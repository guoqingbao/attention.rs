use anyhow::Result;
use cudaforge::KernelBuilder;
use sha2::{Digest, Sha256};
use std::io::Read as _;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// nvcc resolution
// ---------------------------------------------------------------------------

pub fn resolve_nvcc() -> String {
    if let Ok(value) = std::env::var("CUDACXX") {
        if !value.trim().is_empty() {
            return value;
        }
    }

    for root_var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = std::env::var(root_var) {
            let candidate = PathBuf::from(root).join("bin/nvcc");
            if candidate.is_file() {
                return candidate.display().to_string();
            }
        }
    }

    // Cargo build scripts do not necessarily inherit the interactive shell's
    // CUDA PATH setup.  CUDA's conventional installation symlink is therefore
    // a useful fallback before trying bare `nvcc` through PATH.
    for candidate in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
    ] {
        if std::path::Path::new(candidate).is_file() {
            return candidate.to_owned();
        }
    }

    "nvcc".to_owned()
}

// ---------------------------------------------------------------------------
// TRT-LLM artifact download
// ---------------------------------------------------------------------------

const ARTIFACT_BASE: &str =
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/";

pub const TRTLLM_GEN_BMM_PATH: &str =
    "b55211623be7f5697c5262ffd8361fc06c147bc9/batched_gemm-b3c1646-c111d7c/";
const TRTLLM_GEN_BMM_CHECKSUM: &str =
    "0af823880730c4f0b3832d2208fab035946694b83444410b9309db5613d60195";

pub const TRTLLM_GEN_GEMM_PATH: &str =
    "b117d5a6b2dd2228aa966a938eac398cf336d8c0/gemm-b3c1646-1fddea2/";
const TRTLLM_GEN_GEMM_CHECKSUM: &str =
    "18262161e624f7da9d2d04c528c645a5ff7f5efd774024a0b2eb92748ab18bb9";

pub const TRTLLM_GEN_FMHA_PATH: &str = "55bba55929d4093682e32d817bd11ffb0441c749/fmha/trtllm-gen/";
const TRTLLM_GEN_FMHA_CHECKSUM: &str =
    "f2c0aad1e74391c4267a2f9a20ec819358b59e04588385cffb452ed341500b99";

const BMM_EXPORT_HEADERS: &[&str] = &[
    "BatchedGemmEnums.h",
    "BatchedGemmInterface.h",
    "BatchedGemmOptions.h",
    "Enums.h",
    "GemmGatedActOptions.h",
    "GemmOptions.h",
    "KernelParams.h",
    "KernelParamsDecl.h",
    "KernelTraits.h",
    "TmaDescriptor.h",
    "trtllm/gen/CommonUtils.h",
    "trtllm/gen/CudaArchDecl.h",
    "trtllm/gen/CudaKernelLauncher.h",
    "trtllm/gen/DtypeDecl.h",
    "trtllm/gen/MmaDecl.h",
    "trtllm/gen/SfLayoutDecl.h",
    "trtllm/gen/SparsityDecl.h",
];

fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn download_file(url: &str) -> Result<Vec<u8>> {
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(30))
        .build();
    for attempt in 1..=4 {
        match agent.get(url).call() {
            Ok(resp) => {
                let mut data = Vec::new();
                resp.into_reader().read_to_end(&mut data)?;
                return Ok(data);
            }
            Err(e) => {
                if attempt < 4 {
                    let delay = std::time::Duration::from_secs(2u64.pow(attempt));
                    eprintln!(
                        "cargo:warning=Download attempt {attempt} failed for {url}: {e}, retrying in {delay:?}"
                    );
                    std::thread::sleep(delay);
                } else {
                    anyhow::bail!("Failed to download {url} after 4 attempts: {e}");
                }
            }
        }
    }
    unreachable!()
}

fn download_and_cache(
    url: &str,
    local_path: &Path,
    expected_sha256: Option<&str>,
) -> Result<Vec<u8>> {
    if local_path.exists() {
        let data = std::fs::read(local_path)?;
        if let Some(expected) = expected_sha256 {
            if sha256_bytes(&data) == expected {
                return Ok(data);
            }
            eprintln!(
                "cargo:warning=SHA256 mismatch for cached {}, re-downloading",
                local_path.display()
            );
        } else {
            return Ok(data);
        }
    }

    if let Some(parent) = local_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let data = download_file(url)?;
    if let Some(expected) = expected_sha256 {
        let actual = sha256_bytes(&data);
        if actual != expected {
            anyhow::bail!("SHA256 mismatch for {url}: expected {expected}, got {actual}");
        }
    }
    std::fs::write(local_path, &data)?;
    Ok(data)
}

fn parse_checksums(data: &[u8]) -> std::collections::HashMap<String, String> {
    let text = String::from_utf8_lossy(data);
    let mut map = std::collections::HashMap::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.trim().splitn(2, ' ').collect();
        if parts.len() == 2 {
            let sha = parts[0].trim();
            let name = parts[1].trim();
            map.insert(name.to_string(), sha.to_string());
        }
    }
    map
}

fn find_hash<'a>(
    checksums: &'a std::collections::HashMap<String, String>,
    target: &str,
) -> Option<&'a str> {
    let target_lower = target.to_lowercase();
    for (name, hash) in checksums {
        let name_lower = name.to_lowercase();
        if name_lower == target_lower || name_lower.ends_with(&format!("/{target_lower}")) {
            return Some(hash.as_str());
        }
    }
    None
}

/// Download BMM export headers into `<dest_dir>/trtllmGen_bmm_export/...`.
pub fn download_bmm_headers(cache_dir: &Path, dest_dir: &Path) -> Result<()> {
    let checksums_url = format!("{ARTIFACT_BASE}{TRTLLM_GEN_BMM_PATH}checksums.txt");
    let checksums_path = cache_dir.join("bmm_checksums.txt");
    let checksums_data = download_and_cache(
        &checksums_url,
        &checksums_path,
        Some(TRTLLM_GEN_BMM_CHECKSUM),
    )?;
    let checksums = parse_checksums(&checksums_data);

    let meta_hash = find_hash(&checksums, "flashinferMetaInfo.h")
        .ok_or_else(|| anyhow::anyhow!("flashinferMetaInfo.h not found in BMM checksums"))?;
    let meta_url = format!("{ARTIFACT_BASE}{TRTLLM_GEN_BMM_PATH}include/flashinferMetaInfo.h");
    let meta_dest = dest_dir.join("flashinferMetaInfo.h");
    download_and_cache(&meta_url, &meta_dest, Some(meta_hash))?;
    println!("cargo:warning=Downloaded BMM flashinferMetaInfo.h");

    let export_dir = dest_dir.join("trtllmGen_bmm_export");
    for header in BMM_EXPORT_HEADERS {
        let header_hash = find_hash(&checksums, header);
        let url =
            format!("{ARTIFACT_BASE}{TRTLLM_GEN_BMM_PATH}include/trtllmGen_bmm_export/{header}");
        let dest = export_dir.join(header);
        download_and_cache(&url, &dest, header_hash)?;
    }
    println!(
        "cargo:warning=Downloaded {} BMM export headers",
        BMM_EXPORT_HEADERS.len()
    );

    Ok(())
}

/// Download GEMM metainfo header.
pub fn download_gemm_metainfo(cache_dir: &Path, dest_dir: &Path) -> Result<()> {
    let checksums_url = format!("{ARTIFACT_BASE}{TRTLLM_GEN_GEMM_PATH}checksums.txt");
    let checksums_path = cache_dir.join("gemm_checksums.txt");
    let checksums_data = download_and_cache(
        &checksums_url,
        &checksums_path,
        Some(TRTLLM_GEN_GEMM_CHECKSUM),
    )?;
    let checksums = parse_checksums(&checksums_data);

    let meta_hash = find_hash(&checksums, "flashinferMetaInfo.h")
        .ok_or_else(|| anyhow::anyhow!("flashinferMetaInfo.h not found in GEMM checksums"))?;
    let meta_url = format!("{ARTIFACT_BASE}{TRTLLM_GEN_GEMM_PATH}include/flashinferMetaInfo.h");
    let meta_dest = dest_dir.join("flashinferMetaInfo.h");
    download_and_cache(&meta_url, &meta_dest, Some(meta_hash))?;
    println!("cargo:warning=Downloaded GEMM flashinferMetaInfo.h");

    Ok(())
}

/// Download FMHA metainfo header. Returns the SHA256 hash of the metainfo file.
pub fn download_fmha_metainfo(cache_dir: &Path, dest_dir: &Path) -> Result<String> {
    let checksums_url = format!("{ARTIFACT_BASE}{TRTLLM_GEN_FMHA_PATH}checksums.txt");
    let checksums_path = cache_dir.join("fmha_checksums.txt");
    let checksums_data = download_and_cache(
        &checksums_url,
        &checksums_path,
        Some(TRTLLM_GEN_FMHA_CHECKSUM),
    )?;
    let checksums = parse_checksums(&checksums_data);

    let meta_hash = find_hash(&checksums, "flashInferMetaInfo.h")
        .ok_or_else(|| anyhow::anyhow!("flashInferMetaInfo.h not found in FMHA checksums"))?;
    let meta_url = format!("{ARTIFACT_BASE}{TRTLLM_GEN_FMHA_PATH}include/flashInferMetaInfo.h");
    let meta_dest = dest_dir.join("flashInferMetaInfo.h");
    download_and_cache(&meta_url, &meta_dest, Some(meta_hash))?;
    println!("cargo:warning=Downloaded FMHA flashInferMetaInfo.h");

    Ok(meta_hash.to_string())
}

pub fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else if ty.is_symlink() {
            let target = std::fs::read_link(entry.path())?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(target, dest_path)?;
            #[cfg(not(unix))]
            std::fs::copy(entry.path(), dest_path)?;
        } else {
            std::fs::copy(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// TRT-LLM builder configuration (Blackwell cubin artifacts)
// ---------------------------------------------------------------------------

/// Wire TRT-LLM BMM/GEMM/FMHA artifacts into `builder` when the `trtllm`
/// feature is enabled. Cubins are Blackwell-only (SM100+).
pub fn configure_trtllm(
    mut builder: KernelBuilder,
    flashinfer_root: &Path,
    build_dir: &Path,
    compute_cap: usize,
    trtllm_enabled: bool,
) -> Result<KernelBuilder> {
    if !trtllm_enabled {
        return Ok(builder);
    }

    if compute_cap < 100 {
        panic!(
            "trtllm feature requires SM100+ (Blackwell). Detected compute_cap={compute_cap}. \
             TRT-LLM fused MoE cubins are Blackwell-only. \
             Remove the trtllm feature to build for this GPU."
        );
    }

    let trtllm_cache = build_dir.join("trtllm_artifacts");
    std::fs::create_dir_all(&trtllm_cache)?;

    // The bmm_export headers go into the FlashInfer include tree so that
    // `#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/Enums.h"` resolves.
    let bmm_dest = flashinfer_root.join("include/flashinfer/trtllm/batched_gemm");
    std::fs::create_dir_all(&bmm_dest)?;

    // BMM metainfo goes into the artifact cache dir (included separately)
    let bmm_include_dir = trtllm_cache.join("bmm_include");
    std::fs::create_dir_all(&bmm_include_dir)?;

    match download_bmm_headers(&trtllm_cache, &bmm_include_dir) {
        Ok(()) => {
            // Symlink/copy the downloaded bmm_export into the FlashInfer tree
            let export_src = bmm_include_dir.join("trtllmGen_bmm_export");
            let export_dst = bmm_dest.join("trtllmGen_bmm_export");
            if export_src.exists() && !export_dst.exists() {
                #[cfg(unix)]
                std::os::unix::fs::symlink(&export_src, &export_dst)
                    .or_else(|_| copy_dir_recursive(&export_src, &export_dst))?;
                #[cfg(not(unix))]
                copy_dir_recursive(&export_src, &export_dst)?;
            }

            // GEMM metainfo
            let gemm_include_dir = trtllm_cache.join("gemm_include");
            let _ = download_gemm_metainfo(&trtllm_cache, &gemm_include_dir);

            // FMHA metainfo
            let fmha_include_dir = trtllm_cache.join("fmha_include");
            let fmha_meta_hash =
                download_fmha_metainfo(&trtllm_cache, &fmha_include_dir).unwrap_or_default();

            builder = builder
                .arg("-DUSE_TRTLLM")
                .arg("-DTLLM_GEN_EXPORT_INTERFACE")
                .arg("-DTLLM_GEN_EXPORT_FLASHINFER")
                .arg("-DTLLM_ENABLE_CUDA")
                .arg(&format!(
                    "-DTLLM_GEN_GEMM_CUBIN_PATH=\\\"{}\\\"",
                    TRTLLM_GEN_BMM_PATH
                ))
                .arg(&format!(
                    "-DTLLM_GEN_FMHA_CUBIN_PATH=\\\"{}\\\"",
                    TRTLLM_GEN_FMHA_PATH
                ))
                .arg(&format!(
                    "-DTLLM_GEN_FMHA_METAINFO_HASH=\\\"{fmha_meta_hash}\\\""
                ))
                .include_path(&bmm_include_dir)
                .include_path(bmm_include_dir.join("trtllmGen_bmm_export"));

            if gemm_include_dir.exists() {
                builder = builder.include_path(&gemm_include_dir);
            }
            if fmha_include_dir.exists() {
                builder = builder.include_path(&fmha_include_dir);
            }

            let csrc_path = PathBuf::from("src/trtllm/trtllm_cutlass_heuristic.cpp");
            if csrc_path.exists() {
                builder = builder.source_files(vec![csrc_path]);
            }

            println!("cargo:warning=TRT-LLM artifacts downloaded successfully");
        }
        Err(e) => {
            println!("cargo:warning=Failed to download TRT-LLM BMM artifacts: {e}");
            println!("cargo:warning=TRT-LLM fused MoE backend will be disabled");
        }
    }

    Ok(builder)
}

// ---------------------------------------------------------------------------
// FlashMLA (SM90 DeepSeek V4 sparse) — separate static lib
// ---------------------------------------------------------------------------

/// Build FlashMLA DSV4 sparse as a separate static library with FlashMLA's
/// own CUTLASS pin so it does not clash with attention.rs CUTLASS.
///
/// Returns `(builder, link_flashmla)`. When `link_flashmla` is true the
/// caller must emit `cargo:rustc-link-lib=static=flashmla_dsv4`.
pub fn configure_flashmla(
    mut builder: KernelBuilder,
    build_dir: &Path,
    compute_cap: usize,
) -> Result<(KernelBuilder, bool)> {
    let mut link_flashmla = false;

    if !(std::env::var("CARGO_FEATURE_FLASHINFER").is_ok() && compute_cap >= 90) {
        return Ok((builder, link_flashmla));
    }

    // cudaforge only treats a checkout as cached when <root>/include exists.
    // FlashMLA has no top-level include/, so without this marker every build
    // re-runs `git fetch` (and hangs when GitHub is slow/unreachable).
    let fm_commit = "05e26647fe840b8baedae486c2d86d5ce4efeb7c";
    if let Some(cache) = std::env::var_os("HOME").map(|h| {
        PathBuf::from(h).join(format!(
            ".cudaforge/git/checkouts/flashmla-{}",
            &fm_commit[..16]
        ))
    }) {
        if cache.exists() {
            let _ = std::fs::create_dir_all(cache.join("include"));
        }
    }
    builder = builder.with_git_dependency(
        "flashmla",
        "https://github.com/sgl-project/FlashMLA.git",
        fm_commit,
        vec![
            "csrc",
            "csrc/kerutils/include",
            "csrc/sm90",
            "csrc/cutlass/include",
            "csrc/cutlass/tools/util/include",
        ],
        vec![
            "csrc/sm90/decode/sparse_fp8",
            "csrc/sm90/prefill/sparse",
            "csrc/smxx/decode",
            "csrc/kerutils",
            "csrc/params.h",
            "csrc/cutlass",
        ],
        true,
    );
    let flashmla_root = builder.fetch_git_dependency("flashmla")?;
    let fm_csrc = flashmla_root.join("csrc");
    let fm_cutlass_hdr = fm_csrc.join("cutlass/include/cutlass/bfloat16.h");
    if !fm_cutlass_hdr.exists() {
        let _ = std::process::Command::new("git")
            .args([
                "submodule",
                "update",
                "--init",
                "--depth",
                "1",
                "csrc/cutlass",
            ])
            .current_dir(&flashmla_root)
            .status();
    }
    if fm_csrc.join("cutlass/include/cutlass/bfloat16.h").exists() {
        let fm_out = build_dir.join("flashmla_objs");
        std::fs::create_dir_all(&fm_out)?;
        let nvcc = resolve_nvcc();
        let sources = [
            fm_csrc.join("sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu"),
            fm_csrc.join("sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu"),
            fm_csrc.join("sm90/prefill/sparse/instantiations/phase1_k512.cu"),
            fm_csrc.join("sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu"),
            fm_csrc.join("smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu"),
            fm_csrc.join("smxx/decode/combine/combine.cu"),
            PathBuf::from("src/flashmla_sparse_mla.cu"),
        ];
        let mut objs = Vec::new();
        let mut ok = true;
        for src in &sources {
            let obj = fm_out.join(format!(
                "{}.o",
                src.file_stem().and_then(|s| s.to_str()).unwrap_or("obj")
            ));
            let mut cmd = std::process::Command::new(&nvcc);
            cmd.arg("-c")
                .arg(src)
                .arg("-o")
                .arg(&obj)
                .arg("-std=c++20")
                .arg("-O3")
                .arg("--expt-relaxed-constexpr")
                .arg("--expt-extended-lambda")
                .arg("--use_fast_math")
                .arg("-Xcompiler")
                .arg("-fPIC")
                .arg("-DATTENTION_RS_USE_FLASHMLA")
                .arg(format!("-I{}", fm_csrc.display()))
                .arg(format!("-I{}", fm_csrc.join("kerutils/include").display()))
                .arg(format!("-I{}", fm_csrc.join("sm90").display()))
                .arg(format!("-I{}", fm_csrc.join("cutlass/include").display()))
                .arg(format!(
                    "-I{}",
                    fm_csrc.join("cutlass/tools/util/include").display()
                ))
                .arg("-gencode=arch=compute_90a,code=sm_90a");
            let status = cmd.status();
            match status {
                Ok(st) if st.success() => objs.push(obj),
                Ok(st) => {
                    println!(
                        "cargo:warning=FlashMLA nvcc failed for {}: status={}",
                        src.display(),
                        st
                    );
                    ok = false;
                    break;
                }
                Err(e) => {
                    println!("cargo:warning=FlashMLA nvcc spawn failed: {e}");
                    ok = false;
                    break;
                }
            }
        }
        if ok && !objs.is_empty() {
            let lib = build_dir.join("libflashmla_dsv4.a");
            let _ = std::fs::remove_file(&lib);
            let mut ar = std::process::Command::new("ar");
            ar.arg("rcs").arg(&lib);
            for o in &objs {
                ar.arg(o);
            }
            if ar.status().map(|s| s.success()).unwrap_or(false) {
                link_flashmla = true;
                // Exclude the in-tree wrapper from the main lib (compiled into flashmla lib).
                builder = builder.exclude(&["flashmla_sparse_mla.cu"]);
                println!("cargo:warning=FlashMLA DSV4 sparse static lib built");
            } else {
                println!("cargo:warning=Failed to archive FlashMLA objects");
            }
        }
    } else {
        println!("cargo:warning=FlashMLA CUTLASS missing; sparse FlashMLA stub only");
    }

    Ok((builder, link_flashmla))
}
