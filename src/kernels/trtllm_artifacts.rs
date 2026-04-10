use anyhow::Result;
use sha2::{Digest, Sha256};
use std::io::Read as _;
use std::path::Path;

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
