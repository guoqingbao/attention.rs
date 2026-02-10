use anyhow::{Context, Result};
use cudaforge::KernelBuilder;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::{fs, path::Path};

const TRTLLM_CUBIN_REPOSITORY_DEFAULT: &str =
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local";
const TRTLLM_GEN_FMHA_REL_PATH: &str = "75d477a640f268ea9ad117cc596eb39245713b9e/fmha/trtllm-gen";
const TRTLLM_GEN_FMHA_CHECKSUMS_SHA256: &str =
    "e014d7a54c396733ef012b223603c1be2861019f88faa5dcc882ed1ecfe5c2d9";

fn env_true(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        })
        .unwrap_or(false)
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)
        .with_context(|| format!("failed to open file for sha256: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f
            .read(&mut buf)
            .with_context(|| format!("failed to read file for sha256: {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn download_file(url: &str, output: &Path) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create dir {}", parent.display()))?;
    }
    let mut last_err: Option<anyhow::Error> = None;
    for _ in 0..3 {
        match ureq::get(url).call() {
            Ok(resp) => {
                let mut reader = resp.into_reader();
                let mut out = fs::File::create(output)
                    .with_context(|| format!("failed to create {}", output.display()))?;
                std::io::copy(&mut reader, &mut out)
                    .with_context(|| format!("failed to write {}", output.display()))?;
                return Ok(());
            }
            Err(e) => {
                last_err = Some(anyhow::anyhow!("download error for {}: {}", url, e));
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("download failed for {}", url)))
}

fn parse_checksums(checksums_path: &Path) -> Result<HashMap<String, String>> {
    let text = fs::read_to_string(checksums_path)
        .with_context(|| format!("failed to read {}", checksums_path.display()))?;
    let mut m = HashMap::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let sha = parts.next();
        let name = parts.next();
        if let (Some(sha), Some(name)) = (sha, name) {
            m.insert(name.to_string(), sha.to_string());
        }
    }
    if m.is_empty() {
        anyhow::bail!("no entries parsed from {}", checksums_path.display());
    }
    Ok(m)
}

fn count_cached_cubins(trtllm_dir: &Path, checksums_path: &Path) -> Result<(usize, usize)> {
    let entries = parse_checksums(checksums_path)?;
    let mut total = 0usize;
    let mut cached = 0usize;
    for name in entries.keys() {
        if !name.ends_with(".cubin") {
            continue;
        }
        total += 1;
        if trtllm_dir.join(name).exists() {
            cached += 1;
        }
    }
    Ok((cached, total))
}

fn ensure_trtllm_artifacts(
    trtllm_dir: &Path,
    repository: &str,
    no_download: bool,
    download_cubins: bool,
) -> Result<String> {
    let checksums_path = trtllm_dir.join("checksums.txt");
    if !checksums_path.exists() {
        if no_download {
            anyhow::bail!(
                "missing {} and ATTENTION_RS_TRTLLM_NO_DOWNLOAD is set",
                checksums_path.display()
            );
        }
        let url = format!("{}/{}/checksums.txt", repository, TRTLLM_GEN_FMHA_REL_PATH);
        download_file(&url, &checksums_path)
            .with_context(|| format!("failed to download TRTLLM checksums from {}", url))?;
    }
    let checksums_sha = file_sha256(&checksums_path)?;
    if checksums_sha != TRTLLM_GEN_FMHA_CHECKSUMS_SHA256 {
        if no_download {
            anyhow::bail!(
                "checksums hash mismatch for {} (got {}, expected {})",
                checksums_path.display(),
                checksums_sha,
                TRTLLM_GEN_FMHA_CHECKSUMS_SHA256
            );
        }
        let url = format!("{}/{}/checksums.txt", repository, TRTLLM_GEN_FMHA_REL_PATH);
        download_file(&url, &checksums_path)
            .with_context(|| format!("failed to refresh TRTLLM checksums from {}", url))?;
    }

    let entries = parse_checksums(&checksums_path)?;
    let meta_hash = entries
        .get("include/flashInferMetaInfo.h")
        .cloned()
        .context("missing include/flashInferMetaInfo.h in checksums.txt")?;

    for (name, expected_sha) in entries.iter() {
        let is_meta = name == "include/flashInferMetaInfo.h";
        let is_cubin = name.ends_with(".cubin");
        if !is_meta && !(download_cubins && is_cubin) {
            continue;
        }
        let local = trtllm_dir.join(name);
        let mut ok = false;
        if local.exists() {
            let got = file_sha256(&local)?;
            ok = got == *expected_sha;
        }
        if !ok {
            if no_download {
                anyhow::bail!(
                    "missing or invalid artifact {} with ATTENTION_RS_TRTLLM_NO_DOWNLOAD set",
                    local.display()
                );
            }
            let url = format!("{}/{}/{}", repository, TRTLLM_GEN_FMHA_REL_PATH, name);
            download_file(&url, &local)
                .with_context(|| format!("failed to download TRTLLM artifact {}", url))?;
            let got = file_sha256(&local)?;
            if got != *expected_sha {
                anyhow::bail!(
                    "sha256 mismatch for {}: got {}, expected {}",
                    local.display(),
                    got,
                    expected_sha
                );
            }
        }
    }

    Ok(meta_hash)
}

pub fn emit_rerun_if_env_changed() {
    println!("cargo:rerun-if-env-changed=FLASHINFER_BACKEND");
    println!("cargo:rerun-if-env-changed=ATTENTION_RS_TRTLLM_FMHA_DIR");
    println!("cargo:rerun-if-env-changed=ATTENTION_RS_TRTLLM_REPOSITORY");
    println!("cargo:rerun-if-env-changed=ATTENTION_RS_TRTLLM_NO_DOWNLOAD");
    println!("cargo:rerun-if-env-changed=ATTENTION_RS_TRTLLM_DOWNLOAD_CUBINS");
}

pub fn trtllm_backend_requested() -> bool {
    std::env::var("FLASHINFER_BACKEND")
        .ok()
        .map(|v| v.eq_ignore_ascii_case("trtllm"))
        .unwrap_or(false)
}

pub fn configure(builder: KernelBuilder, compute_cap: usize) -> Result<KernelBuilder> {
    if compute_cap != 100 && compute_cap != 103 {
        anyhow::bail!(
            "TRTLLM backend currently supports only SM100/SM103 cubins, got SM{}",
            compute_cap
        );
    }

    let user_dir = std::env::var("ATTENTION_RS_TRTLLM_FMHA_DIR")
        .ok()
        .map(PathBuf::from);
    let default_dir = std::env::var("HOME").ok().map(PathBuf::from).map(|p| {
        p.join(".cache/flashinfer/cubins")
            .join(TRTLLM_GEN_FMHA_REL_PATH)
    });
    let trtllm_dir = user_dir.or(default_dir);
    let Some(trtllm_dir) = trtllm_dir else {
        anyhow::bail!("TRTLLM support requested but no artifact dir could be resolved");
    };

    let repository = std::env::var("ATTENTION_RS_TRTLLM_REPOSITORY")
        .unwrap_or_else(|_| TRTLLM_CUBIN_REPOSITORY_DEFAULT.to_string());
    let repository = repository.trim_end_matches('/');
    let no_download = env_true("ATTENTION_RS_TRTLLM_NO_DOWNLOAD");
    let download_cubins = env_true("ATTENTION_RS_TRTLLM_DOWNLOAD_CUBINS");

    let meta_hash = ensure_trtllm_artifacts(&trtllm_dir, repository, no_download, download_cubins)?;

    let include_dir = trtllm_dir.join("include");
    if !include_dir.join("flashInferMetaInfo.h").exists() {
        anyhow::bail!(
            "TRTLLM support requested but missing {}",
            include_dir.join("flashInferMetaInfo.h").display()
        );
    }
    let checksums_path = trtllm_dir.join("checksums.txt");
    if let Ok((cached, total)) = count_cached_cubins(&trtllm_dir, &checksums_path) {
        println!(
            "cargo:warning=TRTLLM cubin cache status: {}/{} files present in {}",
            cached,
            total,
            trtllm_dir.display()
        );
    }
    if !download_cubins {
        println!(
            "cargo:warning=TRTLLM build enabled with metadata only. Set ATTENTION_RS_TRTLLM_DOWNLOAD_CUBINS=1 to prefetch all cubins."
        );
    }

    let cubin_path = trtllm_dir.to_string_lossy().replace('\\', "/");
    let include_arg = format!("-I{}", include_dir.display());
    let cubin_arg = format!("-DTLLM_GEN_FMHA_CUBIN_PATH=\\\"{}\\\"", cubin_path);
    let meta_hash_arg = format!("-DTLLM_GEN_FMHA_METAINFO_HASH=\\\"{}\\\"", meta_hash);
    let mut builder = builder
        .arg("-DENABLE_TRTLLM")
        .arg(&include_arg)
        .arg(&cubin_arg)
        .arg(&meta_hash_arg);
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("windows") {
        builder = builder.arg("-Wno-deprecated-declarations");
    }

    println!(
        "cargo:warning=TRTLLM support enabled with artifacts from {}",
        trtllm_dir.display()
    );
    Ok(builder)
}
