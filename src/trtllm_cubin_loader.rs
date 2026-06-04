use std::ffi::{c_char, c_int, CStr};
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::Once;

static INIT: Once = Once::new();

const ARTIFACT_BASE: &str =
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/";

fn cubin_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("FLASHINFER_CUBIN_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache/flashinfer/cubins");
    }
    PathBuf::from("/tmp/flashinfer_cubins")
}

fn download_cubin(relative_path: &str, expected_sha256: &str) -> Option<Vec<u8>> {
    let cache_dir = cubin_cache_dir();
    let local_path = cache_dir.join(relative_path);

    if local_path.exists() {
        if let Ok(data) = std::fs::read(&local_path) {
            if expected_sha256.is_empty() || verify_sha256(&data, expected_sha256) {
                return Some(data);
            }
            tracing::warn!(
                "SHA256 mismatch for cached cubin {}, re-downloading",
                local_path.display()
            );
        }
    }

    if std::env::var("FLASHINFER_NO_DOWNLOAD").is_ok() {
        tracing::error!(
            "Cubin not found locally and FLASHINFER_NO_DOWNLOAD is set: {}",
            relative_path
        );
        return None;
    }

    let url = format!("{ARTIFACT_BASE}{relative_path}");
    tracing::info!("Downloading cubin: {url}");

    for attempt in 1..=4u32 {
        match ureq::get(&url)
            .timeout(std::time::Duration::from_secs(60))
            .call()
        {
            Ok(resp) => {
                let mut data = Vec::new();
                if resp.into_reader().read_to_end(&mut data).is_ok() {
                    if !expected_sha256.is_empty() && !verify_sha256(&data, expected_sha256) {
                        tracing::error!("SHA256 mismatch for downloaded cubin {relative_path}");
                        return None;
                    }
                    if let Some(parent) = local_path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = std::fs::write(&local_path, &data);
                    return Some(data);
                }
            }
            Err(e) => {
                if attempt < 4 {
                    let delay = std::time::Duration::from_secs(2u64.pow(attempt));
                    tracing::warn!(
                        "Download attempt {attempt} failed for {relative_path}: {e}, retrying in {delay:?}"
                    );
                    std::thread::sleep(delay);
                } else {
                    tracing::error!(
                        "Failed to download cubin {relative_path} after 4 attempts: {e}"
                    );
                }
            }
        }
    }
    None
}

fn verify_sha256(data: &[u8], expected: &str) -> bool {
    use sha2::{Digest, Sha256};
    let hash = format!("{:x}", Sha256::digest(data));
    hash == expected
}

/// C callback invoked by the TRT-LLM C++ code via `getCubin()`.
/// Downloads the cubin from NVIDIA artifactory (or reads from cache),
/// then passes it back via `FlashInferSetCurrentCubin`.
unsafe extern "C" fn cubin_callback(path_ptr: *const c_char, sha256_ptr: *const c_char) {
    let path = if path_ptr.is_null() {
        ""
    } else {
        CStr::from_ptr(path_ptr).to_str().unwrap_or("")
    };
    let sha256 = if sha256_ptr.is_null() {
        ""
    } else {
        CStr::from_ptr(sha256_ptr).to_str().unwrap_or("")
    };

    if let Some(data) = download_cubin(path, sha256) {
        kernels::ffi::FlashInferSetCurrentCubin(
            data.as_ptr() as *const c_char,
            data.len() as c_int,
        );
    } else {
        tracing::error!("Failed to load cubin: {path}");
        kernels::ffi::FlashInferSetCurrentCubin(std::ptr::null(), 0);
    }
}

/// Initialize the TRT-LLM cubin loader callback.
/// Must be called once before any TRT-LLM fused MoE operations.
pub fn init_cubin_loader() {
    INIT.call_once(|| {
        unsafe {
            kernels::ffi::FlashInferSetCubinCallback(Some(cubin_callback));
        }
        tracing::info!(
            "TRT-LLM cubin loader initialized (cache: {})",
            cubin_cache_dir().display()
        );
    });
}
