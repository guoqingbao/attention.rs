use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;
fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/pagedattention.cuh");
    println!("cargo:rerun-if-changed=src/prefill_paged_attn.cu");
    println!("cargo:rerun-if-changed=src/prefill_paged_attn_opt.cu");
    println!("cargo:rerun-if-changed=src/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/reshape_and_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/sort.cu");
    println!("cargo:rerun-if-changed=src/update_kvscales.cu");
    println!("cargo:rerun-if-changed=src/mask.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm_wmma.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm_gguf.cu");
    println!("cargo:rerun-if-changed=src/moe_gguf_small_m.cu");
    println!("cargo:rerun-if-changed=src/moe_wmma_gguf.cu");
    println!("cargo:rerun-if-changed=src/gpu_sampling.cuh");
    println!("cargo:rerun-if-changed=src/gpu_sampling.cu");
    println!("cargo:rerun-if-changed=src/fused_rope.cu");
    let marlin_disabled = std::env::var("CARGO_FEATURE_NO_MARLIN").is_ok();
    let fp8_kvcache_disabled = std::env::var("CARGO_FEATURE_NO_FP8_KVCACHE").is_ok();
    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or("".to_string()));
    let mut builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math");

    let compute_cap = {
        if let Ok(var) = std::env::var("CUDA_COMPUTE_CAP") {
            var.parse::<usize>().unwrap() * 10
        } else {
            let mut cmd = Command::new("nvidia-smi");
            match cmd
                .args(["--query-gpu=compute_cap", "--format=csv"])
                .output()
            {
                Ok(out) => {
                    let output =
                        String::from_utf8(out.stdout).expect("Output of nvidia-smi was not utf8.");
                    (output
                        .split('\n')
                        .nth(1)
                        .unwrap()
                        .trim()
                        .parse::<f32>()
                        .unwrap()
                        * 100.) as usize
                }
                Err(_) => {
                    panic!(
                        "`CUDA_COMPUTE_CAP` env var not specified and `nvidia-smi` was not found."
                    );
                }
            }
        }
    };

    if compute_cap < 800 {
        builder = builder.arg("-DNO_BF16_KERNEL");
        builder = builder.arg("-DNO_MARLIN_KERNEL");
        builder = builder.arg("-DNO_HARDWARE_FP8");
    }

    if marlin_disabled {
        builder = builder.arg("-DNO_MARLIN_KERNEL");
    }

    if fp8_kvcache_disabled {
        builder = builder.arg("-DNO_FP8_KVCACHE");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC").arg("-std=c++17");
    }

    println!("cargo:info={builder:?}");
    builder.build_lib(build_dir.join("libpagedattention.a"));
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=pagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    Ok(())
}
