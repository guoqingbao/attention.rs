use anyhow::Result;
use cudaforge::KernelBuilder;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/pagedattention.cuh");
    println!("cargo:rerun-if-changed=src/prefill_paged_attn.cu");
    println!("cargo:rerun-if-changed=src/prefill_paged_attn_opt.cu");
    println!("cargo:rerun-if-changed=src/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/mamba_scatter_kernel.cu");
    println!("cargo:rerun-if-changed=src/reshape_and_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/sort.cu");
    println!("cargo:rerun-if-changed=src/update_kvscales.cu");
    println!("cargo:rerun-if-changed=src/mask.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm.cu");
    println!("cargo:rerun-if-changed=src/moe_gemv.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm_wmma.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm_gguf.cu");
    println!("cargo:rerun-if-changed=src/moe_gguf_small_m.cu");
    println!("cargo:rerun-if-changed=src/moe_wmma_gguf.cu");
    println!("cargo:rerun-if-changed=src/gpu_sampling.cuh");
    println!("cargo:rerun-if-changed=src/gpu_sampling.cu");
    println!("cargo:rerun-if-changed=src/fused_rope.cu");
    println!("cargo:rerun-if-changed=src/fp8_matmul.cu");
    println!("cargo:rerun-if-changed=src/fp8_gemm_cutlass.cu");
    println!("cargo:rerun-if-changed=src/fp8_moe_cutlass.cu");
    println!("cargo:rerun-if-changed=src/flashinfer_fp8_qquant.cu");
    println!("cargo:rerun-if-changed=src/flashinfer_adapter_fp8.cu");
    println!("cargo:rerun-if-changed=src/flashinfer_bmm_fp8.cu");
    println!("cargo:rerun-if-changed=src/flashinfer_moe_adapter.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_batched_gemm_runner.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_fused_moe_runner.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_fused_moe_dev_kernel.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_fused_moe_routing_renormalize.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_fused_moe_routing_deepseek.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_fused_moe_routing_llama4.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_cutlass_heuristic.cpp");
    println!("cargo:rerun-if-changed=src/gdn.cu");

    let marlin_disabled = std::env::var("CARGO_FEATURE_NO_MARLIN").is_ok();
    let fp8_kvcache_disabled = std::env::var("CARGO_FEATURE_NO_FP8_KVCACHE").is_ok();

    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());

    let mut builder = KernelBuilder::new()
        .source_dir("src")
        .nvcc_thread_patterns(&["flash_api", "cutlass", "flashinfer"], 2)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    let compute_cap = builder.get_compute_cap().unwrap_or(80);

    println!("cargo:info=compute capability: {:?}", compute_cap);

    if compute_cap < 80 {
        builder = builder.arg("-DNO_BF16_KERNEL");
        builder = builder.arg("-DNO_MARLIN_KERNEL");
    }

    if compute_cap < 90 {
        builder = builder.arg("-DNO_HARDWARE_FP8");
    }

    if marlin_disabled {
        builder = builder.arg("-DNO_MARLIN_KERNEL");
    }

    if fp8_kvcache_disabled {
        builder = builder.arg("-DNO_FP8_KVCACHE");
    }

    if std::env::var("CARGO_FEATURE_CUTLASS").is_ok()
        || std::env::var("CARGO_FEATURE_FLASHINFER").is_ok()
    {
        builder = builder.arg("-DUSE_CUTLASS").with_cutlass(None);

        if std::env::var("CARGO_FEATURE_FLASHINFER").is_ok() {
            builder = builder.arg("-DENABLE_BF16").arg("-DENABLE_FP8");
            if compute_cap >= 89 {
                builder = builder.arg("-DFLASHINFER_ENABLE_FP8_E8M0");
            }
            if compute_cap == 90 {
                builder = builder.arg("-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED");
                builder = builder.arg("-DSM_90_PASS");
            }
            if compute_cap >= 90 {
                builder = builder.arg("-DFLASHINFER_ENABLE_FP8_E4M3");
                builder = builder.arg("-DFLASHINFER_ENABLE_FP4_E2M1");
            }
        }
    }

    if std::env::var("CARGO_FEATURE_FLASHINFER").is_ok() {
        println!("cargo:rerun-if-changed=src/flashinfer_adapter.cu");
        // DO not change this, this featch custom flashinfer v0.6.2 headers
        // which is compatible with our code (added more gqa group_size)
        builder = builder.arg("-DUSE_FLASHINFER").with_git_dependency(
            "flashinfer",
            "https://github.com/guoqingbao/flashinfer.git",
            "35e1a37733b452fe26c2a51ac38fca134b1efaa4", // v0.6.2
            vec![
                "include",
                "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export",
                "include/flashinfer/trtllm/gemm/trtllmGen_gemm_export",
                "csrc/nv_internal",
                "csrc/nv_internal/include",
                "csrc/nv_internal/tensorrt_llm/cutlass_extensions/include",
            ],
            vec![
                "csrc/nv_internal/cpp/common",
                "csrc/nv_internal/tensorrt_llm",
            ],
            false,
        );

        let flashinfer_root = builder.fetch_git_dependency("flashinfer")?;
        let csrc_dir = flashinfer_root.join("csrc");
        let trtllm_dir = csrc_dir.join("nv_internal").join("tensorrt_llm");

        if matches!(compute_cap, 90 | 100) && trtllm_dir.exists() {
            let include_define = format!(
                "-DATTENTION_RS_FLASHINFER_TRTLLM_INCLUDE_DIR=\\\"{}\\\"",
                trtllm_dir.display()
            );
            builder = builder
                .arg("-DATTENTION_RS_USE_FLASHINFER_BLOCKSCALE")
                .arg("-DCOMPILE_HOPPER_TMA_GEMMS")
                .arg("-DENABLE_FP8_BLOCK_SCALE")
                .arg(&include_define)
                .include_path(csrc_dir.join("nv_internal/tensorrt_llm/kernels/cutlass_kernels/include"))
                .include_path(csrc_dir.join("nv_internal/tensorrt_llm/kernels/cutlass_kernels"))
                .source_files(vec![
                    csrc_dir.join(
                        "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu",
                    ),
                    csrc_dir.join("nv_internal/cpp/common/envUtils.cpp"),
                    csrc_dir.join("nv_internal/cpp/common/logger.cpp"),
                    csrc_dir.join("nv_internal/cpp/common/stringUtils.cpp"),
                    csrc_dir.join("nv_internal/cpp/common/tllmException.cpp"),
                    csrc_dir.join("nv_internal/cpp/common/memoryUtils.cu"),
                ]);
        } else if matches!(compute_cap, 90 | 100) {
            println!(
                "cargo:warning=flashinfer TensorRT-LLM sources not found at {}, skipping blockscale fp8 wrapper",
                trtllm_dir.display()
            );
        }
    }

    // Target handling
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

    let _ = builder.build_lib(build_dir.join("libpagedattention.a"))?;

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=pagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    // println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}
