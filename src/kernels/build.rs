mod trtllm_artifacts;

use anyhow::Result;
use cudaforge::KernelBuilder;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=trtllm_artifacts.rs");
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
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_fused_moe_routing_custom_block.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_fused_moe_routing_custom_cluster.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_fused_moe_routing_deepseek.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_fused_moe_routing_llama4.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_fused_moe_routing_common.cu");
    println!("cargo:rerun-if-changed=src/trtllm/trtllm_cutlass_heuristic.cpp");
    println!("cargo:rerun-if-changed=src/gdn.cu");
    println!("cargo:rerun-if-changed=src/mxfp4_gemm.cu");
    println!("cargo:rerun-if-changed=src/mxfp4_gemm_wmma.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_gemm.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_gemm_cutlass.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_moe_cutlass.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_quant.cu");
    println!("cargo:rerun-if-changed=src/mxfp4_gemm_cutlass.cu");
    println!("cargo:rerun-if-changed=src/mxfp4_quant.cu");
    println!("cargo:rerun-if-changed=src/gptoss_swiglu.cu");
    println!("cargo:rerun-if-changed=src/concat_and_cache_mla_kernel.cu");
    println!("cargo:rerun-if-changed=src/mla_paged_attention.cu");

    let marlin_disabled = std::env::var("CARGO_FEATURE_NO_MARLIN").is_ok();
    let fp8_kvcache_disabled = std::env::var("CARGO_FEATURE_NO_FP8_KVCACHE").is_ok();
    let trtllm_enabled = std::env::var("CARGO_FEATURE_TRTLLM").is_ok();

    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());

    let mut builder = KernelBuilder::new()
        .source_dir("src")
        .nvcc_thread_patterns(&["flash_api", "cutlass", "flashinfer"], 2)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    if !trtllm_enabled {
        builder = builder.exclude(&["trtllm/*"]);
    }

    let compute_cap = builder.get_compute_cap().unwrap_or(80);

    println!("cargo:info=compute capability: {:?}", compute_cap);

    if compute_cap < 80 {
        builder = builder.arg("-DNO_BF16_KERNEL");
        builder = builder.arg("-DNO_MARLIN_KERNEL");
    }

    if compute_cap < 89 {
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
        builder = builder
            .arg("-DUSE_CUTLASS")
            .with_cutlass(Some("da5e086dab31d63815acafdac9a9c5893b1c69e2"));

        if compute_cap >= 100 {
            builder = builder.arg("-DENABLE_FP4");
        }
        if (100..120).contains(&compute_cap) {
            builder = builder.arg("-DENABLE_FP4_SM100");
        }
        if compute_cap >= 120 {
            builder = builder.arg("-DENABLE_FP4_SM120");
        }

        if std::env::var("CARGO_FEATURE_FLASHINFER").is_ok() {
            builder = builder.arg("-DENABLE_BF16").arg("-DENABLE_FP8");
            if compute_cap >= 89 {
                builder = builder.arg("-DFLASHINFER_ENABLE_FP8_E8M0");
            }
            if (90..100).contains(&compute_cap) {
                builder = builder.arg("-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED");
                builder = builder.arg("-DSM_90_PASS");
            }
            if compute_cap >= 89 {
                builder = builder.arg("-DFLASHINFER_ENABLE_FP8_E4M3");
            }
            if compute_cap >= 90 {
                builder = builder.arg("-DFLASHINFER_ENABLE_FP4_E2M1");
            }
        }
    }

    if std::env::var("CARGO_FEATURE_FLASHINFER").is_ok() {
        println!("cargo:rerun-if-changed=src/flashinfer_adapter.cu");
        println!("cargo:rerun-if-changed=src/flashinfer_mla.cu");
        // Custom flashinfer v0.6.7 with GQA fixes (guoqingbao fork)
        // Synced with CUTLASS 4.4.2 (da5e086d) for SM100+/SM121 support
        builder = builder.arg("-DUSE_FLASHINFER").with_git_dependency(
            "flashinfer",
            "https://github.com/guoqingbao/flashinfer.git",
            "d586bc467d1e2f3a92e0f13c1bf5a09cd25027d7", // v0.6.7
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

        if compute_cap >= 90 && trtllm_dir.exists() {
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
                    csrc_dir.join("nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp"),
                ]);
        } else if compute_cap >= 90 {
            println!(
                "cargo:warning=flashinfer TensorRT-LLM sources not found at {}, skipping blockscale fp8 wrapper",
                trtllm_dir.display()
            );
        }

        // TRT-LLM backend: download BMM/GEMM/FMHA artifacts from NVIDIA artifactory.
        // Cubins are Blackwell-only (SM100+); fail the build on older architectures.
        if trtllm_enabled && compute_cap < 100 {
            panic!(
                "trtllm feature requires SM100+ (Blackwell). Detected compute_cap={compute_cap}. \
                 TRT-LLM fused MoE cubins are Blackwell-only. \
                 Remove the trtllm feature to build for this GPU."
            );
        }
        if trtllm_enabled && compute_cap >= 100 {
            let trtllm_cache = build_dir.join("trtllm_artifacts");
            std::fs::create_dir_all(&trtllm_cache)?;

            // The bmm_export headers go into the FlashInfer include tree so that
            // `#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/Enums.h"` resolves.
            let bmm_dest = flashinfer_root.join("include/flashinfer/trtllm/batched_gemm");
            std::fs::create_dir_all(&bmm_dest)?;

            // BMM metainfo goes into the artifact cache dir (included separately)
            let bmm_include_dir = trtllm_cache.join("bmm_include");
            std::fs::create_dir_all(&bmm_include_dir)?;

            match trtllm_artifacts::download_bmm_headers(&trtllm_cache, &bmm_include_dir) {
                Ok(()) => {
                    // Symlink/copy the downloaded bmm_export into the FlashInfer tree
                    let export_src = bmm_include_dir.join("trtllmGen_bmm_export");
                    let export_dst = bmm_dest.join("trtllmGen_bmm_export");
                    if export_src.exists() && !export_dst.exists() {
                        #[cfg(unix)]
                        std::os::unix::fs::symlink(&export_src, &export_dst).or_else(|_| {
                            trtllm_artifacts::copy_dir_recursive(&export_src, &export_dst)
                        })?;
                        #[cfg(not(unix))]
                        trtllm_artifacts::copy_dir_recursive(&export_src, &export_dst)?;
                    }

                    // GEMM metainfo
                    let gemm_include_dir = trtllm_cache.join("gemm_include");
                    let _ =
                        trtllm_artifacts::download_gemm_metainfo(&trtllm_cache, &gemm_include_dir);

                    // FMHA metainfo
                    let fmha_include_dir = trtllm_cache.join("fmha_include");
                    let fmha_meta_hash =
                        trtllm_artifacts::download_fmha_metainfo(&trtllm_cache, &fmha_include_dir)
                            .unwrap_or_default();

                    builder = builder
                        .arg("-DUSE_TRTLLM")
                        .arg("-DTLLM_GEN_EXPORT_INTERFACE")
                        .arg("-DTLLM_GEN_EXPORT_FLASHINFER")
                        .arg("-DTLLM_ENABLE_CUDA")
                        .arg(&format!(
                            "-DTLLM_GEN_GEMM_CUBIN_PATH=\\\"{}\\\"",
                            trtllm_artifacts::TRTLLM_GEN_BMM_PATH
                        ))
                        .arg(&format!(
                            "-DTLLM_GEN_FMHA_CUBIN_PATH=\\\"{}\\\"",
                            trtllm_artifacts::TRTLLM_GEN_FMHA_PATH
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

    Ok(())
}
