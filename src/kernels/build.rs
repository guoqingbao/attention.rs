mod others;

use anyhow::Result;
use cudaforge::KernelBuilder;
use std::path::{Path, PathBuf};

fn unix_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "/")
}

fn main() -> Result<()> {
    // CUDA translation units use host-side C++ state/workspaces (including
    // DeepSeek V4's persistent per-device scratch guards).  Consumers that
    // do not otherwise pull in a C++ dependency, such as the library test
    // binary, still need the C++ ABI and standard-library symbols.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDACXX");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=others.rs");
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
    println!("cargo:rerun-if-changed=src/moe_w2_unpack.cu");
    println!("cargo:rerun-if-changed=src/moe_w2_pack.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm_wmma.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm_gguf.cu");
    println!("cargo:rerun-if-changed=src/gguf_gemm.cu");
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
    println!("cargo:rerun-if-changed=src/deepseek_v4/ds_hc.cu");
    println!("cargo:rerun-if-changed=src/deepseek_v4/ds_compressor.cu");
    println!("cargo:rerun-if-changed=src/deepseek_v4/ds_indexer.cu");
    println!("cargo:rerun-if-changed=src/deepseek_v4/ds_sparse_attn.cu");
    println!("cargo:rerun-if-changed=src/deepseek_v4/ds_quant.cu");
    println!("cargo:rerun-if-changed=src/deepseek_v4/ds_moe.cu");
    println!("cargo:rerun-if-changed=src/deepseek_v4/ds_fp8_kv_pack.cu");
    println!("cargo:rerun-if-changed=src/flashmla_sparse_mla.cu");
    println!("cargo:rerun-if-changed=src/flashinfer_sparse_mla_dsv4.cu");
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
    println!("cargo:rerun-if-changed=src/qwen4/qwen4_hc.cu");
    println!("cargo:rerun-if-changed=src/qwen4/qwen4_qsa.cu");
    println!("cargo:rerun-if-changed=src/gdn.cu");
    println!("cargo:rerun-if-changed=src/gdn_flashinfer_prefill.cu");
    println!("cargo:rerun-if-changed=src/mxfp4_gemm.cu");
    println!("cargo:rerun-if-changed=src/mxfp4_gemm_wmma.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_gemm.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_gemm_cutlass.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_gemm_flashinfer.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_gemm_flashinfer_sm103.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_moe_cutlass.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_quant.cu");
    println!("cargo:rerun-if-changed=src/nvfp4_quant_flashinfer.cu");
    println!("cargo:rerun-if-changed=src/flashinfer_cccl_compat.h");
    println!("cargo:rerun-if-changed=src/mlx_nvfp4_utils.cu");
    println!("cargo:rerun-if-changed=src/mxfp4_gemm_cutlass.cu");
    println!("cargo:rerun-if-changed=src/mxfp4_quant.cu");
    println!("cargo:rerun-if-changed=src/gptoss_swiglu.cu");
    println!("cargo:rerun-if-changed=src/silu_and_mul.cu");
    println!("cargo:rerun-if-changed=src/concat_and_cache_mla_kernel.cu");
    println!("cargo:rerun-if-changed=src/mla_paged_attention.cu");
    println!("cargo:rerun-if-changed=src/mla_sparse_attention.cu");
    println!("cargo:rerun-if-changed=src/fast_topk.cu");
    println!("cargo:rerun-if-changed=src/flash/flash_instantiate.cu");
    println!("cargo:rerun-if-changed=src/flash/flash_decode.cu");
    println!("cargo:rerun-if-changed=src/flash/flash_prefill_paged.cuh");
    println!("cargo:rerun-if-changed=src/flash/flash_prefill_paged_fp8.cuh");
    println!("cargo:rerun-if-changed=src/flash/flash_decode_paged.cuh");
    println!("cargo:rerun-if-changed=src/flash/flash_decode_paged_fp8.cuh");
    println!("cargo:rerun-if-changed=src/flash/flash_reshape_cache.cuh");
    println!("cargo:rerun-if-changed=src/flash/flash_turboquant.cuh");
    println!("cargo:rerun-if-changed=src/flash/flash_turboquant_lowbit.cuh");
    println!("cargo:rerun-if-changed=src/flash/flash_prefill_tq4.cuh");
    println!("cargo:rerun-if-changed=src/flash/flash_prefill_tq3.cuh");
    println!("cargo:rerun-if-changed=src/flash/flash_sm_compat.cuh");

    let marlin_disabled = std::env::var("CARGO_FEATURE_NO_MARLIN").is_ok();
    let fp8_kvcache_disabled = std::env::var("CARGO_FEATURE_NO_FP8_KVCACHE").is_ok();
    let trtllm_enabled = std::env::var("CARGO_FEATURE_TRTLLM").is_ok();

    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());

    let mut builder = KernelBuilder::new()
        .source_dir("src")
        .nvcc_thread_patterns(&["flash_api", "flash_decode", "cutlass", "flashinfer"], 2)
        .arg("--expt-relaxed-constexpr")
        .arg("-O3");

    let flash_enabled = std::env::var("CARGO_FEATURE_FLASH").is_ok();

    if !trtllm_enabled {
        builder = builder.exclude(&["trtllm/*"]);
    }

    let compute_cap = builder.get_compute_cap().unwrap_or(80);

    if !flash_enabled {
        builder = builder.exclude(&["flash/*"]);
    } else {
        builder = builder.arg("-Isrc/flash");
        if compute_cap <= 70 {
            println!(
                "cargo:warning=Native flash kernels using m8n8k4 Tensor Core MMA for SM{}.",
                compute_cap
            );
        } else if compute_cap <= 75 {
            println!(
                "cargo:warning=Native flash kernels using FP16 m16n8k8 Tensor Core MMA for SM{}. \
                 SM80+ builds both BF16 and F16 m16n8k16 native flash paths.",
                compute_cap
            );
        } else {
            println!(
                "cargo:warning=Native flash kernels: BF16 + F16 m16n8k16 Tensor Core MMA for SM{}.",
                compute_cap
            );
        }
    }

    println!("cargo:info=compute capability: {:?}", compute_cap);

    if compute_cap < 80 {
        builder = builder.arg("-DNO_BF16_KERNEL");
        builder = builder.arg("-DNO_MARLIN_KERNEL");
    }

    if compute_cap < 89 {
        builder = builder.arg("-DNO_HARDWARE_FP8");
    }

    if compute_cap >= 100 && !std::env::var("NO_HARDWARE_FP4_DECODING").is_ok() {
        builder = builder.arg("-DNVFP4_BLACKWELL");
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
            .with_cutlass(Some("b46b16d003484063bca4ed365e44095c4c6ed633"));

        if compute_cap >= 100 {
            builder = builder
                .arg("-DENABLE_FP4")
                .arg("-DCUTLASS_ENABLE_GDC_FOR_SM100");
        }
        if (100..120).contains(&compute_cap) {
            builder = builder.arg("-DENABLE_FP4_SM100");
            if compute_cap >= 103 {
                // FlashInfer's SM103 header adds an optional Store256 path.
                // Keep it out of SM100 builds, where that implementation and
                // its symbols are not part of the SM100 kernel set.
                builder = builder.arg("-DATTENTION_RS_FLASHINFER_SM103");
            }
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
            let flashinfer_sw_fp8 = std::env::var("ENABLE_FLASHINFER_SOFTWARE_FP8").is_ok();
            if compute_cap >= 90 || (compute_cap >= 80 && flashinfer_sw_fp8) {
                builder = builder.arg("-DFLASHINFER_ENABLE_FP8_E4M3");
            }
            if compute_cap >= 90 {
                builder = builder.arg("-DFLASHINFER_ENABLE_FP4_E2M1");
            }
            if compute_cap == 90 {
                // The FlashInfer delta-rule launcher checks this host-side
                // feature macro before instantiating the SM90A operation.
                // CudaForge emits sm_90a for compute capability 90, but it
                // does not define FLAT_SM90A_ENABLED itself.
                builder = builder
                    .arg("-DATTENTION_RS_ENABLE_FLASHINFER_GDN_PREFILL_SM90")
                    .arg("-DFLAT_SM90A_ENABLED");
            }
        }
    }

    if std::env::var("CARGO_FEATURE_FLASHINFER").is_ok() {
        println!("cargo:rerun-if-changed=src/flashinfer_common.cuh");
        println!("cargo:rerun-if-changed=src/flashinfer_adapter_decode.cu");
        println!("cargo:rerun-if-changed=src/flashinfer_adapter_prefill.cu");
        println!("cargo:rerun-if-changed=src/flashinfer_prefill_fp8_fa2.cu");
        println!("cargo:rerun-if-changed=src/flashinfer_mla.cu");
        // guoqingbao/flashinfer upstream branch: official DSV4 sparse MLA + GQA/FP8 patches
        // Pin: github/upstream @ 0f06c230 (DSV4 sparse + GQA patches + fastdiv/flat compat)
        builder = builder.arg("-DUSE_FLASHINFER").with_git_dependency(
            "flashinfer",
            "https://github.com/guoqingbao/flashinfer.git",
            "0f06c2305a276bcb704277705b32025575cb567f", // upstream + fastdiv/flat compat
            vec![
                "include",
                "include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export",
                "include/flashinfer/trtllm/gemm/trtllmGen_gemm_export",
                "include/flashinfer/attention/sparse_mla_sm120",
                "csrc/nv_internal",
                "csrc/nv_internal/include",
                "csrc/nv_internal/tensorrt_llm/cutlass_extensions/include",
            ],
            vec![
                "csrc/nv_internal/cpp/common",
                "csrc/nv_internal/tensorrt_llm",
                "csrc/sparse_mla_sm120_decode_dsv4.cu",
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

        builder = others::configure_trtllm(
            builder,
            &flashinfer_root,
            &build_dir,
            compute_cap,
            trtllm_enabled,
        )?;
    }

    let (builder, link_flashmla) = others::configure_flashmla(builder, &build_dir, compute_cap)?;
    let mut builder = builder;

    if std::env::var("CARGO_FEATURE_FLASHINFER").is_ok() && compute_cap >= 120 {
        if let Ok(flashinfer_root) = builder.fetch_git_dependency("flashinfer") {
            let csrc_dir = flashinfer_root.join("csrc");
            let quant_cu = csrc_dir.join("nv_internal/cpp/kernels/quantization.cu");
            if quant_cu.exists() {
                // Do not use global nvcc `-include`: extra_args apply to every
                // kernel. A `cuda::maximum` polyfill would then redeclare CCCL
                // types on CUDA 13. Wrap only this translation unit.
                let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
                let compat = manifest.join("src/flashinfer_cccl_compat.h");
                let wrap = build_dir.join("flashinfer_quantization_sm120.cu");
                std::fs::write(
                    &wrap,
                    format!(
                        "#include \"{}\"\n#include \"{}\"\n",
                        unix_path(&compat),
                        unix_path(&quant_cu)
                    ),
                )?;
                builder = builder
                    .arg("-DATTENTION_RS_USE_FLASHINFER_FP4_QUANT")
                    .source_files(vec![wrap]);
                println!(
                    "cargo:warning=FlashInfer SM120 NVFP4 quant (invokeFP4Quantization) enabled"
                );
            }
            let dsv4_cu = csrc_dir.join("sparse_mla_sm120_decode_dsv4.cu");
            if dsv4_cu.exists() {
                builder = builder
                    .arg("-DATTENTION_RS_USE_FLASHINFER_SPARSE_MLA_SM120")
                    .source_files(vec![dsv4_cu]);
                println!("cargo:warning=FlashInfer SM120 DSV4 sparse MLA enabled");
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
        builder = builder.arg("-Xcompiler").arg("-fPIC");
        if compute_cap >= 90 {
            builder = builder.arg("-std=c++20");
        } else {
            builder = builder.arg("-std=c++17");
        }
    }

    println!("cargo:info={builder:?}");

    let _ = builder.build_lib(build_dir.join("libpagedattention.a"))?;

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=pagedattention");
    if link_flashmla {
        println!("cargo:rustc-link-lib=static=flashmla_dsv4");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");

    Ok(())
}
