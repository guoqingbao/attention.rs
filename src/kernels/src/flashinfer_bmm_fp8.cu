#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(USE_FLASHINFER) && defined(ATTENTION_RS_USE_FLASHINFER_BLOCKSCALE) && \
    defined(FLASHINFER_ENABLE_FP8_E4M3)
#include "tensorrt_llm/deep_gemm/compiler.cuh"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"

namespace {

using Runner = tensorrt_llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<
    __nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>;
using RunnerFp8 = tensorrt_llm::kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<
    __nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;

thread_local std::unique_ptr<Runner> g_runner;
thread_local std::unique_ptr<RunnerFp8> g_runner_fp8;
thread_local bool g_runner_initialized = false;

Runner& get_runner() {
  if (!g_runner) {
    g_runner = std::make_unique<Runner>();
  }
  if (!g_runner_initialized) {
    deep_gemm::jit::Compiler::setIncludeDirs(
        {std::filesystem::path(ATTENTION_RS_FLASHINFER_TRTLLM_INCLUDE_DIR)});
    g_runner_initialized = true;
  }
  return *g_runner;
}

RunnerFp8& get_runner_fp8() {
  if (!g_runner_fp8) {
    g_runner_fp8 = std::make_unique<RunnerFp8>();
  }
  if (!g_runner_initialized) {
    deep_gemm::jit::Compiler::setIncludeDirs(
        {std::filesystem::path(ATTENTION_RS_FLASHINFER_TRTLLM_INCLUDE_DIR)});
    g_runner_initialized = true;
  }
  return *g_runner_fp8;
}

}  // namespace

extern "C" size_t flashinfer_fp8_blockscale_workspace_size_bf16(int m, int n, int k) {
  try {
    if (m <= 0 || n <= 0 || k <= 0) {
      return 0;
    }
    return get_runner().getWorkspaceSize(static_cast<size_t>(m), static_cast<size_t>(n),
                                         static_cast<size_t>(k), 1, 1);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "flashinfer_fp8_blockscale_workspace_size_bf16 failed: %s\n", e.what());
    return 0;
  }
}

extern "C" int flashinfer_fp8_blockscale_bf16(const void* input, const void* weight,
                                              const float* weight_scale, void* output, int m,
                                              int n, int k, void* workspace,
                                              size_t workspace_size, int64_t stream_) {
  try {
    if (input == nullptr || weight == nullptr || weight_scale == nullptr || output == nullptr) {
      return 1;
    }

    auto& runner = get_runner();
    if (workspace == nullptr && workspace_size > 0) {
      return 2;
    }
    if (workspace_size > 0) {
      runner.configureWorkspace(reinterpret_cast<char*>(workspace));
    }
    runner.gemm(output, input, weight, m, n, k, reinterpret_cast<cudaStream_t>(stream_), nullptr,
                weight_scale);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "flashinfer_fp8_blockscale_bf16 failed: %s\n", e.what());
    return -1;
  } catch (...) {
    std::fprintf(stderr, "flashinfer_fp8_blockscale_bf16 failed with unknown exception\n");
    return -2;
  }
}

extern "C" size_t flashinfer_fp8_blockscale_workspace_size_fp8(int m, int n, int k) {
  try {
    if (m <= 0 || n <= 0 || k <= 0) {
      return 0;
    }
    return get_runner_fp8().getWorkspaceSize(static_cast<size_t>(m), static_cast<size_t>(n),
                                             static_cast<size_t>(k), 1, 1);
  } catch (std::exception const& e) {
    std::fprintf(stderr, "flashinfer_fp8_blockscale_workspace_size_fp8 failed: %s\n", e.what());
    return 0;
  }
}

extern "C" int flashinfer_fp8_blockscale_fp8(const void* input, const float* input_scale,
                                             const void* weight, const float* weight_scale,
                                             void* output, int m, int n, int k, void* workspace,
                                             size_t workspace_size, int64_t stream_) {
  try {
    if (input == nullptr || input_scale == nullptr || weight == nullptr || weight_scale == nullptr ||
        output == nullptr) {
      return 1;
    }

    auto& runner = get_runner_fp8();
    if (workspace == nullptr && workspace_size > 0) {
      return 2;
    }
    if (workspace_size > 0) {
      runner.configureWorkspace(reinterpret_cast<char*>(workspace));
    }

    runner.gemm(reinterpret_cast<__nv_fp8_e4m3 const*>(input), k,
                reinterpret_cast<__nv_fp8_e4m3 const*>(weight), k,
                reinterpret_cast<__nv_bfloat16*>(output), n, m, n, k, input_scale, weight_scale,
                reinterpret_cast<cudaStream_t>(stream_));
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "flashinfer_fp8_blockscale_fp8 failed: %s\n", e.what());
    return -1;
  } catch (...) {
    std::fprintf(stderr, "flashinfer_fp8_blockscale_fp8 failed with unknown exception\n");
    return -2;
  }
}

#else

extern "C" size_t flashinfer_fp8_blockscale_workspace_size_bf16(int, int, int) { return 0; }

extern "C" int flashinfer_fp8_blockscale_bf16(const void*, const void*, const float*, void*, int,
                                              int, int, void*, size_t, int64_t) {
  return -1;
}

extern "C" size_t flashinfer_fp8_blockscale_workspace_size_fp8(int, int, int) { return 0; }

extern "C" int flashinfer_fp8_blockscale_fp8(const void*, const float*, const void*, const float*,
                                             void*, int, int, int, void*, size_t, int64_t) {
  return -1;
}

#endif
