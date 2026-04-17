/*
 * FlashInfer NVFP4 CUTLASS GEMM wrapper for attention.rs.
 *
 * This file is a thin C-ABI wrapper around FlashInfer's CutlassFp4GemmRunner.
 * It #includes the FlashInfer template headers (fetched by build.rs) and
 * instantiates the CUTLASS kernel templates for the required tile/cluster
 * configurations.
 *
 * Scale layout contract (matches our nvfp4_quant.cu swizzle output):
 *   input_sf:  swizzled [ceil(M/128)*128, ceil(K/sfVecSize/4)*4] UE4M3
 *   weight_sf: swizzled [ceil(N/128)*128, ceil(K/sfVecSize/4)*4] UE4M3
 *   global_sf: single float = input_scale * weight_global_scale
 *
 * Both our swizzle kernel and FlashInfer's runner use
 * Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB internally, so the
 * 128×4 swizzled layout is identical.
 */

#if defined(ENABLE_FP4) && defined(USE_FLASHINFER)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <vector>

// ============================================================================
// SM100/SM103 path (Blackwell)
// fp4_gemm_cutlass_template_sm103.h is the superset: it includes both
// fp4_gemm_template_sm100.h and fp4_gemm_template_sm103.h, and provides
// the CutlassFp4GemmRunner with dispatch to both SM100 and SM103 kernels.
// ============================================================================

#if defined(ENABLE_FP4_SM100)

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_cutlass.h"
#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm103.h"

namespace flashinfer { namespace gemm {

// SM100 tile configs: 128×64×128, 128×256×128, 128×128×256, 128×256×256
// Each with all 9 cluster shapes: {1,2,4}×{1,2,4}×1

#define INSTANTIATE_ALL_CLUSTERS(TYPE, CTA_M, CTA_N, CTA_K) \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 1, 1, 1, _1SM) \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 1, 2, 1, _1SM) \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 1, 4, 1, _1SM) \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 2, 1, 1, _2SM) \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 2, 2, 1, _2SM) \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 2, 4, 1, _2SM) \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 4, 1, 1, _2SM) \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 4, 2, 1, _2SM) \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 4, 4, 1, _2SM)

INSTANTIATE_ALL_CLUSTERS(half, 128, 64, 128)
INSTANTIATE_ALL_CLUSTERS(half, 128, 256, 128)
INSTANTIATE_ALL_CLUSTERS(half, 128, 128, 256)
INSTANTIATE_ALL_CLUSTERS(half, 128, 256, 256)

INSTANTIATE_ALL_CLUSTERS(__nv_bfloat16, 128, 64, 128)
INSTANTIATE_ALL_CLUSTERS(__nv_bfloat16, 128, 256, 128)
INSTANTIATE_ALL_CLUSTERS(__nv_bfloat16, 128, 128, 256)
INSTANTIATE_ALL_CLUSTERS(__nv_bfloat16, 128, 256, 256)

#undef INSTANTIATE_ALL_CLUSTERS

// SM103 "Ultra" tile configs: 128×128×768, 128×192×768, 128×256×768
// These use a different CUTLASS kernel path with enhanced register file.

#define INSTANTIATE_ALL_CLUSTERS_SM103(TYPE, CTA_M, CTA_N, CTA_K) \
  INSTANTIATE_FP4_ULTRA_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 1, 1, 1, _1SM_sm103) \
  INSTANTIATE_FP4_ULTRA_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 1, 2, 1, _1SM_sm103) \
  INSTANTIATE_FP4_ULTRA_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 1, 4, 1, _1SM_sm103) \
  INSTANTIATE_FP4_ULTRA_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 2, 1, 1, _2SM_sm103) \
  INSTANTIATE_FP4_ULTRA_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 2, 2, 1, _2SM_sm103) \
  INSTANTIATE_FP4_ULTRA_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 2, 4, 1, _2SM_sm103) \
  INSTANTIATE_FP4_ULTRA_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 4, 1, 1, _2SM_sm103) \
  INSTANTIATE_FP4_ULTRA_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 4, 2, 1, _2SM_sm103) \
  INSTANTIATE_FP4_ULTRA_GEMM_KERNEL_LAUNCHER(TYPE, CTA_M, CTA_N, CTA_K, 4, 4, 1, _2SM_sm103)

INSTANTIATE_ALL_CLUSTERS_SM103(half, 128, 128, 768)
INSTANTIATE_ALL_CLUSTERS_SM103(half, 128, 192, 768)
INSTANTIATE_ALL_CLUSTERS_SM103(half, 128, 256, 768)

INSTANTIATE_ALL_CLUSTERS_SM103(__nv_bfloat16, 128, 128, 768)
INSTANTIATE_ALL_CLUSTERS_SM103(__nv_bfloat16, 128, 192, 768)
INSTANTIATE_ALL_CLUSTERS_SM103(__nv_bfloat16, 128, 256, 768)

#undef INSTANTIATE_ALL_CLUSTERS_SM103

template class CutlassFp4GemmRunner<half, FP4GemmType::W4A4_NVFP4_NVFP4>;
template class CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4>;

} // namespace gemm
} // namespace flashinfer

#endif // ENABLE_FP4_SM100


// ============================================================================
// SM120 path (Rubin)
// fp4_gemm_cutlass_template_sm120.h provides the CutlassFp4GemmRunner with
// DP and StreamK schedulers. 1×1×1 cluster only.
// ============================================================================

#if defined(ENABLE_FP4_SM120)

#include "flashinfer/gemm/cutlass_gemm_configs.h"
#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"

namespace flashinfer { namespace gemm {

// SM120 tiles: 128×128×128, 128×128×256, 256×128×128
// 1×1×1 cluster only

INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(half, 128, 128, 128, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(half, 128, 128, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(half, 256, 128, 128, 1, 1, 1, _1SM)

INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 128, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 256, 128, 128, 1, 1, 1, _1SM)

template class CutlassFp4GemmRunner<half, FP4GemmType::W4A4_NVFP4_NVFP4>;
template class CutlassFp4GemmRunner<__nv_bfloat16, FP4GemmType::W4A4_NVFP4_NVFP4>;

} // namespace gemm
} // namespace flashinfer

#endif // ENABLE_FP4_SM120


// ============================================================================
// C ABI wrappers
// ============================================================================

#if defined(ENABLE_FP4_SM100) || defined(ENABLE_FP4_SM120)

using flashinfer::gemm::CutlassFp4GemmRunner;
using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::FP4GemmType;

namespace {

template <typename T>
void run_flashinfer_fp4_gemm(
    void* D, const void* A, const void* B,
    const void* input_sf, const void* weight_sf,
    const float* global_sf,
    int m, int n, int k,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream)
{
  CutlassFp4GemmRunner<T, FP4GemmType::W4A4_NVFP4_NVFP4> runner;

  auto configs = runner.getConfigs();
  if (configs.empty()) return;

  // Tactic 0 is the best from FlashInfer's profiling-based reordering
  CutlassGemmConfig config = configs[0];

  size_t required_ws = runner.getWorkspaceSize(m, n, k, 1);
  if (required_ws > workspace_bytes) {
    fprintf(stderr, "[FlashInfer FP4] workspace too small: need %zu, have %zu\n",
            required_ws, workspace_bytes);
    return;
  }

  try {
    runner.gemm(D, A, B, input_sf, weight_sf, global_sf,
                m, n, k, /*batch_count=*/1, config,
                reinterpret_cast<char*>(workspace), workspace_bytes, stream);
  } catch (const std::exception& e) {
    fprintf(stderr, "[FlashInfer FP4] GEMM failed: %s (M=%d N=%d K=%d)\n",
            e.what(), m, n, k);
  }
}

} // namespace

#endif // ENABLE_FP4_SM100 || ENABLE_FP4_SM120


extern "C" {

void flashinfer_nvfp4_cutlass_gemm_f16(
    const void* input,
    const void* weight,
    const void* input_sf,
    const void* weight_sf,
    const float* global_sf,
    void* output,
    int M, int N, int K,
    void* workspace, int64_t workspace_bytes,
    int64_t stream)
{
#if defined(ENABLE_FP4_SM100) || defined(ENABLE_FP4_SM120)
  run_flashinfer_fp4_gemm<half>(
      output, input, weight, input_sf, weight_sf, global_sf,
      M, N, K, workspace, static_cast<size_t>(workspace_bytes),
      reinterpret_cast<cudaStream_t>(stream));
#else
  (void)input; (void)weight; (void)input_sf; (void)weight_sf;
  (void)global_sf; (void)output; (void)M; (void)N; (void)K;
  (void)workspace; (void)workspace_bytes; (void)stream;
#endif
}

void flashinfer_nvfp4_cutlass_gemm_bf16(
    const void* input,
    const void* weight,
    const void* input_sf,
    const void* weight_sf,
    const float* global_sf,
    void* output,
    int M, int N, int K,
    void* workspace, int64_t workspace_bytes,
    int64_t stream)
{
#if defined(ENABLE_FP4_SM100) || defined(ENABLE_FP4_SM120)
  run_flashinfer_fp4_gemm<__nv_bfloat16>(
      output, input, weight, input_sf, weight_sf, global_sf,
      M, N, K, workspace, static_cast<size_t>(workspace_bytes),
      reinterpret_cast<cudaStream_t>(stream));
#else
  (void)input; (void)weight; (void)input_sf; (void)weight_sf;
  (void)global_sf; (void)output; (void)M; (void)N; (void)K;
  (void)workspace; (void)workspace_bytes; (void)stream;
#endif
}

} // extern "C"

#else // !(ENABLE_FP4 && USE_FLASHINFER)

extern "C" {

void flashinfer_nvfp4_cutlass_gemm_f16(
    const void*, const void*, const void*, const void*,
    const float*, void*, int, int, int,
    void*, int64_t, int64_t)
{
}

void flashinfer_nvfp4_cutlass_gemm_bf16(
    const void*, const void*, const void*, const void*,
    const float*, void*, int, int, int,
    void*, int64_t, int64_t)
{
}

} // extern "C"

#endif // ENABLE_FP4 && USE_FLASHINFER
