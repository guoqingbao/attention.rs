/*
 * FlashInfer-style NVFP4 grouped GEMM for MoE.
 *
 * SM120: Uses FlashInfer's CutlassNVFP4GroupwiseScaledGroupGEMMSM120 via
 *        the group_gemm_nvfp4_groupwise_sm120.cuh header.
 * SM100: FlashInfer has no grouped NVFP4 GEMM for SM100.
 *        The caller should fall back to the existing CUTLASS MoE path.
 *
 * For the dense linear path (non-MoE), use nvfp4_gemm_flashinfer.cu instead.
 */

#ifdef ENABLE_FP4

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#if defined(ENABLE_FP4_SM120) && defined(USE_FLASHINFER)

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include <flashinfer/gemm/group_gemm_nvfp4_groupwise_sm120.cuh>

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

using namespace flashinfer;
using namespace flashinfer::group_gemm;

namespace flashinfer {
namespace group_gemm {

INSTANTIATE_GROUP_GEMM_NVFP4_GROUPWISE_SM120(
    128, 128, 128,
    cutlass::float_e2m1_t, cutlass::float_e2m1_t,
    cutlass::float_ue4m3_t, cutlass::float_ue4m3_t,
    cutlass::half_t,
    float_e2m1_t, float_e2m1_t,
    float_ue4m3_t, float_ue4m3_t,
    half_t
)

INSTANTIATE_GROUP_GEMM_NVFP4_GROUPWISE_SM120(
    128, 128, 256,
    cutlass::float_e2m1_t, cutlass::float_e2m1_t,
    cutlass::float_ue4m3_t, cutlass::float_ue4m3_t,
    cutlass::half_t,
    float_e2m1_t, float_e2m1_t,
    float_ue4m3_t, float_ue4m3_t,
    half_t
)

INSTANTIATE_GROUP_GEMM_NVFP4_GROUPWISE_SM120(
    128, 128, 128,
    cutlass::float_e2m1_t, cutlass::float_e2m1_t,
    cutlass::float_ue4m3_t, cutlass::float_ue4m3_t,
    cutlass::bfloat16_t,
    float_e2m1_t, float_e2m1_t,
    float_ue4m3_t, float_ue4m3_t,
    bfloat16_t
)

INSTANTIATE_GROUP_GEMM_NVFP4_GROUPWISE_SM120(
    128, 128, 256,
    cutlass::float_e2m1_t, cutlass::float_e2m1_t,
    cutlass::float_ue4m3_t, cutlass::float_ue4m3_t,
    cutlass::bfloat16_t,
    float_e2m1_t, float_e2m1_t,
    float_ue4m3_t, float_ue4m3_t,
    bfloat16_t
)

}  // namespace group_gemm
}  // namespace flashinfer

namespace {

/*
 * FlashInfer computes per-group SFA offsets with this formula:
 *   sf_m_offset_i = floor((m_offset_i + i * 127) / 128) * 128
 * where m_offset_i = m_indptr[i] (cumulative token count).
 *
 * This differs from our existing formula:
 *   sf_offsets[i] = sum_{j<i} ceil(rows_j / 128) * 128
 *
 * Both ensure 128-aligned offsets, but produce different values when
 * a group's rows aren't a multiple of 128. The FlashInfer kernel
 * internally uses its formula, so we must produce activation scales
 * in the same layout.
 *
 * This kernel builds sf_offsets matching FlashInfer's formula so the
 * existing grouped quantization kernel writes scales to the right places.
 */
__global__ void flashinfer_moe_build_metadata_kernel(
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ weight_global_scales,
    const float* __restrict__ input_scales,
    int32_t* __restrict__ sf_offsets,
    float* __restrict__ alphas,
    float* __restrict__ input_scale_invs,
    int num_experts) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  for (int i = 0; i < num_experts; ++i) {
    int m_offset = expert_offsets[i];
    int sf_m_offset = (static_cast<int64_t>(m_offset) +
                       static_cast<int64_t>(i) * 127) / 128 * 128;
    sf_offsets[i] = static_cast<int32_t>(sf_m_offset);

    float input_scale = input_scales != nullptr ? input_scales[i] : 1.0f;
    float input_scale_inv = input_scale != 0.0f ? 1.0f / input_scale : 1.0f;
    alphas[i] = input_scale * weight_global_scales[i];
    input_scale_invs[i] = input_scale_inv;
  }
}

template <typename DTypeOut>
int run_flashinfer_nvfp4_moe_gemm_sm120(
    const void* gathered_input,
    const void* weights,
    const void* input_sf,
    const void* weight_sf,
    const float* alphas,
    const int32_t* expert_offsets,
    void* output,
    int num_experts, int N, int K,
    int total_rows,
    void* int_workspace, int64_t int_workspace_bytes,
    void* float_workspace, int64_t float_workspace_bytes,
    cudaStream_t stream)
{
    if (num_experts == 0 || total_rows == 0) {
        return 0;
    }

    int device_id = 0;
    cudaGetDevice(&device_id);

    int tile_k = (K >= 512) ? 256 : 128;

    cudaError_t err;
    if (tile_k == 256) {
        err = CutlassNVFP4GroupwiseScaledGroupGEMMSM120<
            128, 128, 256,
            cutlass::float_e2m1_t, cutlass::float_e2m1_t,
            cutlass::float_ue4m3_t, cutlass::float_ue4m3_t,
            DTypeOut>(
            int_workspace, static_cast<size_t>(int_workspace_bytes),
            float_workspace, static_cast<size_t>(float_workspace_bytes),
            static_cast<cutlass::float_e2m1_t*>(const_cast<void*>(gathered_input)),
            static_cast<cutlass::float_e2m1_t*>(const_cast<void*>(weights)),
            static_cast<cutlass::float_ue4m3_t*>(const_cast<void*>(input_sf)),
            static_cast<cutlass::float_ue4m3_t*>(const_cast<void*>(weight_sf)),
            static_cast<DTypeOut*>(output),
            const_cast<float*>(alphas),
            const_cast<int32_t*>(expert_offsets),
            N, K, num_experts, stream, device_id);
    } else {
        err = CutlassNVFP4GroupwiseScaledGroupGEMMSM120<
            128, 128, 128,
            cutlass::float_e2m1_t, cutlass::float_e2m1_t,
            cutlass::float_ue4m3_t, cutlass::float_ue4m3_t,
            DTypeOut>(
            int_workspace, static_cast<size_t>(int_workspace_bytes),
            float_workspace, static_cast<size_t>(float_workspace_bytes),
            static_cast<cutlass::float_e2m1_t*>(const_cast<void*>(gathered_input)),
            static_cast<cutlass::float_e2m1_t*>(const_cast<void*>(weights)),
            static_cast<cutlass::float_ue4m3_t*>(const_cast<void*>(input_sf)),
            static_cast<cutlass::float_ue4m3_t*>(const_cast<void*>(weight_sf)),
            static_cast<DTypeOut*>(output),
            const_cast<float*>(alphas),
            const_cast<int32_t*>(expert_offsets),
            N, K, num_experts, stream, device_id);
    }

    if (err != cudaSuccess) {
        fprintf(stderr, "[FlashInfer NVFP4 MoE SM120] kernel failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

}  // namespace

#endif  // ENABLE_FP4_SM120 && USE_FLASHINFER

extern "C" {

#if defined(ENABLE_FP4_SM120) && defined(USE_FLASHINFER)
void flashinfer_nvfp4_moe_build_metadata(
    const int32_t* expert_offsets,
    const float* weight_global_scales,
    const float* input_scales,
    int32_t* sf_offsets,
    float* alphas,
    float* input_scale_invs,
    int num_experts,
    int64_t stream)
{
    flashinfer_moe_build_metadata_kernel<<<1, 1, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        expert_offsets, weight_global_scales, input_scales,
        sf_offsets, alphas, input_scale_invs, num_experts);
}
#else
void flashinfer_nvfp4_moe_build_metadata(
    const int32_t*, const float*, const float*,
    int32_t*, float*, float*, int, int64_t) {}
#endif

int flashinfer_nvfp4_moe_gemm_f16(
    const void* gathered_input,
    const void* weights,
    const void* input_sf,
    const void* weight_sf,
    const float* alphas,
    const int32_t* expert_offsets,
    void* output,
    int num_experts, int N, int K,
    int total_rows,
    void* int_workspace, int64_t int_workspace_bytes,
    void* float_workspace, int64_t float_workspace_bytes,
    int64_t stream)
{
#if defined(ENABLE_FP4_SM120) && defined(USE_FLASHINFER)
    return run_flashinfer_nvfp4_moe_gemm_sm120<cutlass::half_t>(
        gathered_input, weights, input_sf, weight_sf,
        alphas, expert_offsets, output,
        num_experts, N, K, total_rows,
        int_workspace, int_workspace_bytes,
        float_workspace, float_workspace_bytes,
        reinterpret_cast<cudaStream_t>(stream));
#else
    (void)gathered_input; (void)weights; (void)input_sf; (void)weight_sf;
    (void)alphas; (void)expert_offsets; (void)output;
    (void)num_experts; (void)N; (void)K; (void)total_rows;
    (void)int_workspace; (void)int_workspace_bytes;
    (void)float_workspace; (void)float_workspace_bytes;
    (void)stream;
    return -1;
#endif
}

int flashinfer_nvfp4_moe_gemm_bf16(
    const void* gathered_input,
    const void* weights,
    const void* input_sf,
    const void* weight_sf,
    const float* alphas,
    const int32_t* expert_offsets,
    void* output,
    int num_experts, int N, int K,
    int total_rows,
    void* int_workspace, int64_t int_workspace_bytes,
    void* float_workspace, int64_t float_workspace_bytes,
    int64_t stream)
{
#if defined(ENABLE_FP4_SM120) && defined(USE_FLASHINFER)
    return run_flashinfer_nvfp4_moe_gemm_sm120<cutlass::bfloat16_t>(
        gathered_input, weights, input_sf, weight_sf,
        alphas, expert_offsets, output,
        num_experts, N, K, total_rows,
        int_workspace, int_workspace_bytes,
        float_workspace, float_workspace_bytes,
        reinterpret_cast<cudaStream_t>(stream));
#else
    (void)gathered_input; (void)weights; (void)input_sf; (void)weight_sf;
    (void)alphas; (void)expert_offsets; (void)output;
    (void)num_experts; (void)N; (void)K; (void)total_rows;
    (void)int_workspace; (void)int_workspace_bytes;
    (void)float_workspace; (void)float_workspace_bytes;
    (void)stream;
    return -1;
#endif
}

}  // extern "C"

#else  // !ENABLE_FP4

extern "C" {

void flashinfer_nvfp4_moe_build_metadata(
    const int32_t*, const float*, const float*,
    int32_t*, float*, float*, int, int64_t) {}

int flashinfer_nvfp4_moe_gemm_f16(
    const void*, const void*, const void*, const void*,
    const float*, const int32_t*, void*,
    int, int, int, int,
    void*, int64_t, void*, int64_t, int64_t)
{ return -1; }

int flashinfer_nvfp4_moe_gemm_bf16(
    const void*, const void*, const void*, const void*,
    const float*, const int32_t*, void*,
    int, int, int, int,
    void*, int64_t, void*, int64_t, int64_t)
{ return -1; }

}  // extern "C"

#endif  // ENABLE_FP4
