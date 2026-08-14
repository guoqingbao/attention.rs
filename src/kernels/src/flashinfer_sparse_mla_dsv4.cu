// FlashInfer SM120 DeepSeek V4 sparse MLA decode wrapper.
// Compiles the launch entry when ATTENTION_RS_USE_FLASHINFER_SPARSE_MLA_SM120
// is set (compute_cap >= 120). On other archs this is a stub.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

#if defined(ATTENTION_RS_USE_FLASHINFER_SPARSE_MLA_SM120) && defined(USE_FLASHINFER)

#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

namespace flashinfer::sparse_mla_sm120 {
bool launch_sparse_mla_decode_dsv4(ModelType mt, int num_heads, int topk,
                                   int page_block_size, int num_tokens, int num_splits,
                                   const bf16* Q, const uint8_t* KV_cache,
                                   const int32_t* indices, bf16* mid_out, float* mid_lse,
                                   bf16* output, float* out_lse, const int* topk_length,
                                   const float* attn_sink, const uint8_t* extra_KV_cache,
                                   const int32_t* extra_indices, const int* extra_topk_length,
                                   int extra_topk, int pbs_extra, size_t stride_extra_kv_block,
                                   int chunks_per_block_override, float sm_scale,
                                   size_t stride_kv_block, cudaStream_t stream);
}

static int flashinfer_dsv4_sm120_kernel_topk(int topk) {
  if (topk <= 128) return 128;
  if (topk <= 512) return 512;
  if (topk <= 1024) return 1024;
  return 0;
}

extern "C" int flashinfer_dsv4_sparse_sm120_supported(int num_heads, int topk) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
  // FlashInfer calls this the SM12x sparse backend.  Do not make the model
  // dispatch depend on an exact minor number: CUDA 13.x and early SM120
  // drivers have reported the same SM120 device through slightly different
  // minor values.  The actual launch still returns the CUDA error if a
  // future SM12x device is not executable by this cubin.
  if (prop.major != 12) return 0;
  const int heads_ok =
      num_heads == 8 || num_heads == 16 || num_heads == 32 || num_heads == 64 ||
      num_heads == 128;
  return heads_ok && flashinfer_dsv4_sm120_kernel_topk(topk) != 0 ? 1 : 0;
}

extern "C" int flashinfer_dsv4_sparse_sm120_compiled() { return 1; }

extern "C" int flashinfer_dsv4_sparse_decode_sm120(
    const void* q_bf16, const void* kv_fp8, const int* indices, const int* topk_length,
    const float* attn_sink, void* out_bf16, float* out_lse, void* mid_out_bf16,
    float* mid_lse, const void* extra_kv_fp8, const int* extra_indices,
    const int* extra_topk_length, int num_tokens, int num_heads, int topk, int num_splits,
    int page_block_size, int extra_topk, int extra_page_block_size,
    int chunks_per_block_override, float sm_scale, cudaStream_t stream) {
  const int dispatch_topk = flashinfer_dsv4_sm120_kernel_topk(topk);
  if (dispatch_topk == 0 || !flashinfer_dsv4_sparse_sm120_supported(num_heads, topk)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (!q_bf16 || !kv_fp8 || !indices || !out_bf16 || !mid_out_bf16 || !mid_lse ||
      !topk_length) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (page_block_size != 64) {
    // FlashInfer DSV4 decode instantiations are page_block_size=64 only.
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr size_t kBytesPerToken = 584;
  size_t stride_kv_block = static_cast<size_t>(page_block_size) * kBytesPerToken;
  size_t stride_extra =
      extra_kv_fp8 ? static_cast<size_t>(extra_page_block_size) * kBytesPerToken : 0;

  bool ok = flashinfer::sparse_mla_sm120::launch_sparse_mla_decode_dsv4(
      ModelType::DSV4, num_heads, dispatch_topk, page_block_size, num_tokens, num_splits,
      reinterpret_cast<const bf16*>(q_bf16), reinterpret_cast<const uint8_t*>(kv_fp8),
      indices, reinterpret_cast<bf16*>(mid_out_bf16), mid_lse,
      reinterpret_cast<bf16*>(out_bf16), out_lse, topk_length, attn_sink,
      reinterpret_cast<const uint8_t*>(extra_kv_fp8), extra_indices, extra_topk_length,
      extra_topk, extra_page_block_size, stride_extra, chunks_per_block_override, sm_scale,
      stride_kv_block, stream);
  if (!ok) return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(cudaGetLastError());
}

#else

extern "C" int flashinfer_dsv4_sparse_sm120_supported(int, int) { return 0; }

extern "C" int flashinfer_dsv4_sparse_sm120_compiled() { return 0; }

extern "C" int flashinfer_dsv4_sparse_decode_sm120(
    const void*, const void*, const int*, const int*, const float*, void*, float*, void*,
    float*, const void*, const int*, const int*, int, int, int, int, int, int, int, int,
    float, cudaStream_t) {
  return static_cast<int>(cudaErrorNotSupported);
}

#endif
