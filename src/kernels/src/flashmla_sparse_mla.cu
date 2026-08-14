// Torch-free FlashMLA sparse MLA wrappers for DeepSeek V4 (MODEL1 / d_qk=512).
// SM90 decode: FP8 FOOTER paged KV. Prefill: BF16 contiguous KV (FlashMLA API).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#if defined(ATTENTION_RS_USE_FLASHMLA)

#include "params.h"
#include "sm90/decode/sparse_fp8/splitkv_mla.h"
#include "sm90/prefill/sparse/phase1.h"
#include "smxx/decode/combine/combine.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

namespace {

constexpr float kLog2e = 1.4426950408889634f;
constexpr int kBytesPerToken = 584;
constexpr int kHeadDim = 512;
constexpr int kHeadDimV = 512;

struct DeviceArch {
  int major = 0;
  int minor = 0;
  int num_sms = 0;
  bool ok = false;

  DeviceArch() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return;
    major = prop.major;
    minor = prop.minor;
    num_sms = prop.multiProcessorCount;
    ok = true;
  }

  bool is_sm90a() const { return major == 9; }
  bool is_sm120() const { return major == 12; }
};

inline int safe_stride(int64_t s) {
  if (s > static_cast<int64_t>(std::numeric_limits<int>::max())) return -1;
  return static_cast<int>(s);
}

}  // namespace

extern "C" int flashmla_dsv4_supported(int num_heads) {
  DeviceArch arch;
  if (!arch.ok || !arch.is_sm90a()) return 0;
  // FlashMLA MODEL1 sparse FP8 instantiates h_q ∈ {64, 128}. TP shards with
  // fewer local heads (e.g. 32 on TP=2) are padded up to 64 in the Rust wrapper
  // (same as SGLang deepseek_v4).
  if (num_heads <= 0) return 0;
  if (num_heads <= 64 || num_heads == 128) return 1;
  return 0;
}

extern "C" int flashmla_dsv4_bytes_per_token() { return kBytesPerToken; }

extern "C" int flashmla_dsv4_decode_workspace_bytes(
    int batch_size, int s_q, int num_heads, int* num_sm_parts_out,
    size_t* tile_meta_bytes_out, size_t* num_splits_bytes_out,
    size_t* lse_accum_bytes_out, size_t* o_accum_bytes_out) {
  DeviceArch arch;
  if (!arch.ok || !arch.is_sm90a()) return static_cast<int>(cudaErrorNotSupported);
  // Caller must pass FlashMLA specialization width (64 or 128), after TP pad.
  if (num_heads != 64 && num_heads != 128) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int heads_per_64 = std::max(1, num_heads / 64);
  int num_sm_parts = std::max(arch.num_sms / std::max(s_q, 1) / heads_per_64, 1);
  int total_splits = batch_size + num_sm_parts;
  if (num_sm_parts_out) *num_sm_parts_out = num_sm_parts;
  if (tile_meta_bytes_out)
    *tile_meta_bytes_out =
        static_cast<size_t>(num_sm_parts) * (sizeof(DecodingSchedMeta) / 4) * sizeof(int);
  if (num_splits_bytes_out)
    *num_splits_bytes_out = static_cast<size_t>(batch_size + 1) * sizeof(int);
  if (lse_accum_bytes_out)
    *lse_accum_bytes_out =
        static_cast<size_t>(total_splits) * s_q * num_heads * sizeof(float);
  if (o_accum_bytes_out)
    *o_accum_bytes_out = static_cast<size_t>(total_splits) * s_q * num_heads *
                         kHeadDimV * sizeof(float);
  return 0;
}

extern "C" int flashmla_dsv4_sparse_decode(
    const void* q_bf16,             // [b, s_q, h_q, 512]
    const void* kv_fp8,             // [num_blocks, page_block_size, 1, 584] FOOTER
    const int* indices,             // [b, s_q, topk]
    const int* topk_length,         // [b] or nullptr
    const float* attn_sink,         // [h_q] or nullptr
    void* out_bf16,                 // [b, s_q, h_q, 512]
    float* lse,                     // [b, s_q, h_q] or nullptr
    const void* extra_kv_fp8,       // optional
    const int* extra_indices,       // optional [b,s_q,extra_topk]
    const int* extra_topk_length,   // optional [b]
    int* tile_scheduler_metadata,   // workspace
    int* num_splits,                // workspace [b+1]
    float* lse_accum,               // workspace
    float* o_accum,                 // workspace
    int batch_size, int s_q, int num_heads, int topk, int num_blocks,
    int page_block_size, int extra_num_blocks, int extra_page_block_size,
    int extra_topk, int num_sm_parts, float sm_scale, cudaStream_t stream) {
  DeviceArch arch;
  if (!arch.ok || !arch.is_sm90a()) return static_cast<int>(cudaErrorNotSupported);
  if (!q_bf16 || !kv_fp8 || !indices || !out_bf16 || !tile_scheduler_metadata ||
      !num_splits || !lse_accum || !o_accum) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((num_heads != 64 && num_heads != 128) || batch_size <= 0 || s_q <= 0 ||
      topk <= 0 || page_block_size <= 0 || num_sm_parts <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((extra_kv_fp8 != nullptr) != (extra_indices != nullptr)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  using bf16 = cutlass::bfloat16_t;
  bool have_extra = extra_kv_fp8 != nullptr;

  // Local LSE if caller did not provide one.
  float* lse_ptr = lse;
  // Caller must provide lse buffer; allocate is not graph-safe.
  if (!lse_ptr) return static_cast<int>(cudaErrorInvalidValue);

  SparseAttnDecodeParams params{};
  params.b = batch_size;
  params.s_q = s_q;
  params.h_q = num_heads;
  params.h_kv = 1;
  params.d_qk = kHeadDim;
  params.d_v = kHeadDimV;
  params.sm_scale = sm_scale;
  params.sm_scale_div_log2 = sm_scale * kLog2e;
  params.num_blocks = num_blocks;
  params.page_block_size = page_block_size;
  params.topk = topk;
  params.model_type = ModelType::MODEL1;

  params.q = reinterpret_cast<bf16*>(const_cast<void*>(q_bf16));
  params.kv = reinterpret_cast<bf16*>(const_cast<void*>(kv_fp8));
  params.indices = const_cast<int*>(indices);
  params.topk_length = const_cast<int*>(topk_length);
  params.attn_sink = const_cast<float*>(attn_sink);
  params.lse = lse_ptr;
  params.out = reinterpret_cast<bf16*>(out_bf16);

  params.extra_num_blocks = have_extra ? extra_num_blocks : 0;
  params.extra_page_block_size = have_extra ? extra_page_block_size : 0;
  params.extra_topk = have_extra ? extra_topk : 0;
  params.extra_kv =
      have_extra ? reinterpret_cast<bf16*>(const_cast<void*>(extra_kv_fp8)) : nullptr;
  params.extra_indices = have_extra ? const_cast<int*>(extra_indices) : nullptr;
  params.extra_topk_length =
      have_extra ? const_cast<int*>(extra_topk_length) : nullptr;

  // Contiguous layout strides.
  params.stride_q_b = s_q * num_heads * kHeadDim;
  params.stride_q_s_q = num_heads * kHeadDim;
  params.stride_q_h_q = kHeadDim;
  params.stride_kv_block = page_block_size * kBytesPerToken;
  params.stride_kv_row = kBytesPerToken;
  params.stride_indices_b = s_q * topk;
  params.stride_indices_s_q = topk;
  params.stride_lse_b = s_q * num_heads;
  params.stride_lse_s_q = num_heads;
  params.stride_o_b = s_q * num_heads * kHeadDimV;
  params.stride_o_s_q = num_heads * kHeadDimV;
  params.stride_o_h_q = kHeadDimV;
  if (have_extra) {
    params.stride_extra_kv_block = extra_page_block_size * kBytesPerToken;
    params.stride_extra_kv_row = kBytesPerToken;
    params.stride_extra_indices_b = s_q * extra_topk;
    params.stride_extra_indices_s_q = extra_topk;
  }
  params.stream = stream;

  GetDecodeSchedMetaParams meta_params{};
  meta_params.b = batch_size;
  meta_params.s_q = s_q;
  meta_params.block_size_n = 64;
  meta_params.fixed_overhead_num_blocks = 5;
  meta_params.topk = topk;
  meta_params.extra_topk = have_extra ? extra_topk : -1;
  meta_params.topk_length = const_cast<int*>(topk_length);
  meta_params.extra_topk_length =
      have_extra ? const_cast<int*>(extra_topk_length) : nullptr;
  meta_params.seqlens_k_ptr = nullptr;
  meta_params.tile_scheduler_metadata_ptr =
      reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata);
  meta_params.num_splits_ptr = num_splits;
  meta_params.num_sm_parts = num_sm_parts;
  meta_params.stream = stream;
  smxx::decode::run_get_decoding_sched_meta_kernel(meta_params);

  params.tile_scheduler_metadata_ptr =
      reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata);
  params.num_splits_ptr = num_splits;
  params.num_sm_parts = num_sm_parts;

  const int total_num_splits = batch_size + num_sm_parts;
  params.lse_accum = lse_accum;
  params.o_accum = o_accum;
  params.stride_lse_accum_split = s_q * num_heads;
  params.stride_lse_accum_s_q = num_heads;
  params.stride_o_accum_split = s_q * num_heads * kHeadDimV;
  params.stride_o_accum_s_q = num_heads * kHeadDimV;
  params.stride_o_accum_h_q = kHeadDimV;
  (void)total_num_splits;

  if (num_heads == 64) {
    sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1,
                                                                      64>(params);
  } else {
    sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1,
                                                                      128>(params);
  }

  CombineParams combine{};
  combine.b = batch_size;
  combine.s_q = s_q;
  combine.h_q = num_heads;
  combine.d_v = kHeadDimV;
  combine.lse = params.lse;
  combine.out = params.out;
  combine.stride_lse_b = params.stride_lse_b;
  combine.stride_lse_s_q = params.stride_lse_s_q;
  combine.stride_o_b = params.stride_o_b;
  combine.stride_o_s_q = params.stride_o_s_q;
  combine.stride_o_h_q = params.stride_o_h_q;
  combine.lse_accum = params.lse_accum;
  combine.o_accum = params.o_accum;
  combine.stride_lse_accum_split = params.stride_lse_accum_split;
  combine.stride_lse_accum_s_q = params.stride_lse_accum_s_q;
  combine.stride_o_accum_split = params.stride_o_accum_split;
  combine.stride_o_accum_s_q = params.stride_o_accum_s_q;
  combine.stride_o_accum_h_q = params.stride_o_accum_h_q;
  combine.tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
  combine.num_splits_ptr = params.num_splits_ptr;
  combine.num_sm_parts = params.num_sm_parts;
  combine.attn_sink = params.attn_sink;
  combine.stream = stream;
  smxx::decode::run_flash_mla_combine_kernel<bf16>(combine);

  return static_cast<int>(cudaGetLastError());
}

extern "C" int flashmla_dsv4_sparse_prefill(
    const void* q_bf16,      // [s_q, h_q, 512]
    const void* kv_bf16,     // [s_kv, 1, 512]
    const int* indices,      // [s_q, 1, topk]
    const float* attn_sink,  // [h_q] or nullptr
    const int* topk_length,  // [s_q] or nullptr
    void* out_bf16,          // [s_q, h_q, 512]
    float* lse,              // [s_q, h_q]
    float* max_logits,       // [s_q, h_q]
    int s_q, int s_kv, int num_heads, int topk, float sm_scale,
    cudaStream_t stream) {
  DeviceArch arch;
  if (!arch.ok || !arch.is_sm90a()) return static_cast<int>(cudaErrorNotSupported);
  if (!q_bf16 || !kv_bf16 || !indices || !out_bf16 || !lse || !max_logits) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((num_heads != 64 && num_heads != 128) || s_q <= 0 || s_kv <= 0 || topk <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  using bf16 = cutlass::bfloat16_t;
  SparseAttnFwdParams params{};
  params.s_q = s_q;
  params.s_kv = s_kv;
  params.h_q = num_heads;
  params.h_kv = 1;
  params.d_qk = kHeadDim;
  params.d_v = kHeadDimV;
  params.topk = topk;
  params.sm_scale = sm_scale;
  params.sm_scale_div_log2 = sm_scale * kLog2e;
  params.q = reinterpret_cast<bf16*>(const_cast<void*>(q_bf16));
  params.kv = reinterpret_cast<bf16*>(const_cast<void*>(kv_bf16));
  params.indices = const_cast<int*>(indices);
  params.attn_sink = const_cast<float*>(attn_sink);
  params.topk_length = const_cast<int*>(topk_length);
  params.stride_q_s_q = num_heads * kHeadDim;
  params.stride_q_h_q = kHeadDim;
  params.stride_kv_s_kv = kHeadDim;
  params.stride_kv_h_kv = kHeadDim;
  params.stride_indices_s_q = topk;
  params.stride_indices_h_kv = topk;
  params.out = reinterpret_cast<bf16*>(out_bf16);
  params.max_logits = max_logits;
  params.lse = lse;
  params.num_sm = arch.num_sms;
  params.stream = stream;

  if (topk_length != nullptr) {
    sm90::fwd::run_fwd_phase1_kernel<512, true>(params);
  } else {
    sm90::fwd::run_fwd_phase1_kernel<512, false>(params);
  }
  return static_cast<int>(cudaGetLastError());
}

#else  // !ATTENTION_RS_USE_FLASHMLA

extern "C" int flashmla_dsv4_supported(int) { return 0; }
extern "C" int flashmla_dsv4_bytes_per_token() { return 584; }
extern "C" int flashmla_dsv4_decode_workspace_bytes(int, int, int, int*, size_t*,
                                                    size_t*, size_t*, size_t*) {
  return static_cast<int>(cudaErrorNotSupported);
}
extern "C" int flashmla_dsv4_sparse_decode(const void*, const void*, const int*,
                                           const int*, const float*, void*, float*,
                                           const void*, const int*, const int*, int*,
                                           int*, float*, float*, int, int, int, int, int,
                                           int, int, int, int, int, float,
                                           cudaStream_t) {
  return static_cast<int>(cudaErrorNotSupported);
}
extern "C" int flashmla_dsv4_sparse_prefill(const void*, const void*, const int*,
                                            const float*, const int*, void*, float*,
                                            float*, int, int, int, int, float,
                                            cudaStream_t) {
  return static_cast<int>(cudaErrorNotSupported);
}

#endif
