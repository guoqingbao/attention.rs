/**
 * @brief CUDA kernel for chunked prefill attention with paged KV-cache.
 * Copyright (c) 2025, Guoqing Bao.  All rights reserved.
 * This kernel computes attention during the prefill stage (processing prompt tokens)
 * using a **paged key-value cache**. Each thread computes the output for a token/head pair.
 *
 * This CUDA kernel is part of the vllm.rs project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/prefill_paged_attn.cu
 * Features:
 *  - Support Chunked Prefill (prefilled attention with kvcache)
 *  - Supports **paged KV-cache** (blocks of tokens stored in memory).
 *  - Handles **sliding window attention**.
 *  - Optionally applies **ALiBi positional bias**.
 *  - Uses **online softmax** for numerical stability.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif

#include "attention/attention_dtypes.h"
#include "attention/attention_utils.cuh"
#include <stdexcept>
#include <algorithm>

#ifndef USE_ROCM
#define WARP_SIZE 32
#else
#define WARP_SIZE warpSize
#endif
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define TOKEN_CHUNK_SIZE 128
using namespace vllm;

namespace vllm_rs {

inline __device__ float fast_tanh(float x) {
  #if defined(__CUDA_ARCH__)
    #if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
      float y;
      asm volatile ( "tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
      return y;
    #else
      return ::tanhf(x);
    #endif
  #else
  return std::tanh(x);
  #endif
}

template <typename T>
__device__ inline T make_zero() {
    T x;
    memset(&x, 0, sizeof(T));
    return x;
}

/**
 * @brief CUDA kernel for chunked prefill attention with paged KV-cache.
 *
 * This kernel computes attention during the prefill stage (processing prompt tokens)
 * using a **paged key-value cache**. Each thread computes the output for a token/head pair.
 *
 * @author Guoqing Bao
 * This CUDA kernel is part of the vllm.rs project:
 * Original implmentation: https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/prefill_paged_attn.cu
 * @tparam scalar_t   Data type (e.g., half, float, bfloat16).
 * @tparam HEAD_SIZE  Dimension of each attention head.
 * @tparam BLOCK_SIZE Number of tokens per KV cache block.
 *
 * @param[out] out             [num_all_tokens_new, num_query_heads, HEAD] 
 *                             Output tensor: attention result for each token/head.
 * @param[in]  q               [num_all_tokens_new, num_query_heads, HEAD] 
 *                             Query tensor for all new tokens.
 * @param[in]  k_cache         [num_k_blocks, num_kv_heads, HEAD/x, BLOCK_SIZE, x] 
 *                             Paged cache for keys.
 * @param[in]  v_cache         [num_k_blocks, num_kv_heads, HEAD, BLOCK_SIZE] 
 *                             Paged cache for values.
 * @param[in]  num_kv_heads    Number of KV heads (can be smaller than query heads if sharing).
 * @param[in]  sm_scale        Scaling factor for q·k (usually 1/sqrt(HEAD_SIZE)).
 * @param[in]  block_tables    [num_seqs, block_table_stride] 
 *                             Maps logical block indices → physical block indices in cache.
 * @param[in]  seq_lens        [num_seqs] 
 *                             Full sequence lengths (used to find valid context length).
 * @param[in]  block_table_stride Stride for indexing block_tables per sequence.
 * @param[in]  num_seqs        Number of sequences in this batch.
 * @param[in]  num_query_heads Total number of query heads.
 * @param[in]  num_query_tokens Number of query tokens across all sequences.
 * @param[in]  softscapping    Factor for softcapping logits (applies tanh rescaling).
 * @param[in]  o_stride_tokens Stride for writing to `out` across tokens.
 * @param[in]  query_start_len [num_seqs+1] 
 *                             Prefix sum array giving token index ranges for each sequence.
 * @param[in]  alibi_slopes    [num_query_heads] 
 *                             ALiBi slopes per head (nullptr if not used).
 * @param[in]  k_scale         Optional scaling factor for keys (not always used).
 * @param[in]  v_scale         Optional scaling factor for values (not always used).
 * @param[in]  sinks           [num_query_heads] 
 *                             Sink logits for stabilizing softmax initialization (nullptr if not used).
 * @param[in]  sliding_window  If > 0, restricts attention to a recent window of tokens.
 * @param[in]  total_num_blocks Total number of KV blocks allocated in cache.
 * @param[in]  kv_block_stride Stride between consecutive KV blocks in memory.
 * @param[in]  kv_head_stride  Stride between consecutive KV heads inside a block.
 */
template<typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE>
__global__ void chunked_prefill_paged_attention_kernel(
    scalar_t* __restrict__ out,              // [num_all_tokens_new, num_query_heads, HEAD]
    const scalar_t* __restrict__ q,          // [num_all_tokens_new, num_query_heads, HEAD]
    const cache_t* __restrict__ k_cache,    // [num_k_blocks, num_kv_heads, HEAD/x, BLOCK_SIZE, x]
    const cache_t* __restrict__ v_cache,    // [num_k_blocks, num_kv_heads, HEAD, BLOCK_SIZE]
    const float* __restrict__ k_scales, const float* v_scales,
    int32_t num_kv_heads,
    float sm_scale,
    const uint32_t* __restrict__ block_tables,
    const uint32_t* __restrict__ seq_lens,
    int32_t block_table_stride,
    int32_t num_seqs,
    int32_t num_query_heads,
    int32_t num_query_tokens,
    float softscapping,
    int32_t o_stride_tokens,
    const uint32_t* __restrict__ query_start_len,
    const float* __restrict__ alibi_slopes,
    const float* __restrict__ sinks,//not used
    int32_t sliding_window,
    int32_t total_num_blocks,
    int32_t kv_block_stride,
    int32_t kv_head_stride
) {
    const bool is_quantized = !std::is_same<scalar_t, cache_t>::value;
    constexpr int THREAD_GROUP_SIZE = 1;
    constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
    constexpr int NUM_VECS  = HEAD_SIZE / VEC_SIZE;

    const int tid     = threadIdx.x;
    const int lane    = tid % TOKEN_CHUNK_SIZE; // each lane processes one token

    const int NUM_BLOCK_VECS = BLOCK_SIZE / VEC_SIZE;
    // Grid setup: (query_head_per_kv, kv_heads, tokens / chunk_size)
    const int qh_base_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int token_start = blockIdx.z * TOKEN_CHUNK_SIZE + lane;

     // Total number of elements in K and V cache
    // int64_t k_cache_num_elems = total_num_blocks * kv_block_stride;
    // int64_t v_cache_num_elems = total_num_blocks * kv_block_stride;

    const int num_queries_per_kv = num_query_heads / num_kv_heads;
    const int X = 16 / sizeof(cache_t); // sub-vector size
    const bool use_alibi = (alibi_slopes != nullptr);
    const bool use_sinks  = (sinks != nullptr);

    // Strides for indexing q/o tensors
    const int64_t q_stride_tokens = (int64_t)num_query_heads * (int64_t)HEAD_SIZE;
    const int64_t q_stride_heads  = (int64_t)HEAD_SIZE;
    const int64_t o_stride_heads  = (int64_t)HEAD_SIZE;

    // --- Find which sequence this token belongs to ---
    int seq_idx = 0;
    for (int i = 0; i < num_seqs; i++) {
        int s = query_start_len[i];
        int e = query_start_len[i + 1];
        if (token_start >= s && token_start < e) {
          seq_idx = i;
          int seq_len = (e - s);
          if (seq_len <= 0) return;
          break;
        }
    }

    const uint32_t seq_len_full = seq_lens[seq_idx];
    const int num_blocks = (int)((seq_len_full + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const uint32_t* block_table_for_seq = block_tables + (int64_t)seq_idx * (int64_t)block_table_stride;
    // const int context_len = (int)seq_len_full - 1;

    // Vectorized types for Q and K
    using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
    using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
    using Float_vec = typename Vec<float, VEC_SIZE>::Type;
    using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

    // Buffers for q, k, and temporary storage
    Q_vec q_vec[NUM_VECS];
    K_vec k_vec[NUM_VECS];
    float qk_block[BLOCK_SIZE];

    // Determine active head and token
    const int query_head_idx = kv_head_idx * num_queries_per_kv + qh_base_idx;
    const bool head_active = (qh_base_idx < num_queries_per_kv) && (query_head_idx < num_query_heads);
    const bool lane_active = token_start < num_query_tokens;

    // // Compute offsets in q and output tensors
    const int64_t q_off = (int64_t)token_start * q_stride_tokens + (int64_t)query_head_idx * q_stride_heads;
    const int64_t o_off = (int64_t)token_start * (int64_t)o_stride_tokens + (int64_t)query_head_idx * o_stride_heads;
    
    // --- Load Q vector for this token and head ---
    if (head_active && lane_active) {
      #pragma unroll
      for (int k = 0; k < NUM_VECS; k++) {
        int d_base = k * VEC_SIZE;
        q_vec[k] = *reinterpret_cast<const Q_vec*>(&q[q_off + d_base]);
      }
    }

    // Accumulators for output
    float acc_vec[HEAD_SIZE] = { 0.f };
    float M = use_sinks && head_active && lane_active ? sinks[query_head_idx] : -INFINITY;
    float alibi = (use_alibi && head_active && lane_active) ? alibi_slopes[query_head_idx] : 0.f;
    float L = 1.f;

    // Sliding Window Calculation
    // We only attend to tokens in range [start_token_idx, seq_len_full)
    int start_token_idx = 0;
    int start_block_idx = 0;
    if (sliding_window > 0 && sliding_window < (int)seq_len_full) {
        start_token_idx = (int)seq_len_full - sliding_window;
        start_block_idx = start_token_idx / BLOCK_SIZE;
    }

    // --- Iterate over all KV blocks ---
    // Start from start_block_idx instead of 0
    for (int blk = start_block_idx; blk < num_blocks; ++blk) {
        const uint32_t physical_block = block_table_for_seq[blk];
        const bool valid_block =
            (physical_block != UINT32_MAX) &&
            ((uint64_t)physical_block < (uint64_t)total_num_blocks);

        const int block_in_full = blk * BLOCK_SIZE;

        // --- Compute q·k for each token in this block ---
        bool in_contexts[BLOCK_SIZE] = { false };
        for (int b = 0; b < BLOCK_SIZE; ++b) {
            const int token_idx_in_full = block_in_full + b;
            
            // Masking logic: Must be in valid context AND within sliding window
            bool in_context = (token_idx_in_full < (int)seq_len_full);
            bool in_window = (token_idx_in_full >= start_token_idx);

            // Store status for P·V later
            in_contexts[b] = in_context && in_window;

            // If invalid, mask out QK
            if (!in_context || !in_window || !valid_block || !lane_active) {
              qk_block[b] = -INFINITY;
              continue;
            }
             // Load K vector from kcache
            // #pragma unroll
            const int offset = physical_block * kv_block_stride + kv_head_idx * kv_head_stride;
            for (int k = 0; k < NUM_VECS; k++) {
              int d = k * VEC_SIZE;
              int gy = d / X;
              int gx = d % X;
              int64_t k_idx = offset + b * X + gy * (BLOCK_SIZE * X) + gx;
              if constexpr (!is_quantized) {
                k_vec[k] = *reinterpret_cast<const K_vec*>(&k_cache[k_idx]);
              } else {
                Quant_vec fp8_k_vec = *reinterpret_cast<const Quant_vec*>(
                    &k_cache[k_idx]);
                k_vec[k] = vllm::fp8::scaled_convert<K_vec, Quant_vec>(fp8_k_vec, *k_scales);
              }
            }

            // Compute dot product q·k
            float qk = Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vec, k_vec) * sm_scale;
            if (softscapping != 1.0) {
              qk = fast_tanh(qk / softscapping) * softscapping;
            }
            qk_block[b] = qk;

            // Add ALiBi positional bias if enabled
            if (use_alibi) {
                const int context_len = (int)seq_len_full - 1;
                qk_block[b] += alibi * float(token_idx_in_full - context_len);
            }
        } // blk

        if (!head_active || !lane_active || !valid_block) continue;

        // --- Softmax computation (online normalization) ---
        float Smax = -INFINITY;
        #pragma unroll
        for (int b = 0; b < BLOCK_SIZE; ++b) Smax = fmaxf(Smax, qk_block[b]);

        // Update running maximum and normalization factors
        const float m_j = fmaxf(M, Smax);
        const float alpha = __expf(M - m_j);
        M = m_j;
        L = L * alpha;
        #pragma unroll
        for (int i = 0; i < HEAD_SIZE; ++i) acc_vec[i] *= alpha;

        // --- Compute softmax weights and accumulate P·V ---
        float acc_lane = 0.f;
        Float_vec p_vec[NUM_BLOCK_VECS];

        #pragma unroll
        for (int b = 0; b < BLOCK_SIZE; ++b) {
          if (in_contexts[b]) { // Uses the updated context check (window + valid)
              const float P = __expf(qk_block[b] - M);
              reinterpret_cast<float*>(&p_vec[b/VEC_SIZE])[b % VEC_SIZE] = P;
              acc_lane += P;
          } else {
            reinterpret_cast<float*>(&p_vec[b/VEC_SIZE])[b % VEC_SIZE] = 0.f;
          }
        }

        // Load V block and compute weighted sum
        const int64_t v_base_block = (int64_t)physical_block * kv_block_stride + (int64_t)kv_head_idx * kv_head_stride;
        Float_vec v_vec[NUM_BLOCK_VECS];

        // Float_vec* v_vec_ptr = reinterpret_cast<Float_vec*>(&v_vec);
        for (int k = 0; k < HEAD_SIZE; ++k) {
          const cache_t* v_row_ptr = &v_cache[v_base_block + (int64_t)k * BLOCK_SIZE];
          for (int b = 0; b < NUM_BLOCK_VECS; b++) {
            const cache_t* src = v_row_ptr + b * VEC_SIZE;
            Float_vec v;
            if constexpr (!is_quantized) {
              v = to_float(*reinterpret_cast<const K_vec*>(src));
            } else {
              Quant_vec fp8_v_vec = *reinterpret_cast<const Quant_vec *>(src);
              v = vllm::fp8::scaled_convert<Float_vec, Quant_vec>(fp8_v_vec, *v_scales);
            }

            acc_vec[k] += dot(p_vec[b], v);
          }
        }

        L += acc_lane; // update softmax normalization
    } // blk loop

    using O_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

    // Write output for the current token
    if (head_active && lane_active) {
      O_vec o_vec[NUM_VECS];
      #pragma unroll
      for (int k = 0; k < HEAD_SIZE; k++) {
        float outv = acc_vec[k] / (L + 1e-6f);
        from_float(reinterpret_cast<scalar_t*>(&o_vec[k / VEC_SIZE])[k % VEC_SIZE], outv);
      }
      #pragma unroll
      for (int k = 0; k < NUM_VECS; k++) {
        *reinterpret_cast<O_vec*>(out + o_off + k * VEC_SIZE) = o_vec[k];
      }
    }

}

}

#define LAUNCH_PAGED_ATTENTION_PREFILL(HEAD_SIZE)   \
  vllm_rs::chunked_prefill_paged_attention_kernel<T, cache_T, HEAD_SIZE, BLOCK_SIZE>  \
  <<<grid, block, 0, stream>>>(                                                 \
    reinterpret_cast<T*>(out),                                                                \
    reinterpret_cast<T*>(query),                                                              \
    reinterpret_cast<cache_T*>(key_cache),                                                          \
    reinterpret_cast<cache_T*>(value_cache),                                                        \
    k_scales, v_scales,                                                                       \
    num_kv_heads,                                                                             \
    scale,                                                                                    \
    block_tables,                                                                             \
    context_lens,                                                                             \
    max_num_blocks_per_seq,                                                                   \
    num_seqs,\
    num_query_heads,\
    num_query_tokens,\
    softscapping,\
    o_stride_tokens,\
    query_start_len,\
    alibi_slopes_ptr,                                                                         \
    sinks,\
    sliding_window,\
    num_blocks, \
    kv_block_stride,\
    kv_head_stride);


template<
  typename T,
  typename cache_T,
  int BLOCK_SIZE
  >
void paged_attention_prefill_launcher(
  void *out,
  void *query,
  void *key_cache,
  void *value_cache,
  float* k_scales,
  float* v_scales,
  int32_t num_kv_heads,
  float scale,
  uint32_t *block_tables,
  uint32_t *context_lens,
  int32_t max_num_blocks_per_seq,
  int32_t num_seqs,
  int32_t num_query_heads,
  int32_t num_query_tokens,
  int32_t head_size,
  float softscapping,
  int32_t o_stride_tokens,      // out.stride(0)
  uint32_t* __restrict__ query_start_len, // [num_seqs+1] or nullptr
  float*    __restrict__ sinks,  // [num_query_heads] or nullptr
  int32_t sliding_window,
  int32_t num_blocks,
  int32_t kv_block_stride,   // stride between consecutive physical blocks for k_cache (elements)
  int32_t kv_head_stride,    // stride between consecutive kv heads for k_cache (elements)
  int64_t stream_) {

  const float* alibi_slopes_ptr = nullptr;
  const int num_queries_per_kv = num_query_heads / num_kv_heads;
  int VEC_SIZE = 16 / sizeof(T);
  // int NUM_VECS  = head_size / VEC_SIZE;

  int num_token_chunks = (num_query_tokens + TOKEN_CHUNK_SIZE - 1) / TOKEN_CHUNK_SIZE;
  dim3 grid(num_queries_per_kv, num_kv_heads, num_token_chunks);
  dim3 block(TOKEN_CHUNK_SIZE);
  const cudaStream_t stream = (cudaStream_t)stream_;
  switch (head_size) {
    case 64:
      LAUNCH_PAGED_ATTENTION_PREFILL(64);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_PREFILL(96);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_PREFILL(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_PREFILL(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_PREFILL(256);
      break;
    default:
      break;
  }
}

#define CALL_PREFILL_LAUNCHER(T, cache_T, BLOCK_SIZE)                             \
  paged_attention_prefill_launcher<T, cache_T, BLOCK_SIZE>(                       \
    out,                                                            \
    query,                                                          \
    key_cache,                                                      \
    value_cache,                                                    \
    reinterpret_cast<float*>(k_scales),                                                      \
    reinterpret_cast<float*>(v_scales),                                                    \
    num_kv_heads,                                                   \
    scale,                                                          \
    block_tables,                                                   \
    context_lens,                                                   \
    max_num_blocks_per_seq,                                                \
    num_seqs,\
    num_query_heads,                                                       \
    num_query_tokens,\
    head_size, \
    softscapping,                                                      \
    o_stride_tokens,                                                      \
    query_start_len,                                         \
    sinks,                                                       \
    sliding_window,                                                \
    num_blocks,\
    kv_block_stride,\
    kv_head_stride,\
    stream);

#define CALL_PREFILL_LAUNCHER_BLOCK_SIZE(T, cache_T)                              \
  switch (block_size) {                                             \
    case 32:                                                        \
      CALL_PREFILL_LAUNCHER(T, cache_T, 32);                                      \
      break;                                                        \
    case 64:                                                        \
      CALL_PREFILL_LAUNCHER(T, cache_T, 64);                                      \
      break;                                                        \
    default:                                                        \
      break;                                                        \
  }

extern "C" void paged_attention_prefill(
  void *out,             // [num_seqs, num_heads, head_size]
  void *query,           // [num_seqs, num_heads, head_size]
  void *key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  void *value_cache,     // [num_blocks, num_heads, head_size, block_size]
  void * k_scales,
  void * v_scales,
  int32_t num_kv_heads,               // [num_heads]
  float scale,
  uint32_t *block_tables,    // [num_seqs, max_num_blocks_per_seq]
  uint32_t *context_lens,    // [num_seqs]
  int32_t block_size,
  int32_t max_context_len,

  int32_t num_seqs,
  int32_t num_query_heads,
  int32_t num_query_tokens,
  int32_t head_size,
  int32_t max_num_blocks_per_seq,
  int32_t q_stride,
  int32_t num_blocks,
  int32_t kv_block_stride,
  int32_t kv_head_stride,

  uint32_t dtype,      // 0 => f16; 1 => bf16; 2 => f32
  float softscapping,

  int32_t o_stride_tokens,      // out.stride(0)
  uint32_t* query_start_len, // [num_seqs+1] or nullptr
  float* sinks,  // [num_query_heads] or nullptr
  int32_t sliding_window,
  int64_t stream
  ) {

  if (k_scales != nullptr || v_scales != nullptr) {
#ifndef NO_FP8_KVCACHE
    if (dtype == 2) {
      // CALL_PREFILL_LAUNCHER_BLOCK_SIZE(float);
    } else if (dtype == 0) {
      CALL_PREFILL_LAUNCHER_BLOCK_SIZE(uint16_t, uint8_t);
    } else if (dtype == 1) {
      #ifndef NO_BF16_KERNEL //cuda_arc < 800 (no bf16 support)
      CALL_PREFILL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16, uint8_t);
      #endif
    }
#else
    throw std::runtime_error("Error: FP8 KV-cache is disabled (possiblly because flash-attn or context-cache enabled).");
#endif
  } else {
    if (dtype == 2) {
      // CALL_PREFILL_LAUNCHER_BLOCK_SIZE(float);
    } else if (dtype == 0) {
      CALL_PREFILL_LAUNCHER_BLOCK_SIZE(uint16_t, uint16_t);
    } else if (dtype == 1) {
      #ifndef NO_BF16_KERNEL //cuda_arc < 800 (no bf16 support)
      CALL_PREFILL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16, __nv_bfloat16);
      #endif
    }
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
