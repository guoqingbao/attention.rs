/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "pagedattention.cuh"
#include <stdexcept>
#include <algorithm>

#ifndef USE_ROCM
#define WARP_SIZE 32
#else
#define WARP_SIZE warpSize
#endif
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                                  \
  vllm::paged_attention_v2_kernel<T, cache_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>      \
  <<<grid, block, shared_mem_size, stream>>>(                                                 \
    exp_sums,                                                                                 \
    max_logits,                                                                               \
    tmp_out_ptr,                                                                              \
    reinterpret_cast<T*>(query),                                                              \
    reinterpret_cast<cache_T*>(key_cache),                                                          \
    reinterpret_cast<cache_T*>(value_cache),                                                        \
    reinterpret_cast<float*>(k_scales),                                                          \
    reinterpret_cast<float*>(v_scales),                                                        \
    num_kv_heads,                                                                             \
    scale,                                                                                    \
    block_tables,                                                                             \
    context_lens,                                                                             \
    max_num_blocks_per_seq,                                                                   \
    alibi_slopes,                                                                             \
    q_stride,                                                                                 \
    kv_block_stride,                                                                          \
    kv_head_stride,\
    softscapping,\
    sliding_window);                                                                          \
  vllm::paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>           \
  <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                                   \
    reinterpret_cast<T*>(out),                                                                \
    exp_sums,                                                                                 \
    max_logits,                                                                               \
    tmp_out_ptr,                                                                              \
    context_lens,                                                                             \
    max_num_partitions);

template<
  typename T,
  typename cache_T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128,
  int PARTITION_SIZE = 512>
void paged_attention_v2_launcher(
  void *out,
  float *exp_sums,
  float *max_logits,
  void *tmp_out,
  void *query,
  void *key_cache,
  void *value_cache,
  void* k_scales,
  void* v_scales,
  int num_kv_heads,
  float scale,
  int32_t *block_tables,
  int32_t *context_lens,
  int max_context_len,

  int num_seqs,
  int num_heads,
  int head_size,
  int max_num_blocks_per_seq,
  int q_stride,
  int kv_block_stride,
  int kv_head_stride,
  float softscapping,
  int sliding_window,
  int64_t stream_
  ) {

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes = nullptr;

  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out);

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

  // For paged attention v2 kernel.
  dim3 grid(num_heads, num_seqs, max_num_partitions);
  int shared_mem_size = std::max(logits_size, outputs_size);
  // For paged attention v2 reduce kernel.
  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  dim3 block(NUM_THREADS);
  const cudaStream_t stream = (cudaStream_t)stream_;
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V2(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V2(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V2(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V2(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V2(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V2(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V2(256);
      break;
    default:
      break;
  }
}

#define CALL_V2_LAUNCHER(T, cache_T, BLOCK_SIZE)                             \
  paged_attention_v2_launcher<T, cache_T, BLOCK_SIZE>(                       \
    out,                                                            \
    exp_sums,                                                       \
    max_logits,                                                     \
    tmp_out,                                                        \
    query,                                                          \
    key_cache,                                                      \
    value_cache,                                                    \
    k_scales,                                                      \
    v_scales,                                                      \
    num_kv_heads,                                                   \
    scale,                                                          \
    block_tables,                                                   \
    context_lens,                                                   \
    max_context_len,                                                \
    num_seqs,                                                       \
    num_heads,                                                      \
    head_size,                                                      \
    max_num_blocks_per_seq,                                         \
    q_stride,                                                       \
    kv_block_stride,                                                \
    kv_head_stride,\
    softscapping, \
    sliding_window,\
    stream);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V2_LAUNCHER_BLOCK_SIZE(T, cache_T)                              \
  switch (block_size) {                                             \
    case 32:                                                        \
      CALL_V2_LAUNCHER(T, cache_T, 32);                                      \
      break;                                                        \
    case 64:                                                        \
      CALL_V2_LAUNCHER(T, cache_T, 64);                                      \
      break;                                                        \
    default:                                                        \
      break;                                                        \
  }

extern "C" void paged_attention_v2(
  void *out,             // [num_seqs, num_heads, head_size]
  float *exp_sums,        // [num_seqs, num_heads, max_num_partitions]
  float *max_logits,      // [num_seqs, num_heads, max_num_partitions]
  void *tmp_out,         // [num_seqs, num_heads, max_num_partitions, head_size]
  void *query,           // [num_seqs, num_heads, head_size]
  void *key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  void *value_cache,     // [num_blocks, num_heads, head_size, block_size]
  void* k_scales,
  void* v_scales,
  int32_t num_kv_heads,
  float scale,
  int32_t *block_tables,    // [num_seqs, max_num_blocks_per_seq]
  int32_t *context_lens,    // [num_seqs]
  int32_t block_size,
  int32_t max_context_len,

  int32_t num_seqs,
  int32_t num_heads,
  int32_t head_size,
  int32_t max_num_blocks_per_seq,
  int32_t q_stride,
  int32_t kv_block_stride,
  int32_t kv_head_stride,

  uint32_t dtype,      // 0 => f16; 1 => bf16; 2 => f32
  float softscapping,
  int sliding_window,
  int64_t stream
  ) {
  bool is_quantized = (k_scales != nullptr) && (v_scales != nullptr);
  if (!is_quantized) {
    if (dtype == 2) {
      CALL_V2_LAUNCHER_BLOCK_SIZE(float, float);
    } else if (dtype == 0) {
      CALL_V2_LAUNCHER_BLOCK_SIZE(uint16_t, uint16_t);
    } else if (dtype == 1) {
      #ifndef NO_BF16_KERNEL //cuda_arc < 800 (no bf16 support)
      CALL_V2_LAUNCHER_BLOCK_SIZE(__nv_bfloat16, __nv_bfloat16);
      #endif
    }
  } else {
#ifndef NO_FP8_KVCACHE
    if (dtype == 2) {
      CALL_V2_LAUNCHER_BLOCK_SIZE(float, uint8_t);
    } else if (dtype == 0) {
      CALL_V2_LAUNCHER_BLOCK_SIZE(uint16_t, uint8_t);
    } else if (dtype == 1) {
      #ifndef NO_BF16_KERNEL //cuda_arc < 800 (no bf16 support)
      CALL_V2_LAUNCHER_BLOCK_SIZE(__nv_bfloat16, uint8_t);
      #endif
    }
#else
    throw std::runtime_error("Error: FP8 KV-cache is disabled (possiblly because flashattn or context-cache enabled).");
#endif
  }
}

#undef WARP_SIZE
#undef DIVIDE_ROUND_UP
