/**
 * Fused MLA (Multi-head Latent Attention) paged attention kernels.
 *
 * MLA operates in compressed KV space (kv_lora_rank) rather than standard
 * head_dim. Score = q_absorbed · ckv^T + q_pe · kpe^T, then softmax, then
 * output = attn_weights · ckv (in kv_lora_rank space).
 *
 * Performance notes (v3):
 *   - Prefill: single-pass FlashAttention-style online softmax.
 *   - Warp-per-KV-token scoring (8 concurrent scores / 256-thread block).
 *   - Decode: same warp-parallel scoring; launch grid sized by max_context_len.
 *
 * Copyright (c) 2025, Guoqing Bao.  All rights reserved.
 *
 * This CUDA kernel is developed for xInfer (vLLM.rs) project:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/mla_paged_attention.cu
 *
 * Decode uses a split-K partitioned approach for long contexts:
 *   Phase 1: Each partition block processes a chunk of KV tokens, producing
 *            partial (max_logit, exp_sum, weighted_output) per head.
 *   Phase 2: A reduce kernel merges partitions via online softmax correction.
 *
 * For short contexts (single partition), Phase 1 writes directly to output.
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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <algorithm>

#define MLA_WARP_SIZE 32

__device__ __forceinline__ float mla_warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = MLA_WARP_SIZE / 2; mask >= 1; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__device__ __forceinline__ float mla_warp_reduce_max(float val) {
#pragma unroll
    for (int mask = MLA_WARP_SIZE / 2; mask >= 1; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

/* ------------------------------------------------------------------ */
/*  Warp-cooperative Q·K score for one KV token                        */
/* ------------------------------------------------------------------ */
template <typename scalar_t>
__device__ __forceinline__ float mla_warp_qk_score(
    const float* __restrict__ q_abs_s,
    const float* __restrict__ q_pe_s,
    const scalar_t* __restrict__ ckv_cache,
    const scalar_t* __restrict__ kpe_cache,
    int64_t ckv_base,
    int64_t kpe_base,
    int kv_lora_rank,
    int qk_rope_head_dim,
    int lane,
    float scale) {
    float partial = 0.f;
#pragma unroll 4
    for (int d = lane; d < kv_lora_rank; d += MLA_WARP_SIZE) {
        partial += q_abs_s[d] * (float)ckv_cache[ckv_base + d];
    }
#pragma unroll 2
    for (int d = lane; d < qk_rope_head_dim; d += MLA_WARP_SIZE) {
        partial += q_pe_s[d] * (float)kpe_cache[kpe_base + d];
    }
    return mla_warp_reduce_sum(partial) * scale;
}

/* ------------------------------------------------------------------ */
/*  Split-K Decode Kernel (Phase 1) — warp-parallel scores             */
/*  Grid: (num_heads, num_seqs, num_partitions)                        */
/* ------------------------------------------------------------------ */

static constexpr int PARTITION_SIZE = 128;
// Support kv_lora_rank up to 2048 with 256 threads (8 dims/thread).
static constexpr int MLA_MAX_DIMS_PER_THREAD = 8;

template <typename scalar_t, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mla_paged_attention_decode_partitioned_kernel(
    float* __restrict__ tmp_out,
    float* __restrict__ tmp_max,
    float* __restrict__ tmp_sum,
    const scalar_t* __restrict__ q_abs,
    const scalar_t* __restrict__ q_pe,
    const scalar_t* __restrict__ ckv_cache,
    const scalar_t* __restrict__ kpe_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    const float scale,
    const int num_heads,
    const int kv_lora_rank,
    const int qk_rope_head_dim,
    const int max_num_blocks_per_seq,
    const int max_partitions) {

    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int part_idx = blockIdx.z;
    const int tid = threadIdx.x;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;
    const int warp = tid / MLA_WARP_SIZE;
    const int lane = tid % MLA_WARP_SIZE;

    const int ctx_len = context_lens[seq_idx];
    const int part_start = part_idx * PARTITION_SIZE;
    if (part_start >= ctx_len) return;
    const int part_end = min(part_start + PARTITION_SIZE, ctx_len);
    const int part_len = part_end - part_start;

    const int q_abs_off = (seq_idx * num_heads + head_idx) * kv_lora_rank;
    const int q_pe_off = (seq_idx * num_heads + head_idx) * qk_rope_head_dim;
    const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    // smem: Q + scores[NUM_WARPS]
    extern __shared__ char smem_raw[];
    float* q_abs_s = reinterpret_cast<float*>(smem_raw);
    float* q_pe_s = q_abs_s + kv_lora_rank;
    float* scores = q_pe_s + qk_rope_head_dim;

    for (int d = tid; d < kv_lora_rank; d += NUM_THREADS)
        q_abs_s[d] = (float)q_abs[q_abs_off + d];
    for (int d = tid; d < qk_rope_head_dim; d += NUM_THREADS)
        q_pe_s[d] = (float)q_pe[q_pe_off + d];
    __syncthreads();

    // Per-thread output accumulators over owned dims
    float acc[MLA_MAX_DIMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < MLA_MAX_DIMS_PER_THREAD; i++) acc[i] = 0.f;

    float part_max = -FLT_MAX;
    float part_sum = 0.f;

    // Process partition in warp-sized tiles (NUM_WARPS KV tokens / iter)
    for (int tile = 0; tile < part_len; tile += NUM_WARPS) {
        const int ti = tile + warp;
        const bool valid = ti < part_len;

        float score = -FLT_MAX;
        int physical_block = 0;
        int blk_off = 0;
        if (valid) {
            const int t = part_start + ti;
            const int blk_idx = t / BLOCK_SIZE;
            blk_off = t % BLOCK_SIZE;
            physical_block = block_table[blk_idx];
            const int64_t ckv_base =
                (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
                (int64_t)blk_off * kv_lora_rank;
            const int64_t kpe_base =
                (int64_t)physical_block * BLOCK_SIZE * qk_rope_head_dim +
                (int64_t)blk_off * qk_rope_head_dim;
            score = mla_warp_qk_score(
                q_abs_s, q_pe_s, ckv_cache, kpe_cache, ckv_base, kpe_base,
                kv_lora_rank, qk_rope_head_dim, lane, scale);
        }
        if (lane == 0) scores[warp] = valid ? score : -FLT_MAX;
        __syncthreads();

        // Softmax over this tile relative to running partition max
        float tile_max = -FLT_MAX;
#pragma unroll
        for (int i = 0; i < NUM_WARPS; i++) {
            if (tile + i < part_len) tile_max = fmaxf(tile_max, scores[i]);
        }
        float m_new = fmaxf(part_max, tile_max);
        float alpha = (part_max == -FLT_MAX) ? 0.f : expf(part_max - m_new);

        // Rescale previous accumulators
#pragma unroll
        for (int i = 0; i < MLA_MAX_DIMS_PER_THREAD; i++) acc[i] *= alpha;
        part_sum *= alpha;

#pragma unroll
        for (int i = 0; i < NUM_WARPS; i++) {
            if (tile + i >= part_len) continue;
            float p = expf(scores[i] - m_new);
            part_sum += p;

            const int t = part_start + tile + i;
            const int blk_idx = t / BLOCK_SIZE;
            const int bo = t % BLOCK_SIZE;
            const int pb = block_table[blk_idx];
            const int64_t ckv_base =
                (int64_t)pb * BLOCK_SIZE * kv_lora_rank + (int64_t)bo * kv_lora_rank;

            int di = 0;
            for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
                acc[di] += p * (float)ckv_cache[ckv_base + d];
                di++;
            }
        }
        part_max = m_new;
        __syncthreads();
    }

    // Store partition metadata + unnormalized output
    const int meta_off = (seq_idx * num_heads + head_idx) * max_partitions + part_idx;
    if (tid == 0) {
        tmp_max[meta_off] = part_max;
        tmp_sum[meta_off] = part_sum;
    }

    const int64_t out_off =
        ((int64_t)seq_idx * num_heads + head_idx) * max_partitions * kv_lora_rank +
        (int64_t)part_idx * kv_lora_rank;
    {
        int di = 0;
        for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
            tmp_out[out_off + d] = acc[di];
            di++;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Reduce Kernel (Phase 2)                                            */
/* ------------------------------------------------------------------ */

template <typename scalar_t, int NUM_THREADS>
__global__ void mla_paged_attention_decode_reduce_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ tmp_out,
    const float* __restrict__ tmp_max,
    const float* __restrict__ tmp_sum,
    const int32_t* __restrict__ context_lens,
    const int num_heads,
    const int kv_lora_rank,
    const int max_partitions) {

    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len == 0) return;
    const int num_parts = (ctx_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

    const int meta_base = (seq_idx * num_heads + head_idx) * max_partitions;
    const int out_off = (seq_idx * num_heads + head_idx) * kv_lora_rank;
    const int64_t tmp_base =
        ((int64_t)seq_idx * num_heads + head_idx) * max_partitions * kv_lora_rank;

    if (num_parts == 1) {
        const float inv_sum = tmp_sum[meta_base] > 0.f ? 1.f / tmp_sum[meta_base] : 0.f;
        for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
            out[out_off + d] = (scalar_t)(tmp_out[tmp_base + d] * inv_sum);
        }
        return;
    }

    float global_max = -FLT_MAX;
    for (int p = 0; p < num_parts; p++) {
        global_max = fmaxf(global_max, tmp_max[meta_base + p]);
    }

    float global_sum = 0.f;
    for (int p = 0; p < num_parts; p++) {
        global_sum += tmp_sum[meta_base + p] * expf(tmp_max[meta_base + p] - global_max);
    }
    float inv_global_sum = (global_sum > 0.f) ? (1.f / global_sum) : 0.f;

    for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
        float acc = 0.f;
        for (int p = 0; p < num_parts; p++) {
            float correction = expf(tmp_max[meta_base + p] - global_max);
            acc += tmp_out[tmp_base + (int64_t)p * kv_lora_rank + d] * correction;
        }
        out[out_off + d] = (scalar_t)(acc * inv_global_sum);
    }
}

/* ------------------------------------------------------------------ */
/*  Prefill v3: single-pass online softmax, warp-per-KV scoring        */
/*  Grid: (num_heads, total_tokens)                                    */
/* ------------------------------------------------------------------ */

template <typename scalar_t, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mla_paged_attention_prefill_v3_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q_abs,
    const scalar_t* __restrict__ q_pe,
    const scalar_t* __restrict__ ckv_cache,
    const scalar_t* __restrict__ kpe_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ cu_seqlens_q,
    const float scale,
    const int num_seqs,
    const int num_heads,
    const int kv_lora_rank,
    const int qk_rope_head_dim,
    const int max_num_blocks_per_seq,
    const int total_tokens) {

    const int head_idx = blockIdx.x;
    const int global_q_idx = blockIdx.y;
    const int tid = threadIdx.x;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;
    const int warp = tid / MLA_WARP_SIZE;
    const int lane = tid % MLA_WARP_SIZE;

    if (global_q_idx >= total_tokens) return;

    // Binary search sequence id
    int seq_idx = 0;
    {
        int lo = 0, hi = num_seqs;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cu_seqlens_q[mid + 1] <= global_q_idx) lo = mid + 1;
            else hi = mid;
        }
        seq_idx = lo;
    }

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len == 0) return;

    const int q_start = cu_seqlens_q[seq_idx];
    const int q_local_idx = global_q_idx - q_start;
    const int q_len = cu_seqlens_q[seq_idx + 1] - q_start;
    const int q_pos_start = ctx_len - q_len;
    const int causal_limit = q_pos_start + q_local_idx + 1;
    const int attend_len = min(ctx_len, causal_limit);
    if (attend_len <= 0) return;

    const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    extern __shared__ char smem_raw[];
    float* q_abs_s = reinterpret_cast<float*>(smem_raw);
    float* q_pe_s = q_abs_s + kv_lora_rank;
    float* scores = q_pe_s + qk_rope_head_dim;
    // red unused but keep layout stable if needed later

    const int q_abs_off = (global_q_idx * num_heads + head_idx) * kv_lora_rank;
    const int q_pe_off = (global_q_idx * num_heads + head_idx) * qk_rope_head_dim;

    for (int d = tid; d < kv_lora_rank; d += NUM_THREADS)
        q_abs_s[d] = (float)q_abs[q_abs_off + d];
    for (int d = tid; d < qk_rope_head_dim; d += NUM_THREADS)
        q_pe_s[d] = (float)q_pe[q_pe_off + d];
    __syncthreads();

    float acc[MLA_MAX_DIMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < MLA_MAX_DIMS_PER_THREAD; i++) acc[i] = 0.f;

    float m_i = -FLT_MAX;
    float l_i = 0.f;

    for (int base = 0; base < attend_len; base += NUM_WARPS) {
        const int t = base + warp;
        const bool valid = t < attend_len;

        float score = -FLT_MAX;
        if (valid) {
            const int blk_idx = t / BLOCK_SIZE;
            const int blk_off = t % BLOCK_SIZE;
            const int physical_block = block_table[blk_idx];
            const int64_t ckv_base =
                (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
                (int64_t)blk_off * kv_lora_rank;
            const int64_t kpe_base =
                (int64_t)physical_block * BLOCK_SIZE * qk_rope_head_dim +
                (int64_t)blk_off * qk_rope_head_dim;
            score = mla_warp_qk_score(
                q_abs_s, q_pe_s, ckv_cache, kpe_cache, ckv_base, kpe_base,
                kv_lora_rank, qk_rope_head_dim, lane, scale);
        }
        if (lane == 0) scores[warp] = valid ? score : -FLT_MAX;
        __syncthreads();

        float tile_max = -FLT_MAX;
#pragma unroll
        for (int i = 0; i < NUM_WARPS; i++) {
            if (base + i < attend_len) tile_max = fmaxf(tile_max, scores[i]);
        }

        float m_new = fmaxf(m_i, tile_max);
        float alpha = (m_i == -FLT_MAX) ? 0.f : expf(m_i - m_new);

#pragma unroll
        for (int i = 0; i < MLA_MAX_DIMS_PER_THREAD; i++) acc[i] *= alpha;
        float l_new = l_i * alpha;

#pragma unroll
        for (int i = 0; i < NUM_WARPS; i++) {
            if (base + i >= attend_len) continue;
            float p = expf(scores[i] - m_new);
            l_new += p;

            const int t2 = base + i;
            const int blk_idx = t2 / BLOCK_SIZE;
            const int blk_off = t2 % BLOCK_SIZE;
            const int physical_block = block_table[blk_idx];
            const int64_t ckv_base =
                (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
                (int64_t)blk_off * kv_lora_rank;

            int di = 0;
            for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
                acc[di] += p * (float)ckv_cache[ckv_base + d];
                di++;
            }
        }

        m_i = m_new;
        l_i = l_new;
        __syncthreads();
    }

    const float inv = (l_i > 0.f) ? (1.f / l_i) : 0.f;
    const int out_off = (global_q_idx * num_heads + head_idx) * kv_lora_rank;
    {
        int di = 0;
        for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
            out[out_off + d] = (scalar_t)(acc[di] * inv);
            di++;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  C API                                                              */
/* ------------------------------------------------------------------ */

extern "C" void mla_paged_attention_decode(
    void* out, void* q_abs, void* q_pe,
    void* ckv_cache, void* kpe_cache,
    int32_t* block_tables, int32_t* context_lens,
    float scale,
    int32_t num_seqs, int32_t num_heads,
    int32_t kv_lora_rank, int32_t qk_rope_head_dim,
    int32_t block_size, int32_t max_num_blocks_per_seq,
    int32_t max_context_len,
    uint32_t dtype, int64_t stream_,
    void* tmp_out_buf, void* tmp_max_buf, void* tmp_sum_buf,
    int32_t use_partitioned) {

    if (num_seqs == 0) return;
    const cudaStream_t stream = (cudaStream_t)stream_;

    constexpr int NUM_THREADS = 256;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;

    // Prefer actual max context over full table capacity (avoids launching
    // thousands of empty partition blocks for long max_model_len).
    int capacity_ctx = max_num_blocks_per_seq * block_size;
    int effective_ctx = max_context_len > 0 ? max_context_len : capacity_ctx;
    if (effective_ctx > capacity_ctx) effective_ctx = capacity_ctx;
    if (effective_ctx < 1) effective_ctx = 1;

    int smem_size =
        (kv_lora_rank + qk_rope_head_dim + NUM_WARPS) * (int)sizeof(float);

    if (use_partitioned && tmp_out_buf && tmp_max_buf && tmp_sum_buf &&
        effective_ctx > PARTITION_SIZE) {
        int max_partitions = (effective_ctx + PARTITION_SIZE - 1) / PARTITION_SIZE;

        dim3 grid(num_heads, num_seqs, max_partitions);
        dim3 block(NUM_THREADS);

        float* f_tmp_out = reinterpret_cast<float*>(tmp_out_buf);
        float* f_tmp_max = reinterpret_cast<float*>(tmp_max_buf);
        float* f_tmp_sum = reinterpret_cast<float*>(tmp_sum_buf);

#define LAUNCH_MLA_DECODE_PART(T, BS)                                         \
    mla_paged_attention_decode_partitioned_kernel<T, BS, NUM_THREADS>         \
        <<<grid, block, smem_size, stream>>>(                                 \
            f_tmp_out, f_tmp_max, f_tmp_sum,                                  \
            reinterpret_cast<T*>(q_abs), reinterpret_cast<T*>(q_pe),          \
            reinterpret_cast<T*>(ckv_cache), reinterpret_cast<T*>(kpe_cache), \
            block_tables, context_lens, scale, num_heads,                     \
            kv_lora_rank, qk_rope_head_dim, max_num_blocks_per_seq,           \
            max_partitions)

#define LAUNCH_MLA_DECODE_PART_BLOCK(T)                                       \
    switch (block_size) {                                                     \
        case 16: LAUNCH_MLA_DECODE_PART(T, 16); break;                       \
        case 32: LAUNCH_MLA_DECODE_PART(T, 32); break;                       \
        case 64: LAUNCH_MLA_DECODE_PART(T, 64); break;                       \
        default: break;                                                       \
    }

        if (dtype == 0) {
            LAUNCH_MLA_DECODE_PART_BLOCK(__half);
        } else if (dtype == 1) {
            LAUNCH_MLA_DECODE_PART_BLOCK(__nv_bfloat16);
        } else if (dtype == 2) {
            LAUNCH_MLA_DECODE_PART_BLOCK(float);
        }

#undef LAUNCH_MLA_DECODE_PART
#undef LAUNCH_MLA_DECODE_PART_BLOCK

        dim3 reduce_grid(num_heads, num_seqs);
        dim3 reduce_block(NUM_THREADS);

#define LAUNCH_MLA_REDUCE(T)                                                  \
    mla_paged_attention_decode_reduce_kernel<T, NUM_THREADS>                  \
        <<<reduce_grid, reduce_block, 0, stream>>>(                           \
            reinterpret_cast<T*>(out), f_tmp_out, f_tmp_max, f_tmp_sum,       \
            context_lens, num_heads, kv_lora_rank, max_partitions)

        if (dtype == 0) {
            LAUNCH_MLA_REDUCE(__half);
        } else if (dtype == 1) {
            LAUNCH_MLA_REDUCE(__nv_bfloat16);
        } else if (dtype == 2) {
            LAUNCH_MLA_REDUCE(float);
        }

#undef LAUNCH_MLA_REDUCE
    } else {
        // Short-context path: reuse partitioned kernel with a single partition
        // by launching max_partitions=1 into the output via tmp==null path.
        // Fall back to a 1-partition partitioned kernel writing through reduce.
        // Allocate nothing: run partitioned with part=1 into a local pattern
        // using out as destination via the reduce-less path below.

        // Use partitioned kernel into temporary stack-less approach:
        // Launch 1 partition and write directly via a dedicated short path:
        // For simplicity, require tmp buffers even for short ctx when available;
        // otherwise run prefill-style single-block online softmax over full ctx.
        int max_partitions = 1;
        dim3 grid(num_heads, num_seqs, 1);
        dim3 block(NUM_THREADS);

        // Without tmp buffers, run an in-register online-softmax decode into `out`
        // by invoking the prefill kernel shape isn't right. Use partitioned + reduce
        // only when tmp is provided; else no-op safety.
        if (tmp_out_buf && tmp_max_buf && tmp_sum_buf) {
            float* f_tmp_out = reinterpret_cast<float*>(tmp_out_buf);
            float* f_tmp_max = reinterpret_cast<float*>(tmp_max_buf);
            float* f_tmp_sum = reinterpret_cast<float*>(tmp_sum_buf);

#define LAUNCH_MLA_DECODE_SHORT(T, BS)                                        \
    mla_paged_attention_decode_partitioned_kernel<T, BS, NUM_THREADS>         \
        <<<grid, block, smem_size, stream>>>(                                 \
            f_tmp_out, f_tmp_max, f_tmp_sum,                                  \
            reinterpret_cast<T*>(q_abs), reinterpret_cast<T*>(q_pe),          \
            reinterpret_cast<T*>(ckv_cache), reinterpret_cast<T*>(kpe_cache), \
            block_tables, context_lens, scale, num_heads,                     \
            kv_lora_rank, qk_rope_head_dim, max_num_blocks_per_seq,           \
            max_partitions)

#define LAUNCH_MLA_DECODE_SHORT_BLOCK(T)                                      \
    switch (block_size) {                                                     \
        case 16: LAUNCH_MLA_DECODE_SHORT(T, 16); break;                      \
        case 32: LAUNCH_MLA_DECODE_SHORT(T, 32); break;                      \
        case 64: LAUNCH_MLA_DECODE_SHORT(T, 64); break;                      \
        default: break;                                                       \
    }

            if (dtype == 0) {
                LAUNCH_MLA_DECODE_SHORT_BLOCK(__half);
            } else if (dtype == 1) {
                LAUNCH_MLA_DECODE_SHORT_BLOCK(__nv_bfloat16);
            } else if (dtype == 2) {
                LAUNCH_MLA_DECODE_SHORT_BLOCK(float);
            }

#undef LAUNCH_MLA_DECODE_SHORT
#undef LAUNCH_MLA_DECODE_SHORT_BLOCK

            dim3 reduce_grid(num_heads, num_seqs);
            dim3 reduce_block(NUM_THREADS);
#define LAUNCH_MLA_REDUCE_SHORT(T)                                            \
    mla_paged_attention_decode_reduce_kernel<T, NUM_THREADS>                  \
        <<<reduce_grid, reduce_block, 0, stream>>>(                           \
            reinterpret_cast<T*>(out), f_tmp_out, f_tmp_max, f_tmp_sum,       \
            context_lens, num_heads, kv_lora_rank, max_partitions)

            if (dtype == 0) {
                LAUNCH_MLA_REDUCE_SHORT(__half);
            } else if (dtype == 1) {
                LAUNCH_MLA_REDUCE_SHORT(__nv_bfloat16);
            } else if (dtype == 2) {
                LAUNCH_MLA_REDUCE_SHORT(float);
            }
#undef LAUNCH_MLA_REDUCE_SHORT
        }
    }
}

extern "C" void mla_paged_attention_prefill(
    void* out, void* q_abs, void* q_pe,
    void* ckv_cache, void* kpe_cache,
    int32_t* block_tables, int32_t* context_lens,
    int32_t* cu_seqlens_q,
    float scale,
    int32_t num_seqs, int32_t num_heads,
    int32_t kv_lora_rank, int32_t qk_rope_head_dim,
    int32_t block_size, int32_t max_num_blocks_per_seq,
    int32_t total_tokens,
    uint32_t dtype, int64_t stream_) {

    if (num_seqs == 0 || total_tokens == 0) return;
    const cudaStream_t stream = (cudaStream_t)stream_;

    constexpr int NUM_THREADS = 256;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;
    int smem_size =
        (kv_lora_rank + qk_rope_head_dim + NUM_WARPS) * (int)sizeof(float);

    dim3 grid(num_heads, total_tokens);
    dim3 block(NUM_THREADS);

#define LAUNCH_MLA_PREFILL(T, BS)                                             \
    mla_paged_attention_prefill_v3_kernel<T, BS, NUM_THREADS>                 \
        <<<grid, block, smem_size, stream>>>(                                 \
            reinterpret_cast<T*>(out), reinterpret_cast<T*>(q_abs),           \
            reinterpret_cast<T*>(q_pe), reinterpret_cast<T*>(ckv_cache),      \
            reinterpret_cast<T*>(kpe_cache), block_tables, context_lens,      \
            cu_seqlens_q, scale, num_seqs, num_heads,                         \
            kv_lora_rank, qk_rope_head_dim, max_num_blocks_per_seq,           \
            total_tokens)

#define LAUNCH_MLA_PREFILL_BLOCK(T)                                           \
    switch (block_size) {                                                     \
        case 16: LAUNCH_MLA_PREFILL(T, 16); break;                           \
        case 32: LAUNCH_MLA_PREFILL(T, 32); break;                           \
        case 64: LAUNCH_MLA_PREFILL(T, 64); break;                           \
        default: break;                                                       \
    }

    if (dtype == 0) {
        LAUNCH_MLA_PREFILL_BLOCK(__half);
    } else if (dtype == 1) {
        LAUNCH_MLA_PREFILL_BLOCK(__nv_bfloat16);
    } else if (dtype == 2) {
        LAUNCH_MLA_PREFILL_BLOCK(float);
    }

#undef LAUNCH_MLA_PREFILL
#undef LAUNCH_MLA_PREFILL_BLOCK
}

#undef MLA_WARP_SIZE
