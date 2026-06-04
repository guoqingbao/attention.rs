/**
 * Fused MLA (Multi-head Latent Attention) paged attention kernels.
 *
 * MLA operates in compressed KV space (kv_lora_rank) rather than standard
 * head_dim. Score = q_absorbed · ckv^T + q_pe · kpe^T, then softmax, then
 * output = attn_weights · ckv (in kv_lora_rank space).
 *
 * Copyright (c) 2025, Guoqing Bao.  All rights reserved.
 *
 * This CUDA kernel is developed for vLLM.rs project:
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
/*  Split-K Decode Kernel (Phase 1)                                    */
/*  Grid: (num_heads, num_seqs, num_partitions)                        */
/*  Each block handles a partition of KV tokens for one (head, seq).   */
/*  Writes partial results to tmp buffers.                             */
/* ------------------------------------------------------------------ */

static constexpr int PARTITION_SIZE = 128;

template <typename scalar_t, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mla_paged_attention_decode_partitioned_kernel(
    float* __restrict__ tmp_out,       // [num_seqs, num_heads, max_partitions, kv_lora_rank]
    float* __restrict__ tmp_max,       // [num_seqs, num_heads, max_partitions]
    float* __restrict__ tmp_sum,       // [num_seqs, num_heads, max_partitions]
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

    const int ctx_len = context_lens[seq_idx];
    const int part_start = part_idx * PARTITION_SIZE;
    if (part_start >= ctx_len) return;
    const int part_end = min(part_start + PARTITION_SIZE, ctx_len);
    const int part_len = part_end - part_start;

    const int q_abs_off = (seq_idx * num_heads + head_idx) * kv_lora_rank;
    const int q_pe_off = (seq_idx * num_heads + head_idx) * qk_rope_head_dim;
    const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    // Shared memory: logits[PARTITION_SIZE] + red[NUM_WARPS]
    extern __shared__ char smem_raw[];
    float* logits = reinterpret_cast<float*>(smem_raw);
    float* red_smem = logits + PARTITION_SIZE;

    // Pass 1: compute logits for this partition's KV tokens
    for (int ti = 0; ti < part_len; ti++) {
        const int t = part_start + ti;
        const int blk_idx = t / BLOCK_SIZE;
        const int blk_off = t % BLOCK_SIZE;
        const int physical_block = block_table[blk_idx];

        const int64_t ckv_base =
            (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
            (int64_t)blk_off * kv_lora_rank;
        const int64_t kpe_base =
            (int64_t)physical_block * BLOCK_SIZE * qk_rope_head_dim +
            (int64_t)blk_off * qk_rope_head_dim;

        float partial = 0.f;
        for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
            partial += (float)q_abs[q_abs_off + d] *
                       (float)ckv_cache[ckv_base + d];
        }
        for (int d = tid; d < qk_rope_head_dim; d += NUM_THREADS) {
            partial += (float)q_pe[q_pe_off + d] *
                       (float)kpe_cache[kpe_base + d];
        }

        int warp = tid / MLA_WARP_SIZE;
        int lane = tid % MLA_WARP_SIZE;
        partial = mla_warp_reduce_sum(partial);
        if (lane == 0) red_smem[warp] = partial;
        __syncthreads();
        if (warp == 0) {
            partial = (lane < NUM_WARPS) ? red_smem[lane] : 0.f;
            partial = mla_warp_reduce_sum(partial);
            if (lane == 0) logits[ti] = partial * scale;
        }
        __syncthreads();
    }

    // Pass 2: local softmax over this partition
    float local_max = -FLT_MAX;
    for (int ti = tid; ti < part_len; ti += NUM_THREADS) {
        local_max = fmaxf(local_max, logits[ti]);
    }
    int warp = tid / MLA_WARP_SIZE;
    int lane = tid % MLA_WARP_SIZE;
    local_max = mla_warp_reduce_max(local_max);
    if (lane == 0) red_smem[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        local_max = (lane < NUM_WARPS) ? red_smem[lane] : -FLT_MAX;
        local_max = mla_warp_reduce_max(local_max);
        if (lane == 0) red_smem[0] = local_max;
    }
    __syncthreads();
    float part_max = red_smem[0];

    float local_sum = 0.f;
    for (int ti = tid; ti < part_len; ti += NUM_THREADS) {
        float e = expf(logits[ti] - part_max);
        logits[ti] = e;
        local_sum += e;
    }
    local_sum = mla_warp_reduce_sum(local_sum);
    if (lane == 0) red_smem[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        local_sum = (lane < NUM_WARPS) ? red_smem[lane] : 0.f;
        local_sum = mla_warp_reduce_sum(local_sum);
        if (lane == 0) red_smem[0] = local_sum;
    }
    __syncthreads();
    float part_sum = red_smem[0];

    // Store partition metadata
    const int meta_off = (seq_idx * num_heads + head_idx) * max_partitions + part_idx;
    if (tid == 0) {
        tmp_max[meta_off] = part_max;
        tmp_sum[meta_off] = part_sum;
    }

    // Pass 3: weighted sum of ckv values (unnormalized — divide by global sum in reduce)
    // Store as sum_i(exp(logit_i - part_max) * ckv_i) — the reduce kernel handles correction
    const int64_t out_off =
        ((int64_t)seq_idx * num_heads + head_idx) * max_partitions * kv_lora_rank +
        (int64_t)part_idx * kv_lora_rank;
    for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
        float acc = 0.f;
        for (int ti = 0; ti < part_len; ti++) {
            const int t = part_start + ti;
            const int blk_idx = t / BLOCK_SIZE;
            const int blk_off = t % BLOCK_SIZE;
            const int physical_block = block_table[blk_idx];
            const int64_t ckv_base =
                (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
                (int64_t)blk_off * kv_lora_rank;
            acc += logits[ti] * (float)ckv_cache[ckv_base + d];
        }
        tmp_out[out_off + d] = acc;
    }
}

/* ------------------------------------------------------------------ */
/*  Reduce Kernel (Phase 2)                                            */
/*  Grid: (num_heads, num_seqs)                                        */
/*  Merges partition results using online softmax correction.          */
/* ------------------------------------------------------------------ */

template <typename scalar_t, int NUM_THREADS>
__global__ void mla_paged_attention_decode_reduce_kernel(
    scalar_t* __restrict__ out,        // [num_seqs, num_heads, kv_lora_rank]
    const float* __restrict__ tmp_out, // [num_seqs, num_heads, max_partitions, kv_lora_rank]
    const float* __restrict__ tmp_max, // [num_seqs, num_heads, max_partitions]
    const float* __restrict__ tmp_sum, // [num_seqs, num_heads, max_partitions]
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

    if (num_parts == 1) {
        // Single partition: just copy and convert
        const int64_t tmp_off =
            ((int64_t)seq_idx * num_heads + head_idx) * max_partitions * kv_lora_rank;
        const int out_off = (seq_idx * num_heads + head_idx) * kv_lora_rank;
        const float inv_sum = tmp_sum[(seq_idx * num_heads + head_idx) * max_partitions] > 0.f
            ? 1.f / tmp_sum[(seq_idx * num_heads + head_idx) * max_partitions]
            : 0.f;
        for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
            out[out_off + d] = (scalar_t)(tmp_out[tmp_off + d] * inv_sum);
        }
        return;
    }

    // Find global max across partitions
    const int meta_base = (seq_idx * num_heads + head_idx) * max_partitions;
    float global_max = -FLT_MAX;
    for (int p = 0; p < num_parts; p++) {
        global_max = fmaxf(global_max, tmp_max[meta_base + p]);
    }

    // Compute corrected global sum and merge outputs
    // For each partition p:
    //   correction_p = exp(part_max_p - global_max)
    //   corrected_sum_p = part_sum_p * correction_p
    //   corrected_out_p = tmp_out_p * correction_p  (per dimension)
    // global_sum = sum of corrected_sum_p
    // final_out = sum(corrected_out_p) / global_sum

    float global_sum = 0.f;
    for (int p = 0; p < num_parts; p++) {
        float correction = expf(tmp_max[meta_base + p] - global_max);
        global_sum += tmp_sum[meta_base + p] * correction;
    }
    float inv_global_sum = (global_sum > 0.f) ? (1.f / global_sum) : 0.f;

    const int out_off = (seq_idx * num_heads + head_idx) * kv_lora_rank;
    const int64_t tmp_base =
        ((int64_t)seq_idx * num_heads + head_idx) * max_partitions * kv_lora_rank;

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
/*  Single-block Decode Kernel (for short contexts / fallback)         */
/*  Grid: (num_heads, num_seqs)                                        */
/* ------------------------------------------------------------------ */

template <typename scalar_t, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mla_paged_attention_decode_kernel(
    scalar_t* __restrict__ out,
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
    const int max_num_blocks_per_seq) {

    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len == 0) return;

    const int q_abs_off = (seq_idx * num_heads + head_idx) * kv_lora_rank;
    const int q_pe_off = (seq_idx * num_heads + head_idx) * qk_rope_head_dim;
    const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    extern __shared__ char smem_raw[];
    float* logits = reinterpret_cast<float*>(smem_raw);
    float* red_smem = logits + ctx_len;

    for (int t = 0; t < ctx_len; t++) {
        const int blk_idx = t / BLOCK_SIZE;
        const int blk_off = t % BLOCK_SIZE;
        const int physical_block = block_table[blk_idx];

        const int64_t ckv_base =
            (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
            (int64_t)blk_off * kv_lora_rank;
        const int64_t kpe_base =
            (int64_t)physical_block * BLOCK_SIZE * qk_rope_head_dim +
            (int64_t)blk_off * qk_rope_head_dim;

        float partial = 0.f;
        for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
            partial += (float)q_abs[q_abs_off + d] *
                       (float)ckv_cache[ckv_base + d];
        }
        for (int d = tid; d < qk_rope_head_dim; d += NUM_THREADS) {
            partial += (float)q_pe[q_pe_off + d] *
                       (float)kpe_cache[kpe_base + d];
        }

        int warp = tid / MLA_WARP_SIZE;
        int lane = tid % MLA_WARP_SIZE;
        partial = mla_warp_reduce_sum(partial);
        if (lane == 0) red_smem[warp] = partial;
        __syncthreads();
        if (warp == 0) {
            partial = (lane < NUM_WARPS) ? red_smem[lane] : 0.f;
            partial = mla_warp_reduce_sum(partial);
            if (lane == 0) logits[t] = partial * scale;
        }
        __syncthreads();
    }

    float local_max = -FLT_MAX;
    for (int t = tid; t < ctx_len; t += NUM_THREADS) {
        local_max = fmaxf(local_max, logits[t]);
    }
    int warp = tid / MLA_WARP_SIZE;
    int lane = tid % MLA_WARP_SIZE;
    local_max = mla_warp_reduce_max(local_max);
    if (lane == 0) red_smem[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        local_max = (lane < NUM_WARPS) ? red_smem[lane] : -FLT_MAX;
        local_max = mla_warp_reduce_max(local_max);
        if (lane == 0) red_smem[0] = local_max;
    }
    __syncthreads();
    float global_max = red_smem[0];

    float local_sum = 0.f;
    for (int t = tid; t < ctx_len; t += NUM_THREADS) {
        float e = expf(logits[t] - global_max);
        logits[t] = e;
        local_sum += e;
    }
    local_sum = mla_warp_reduce_sum(local_sum);
    if (lane == 0) red_smem[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        local_sum = (lane < NUM_WARPS) ? red_smem[lane] : 0.f;
        local_sum = mla_warp_reduce_sum(local_sum);
        if (lane == 0) red_smem[0] = local_sum;
    }
    __syncthreads();
    float inv_sum = (red_smem[0] > 0.f) ? (1.f / red_smem[0]) : 0.f;

    for (int t = tid; t < ctx_len; t += NUM_THREADS) {
        logits[t] *= inv_sum;
    }
    __syncthreads();

    const int out_off = (seq_idx * num_heads + head_idx) * kv_lora_rank;
    for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
        float acc = 0.f;
        for (int t = 0; t < ctx_len; t++) {
            const int blk_idx = t / BLOCK_SIZE;
            const int blk_off = t % BLOCK_SIZE;
            const int physical_block = block_table[blk_idx];
            const int64_t ckv_base =
                (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
                (int64_t)blk_off * kv_lora_rank;
            acc += logits[t] * (float)ckv_cache[ckv_base + d];
        }
        out[out_off + d] = (scalar_t)acc;
    }
}

/* ------------------------------------------------------------------ */
/*  MLA Prefill Kernel                                                 */
/* ------------------------------------------------------------------ */

template <typename scalar_t, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mla_paged_attention_prefill_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q_abs,
    const scalar_t* __restrict__ q_pe,
    const scalar_t* __restrict__ ckv_cache,
    const scalar_t* __restrict__ kpe_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ cu_seqlens_q,
    const float scale,
    const int num_heads,
    const int kv_lora_rank,
    const int qk_rope_head_dim,
    const int max_num_blocks_per_seq) {

    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len == 0) return;

    const int q_start = cu_seqlens_q[seq_idx];
    const int q_end = cu_seqlens_q[seq_idx + 1];
    const int q_len = q_end - q_start;
    if (q_len == 0) return;

    const int q_pos_start = ctx_len - q_len;
    const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    extern __shared__ char smem_raw[];
    float* logits = reinterpret_cast<float*>(smem_raw);
    float* red_smem = logits + ctx_len;

    for (int qi = 0; qi < q_len; qi++) {
        const int global_q_idx = q_start + qi;
        const int causal_limit = q_pos_start + qi + 1;
        const int attend_len = min(ctx_len, causal_limit);

        const int q_abs_off =
            (global_q_idx * num_heads + head_idx) * kv_lora_rank;
        const int q_pe_off =
            (global_q_idx * num_heads + head_idx) * qk_rope_head_dim;

        for (int t = 0; t < attend_len; t++) {
            const int blk_idx = t / BLOCK_SIZE;
            const int blk_off = t % BLOCK_SIZE;
            const int physical_block = block_table[blk_idx];

            const int64_t ckv_base =
                (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
                (int64_t)blk_off * kv_lora_rank;
            const int64_t kpe_base =
                (int64_t)physical_block * BLOCK_SIZE * qk_rope_head_dim +
                (int64_t)blk_off * qk_rope_head_dim;

            float partial = 0.f;
            for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
                partial += (float)q_abs[q_abs_off + d] *
                           (float)ckv_cache[ckv_base + d];
            }
            for (int d = tid; d < qk_rope_head_dim; d += NUM_THREADS) {
                partial += (float)q_pe[q_pe_off + d] *
                           (float)kpe_cache[kpe_base + d];
            }

            int warp = tid / MLA_WARP_SIZE;
            int lane = tid % MLA_WARP_SIZE;
            partial = mla_warp_reduce_sum(partial);
            if (lane == 0) red_smem[warp] = partial;
            __syncthreads();
            if (warp == 0) {
                partial = (lane < NUM_WARPS) ? red_smem[lane] : 0.f;
                partial = mla_warp_reduce_sum(partial);
                if (lane == 0) logits[t] = partial * scale;
            }
            __syncthreads();
        }

        for (int t = attend_len + tid; t < ctx_len; t += NUM_THREADS) {
            logits[t] = -FLT_MAX;
        }
        __syncthreads();

        float local_max = -FLT_MAX;
        for (int t = tid; t < attend_len; t += NUM_THREADS) {
            local_max = fmaxf(local_max, logits[t]);
        }
        int warp = tid / MLA_WARP_SIZE;
        int lane = tid % MLA_WARP_SIZE;
        local_max = mla_warp_reduce_max(local_max);
        if (lane == 0) red_smem[warp] = local_max;
        __syncthreads();
        if (warp == 0) {
            local_max = (lane < NUM_WARPS) ? red_smem[lane] : -FLT_MAX;
            local_max = mla_warp_reduce_max(local_max);
            if (lane == 0) red_smem[0] = local_max;
        }
        __syncthreads();
        float global_max = red_smem[0];

        float local_sum = 0.f;
        for (int t = tid; t < attend_len; t += NUM_THREADS) {
            float e = expf(logits[t] - global_max);
            logits[t] = e;
            local_sum += e;
        }
        local_sum = mla_warp_reduce_sum(local_sum);
        if (lane == 0) red_smem[warp] = local_sum;
        __syncthreads();
        if (warp == 0) {
            local_sum = (lane < NUM_WARPS) ? red_smem[lane] : 0.f;
            local_sum = mla_warp_reduce_sum(local_sum);
            if (lane == 0) red_smem[0] = local_sum;
        }
        __syncthreads();
        float inv_sum = (red_smem[0] > 0.f) ? (1.f / red_smem[0]) : 0.f;

        for (int t = tid; t < attend_len; t += NUM_THREADS) {
            logits[t] *= inv_sum;
        }
        __syncthreads();

        const int out_off =
            (global_q_idx * num_heads + head_idx) * kv_lora_rank;
        for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
            float acc = 0.f;
            for (int t = 0; t < attend_len; t++) {
                const int blk_idx = t / BLOCK_SIZE;
                const int blk_off = t % BLOCK_SIZE;
                const int physical_block = block_table[blk_idx];
                const int64_t ckv_base =
                    (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
                    (int64_t)blk_off * kv_lora_rank;
                acc += logits[t] * (float)ckv_cache[ckv_base + d];
            }
            out[out_off + d] = (scalar_t)acc;
        }
        __syncthreads();
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
    uint32_t dtype, int64_t stream_,
    void* tmp_out_buf, void* tmp_max_buf, void* tmp_sum_buf,
    int32_t use_partitioned) {

    if (num_seqs == 0) return;
    const cudaStream_t stream = (cudaStream_t)stream_;

    constexpr int NUM_THREADS = 256;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;

    if (use_partitioned && tmp_out_buf && tmp_max_buf && tmp_sum_buf) {
        int max_ctx = max_num_blocks_per_seq * block_size;
        int max_partitions = (max_ctx + PARTITION_SIZE - 1) / PARTITION_SIZE;

        int smem_size = (PARTITION_SIZE + NUM_WARPS) * sizeof(float);

        dim3 grid(num_heads, num_seqs, max_partitions);
        dim3 block(NUM_THREADS);

        float* f_tmp_out = reinterpret_cast<float*>(tmp_out_buf);
        float* f_tmp_max = reinterpret_cast<float*>(tmp_max_buf);
        float* f_tmp_sum = reinterpret_cast<float*>(tmp_sum_buf);

#define LAUNCH_MLA_DECODE_PART(T, BS)                                         \
    mla_paged_attention_decode_partitioned_kernel<T, BS, NUM_THREADS>          \
        <<<grid, block, smem_size, stream>>>(                                  \
            f_tmp_out, f_tmp_max, f_tmp_sum,                                   \
            reinterpret_cast<T*>(q_abs), reinterpret_cast<T*>(q_pe),           \
            reinterpret_cast<T*>(ckv_cache), reinterpret_cast<T*>(kpe_cache),  \
            block_tables, context_lens, scale, num_heads,                      \
            kv_lora_rank, qk_rope_head_dim, max_num_blocks_per_seq,            \
            max_partitions)

#define LAUNCH_MLA_DECODE_PART_BLOCK(T)                                       \
    switch (block_size) {                                                      \
        case 16: LAUNCH_MLA_DECODE_PART(T, 16); break;                        \
        case 32: LAUNCH_MLA_DECODE_PART(T, 32); break;                        \
        case 64: LAUNCH_MLA_DECODE_PART(T, 64); break;                        \
        default: break;                                                        \
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

        // Phase 2: reduce
        dim3 reduce_grid(num_heads, num_seqs);
        dim3 reduce_block(NUM_THREADS);

#define LAUNCH_MLA_REDUCE(T)                                                  \
    mla_paged_attention_decode_reduce_kernel<T, NUM_THREADS>                   \
        <<<reduce_grid, reduce_block, 0, stream>>>(                            \
            reinterpret_cast<T*>(out), f_tmp_out, f_tmp_max, f_tmp_sum,        \
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
        // Fallback: single-block kernel for short contexts
        int max_ctx = max_num_blocks_per_seq * block_size;
        int smem_size = (max_ctx + NUM_WARPS) * sizeof(float);
        if (smem_size > 48 * 1024) {
            max_ctx = (48 * 1024) / sizeof(float) - NUM_WARPS;
            smem_size = 48 * 1024;
        }

        dim3 grid(num_heads, num_seqs);
        dim3 block(NUM_THREADS);

#define LAUNCH_MLA_DECODE(T, BS)                                              \
    mla_paged_attention_decode_kernel<T, BS, NUM_THREADS>                      \
        <<<grid, block, smem_size, stream>>>(                                  \
            reinterpret_cast<T*>(out), reinterpret_cast<T*>(q_abs),            \
            reinterpret_cast<T*>(q_pe), reinterpret_cast<T*>(ckv_cache),       \
            reinterpret_cast<T*>(kpe_cache), block_tables, context_lens,       \
            scale, num_heads, kv_lora_rank, qk_rope_head_dim,                 \
            max_num_blocks_per_seq)

#define LAUNCH_MLA_DECODE_BLOCK(T)                                            \
    switch (block_size) {                                                      \
        case 16: LAUNCH_MLA_DECODE(T, 16); break;                             \
        case 32: LAUNCH_MLA_DECODE(T, 32); break;                             \
        case 64: LAUNCH_MLA_DECODE(T, 64); break;                             \
        default: break;                                                        \
    }

        if (dtype == 0) {
            LAUNCH_MLA_DECODE_BLOCK(__half);
        } else if (dtype == 1) {
            LAUNCH_MLA_DECODE_BLOCK(__nv_bfloat16);
        } else if (dtype == 2) {
            LAUNCH_MLA_DECODE_BLOCK(float);
        }

#undef LAUNCH_MLA_DECODE
#undef LAUNCH_MLA_DECODE_BLOCK
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
    uint32_t dtype, int64_t stream_) {

    if (num_seqs == 0) return;
    const cudaStream_t stream = (cudaStream_t)stream_;

    constexpr int NUM_THREADS = 256;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;

    int max_ctx = max_num_blocks_per_seq * block_size;
    int smem_size = (max_ctx + NUM_WARPS) * sizeof(float);
    if (smem_size > 48 * 1024) {
        max_ctx = (48 * 1024) / sizeof(float) - NUM_WARPS;
        smem_size = 48 * 1024;
    }

    dim3 grid(num_heads, num_seqs);
    dim3 block(NUM_THREADS);

#define LAUNCH_MLA_PREFILL(T, BS)                                             \
    mla_paged_attention_prefill_kernel<T, BS, NUM_THREADS>                     \
        <<<grid, block, smem_size, stream>>>(                                  \
            reinterpret_cast<T*>(out), reinterpret_cast<T*>(q_abs),            \
            reinterpret_cast<T*>(q_pe), reinterpret_cast<T*>(ckv_cache),       \
            reinterpret_cast<T*>(kpe_cache), block_tables, context_lens,       \
            cu_seqlens_q, scale, num_heads, kv_lora_rank, qk_rope_head_dim,   \
            max_num_blocks_per_seq)

#define LAUNCH_MLA_PREFILL_BLOCK(T)                                           \
    switch (block_size) {                                                      \
        case 16: LAUNCH_MLA_PREFILL(T, 16); break;                            \
        case 32: LAUNCH_MLA_PREFILL(T, 32); break;                            \
        case 64: LAUNCH_MLA_PREFILL(T, 64); break;                            \
        default: break;                                                        \
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
