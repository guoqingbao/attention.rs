/**
 * Sparse MLA (Multi-head Latent Attention) attention kernels for DSA.
 *
 * These kernels operate on absorbed Q in compressed KV space, attending only
 * to the top-k tokens selected by the DSA indexer. The KV cache uses the same
 * split ckv/k_pe layout as the dense MLA kernels.
 *
 * Score = q_absorbed · ckv[topk_idx]^T + q_pe · kpe[topk_idx]^T
 * Output = softmax(scores) · ckv[topk_idx]
 *
 * Prefill: Grid(num_heads, total_tokens) — one block per (head, query_position).
 *          Much better GPU utilization than iterating queries serially.
 *
 * Decode:  Grid(num_heads, num_seqs) — one block per (head, seq).
 *          Single query token per sequence.
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define MLA_WARP_SIZE 32

__device__ __forceinline__ float sparse_mla_warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = MLA_WARP_SIZE / 2; mask >= 1; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__device__ __forceinline__ float sparse_mla_warp_reduce_max(float val) {
#pragma unroll
    for (int mask = MLA_WARP_SIZE / 2; mask >= 1; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ float vec_dot(
    const scalar_t* __restrict__ a, const scalar_t* __restrict__ b, int len) {
    float acc = 0.f;
    int d = 0;
    // Process pairs via 32-bit loads (2x BF16 or 2x FP16)
    const int vec_len = len & ~1;
    for (; d < vec_len; d += 2) {
        uint32_t va = *reinterpret_cast<const uint32_t*>(a + d);
        uint32_t vb = *reinterpret_cast<const uint32_t*>(b + d);
        scalar_t a0, a1, b0, b1;
        a0 = *reinterpret_cast<const scalar_t*>(&va);
        a1 = *(reinterpret_cast<const scalar_t*>(&va) + 1);
        b0 = *reinterpret_cast<const scalar_t*>(&vb);
        b1 = *(reinterpret_cast<const scalar_t*>(&vb) + 1);
        acc += (float)a0 * (float)b0 + (float)a1 * (float)b1;
    }
    if (d < len) {
        acc += (float)a[d] * (float)b[d];
    }
    return acc;
}

/* ------------------------------------------------------------------ */
/*  Sparse MLA Prefill Kernel — one block per (head, query_position)  */
/*  Grid: (num_heads, total_tokens)                                    */
/*  topk_indices: [total_tokens, topk] I32 — flat KV token indices    */
/* ------------------------------------------------------------------ */

template <typename scalar_t, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mla_sparse_prefill_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q_abs,
    const scalar_t* __restrict__ q_pe,
    const scalar_t* __restrict__ ckv_cache,
    const scalar_t* __restrict__ kpe_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ topk_indices,
    const float scale,
    const int num_seqs,
    const int num_heads,
    const int kv_lora_rank,
    const int qk_rope_head_dim,
    const int max_num_blocks_per_seq,
    const int topk,
    const int total_tokens) {

    const int head_idx = blockIdx.x;
    const int global_q_idx = blockIdx.y;
    const int tid = threadIdx.x;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;

    if (global_q_idx >= total_tokens) return;

    // Find which sequence this query belongs to (binary search on cu_seqlens_q)
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
    const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
    const int32_t* qi_topk = topk_indices + global_q_idx * topk;

    extern __shared__ char smem_raw[];
    float* logits = reinterpret_cast<float*>(smem_raw);
    float* red_smem = logits + topk;

    const int q_abs_off = (global_q_idx * num_heads + head_idx) * kv_lora_rank;
    const int q_pe_off = (global_q_idx * num_heads + head_idx) * qk_rope_head_dim;

    // Compute attention scores for each top-k KV token (vectorized loads)
    for (int ki = tid; ki < topk; ki += NUM_THREADS) {
        const int kv_flat_idx = qi_topk[ki];
        if (kv_flat_idx < 0 || kv_flat_idx >= ctx_len) {
            logits[ki] = -FLT_MAX;
            continue;
        }

        const int blk_idx = kv_flat_idx / BLOCK_SIZE;
        const int blk_off = kv_flat_idx % BLOCK_SIZE;
        const int physical_block = block_table[blk_idx];

        const int64_t ckv_base =
            (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
            (int64_t)blk_off * kv_lora_rank;
        const int64_t kpe_base =
            (int64_t)physical_block * BLOCK_SIZE * qk_rope_head_dim +
            (int64_t)blk_off * qk_rope_head_dim;

        float dot = vec_dot(q_abs + q_abs_off, ckv_cache + ckv_base, kv_lora_rank);
        dot += vec_dot(q_pe + q_pe_off, kpe_cache + kpe_base, qk_rope_head_dim);
        logits[ki] = dot * scale;
    }
    __syncthreads();

    // Softmax over topk logits
    float local_max = -FLT_MAX;
    for (int ki = tid; ki < topk; ki += NUM_THREADS) {
        local_max = fmaxf(local_max, logits[ki]);
    }
    int warp = tid / MLA_WARP_SIZE;
    int lane = tid % MLA_WARP_SIZE;
    local_max = sparse_mla_warp_reduce_max(local_max);
    if (lane == 0) red_smem[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        local_max = (lane < NUM_WARPS) ? red_smem[lane] : -FLT_MAX;
        local_max = sparse_mla_warp_reduce_max(local_max);
        if (lane == 0) red_smem[0] = local_max;
    }
    __syncthreads();
    float global_max = red_smem[0];

    float local_sum = 0.f;
    for (int ki = tid; ki < topk; ki += NUM_THREADS) {
        float e = expf(logits[ki] - global_max);
        logits[ki] = e;
        local_sum += e;
    }
    local_sum = sparse_mla_warp_reduce_sum(local_sum);
    if (lane == 0) red_smem[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        local_sum = (lane < NUM_WARPS) ? red_smem[lane] : 0.f;
        local_sum = sparse_mla_warp_reduce_sum(local_sum);
        if (lane == 0) red_smem[0] = local_sum;
    }
    __syncthreads();
    float inv_sum = (red_smem[0] > 0.f) ? (1.f / red_smem[0]) : 0.f;

    for (int ki = tid; ki < topk; ki += NUM_THREADS) {
        logits[ki] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum: out = sum_i(attn_weight_i * ckv[topk_i])
    // Optimization: skip near-zero weights and pre-filter valid entries
    const int out_off = (global_q_idx * num_heads + head_idx) * kv_lora_rank;
    for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
        float acc = 0.f;
        for (int ki = 0; ki < topk; ki++) {
            float w = logits[ki];
            if (w < 1e-8f) continue;
            const int kv_flat_idx = qi_topk[ki];
            if (kv_flat_idx < 0 || kv_flat_idx >= ctx_len) continue;
            const int blk_idx = kv_flat_idx / BLOCK_SIZE;
            const int blk_off = kv_flat_idx % BLOCK_SIZE;
            const int physical_block = block_table[blk_idx];
            const int64_t ckv_base =
                (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
                (int64_t)blk_off * kv_lora_rank;
            acc += w * (float)ckv_cache[ckv_base + d];
        }
        out[out_off + d] = (scalar_t)acc;
    }
}

/* ------------------------------------------------------------------ */
/*  Sparse MLA Decode Kernel                                           */
/*  Grid: (num_heads, num_seqs)                                        */
/*  Single query token per sequence, attends to topk KV tokens.        */
/*  topk_indices: [num_seqs, topk] I32                                 */
/* ------------------------------------------------------------------ */

template <typename scalar_t, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mla_sparse_decode_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q_abs,
    const scalar_t* __restrict__ q_pe,
    const scalar_t* __restrict__ ckv_cache,
    const scalar_t* __restrict__ kpe_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    const int32_t* __restrict__ topk_indices,
    const float scale,
    const int num_heads,
    const int kv_lora_rank,
    const int qk_rope_head_dim,
    const int max_num_blocks_per_seq,
    const int topk) {

    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len == 0) return;

    const int q_abs_off = (seq_idx * num_heads + head_idx) * kv_lora_rank;
    const int q_pe_off = (seq_idx * num_heads + head_idx) * qk_rope_head_dim;
    const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
    const int32_t* seq_topk = topk_indices + seq_idx * topk;

    extern __shared__ char smem_raw[];
    float* logits = reinterpret_cast<float*>(smem_raw);
    float* red_smem = logits + topk;

    // Compute logits: each thread handles multiple KV tokens (vectorized loads)
    for (int ki = tid; ki < topk; ki += NUM_THREADS) {
        const int kv_flat_idx = seq_topk[ki];
        if (kv_flat_idx < 0 || kv_flat_idx >= ctx_len) {
            logits[ki] = -FLT_MAX;
            continue;
        }

        const int blk_idx = kv_flat_idx / BLOCK_SIZE;
        const int blk_off = kv_flat_idx % BLOCK_SIZE;
        const int physical_block = block_table[blk_idx];

        const int64_t ckv_base =
            (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
            (int64_t)blk_off * kv_lora_rank;
        const int64_t kpe_base =
            (int64_t)physical_block * BLOCK_SIZE * qk_rope_head_dim +
            (int64_t)blk_off * qk_rope_head_dim;

        float dot = vec_dot(q_abs + q_abs_off, ckv_cache + ckv_base, kv_lora_rank);
        dot += vec_dot(q_pe + q_pe_off, kpe_cache + kpe_base, qk_rope_head_dim);
        logits[ki] = dot * scale;
    }
    __syncthreads();

    // Softmax
    float local_max = -FLT_MAX;
    for (int ki = tid; ki < topk; ki += NUM_THREADS) {
        local_max = fmaxf(local_max, logits[ki]);
    }
    int warp = tid / MLA_WARP_SIZE;
    int lane = tid % MLA_WARP_SIZE;
    local_max = sparse_mla_warp_reduce_max(local_max);
    if (lane == 0) red_smem[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        local_max = (lane < NUM_WARPS) ? red_smem[lane] : -FLT_MAX;
        local_max = sparse_mla_warp_reduce_max(local_max);
        if (lane == 0) red_smem[0] = local_max;
    }
    __syncthreads();
    float global_max = red_smem[0];

    float local_sum = 0.f;
    for (int ki = tid; ki < topk; ki += NUM_THREADS) {
        float e = expf(logits[ki] - global_max);
        logits[ki] = e;
        local_sum += e;
    }
    local_sum = sparse_mla_warp_reduce_sum(local_sum);
    if (lane == 0) red_smem[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        local_sum = (lane < NUM_WARPS) ? red_smem[lane] : 0.f;
        local_sum = sparse_mla_warp_reduce_sum(local_sum);
        if (lane == 0) red_smem[0] = local_sum;
    }
    __syncthreads();
    float inv_sum = (red_smem[0] > 0.f) ? (1.f / red_smem[0]) : 0.f;

    for (int ki = tid; ki < topk; ki += NUM_THREADS) {
        logits[ki] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum (skip near-zero weights)
    const int out_off = (seq_idx * num_heads + head_idx) * kv_lora_rank;
    for (int d = tid; d < kv_lora_rank; d += NUM_THREADS) {
        float acc = 0.f;
        for (int ki = 0; ki < topk; ki++) {
            float w = logits[ki];
            if (w < 1e-8f) continue;
            const int kv_flat_idx = seq_topk[ki];
            if (kv_flat_idx < 0 || kv_flat_idx >= ctx_len) continue;
            const int blk_idx = kv_flat_idx / BLOCK_SIZE;
            const int blk_off = kv_flat_idx % BLOCK_SIZE;
            const int physical_block = block_table[blk_idx];
            const int64_t ckv_base =
                (int64_t)physical_block * BLOCK_SIZE * kv_lora_rank +
                (int64_t)blk_off * kv_lora_rank;
            acc += w * (float)ckv_cache[ckv_base + d];
        }
        out[out_off + d] = (scalar_t)acc;
    }
}

/* ------------------------------------------------------------------ */
/*  C API                                                              */
/* ------------------------------------------------------------------ */

extern "C" void mla_sparse_attention_prefill(
    void* out, void* q_abs, void* q_pe,
    void* ckv_cache, void* kpe_cache,
    int32_t* block_tables, int32_t* context_lens,
    int32_t* cu_seqlens_q, int32_t* topk_indices,
    float scale,
    int32_t num_seqs, int32_t num_heads,
    int32_t kv_lora_rank, int32_t qk_rope_head_dim,
    int32_t block_size, int32_t max_num_blocks_per_seq,
    int32_t topk, int32_t total_tokens,
    uint32_t dtype, int64_t stream_) {

    if (num_seqs == 0 || total_tokens == 0) return;
    const cudaStream_t stream = (cudaStream_t)stream_;

    const int32_t total_tokens_host = total_tokens;

    constexpr int NUM_THREADS = 256;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;

    int smem_size = (topk + NUM_WARPS) * sizeof(float);

    dim3 grid(num_heads, total_tokens_host);
    dim3 block(NUM_THREADS);

#define LAUNCH_SPARSE_MLA_PREFILL(T, BS) \
    mla_sparse_prefill_kernel<T, BS, NUM_THREADS> \
        <<<grid, block, smem_size, stream>>>( \
            reinterpret_cast<T*>(out), reinterpret_cast<T*>(q_abs), \
            reinterpret_cast<T*>(q_pe), reinterpret_cast<T*>(ckv_cache), \
            reinterpret_cast<T*>(kpe_cache), block_tables, context_lens, \
            cu_seqlens_q, topk_indices, scale, num_seqs, num_heads, \
            kv_lora_rank, qk_rope_head_dim, max_num_blocks_per_seq, topk, \
            total_tokens_host)

#define LAUNCH_SPARSE_MLA_PREFILL_BLOCK(T) \
    switch (block_size) { \
        case 16: LAUNCH_SPARSE_MLA_PREFILL(T, 16); break; \
        case 32: LAUNCH_SPARSE_MLA_PREFILL(T, 32); break; \
        case 64: LAUNCH_SPARSE_MLA_PREFILL(T, 64); break; \
        default: break; \
    }

    if (dtype == 0) {
        LAUNCH_SPARSE_MLA_PREFILL_BLOCK(__half);
    } else if (dtype == 1) {
        LAUNCH_SPARSE_MLA_PREFILL_BLOCK(__nv_bfloat16);
    } else if (dtype == 2) {
        LAUNCH_SPARSE_MLA_PREFILL_BLOCK(float);
    }

#undef LAUNCH_SPARSE_MLA_PREFILL
#undef LAUNCH_SPARSE_MLA_PREFILL_BLOCK
}

extern "C" void mla_sparse_attention_decode(
    void* out, void* q_abs, void* q_pe,
    void* ckv_cache, void* kpe_cache,
    int32_t* block_tables, int32_t* context_lens,
    int32_t* topk_indices,
    float scale,
    int32_t num_seqs, int32_t num_heads,
    int32_t kv_lora_rank, int32_t qk_rope_head_dim,
    int32_t block_size, int32_t max_num_blocks_per_seq,
    int32_t topk,
    uint32_t dtype, int64_t stream_) {

    if (num_seqs == 0) return;
    const cudaStream_t stream = (cudaStream_t)stream_;

    constexpr int NUM_THREADS = 256;
    constexpr int NUM_WARPS = NUM_THREADS / MLA_WARP_SIZE;

    int smem_size = (topk + NUM_WARPS) * sizeof(float);

    dim3 grid(num_heads, num_seqs);
    dim3 block(NUM_THREADS);

#define LAUNCH_SPARSE_MLA_DECODE(T, BS) \
    mla_sparse_decode_kernel<T, BS, NUM_THREADS> \
        <<<grid, block, smem_size, stream>>>( \
            reinterpret_cast<T*>(out), reinterpret_cast<T*>(q_abs), \
            reinterpret_cast<T*>(q_pe), reinterpret_cast<T*>(ckv_cache), \
            reinterpret_cast<T*>(kpe_cache), block_tables, context_lens, \
            topk_indices, scale, num_heads, \
            kv_lora_rank, qk_rope_head_dim, max_num_blocks_per_seq, topk)

#define LAUNCH_SPARSE_MLA_DECODE_BLOCK(T) \
    switch (block_size) { \
        case 16: LAUNCH_SPARSE_MLA_DECODE(T, 16); break; \
        case 32: LAUNCH_SPARSE_MLA_DECODE(T, 32); break; \
        case 64: LAUNCH_SPARSE_MLA_DECODE(T, 64); break; \
        default: break; \
    }

    if (dtype == 0) {
        LAUNCH_SPARSE_MLA_DECODE_BLOCK(__half);
    } else if (dtype == 1) {
        LAUNCH_SPARSE_MLA_DECODE_BLOCK(__nv_bfloat16);
    } else if (dtype == 2) {
        LAUNCH_SPARSE_MLA_DECODE_BLOCK(float);
    }

#undef LAUNCH_SPARSE_MLA_DECODE
#undef LAUNCH_SPARSE_MLA_DECODE_BLOCK
}

#undef MLA_WARP_SIZE
