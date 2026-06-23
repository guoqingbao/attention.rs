/**
 * Fused DSA Lightning Indexer kernel for GLM-5.2 / DeepSeek V3.2.
 *
 * Computes per-position index scores from multi-head Q and single-head K
 * (MQA-style indexer), applies causal mask, and selects top-k indices —
 * all without materializing the O(n²) intermediate score matrix.
 *
 * Score formula per (query_pos i, key_pos j):
 *   raw_score = sum_h( relu(q[i,h,:] · k[j,:]) * w[i,h] ) * scale
 *   index_score[i,j] = raw_score   if j <= i   (causal)
 *                     = -inf        otherwise
 *
 * Then per-row top-k selection picks the k highest-scoring key positions.
 *
 * Layout:
 *   q:       [seq_len, n_heads, head_dim]  BF16
 *   k:       [seq_len, head_dim]           BF16
 *   weights: [seq_len, n_heads]            F32
 *   output:  [seq_len, topk]               I32
 *
 * Grid: (seq_len, 1, 1)   — one block per query position
 * Block: 256 threads
 *
 * Each block computes scores for its query position against all valid key
 * positions (j=0..i for causal), stores them in shared memory or processes
 * in chunks, then selects top-k.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define IDX_WARP_SIZE 32

/* ------------------------------------------------------------------ */
/*  Fused score + causal top-k kernel (small seq, fits in shared mem) */
/*  Grid: (seq_len,)   Block: (256,)                                  */
/*  Each block handles one query position.                             */
/* ------------------------------------------------------------------ */

template <int NUM_THREADS>
__global__ void dsa_lightning_indexer_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ weights,
    int32_t* __restrict__ topk_out,
    const int seq_len,
    const int n_heads,
    const int head_dim,
    const int topk,
    const float score_scale) {

    const int qi = blockIdx.x;
    if (qi >= seq_len) return;
    const int tid = threadIdx.x;
    const int causal_len = qi + 1;

    extern __shared__ char smem[];
    float* scores = reinterpret_cast<float*>(smem);

    const int q_base = qi * n_heads * head_dim;
    const float* w_row = weights + qi * n_heads;

    for (int kj = tid; kj < causal_len; kj += NUM_THREADS) {
        const int k_base = kj * head_dim;
        float acc = 0.0f;

        for (int h = 0; h < n_heads; ++h) {
            float dot = 0.0f;
            const int qh_base = q_base + h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                dot += __bfloat162float(q[qh_base + d]) *
                       __bfloat162float(k[k_base + d]);
            }
            acc += fmaxf(dot, 0.0f) * w_row[h];
        }

        scores[kj] = acc * score_scale;
    }

    for (int kj = causal_len + tid; kj < seq_len; kj += NUM_THREADS) {
        scores[kj] = -FLT_MAX;
    }
    __syncthreads();

    /* Top-k selection via iterative argmax + invalidation.
       Efficient when topk << seq_len. Shared memory stays hot. */
    __shared__ float thread_best_scores[NUM_THREADS];
    __shared__ int   thread_best_indices[NUM_THREADS];

    const int actual_topk = min(topk, causal_len);

    for (int route = 0; route < actual_topk; ++route) {
        float best_score = -FLT_MAX;
        int best_idx = -1;
        for (int kj = tid; kj < causal_len; kj += NUM_THREADS) {
            float s = scores[kj];
            if (s > best_score || (s == best_score && kj < best_idx)) {
                best_score = s;
                best_idx = kj;
            }
        }
        thread_best_scores[tid] = best_score;
        thread_best_indices[tid] = best_idx;
        __syncthreads();

        for (int stride = NUM_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float other_s = thread_best_scores[tid + stride];
                int   other_i = thread_best_indices[tid + stride];
                float curr_s = thread_best_scores[tid];
                int   curr_i = thread_best_indices[tid];
                if (other_s > curr_s ||
                    (other_s == curr_s && other_i >= 0 &&
                     (curr_i < 0 || other_i < curr_i))) {
                    thread_best_scores[tid] = other_s;
                    thread_best_indices[tid] = other_i;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            int idx = thread_best_indices[0];
            topk_out[qi * topk + route] = idx;
            if (idx >= 0) scores[idx] = -FLT_MAX;
        }
        __syncthreads();
    }

    for (int route = actual_topk + tid; route < topk; route += NUM_THREADS) {
        topk_out[qi * topk + route] = -1;
    }
}


/* ------------------------------------------------------------------ */
/*  Chunked version for large sequences (scores don't fit in smem).   */
/*  Uses streaming accumulation: compute per-chunk scores into smem,  */
/*  maintain a running top-k buffer.                                   */
/* ------------------------------------------------------------------ */

template <int NUM_THREADS, int CHUNK_SIZE>
__global__ void dsa_lightning_indexer_chunked_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ weights,
    int32_t* __restrict__ topk_out,
    const int seq_len,
    const int n_heads,
    const int head_dim,
    const int topk,
    const float score_scale) {

    const int qi = blockIdx.x;
    if (qi >= seq_len) return;
    const int tid = threadIdx.x;
    const int causal_len = qi + 1;

    extern __shared__ char smem[];
    /* Layout: [CHUNK_SIZE scores] [topk_vals F32] [topk_idxs I32]
               [NUM_THREADS reduce_scores] [NUM_THREADS reduce_idxs] */
    float* chunk_scores = reinterpret_cast<float*>(smem);
    float* topk_vals    = chunk_scores + CHUNK_SIZE;
    int*   topk_idxs    = reinterpret_cast<int*>(topk_vals + topk);
    float* reduce_scores = reinterpret_cast<float*>(topk_idxs + topk);
    int*   reduce_idxs   = reinterpret_cast<int*>(reduce_scores + NUM_THREADS);

    // Initialize top-k buffer with -inf
    for (int i = tid; i < topk; i += NUM_THREADS) {
        topk_vals[i] = -FLT_MAX;
        topk_idxs[i] = -1;
    }
    __syncthreads();

    // Find minimum in current top-k buffer (the entry to potentially replace)
    // We maintain a simple heap-free approach: find global min, compare with chunk candidates.

    const int q_base = qi * n_heads * head_dim;
    const float* w_row = weights + qi * n_heads;

    for (int chunk_start = 0; chunk_start < causal_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = min(chunk_start + CHUNK_SIZE, causal_len);
        int chunk_len = chunk_end - chunk_start;

        // Compute scores for this chunk
        for (int local_j = tid; local_j < chunk_len; local_j += NUM_THREADS) {
            int kj = chunk_start + local_j;
            const int k_base = kj * head_dim;
            float acc = 0.0f;

            for (int h = 0; h < n_heads; ++h) {
                float dot = 0.0f;
                const int qh_base = q_base + h * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    dot += __bfloat162float(q[qh_base + d]) *
                           __bfloat162float(k[k_base + d]);
                }
                acc += fmaxf(dot, 0.0f) * w_row[h];
            }
            chunk_scores[local_j] = acc * score_scale;
        }
        for (int local_j = chunk_len + tid; local_j < CHUNK_SIZE; local_j += NUM_THREADS) {
            chunk_scores[local_j] = -FLT_MAX;
        }
        __syncthreads();

        /* Merge chunk candidates into top-k buffer.
           Strategy: for each candidate in the chunk, if it's better than
           the current min of topk_vals, replace the min. Thread 0 does serial
           merge (topk is small, typically 2048, and we only check chunk_len items). */
        if (tid == 0) {
            // Find current min in topk buffer
            for (int cj = 0; cj < chunk_len; ++cj) {
                float cand = chunk_scores[cj];
                // Find min in topk buffer
                int min_pos = 0;
                float min_val = topk_vals[0];
                for (int t = 1; t < topk; ++t) {
                    if (topk_vals[t] < min_val) {
                        min_val = topk_vals[t];
                        min_pos = t;
                    }
                }
                if (cand > min_val) {
                    topk_vals[min_pos] = cand;
                    topk_idxs[min_pos] = chunk_start + cj;
                }
            }
        }
        __syncthreads();
    }

    // Write output
    for (int i = tid; i < topk; i += NUM_THREADS) {
        topk_out[qi * topk + i] = topk_idxs[i];
    }
}


/* ------------------------------------------------------------------ */
/*  C API                                                              */
/* ------------------------------------------------------------------ */

extern "C" {

cudaError_t dsa_lightning_indexer_prefill(
    const void* q,           // [seq_len, n_heads, head_dim] BF16
    const void* k,           // [seq_len, head_dim]          BF16
    const void* weights,     // [seq_len, n_heads]           F32
    int32_t* topk_out,       // [seq_len, topk]              I32
    int seq_len,
    int n_heads,
    int head_dim,
    int topk,
    float score_scale,
    int64_t stream_) {

    if (seq_len <= 0 || n_heads <= 0 || head_dim <= 0 || topk <= 0)
        return cudaErrorInvalidValue;

    const cudaStream_t stream = (cudaStream_t)stream_;
    constexpr int NUM_THREADS = 256;

    // Decide: shared-memory based (small seq) or chunked (large seq)
    // smem budget: scores[seq_len] + reduce buffers
    size_t scores_bytes = seq_len * sizeof(float);
    size_t reduce_bytes = 2 * NUM_THREADS * (sizeof(float) + sizeof(int));
    size_t total_smem = scores_bytes + reduce_bytes;

    // Max shared memory is typically 48KB (default) or 100KB+ (configurable)
    if (total_smem <= 48 * 1024) {
        // Small-seq path: all scores fit in shared memory
        dim3 grid(seq_len);
        dim3 block(NUM_THREADS);
        dsa_lightning_indexer_kernel<NUM_THREADS>
            <<<grid, block, total_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q),
                reinterpret_cast<const __nv_bfloat16*>(k),
                reinterpret_cast<const float*>(weights),
                topk_out,
                seq_len, n_heads, head_dim, topk, score_scale);
    } else {
        // Large-seq path: chunked processing
        constexpr int CHUNK_SIZE = 4096;
        size_t chunk_smem =
            CHUNK_SIZE * sizeof(float) +           // chunk_scores
            topk * (sizeof(float) + sizeof(int)) + // topk buffer
            NUM_THREADS * (sizeof(float) + sizeof(int)); // reduce buffers
        dim3 grid(seq_len);
        dim3 block(NUM_THREADS);

        // Set max shared memory if needed
        if (chunk_smem > 48 * 1024) {
            cudaFuncSetAttribute(
                dsa_lightning_indexer_chunked_kernel<NUM_THREADS, CHUNK_SIZE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                chunk_smem);
        }

        dsa_lightning_indexer_chunked_kernel<NUM_THREADS, CHUNK_SIZE>
            <<<grid, block, chunk_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q),
                reinterpret_cast<const __nv_bfloat16*>(k),
                reinterpret_cast<const float*>(weights),
                topk_out,
                seq_len, n_heads, head_dim, topk, score_scale);
    }

    return cudaGetLastError();
}

}  // extern "C"

#undef IDX_WARP_SIZE
