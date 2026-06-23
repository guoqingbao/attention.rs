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

__device__ __forceinline__ float bf16_vec_dot(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    int len) {
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    int d = 0;
    const int vec_len8 = len & ~7;
    for (; d < vec_len8; d += 8) {
        __nv_bfloat162 va0 = *reinterpret_cast<const __nv_bfloat162*>(a + d);
        __nv_bfloat162 vb0 = *reinterpret_cast<const __nv_bfloat162*>(b + d);
        __nv_bfloat162 va1 = *reinterpret_cast<const __nv_bfloat162*>(a + d + 2);
        __nv_bfloat162 vb1 = *reinterpret_cast<const __nv_bfloat162*>(b + d + 2);
        __nv_bfloat162 va2 = *reinterpret_cast<const __nv_bfloat162*>(a + d + 4);
        __nv_bfloat162 vb2 = *reinterpret_cast<const __nv_bfloat162*>(b + d + 4);
        __nv_bfloat162 va3 = *reinterpret_cast<const __nv_bfloat162*>(a + d + 6);
        __nv_bfloat162 vb3 = *reinterpret_cast<const __nv_bfloat162*>(b + d + 6);
        acc0 += __bfloat162float(va0.x) * __bfloat162float(vb0.x)
              + __bfloat162float(va0.y) * __bfloat162float(vb0.y);
        acc1 += __bfloat162float(va1.x) * __bfloat162float(vb1.x)
              + __bfloat162float(va1.y) * __bfloat162float(vb1.y);
        acc2 += __bfloat162float(va2.x) * __bfloat162float(vb2.x)
              + __bfloat162float(va2.y) * __bfloat162float(vb2.y);
        acc3 += __bfloat162float(va3.x) * __bfloat162float(vb3.x)
              + __bfloat162float(va3.y) * __bfloat162float(vb3.y);
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; d + 1 < len; d += 2) {
        __nv_bfloat162 va = *reinterpret_cast<const __nv_bfloat162*>(a + d);
        __nv_bfloat162 vb = *reinterpret_cast<const __nv_bfloat162*>(b + d);
        acc += __bfloat162float(va.x) * __bfloat162float(vb.x)
             + __bfloat162float(va.y) * __bfloat162float(vb.y);
    }
    if (d < len) {
        acc += __bfloat162float(a[d]) * __bfloat162float(b[d]);
    }
    return acc;
}

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
            float dot = bf16_vec_dot(q + q_base + h * head_dim, k + k_base, head_dim);
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
/*  Uses streaming accumulation with PARALLEL merge:                   */
/*  1. Compute chunk scores (parallel across threads)                  */
/*  2. Merge chunk into top-k buffer using parallel threshold filter   */
/*     + cooperative replacement (all threads participate)              */
/*                                                                     */
/*  Algorithm: After scoring a chunk, find the current k-th largest    */
/*  value in the combined buffer (threshold). Keep only entries above  */
/*  threshold. This avoids O(k) min-scan per candidate.                */
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
    /* Layout: [CHUNK_SIZE chunk_scores] [topk merge_vals] [topk merge_idxs]
               [NUM_THREADS reduce buf] */
    float* chunk_scores = reinterpret_cast<float*>(smem);
    float* topk_vals    = chunk_scores + CHUNK_SIZE;
    int*   topk_idxs    = reinterpret_cast<int*>(topk_vals + topk);
    float* reduce_buf   = reinterpret_cast<float*>(topk_idxs + topk);

    for (int i = tid; i < topk; i += NUM_THREADS) {
        topk_vals[i] = -FLT_MAX;
        topk_idxs[i] = -1;
    }
    __syncthreads();

    const int q_base = qi * n_heads * head_dim;
    const float* w_row = weights + qi * n_heads;

    for (int chunk_start = 0; chunk_start < causal_len; chunk_start += CHUNK_SIZE) {
        int chunk_end = min(chunk_start + CHUNK_SIZE, causal_len);
        int chunk_len = chunk_end - chunk_start;

        for (int local_j = tid; local_j < chunk_len; local_j += NUM_THREADS) {
            int kj = chunk_start + local_j;
            const int k_base = kj * head_dim;
            float acc = 0.0f;

            for (int h = 0; h < n_heads; ++h) {
                float dot = bf16_vec_dot(q + q_base + h * head_dim, k + k_base, head_dim);
                acc += fmaxf(dot, 0.0f) * w_row[h];
            }
            chunk_scores[local_j] = acc * score_scale;
        }
        for (int local_j = chunk_len + tid; local_j < CHUNK_SIZE; local_j += NUM_THREADS) {
            chunk_scores[local_j] = -FLT_MAX;
        }
        __syncthreads();

        // Parallel min-reduction to find current threshold (k-th largest value)
        float local_min = FLT_MAX;
        for (int i = tid; i < topk; i += NUM_THREADS) {
            local_min = fminf(local_min, topk_vals[i]);
        }
        reduce_buf[tid] = local_min;
        __syncthreads();

        for (int stride = NUM_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] = fminf(reduce_buf[tid], reduce_buf[tid + stride]);
            }
            __syncthreads();
        }
        float current_threshold = reduce_buf[0];

        /* Step 2: Process chunk candidates via serial-on-thread-0 but with
           early-exit threshold filtering. The key optimization vs the old code:
           - Parallel threshold computation lets us skip ~90% of candidates
           - We only scan topk_vals for the min when we actually have a
             qualifying candidate (amortized cost is much lower)
           - Refresh threshold periodically to tighten the filter */
        if (tid == 0) {
            int insert_count = 0;
            for (int cj = 0; cj < chunk_len; ++cj) {
                float cand = chunk_scores[cj];
                if (cand <= current_threshold) continue;

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
                    // After replacement, the new min is at least min_val
                    // (the second-smallest). Update threshold to min_val found
                    // during next scan. The min_val we just found is the old min
                    // which we replaced, so the actual new min must be >= all
                    // remaining entries. But we don't know it cheaply — use the
                    // second-smallest seen during the scan as a lower bound.
                    // For correctness: refresh threshold every N insertions.
                    insert_count++;
                    if ((insert_count & 31) == 0) {
                        float new_min = topk_vals[0];
                        for (int t = 1; t < topk; ++t) {
                            new_min = fminf(new_min, topk_vals[t]);
                        }
                        current_threshold = new_min;
                    }
                }
            }
        }
        __syncthreads();
    }

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

    // With chunked prefill (max 8K tokens per chunk), scores always fit in
    // shared memory (8K * 4 = 32KB < 48KB). The small-seq path handles all
    // practical cases. The chunked kernel is kept as a safety net for unusual
    // configurations (e.g., disabled chunked prefill).
    size_t scores_bytes = seq_len * sizeof(float);
    size_t reduce_bytes = 2 * NUM_THREADS * (sizeof(float) + sizeof(int));
    size_t total_smem = scores_bytes + reduce_bytes;

    if (total_smem <= 48 * 1024) {
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
        constexpr int CHUNK_SIZE = 4096;
        size_t chunk_smem =
            CHUNK_SIZE * sizeof(float) +           // chunk_scores
            topk * (sizeof(float) + sizeof(int)) + // topk buffer
            NUM_THREADS * sizeof(float);           // reduce buffer
        dim3 grid(seq_len);
        dim3 block(NUM_THREADS);

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
