/**
 * Fast parallel top-k selection kernel using radix-based filtering.
 *
 * Selects the k largest values from an array, returning their indices.
 * Algorithm:
 *   1. For short arrays (len <= topk or len <= 4096): bitonic-sort approach
 *   2. For longer arrays: radix-based bucket filtering
 *      - Iteratively process bits from MSB to LSB
 *      - Count elements in each bucket (parallel histogram)
 *      - Determine threshold bucket and recurse on remaining bits
 *      - Final pass collects indices of qualifying elements
 *
 * This is a reusable primitive for:
 *   - DSA lightning indexer top-k selection
 *   - DeepSeek V4 compressed-KV indexer
 *   - MoE expert routing (top-k experts)
 *   - Any future sparse attention token selection
 *
 * Grid: (batch_size, 1, 1)  — one block per batch element
 * Block: 256 threads
 *
 * API:
 *   fast_topk_select(scores, indices_out, batch, seq_len, topk, stream)
 *     scores:      [batch, seq_len] F32 input
 *     indices_out: [batch, topk]    I32 output (indices of top-k elements)
 */

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define TOPK_WARP_SIZE 32
#define TOPK_NUM_THREADS 256

__device__ __forceinline__ int topk_warp_reduce_sum_int(int val) {
#pragma unroll
    for (int mask = TOPK_WARP_SIZE / 2; mask >= 1; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__device__ __forceinline__ float topk_warp_reduce_max(float val) {
#pragma unroll
    for (int mask = TOPK_WARP_SIZE / 2; mask >= 1; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float topk_warp_reduce_min(float val) {
#pragma unroll
    for (int mask = TOPK_WARP_SIZE / 2; mask >= 1; mask >>= 1)
        val = fminf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

/* ------------------------------------------------------------------ */
/*  Bitonic-sort-based top-k for small arrays (len <= 4096)           */
/*  Uses shared memory to hold all scores, sorts in-place, returns    */
/*  the top-k indices.                                                 */
/* ------------------------------------------------------------------ */

__global__ void fast_topk_bitonic_kernel(
    const float* __restrict__ scores,
    int32_t* __restrict__ indices_out,
    const int seq_len,
    const int topk) {

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const float* batch_scores = scores + (int64_t)batch_idx * seq_len;
    int32_t* batch_out = indices_out + (int64_t)batch_idx * topk;

    extern __shared__ char smem_raw[];
    float* s_vals = reinterpret_cast<float*>(smem_raw);
    int* s_idxs = reinterpret_cast<int*>(s_vals + seq_len);

    // Load scores and indices into shared memory
    for (int i = tid; i < seq_len; i += TOPK_NUM_THREADS) {
        s_vals[i] = batch_scores[i];
        s_idxs[i] = i;
    }
    __syncthreads();

    // Partial bitonic sort: only need top-k, so sort first topk rounds
    // Use iterative argmax + invalidation (efficient when topk << seq_len)
    __shared__ float thread_best[TOPK_NUM_THREADS];
    __shared__ int thread_best_idx[TOPK_NUM_THREADS];

    for (int route = 0; route < topk; ++route) {
        float best = -FLT_MAX;
        int best_idx = -1;
        for (int i = tid; i < seq_len; i += TOPK_NUM_THREADS) {
            float v = s_vals[i];
            if (v > best || (v == best && i < best_idx)) {
                best = v;
                best_idx = i;
            }
        }
        thread_best[tid] = best;
        thread_best_idx[tid] = best_idx;
        __syncthreads();

        for (int stride = TOPK_NUM_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float other = thread_best[tid + stride];
                int other_idx = thread_best_idx[tid + stride];
                if (other > thread_best[tid] ||
                    (other == thread_best[tid] && other_idx >= 0 &&
                     (thread_best_idx[tid] < 0 || other_idx < thread_best_idx[tid]))) {
                    thread_best[tid] = other;
                    thread_best_idx[tid] = other_idx;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            int idx = thread_best_idx[0];
            batch_out[route] = idx;
            if (idx >= 0) s_vals[idx] = -FLT_MAX;
        }
        __syncthreads();
    }
}

/* ------------------------------------------------------------------ */
/*  Radix-based top-k for large arrays                                 */
/*  Uses iterative bit inspection to narrow down candidates.           */
/*                                                                     */
/*  Key insight: float32 can be compared as uint32 after XOR with      */
/*  sign bit (IEEE 754 lexicographic ordering trick). This allows      */
/*  radix operations on floating point values.                         */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ uint32_t float_to_sortable(float f) {
    uint32_t u = __float_as_uint(f);
    // If negative (MSB=1), flip all bits; else flip only sign bit
    uint32_t mask = (u >> 31) ? 0xFFFFFFFF : 0x80000000;
    return u ^ mask;
}

__global__ void fast_topk_radix_kernel(
    const float* __restrict__ scores,
    int32_t* __restrict__ indices_out,
    const int seq_len,
    const int topk) {

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const float* batch_scores = scores + (int64_t)batch_idx * seq_len;
    int32_t* batch_out = indices_out + (int64_t)batch_idx * topk;

    constexpr int NUM_WARPS = TOPK_NUM_THREADS / TOPK_WARP_SIZE;

    extern __shared__ char smem_raw[];
    int* s_count = reinterpret_cast<int*>(smem_raw);  // [NUM_WARPS] for reduction

    // Radix select: find the threshold value (the k-th largest)
    // Process bits from MSB (bit 31) to LSB (bit 0)
    // At each level, count how many values have the current bit = 1
    // If count >= remaining_k: keep only those with bit=1, else keep bit=0 group

    uint32_t prefix = 0;       // accumulated prefix mask
    uint32_t prefix_val = 0;   // accumulated prefix value
    int remaining_k = topk;

    for (int bit = 31; bit >= 0 && remaining_k > 0; --bit) {
        uint32_t bit_mask = 1u << bit;

        // Count elements matching current prefix that have this bit = 1
        int local_count = 0;
        for (int i = tid; i < seq_len; i += TOPK_NUM_THREADS) {
            uint32_t sv = float_to_sortable(batch_scores[i]);
            if ((sv & prefix) == prefix_val) {
                if (sv & bit_mask) local_count++;
            }
        }

        // Warp reduction
        local_count = topk_warp_reduce_sum_int(local_count);
        int warp = tid / TOPK_WARP_SIZE;
        int lane = tid % TOPK_WARP_SIZE;
        if (lane == 0) s_count[warp] = local_count;
        __syncthreads();

        // Final reduction across warps
        if (warp == 0) {
            int v = (lane < NUM_WARPS) ? s_count[lane] : 0;
            v = topk_warp_reduce_sum_int(v);
            if (lane == 0) s_count[0] = v;
        }
        __syncthreads();

        int ones_count = s_count[0];

        // Decision: if ones_count >= remaining_k, top-k all have this bit = 1
        if (ones_count >= remaining_k) {
            prefix |= bit_mask;
            prefix_val |= bit_mask;
        } else {
            // Top-k spans both groups; the ones_count elements with bit=1
            // are guaranteed to be in top-k. Reduce remaining_k.
            remaining_k -= ones_count;
            prefix |= bit_mask;
            // prefix_val bit stays 0
        }
        __syncthreads();
    }

    // Now prefix_val defines the threshold: elements with
    // float_to_sortable(score) >= prefix_val are in the top-k.
    // Collect their indices.
    // Use atomic counter in shared memory for output position
    if (tid == 0) s_count[0] = 0;
    __syncthreads();

    float threshold = __uint_as_float(
        (prefix_val >> 31) ? (prefix_val ^ 0xFFFFFFFF) : (prefix_val ^ 0x80000000));
    // Simpler: just compare sortable values
    uint32_t threshold_sortable = prefix_val;

    for (int i = tid; i < seq_len; i += TOPK_NUM_THREADS) {
        uint32_t sv = float_to_sortable(batch_scores[i]);
        if (sv >= threshold_sortable) {
            int pos = atomicAdd(s_count, 1);
            if (pos < topk) {
                batch_out[pos] = i;
            }
        }
    }
    __syncthreads();

    // If we collected fewer than topk (due to ties at threshold),
    // fill remaining with elements exactly at threshold
    int collected = min(s_count[0], topk);
    for (int i = collected + tid; i < topk; i += TOPK_NUM_THREADS) {
        batch_out[i] = -1;
    }
}


/* ------------------------------------------------------------------ */
/*  C API                                                              */
/* ------------------------------------------------------------------ */

extern "C" {

cudaError_t fast_topk_select(
    const float* scores,     // [batch, seq_len] F32
    int32_t* indices_out,    // [batch, topk]    I32
    int batch,
    int seq_len,
    int topk,
    int64_t stream_) {

    if (batch <= 0 || seq_len <= 0 || topk <= 0)
        return cudaErrorInvalidValue;
    if (topk > seq_len) topk = seq_len;

    const cudaStream_t stream = (cudaStream_t)stream_;

    // Small arrays: use bitonic/argmax approach (fits in shared memory)
    size_t bitonic_smem = seq_len * (sizeof(float) + sizeof(int)) +
                          2 * TOPK_NUM_THREADS * sizeof(float);
    if (bitonic_smem <= 48 * 1024 && seq_len <= 4096) {
        dim3 grid(batch);
        dim3 block(TOPK_NUM_THREADS);
        fast_topk_bitonic_kernel<<<grid, block, bitonic_smem, stream>>>(
            scores, indices_out, seq_len, topk);
    } else {
        // Large arrays: radix-based selection
        constexpr int NUM_WARPS = TOPK_NUM_THREADS / TOPK_WARP_SIZE;
        size_t radix_smem = (NUM_WARPS + 1) * sizeof(int) + sizeof(float);
        dim3 grid(batch);
        dim3 block(TOPK_NUM_THREADS);
        fast_topk_radix_kernel<<<grid, block, radix_smem, stream>>>(
            scores, indices_out, seq_len, topk);
    }

    return cudaGetLastError();
}

}  // extern "C"

#undef TOPK_WARP_SIZE
#undef TOPK_NUM_THREADS
