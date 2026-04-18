// #undef __CUDA_FP8_TYPES_EXIST__
#ifndef NO_HARDWARE_FP8
  #include "cuda_fp8.h"
#endif
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// High-performance expert offset computation — thrust-free, CUDA-graph safe.
//
// Two paths:
//   * Small path  (size_m <= 128): fused count + scan in a single warp launch.
//   * Large path  (size_m > 128):  multi-block warp-reduction count +
//                                  single-block Blelloch exclusive scan.
// ---------------------------------------------------------------------------

// ---- helpers ---------------------------------------------------------------
static __device__ __forceinline__ int next_pow2_dev(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

static __host__ __forceinline__ int next_pow2_host(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// ---- Large path: optimised count kernel with warp-level reduction ----------
static __global__ void count_tokens_per_expert_warp(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ expert_counts,
    int size_m,
    int num_experts)
{
    extern __shared__ int32_t s_counts[];

    const int tid = threadIdx.x;

    for (int e = tid; e < num_experts; e += blockDim.x)
        s_counts[e] = 0;
    __syncthreads();

    const int gid = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < size_m; i += stride) {
        int32_t eid = expert_ids[i];
        if (eid >= 0 && eid < num_experts)
            atomicAdd(&s_counts[eid], 1);
    }
    __syncthreads();

    for (int e = tid; e < num_experts; e += blockDim.x)
        if (s_counts[e] != 0)
            atomicAdd(&expert_counts[e], s_counts[e]);
}

// Blelloch exclusive scan (work-efficient, single block, any num_experts<=1024)
static __global__ void blelloch_exclusive_scan_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int num_experts)
{
    extern __shared__ int32_t temp[];
    const int tid = threadIdx.x;
    const int n = blockDim.x;  // next power of 2 >= num_experts

    temp[tid] = (tid < num_experts) ? counts[tid] : 0;
    __syncthreads();

    // Up-sweep (reduce)
    for (int d = 1; d < n; d <<= 1) {
        int idx = (tid + 1) * (d << 1) - 1;
        if (idx < n)
            temp[idx] += temp[idx - d];
        __syncthreads();
    }

    if (tid == 0) temp[n - 1] = 0;
    __syncthreads();

    // Down-sweep
    for (int d = n >> 1; d >= 1; d >>= 1) {
        int idx = (tid + 1) * (d << 1) - 1;
        if (idx < n) {
            int32_t t = temp[idx - d];
            temp[idx - d] = temp[idx];
            temp[idx] += t;
        }
        __syncthreads();
    }

    if (tid < num_experts)
        offsets[tid] = temp[tid];
    if (tid == 0)
        offsets[num_experts] = temp[num_experts - 1] + counts[num_experts - 1];
}

// ---- Small path: fused count + scan in a single block ----------------------
static __global__ void fused_count_and_scan_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    int size_m,
    int num_experts)
{
    extern __shared__ int32_t smem[];

    const int tid = threadIdx.x;

    for (int e = tid; e < num_experts; e += blockDim.x)
        smem[e] = 0;
    __syncthreads();

    for (int i = tid; i < size_m; i += blockDim.x) {
        int32_t eid = expert_ids[i];
        if (eid >= 0 && eid < num_experts)
            atomicAdd(&smem[eid], 1);
    }
    __syncthreads();

    if (tid < num_experts)
        expert_counts[tid] = smem[tid];
    __syncthreads();

    // Warp-style exclusive scan in shared memory (Blelloch in-place)
    const int n = blockDim.x;  // power of 2 >= num_experts
    int val = (tid < num_experts) ? smem[tid] : 0;
    smem[tid] = val;
    __syncthreads();

    for (int d = 1; d < n; d <<= 1) {
        int idx = (tid + 1) * (d << 1) - 1;
        if (idx < n)
            smem[idx] += smem[idx - d];
        __syncthreads();
    }

    if (tid == 0) smem[n - 1] = 0;
    __syncthreads();

    for (int d = n >> 1; d >= 1; d >>= 1) {
        int idx = (tid + 1) * (d << 1) - 1;
        if (idx < n) {
            int32_t t = smem[idx - d];
            smem[idx - d] = smem[idx];
            smem[idx] += t;
        }
        __syncthreads();
    }

    if (tid < num_experts)
        expert_offsets[tid] = smem[tid];
    if (tid == 0)
        expert_offsets[num_experts] = smem[num_experts - 1] + expert_counts[num_experts - 1];
}

// ---- Public API (same interface as the old functions) ----------------------

static void g_calculate_expert_offsets(
    const int32_t* d_expert_ids,
    int size_m,
    int32_t* d_expert_counts,
    int32_t* d_expert_offsets,
    int num_experts,
    cudaStream_t stream)
{
    const int scan_threads = next_pow2_host(num_experts < 32 ? 32 : num_experts);

    if (size_m <= 128) {
        // Fused path: single kernel launch
        size_t smem = scan_threads * sizeof(int32_t);
        fused_count_and_scan_kernel<<<1, scan_threads, smem, stream>>>(
            d_expert_ids, d_expert_counts, d_expert_offsets, size_m, num_experts);
    } else {
        // Two-kernel path: optimised count then Blelloch scan
        cudaMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int32_t), stream);

        const int count_threads = 256;
        const int max_blocks = 128;
        int count_blocks = (size_m + count_threads - 1) / count_threads;
        if (count_blocks > max_blocks) count_blocks = max_blocks;
        size_t smem_count = num_experts * sizeof(int32_t);
        count_tokens_per_expert_warp<<<count_blocks, count_threads, smem_count, stream>>>(
            d_expert_ids, d_expert_counts, size_m, num_experts);

        size_t smem_scan = scan_threads * sizeof(int32_t);
        blelloch_exclusive_scan_kernel<<<1, scan_threads, smem_scan, stream>>>(
            d_expert_counts, d_expert_offsets, num_experts);
    }
}
