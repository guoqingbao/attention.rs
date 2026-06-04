#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#if defined(USE_FLASHINFER) && defined(FLAT_SM90A_ENABLED)
#include "cutlass/arch/arch.h"
#include "flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh"
#endif

template <typename T>
__global__ void gdn_gather_state_kv_to_vk_kernel(
    const float* __restrict__ state_kv,
    const int64_t* __restrict__ slots,
    float* __restrict__ state_vk,
    int batch,
    int num_heads,
    int k_dim,
    int v_dim) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(batch) * num_heads * v_dim * k_dim;
    if (idx >= total) return;

    const int k_idx = idx % k_dim;
    const int v_idx = (idx / k_dim) % v_dim;
    const int head = (idx / (static_cast<int64_t>(k_dim) * v_dim)) % num_heads;
    const int b = idx / (static_cast<int64_t>(num_heads) * v_dim * k_dim);
    const int64_t slot = slots[b];

    if (slot < 0) {
        state_vk[idx] = 0.0f;
        return;
    }

    state_vk[idx] = state_kv[((slot * num_heads + head) * k_dim + k_idx) * v_dim + v_idx];
}

__global__ void gdn_scatter_state_vk_to_kv_kernel(
    const float* __restrict__ state_vk,
    const int64_t* __restrict__ slots,
    float* __restrict__ state_kv,
    int batch,
    int num_heads,
    int k_dim,
    int v_dim) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(batch) * num_heads * v_dim * k_dim;
    if (idx >= total) return;

    const int k_idx = idx % k_dim;
    const int v_idx = (idx / k_dim) % v_dim;
    const int head = (idx / (static_cast<int64_t>(k_dim) * v_dim)) % num_heads;
    const int b = idx / (static_cast<int64_t>(num_heads) * v_dim * k_dim);
    const int64_t slot = slots[b];
    if (slot < 0) return;

    state_kv[((slot * num_heads + head) * k_dim + k_idx) * v_dim + v_idx] = state_vk[idx];
}

__global__ void gdn_cu_seqlens_u32_to_i64_kernel(
    const uint32_t* __restrict__ cu_u32,
    int64_t* __restrict__ cu_i64,
    int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        cu_i64[idx] = static_cast<int64_t>(cu_u32[idx]);
    }
}

template <typename T>
static int launch_flashinfer_gdn_prefill_gva(
    const T* q,
    const T* k,
    const T* v,
    const float* alpha,
    const float* beta,
    float* state,
    const int64_t* slots,
    T* out,
    const uint32_t* cu_seqlens,
    int total_tokens,
    int batch,
    int num_v_heads,
    int num_k_heads,
    int k_dim,
    int v_dim,
    float q_scale,
    int64_t stream_i64) {
#if defined(USE_FLASHINFER) && defined(FLAT_SM90A_ENABLED)
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_i64);
    if (k_dim != v_dim || num_v_heads <= num_k_heads || batch <= 0) {
        return 1;
    }

    int dev_id = 0;
    int major = 0;
    int sm_count = 0;
    if (cudaGetDevice(&dev_id) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id) != cudaSuccess ||
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id) != cudaSuccess ||
        major != 9) {
        return 1;
    }

    const size_t state_elems = static_cast<size_t>(batch) * num_v_heads * v_dim * k_dim;
    const size_t state_bytes = state_elems * sizeof(float);
    const size_t cu_bytes = static_cast<size_t>(batch + 1) * sizeof(int64_t);
    const size_t workspace_bytes = static_cast<size_t>(sm_count) * 128;

    float* input_state = nullptr;
    float* output_state = nullptr;
    int64_t* cu_i64 = nullptr;
    uint8_t* workspace = nullptr;

    if (cudaMallocAsync(reinterpret_cast<void**>(&input_state), state_bytes, stream) != cudaSuccess ||
        cudaMallocAsync(reinterpret_cast<void**>(&output_state), state_bytes, stream) != cudaSuccess ||
        cudaMallocAsync(reinterpret_cast<void**>(&cu_i64), cu_bytes, stream) != cudaSuccess ||
        cudaMallocAsync(reinterpret_cast<void**>(&workspace), workspace_bytes, stream) != cudaSuccess) {
        if (input_state) cudaFreeAsync(input_state, stream);
        if (output_state) cudaFreeAsync(output_state, stream);
        if (cu_i64) cudaFreeAsync(cu_i64, stream);
        if (workspace) cudaFreeAsync(workspace, stream);
        return 2;
    }

    const int threads = 256;
    const int64_t state_total = static_cast<int64_t>(state_elems);
    const int state_blocks = static_cast<int>((state_total + threads - 1) / threads);
    const int cu_blocks = (batch + 1 + threads - 1) / threads;

    gdn_gather_state_kv_to_vk_kernel<T><<<state_blocks, threads, 0, stream>>>(
        state, slots, input_state, batch, num_v_heads, k_dim, v_dim);
    gdn_cu_seqlens_u32_to_i64_kernel<<<cu_blocks, threads, 0, stream>>>(
        cu_seqlens, cu_i64, batch + 1);

    int status = 0;
    try {
        flat::launch_delta_rule_prefill_kernel_gbai<
            true, true, true, true, cutlass::arch::Sm90, T, T, float>(
            stream,
            out,
            output_state,
            q,
            k,
            v,
            input_state,
            alpha,
            beta,
            cu_i64,
            workspace,
            batch,
            num_k_heads,
            num_k_heads,
            num_v_heads,
            num_v_heads,
            k_dim,
            static_cast<int64_t>(total_tokens),
            q_scale,
            sm_count);
    } catch (...) {
        status = 3;
    }

    if (status == 0) {
        gdn_scatter_state_vk_to_kv_kernel<<<state_blocks, threads, 0, stream>>>(
            output_state, slots, state, batch, num_v_heads, k_dim, v_dim);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = 4;
        }
    }

    cudaFreeAsync(input_state, stream);
    cudaFreeAsync(output_state, stream);
    cudaFreeAsync(cu_i64, stream);
    cudaFreeAsync(workspace, stream);
    return status;
#else
    (void)q;
    (void)k;
    (void)v;
    (void)alpha;
    (void)beta;
    (void)state;
    (void)slots;
    (void)out;
    (void)cu_seqlens;
    (void)total_tokens;
    (void)batch;
    (void)num_v_heads;
    (void)num_k_heads;
    (void)k_dim;
    (void)v_dim;
    (void)q_scale;
    (void)stream_i64;
    return 1;
#endif
}

extern "C" int gated_delta_rule_prefill_persistent_varlen_gqa_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    float* state,
    const int64_t* slots,
    void* out,
    const uint32_t* cu_seqlens,
    int total_tokens,
    int batch,
    int num_v_heads,
    int num_k_heads,
    int k_dim,
    int v_dim,
    float q_scale,
    int64_t stream) {
    return launch_flashinfer_gdn_prefill_gva<__nv_bfloat16>(
        static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v),
        static_cast<const float*>(g),
        static_cast<const float*>(beta),
        state,
        slots,
        static_cast<__nv_bfloat16*>(out),
        cu_seqlens,
        total_tokens,
        batch,
        num_v_heads,
        num_k_heads,
        k_dim,
        v_dim,
        q_scale,
        stream);
}

extern "C" int gated_delta_rule_prefill_persistent_varlen_gqa_f16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    float* state,
    const int64_t* slots,
    void* out,
    const uint32_t* cu_seqlens,
    int total_tokens,
    int batch,
    int num_v_heads,
    int num_k_heads,
    int k_dim,
    int v_dim,
    float q_scale,
    int64_t stream) {
    return launch_flashinfer_gdn_prefill_gva<half>(
        static_cast<const half*>(q),
        static_cast<const half*>(k),
        static_cast<const half*>(v),
        static_cast<const float*>(g),
        static_cast<const float*>(beta),
        state,
        slots,
        static_cast<half*>(out),
        cu_seqlens,
        total_tokens,
        batch,
        num_v_heads,
        num_k_heads,
        k_dim,
        v_dim,
        q_scale,
        stream);
}
