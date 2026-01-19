/**
 * @brief Fused Rotary Position Embedding (RoPE) CUDA Kernels - With Position Selection
 * Copyright (c) 2025, Guoqingbao. All rights reserved.
 *
 * This kernel fuses TWO operations:
 *   1. Position-based cos/sin selection (eliminates index_select kernel)
 *   2. Rotary position embedding application
 *
 * Performance optimizations:
 *  - Single kernel replaces index_select + rope (2 kernels -> 1)
 *  - No shared memory (registers only)
 *  - Grid-stride loop for all tensor sizes
 *  - Native BF16/F32 compute, F16 -> F32 for precision
 *  - Vectorized float2/half2/bfloat162 access
 *
 * Licensed under the Apache License, Version 2.0
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

constexpr int BLOCK_SIZE = 128;

// ============================================================================
// Interleaved RoPE with Position Selection
// ============================================================================

/**
 * @brief F32 Interleaved kernel with position-based cos/sin selection
 * 
 * @param q Query tensor [batch, num_q_heads, seq_len, head_dim]
 * @param k Key tensor [batch, num_kv_heads, seq_len, head_dim]
 * @param cos Full cos tensor [max_seq_len, head_dim/2]
 * @param sin Full sin tensor [max_seq_len, head_dim/2]
 * @param positions Position indices [seq_len] - used to select cos/sin rows
 * @param q_num_pairs Total Q pairs = (batch * num_q_heads * seq_len * head_dim) / 2
 * @param k_num_pairs Total K pairs
 * @param seq_len Sequence length
 * @param half_d head_dim / 2
 */
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_rope_i_f32_kernel(
    float2* __restrict__ q,
    float2* __restrict__ k,
    const float* __restrict__ cos,  // [max_seq_len, half_d]
    const float* __restrict__ sin,  // [max_seq_len, half_d]
    const int64_t* __restrict__ positions,  // [seq_len]
    const uint32_t q_num_pairs,
    const uint32_t k_num_pairs,
    const uint32_t seq_len,
    const uint32_t half_d  // head_dim / 2
) {
    const uint32_t total_pairs = q_num_pairs + k_num_pairs;
    
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_pairs; 
         idx += gridDim.x * blockDim.x) {
        
        const bool is_q = (idx < q_num_pairs);
        const uint32_t local_idx = is_q ? idx : (idx - q_num_pairs);
        
        // Calculate which position and which dimension
        // local_idx = (batch * heads * seq_len * head_dim/2) flattened
        // We need to find: which seq position (t) and which dim (d)
        // const uint32_t pairs_per_seq = half_d;  // head_dim/2 pairs per sequence position
        const uint32_t t_and_d = local_idx;  // Relative to some (b,h)
        const uint32_t d_idx = t_and_d % half_d;
        const uint32_t t_idx = (t_and_d / half_d) % seq_len;
        
        // Look up position for this sequence index
        const int64_t pos = positions[t_idx];
        
        // Index into full cos/sin: [pos, d_idx]
        const uint32_t cs_idx = pos * half_d + d_idx;
        const float c = cos[cs_idx];
        const float s = sin[cs_idx];
        
        // Load, rotate, store
        float2* ptr = is_q ? q : k;
        float2 v = ptr[local_idx];
        
        float2 result;
        result.x = v.x * c - v.y * s;
        result.y = v.x * s + v.y * c;
        
        ptr[local_idx] = result;
    }
}

/**
 * @brief F16 Interleaved kernel with position selection
 */
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_rope_i_f16_kernel(
    __half2* __restrict__ q,
    __half2* __restrict__ k,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    const int64_t* __restrict__ positions,
    const uint32_t q_num_pairs,
    const uint32_t k_num_pairs,
    const uint32_t seq_len,
    const uint32_t half_d
) {
    const uint32_t total_pairs = q_num_pairs + k_num_pairs;
    
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_pairs; 
         idx += gridDim.x * blockDim.x) {
        
        const bool is_q = (idx < q_num_pairs);
        const uint32_t local_idx = is_q ? idx : (idx - q_num_pairs);
        
        const uint32_t d_idx = local_idx % half_d;
        const uint32_t t_idx = (local_idx / half_d) % seq_len;
        
        const int64_t pos = positions[t_idx];
        const uint32_t cs_idx = pos * half_d + d_idx;
        
        // F32 compute for precision
        const float c = __half2float(cos[cs_idx]);
        const float s = __half2float(sin[cs_idx]);
        
        __half2* ptr = is_q ? q : k;
        __half2 v = ptr[local_idx];
        
        float vx = __half2float(v.x);
        float vy = __half2float(v.y);
        
        __half2 result;
        result.x = __float2half(vx * c - vy * s);
        result.y = __float2half(vx * s + vy * c);
        
        ptr[local_idx] = result;
    }
}

#ifndef NO_BF16_KERNEL
/**
 * @brief BF16 Interleaved kernel with position selection
 */
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_rope_i_bf16_kernel(
    __nv_bfloat162* __restrict__ q,
    __nv_bfloat162* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const int64_t* __restrict__ positions,
    const uint32_t q_num_pairs,
    const uint32_t k_num_pairs,
    const uint32_t seq_len,
    const uint32_t half_d
) {
    const uint32_t total_pairs = q_num_pairs + k_num_pairs;
    
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_pairs; 
         idx += gridDim.x * blockDim.x) {
        
        const bool is_q = (idx < q_num_pairs);
        const uint32_t local_idx = is_q ? idx : (idx - q_num_pairs);
        
        const uint32_t d_idx = local_idx % half_d;
        const uint32_t t_idx = (local_idx / half_d) % seq_len;
        
        const int64_t pos = positions[t_idx];
        const uint32_t cs_idx = pos * half_d + d_idx;
        
        // Native BF16 compute
        const __nv_bfloat16 c = cos[cs_idx];
        const __nv_bfloat16 s = sin[cs_idx];
        
        __nv_bfloat162* ptr = is_q ? q : k;
        __nv_bfloat162 v = ptr[local_idx];
        
        __nv_bfloat162 result;
        result.x = __hsub(__hmul(v.x, c), __hmul(v.y, s));
        result.y = __hadd(__hmul(v.x, s), __hmul(v.y, c));
        
        ptr[local_idx] = result;
    }
}
#endif

// ============================================================================
// Non-Interleaved RoPE with Position Selection
// ============================================================================

/**
 * @brief F32 Non-interleaved kernel with position selection
 */
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_rope_f32_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int64_t* __restrict__ positions,
    const uint32_t q_bh,  // batch * num_q_heads
    const uint32_t k_bh,  // batch * num_kv_heads
    const uint32_t seq_len,
    const uint32_t d  // head_dim
) {
    const uint32_t half_d = d / 2;
    const uint32_t q_pairs = q_bh * seq_len * half_d;
    const uint32_t k_pairs = k_bh * seq_len * half_d;
    const uint32_t total_pairs = q_pairs + k_pairs;
    
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_pairs; 
         idx += gridDim.x * blockDim.x) {
        
        const bool is_q = (idx < q_pairs);
        const uint32_t local_idx = is_q ? idx : (idx - q_pairs);
        // const uint32_t bh = is_q ? q_bh : k_bh;
        
        // Decompose index: local_idx = i_bh * (seq_len * half_d) + i_t * half_d + i_d
        const uint32_t pairs_per_bh = seq_len * half_d;
        const uint32_t i_bh = local_idx / pairs_per_bh;
        const uint32_t remainder = local_idx % pairs_per_bh;
        const uint32_t i_t = remainder / half_d;
        const uint32_t i_d = remainder % half_d;
        
        // Get position for this sequence index
        const int64_t pos = positions[i_t];
        
        // cos/sin index: [pos, i_d]
        const uint32_t cs_idx = pos * half_d + i_d;
        const float c = cos[cs_idx];
        const float s = sin[cs_idx];
        
        // Calculate tensor indices (non-interleaved: pairs at d/2 offset)
        const uint32_t td = seq_len * d;
        const uint32_t i1 = i_bh * td + i_t * d + i_d;
        const uint32_t i2 = i1 + half_d;
        
        float* ptr = is_q ? q : k;
        float x1 = ptr[i1];
        float x2 = ptr[i2];
        
        ptr[i1] = x1 * c - x2 * s;
        ptr[i2] = x1 * s + x2 * c;
    }
}

/**
 * @brief F16 Non-interleaved kernel with position selection
 */
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_rope_f16_kernel(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    const int64_t* __restrict__ positions,
    const uint32_t q_bh,
    const uint32_t k_bh,
    const uint32_t seq_len,
    const uint32_t d
) {
    const uint32_t half_d = d / 2;
    const uint32_t q_pairs = q_bh * seq_len * half_d;
    const uint32_t k_pairs = k_bh * seq_len * half_d;
    const uint32_t total_pairs = q_pairs + k_pairs;
    
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_pairs; 
         idx += gridDim.x * blockDim.x) {
        
        const bool is_q = (idx < q_pairs);
        const uint32_t local_idx = is_q ? idx : (idx - q_pairs);
        
        const uint32_t pairs_per_bh = seq_len * half_d;
        const uint32_t i_bh = local_idx / pairs_per_bh;
        const uint32_t remainder = local_idx % pairs_per_bh;
        const uint32_t i_t = remainder / half_d;
        const uint32_t i_d = remainder % half_d;
        
        const int64_t pos = positions[i_t];
        const uint32_t cs_idx = pos * half_d + i_d;
        
        // F32 compute
        const float c = __half2float(cos[cs_idx]);
        const float s = __half2float(sin[cs_idx]);
        
        const uint32_t td = seq_len * d;
        const uint32_t i1 = i_bh * td + i_t * d + i_d;
        const uint32_t i2 = i1 + half_d;
        
        __half* ptr = is_q ? q : k;
        float x1 = __half2float(ptr[i1]);
        float x2 = __half2float(ptr[i2]);
        
        ptr[i1] = __float2half(x1 * c - x2 * s);
        ptr[i2] = __float2half(x1 * s + x2 * c);
    }
}

#ifndef NO_BF16_KERNEL
/**
 * @brief BF16 Non-interleaved kernel with position selection
 */
__global__ void __launch_bounds__(BLOCK_SIZE)
fused_rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    const int64_t* __restrict__ positions,
    const uint32_t q_bh,
    const uint32_t k_bh,
    const uint32_t seq_len,
    const uint32_t d
) {
    const uint32_t half_d = d / 2;
    const uint32_t q_pairs = q_bh * seq_len * half_d;
    const uint32_t k_pairs = k_bh * seq_len * half_d;
    const uint32_t total_pairs = q_pairs + k_pairs;
    
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_pairs; 
         idx += gridDim.x * blockDim.x) {
        
        const bool is_q = (idx < q_pairs);
        const uint32_t local_idx = is_q ? idx : (idx - q_pairs);
        
        const uint32_t pairs_per_bh = seq_len * half_d;
        const uint32_t i_bh = local_idx / pairs_per_bh;
        const uint32_t remainder = local_idx % pairs_per_bh;
        const uint32_t i_t = remainder / half_d;
        const uint32_t i_d = remainder % half_d;
        
        const int64_t pos = positions[i_t];
        const uint32_t cs_idx = pos * half_d + i_d;
        
        const __nv_bfloat16 c = cos[cs_idx];
        const __nv_bfloat16 s = sin[cs_idx];
        
        const uint32_t td = seq_len * d;
        const uint32_t i1 = i_bh * td + i_t * d + i_d;
        const uint32_t i2 = i1 + half_d;
        
        __nv_bfloat16* ptr = is_q ? q : k;
        __nv_bfloat16 x1 = ptr[i1];
        __nv_bfloat16 x2 = ptr[i2];
        
        ptr[i1] = __hsub(__hmul(x1, c), __hmul(x2, s));
        ptr[i2] = __hadd(__hmul(x1, s), __hmul(x2, c));
    }
}
#endif

// ============================================================================
// Launch helpers
// ============================================================================

inline dim3 get_optimal_grid(uint32_t num_elements) {
    const int max_blocks = 1024;
    int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    return dim3(min(num_blocks, max_blocks), 1, 1);
}

// ============================================================================
// C-linkage wrappers - NEW API with positions
// ============================================================================

extern "C" void fused_rope_f32(
    float* q, float* k,
    const float* cos, const float* sin,
    const int64_t* positions,  // NEW: position indices [seq_len]
    uint32_t q_bh, uint32_t k_bh,
    uint32_t seq_len, uint32_t d,  // Changed: seq_len instead of td
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t half_d = d / 2;
    uint32_t total = (q_bh + k_bh) * seq_len * half_d;
    dim3 grid = get_optimal_grid(total);
    fused_rope_f32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        q, k, cos, sin, positions, q_bh, k_bh, seq_len, d
    );
}

extern "C" void fused_rope_f16(
    void* q, void* k,
    const void* cos, const void* sin,
    const int64_t* positions,
    uint32_t q_bh, uint32_t k_bh,
    uint32_t seq_len, uint32_t d,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t half_d = d / 2;
    uint32_t total = (q_bh + k_bh) * seq_len * half_d;
    dim3 grid = get_optimal_grid(total);
    fused_rope_f16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        (__half*)q, (__half*)k, (const __half*)cos, (const __half*)sin,
        positions, q_bh, k_bh, seq_len, d
    );
}

extern "C" void fused_rope_bf16(
    void* q, void* k,
    const void* cos, const void* sin,
    const int64_t* positions,
    uint32_t q_bh, uint32_t k_bh,
    uint32_t seq_len, uint32_t d,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t half_d = d / 2;
    uint32_t total = (q_bh + k_bh) * seq_len * half_d;
    dim3 grid = get_optimal_grid(total);
#ifndef NO_BF16_KERNEL
    fused_rope_bf16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)q, (__nv_bfloat16*)k,
        (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin,
        positions, q_bh, k_bh, seq_len, d
    );
#endif
}

extern "C" void fused_rope_i_f32(
    float* q, float* k,
    const float* cos, const float* sin,
    const int64_t* positions,
    uint32_t q_bh, uint32_t k_bh,
    uint32_t seq_len, uint32_t d,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t half_d = d / 2;
    uint32_t q_pairs = q_bh * seq_len * half_d;
    uint32_t k_pairs = k_bh * seq_len * half_d;
    dim3 grid = get_optimal_grid(q_pairs + k_pairs);
    fused_rope_i_f32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        (float2*)q, (float2*)k, cos, sin, positions,
        q_pairs, k_pairs, seq_len, half_d
    );
}

extern "C" void fused_rope_i_f16(
    void* q, void* k,
    const void* cos, const void* sin,
    const int64_t* positions,
    uint32_t q_bh, uint32_t k_bh,
    uint32_t seq_len, uint32_t d,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t half_d = d / 2;
    uint32_t q_pairs = q_bh * seq_len * half_d;
    uint32_t k_pairs = k_bh * seq_len * half_d;
    dim3 grid = get_optimal_grid(q_pairs + k_pairs);
    fused_rope_i_f16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        (__half2*)q, (__half2*)k, (const __half*)cos, (const __half*)sin,
        positions, q_pairs, k_pairs, seq_len, half_d
    );
}

extern "C" void fused_rope_i_bf16(
    void* q, void* k,
    const void* cos, const void* sin,
    const int64_t* positions,
    uint32_t q_bh, uint32_t k_bh,
    uint32_t seq_len, uint32_t d,
    int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    uint32_t half_d = d / 2;
    uint32_t q_pairs = q_bh * seq_len * half_d;
    uint32_t k_pairs = k_bh * seq_len * half_d;
#ifndef NO_BF16_KERNEL
    dim3 grid = get_optimal_grid(q_pairs + k_pairs);
    fused_rope_i_bf16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat162*)q, (__nv_bfloat162*)k,
        (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin,
        positions, q_pairs, k_pairs, seq_len, half_d
    );
#endif
}
