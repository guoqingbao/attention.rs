#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define CHECK_CUDA(x)                                                           \
    do {                                                                        \
        cudaError_t err__ = (x);                                                \
        if (err__ != cudaSuccess) {                                             \
            printf("CUDA Error at %s:%d: %s\\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(err__));                                  \
        }                                                                       \
    } while (0)

static constexpr int GDN_MAX_KERNEL_SIZE = 16;

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float silu_float(float x) {
    return x / (1.0f + __expf(-x));
}

static constexpr int GDN_WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = GDN_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

// =============================================================================
// Gated Delta Rule Recurrence (DeltaNet core)
// =============================================================================

template <typename T, int BK, int BV>
__global__ void gated_delta_rule_recurrence_kernel_tiled(
    const T* __restrict__ q,          // [BH, S, K]
    const T* __restrict__ k,          // [BH, S, K]
    const T* __restrict__ v,          // [BH, S, V]
    const float* __restrict__ g,      // [BH, S] decay = exp(g)
    const float* __restrict__ beta,   // [BH, S]
    float* __restrict__ state,        // [BH, K, V] (in/out)
    float* __restrict__ out,          // [BH, S, V]
    int seq_len,
    int v_dim) {
    const int v_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int v_idx = v_tile * BV + tid;

    const bool v_valid = v_idx < v_dim;

    const T* q_bh = q + bh * seq_len * BK;
    const T* k_bh = k + bh * seq_len * BK;
    const T* v_bh = v + bh * seq_len * v_dim;
    const float* g_bh = g + bh * seq_len;
    const float* beta_bh = beta + bh * seq_len;
    float* state_base = state + bh * BK * v_dim;
    float* out_bh = out + bh * seq_len * v_dim;

    __shared__ float k_buf[BK];
    __shared__ float q_buf[BK];
    __shared__ float scalars[2]; // decay, beta_t

    // State layout: [K, V] — each thread loads its v_idx element from each key row
    float s[BK];
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        s[j] = v_valid ? state_base[j * v_dim + v_idx] : 0.0f;
    }

    for (int t = 0; t < seq_len; ++t) {
        // Cooperative load k into shared memory
#pragma unroll
        for (int j = tid; j < BK; j += BV) {
            k_buf[j] = to_float(k_bh[t * BK + j]);
        }
        if (tid == 0) {
            scalars[0] = g_bh[t];
            scalars[1] = beta_bh[t];
        }
        __syncthreads();

        const float decay = scalars[0];
        const float beta_t = scalars[1];
        const float v_t = v_valid ? to_float(v_bh[t * v_dim + v_idx]) : 0.0f;

        float kv_mem = 0.0f;
#pragma unroll
        for (int j = 0; j < BK; ++j) {
            s[j] *= decay;
            kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
        }

        const float delta = (v_t - kv_mem) * beta_t;

        // Load q into shared memory (reuse sync from above — k_buf consumed)
        __syncthreads();
#pragma unroll
        for (int j = tid; j < BK; j += BV) {
            q_buf[j] = to_float(q_bh[t * BK + j]);
        }
        __syncthreads();

        float y_t = 0.0f;
#pragma unroll
        for (int j = 0; j < BK; ++j) {
            s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
            y_t = __fmaf_rn(s[j], q_buf[j], y_t);
        }

        if (v_valid) {
            out_bh[t * v_dim + v_idx] = y_t;
        }
    }

#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (v_valid) {
            state_base[j * v_dim + v_idx] = s[j];
        }
    }
}

template <typename T, int BV, int MAX_K>
__global__ void gated_delta_rule_recurrence_kernel_fallback(
    const T* __restrict__ q,          // [BH, S, K]
    const T* __restrict__ k,          // [BH, S, K]
    const T* __restrict__ v,          // [BH, S, V]
    const float* __restrict__ g,      // [BH, S] decay = exp(g)
    const float* __restrict__ beta,   // [BH, S]
    float* __restrict__ state,        // [BH, K, V] (in/out)
    float* __restrict__ out,          // [BH, S, V]
    int seq_len,
    int k_dim,
    int v_dim) {
    const int v_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int v_idx = v_tile * BV + tid;

    const bool v_valid = v_idx < v_dim;

    const T* q_bh = q + bh * seq_len * k_dim;
    const T* k_bh = k + bh * seq_len * k_dim;
    const T* v_bh = v + bh * seq_len * v_dim;
    const float* g_bh = g + bh * seq_len;
    const float* beta_bh = beta + bh * seq_len;
    float* state_base = state + bh * k_dim * v_dim;
    float* out_bh = out + bh * seq_len * v_dim;

    extern __shared__ float shared[];
    float* k_buf = shared;
    float* q_buf = shared + k_dim;
    float* scalars_buf = shared + 2 * k_dim;

    float s[MAX_K];
    for (int j = 0; j < k_dim; ++j) {
        s[j] = v_valid ? state_base[j * v_dim + v_idx] : 0.0f;
    }

    for (int t = 0; t < seq_len; ++t) {
        for (int j = tid; j < k_dim; j += BV) {
            k_buf[j] = to_float(k_bh[t * k_dim + j]);
        }
        if (tid == 0) {
            scalars_buf[0] = g_bh[t];
            scalars_buf[1] = beta_bh[t];
        }
        __syncthreads();

        const float decay = scalars_buf[0];
        const float beta_t = scalars_buf[1];
        const float v_t = v_valid ? to_float(v_bh[t * v_dim + v_idx]) : 0.0f;

        float kv_mem = 0.0f;
        for (int j = 0; j < k_dim; ++j) {
            s[j] *= decay;
            kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
        }

        const float delta = (v_t - kv_mem) * beta_t;

        __syncthreads();
        for (int j = tid; j < k_dim; j += BV) {
            q_buf[j] = to_float(q_bh[t * k_dim + j]);
        }
        __syncthreads();

        float y_t = 0.0f;
        for (int j = 0; j < k_dim; ++j) {
            s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
            y_t = __fmaf_rn(s[j], q_buf[j], y_t);
        }

        if (v_valid) {
            out_bh[t * v_dim + v_idx] = y_t;
        }
    }

    for (int j = 0; j < k_dim; ++j) {
        if (v_valid) {
            state_base[j * v_dim + v_idx] = s[j];
        }
    }
}

template <typename T>
void launch_gated_delta_rule_recurrence(
    const T* q,
    const T* k,
    const T* v,
    const float* g,
    const float* beta,
    float* state,
    float* out,
    int bh,
    int seq_len,
    int k_dim,
    int v_dim,
    cudaStream_t stream) {
    if (bh <= 0 || seq_len <= 0 || k_dim <= 0 || v_dim <= 0) {
        return;
    }

    if (k_dim == 128) {
        constexpr int BK = 128;
        constexpr int BV = 64;
        dim3 grid((v_dim + BV - 1) / BV, bh);
        dim3 block(BV);
        gated_delta_rule_recurrence_kernel_tiled<T, BK, BV><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, state, out, seq_len, v_dim);
    } else if (k_dim == 64) {
        constexpr int BK = 64;
        constexpr int BV = 64;
        dim3 grid((v_dim + BV - 1) / BV, bh);
        dim3 block(BV);
        gated_delta_rule_recurrence_kernel_tiled<T, BK, BV><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, state, out, seq_len, v_dim);
    } else {
        constexpr int BV = 64;
        constexpr int MAX_K = 256;
        if (k_dim > MAX_K) {
            printf("gated_delta_rule_recurrence: k_dim=%d exceeds MAX_K=%d\\n", k_dim, MAX_K);
            return;
        }
        dim3 grid((v_dim + BV - 1) / BV, bh);
        dim3 block(BV);
        size_t smem = (2 * static_cast<size_t>(k_dim) + 2) * sizeof(float);
        gated_delta_rule_recurrence_kernel_fallback<T, BV, MAX_K><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, out, seq_len, k_dim, v_dim);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void gated_delta_rule_recurrence(
    const float* q,
    const float* k,
    const float* v,
    const float* g,
    const float* beta,
    float* state,
    float* out,
    int bh,
    int seq_len,
    int k_dim,
    int v_dim,
    cudaStream_t stream) {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, state, out, bh, seq_len, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_recurrence_f16(
    const half* q,
    const half* k,
    const half* v,
    const float* g,
    const float* beta,
    float* state,
    float* out,
    int bh,
    int seq_len,
    int k_dim,
    int v_dim,
    cudaStream_t stream) {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, state, out, bh, seq_len, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_recurrence_bf16(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* g,
    const float* beta,
    float* state,
    float* out,
    int bh,
    int seq_len,
    int k_dim,
    int v_dim,
    cudaStream_t stream) {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, state, out, bh, seq_len, k_dim, v_dim, stream);
}

// =============================================================================
// Optimized decode kernel: K3 (scalar broadcast), K4 (exact BK template),
// K5 (cooperative state load), K6 (shared q/k)
// Dynamic shared memory layout: [q_smem: BK] [k_smem: BK] [scalars: 2]
// =============================================================================

template <typename T, int BV, int BK>
__global__ void gated_delta_rule_decode_slots_kernel(
    const T* __restrict__ q,      // [batch, heads, k_dim]
    const T* __restrict__ k,      // [batch, heads, k_dim]
    const T* __restrict__ v,      // [batch, heads, v_dim]
    const float* __restrict__ g,  // [batch, heads] decay = exp(g)
    const float* __restrict__ beta, // [batch, heads]
    T* __restrict__ state,        // [max_batch, heads, k_dim, v_dim]
    const int64_t* __restrict__ slots, // [batch]
    T* __restrict__ out,          // [batch, heads, v_dim]
    int batch,
    int heads,
    int k_dim,
    int v_dim) {
    const int v_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int v_idx = v_tile * BV + tid;
    if (bh >= batch * heads) return;
    const bool v_valid = v_idx < v_dim;

    const int b = bh / heads;
    const int h = bh % heads;
    const int64_t slot = slots[b];
    if (slot < 0) return;

    extern __shared__ float smem[];
    float* q_smem = smem;
    float* k_smem = smem + BK;
    float* scalars = smem + 2 * BK;

    if (tid == 0) {
        scalars[0] = to_float(g[bh]);
        scalars[1] = to_float(beta[bh]);
    }

    // K6: cooperative load of q/k into shared memory
    const T* q_bh = q + bh * k_dim;
    const T* k_bh = k + bh * k_dim;
    for (int j = tid; j < k_dim; j += BV) {
        k_smem[j] = to_float(k_bh[j]);
    }
    __syncthreads();
    const float decay = scalars[0];
    const float beta_t = scalars[1];

    T* state_head = state + ((slot * heads + h) * k_dim * v_dim);

    float s_buf[BK];
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        s_buf[j] = (v_valid && j < k_dim) ? to_float(state_head[j * v_dim + v_idx]) : 0.0f;
    }

    float kv_mem = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (v_valid && j < k_dim) {
            s_buf[j] *= decay;
            kv_mem = __fmaf_rn(s_buf[j], k_smem[j], kv_mem);
        }
    }

    const T* v_bh = v + (bh * v_dim);
    const float delta = (v_valid ? to_float(v_bh[v_idx]) : 0.0f) - kv_mem;
    const float scaled_delta = delta * beta_t;

    __syncthreads();
    for (int j = tid; j < k_dim; j += BV) {
        q_smem[j] = to_float(q_bh[j]);
    }
    __syncthreads();

    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (v_valid && j < k_dim) {
            s_buf[j] = __fmaf_rn(k_smem[j], scaled_delta, s_buf[j]);
            y = __fmaf_rn(s_buf[j], q_smem[j], y);
        }
    }

#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (v_valid && j < k_dim) {
            state_head[j * v_dim + v_idx] = from_float<T>(s_buf[j]);
        }
    }

    if (v_valid) {
        out[bh * v_dim + v_idx] = from_float<T>(y);
    }
}

template <typename T, int BV, int BK>
__global__ void gated_delta_rule_decode_slots_kernel_state_f32(
    const T* __restrict__ q,      // [batch, heads, k_dim]
    const T* __restrict__ k,      // [batch, heads, k_dim]
    const T* __restrict__ v,      // [batch, heads, v_dim]
    const float* __restrict__ g,  // [batch, heads] decay = exp(g)
    const float* __restrict__ beta, // [batch, heads]
    float* __restrict__ state,    // [max_batch, heads, k_dim, v_dim]
    const int64_t* __restrict__ slots, // [batch]
    T* __restrict__ out,          // [batch, heads, v_dim]
    int batch,
    int heads,
    int k_dim,
    int v_dim) {
    const int v_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int v_idx = v_tile * BV + tid;
    if (bh >= batch * heads) return;
    const bool v_valid = v_idx < v_dim;

    const int b = bh / heads;
    const int h = bh % heads;
    const int64_t slot = slots[b];
    if (slot < 0) return;

    extern __shared__ float smem[];
    float* q_smem = smem;
    float* k_smem = smem + BK;
    float* scalars = smem + 2 * BK;

    if (tid == 0) {
        scalars[0] = to_float(g[bh]);
        scalars[1] = to_float(beta[bh]);
    }

    const T* q_bh = q + bh * k_dim;
    const T* k_bh = k + bh * k_dim;
    for (int j = tid; j < k_dim; j += BV) {
        k_smem[j] = to_float(k_bh[j]);
    }
    __syncthreads();
    const float decay = scalars[0];
    const float beta_t = scalars[1];

    float* state_head = state + ((slot * heads + h) * k_dim * v_dim);

    float s_buf[BK];
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        s_buf[j] = (v_valid && j < k_dim) ? state_head[j * v_dim + v_idx] : 0.0f;
    }

    float kv_mem = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (v_valid && j < k_dim) {
            s_buf[j] *= decay;
            kv_mem = __fmaf_rn(s_buf[j], k_smem[j], kv_mem);
        }
    }

    const T* v_bh = v + (bh * v_dim);
    const float delta = ((v_valid ? to_float(v_bh[v_idx]) : 0.0f) - kv_mem) * beta_t;

    __syncthreads();
    for (int j = tid; j < k_dim; j += BV) {
        q_smem[j] = to_float(q_bh[j]);
    }
    __syncthreads();

    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (v_valid && j < k_dim) {
            s_buf[j] = __fmaf_rn(k_smem[j], delta, s_buf[j]);
            y = __fmaf_rn(s_buf[j], q_smem[j], y);
        }
    }

#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (v_valid && j < k_dim) {
            state_head[j * v_dim + v_idx] = s_buf[j];
        }
    }

    if (v_valid) {
        out[bh * v_dim + v_idx] = from_float<T>(y);
    }
}

// K4: dispatch to exact BK sizes to minimize register pressure
template <typename T>
void launch_gated_delta_rule_decode_slots(
    const T* q, const T* k, const T* v, const float* g, const float* beta,
    T* state, const int64_t* slots, T* out,
    int batch, int heads, int k_dim, int v_dim,
    cudaStream_t stream) {
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, batch * heads);
    dim3 block(BV);
    // smem: q[BK] + k[BK] + scalars[2]
    if (k_dim <= 64) {
        constexpr int BK = 64;
        size_t smem = (2 * BK + 2) * sizeof(float);
        gated_delta_rule_decode_slots_kernel<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim);
    } else if (k_dim <= 128) {
        constexpr int BK = 128;
        size_t smem = (2 * BK + 2) * sizeof(float);
        gated_delta_rule_decode_slots_kernel<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim);
    } else {
        constexpr int BK = 256;
        if (k_dim > BK) {
            printf("gated_delta_rule_decode_slots: k_dim=%d exceeds MAX_K=%d\n", k_dim, BK);
            return;
        }
        size_t smem = (2 * BK + 2) * sizeof(float);
        gated_delta_rule_decode_slots_kernel<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim);
    }
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void launch_gated_delta_rule_decode_slots_state_f32(
    const T* q, const T* k, const T* v, const float* g, const float* beta,
    float* state, const int64_t* slots, T* out,
    int batch, int heads, int k_dim, int v_dim,
    cudaStream_t stream) {
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, batch * heads);
    dim3 block(BV);
    if (k_dim <= 64) {
        constexpr int BK = 64;
        size_t smem = (2 * BK + 2) * sizeof(float);
        gated_delta_rule_decode_slots_kernel_state_f32<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim);
    } else if (k_dim <= 128) {
        constexpr int BK = 128;
        size_t smem = (2 * BK + 2) * sizeof(float);
        gated_delta_rule_decode_slots_kernel_state_f32<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim);
    } else {
        constexpr int BK = 256;
        if (k_dim > BK) {
            printf("gated_delta_rule_decode_slots_state_f32: k_dim=%d exceeds MAX_K=%d\n", k_dim, BK);
            return;
        }
        size_t smem = (2 * BK + 2) * sizeof(float);
        gated_delta_rule_decode_slots_kernel_state_f32<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void gated_delta_rule_decode_slots_f32(
    const float* q, const float* k, const float* v, const float* g, const float* beta,
    float* state, const int64_t* slots, float* out, int batch, int heads, int k_dim,
    int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_decode_slots_f16(
    const half* q, const half* k, const half* v, const float* g, const float* beta,
    half* state, const int64_t* slots, half* out, int batch, int heads, int k_dim,
    int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_decode_slots_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const float* beta, __nv_bfloat16* state,
    const int64_t* slots, __nv_bfloat16* out, int batch, int heads, int k_dim,
    int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_decode_slots_f16_state_f32(
    const half* q, const half* k, const half* v, const float* g, const float* beta,
    float* state, const int64_t* slots, half* out, int batch, int heads, int k_dim,
    int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots_state_f32(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_decode_slots_bf16_state_f32(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const float* beta, float* state,
    const int64_t* slots, __nv_bfloat16* out, int batch, int heads, int k_dim,
    int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots_state_f32(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim, stream);
}

// =============================================================================
// GQA Decode: q/k have num_k_heads, v/g/beta/state/out have num_v_heads.
// Fuses exp(g), q_scale. State is FP32.
// =============================================================================

template <typename T, int BV, int BK>
__global__ void gated_delta_rule_decode_slots_gqa_kernel(
    const T* __restrict__ q,      // [batch, num_k_heads, k_dim]
    const T* __restrict__ k,      // [batch, num_k_heads, k_dim]
    const T* __restrict__ v,      // [batch, num_v_heads, v_dim]
    const float* __restrict__ g,  // [batch, num_v_heads]  (log-space, NOT exp'd)
    const float* __restrict__ beta, // [batch, num_v_heads]
    float* __restrict__ state,    // [max_batch, num_v_heads, k_dim, v_dim]
    const int64_t* __restrict__ slots, // [batch]
    T* __restrict__ out,          // [batch, num_v_heads, v_dim]
    int batch,
    int num_v_heads,
    int num_k_heads,
    int k_dim,
    int v_dim,
    float q_scale) {
    const int v_tile = blockIdx.x;
    const int bh = blockIdx.y;    // batch * num_v_heads
    const int tid = threadIdx.x;
    const int v_idx = v_tile * BV + tid;
    if (bh >= batch * num_v_heads) return;
    const bool v_valid = v_idx < v_dim;

    const int b = bh / num_v_heads;
    const int v_head_idx = bh % num_v_heads;
    const int kv_group_size = num_v_heads / num_k_heads;
    const int k_head_idx = v_head_idx / kv_group_size;
    const int64_t slot = slots[b];
    const bool slot_valid = slot >= 0;

    extern __shared__ float smem[];
    float* q_smem = smem;
    float* k_smem = smem + BK;
    float* scalars = smem + 2 * BK;

    if (tid == 0) {
        scalars[0] = expf(to_float(g[b * num_v_heads + v_head_idx]));
        scalars[1] = to_float(beta[b * num_v_heads + v_head_idx]);
    }

    const T* q_bh = q + (b * num_k_heads + k_head_idx) * k_dim;
    const T* k_bh = k + (b * num_k_heads + k_head_idx) * k_dim;
    for (int j = tid; j < k_dim; j += BV) {
        k_smem[j] = to_float(k_bh[j]);
    }
    __syncthreads();
    const float decay = scalars[0];
    const float beta_t = scalars[1];

    float* state_head = slot_valid
        ? state + ((slot * num_v_heads + v_head_idx) * k_dim * v_dim)
        : nullptr;

    float s_buf[BK];
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        s_buf[j] = (v_valid && slot_valid && j < k_dim)
            ? state_head[j * v_dim + v_idx]
            : 0.0f;
    }

    float kv_mem = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (v_valid && slot_valid && j < k_dim) {
            s_buf[j] *= decay;
            kv_mem = __fmaf_rn(s_buf[j], k_smem[j], kv_mem);
        }
    }

    const T* v_bh = v + (b * num_v_heads + v_head_idx) * v_dim;
    const float delta = v_valid && slot_valid
        ? (to_float(v_bh[v_idx]) - kv_mem) * beta_t
        : 0.0f;

    __syncthreads();
    for (int j = tid; j < k_dim; j += BV) {
        q_smem[j] = to_float(q_bh[j]) * q_scale;
    }
    __syncthreads();

    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (v_valid && slot_valid && j < k_dim) {
            s_buf[j] = __fmaf_rn(k_smem[j], delta, s_buf[j]);
            y = __fmaf_rn(s_buf[j], q_smem[j], y);
        }
    }

#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (v_valid && slot_valid && j < k_dim) {
            state_head[j * v_dim + v_idx] = s_buf[j];
        }
    }

    if (v_valid && slot_valid) {
        out[(b * num_v_heads + v_head_idx) * v_dim + v_idx] = from_float<T>(y);
    }
}

// Decode specialization for the Qwen3.5/Qwen3.8 GDN geometry.  The generic
// kernel assigns one output column to one thread, which leaves a Hopper SM
// with too little instruction-level parallelism: every thread executes both
// 128-element dot products serially.  Here a warp computes four columns and
// splits the recurrent dimension over eight lanes per column.  The recurrent
// state and all arithmetic remain FP32; only the work assignment and the
// FP32 reduction order change.
template <typename T>
__global__ void gated_delta_rule_decode_slots_gqa_128_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ state,
    const int64_t* __restrict__ slots,
    T* __restrict__ out,
    int batch,
    int num_v_heads,
    int num_k_heads,
    float q_scale,
    int64_t v_offset,
    int64_t v_row_stride) {
    constexpr int K_DIM = 128;
    constexpr int V_DIM = 128;
    constexpr int COLS_PER_WARP = 4;
    constexpr int LANES_PER_COL = GDN_WARP_SIZE / COLS_PER_WARP;
    constexpr int ROWS_PER_LANE = K_DIM / LANES_PER_COL;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int col_in_warp = lane % COLS_PER_WARP;
    const int lane_in_col = lane / COLS_PER_WARP;
    const int v_idx = (blockIdx.x * blockDim.y + warp) * COLS_PER_WARP + col_in_warp;
    const int bh = blockIdx.y;
    if (bh >= batch * num_v_heads || v_idx >= V_DIM) return;

    const int b = bh / num_v_heads;
    const int v_head_idx = bh % num_v_heads;
    const int kv_group_size = num_v_heads / num_k_heads;
    const int k_head_idx = v_head_idx / kv_group_size;
    const int64_t slot = slots[b];
    if (slot < 0) return;

    const T* q_head = q + (b * num_k_heads + k_head_idx) * K_DIM;
    const T* k_head = k + (b * num_k_heads + k_head_idx) * K_DIM;
    const T* v_head = v + v_offset + static_cast<int64_t>(b) * v_row_stride +
                      v_head_idx * V_DIM;
    float* state_head = state + ((slot * num_v_heads + v_head_idx) * K_DIM * V_DIM);
    const float decay = expf(g[b * num_v_heads + v_head_idx]);
    const float beta_t = beta[b * num_v_heads + v_head_idx];

    float s[ROWS_PER_LANE];
    float kv_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
        const int k_idx = r * LANES_PER_COL + lane_in_col;
        s[r] = state_head[k_idx * V_DIM + v_idx] * decay;
        kv_partial = __fmaf_rn(s[r], to_float(k_head[k_idx]), kv_partial);
    }

    // Lanes for one column are interleaved at a stride of COLS_PER_WARP.
#pragma unroll
    for (int offset = LANES_PER_COL / 2; offset > 0; offset >>= 1) {
        kv_partial += __shfl_down_sync(0xffffffffu, kv_partial, offset * COLS_PER_WARP);
    }
    const float kv = __shfl_sync(0xffffffffu, kv_partial, col_in_warp);
    const float delta = (to_float(v_head[v_idx]) - kv) * beta_t;

    float y_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
        const int k_idx = r * LANES_PER_COL + lane_in_col;
        s[r] = __fmaf_rn(to_float(k_head[k_idx]), delta, s[r]);
        y_partial = __fmaf_rn(s[r], to_float(q_head[k_idx]) * q_scale, y_partial);
        state_head[k_idx * V_DIM + v_idx] = s[r];
    }
#pragma unroll
    for (int offset = LANES_PER_COL / 2; offset > 0; offset >>= 1) {
        y_partial += __shfl_down_sync(0xffffffffu, y_partial, offset * COLS_PER_WARP);
    }
    if (lane_in_col == 0) {
        out[(b * num_v_heads + v_head_idx) * V_DIM + v_idx] = from_float<T>(y_partial);
    }
}

template <typename T>
static int launch_gdn_decode_smem(
    const T* q, const T* k, const T* v, const float* g, const float* beta,
    float* state, const int64_t* slots, T* out, int batch, int num_v_heads,
    int num_k_heads, int k_dim, int v_dim, float q_scale, int64_t v_offset,
    int64_t v_row_stride, cudaStream_t stream);

template <typename T>
void launch_gated_delta_rule_decode_slots_gqa(
    const T* q, const T* k, const T* v, const float* g, const float* beta,
    float* state, const int64_t* slots, T* out,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, cudaStream_t stream) {
    if (k_dim == 128 && v_dim == 128) {
        if (launch_gdn_decode_smem(
                q, k, v, g, beta, state, slots, out, batch, num_v_heads, num_k_heads,
                k_dim, v_dim, q_scale, 0,
                static_cast<int64_t>(num_v_heads) * v_dim, stream) == 0) {
            return;
        }
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int COLS_PER_WARP = 4;
        dim3 grid((v_dim + WARPS_PER_BLOCK * COLS_PER_WARP - 1) /
                      (WARPS_PER_BLOCK * COLS_PER_WARP),
                  batch * num_v_heads);
        dim3 block(GDN_WARP_SIZE, WARPS_PER_BLOCK);
        gated_delta_rule_decode_slots_gqa_128_kernel<T><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, state, slots, out,
            batch, num_v_heads, num_k_heads, q_scale, 0,
            static_cast<int64_t>(num_v_heads) * v_dim);
        CHECK_CUDA(cudaGetLastError());
        return;
    }
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, batch * num_v_heads);
    dim3 block(BV);
    if (k_dim <= 64) {
        constexpr int BK = 64;
        size_t smem = (2 * BK + 2) * sizeof(float);
        gated_delta_rule_decode_slots_gqa_kernel<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out,
            batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale);
    } else if (k_dim <= 128) {
        constexpr int BK = 128;
        size_t smem = (2 * BK + 2) * sizeof(float);
        gated_delta_rule_decode_slots_gqa_kernel<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out,
            batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale);
    } else {
        constexpr int BK = 256;
        if (k_dim > BK) {
            printf("gated_delta_rule_decode_slots_gqa: k_dim=%d exceeds MAX_K=%d\n", k_dim, BK);
            return;
        }
        size_t smem = (2 * BK + 2) * sizeof(float);
        gated_delta_rule_decode_slots_gqa_kernel<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out,
            batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void gated_delta_rule_decode_slots_gqa_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const float* beta, float* state,
    const int64_t* slots, __nv_bfloat16* out,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots_gqa(
        q, k, v, g, beta, state, slots, out,
        batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale, stream);
}

extern "C" void gated_delta_rule_decode_slots_gqa_f16(
    const half* q, const half* k, const half* v,
    const float* g, const float* beta, float* state,
    const int64_t* slots, half* out,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots_gqa(
        q, k, v, g, beta, state, slots, out,
        batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale, stream);
}

// Coalesced one-block-per-head GDN decode.
//
// The register kernel assigns each block 16 value columns and 128 threads;
// global state accesses touch 16B per row with a 512B stride, wasting DRAM
// sectors.  This kernel keeps the whole [K=128, V=128] head state in 64 KiB of
// shared memory, loads and stores it with fully coalesced per-row accesses
// (one thread per value column), and performs the two recurrence passes in
// shared memory.  Global traffic stays one read + one write per element.
template <typename T>
__global__ void gdn_decode_smem_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ state,
    const int64_t* __restrict__ slots,
    T* __restrict__ out,
    int num_v_heads,
    int num_k_heads,
    float q_scale,
    int64_t v_offset,
    int64_t v_row_stride) {
    constexpr int K_DIM = 128;
    constexpr int V_DIM = 128;
    extern __shared__ float smem[];

    const int bh = blockIdx.x;
    const int b = bh / num_v_heads;
    const int vh = bh - b * num_v_heads;
    const int64_t slot = slots[b];
    if (slot < 0) return;

    const int c = threadIdx.x;
    const int kv_group = num_v_heads / num_k_heads;
    const int kh = vh / kv_group;
    const T* q_head = q + (static_cast<size_t>(b) * num_k_heads + kh) * K_DIM;
    const T* k_head = k + (static_cast<size_t>(b) * num_k_heads + kh) * K_DIM;
    const T* v_head = v + v_offset + static_cast<int64_t>(b) * v_row_stride +
                      static_cast<int64_t>(vh) * V_DIM;
    float* state_head =
        state + (static_cast<size_t>(slot) * num_v_heads + vh) * K_DIM * V_DIM;
    const float decay = expf(g[b * num_v_heads + vh]);
    const float beta_t = beta[b * num_v_heads + vh];

    // Coalesced load: warp lanes read consecutive value columns of one row.
    for (int r = 0; r < K_DIM; ++r) {
        smem[r * V_DIM + c] = state_head[static_cast<size_t>(r) * V_DIM + c];
    }
    __syncthreads();

    const float v_val = to_float(v_head[c]);
    float kv = 0.0f;
    for (int r = 0; r < K_DIM; ++r) {
        const float sd = smem[r * V_DIM + c] * decay;
        smem[r * V_DIM + c] = sd;
        kv = __fmaf_rn(sd, to_float(k_head[r]), kv);
    }
    const float delta = (v_val - kv) * beta_t;

    float y = 0.0f;
    for (int r = 0; r < K_DIM; ++r) {
        const float sv = __fmaf_rn(to_float(k_head[r]), delta, smem[r * V_DIM + c]);
        smem[r * V_DIM + c] = sv;
        y = __fmaf_rn(sv, to_float(q_head[r]) * q_scale, y);
    }
    __syncthreads();

    for (int r = 0; r < K_DIM; ++r) {
        state_head[static_cast<size_t>(r) * V_DIM + c] = smem[r * V_DIM + c];
    }
    out[static_cast<size_t>(bh) * V_DIM + c] = from_float<T>(y);
}

template <typename T>
static int launch_gdn_decode_smem(
    const T* q, const T* k, const T* v, const float* g, const float* beta,
    float* state, const int64_t* slots, T* out,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, int64_t v_offset, int64_t v_row_stride,
    cudaStream_t stream) {
    if (k_dim != 128 || v_dim != 128 || num_v_heads % num_k_heads != 0) return 1;
    // One block per head needs enough batch to fill 114 SMs; below ~20
    // sequences the register kernel (8 blocks per head) hides latency better.
    if (batch < 20) return 1;
    // 64 KiB of dynamic shared memory per block is only comfortable on
    // Ampere+; on SM70/SM75 it would occupy the whole SM and slow down.
    static int sm_major = -1;
    if (sm_major < 0) {
        int dev = 0;
        sm_major = 0;
        if (cudaGetDevice(&dev) == cudaSuccess) {
            int major = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) ==
                cudaSuccess) {
                sm_major = major;
            }
        }
    }
    if (sm_major < 8) return 1;
    constexpr int smem_bytes = 128 * 128 * sizeof(float);
    static bool attr_set = false;
    if (!attr_set) {
        if (cudaFuncSetAttribute(
                gdn_decode_smem_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes) != cudaSuccess) {
            cudaGetLastError();
            return 1;
        }
        attr_set = true;
    }
    gdn_decode_smem_kernel<T><<<batch * num_v_heads, 128, smem_bytes, stream>>>(
        q, k, v, g, beta, state, slots, out, num_v_heads, num_k_heads, q_scale,
        v_offset, v_row_stride);
    CHECK_CUDA(cudaGetLastError());
    return 0;
}

extern "C" int gdn_decode_smem_f32(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, float* state, const int64_t* slots, float* out,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, int64_t v_offset, int64_t v_row_stride,
    cudaStream_t stream) {
    return launch_gdn_decode_smem(
        q, k, v, g, beta, state, slots, out, batch, num_v_heads, num_k_heads,
        k_dim, v_dim, q_scale, v_offset, v_row_stride, stream);
}

extern "C" int gdn_decode_smem_f16(
    const half* q, const half* k, const half* v, const float* g,
    const float* beta, float* state, const int64_t* slots, half* out,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, int64_t v_offset, int64_t v_row_stride,
    cudaStream_t stream) {
    return launch_gdn_decode_smem(
        q, k, v, g, beta, state, slots, out, batch, num_v_heads, num_k_heads,
        k_dim, v_dim, q_scale, v_offset, v_row_stride, stream);
}

extern "C" int gdn_decode_smem_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const float* beta, float* state, const int64_t* slots,
    __nv_bfloat16* out, int batch, int num_v_heads, int num_k_heads, int k_dim,
    int v_dim, float q_scale, int64_t v_offset, int64_t v_row_stride,
    cudaStream_t stream) {
    return launch_gdn_decode_smem(
        q, k, v, g, beta, state, slots, out, batch, num_v_heads, num_k_heads,
        k_dim, v_dim, q_scale, v_offset, v_row_stride, stream);
}

template <typename T>
static int launch_gated_delta_rule_decode_slots_gqa_strided(
    const T* q, const T* k, const T* v, const float* g, const float* beta,
    float* state, const int64_t* slots, T* out,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, int64_t v_offset, int64_t v_row_stride,
    cudaStream_t stream) {
    if (k_dim != 128 || v_dim != 128) return 1;
    if (launch_gdn_decode_smem(
            q, k, v, g, beta, state, slots, out, batch, num_v_heads, num_k_heads,
            k_dim, v_dim, q_scale, v_offset, v_row_stride, stream) == 0) {
        return 0;
    }
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int COLS_PER_WARP = 4;
    dim3 grid((v_dim + WARPS_PER_BLOCK * COLS_PER_WARP - 1) /
                  (WARPS_PER_BLOCK * COLS_PER_WARP),
              batch * num_v_heads);
    dim3 block(GDN_WARP_SIZE, WARPS_PER_BLOCK);
    gated_delta_rule_decode_slots_gqa_128_kernel<T><<<grid, block, 0, stream>>>(
        q, k, v, g, beta, state, slots, out,
        batch, num_v_heads, num_k_heads, q_scale, v_offset, v_row_stride);
    CHECK_CUDA(cudaGetLastError());
    return 0;
}

extern "C" int gated_delta_rule_decode_slots_gqa_strided_f32(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, float* state, const int64_t* slots, float* out,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, int64_t v_offset, int64_t v_row_stride,
    cudaStream_t stream) {
    return launch_gated_delta_rule_decode_slots_gqa_strided(
        q, k, v, g, beta, state, slots, out, batch, num_v_heads, num_k_heads,
        k_dim, v_dim, q_scale, v_offset, v_row_stride, stream);
}

extern "C" int gated_delta_rule_decode_slots_gqa_strided_f16(
    const half* q, const half* k, const half* v, const float* g,
    const float* beta, float* state, const int64_t* slots, half* out,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, int64_t v_offset, int64_t v_row_stride,
    cudaStream_t stream) {
    return launch_gated_delta_rule_decode_slots_gqa_strided(
        q, k, v, g, beta, state, slots, out, batch, num_v_heads, num_k_heads,
        k_dim, v_dim, q_scale, v_offset, v_row_stride, stream);
}

extern "C" int gated_delta_rule_decode_slots_gqa_strided_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const float* beta, float* state, const int64_t* slots,
    __nv_bfloat16* out, int batch, int num_v_heads, int num_k_heads, int k_dim,
    int v_dim, float q_scale, int64_t v_offset, int64_t v_row_stride,
    cudaStream_t stream) {
    return launch_gated_delta_rule_decode_slots_gqa_strided(
        q, k, v, g, beta, state, slots, out, batch, num_v_heads, num_k_heads,
        k_dim, v_dim, q_scale, v_offset, v_row_stride, stream);
}

extern "C" void gated_delta_rule_decode_slots_gqa_f32(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta, float* state,
    const int64_t* slots, float* out,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots_gqa(
        q, k, v, g, beta, state, slots, out,
        batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale, stream);
}

// =============================================================================
// Causal Conv1d Forward (Prefill, varlen)
// =============================================================================

template <typename T, int KERNEL_SIZE>
__global__ void causal_conv1d_fwd_varlen_kernel(
    const T* __restrict__ x,            // [total_tokens, d_conv]
    const T* __restrict__ weight,       // [d_conv, kernel_size]
    const T* __restrict__ bias,         // [d_conv] or nullptr
    float* __restrict__ conv_state,     // [batch, d_conv, kernel_size - 1], always F32
    T* __restrict__ out,                // [total_tokens, d_conv]
    T* __restrict__ state_snapshots,    // optional [total_tokens, d_conv, kernel_size - 1]
    const uint32_t* __restrict__ cu_seqlens, // [batch + 1]
    int batch_size,
    int d_conv,
    bool activation_silu) {
    int seq_idx = blockIdx.x;
    int channel_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (seq_idx >= batch_size || channel_idx >= d_conv) {
        return;
    }

    const int start = static_cast<int>(cu_seqlens[seq_idx]);
    const int end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    const int seq_len = end - start;

    const T* w_ptr = weight + channel_idx * KERNEL_SIZE;
    float* state_ptr = conv_state +
                   (seq_idx * d_conv + channel_idx) * (KERNEL_SIZE - 1);

    // Load weights into registers
    float w_reg[KERNEL_SIZE];
#pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) {
        w_reg[k] = to_float(w_ptr[k]);
    }

    float history[KERNEL_SIZE];
#pragma unroll
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        history[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < KERNEL_SIZE - 1; ++i) {
        history[i] = state_ptr[i];
    }

    float bias_val = (bias != nullptr) ? to_float(bias[channel_idx]) : 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        float x_t = to_float(x[(start + t) * d_conv + channel_idx]);
        float sum = x_t * w_reg[KERNEL_SIZE - 1];

#pragma unroll
        for (int k = 0; k < KERNEL_SIZE - 1; ++k) {
            sum += history[k] * w_reg[k];
        }

        if (bias != nullptr) {
            sum += bias_val;
        }
        if (activation_silu) {
            sum = silu_float(sum);
        }

        out[(start + t) * d_conv + channel_idx] = from_float<T>(sum);

        if (KERNEL_SIZE > 1) {
#pragma unroll
            for (int k = 0; k < KERNEL_SIZE - 2; ++k) {
                history[k] = history[k + 1];
            }
            history[KERNEL_SIZE - 2] = x_t;
        }

        if (state_snapshots != nullptr) {
            T* snapshot_ptr = state_snapshots +
                ((start + t) * d_conv + channel_idx) * (KERNEL_SIZE - 1);
#pragma unroll
            for (int i = 0; i < KERNEL_SIZE - 1; ++i) {
                snapshot_ptr[i] = from_float<T>(history[i]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < KERNEL_SIZE - 1; ++i) {
        state_ptr[i] = history[i];
    }
}

template <typename T>
void launch_causal_conv1d_fwd_varlen(const T* x, const T* weight, const T* bias,
                                     float* conv_state, T* out,
                                     T* state_snapshots,
                                     const uint32_t* cu_seqlens, int batch,
                                     int d_conv, int kernel_size, bool silu,
                                     cudaStream_t stream) {
    const int threads = 256;
    dim3 grid(batch, (d_conv + threads - 1) / threads);

    if (kernel_size == 4) {
        causal_conv1d_fwd_varlen_kernel<T, 4><<<grid, threads, 0, stream>>>(
            x, weight, bias, conv_state, out, state_snapshots, cu_seqlens, batch, d_conv, silu);
    } else if (kernel_size == 3) {
        causal_conv1d_fwd_varlen_kernel<T, 3><<<grid, threads, 0, stream>>>(
            x, weight, bias, conv_state, out, state_snapshots, cu_seqlens, batch, d_conv, silu);
    } else if (kernel_size == 2) {
        causal_conv1d_fwd_varlen_kernel<T, 2><<<grid, threads, 0, stream>>>(
            x, weight, bias, conv_state, out, state_snapshots, cu_seqlens, batch, d_conv, silu);
    } else {
         printf("causal_conv1d_fwd kernel_size=%d not supported (only 2,3,4)\\n", kernel_size);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void causal_conv1d_fwd_f32(
    const float* x, const float* weight, const float* bias, float* conv_state,
    float* out, float* state_snapshots, const uint32_t* cu_seqlens, int batch,
    int d_conv, int kernel_size, bool silu, cudaStream_t stream) {
    launch_causal_conv1d_fwd_varlen(x, weight, bias, conv_state, out, state_snapshots, cu_seqlens,
                                    batch, d_conv, kernel_size, silu, stream);
}

// =============================================================================
// Causal Conv1d Forward (Prefill, varlen) - token-parallel, vectorized
// =============================================================================
// The reference kernel walks the whole sequence in one thread per channel.
// Each output element needs only the current input plus K-1 history values, so
// the work is embarrassingly parallel over tokens once history is read back
// from x (or from the carried state at a sequence boundary).  One block per
// token with 16-byte channel vectors lifts this kernel from ~80 GB/s to near
// HBM bandwidth on Hopper.

template <typename T, int VEC>
union ConvAlignedVec {
    uint4 raw;
    T val[VEC];
};

template <typename T, int K, int VEC>
__global__ void causal_conv1d_fwd_parallel_kernel(
    const T* __restrict__ x,                 // [total_tokens, d_conv]
    const T* __restrict__ weight,            // [d_conv, K]
    const T* __restrict__ bias,              // [d_conv] or nullptr
    float* __restrict__ conv_state,          // [batch, d_conv, K-1], F32
    T* __restrict__ out,                     // [total_tokens, d_conv]
    T* __restrict__ state_snapshots,         // optional [total_tokens, d_conv, K-1]
    const uint32_t* __restrict__ cu_seqlens, // [batch + 1]
    int batch,
    int d_conv,
    int total_tokens,
    bool silu) {
    const int t = blockIdx.x;
    if (t >= total_tokens) return;

    // Locate the sequence containing this token.
    int lo = 0;
    int hi = batch - 1;
    int s = 0;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (static_cast<int>(cu_seqlens[mid]) <= t) {
            s = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    const int start = static_cast<int>(cu_seqlens[s]);
    const int end = static_cast<int>(cu_seqlens[s + 1]);
    const int pos = t - start;

    const int vecs = d_conv / VEC;
    for (int g = threadIdx.x; g < vecs; g += blockDim.x) {
        const int c0 = g * VEC;
        ConvAlignedVec<T, VEC> cur;
        cur.raw = *reinterpret_cast<const uint4*>(x + static_cast<size_t>(t) * d_conv + c0);

        float acc[VEC];
        {
            const T* wp = weight + static_cast<size_t>(c0) * K + (K - 1);
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                acc[j] = to_float(cur.val[j]) * to_float(wp[j * K]);
            }
        }
#pragma unroll
        for (int i = 0; i < K - 1; ++i) {
            const int jpos = pos - (K - 1) + i;
            float hv[VEC];
            if (jpos >= 0) {
                ConvAlignedVec<T, VEC> h;
                h.raw = *reinterpret_cast<const uint4*>(
                    x + static_cast<size_t>(start + jpos) * d_conv + c0);
#pragma unroll
                for (int j = 0; j < VEC; ++j) hv[j] = to_float(h.val[j]);
            } else {
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    hv[j] = conv_state[(static_cast<size_t>(s) * d_conv + c0 + j) * (K - 1) +
                                       (jpos + (K - 1))];
                }
            }
            const T* wp = weight + static_cast<size_t>(c0) * K + i;
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                acc[j] += hv[j] * to_float(wp[j * K]);
            }
        }
        if (bias != nullptr) {
#pragma unroll
            for (int j = 0; j < VEC; ++j) acc[j] += to_float(bias[c0 + j]);
        }

        ConvAlignedVec<T, VEC> ov;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float v = silu ? silu_float(acc[j]) : acc[j];
            ov.val[j] = from_float<T>(v);
        }
        *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * d_conv + c0) = ov.raw;

        if (state_snapshots != nullptr) {
            // History carried after this token: x_ext[pos-K+2 .. pos].
#pragma unroll
            for (int i = 0; i < K - 1; ++i) {
                const int jpos = pos - (K - 2) + i;
                T* sp = state_snapshots +
                        (static_cast<size_t>(t) * d_conv + c0) * (K - 1) + i;
#pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    if (jpos >= 0) {
                        sp[static_cast<size_t>(j) * (K - 1)] =
                            x[static_cast<size_t>(start + jpos) * d_conv + c0 + j];
                    } else {
                        sp[static_cast<size_t>(j) * (K - 1)] = from_float<T>(
                            conv_state[(static_cast<size_t>(s) * d_conv + c0 + j) * (K - 1) +
                                       (jpos + (K - 1))]);
                    }
                }
            }
        }
    }
}

// Runs after the main kernel (same stream): commits each sequence's trailing
// K-1 inputs into the carried state.  Kept separate so blocks handling the
// first tokens can still read the previous state without a race.
template <typename T, int K, int VEC>
__global__ void causal_conv1d_state_commit_kernel(
    const T* __restrict__ x,
    float* __restrict__ conv_state,
    const uint32_t* __restrict__ cu_seqlens,
    int d_conv,
    int batch,
    int batch_limit) {
    const int s = blockIdx.x;
    if (s >= batch || s >= batch_limit) return;
    const int start = static_cast<int>(cu_seqlens[s]);
    const int end = static_cast<int>(cu_seqlens[s + 1]);
    const int seq_len = end - start;
    const int vecs = d_conv / VEC;

    for (int g = threadIdx.x; g < vecs; g += blockDim.x) {
        const int c0 = g * VEC;
        float vals[K - 1][VEC];
#pragma unroll
        for (int i = 0; i < K - 1; ++i) {
            const int jpos = seq_len - (K - 1) + i;
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                if (jpos >= 0) {
                    vals[i][j] =
                        to_float(x[static_cast<size_t>(start + jpos) * d_conv + c0 + j]);
                } else {
                    vals[i][j] = conv_state[(static_cast<size_t>(s) * d_conv + c0 + j) *
                                                (K - 1) +
                                            (jpos + (K - 1))];
                }
            }
        }
#pragma unroll
        for (int i = 0; i < K - 1; ++i) {
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                conv_state[(static_cast<size_t>(s) * d_conv + c0 + j) * (K - 1) + i] =
                    vals[i][j];
            }
        }
    }
}

template <typename T, int K, int VEC>
static int launch_causal_conv1d_fwd_parallel_dtype(
    const T* x,
    const T* weight,
    const T* bias,
    float* conv_state,
    T* out,
    T* state_snapshots,
    const uint32_t* cu_seqlens,
    int batch,
    int d_conv,
    int total_tokens,
    bool silu,
    cudaStream_t stream) {
    if (d_conv % VEC != 0 || batch <= 0 || total_tokens <= 0) return 1;
    const int threads = 256;
    causal_conv1d_fwd_parallel_kernel<T, K, VEC><<<total_tokens, threads, 0, stream>>>(
        x, weight, bias, conv_state, out, state_snapshots, cu_seqlens, batch, d_conv,
        total_tokens, silu);
    causal_conv1d_state_commit_kernel<T, K, VEC><<<batch, threads, 0, stream>>>(
        x, conv_state, cu_seqlens, d_conv, batch, batch);
    CHECK_CUDA(cudaGetLastError());
    return 0;
}

template <typename T, int VEC>
static int launch_causal_conv1d_fwd_parallel(
    const T* x,
    const T* weight,
    const T* bias,
    float* conv_state,
    T* out,
    T* state_snapshots,
    const uint32_t* cu_seqlens,
    int batch,
    int d_conv,
    int kernel_size,
    int total_tokens,
    bool silu,
    cudaStream_t stream) {
    switch (kernel_size) {
        case 2:
            return launch_causal_conv1d_fwd_parallel_dtype<T, 2, VEC>(
                x, weight, bias, conv_state, out, state_snapshots, cu_seqlens, batch,
                d_conv, total_tokens, silu, stream);
        case 3:
            return launch_causal_conv1d_fwd_parallel_dtype<T, 3, VEC>(
                x, weight, bias, conv_state, out, state_snapshots, cu_seqlens, batch,
                d_conv, total_tokens, silu, stream);
        case 4:
            return launch_causal_conv1d_fwd_parallel_dtype<T, 4, VEC>(
                x, weight, bias, conv_state, out, state_snapshots, cu_seqlens, batch,
                d_conv, total_tokens, silu, stream);
        default:
            return 1;
    }
}

extern "C" int causal_conv1d_fwd_parallel_f32(
    const float* x, const float* weight, const float* bias, float* conv_state,
    float* out, float* state_snapshots, const uint32_t* cu_seqlens, int batch,
    int d_conv, int kernel_size, int total_tokens, bool silu, cudaStream_t stream) {
    return launch_causal_conv1d_fwd_parallel<float, 4>(
        x, weight, bias, conv_state, out, state_snapshots, cu_seqlens, batch, d_conv,
        kernel_size, total_tokens, silu, stream);
}

extern "C" int causal_conv1d_fwd_parallel_f16(
    const half* x, const half* weight, const half* bias, float* conv_state,
    half* out, half* state_snapshots, const uint32_t* cu_seqlens, int batch,
    int d_conv, int kernel_size, int total_tokens, bool silu, cudaStream_t stream) {
    return launch_causal_conv1d_fwd_parallel<half, 8>(
        x, weight, bias, conv_state, out, state_snapshots, cu_seqlens, batch, d_conv,
        kernel_size, total_tokens, silu, stream);
}

extern "C" int causal_conv1d_fwd_parallel_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    float* conv_state, __nv_bfloat16* out, __nv_bfloat16* state_snapshots,
    const uint32_t* cu_seqlens, int batch, int d_conv, int kernel_size,
    int total_tokens, bool silu, cudaStream_t stream) {
    return launch_causal_conv1d_fwd_parallel<__nv_bfloat16, 8>(
        x, weight, bias, conv_state, out, state_snapshots, cu_seqlens, batch, d_conv,
        kernel_size, total_tokens, silu, stream);
}

extern "C" void causal_conv1d_fwd_f16(
    const half* x, const half* weight, const half* bias, float* conv_state,
    half* out, half* state_snapshots, const uint32_t* cu_seqlens, int batch,
    int d_conv, int kernel_size, bool silu, cudaStream_t stream) {
    launch_causal_conv1d_fwd_varlen(x, weight, bias, conv_state, out, state_snapshots, cu_seqlens,
                                    batch, d_conv, kernel_size, silu, stream);
}

extern "C" void causal_conv1d_fwd_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, float* conv_state, __nv_bfloat16* out,
    __nv_bfloat16* state_snapshots, const uint32_t* cu_seqlens, int batch,
    int d_conv, int kernel_size, bool silu, cudaStream_t stream) {
    launch_causal_conv1d_fwd_varlen(x, weight, bias, conv_state, out, state_snapshots, cu_seqlens,
                                    batch, d_conv, kernel_size, silu, stream);
}

// =============================================================================
// Causal Conv1d Update (Decode)
// =============================================================================

template <typename T>
__global__ void causal_conv1d_update_kernel(
    const T* __restrict__ x,      // [batch, d_conv]
    const T* __restrict__ weight, // [d_conv, kernel_size]
    const T* __restrict__ bias,   // [d_conv] or nullptr
    float* __restrict__ conv_state,   // [batch, d_conv, kernel_size - 1], always F32
    T* __restrict__ out,          // [batch, d_conv]
    int batch_size,
    int d_conv,
    int kernel_size,
    bool activation_silu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * d_conv) {
        return;
    }

    int batch_idx = idx / d_conv;
    int channel_idx = idx % d_conv;

    const T* w_ptr = weight + channel_idx * kernel_size;
    float* state_ptr = conv_state +
                   (batch_idx * d_conv + channel_idx) * (kernel_size - 1);

    float history[GDN_MAX_KERNEL_SIZE];
#pragma unroll
    for (int i = 0; i < GDN_MAX_KERNEL_SIZE; ++i) {
        history[i] = 0.0f;
    }
    for (int i = 0; i < kernel_size - 1; ++i) {
        history[i] = state_ptr[i];
    }

    float x_t = to_float(x[idx]);
    float sum = x_t * to_float(w_ptr[kernel_size - 1]);
    for (int k = 0; k < kernel_size - 1; ++k) {
        sum += history[k] * to_float(w_ptr[k]);
    }

    if (bias != nullptr) {
        sum += to_float(bias[channel_idx]);
    }
    if (activation_silu) {
        sum = silu_float(sum);
    }

    if (kernel_size > 1) {
        for (int k = 0; k < kernel_size - 2; ++k) {
            state_ptr[k] = history[k + 1];
        }
        state_ptr[kernel_size - 2] = x_t;
    }

    out[idx] = from_float<T>(sum);
}

template <typename T>
void launch_causal_conv1d_update(const T* x, const T* weight, const T* bias,
                                 float* conv_state, T* out, int batch, int d_conv,
                                 int kernel_size, bool silu,
                                 cudaStream_t stream) {
    if (kernel_size < 1 || kernel_size > GDN_MAX_KERNEL_SIZE) {
        printf("causal_conv1d_update kernel_size=%d not supported (max=%d)\\n",
               kernel_size, GDN_MAX_KERNEL_SIZE);
        return;
    }
    int total = batch * d_conv;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    causal_conv1d_update_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, conv_state, out, batch, d_conv, kernel_size, silu);
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void causal_conv1d_update_f32(
    const float* x, const float* weight, const float* bias, float* conv_state,
    float* out, int batch, int d_conv, int kernel_size, bool silu,
    cudaStream_t stream) {
    launch_causal_conv1d_update(x, weight, bias, conv_state, out, batch, d_conv,
                                kernel_size, silu, stream);
}

extern "C" void causal_conv1d_update_f16(
    const half* x, const half* weight, const half* bias, float* conv_state,
    half* out, int batch, int d_conv, int kernel_size, bool silu,
    cudaStream_t stream) {
    launch_causal_conv1d_update(x, weight, bias, conv_state, out, batch, d_conv,
                                kernel_size, silu, stream);
}

extern "C" void causal_conv1d_update_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, float* conv_state, __nv_bfloat16* out,
    int batch, int d_conv, int kernel_size, bool silu, cudaStream_t stream) {
    launch_causal_conv1d_update(x, weight, bias, conv_state, out, batch, d_conv,
                                kernel_size, silu, stream);
}

template <typename T, int KERNEL_SIZE>
__global__ void causal_conv1d_update_slots_kernel(
    const T* __restrict__ x,      // [batch, d_conv]
    const T* __restrict__ weight, // [d_conv, kernel_size]
    const T* __restrict__ bias,   // [d_conv] or nullptr
    float* __restrict__ conv_state,   // [max_batch, d_conv, kernel_size - 1], always F32
    const int64_t* __restrict__ slots, // [batch]
    T* __restrict__ out,          // [batch, d_conv]
    int batch_size,
    int d_conv,
    bool activation_silu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * d_conv) {
        return;
    }

    int batch_idx = idx / d_conv;
    int channel_idx = idx % d_conv;
    int64_t slot = slots[batch_idx];
    if (slot < 0) return;

    const T* w_ptr = weight + channel_idx * KERNEL_SIZE;
    float* state_ptr = conv_state +
                   (slot * d_conv + channel_idx) * (KERNEL_SIZE - 1);

    // Load weights to registers
    float w_reg[KERNEL_SIZE];
#pragma unroll
    for (int k = 0; k < KERNEL_SIZE; ++k) {
        w_reg[k] = to_float(w_ptr[k]);
    }

    float history[KERNEL_SIZE];
#pragma unroll
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        history[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < KERNEL_SIZE - 1; ++i) {
        history[i] = state_ptr[i];
    }

    float x_t = to_float(x[idx]);
    float sum = x_t * w_reg[KERNEL_SIZE - 1];

#pragma unroll
    for (int k = 0; k < KERNEL_SIZE - 1; ++k) {
        sum += history[k] * w_reg[k];
    }

    if (bias != nullptr) {
        sum += to_float(bias[channel_idx]);
    }
    if (activation_silu) {
        sum = silu_float(sum);
    }

    if (KERNEL_SIZE > 1) {
#pragma unroll
        for (int k = 0; k < KERNEL_SIZE - 2; ++k) {
            state_ptr[k] = history[k + 1];
        }
        state_ptr[KERNEL_SIZE - 2] = x_t;
    }

    out[idx] = from_float<T>(sum);
}

template <typename T>
void launch_causal_conv1d_update_slots(const T* x, const T* weight, const T* bias,
                                       float* conv_state, const int64_t* slots, T* out,
                                       int batch, int d_conv, int kernel_size, bool silu,
                                       cudaStream_t stream) {
    int total = batch * d_conv;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (kernel_size == 4) {
        causal_conv1d_update_slots_kernel<T, 4><<<blocks, threads, 0, stream>>>(
            x, weight, bias, conv_state, slots, out, batch, d_conv, silu);
    } else if (kernel_size == 3) {
        causal_conv1d_update_slots_kernel<T, 3><<<blocks, threads, 0, stream>>>(
            x, weight, bias, conv_state, slots, out, batch, d_conv, silu);
    } else if (kernel_size == 2) {
        causal_conv1d_update_slots_kernel<T, 2><<<blocks, threads, 0, stream>>>(
            x, weight, bias, conv_state, slots, out, batch, d_conv, silu);
    } else {
        printf("causal_conv1d_update_slots kernel_size=%d not supported (only 2,3,4)\\n", kernel_size);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void causal_conv1d_update_slots_f32(
    const float* x, const float* weight, const float* bias, float* conv_state,
    const int64_t* slots, float* out, int batch, int d_conv, int kernel_size, bool silu,
    cudaStream_t stream) {
    launch_causal_conv1d_update_slots(
        x, weight, bias, conv_state, slots, out, batch, d_conv, kernel_size, silu, stream);
}

extern "C" void causal_conv1d_update_slots_f16(
    const half* x, const half* weight, const half* bias, float* conv_state,
    const int64_t* slots, half* out, int batch, int d_conv, int kernel_size, bool silu,
    cudaStream_t stream) {
    launch_causal_conv1d_update_slots(
        x, weight, bias, conv_state, slots, out, batch, d_conv, kernel_size, silu, stream);
}

extern "C" void causal_conv1d_update_slots_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    float* conv_state, const int64_t* slots, __nv_bfloat16* out,
    int batch, int d_conv, int kernel_size, bool silu, cudaStream_t stream) {
    launch_causal_conv1d_update_slots(
        x, weight, bias, conv_state, slots, out, batch, d_conv, kernel_size, silu, stream);
}

// =============================================================================
// Fused GDN Gating
// =============================================================================

// Helper for vector types
template <typename T>
struct VecType;

template <>
struct VecType<float> {
    using Type = float4;
    static constexpr int size = 4;
};

template <>
struct VecType<half> {
    using Type = float4; // 128 bits = 8 halves
    static constexpr int size = 8;
};

template <>
struct VecType<__nv_bfloat16> {
    using Type = float4; // 128 bits = 8 bfloat16s
    static constexpr int size = 8;
};

template <typename T>
__device__ __forceinline__ void compute_gating(
    T a_val, T b_val, float a_log_val, float dt_val, float& g_val, float& beta_val) {
    float a_f = to_float(a_val);
    float b_f = to_float(b_val);

    float x = a_f + dt_val;
    float softplus_x = (x <= 20.0f) ? log1pf(expf(x)) : x;
    g_val = -expf(a_log_val) * softplus_x;
    beta_val = 1.0f / (1.0f + expf(-b_f));
}

template <typename T>
__global__ void fused_gdn_gating_kernel(
    const float* __restrict__ a_log,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const float* __restrict__ dt_bias,
    float* __restrict__ g,
    float* __restrict__ beta,
    int total_elements,
    int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int h_idx = idx % num_heads;
    compute_gating<T>(
        a[idx],
        b[idx],
        a_log[h_idx],
        dt_bias[h_idx],
        g[idx],
        beta[idx]
    );
}

template <typename T>
__global__ void fused_gdn_gating_kernel_vectorized(
    const float* __restrict__ a_log,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const float* __restrict__ dt_bias,
    float* __restrict__ g,
    float* __restrict__ beta,
    int total_elements,
    int num_heads) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int h_idx = idx % num_heads;
    compute_gating<T>(
        a[idx],
        b[idx],
        a_log[h_idx],
        dt_bias[h_idx],
        g[idx],
        beta[idx]
    );
}

template <typename T>
void launch_fused_gdn_gating(const float* al, const T* a, const T* b, const float* dt,
                             float* g, float* beta, int bat, int seq, int h,
                             cudaStream_t stream) {
    int total = bat * seq * h;
    if (total == 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_gdn_gating_kernel<T><<<blocks, threads, 0, stream>>>(
        al, a, b, dt, g, beta, total, h);
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void fused_gdn_gating_f32(const float* al, const float* a,
                                     const float* b, const float* dt,
                                     float* g, float* beta, int bat,
                                     int seq, int h, cudaStream_t stream) {
    launch_fused_gdn_gating<float>(al, a, b, dt, g, beta, bat, seq, h, stream);
}

extern "C" void fused_gdn_gating_f16(const float* al, const half* a,
                                     const half* b, const float* dt,
                                     float* g, float* beta, int bat,
                                     int seq, int h, cudaStream_t stream) {
    launch_fused_gdn_gating<half>(al, a, b, dt, g, beta, bat, seq, h, stream);
}

extern "C" void fused_gdn_gating_bf16(const float* al,
                                      const __nv_bfloat16* a,
                                      const __nv_bfloat16* b,
                                      const float* dt,
                                      float* g,
                                      float* beta, int bat,
                                      int seq, int h,
                                      cudaStream_t stream) {
    launch_fused_gdn_gating<__nv_bfloat16>(
        al, a, b, dt, g, beta, bat, seq, h, stream
    );
}


// =============================================================================
// Fused Gated RMSNorm + SiLU(z) + Mul
// =============================================================================

template <typename T, typename W, int THREADS>
__global__ void gated_rmsnorm_silu_mul_kernel(
    const T* __restrict__ x,       // [rows, value_dim]
    const T* __restrict__ z,       // [rows, value_dim]
    const W* __restrict__ gamma,   // [group_size] (per-head) or [value_dim] (full)
    const W* __restrict__ bias,    // optional, same shape rule as gamma
    T* __restrict__ out,           // [rows, value_dim]
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    int act) {
    const int row_group = blockIdx.x;
    const int num_groups = value_dim / group_size;
    const int row = row_group / num_groups;
    const int group = row_group % num_groups;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    const int group_offset = row * value_dim + group * group_size;
    const T* x_group = x + group_offset;
    const T* z_group = z + group_offset;
    T* out_group = out + group_offset;

    // K8: Warp shuffle reduction instead of shared memory tree reduction
    float sumsq = 0.0f;
    for (int i = tid; i < group_size; i += THREADS) {
        float xv = to_float(x_group[i]);
        sumsq = __fmaf_rn(xv, xv, sumsq);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }

    // Inter-warp reduction via shared memory (only lane 0 of each warp)
    constexpr int NUM_WARPS = THREADS / 32;
    __shared__ float warp_sums[NUM_WARPS];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sumsq;
    }
    __syncthreads();

    // Final reduction in first warp
    float total = 0.0f;
    if (tid < NUM_WARPS) {
        total = warp_sums[tid];
    }
    if (warp_id == 0) {
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
    }
    // Broadcast result
    if (tid == 0) {
        warp_sums[0] = total;
    }
    __syncthreads();

    const float inv_rms = rsqrtf(warp_sums[0] / static_cast<float>(group_size) + eps);


    for (int i = tid; i < group_size; i += THREADS) {
        const int wb_idx = per_group_weights ? i : (group * group_size + i);
        float normed = to_float(x_group[i]) * inv_rms;
        float y = normed * to_float(gamma[wb_idx]);
        if (has_bias) {
            y += to_float(bias[wb_idx]);
        }
        float zv = to_float(z_group[i]);
        // act: 0 = silu (Qwen3.5), 1 = sigmoid (Qwen4 output_gate_type=sigmoid)
        float gate = (act == 1) ? (1.0f / (1.0f + expf(-zv))) : silu_float(zv);
        out_group[i] = from_float<T>(y * gate);
    }
}

template <typename T, typename W>
void launch_gated_rmsnorm_silu_mul(
    const T* x,
    const T* z,
    const W* gamma,
    const W* bias,
    T* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream,
    int act = 0) {
    if (rows <= 0 || value_dim <= 0 || group_size <= 0 || value_dim % group_size != 0) return;
    constexpr int THREADS = 256;
    const int num_groups = value_dim / group_size;
    dim3 grid(rows * num_groups);
    dim3 block(THREADS);
    gated_rmsnorm_silu_mul_kernel<T, W, THREADS><<<grid, block, 0, stream>>>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, act);
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void gdn_gated_rmsnorm_silu_mul_f32(
    const float* x,
    const float* z,
    const float* gamma,
    const float* bias,
    float* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    launch_gated_rmsnorm_silu_mul<float, float>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream);
}

extern "C" void gdn_gated_rmsnorm_silu_mul_f16(
    const half* x,
    const half* z,
    const half* gamma,
    const half* bias,
    half* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    launch_gated_rmsnorm_silu_mul<half, half>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream);
}

extern "C" void gdn_gated_rmsnorm_silu_mul_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* bias,
    __nv_bfloat16* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    launch_gated_rmsnorm_silu_mul<__nv_bfloat16, __nv_bfloat16>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream);
}

extern "C" void gdn_gated_rmsnorm_silu_mul_f16_wf32(
    const half* x,
    const half* z,
    const float* gamma,
    const float* bias,
    half* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    launch_gated_rmsnorm_silu_mul<half, float>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream);
}

extern "C" void gdn_gated_rmsnorm_silu_mul_bf16_wf32(
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const float* gamma,
    const float* bias,
    __nv_bfloat16* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    launch_gated_rmsnorm_silu_mul<__nv_bfloat16, float>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream);
}

// Sigmoid-gated variants (Qwen4 output_gate_type=sigmoid)
extern "C" void gdn_gated_rmsnorm_sigmoid_mul_f32(
    const float* x,
    const float* z,
    const float* gamma,
    const float* bias,
    float* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    launch_gated_rmsnorm_silu_mul<float, float>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream, 1);
}

extern "C" void gdn_gated_rmsnorm_sigmoid_mul_f16(
    const half* x,
    const half* z,
    const half* gamma,
    const half* bias,
    half* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    launch_gated_rmsnorm_silu_mul<half, half>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream, 1);
}

extern "C" void gdn_gated_rmsnorm_sigmoid_mul_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const __nv_bfloat16* gamma,
    const __nv_bfloat16* bias,
    __nv_bfloat16* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    launch_gated_rmsnorm_silu_mul<__nv_bfloat16, __nv_bfloat16>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream, 1);
}

extern "C" void gdn_gated_rmsnorm_sigmoid_mul_f16_wf32(
    const half* x,
    const half* z,
    const float* gamma,
    const float* bias,
    half* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    launch_gated_rmsnorm_silu_mul<half, float>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream, 1);
}

extern "C" void gdn_gated_rmsnorm_sigmoid_mul_bf16_wf32(
    const __nv_bfloat16* x,
    const __nv_bfloat16* z,
    const float* gamma,
    const float* bias,
    __nv_bfloat16* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    launch_gated_rmsnorm_silu_mul<__nv_bfloat16, float>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream, 1);
}

// =============================================================================
// Fused L2 Norm (last dim) — replaces ~8 Candle kernel launches (S5)
// =============================================================================

template <typename T, int WARPS_PER_BLOCK>
__launch_bounds__(WARPS_PER_BLOCK * 32)
__global__ void l2_norm_last_dim_warp_kernel(
    const T* __restrict__ input,   // [rows, dim]
    T* __restrict__ output,        // [rows, dim]
    int rows,
    int dim,
    float eps) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= rows) return;

    const T* in_row = input + row * dim;
    T* out_row = output + row * dim;

    float sumsq = 0.0f;
    for (int i = lane_id; i < dim; i += 32) {
        const float v = to_float(in_row[i]);
        sumsq = __fmaf_rn(v, v, sumsq);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }
    // After down-shuffle reduction only lane 0 has the full sum; broadcast it.
    const float row_sumsq = __shfl_sync(0xffffffff, sumsq, 0);
    const float inv_norm = rsqrtf(fmaxf(row_sumsq, 0.0f) + eps);

    for (int i = lane_id; i < dim; i += 32) {
        out_row[i] = from_float<T>(to_float(in_row[i]) * inv_norm);
    }
}

template <typename T, int THREADS>
__launch_bounds__(THREADS)
__global__ void l2_norm_last_dim_block_kernel(
    const T* __restrict__ input,   // [rows, dim]
    T* __restrict__ output,        // [rows, dim]
    int rows,
    int dim,
    float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    const T* in_row = input + row * dim;
    T* out_row = output + row * dim;

    float sumsq = 0.0f;
    for (int i = tid; i < dim; i += THREADS) {
        const float v = to_float(in_row[i]);
        sumsq = __fmaf_rn(v, v, sumsq);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }

    constexpr int NUM_WARPS = THREADS / 32;
    __shared__ float warp_sums[NUM_WARPS];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sumsq;
    }
    __syncthreads();

    float total = (tid < NUM_WARPS) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
    }
    if (tid == 0) {
        warp_sums[0] = total;
    }
    __syncthreads();

    const float inv_norm = rsqrtf(fmaxf(warp_sums[0], 0.0f) + eps);
    for (int i = tid; i < dim; i += THREADS) {
        out_row[i] = from_float<T>(to_float(in_row[i]) * inv_norm);
    }
}

template <typename T>
void launch_l2_norm_last_dim(const T* input, T* output, int rows, int dim,
                             float eps, cudaStream_t stream) {
    if (dim <= 256) {
        constexpr int WARPS_PER_BLOCK = 8;
        const int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        l2_norm_last_dim_warp_kernel<T, WARPS_PER_BLOCK>
            <<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(input, output, rows, dim, eps);
    } else {
        constexpr int THREADS = 256;
        l2_norm_last_dim_block_kernel<T, THREADS>
            <<<rows, THREADS, 0, stream>>>(input, output, rows, dim, eps);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void l2_norm_last_dim_f32(
    const float* input, float* output, int rows, int dim, float eps,
    cudaStream_t stream) {
    launch_l2_norm_last_dim(input, output, rows, dim, eps, stream);
}

extern "C" void l2_norm_last_dim_f16(
    const half* input, half* output, int rows, int dim, float eps,
    cudaStream_t stream) {
    launch_l2_norm_last_dim(input, output, rows, dim, eps, stream);
}

extern "C" void l2_norm_last_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, int rows, int dim,
    float eps, cudaStream_t stream) {
    launch_l2_norm_last_dim(input, output, rows, dim, eps, stream);
}

// L2Norm over the last dim of a strided slice of a packed [tokens, token_stride]
// matrix: output row (t*heads + h) is normalized from
// input[t*token_stride + offset + h*dim .. +dim].  Used by the GDN decode path
// to normalize q/k in place from the packed convolution output, avoiding
// per-tensor narrow+contiguous copies.
template <typename T, int WARPS_PER_BLOCK>
__global__ void l2_norm_range_warp_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int tokens,
    int heads,
    int token_stride,
    int offset,
    int dim,
    float eps) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int total_rows = tokens * heads;
    if (row >= total_rows) return;
    const int t = row / heads;
    const int h = row - t * heads;
    const T* in_row = input + static_cast<size_t>(t) * token_stride + offset + h * dim;
    T* out_row = output + static_cast<size_t>(row) * dim;

    float sumsq = 0.0f;
    for (int i = lane_id; i < dim; i += 32) {
        const float v = to_float(in_row[i]);
        sumsq = __fmaf_rn(v, v, sumsq);
    }
    for (int offset_ = 16; offset_ > 0; offset_ >>= 1) {
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset_);
    }
    const float row_sumsq = __shfl_sync(0xffffffff, sumsq, 0);
    const float inv_norm = rsqrtf(fmaxf(row_sumsq, 0.0f) + eps);
    for (int i = lane_id; i < dim; i += 32) {
        out_row[i] = from_float<T>(to_float(in_row[i]) * inv_norm);
    }
}

template <typename T>
static int launch_l2_norm_range(
    const T* input, T* output, int tokens, int heads, int token_stride,
    int offset, int dim, float eps, cudaStream_t stream) {
    if (dim > 256 || tokens <= 0 || heads <= 0) return 1;
    constexpr int WARPS_PER_BLOCK = 8;
    const int rows = tokens * heads;
    const int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    l2_norm_range_warp_kernel<T, WARPS_PER_BLOCK><<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
        input, output, tokens, heads, token_stride, offset, dim, eps);
    CHECK_CUDA(cudaGetLastError());
    return 0;
}

extern "C" int l2_norm_range_f32(
    const float* input, float* output, int tokens, int heads, int token_stride,
    int offset, int dim, float eps, cudaStream_t stream) {
    return launch_l2_norm_range(input, output, tokens, heads, token_stride, offset, dim, eps,
                                stream);
}

extern "C" int l2_norm_range_f16(
    const half* input, half* output, int tokens, int heads, int token_stride,
    int offset, int dim, float eps, cudaStream_t stream) {
    return launch_l2_norm_range(input, output, tokens, heads, token_stride, offset, dim, eps,
                                stream);
}

extern "C" int l2_norm_range_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, int tokens, int heads,
    int token_stride, int offset, int dim, float eps, cudaStream_t stream) {
    return launch_l2_norm_range(input, output, tokens, heads, token_stride, offset, dim, eps,
                                stream);
}

// =============================================================================
// Batched Varlen Recurrence — process multiple sequences in one launch (S1)
// Accepts native dtype inputs, computes in FP32
// =============================================================================

template <typename T, int BK, int WARPS_PER_BLOCK>
__global__ void gated_delta_rule_recurrence_varlen_kernel(
    const T* __restrict__ q,          // [total_tokens, num_heads, k_dim]
    const T* __restrict__ k,          // [total_tokens, num_heads, k_dim]
    const T* __restrict__ v,          // [total_tokens, num_heads, v_dim]
    const float* __restrict__ g,      // [total_tokens, num_heads] decay = exp(g)
    const float* __restrict__ beta,   // [total_tokens, num_heads]
    float* __restrict__ state,        // [max_batch, num_heads, k_dim, v_dim]
    const int64_t* __restrict__ slots, // [batch]
    T* __restrict__ out,              // [total_tokens, num_heads, v_dim]
    float* __restrict__ state_snapshots, // optional [total_tokens, num_heads, k_dim, v_dim]
    const uint32_t* __restrict__ cu_seqlens, // [batch + 1]
    int batch,
    int num_heads,
    int k_dim,
    int v_dim) {
    const int seq_head = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int v_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (seq_head >= batch * num_heads) return;

    const int seq_idx = seq_head / num_heads;
    const int head_idx = seq_head % num_heads;
    const int64_t slot = slots[seq_idx];
    if (slot < 0) return;

    const int start = static_cast<int>(cu_seqlens[seq_idx]);
    const int end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    const int seq_len = end - start;
    if (seq_len <= 0) return;

    const int token_stride_k = num_heads * k_dim;
    const int token_stride_v = num_heads * v_dim;
    const int token_stride_g = num_heads;

    const T* q_base = q + start * token_stride_k + head_idx * k_dim;
    const T* k_base = k + start * token_stride_k + head_idx * k_dim;
    const T* v_base = v + start * token_stride_v + head_idx * v_dim;
    const float* g_base = g + start * token_stride_g + head_idx;
    const float* beta_base = beta + start * token_stride_g + head_idx;
    T* out_base = out + start * token_stride_v + head_idx * v_dim;

    __shared__ float q_buf[2][BK];
    __shared__ float k_buf[2][BK];
    __shared__ float scalars[2][2]; // [buf_idx][0=decay, 1=beta]

    const bool v_valid = (v_idx < v_dim);

    // State layout: [k_dim, v_dim] — for each key row, load v_idx element
    float* state_head = v_valid ?
        state + ((slot * num_heads + head_idx) * k_dim * v_dim) : nullptr;

    constexpr int ROWS_PER_LANE = (BK + GDN_WARP_SIZE - 1) / GDN_WARP_SIZE;
    float s_shard[ROWS_PER_LANE];
    if (v_valid) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; ++r) {
            const int k_idx = r * GDN_WARP_SIZE + lane;
            s_shard[r] = (k_idx < k_dim) ? state_head[k_idx * v_dim + v_idx] : 0.0f;
        }
    }

    constexpr int TOTAL_THREADS = WARPS_PER_BLOCK * GDN_WARP_SIZE;
    const int tid = warp_id * GDN_WARP_SIZE + lane;

    // Preload first timestep into buffer 0
    if (seq_len > 0) {
        const T* q_0 = q_base;
        const T* k_0 = k_base;
        for (int j = tid; j < BK; j += TOTAL_THREADS) {
            if (j < k_dim) {
                q_buf[0][j] = to_float(q_0[j]);
                k_buf[0][j] = to_float(k_0[j]);
            } else {
                q_buf[0][j] = 0.0f;
                k_buf[0][j] = 0.0f;
            }
        }
        if (tid == 0) {
            scalars[0][0] = to_float(g_base[0]);
            scalars[0][1] = to_float(beta_base[0]);
        }
        __syncthreads();
    }

    for (int t = 0; t < seq_len; ++t) {
        const int cur = t & 1;
        const int nxt = 1 - cur;

        // Start loading next timestep into the other buffer (overlaps with compute)
        if (t + 1 < seq_len) {
            const T* q_next = q_base + (t + 1) * token_stride_k;
            const T* k_next = k_base + (t + 1) * token_stride_k;
            for (int j = tid; j < BK; j += TOTAL_THREADS) {
                if (j < k_dim) {
                    q_buf[nxt][j] = to_float(q_next[j]);
                    k_buf[nxt][j] = to_float(k_next[j]);
                } else {
                    q_buf[nxt][j] = 0.0f;
                    k_buf[nxt][j] = 0.0f;
                }
            }
            if (tid == 0) {
                scalars[nxt][0] = to_float(g_base[(t + 1) * token_stride_g]);
                scalars[nxt][1] = to_float(beta_base[(t + 1) * token_stride_g]);
            }
        }

        if (v_valid) {
            const float decay = scalars[cur][0];
            const float beta_t = scalars[cur][1];

            float kv_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < ROWS_PER_LANE; ++r) {
                const int k_idx = r * GDN_WARP_SIZE + lane;
                s_shard[r] *= decay;
                kv_partial = __fmaf_rn(s_shard[r], k_buf[cur][k_idx], kv_partial);
            }
            const float kv_mem = warp_reduce_sum(kv_partial);
            const float delta = (to_float(v_base[t * token_stride_v + v_idx]) - kv_mem) * beta_t;

            float y_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < ROWS_PER_LANE; ++r) {
                const int k_idx = r * GDN_WARP_SIZE + lane;
                s_shard[r] = __fmaf_rn(k_buf[cur][k_idx], delta, s_shard[r]);
                y_partial = __fmaf_rn(s_shard[r], q_buf[cur][k_idx], y_partial);
            }
            const float y_t = warp_reduce_sum(y_partial);

            if (lane == 0) {
                out_base[t * token_stride_v + v_idx] = from_float<T>(y_t);
            }
            if (state_snapshots != nullptr) {
                float* snapshot_head = state_snapshots +
                    (((start + t) * num_heads + head_idx) * k_dim * v_dim);
#pragma unroll
                for (int r = 0; r < ROWS_PER_LANE; ++r) {
                    const int k_idx = r * GDN_WARP_SIZE + lane;
                    if (k_idx < k_dim) {
                        snapshot_head[k_idx * v_dim + v_idx] = s_shard[r];
                    }
                }
            }
        }

        // Single sync: ensures next buffer load is complete before it's used
        __syncthreads();
    }

    if (v_valid) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; ++r) {
            const int k_idx = r * GDN_WARP_SIZE + lane;
            if (k_idx < k_dim) {
                state_head[k_idx * v_dim + v_idx] = s_shard[r];
            }
        }
    }
}

template <typename T>
void launch_gated_delta_rule_recurrence_varlen(
    const T* q, const T* k, const T* v, const float* g, const float* beta,
    float* state, const int64_t* slots, T* out, float* state_snapshots,
    const uint32_t* cu_seqlens,
    int batch, int num_heads, int k_dim, int v_dim,
    cudaStream_t stream) {

    if (k_dim == 64) {
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int BK = 64;
        dim3 grid((v_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batch * num_heads);
        dim3 block(GDN_WARP_SIZE, WARPS_PER_BLOCK);
        gated_delta_rule_recurrence_varlen_kernel<T, BK, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, state, slots, out, state_snapshots, cu_seqlens,
            batch, num_heads, k_dim, v_dim);
    } else if (k_dim == 128) {
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int BK = 128;
        dim3 grid((v_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batch * num_heads);
        dim3 block(GDN_WARP_SIZE, WARPS_PER_BLOCK);
        gated_delta_rule_recurrence_varlen_kernel<T, BK, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, state, slots, out, state_snapshots, cu_seqlens,
            batch, num_heads, k_dim, v_dim);
    } else {
         printf("gated_delta_rule_recurrence_varlen: k_dim=%d not supported (only 64, 128)\n", k_dim);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void gated_delta_rule_recurrence_varlen_f32(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, float* state, const int64_t* slots, float* out,
    float* state_snapshots, const uint32_t* cu_seqlens, int batch,
    int num_heads, int k_dim, int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_recurrence_varlen(
        q, k, v, g, beta, state, slots, out, state_snapshots, cu_seqlens,
        batch, num_heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_recurrence_varlen_f16(
    const half* q, const half* k, const half* v, const float* g,
    const float* beta, float* state, const int64_t* slots, half* out,
    float* state_snapshots, const uint32_t* cu_seqlens, int batch,
    int num_heads, int k_dim, int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_recurrence_varlen(
        q, k, v, g, beta, state, slots, out, state_snapshots, cu_seqlens,
        batch, num_heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_recurrence_varlen_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const float* beta, float* state,
    const int64_t* slots, __nv_bfloat16* out, float* state_snapshots,
    const uint32_t* cu_seqlens, int batch, int num_heads, int k_dim,
    int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_recurrence_varlen(
        q, k, v, g, beta, state, slots, out, state_snapshots, cu_seqlens,
        batch, num_heads, k_dim, v_dim, stream);
}

// =============================================================================
// Grouped-Query Varlen Recurrence — handles num_k_heads != num_v_heads.
// q/k have num_k_heads, v/g/beta/state/out have num_v_heads.
// Each v_head maps to k_head via: k_head = v_head / kv_group_size.
// Fuses the q_scale multiplication to avoid separate kernel launch.
// Uses shared memory double buffering for q/k with k cached in registers.
// Grid:  (v_dim / WARPS_PER_BLOCK, batch * num_v_heads)  Block: (32, WARPS_PER_BLOCK)
// =============================================================================

template <typename T, int BK, int WARPS_PER_BLOCK>
__global__ void gated_delta_rule_recurrence_varlen_gqa_kernel(
    const T* __restrict__ q,          // [total_tokens, num_k_heads, k_dim]
    const T* __restrict__ k,          // [total_tokens, num_k_heads, k_dim]
    const T* __restrict__ v,          // [total_tokens, num_v_heads, v_dim]
    const float* __restrict__ g,      // [total_tokens, num_v_heads] decay = exp(g)
    const float* __restrict__ beta,   // [total_tokens, num_v_heads]
    float* __restrict__ state,        // [max_batch, num_v_heads, k_dim, v_dim]
    const int64_t* __restrict__ slots, // [batch]
    T* __restrict__ out,              // [total_tokens, num_v_heads, v_dim]
    float* __restrict__ state_snapshots, // optional [total_tokens, num_v_heads, k_dim, v_dim]
    const uint32_t* __restrict__ cu_seqlens, // [batch + 1]
    int batch,
    int num_v_heads,
    int num_k_heads,
    int k_dim,
    int v_dim,
    float q_scale) {
    const int seq_head = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int v_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (seq_head >= batch * num_v_heads) return;

    const int seq_idx = seq_head / num_v_heads;
    const int v_head_idx = seq_head % num_v_heads;
    const int kv_group_size = num_v_heads / num_k_heads;
    const int k_head_idx = v_head_idx / kv_group_size;
    const int64_t slot = slots[seq_idx];
    if (slot < 0) return;

    const int start = static_cast<int>(cu_seqlens[seq_idx]);
    const int end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    const int seq_len = end - start;
    if (seq_len <= 0) return;

    const int token_stride_qk = num_k_heads * k_dim;
    const int token_stride_v = num_v_heads * v_dim;
    const int token_stride_g = num_v_heads;

    const T* q_base = q + start * token_stride_qk + k_head_idx * k_dim;
    const T* k_base = k + start * token_stride_qk + k_head_idx * k_dim;
    const T* v_base = v + start * token_stride_v + v_head_idx * v_dim;
    const float* g_base = g + start * token_stride_g + v_head_idx;
    const float* beta_base = beta + start * token_stride_g + v_head_idx;
    T* out_base = out + start * token_stride_v + v_head_idx * v_dim;

    __shared__ float q_buf[2][BK];
    __shared__ float k_buf[2][BK];
    __shared__ float scalars[2][2];

    const bool v_valid = (v_idx < v_dim);

    float* state_head = v_valid ?
        state + ((slot * num_v_heads + v_head_idx) * k_dim * v_dim) : nullptr;

    constexpr int ROWS_PER_LANE = (BK + GDN_WARP_SIZE - 1) / GDN_WARP_SIZE;
    float s_shard[ROWS_PER_LANE];
    if (v_valid) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; ++r) {
            const int k_idx = r * GDN_WARP_SIZE + lane;
            s_shard[r] = (k_idx < k_dim) ? state_head[k_idx * v_dim + v_idx] : 0.0f;
        }
    }

    constexpr int TOTAL_THREADS = WARPS_PER_BLOCK * GDN_WARP_SIZE;
    const int tid = warp_id * GDN_WARP_SIZE + lane;

    if (seq_len > 0) {
        for (int j = tid; j < BK; j += TOTAL_THREADS) {
            if (j < k_dim) {
                q_buf[0][j] = to_float(q_base[j]) * q_scale;
                k_buf[0][j] = to_float(k_base[j]);
            } else {
                q_buf[0][j] = 0.0f;
                k_buf[0][j] = 0.0f;
            }
        }
        if (tid == 0) {
            scalars[0][0] = expf(to_float(g_base[0]));
            scalars[0][1] = to_float(beta_base[0]);
        }
        __syncthreads();
    }

    for (int t = 0; t < seq_len; ++t) {
        const int cur = t & 1;
        const int nxt = 1 - cur;

        if (t + 1 < seq_len) {
            const T* q_next = q_base + (t + 1) * token_stride_qk;
            const T* k_next = k_base + (t + 1) * token_stride_qk;
            for (int j = tid; j < BK; j += TOTAL_THREADS) {
                if (j < k_dim) {
                    q_buf[nxt][j] = to_float(q_next[j]) * q_scale;
                    k_buf[nxt][j] = to_float(k_next[j]);
                } else {
                    q_buf[nxt][j] = 0.0f;
                    k_buf[nxt][j] = 0.0f;
                }
            }
            if (tid == 0) {
                scalars[nxt][0] = expf(to_float(g_base[(t + 1) * token_stride_g]));
                scalars[nxt][1] = to_float(beta_base[(t + 1) * token_stride_g]);
            }
        }

        if (v_valid) {
            const float decay = scalars[cur][0];
            const float beta_t = scalars[cur][1];

            float kv_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < ROWS_PER_LANE; ++r) {
                const int k_idx = r * GDN_WARP_SIZE + lane;
                s_shard[r] *= decay;
                kv_partial = __fmaf_rn(s_shard[r], k_buf[cur][k_idx], kv_partial);
            }
            const float kv_mem = warp_reduce_sum(kv_partial);
            const float delta = (to_float(v_base[t * token_stride_v + v_idx]) - kv_mem) * beta_t;

            float y_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < ROWS_PER_LANE; ++r) {
                const int k_idx = r * GDN_WARP_SIZE + lane;
                s_shard[r] = __fmaf_rn(k_buf[cur][k_idx], delta, s_shard[r]);
                y_partial = __fmaf_rn(s_shard[r], q_buf[cur][k_idx], y_partial);
            }
            const float y_t = warp_reduce_sum(y_partial);

            if (lane == 0) {
                out_base[t * token_stride_v + v_idx] = from_float<T>(y_t);
            }
            if (state_snapshots != nullptr) {
                float* snapshot_head = state_snapshots +
                    (((start + t) * num_v_heads + v_head_idx) * k_dim * v_dim);
#pragma unroll
                for (int r = 0; r < ROWS_PER_LANE; ++r) {
                    const int k_idx = r * GDN_WARP_SIZE + lane;
                    if (k_idx < k_dim) {
                        snapshot_head[k_idx * v_dim + v_idx] = s_shard[r];
                    }
                }
            }
        }

        __syncthreads();
    }

    if (v_valid) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_LANE; ++r) {
            const int k_idx = r * GDN_WARP_SIZE + lane;
            if (k_idx < k_dim) {
                state_head[k_idx * v_dim + v_idx] = s_shard[r];
            }
        }
    }
}

template <typename T>
void launch_gated_delta_rule_recurrence_varlen_gqa(
    const T* q, const T* k, const T* v, const float* g, const float* beta,
    float* state, const int64_t* slots, T* out, float* state_snapshots,
    const uint32_t* cu_seqlens,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, cudaStream_t stream) {

    if (k_dim == 128) {
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int BK = 128;
        dim3 grid((v_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batch * num_v_heads);
        dim3 block(GDN_WARP_SIZE, WARPS_PER_BLOCK);
        gated_delta_rule_recurrence_varlen_gqa_kernel<T, BK, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, state, slots, out, state_snapshots, cu_seqlens,
            batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale);
    } else if (k_dim == 64) {
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int BK = 64;
        dim3 grid((v_dim + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batch * num_v_heads);
        dim3 block(GDN_WARP_SIZE, WARPS_PER_BLOCK);
        gated_delta_rule_recurrence_varlen_gqa_kernel<T, BK, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, state, slots, out, state_snapshots, cu_seqlens,
            batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale);
    } else {
        printf("gated_delta_rule_recurrence_varlen_gqa: k_dim=%d not supported\n", k_dim);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void gated_delta_rule_recurrence_varlen_gqa_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const float* beta, float* state,
    const int64_t* slots, __nv_bfloat16* out, float* state_snapshots,
    const uint32_t* cu_seqlens,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, cudaStream_t stream) {
    launch_gated_delta_rule_recurrence_varlen_gqa(
        q, k, v, g, beta, state, slots, out, state_snapshots, cu_seqlens,
        batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale, stream);
}

extern "C" void gated_delta_rule_recurrence_varlen_gqa_f16(
    const half* q, const half* k, const half* v, const float* g,
    const float* beta, float* state, const int64_t* slots, half* out,
    float* state_snapshots, const uint32_t* cu_seqlens,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, cudaStream_t stream) {
    launch_gated_delta_rule_recurrence_varlen_gqa(
        q, k, v, g, beta, state, slots, out, state_snapshots, cu_seqlens,
        batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale, stream);
}

extern "C" void gated_delta_rule_recurrence_varlen_gqa_f32(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, float* state, const int64_t* slots, float* out,
    float* state_snapshots, const uint32_t* cu_seqlens,
    int batch, int num_v_heads, int num_k_heads, int k_dim, int v_dim,
    float q_scale, cudaStream_t stream) {
    launch_gated_delta_rule_recurrence_varlen_gqa(
        q, k, v, g, beta, state, slots, out, state_snapshots, cu_seqlens,
        batch, num_v_heads, num_k_heads, k_dim, v_dim, q_scale, stream);
}
