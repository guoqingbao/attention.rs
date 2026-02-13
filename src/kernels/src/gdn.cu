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
    return x / (1.0f + expf(-x));
}

// =============================================================================
// Gated Delta Rule Recurrence (DeltaNet core)
// =============================================================================

template <typename T, int BK, int BV>
__global__ void gated_delta_rule_recurrence_kernel_tiled(
    const T* __restrict__ q,          // [BH, S, K]
    const T* __restrict__ k,          // [BH, S, K]
    const T* __restrict__ v,          // [BH, S, V]
    const float* __restrict__ g,      // [BH, S]
    const float* __restrict__ beta,   // [BH, S]
    float* __restrict__ state,        // [BH, K, V] (in/out)
    float* __restrict__ out,          // [BH, S, V]
    int seq_len,
    int v_dim) {
    const int v_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int tid = threadIdx.x;
    const int v_idx = v_tile * BV + tid;

    if (v_idx >= v_dim) {
        return;
    }

    const T* q_bh = q + bh * seq_len * BK;
    const T* k_bh = k + bh * seq_len * BK;
    const T* v_bh = v + bh * seq_len * v_dim;
    const float* g_bh = g + bh * seq_len;
    const float* beta_bh = beta + bh * seq_len;
    float* state_bh = state + bh * BK * v_dim;
    float* out_bh = out + bh * seq_len * v_dim;

    // Shared memory layout: k_buf[BK] + q_buf[BK] + scalars[2]
    __shared__ float k_buf[BK];
    __shared__ float q_buf[BK];
    __shared__ float scalars[2]; // decay, beta_t

    // K1: Load state — state is [K, V], access pattern state[j * v_dim + v_idx]
    // is non-coalesced. Each thread loads its own column (v_idx is unique per thread).
    float s[BK];
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        s[j] = state_bh[j * v_dim + v_idx];
    }

    for (int t = 0; t < seq_len; ++t) {
        // Cooperative load k into shared memory
#pragma unroll
        for (int j = tid; j < BK; j += BV) {
            k_buf[j] = to_float(k_bh[t * BK + j]);
        }
        // Broadcast scalars via shared memory (K3 for prefill)
        if (tid == 0) {
            scalars[0] = expf(g_bh[t]);
            scalars[1] = beta_bh[t];
        }
        __syncthreads();

        const float decay = scalars[0];
        const float beta_t = scalars[1];
        const float v_t = to_float(v_bh[t * v_dim + v_idx]);

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

        out_bh[t * v_dim + v_idx] = y_t;
    }

    // Store state back
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        state_bh[j * v_dim + v_idx] = s[j];
    }
}

template <typename T, int BV, int MAX_K>
__global__ void gated_delta_rule_recurrence_kernel_fallback(
    const T* __restrict__ q,          // [BH, S, K]
    const T* __restrict__ k,          // [BH, S, K]
    const T* __restrict__ v,          // [BH, S, V]
    const float* __restrict__ g,      // [BH, S]
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

    if (v_idx >= v_dim) {
        return;
    }

    const T* q_bh = q + bh * seq_len * k_dim;
    const T* k_bh = k + bh * seq_len * k_dim;
    const T* v_bh = v + bh * seq_len * v_dim;
    const float* g_bh = g + bh * seq_len;
    const float* beta_bh = beta + bh * seq_len;
    float* state_bh = state + bh * k_dim * v_dim;
    float* out_bh = out + bh * seq_len * v_dim;

    // Dynamic shared memory: k_buf[k_dim] + q_buf[k_dim] + scalars[2]
    extern __shared__ float shared[];
    float* k_buf = shared;
    float* q_buf = shared + k_dim;
    float* scalars_buf = shared + 2 * k_dim;

    float s[MAX_K];
    for (int j = 0; j < k_dim; ++j) {
        s[j] = state_bh[j * v_dim + v_idx];
    }

    for (int t = 0; t < seq_len; ++t) {
        for (int j = tid; j < k_dim; j += BV) {
            k_buf[j] = to_float(k_bh[t * k_dim + j]);
        }
        if (tid == 0) {
            scalars_buf[0] = expf(g_bh[t]);
            scalars_buf[1] = beta_bh[t];
        }
        __syncthreads();

        const float decay = scalars_buf[0];
        const float beta_t = scalars_buf[1];
        const float v_t = to_float(v_bh[t * v_dim + v_idx]);

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

        out_bh[t * v_dim + v_idx] = y_t;
    }

    for (int j = 0; j < k_dim; ++j) {
        state_bh[j * v_dim + v_idx] = s[j];
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
    const T* __restrict__ g,      // [batch, heads]
    const T* __restrict__ beta,   // [batch, heads]
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
    if (v_idx >= v_dim || bh >= batch * heads) return;

    const int b = bh / heads;
    const int h = bh % heads;
    const int64_t slot = slots[b];
    if (slot < 0) return;

    // K3: broadcast scalars via shared memory — one expf instead of BV
    extern __shared__ float smem[];
    float* q_smem = smem;                  // [BK]
    float* k_smem = smem + BK;             // [BK]
    float* scalars = smem + 2 * BK;        // [2]: decay, beta_t

    if (tid == 0) {
        scalars[0] = expf(to_float(g[bh]));
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

    // K5: state pointer — stride v_dim between k elements (non-coalesced)
    // We load into registers; cooperative smem load not beneficial for decode
    // because each thread accesses a different v_idx column.
    T* state_bh = state + (((slot * heads + h) * k_dim) * v_dim + v_idx);

    float s_buf[BK];
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        s_buf[j] = (j < k_dim) ? to_float(state_bh[j * v_dim]) : 0.0f;
    }

    float kv_mem = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (j < k_dim) {
            s_buf[j] *= decay;
            kv_mem = __fmaf_rn(s_buf[j], k_smem[j], kv_mem);
        }
    }

    const T* v_bh = v + (bh * v_dim);
    const float delta = (to_float(v_bh[v_idx]) - kv_mem) * beta_t;

    // Load q into shared memory (reuse k_smem space — k no longer needed)
    __syncthreads();
    for (int j = tid; j < k_dim; j += BV) {
        q_smem[j] = to_float(q_bh[j]);
    }
    __syncthreads();

    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (j < k_dim) {
            s_buf[j] = __fmaf_rn(k_smem[j], delta, s_buf[j]);
            y = __fmaf_rn(s_buf[j], q_smem[j], y);
        }
    }

#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (j < k_dim) {
            state_bh[j * v_dim] = from_float<T>(s_buf[j]);
        }
    }

    out[bh * v_dim + v_idx] = from_float<T>(y);
}

template <typename T, int BV, int BK>
__global__ void gated_delta_rule_decode_slots_kernel_state_f32(
    const T* __restrict__ q,      // [batch, heads, k_dim]
    const T* __restrict__ k,      // [batch, heads, k_dim]
    const T* __restrict__ v,      // [batch, heads, v_dim]
    const T* __restrict__ g,      // [batch, heads]
    const T* __restrict__ beta,   // [batch, heads]
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
    if (v_idx >= v_dim || bh >= batch * heads) return;

    const int b = bh / heads;
    const int h = bh % heads;
    const int64_t slot = slots[b];
    if (slot < 0) return;

    // K3: broadcast scalars via shared memory
    extern __shared__ float smem[];
    float* q_smem = smem;
    float* k_smem = smem + BK;
    float* scalars = smem + 2 * BK;        // [2]: decay, beta_t

    if (tid == 0) {
        scalars[0] = expf(to_float(g[bh]));
        scalars[1] = to_float(beta[bh]);
    }

    // K6: cooperative load of k into shared memory
    const T* q_bh = q + bh * k_dim;
    const T* k_bh = k + bh * k_dim;
    for (int j = tid; j < k_dim; j += BV) {
        k_smem[j] = to_float(k_bh[j]);
    }
    __syncthreads();
    const float decay = scalars[0];
    const float beta_t = scalars[1];

    float* state_bh = state + (((slot * heads + h) * k_dim) * v_dim + v_idx);

    float s_buf[BK];
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        s_buf[j] = (j < k_dim) ? state_bh[j * v_dim] : 0.0f;
    }

    float kv_mem = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (j < k_dim) {
            s_buf[j] *= decay;
            kv_mem = __fmaf_rn(s_buf[j], k_smem[j], kv_mem);
        }
    }

    const T* v_bh = v + (bh * v_dim);
    const float delta = (to_float(v_bh[v_idx]) - kv_mem) * beta_t;

    // Load q into shared memory
    __syncthreads();
    for (int j = tid; j < k_dim; j += BV) {
        q_smem[j] = to_float(q_bh[j]);
    }
    __syncthreads();

    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (j < k_dim) {
            s_buf[j] = __fmaf_rn(k_smem[j], delta, s_buf[j]);
            y = __fmaf_rn(s_buf[j], q_smem[j], y);
        }
    }

#pragma unroll
    for (int j = 0; j < BK; ++j) {
        if (j < k_dim) {
            state_bh[j * v_dim] = s_buf[j];
        }
    }

    out[bh * v_dim + v_idx] = from_float<T>(y);
}

// K4: dispatch to exact BK sizes to minimize register pressure
template <typename T>
void launch_gated_delta_rule_decode_slots(
    const T* q, const T* k, const T* v, const T* g, const T* beta,
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
    const T* q, const T* k, const T* v, const T* g, const T* beta,
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
    const half* q, const half* k, const half* v, const half* g, const half* beta,
    half* state, const int64_t* slots, half* out, int batch, int heads, int k_dim,
    int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_decode_slots_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* g, const __nv_bfloat16* beta, __nv_bfloat16* state,
    const int64_t* slots, __nv_bfloat16* out, int batch, int heads, int k_dim,
    int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_decode_slots_f16_state_f32(
    const half* q, const half* k, const half* v, const half* g, const half* beta,
    float* state, const int64_t* slots, half* out, int batch, int heads, int k_dim,
    int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots_state_f32(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_decode_slots_bf16_state_f32(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* g, const __nv_bfloat16* beta, float* state,
    const int64_t* slots, __nv_bfloat16* out, int batch, int heads, int k_dim,
    int v_dim, cudaStream_t stream) {
    launch_gated_delta_rule_decode_slots_state_f32(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim, stream);
}

// =============================================================================
// Causal Conv1d Forward (Prefill, varlen)
// =============================================================================

template <typename T, int KERNEL_SIZE>
__global__ void causal_conv1d_fwd_varlen_kernel(
    const T* __restrict__ x,            // [total_tokens, d_conv]
    const T* __restrict__ weight,       // [d_conv, kernel_size]
    const T* __restrict__ bias,         // [d_conv] or nullptr
    T* __restrict__ conv_state,         // [batch, d_conv, kernel_size - 1]
    T* __restrict__ out,                // [total_tokens, d_conv]
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
    T* state_ptr = conv_state +
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
        history[i] = to_float(state_ptr[i]);
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
    }

#pragma unroll
    for (int i = 0; i < KERNEL_SIZE - 1; ++i) {
        state_ptr[i] = from_float<T>(history[i]);
    }
}

template <typename T>
void launch_causal_conv1d_fwd_varlen(const T* x, const T* weight, const T* bias,
                                     T* conv_state, T* out,
                                     const uint32_t* cu_seqlens, int batch,
                                     int d_conv, int kernel_size, bool silu,
                                     cudaStream_t stream) {
    const int threads = 256;
    dim3 grid(batch, (d_conv + threads - 1) / threads);

    if (kernel_size == 4) {
        causal_conv1d_fwd_varlen_kernel<T, 4><<<grid, threads, 0, stream>>>(
            x, weight, bias, conv_state, out, cu_seqlens, batch, d_conv, silu);
    } else if (kernel_size == 3) {
        causal_conv1d_fwd_varlen_kernel<T, 3><<<grid, threads, 0, stream>>>(
            x, weight, bias, conv_state, out, cu_seqlens, batch, d_conv, silu);
    } else if (kernel_size == 2) {
        causal_conv1d_fwd_varlen_kernel<T, 2><<<grid, threads, 0, stream>>>(
            x, weight, bias, conv_state, out, cu_seqlens, batch, d_conv, silu);
    } else {
         printf("causal_conv1d_fwd kernel_size=%d not supported (only 2,3,4)\\n", kernel_size);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void causal_conv1d_fwd_f32(
    const float* x, const float* weight, const float* bias, float* conv_state,
    float* out, const uint32_t* cu_seqlens, int batch, int d_conv, int kernel_size,
    bool silu, cudaStream_t stream) {
    launch_causal_conv1d_fwd_varlen(x, weight, bias, conv_state, out, cu_seqlens,
                                    batch, d_conv, kernel_size, silu, stream);
}

extern "C" void causal_conv1d_fwd_f16(
    const half* x, const half* weight, const half* bias, half* conv_state,
    half* out, const uint32_t* cu_seqlens, int batch, int d_conv, int kernel_size,
    bool silu, cudaStream_t stream) {
    launch_causal_conv1d_fwd_varlen(x, weight, bias, conv_state, out, cu_seqlens,
                                    batch, d_conv, kernel_size, silu, stream);
}

extern "C" void causal_conv1d_fwd_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* conv_state, __nv_bfloat16* out,
    const uint32_t* cu_seqlens, int batch, int d_conv, int kernel_size, bool silu,
    cudaStream_t stream) {
    launch_causal_conv1d_fwd_varlen(x, weight, bias, conv_state, out, cu_seqlens,
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
    T* __restrict__ conv_state,   // [batch, d_conv, kernel_size - 1]
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
    T* state_ptr = conv_state +
                   (batch_idx * d_conv + channel_idx) * (kernel_size - 1);

    float history[GDN_MAX_KERNEL_SIZE];
#pragma unroll
    for (int i = 0; i < GDN_MAX_KERNEL_SIZE; ++i) {
        history[i] = 0.0f;
    }
    for (int i = 0; i < kernel_size - 1; ++i) {
        history[i] = to_float(state_ptr[i]);
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
            state_ptr[k] = from_float<T>(history[k + 1]);
        }
        state_ptr[kernel_size - 2] = from_float<T>(x_t);
    }

    out[idx] = from_float<T>(sum);
}

template <typename T>
void launch_causal_conv1d_update(const T* x, const T* weight, const T* bias,
                                 T* conv_state, T* out, int batch, int d_conv,
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
    const half* x, const half* weight, const half* bias, half* conv_state,
    half* out, int batch, int d_conv, int kernel_size, bool silu,
    cudaStream_t stream) {
    launch_causal_conv1d_update(x, weight, bias, conv_state, out, batch, d_conv,
                                kernel_size, silu, stream);
}

extern "C" void causal_conv1d_update_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* conv_state, __nv_bfloat16* out,
    int batch, int d_conv, int kernel_size, bool silu, cudaStream_t stream) {
    launch_causal_conv1d_update(x, weight, bias, conv_state, out, batch, d_conv,
                                kernel_size, silu, stream);
}

template <typename T, int KERNEL_SIZE>
__global__ void causal_conv1d_update_slots_kernel(
    const T* __restrict__ x,      // [batch, d_conv]
    const T* __restrict__ weight, // [d_conv, kernel_size]
    const T* __restrict__ bias,   // [d_conv] or nullptr
    T* __restrict__ conv_state,   // [max_batch, d_conv, kernel_size - 1]
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
    T* state_ptr = conv_state +
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
        history[i] = to_float(state_ptr[i]);
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
            state_ptr[k] = from_float<T>(history[k + 1]);
        }
        state_ptr[KERNEL_SIZE - 2] = from_float<T>(x_t);
    }

    out[idx] = from_float<T>(sum);
}

template <typename T>
void launch_causal_conv1d_update_slots(const T* x, const T* weight, const T* bias,
                                       T* conv_state, const int64_t* slots, T* out,
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
    const half* x, const half* weight, const half* bias, half* conv_state,
    const int64_t* slots, half* out, int batch, int d_conv, int kernel_size, bool silu,
    cudaStream_t stream) {
    launch_causal_conv1d_update_slots(
        x, weight, bias, conv_state, slots, out, batch, d_conv, kernel_size, silu, stream);
}

extern "C" void causal_conv1d_update_slots_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    __nv_bfloat16* conv_state, const int64_t* slots, __nv_bfloat16* out,
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
    T a_val, T b_val, T a_log_val, T dt_val, T& g_val, T& beta_val) {
    float a_f = to_float(a_val);
    float b_f = to_float(b_val);
    float alog_f = to_float(a_log_val);
    float dt_f = to_float(dt_val);

    float x = a_f + dt_f;
    float softplus_x = (x <= 20.0f) ? log1pf(expf(x)) : x;
    float g_f = -expf(alog_f) * softplus_x;
    float beta_f = 1.0f / (1.0f + expf(-b_f));

    g_val = from_float<T>(g_f);
    beta_val = from_float<T>(beta_f);
}

template <typename T>
__global__ void fused_gdn_gating_kernel(
    const T* __restrict__ a_log,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ dt_bias,
    T* __restrict__ g,
    T* __restrict__ beta,
    int total_elements,
    int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    int h_idx = idx % num_heads;
    compute_gating(a[idx], b[idx], a_log[h_idx], dt_bias[h_idx], g[idx], beta[idx]);
}

template <typename T>
__global__ void fused_gdn_gating_kernel_vectorized(
    const T* __restrict__ a_log,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ dt_bias,
    T* __restrict__ g,
    T* __restrict__ beta,
    int total_elements,
    int num_heads) {
    
    using VecT = typename VecType<T>::Type;
    constexpr int VecSize = VecType<T>::size;
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
    if (idx >= total_elements) {
        return;
    }

    const VecT* a_vec_ptr = reinterpret_cast<const VecT*>(a);
    const VecT* b_vec_ptr = reinterpret_cast<const VecT*>(b);
    VecT* g_vec_ptr = reinterpret_cast<VecT*>(g);
    VecT* beta_vec_ptr = reinterpret_cast<VecT*>(beta);

    VecT a_vec = a_vec_ptr[blockIdx.x * blockDim.x + threadIdx.x];
    VecT b_vec = b_vec_ptr[blockIdx.x * blockDim.x + threadIdx.x];
    VecT g_vec;
    VecT beta_vec;

    T* a_arr = reinterpret_cast<T*>(&a_vec);
    T* b_arr = reinterpret_cast<T*>(&b_vec);
    T* g_arr = reinterpret_cast<T*>(&g_vec);
    T* beta_arr = reinterpret_cast<T*>(&beta_vec);

    #pragma unroll
    for (int i = 0; i < VecSize; ++i) {
        int curr_idx = idx + i;
        if (curr_idx < total_elements) {
            int h_idx = curr_idx % num_heads;
            compute_gating(a_arr[i], b_arr[i], a_log[h_idx], dt_bias[h_idx],
                           g_arr[i], beta_arr[i]);
        }
    }

    g_vec_ptr[blockIdx.x * blockDim.x + threadIdx.x] = g_vec;
    beta_vec_ptr[blockIdx.x * blockDim.x + threadIdx.x] = beta_vec;
}

template <typename T>
void launch_fused_gdn_gating(const T* al, const T* a, const T* b, const T* dt,
                             T* g, T* beta, int bat, int seq, int h,
                             cudaStream_t stream) {
    int total = bat * seq * h;
    if (total == 0) return;

    constexpr int VecSize = VecType<T>::size;
    
    // Check alignment
    bool aligned = (reinterpret_cast<uintptr_t>(a) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(b) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(g) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(beta) % 16 == 0) &&
                   (total % VecSize == 0);

    int threads = 256;

    if (aligned) {
        int vec_elements = total / VecSize;
        int blocks = (vec_elements + threads - 1) / threads;
        fused_gdn_gating_kernel_vectorized<T><<<blocks, threads, 0, stream>>>(
            al, a, b, dt, g, beta, total, h);
    } else {
        int blocks = (total + threads - 1) / threads;
        fused_gdn_gating_kernel<T><<<blocks, threads, 0, stream>>>(
            al, a, b, dt, g, beta, total, h);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void fused_gdn_gating_f32(const float* al, const float* a,
                                     const float* b, const float* dt,
                                     float* g, float* beta, int bat,
                                     int seq, int h, cudaStream_t stream) {
    launch_fused_gdn_gating(al, a, b, dt, g, beta, bat, seq, h, stream);
}

extern "C" void fused_gdn_gating_f16(const half* al, const half* a,
                                     const half* b, const half* dt,
                                     half* g, half* beta, int bat,
                                     int seq, int h, cudaStream_t stream) {
    launch_fused_gdn_gating(al, a, b, dt, g, beta, bat, seq, h, stream);
}

extern "C" void fused_gdn_gating_bf16(const __nv_bfloat16* al,
                                      const __nv_bfloat16* a,
                                      const __nv_bfloat16* b,
                                      const __nv_bfloat16* dt,
                                      __nv_bfloat16* g,
                                      __nv_bfloat16* beta, int bat,
                                      int seq, int h,
                                      cudaStream_t stream) {
    launch_fused_gdn_gating(al, a, b, dt, g, beta, bat, seq, h, stream);
}


// =============================================================================
// Fused Gated RMSNorm + SiLU(z) + Mul
// =============================================================================

template <typename T, int THREADS>
__global__ void gated_rmsnorm_silu_mul_kernel(
    const T* __restrict__ x,       // [rows, value_dim]
    const T* __restrict__ z,       // [rows, value_dim]
    const T* __restrict__ gamma,   // [group_size] (per-head) or [value_dim] (full)
    const T* __restrict__ bias,    // optional, same shape rule as gamma
    T* __restrict__ out,           // [rows, value_dim]
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias) {
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
        float gate = silu_float(to_float(z_group[i]));
        out_group[i] = from_float<T>(y * gate);
    }
}

template <typename T>
void launch_gated_rmsnorm_silu_mul(
    const T* x,
    const T* z,
    const T* gamma,
    const T* bias,
    T* out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    bool per_group_weights,
    bool has_bias,
    cudaStream_t stream) {
    if (rows <= 0 || value_dim <= 0 || group_size <= 0 || value_dim % group_size != 0) return;
    constexpr int THREADS = 256;
    const int num_groups = value_dim / group_size;
    dim3 grid(rows * num_groups);
    dim3 block(THREADS);
    gated_rmsnorm_silu_mul_kernel<T, THREADS><<<grid, block, 0, stream>>>(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias);
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
    launch_gated_rmsnorm_silu_mul(
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
    launch_gated_rmsnorm_silu_mul(
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
    launch_gated_rmsnorm_silu_mul(
        x, z, gamma, bias, out, rows, value_dim, group_size, eps, per_group_weights, has_bias, stream);
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
    const float inv_norm = rsqrtf(fmaxf(sumsq, 0.0f) + eps);

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

// =============================================================================
// Batched Varlen Recurrence — process multiple sequences in one launch (S1)
// Accepts native dtype inputs, computes in FP32
// =============================================================================

template <typename T, int BV, int BK>
__global__ void gated_delta_rule_recurrence_varlen_kernel(
    const T* __restrict__ q,          // [total_tokens, num_heads, k_dim]
    const T* __restrict__ k,          // [total_tokens, num_heads, k_dim]
    const T* __restrict__ v,          // [total_tokens, num_heads, v_dim]
    const T* __restrict__ g,          // [total_tokens, num_heads]
    const T* __restrict__ beta,       // [total_tokens, num_heads]
    float* __restrict__ state,        // [max_batch, num_heads, k_dim, v_dim]
    const int64_t* __restrict__ slots, // [batch]
    T* __restrict__ out,              // [total_tokens, num_heads, v_dim]
    const uint32_t* __restrict__ cu_seqlens, // [batch + 1]
    int batch,
    int num_heads,
    int k_dim,
    int v_dim) {
    const int v_tile = blockIdx.x;
    const int seq_head = blockIdx.y; // batch_idx * num_heads + head_idx
    const int tid = threadIdx.x;
    const int v_idx = v_tile * BV + tid;
    if (v_idx >= v_dim || seq_head >= batch * num_heads) return;

    const int seq_idx = seq_head / num_heads;
    const int head_idx = seq_head % num_heads;
    const int head_k_idx = head_idx;
    const int64_t slot = slots[seq_idx];
    if (slot < 0) return;

    const int start = static_cast<int>(cu_seqlens[seq_idx]);
    const int end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    const int seq_len = end - start;
    if (seq_len <= 0) return;

    // Pointers: input layout is [total_tokens, num_heads, dim]
    const int token_stride_k = num_heads * k_dim;
    const int token_stride_v = num_heads * v_dim;
    const int token_stride_g = num_heads;

    const T* q_base = q + start * token_stride_k + head_k_idx * k_dim;
    const T* k_base = k + start * token_stride_k + head_k_idx * k_dim;
    const T* v_base = v + start * token_stride_v + head_idx * v_dim;
    const T* g_base = g + start * token_stride_g + head_idx;
    const T* beta_base = beta + start * token_stride_g + head_idx;
    T* out_base = out + start * token_stride_v + head_idx * v_dim;

    float* state_bh = state + ((slot * num_heads + head_idx) * k_dim) * v_dim + v_idx;

    // Shared memory: 
    // k_buf[4][BK]
    // q_buf[4][BK]
    // scl[4][2]
    // Total: 2 * 4 * BK * 2 (k+q) + 16 (scl) floats.
    // For BK=128: 1024 + 1024 + 16 = 2064 floats = ~8KB.
    extern __shared__ float smem[];
    float* k_buf = smem;                 // [4*BK]
    float* q_buf = smem + 4 * BK;        // [4*BK]
    float* scl = smem + 8 * BK;          // [8]

    // Load state into registers
    float s[BK];
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        s[j] = state_bh[j * v_dim];
    }

    for (int t = 0; t < seq_len; t += 4) {
        // Load phase: Load t..t+3
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            if (t + sub < seq_len) {
                // Cooperative load k
                for (int j = tid; j < k_dim; j += BV) {
                    k_buf[sub * BK + j] = to_float(k_base[(t + sub) * token_stride_k + j]);
                }
                // Cooperative load q
                for (int j = tid; j < k_dim; j += BV) {
                    q_buf[sub * BK + j] = to_float(q_base[(t + sub) * token_stride_k + j]);
                }
                if (tid == 0) {
                    scl[sub * 2 + 0] = expf(to_float(g_base[(t + sub) * token_stride_g]));
                    scl[sub * 2 + 1] = to_float(beta_base[(t + sub) * token_stride_g]);
                }
            }
        }
        
        __syncthreads();

        // Compute phase: Process t..t+3
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            if (t + sub < seq_len) {
                const float decay = scl[sub * 2 + 0];
                const float beta_t = scl[sub * 2 + 1];
                const float v_t = to_float(v_base[(t + sub) * token_stride_v + v_idx]);

                float kv_mem = 0.0f;
                #pragma unroll
                for (int j = 0; j < BK; ++j) {
                    s[j] *= decay;
                    kv_mem = __fmaf_rn(s[j], k_buf[sub * BK + j], kv_mem);
                }

                const float delta = (v_t - kv_mem) * beta_t;

                float y_t = 0.0f;
                #pragma unroll
                for (int j = 0; j < BK; ++j) {
                    s[j] = __fmaf_rn(k_buf[sub * BK + j], delta, s[j]);
                    y_t = __fmaf_rn(s[j], q_buf[sub * BK + j], y_t);
                }

                out_base[(t + sub) * token_stride_v + v_idx] = from_float<T>(y_t);
            }
        }

        __syncthreads();
    }

    // Store state back
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        state_bh[j * v_dim] = s[j];
    }
}

template <typename T>
void launch_gated_delta_rule_recurrence_varlen(
    const T* q, const T* k, const T* v, const T* g, const T* beta,
    float* state, const int64_t* slots, T* out,
    const uint32_t* cu_seqlens,
    int batch, int num_heads, int k_dim, int v_dim,
    cudaStream_t stream) {
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, batch * num_heads);
    dim3 block(BV);

    if (k_dim == 64) {
        constexpr int BK = 64;
        size_t smem = (8 * BK + 8) * sizeof(float); // 4 way * 2 (k+q)
        gated_delta_rule_recurrence_varlen_kernel<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out, cu_seqlens,
            batch, num_heads, k_dim, v_dim);
    } else if (k_dim == 128) {
        constexpr int BK = 128;
        size_t smem = (8 * BK + 8) * sizeof(float);
        gated_delta_rule_recurrence_varlen_kernel<T, BV, BK><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, slots, out, cu_seqlens,
            batch, num_heads, k_dim, v_dim);
    } else {
         printf("gated_delta_rule_recurrence_varlen: k_dim=%d not supported (only 64, 128)\n", k_dim);
    }
    CHECK_CUDA(cudaGetLastError());
}

extern "C" void gated_delta_rule_recurrence_varlen_f32(
    const float* q, const float* k, const float* v, const float* g,
    const float* beta, float* state, const int64_t* slots, float* out,
    const uint32_t* cu_seqlens, int batch, int num_heads, int k_dim, int v_dim,
    cudaStream_t stream) {
    launch_gated_delta_rule_recurrence_varlen(
        q, k, v, g, beta, state, slots, out, cu_seqlens,
        batch, num_heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_recurrence_varlen_f16(
    const half* q, const half* k, const half* v, const half* g,
    const half* beta, float* state, const int64_t* slots, half* out,
    const uint32_t* cu_seqlens, int batch, int num_heads, int k_dim, int v_dim,
    cudaStream_t stream) {
    launch_gated_delta_rule_recurrence_varlen(
        q, k, v, g, beta, state, slots, out, cu_seqlens,
        batch, num_heads, k_dim, v_dim, stream);
}

extern "C" void gated_delta_rule_recurrence_varlen_bf16(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* g, const __nv_bfloat16* beta, float* state,
    const int64_t* slots, __nv_bfloat16* out, const uint32_t* cu_seqlens,
    int batch, int num_heads, int k_dim, int v_dim,
    cudaStream_t stream) {
    launch_gated_delta_rule_recurrence_varlen(
        q, k, v, g, beta, state, slots, out, cu_seqlens,
        batch, num_heads, k_dim, v_dim, stream);
}
