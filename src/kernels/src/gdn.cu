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

template <int BK, int BV>
__global__ void gated_delta_rule_recurrence_kernel_tiled(
    const float* __restrict__ q,      // [BH, S, K]
    const float* __restrict__ k,      // [BH, S, K]
    const float* __restrict__ v,      // [BH, S, V]
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

    const float* q_bh = q + bh * seq_len * BK;
    const float* k_bh = k + bh * seq_len * BK;
    const float* v_bh = v + bh * seq_len * v_dim;
    const float* g_bh = g + bh * seq_len;
    const float* beta_bh = beta + bh * seq_len;
    float* state_bh = state + bh * BK * v_dim;
    float* out_bh = out + bh * seq_len * v_dim;

    __shared__ float k_buf[BK];
    __shared__ float q_buf[BK];

    float s[BK];
#pragma unroll
    for (int j = 0; j < BK; ++j) {
        s[j] = state_bh[j * v_dim + v_idx];
    }

    for (int t = 0; t < seq_len; ++t) {
#pragma unroll
        for (int j = tid; j < BK; j += BV) {
            k_buf[j] = k_bh[t * BK + j];
        }
        __syncthreads();

        const float decay = expf(g_bh[t]);
        const float beta_t = beta_bh[t];
        const float v_t = v_bh[t * v_dim + v_idx];

        float kv_mem = 0.0f;
#pragma unroll
        for (int j = 0; j < BK; ++j) {
            s[j] *= decay;
            kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
        }

        const float delta = (v_t - kv_mem) * beta_t;

#pragma unroll
        for (int j = tid; j < BK; j += BV) {
            q_buf[j] = q_bh[t * BK + j];
        }
        __syncthreads();

        float y_t = 0.0f;
#pragma unroll
        for (int j = 0; j < BK; ++j) {
            s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
            y_t = __fmaf_rn(s[j], q_buf[j], y_t);
        }

        out_bh[t * v_dim + v_idx] = y_t;
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < BK; ++j) {
        state_bh[j * v_dim + v_idx] = s[j];
    }
}

template <int BV, int MAX_K>
__global__ void gated_delta_rule_recurrence_kernel_fallback(
    const float* __restrict__ q,      // [BH, S, K]
    const float* __restrict__ k,      // [BH, S, K]
    const float* __restrict__ v,      // [BH, S, V]
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

    const float* q_bh = q + bh * seq_len * k_dim;
    const float* k_bh = k + bh * seq_len * k_dim;
    const float* v_bh = v + bh * seq_len * v_dim;
    const float* g_bh = g + bh * seq_len;
    const float* beta_bh = beta + bh * seq_len;
    float* state_bh = state + bh * k_dim * v_dim;
    float* out_bh = out + bh * seq_len * v_dim;

    extern __shared__ float shared[];
    float* k_buf = shared;
    float* q_buf = shared + k_dim;

    float s[MAX_K];
    for (int j = 0; j < k_dim; ++j) {
        s[j] = state_bh[j * v_dim + v_idx];
    }

    for (int t = 0; t < seq_len; ++t) {
        for (int j = tid; j < k_dim; j += BV) {
            k_buf[j] = k_bh[t * k_dim + j];
        }
        __syncthreads();

        const float decay = expf(g_bh[t]);
        const float beta_t = beta_bh[t];
        const float v_t = v_bh[t * v_dim + v_idx];

        float kv_mem = 0.0f;
        for (int j = 0; j < k_dim; ++j) {
            s[j] *= decay;
            kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
        }

        const float delta = (v_t - kv_mem) * beta_t;

        for (int j = tid; j < k_dim; j += BV) {
            q_buf[j] = q_bh[t * k_dim + j];
        }
        __syncthreads();

        float y_t = 0.0f;
        for (int j = 0; j < k_dim; ++j) {
            s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
            y_t = __fmaf_rn(s[j], q_buf[j], y_t);
        }

        out_bh[t * v_dim + v_idx] = y_t;
        __syncthreads();
    }

    for (int j = 0; j < k_dim; ++j) {
        state_bh[j * v_dim + v_idx] = s[j];
    }
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
    if (bh <= 0 || seq_len <= 0 || k_dim <= 0 || v_dim <= 0) {
        return;
    }

    if (k_dim == 128) {
        constexpr int BK = 128;
        constexpr int BV = 64;
        dim3 grid((v_dim + BV - 1) / BV, bh);
        dim3 block(BV);
        gated_delta_rule_recurrence_kernel_tiled<BK, BV><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, state, out, seq_len, v_dim);
    } else if (k_dim == 64) {
        constexpr int BK = 64;
        constexpr int BV = 64;
        dim3 grid((v_dim + BV - 1) / BV, bh);
        dim3 block(BV);
        gated_delta_rule_recurrence_kernel_tiled<BK, BV><<<grid, block, 0, stream>>>(
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
        size_t smem = 2 * static_cast<size_t>(k_dim) * sizeof(float);
        gated_delta_rule_recurrence_kernel_fallback<BV, MAX_K><<<grid, block, smem, stream>>>(
            q, k, v, g, beta, state, out, seq_len, k_dim, v_dim);
    }
    CHECK_CUDA(cudaGetLastError());
}

template <typename T, int BV, int MAX_K>
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

    const T* q_bh = q + (bh * k_dim);
    const T* k_bh = k + (bh * k_dim);
    const T* v_bh = v + (bh * v_dim);
    const float g_t = to_float(g[bh]);
    const float beta_t = to_float(beta[bh]);

    T* state_bh = state + (((slot * heads + h) * k_dim) * v_dim + v_idx);
    T* out_bh = out + (bh * v_dim + v_idx);

    if (k_dim > MAX_K) return;

    const float decay = expf(g_t);
    const float scale = rsqrtf(static_cast<float>(k_dim));

    float s_buf[MAX_K];
    for (int j = 0; j < k_dim; ++j) {
        s_buf[j] = to_float(state_bh[j * v_dim]);
    }

    float kv_mem = 0.0f;
    for (int j = 0; j < k_dim; ++j) {
        s_buf[j] *= decay;
        kv_mem = __fmaf_rn(s_buf[j], to_float(k_bh[j]), kv_mem);
    }

    const float delta = (to_float(v_bh[v_idx]) - kv_mem) * beta_t;

    float y = 0.0f;
    for (int j = 0; j < k_dim; ++j) {
        s_buf[j] = __fmaf_rn(to_float(k_bh[j]), delta, s_buf[j]);
        y = __fmaf_rn(s_buf[j], to_float(q_bh[j]) * scale, y);
    }

    for (int j = 0; j < k_dim; ++j) {
        state_bh[j * v_dim] = from_float<T>(s_buf[j]);
    }

    *out_bh = from_float<T>(y);
}

template <typename T, int BV, int MAX_K>
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

    const T* q_bh = q + (bh * k_dim);
    const T* k_bh = k + (bh * k_dim);
    const T* v_bh = v + (bh * v_dim);
    const float g_t = to_float(g[bh]);
    const float beta_t = to_float(beta[bh]);

    float* state_bh = state + (((slot * heads + h) * k_dim) * v_dim + v_idx);
    T* out_bh = out + (bh * v_dim + v_idx);

    if (k_dim > MAX_K) return;

    const float decay = expf(g_t);
    const float scale = rsqrtf(static_cast<float>(k_dim));

    float s_buf[MAX_K];
    for (int j = 0; j < k_dim; ++j) {
        s_buf[j] = state_bh[j * v_dim];
    }

    float kv_mem = 0.0f;
    for (int j = 0; j < k_dim; ++j) {
        s_buf[j] *= decay;
        kv_mem = __fmaf_rn(s_buf[j], to_float(k_bh[j]), kv_mem);
    }

    const float delta = (to_float(v_bh[v_idx]) - kv_mem) * beta_t;

    float y = 0.0f;
    for (int j = 0; j < k_dim; ++j) {
        s_buf[j] = __fmaf_rn(to_float(k_bh[j]), delta, s_buf[j]);
        y = __fmaf_rn(s_buf[j], to_float(q_bh[j]) * scale, y);
    }

    for (int j = 0; j < k_dim; ++j) {
        state_bh[j * v_dim] = s_buf[j];
    }

    *out_bh = from_float<T>(y);
}

template <typename T>
void launch_gated_delta_rule_decode_slots(
    const T* q,
    const T* k,
    const T* v,
    const T* g,
    const T* beta,
    T* state,
    const int64_t* slots,
    T* out,
    int batch,
    int heads,
    int k_dim,
    int v_dim,
    cudaStream_t stream) {
    constexpr int BV = 64;
    constexpr int MAX_K = 256;
    if (k_dim > MAX_K) {
        printf("gated_delta_rule_decode_slots: k_dim=%d exceeds MAX_K=%d\\n", k_dim, MAX_K);
        return;
    }
    dim3 grid((v_dim + BV - 1) / BV, batch * heads);
    dim3 block(BV);
    gated_delta_rule_decode_slots_kernel<T, BV, MAX_K><<<grid, block, 0, stream>>>(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
void launch_gated_delta_rule_decode_slots_state_f32(
    const T* q,
    const T* k,
    const T* v,
    const T* g,
    const T* beta,
    float* state,
    const int64_t* slots,
    T* out,
    int batch,
    int heads,
    int k_dim,
    int v_dim,
    cudaStream_t stream) {
    constexpr int BV = 64;
    constexpr int MAX_K = 256;
    if (k_dim > MAX_K) {
        printf("gated_delta_rule_decode_slots_state_f32: k_dim=%d exceeds MAX_K=%d\\n", k_dim, MAX_K);
        return;
    }
    dim3 grid((v_dim + BV - 1) / BV, batch * heads);
    dim3 block(BV);
    gated_delta_rule_decode_slots_kernel_state_f32<T, BV, MAX_K><<<grid, block, 0, stream>>>(
        q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim);
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

template <typename T>
__global__ void causal_conv1d_fwd_varlen_kernel(
    const T* __restrict__ x,            // [total_tokens, d_conv]
    const T* __restrict__ weight,       // [d_conv, kernel_size]
    const T* __restrict__ bias,         // [d_conv] or nullptr
    T* __restrict__ conv_state,         // [batch, d_conv, kernel_size - 1]
    T* __restrict__ out,                // [total_tokens, d_conv]
    const uint32_t* __restrict__ cu_seqlens, // [batch + 1]
    int batch_size,
    int d_conv,
    int kernel_size,
    bool activation_silu) {
    int seq_idx = blockIdx.x;
    int channel_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (seq_idx >= batch_size || channel_idx >= d_conv) {
        return;
    }

    const int start = static_cast<int>(cu_seqlens[seq_idx]);
    const int end = static_cast<int>(cu_seqlens[seq_idx + 1]);
    const int seq_len = end - start;

    const T* w_ptr = weight + channel_idx * kernel_size;
    T* state_ptr = conv_state +
                   (seq_idx * d_conv + channel_idx) * (kernel_size - 1);

    float history[GDN_MAX_KERNEL_SIZE];
#pragma unroll
    for (int i = 0; i < GDN_MAX_KERNEL_SIZE; ++i) {
        history[i] = 0.0f;
    }
    for (int i = 0; i < kernel_size - 1; ++i) {
        history[i] = to_float(state_ptr[i]);
    }

    for (int t = 0; t < seq_len; ++t) {
        float x_t = to_float(x[(start + t) * d_conv + channel_idx]);
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

        out[(start + t) * d_conv + channel_idx] = from_float<T>(sum);

        if (kernel_size > 1) {
            for (int k = 0; k < kernel_size - 2; ++k) {
                history[k] = history[k + 1];
            }
            history[kernel_size - 2] = x_t;
        }
    }

    for (int i = 0; i < kernel_size - 1; ++i) {
        state_ptr[i] = from_float<T>(history[i]);
    }
}

template <typename T>
void launch_causal_conv1d_fwd_varlen(const T* x, const T* weight, const T* bias,
                                     T* conv_state, T* out,
                                     const uint32_t* cu_seqlens, int batch,
                                     int d_conv, int kernel_size, bool silu,
                                     cudaStream_t stream) {
    if (kernel_size < 1 || kernel_size > GDN_MAX_KERNEL_SIZE) {
        printf("causal_conv1d_fwd kernel_size=%d not supported (max=%d)\\n",
               kernel_size, GDN_MAX_KERNEL_SIZE);
        return;
    }
    const int threads = 256;
    dim3 grid(batch, (d_conv + threads - 1) / threads);
    causal_conv1d_fwd_varlen_kernel<<<grid, threads, 0, stream>>>(
        x, weight, bias, conv_state, out, cu_seqlens, batch, d_conv, kernel_size,
        silu);
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

template <typename T>
__global__ void causal_conv1d_update_slots_kernel(
    const T* __restrict__ x,      // [batch, d_conv]
    const T* __restrict__ weight, // [d_conv, kernel_size]
    const T* __restrict__ bias,   // [d_conv] or nullptr
    T* __restrict__ conv_state,   // [max_batch, d_conv, kernel_size - 1]
    const int64_t* __restrict__ slots, // [batch]
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
    int64_t slot = slots[batch_idx];
    if (slot < 0) return;

    const T* w_ptr = weight + channel_idx * kernel_size;
    T* state_ptr = conv_state +
                   (slot * d_conv + channel_idx) * (kernel_size - 1);

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
void launch_causal_conv1d_update_slots(const T* x, const T* weight, const T* bias,
                                       T* conv_state, const int64_t* slots, T* out,
                                       int batch, int d_conv, int kernel_size, bool silu,
                                       cudaStream_t stream) {
    if (kernel_size < 1 || kernel_size > GDN_MAX_KERNEL_SIZE) {
        printf("causal_conv1d_update_slots kernel_size=%d not supported (max=%d)\\n",
               kernel_size, GDN_MAX_KERNEL_SIZE);
        return;
    }
    int total = batch * d_conv;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    causal_conv1d_update_slots_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, conv_state, slots, out, batch, d_conv, kernel_size, silu);
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

template <typename T>
__global__ void fused_gdn_gating_kernel(const T* __restrict__ a_log,
                                        const T* __restrict__ a,
                                        const T* __restrict__ b,
                                        const T* __restrict__ dt_bias,
                                        T* __restrict__ g,
                                        T* __restrict__ beta, int batch,
                                        int seq_len, int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * num_heads;
    if (idx >= total) {
        return;
    }

    int h_idx = idx % num_heads;

    float a_f = to_float(a[idx]);
    float b_f = to_float(b[idx]);
    float alog_f = to_float(a_log[h_idx]);
    float dt_f = to_float(dt_bias[h_idx]);

    float x = a_f + dt_f;
    float softplus_x = (x <= 20.0f) ? log1pf(expf(x)) : x;
    float g_f = -expf(alog_f) * softplus_x;
    float beta_f = 1.0f / (1.0f + expf(-b_f));

    g[idx] = from_float<T>(g_f);
    beta[idx] = from_float<T>(beta_f);
}

template <typename T>
void launch_fused_gdn_gating(const T* al, const T* a, const T* b, const T* dt,
                             T* g, T* beta, int bat, int seq, int h,
                             cudaStream_t stream) {
    int total = bat * seq * h;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_gdn_gating_kernel<<<blocks, threads, 0, stream>>>(al, a, b, dt, g,
                                                             beta, bat, seq, h);
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

    __shared__ float sumsq_buf[THREADS];
    float sumsq = 0.0f;
    for (int i = tid; i < group_size; i += THREADS) {
        float xv = to_float(x_group[i]);
        sumsq = __fmaf_rn(xv, xv, sumsq);
    }
    sumsq_buf[tid] = sumsq;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sumsq_buf[tid] += sumsq_buf[tid + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(sumsq_buf[0] / static_cast<float>(group_size) + eps);

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
