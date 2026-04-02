#include "metal_dtype.metal"
#include <metal_stdlib>

using namespace metal;

static constant uint GDN_THREADS = 256;
static constant uint GDN_REC_THREADS = 64;

template <typename T>
inline float gdn_to_float(T x) {
    return static_cast<float>(x);
}

template <>
inline float gdn_to_float<float>(float x) {
    return x;
}

template <typename T>
inline T gdn_from_float(float x) {
    return static_cast<T>(x);
}

template <>
inline float gdn_from_float<float>(float x) {
    return x;
}

inline float gdn_silu(float x) {
    return x / (1.0f + exp(-x));
}

inline float gdn_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float gdn_softplus(float x) {
    return x <= 20.0f ? log(1.0f + exp(x)) : x;
}

template <typename T, uint KERNEL_SIZE>
kernel void causal_conv1d_fwd_varlen_kernel(
    const device T* x [[buffer(0)]],
    const device T* weight [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* conv_state [[buffer(3)]],
    device T* out [[buffer(4)]],
    const device uint* cu_seqlens [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    constant int& d_conv [[buffer(7)]],
    constant bool& activation_silu [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint channel_idx = gid.x;
    const uint seq_idx = gid.y;
    if (seq_idx >= static_cast<uint>(batch_size) || channel_idx >= static_cast<uint>(d_conv)) {
        return;
    }

    const uint start = cu_seqlens[seq_idx];
    const uint end = cu_seqlens[seq_idx + 1];
    const uint seq_len = end - start;
    const device T* w_ptr = weight + channel_idx * KERNEL_SIZE;
    device T* state_ptr = conv_state + (seq_idx * d_conv + channel_idx) * (KERNEL_SIZE - 1);

    float history[KERNEL_SIZE];
    for (uint i = 0; i < KERNEL_SIZE; ++i) {
        history[i] = 0.0f;
    }
    for (uint i = 0; i + 1 < KERNEL_SIZE; ++i) {
        history[i] = gdn_to_float(state_ptr[i]);
    }

    float w_reg[KERNEL_SIZE];
    for (uint i = 0; i < KERNEL_SIZE; ++i) {
        w_reg[i] = gdn_to_float(w_ptr[i]);
    }
    const float bias_val = bias != nullptr ? gdn_to_float(bias[channel_idx]) : 0.0f;

    for (uint t = 0; t < seq_len; ++t) {
        const float x_t = gdn_to_float(x[(start + t) * d_conv + channel_idx]);
        float sum = x_t * w_reg[KERNEL_SIZE - 1];
        for (uint i = 0; i + 1 < KERNEL_SIZE; ++i) {
            sum += history[i] * w_reg[i];
        }
        if (bias != nullptr) {
            sum += bias_val;
        }
        if (activation_silu) {
            sum = gdn_silu(sum);
        }
        out[(start + t) * d_conv + channel_idx] = gdn_from_float<T>(sum);

        if (KERNEL_SIZE > 1) {
            for (uint i = 0; i + 2 < KERNEL_SIZE; ++i) {
                history[i] = history[i + 1];
            }
            history[KERNEL_SIZE - 2] = x_t;
        }
    }

    for (uint i = 0; i + 1 < KERNEL_SIZE; ++i) {
        state_ptr[i] = gdn_from_float<T>(history[i]);
    }
}

template <typename T, uint KERNEL_SIZE>
kernel void causal_conv1d_update_kernel(
    const device T* x [[buffer(0)]],
    const device T* weight [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* conv_state [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant int& batch_size [[buffer(5)]],
    constant int& d_conv [[buffer(6)]],
    constant bool& activation_silu [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint channel_idx = gid.x;
    const uint batch_idx = gid.y;
    if (batch_idx >= static_cast<uint>(batch_size) || channel_idx >= static_cast<uint>(d_conv)) {
        return;
    }

    device T* state_ptr = conv_state + (batch_idx * d_conv + channel_idx) * (KERNEL_SIZE - 1);
    const device T* w_ptr = weight + channel_idx * KERNEL_SIZE;
    const float x_t = gdn_to_float(x[batch_idx * d_conv + channel_idx]);

    float sum = x_t * gdn_to_float(w_ptr[KERNEL_SIZE - 1]);
    float history[KERNEL_SIZE];
    for (uint i = 0; i < KERNEL_SIZE; ++i) {
        history[i] = 0.0f;
    }
    for (uint i = 0; i + 1 < KERNEL_SIZE; ++i) {
        history[i] = gdn_to_float(state_ptr[i]);
        sum += history[i] * gdn_to_float(w_ptr[i]);
    }

    if (bias != nullptr) {
        sum += gdn_to_float(bias[channel_idx]);
    }
    if (activation_silu) {
        sum = gdn_silu(sum);
    }
    out[batch_idx * d_conv + channel_idx] = gdn_from_float<T>(sum);

    if (KERNEL_SIZE > 1) {
        for (uint i = 0; i + 2 < KERNEL_SIZE; ++i) {
            state_ptr[i] = gdn_from_float<T>(history[i + 1]);
        }
        state_ptr[KERNEL_SIZE - 2] = gdn_from_float<T>(x_t);
    }
}

template <typename T, uint KERNEL_SIZE>
kernel void causal_conv1d_update_slots_kernel(
    const device T* x [[buffer(0)]],
    const device T* weight [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* conv_state [[buffer(3)]],
    const device int64_t* slots [[buffer(4)]],
    device T* out [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    constant int& d_conv [[buffer(7)]],
    constant bool& activation_silu [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint channel_idx = gid.x;
    const uint batch_idx = gid.y;
    if (batch_idx >= static_cast<uint>(batch_size) || channel_idx >= static_cast<uint>(d_conv)) {
        return;
    }

    const int64_t slot = slots[batch_idx];
    if (slot < 0) {
        return;
    }

    device T* state_ptr = conv_state + (static_cast<uint>(slot) * d_conv + channel_idx) * (KERNEL_SIZE - 1);
    const device T* w_ptr = weight + channel_idx * KERNEL_SIZE;
    const float x_t = gdn_to_float(x[batch_idx * d_conv + channel_idx]);

    float sum = x_t * gdn_to_float(w_ptr[KERNEL_SIZE - 1]);
    float history[KERNEL_SIZE];
    for (uint i = 0; i < KERNEL_SIZE; ++i) {
        history[i] = 0.0f;
    }
    for (uint i = 0; i + 1 < KERNEL_SIZE; ++i) {
        history[i] = gdn_to_float(state_ptr[i]);
        sum += history[i] * gdn_to_float(w_ptr[i]);
    }

    if (bias != nullptr) {
        sum += gdn_to_float(bias[channel_idx]);
    }
    if (activation_silu) {
        sum = gdn_silu(sum);
    }
    out[batch_idx * d_conv + channel_idx] = gdn_from_float<T>(sum);

    if (KERNEL_SIZE > 1) {
        for (uint i = 0; i + 2 < KERNEL_SIZE; ++i) {
            state_ptr[i] = gdn_from_float<T>(history[i + 1]);
        }
        state_ptr[KERNEL_SIZE - 2] = gdn_from_float<T>(x_t);
    }
}

template <typename T, typename ALogT>
kernel void fused_gdn_gating_kernel(
    const device ALogT* a_log [[buffer(0)]],
    const device T* a [[buffer(1)]],
    const device T* b [[buffer(2)]],
    const device T* dt_bias [[buffer(3)]],
    device T* g [[buffer(4)]],
    device T* beta [[buffer(5)]],
    constant int& total_elements [[buffer(6)]],
    constant int& num_heads [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= static_cast<uint>(total_elements)) {
        return;
    }
    const int h_idx = static_cast<int>(gid) % num_heads;
    const float a_val = gdn_to_float(a[gid]);
    const float b_val = gdn_to_float(b[gid]);
    const float alog_val = gdn_to_float(a_log[h_idx]);
    const float dt_val = gdn_to_float(dt_bias[h_idx]);
    const float g_val = -exp(alog_val) * gdn_softplus(a_val + dt_val);
    const float beta_val = gdn_sigmoid(b_val);
    g[gid] = gdn_from_float<T>(g_val);
    beta[gid] = gdn_from_float<T>(beta_val);
}

template <typename T>
kernel void l2_norm_last_dim_kernel(
    const device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant int& rows [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup3 [[threads_per_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint threads_per_threadgroup = threads_per_threadgroup3.x;
    const uint row = tgid.x;
    if (row >= static_cast<uint>(rows)) {
        return;
    }

    threadgroup float partial[GDN_THREADS];
    float sumsq = 0.0f;
    const uint base = row * dim;
    for (uint i = tid; i < static_cast<uint>(dim); i += threads_per_threadgroup) {
        const float v = gdn_to_float(input[base + i]);
        sumsq += v * v;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_norm = rsqrt(max(partial[0], 0.0f) + eps);
    for (uint i = tid; i < static_cast<uint>(dim); i += threads_per_threadgroup) {
        output[base + i] = gdn_from_float<T>(gdn_to_float(input[base + i]) * inv_norm);
    }
}

template <typename T, typename W>
kernel void gated_rmsnorm_silu_mul_kernel(
    const device T* x [[buffer(0)]],
    const device T* z [[buffer(1)]],
    const device W* gamma [[buffer(2)]],
    const device W* bias [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant int& rows [[buffer(5)]],
    constant int& value_dim [[buffer(6)]],
    constant int& group_size [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant bool& per_group_weights [[buffer(9)]],
    constant bool& has_bias [[buffer(10)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup3 [[threads_per_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint threads_per_threadgroup = threads_per_threadgroup3.x;
    const uint num_groups = static_cast<uint>(value_dim / group_size);
    const uint row_group = tgid.x;
    const uint row = row_group / num_groups;
    const uint group = row_group % num_groups;
    if (row >= static_cast<uint>(rows)) {
        return;
    }

    threadgroup float partial[GDN_THREADS];
    const uint group_offset = row * value_dim + group * group_size;

    float sumsq = 0.0f;
    for (uint i = tid; i < static_cast<uint>(group_size); i += threads_per_threadgroup) {
        const float v = gdn_to_float(x[group_offset + i]);
        sumsq += v * v;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt(partial[0] / static_cast<float>(group_size) + eps);
    for (uint i = tid; i < static_cast<uint>(group_size); i += threads_per_threadgroup) {
        const uint wb_idx = per_group_weights ? i : group * group_size + i;
        float value = gdn_to_float(x[group_offset + i]) * inv_rms * gdn_to_float(gamma[wb_idx]);
        if (has_bias) {
            value += gdn_to_float(bias[wb_idx]);
        }
        out[group_offset + i] = gdn_from_float<T>(value * gdn_silu(gdn_to_float(z[group_offset + i])));
    }
}

template <typename T, uint BK>
kernel void gated_delta_rule_recurrence_kernel(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device float* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device T* out [[buffer(6)]],
    constant int& bh [[buffer(7)]],
    constant int& seq_len [[buffer(8)]],
    constant int& v_dim [[buffer(9)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint v_idx = tgid.x * GDN_REC_THREADS + tid;
    const uint bh_idx = tgid.y;
    if (bh_idx >= static_cast<uint>(bh) || v_idx >= static_cast<uint>(v_dim)) {
        return;
    }

    const device T* q_bh = q + bh_idx * seq_len * BK;
    const device T* k_bh = k + bh_idx * seq_len * BK;
    const device T* v_bh = v + bh_idx * seq_len * v_dim;
    const device float* g_bh = g + bh_idx * seq_len;
    const device float* beta_bh = beta + bh_idx * seq_len;
    // State layout: [V, K] — each thread owns one contiguous K-row for its v_idx
    device float* state_bh = state + (bh_idx * BK * v_dim) + v_idx * BK;
    device T* out_bh = out + bh_idx * seq_len * v_dim;

    threadgroup float k_shared[BK];
    threadgroup float q_shared[BK];
    threadgroup float scalars[2];

    float s[BK];
    for (uint j = 0; j < BK; ++j) {
        s[j] = state_bh[j];
    }

    for (int t = 0; t < seq_len; ++t) {
        for (uint j = tid; j < BK; j += GDN_REC_THREADS) {
            k_shared[j] = gdn_to_float(k_bh[t * BK + j]);
        }
        if (tid == 0) {
            // g is pre-computed decay = exp(g) by caller
            scalars[0] = g_bh[t];
            scalars[1] = beta_bh[t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float decay = scalars[0];
        const float beta_t = scalars[1];
        const float v_t = gdn_to_float(v_bh[t * v_dim + v_idx]);

        float kv_mem = 0.0f;
        for (uint j = 0; j < BK; ++j) {
            s[j] *= decay;
            kv_mem += s[j] * k_shared[j];
        }
        const float delta = (v_t - kv_mem) * beta_t;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = tid; j < BK; j += GDN_REC_THREADS) {
            q_shared[j] = gdn_to_float(q_bh[t * BK + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float y_t = 0.0f;
        for (uint j = 0; j < BK; ++j) {
            s[j] += k_shared[j] * delta;
            y_t += s[j] * q_shared[j];
        }
        out_bh[t * v_dim + v_idx] = gdn_from_float<T>(y_t);
    }

    for (uint j = 0; j < BK; ++j) {
        state_bh[j] = s[j];
    }
}

template <typename T, uint BK>
kernel void gated_delta_rule_decode_slots_kernel(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device T* g [[buffer(3)]],
    const device T* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    const device int64_t* slots [[buffer(6)]],
    device T* out [[buffer(7)]],
    constant int& batch [[buffer(8)]],
    constant int& heads [[buffer(9)]],
    constant int& k_dim [[buffer(10)]],
    constant int& v_dim [[buffer(11)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint v_idx = tgid.x * GDN_REC_THREADS + tid;
    const uint bh = tgid.y;
    if (bh >= static_cast<uint>(batch * heads) || v_idx >= static_cast<uint>(v_dim)) {
        return;
    }

    const uint b = bh / heads;
    const uint h = bh % heads;
    const int64_t slot = slots[b];
    if (slot < 0) {
        return;
    }

    threadgroup float k_shared[BK];
    threadgroup float q_shared[BK];
    threadgroup float scalars[2];

    const device T* q_bh = q + bh * k_dim;
    const device T* k_bh = k + bh * k_dim;
    const device T* v_bh = v + bh * v_dim;
    if (tid == 0) {
        // g is pre-computed decay = exp(g) by caller
        scalars[0] = gdn_to_float(g[bh]);
        scalars[1] = gdn_to_float(beta[bh]);
    }
    for (uint j = tid; j < static_cast<uint>(k_dim); j += GDN_REC_THREADS) {
        k_shared[j] = gdn_to_float(k_bh[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // State layout: [V, K] — contiguous K-row per v_idx
    device float* state_bh = state + (((static_cast<uint>(slot) * heads + h) * v_dim + v_idx) * k_dim);
    float s[BK];
    for (uint j = 0; j < BK; ++j) {
        s[j] = j < static_cast<uint>(k_dim) ? state_bh[j] : 0.0f;
    }

    const float decay = scalars[0];
    const float beta_t = scalars[1];
    float kv_mem = 0.0f;
    for (uint j = 0; j < BK; ++j) {
        if (j < static_cast<uint>(k_dim)) {
            s[j] *= decay;
            kv_mem += s[j] * k_shared[j];
        }
    }
    const float delta = (gdn_to_float(v_bh[v_idx]) - kv_mem) * beta_t;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < static_cast<uint>(k_dim); j += GDN_REC_THREADS) {
        q_shared[j] = gdn_to_float(q_bh[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float y = 0.0f;
    for (uint j = 0; j < BK; ++j) {
        if (j < static_cast<uint>(k_dim)) {
            s[j] += k_shared[j] * delta;
            y += s[j] * q_shared[j];
            state_bh[j] = s[j];
        }
    }
    out[bh * v_dim + v_idx] = gdn_from_float<T>(y);
}

template <typename T, uint BK>
kernel void gated_delta_rule_recurrence_varlen_kernel(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device T* g [[buffer(3)]],
    const device T* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    const device int64_t* slots [[buffer(6)]],
    device T* out [[buffer(7)]],
    const device uint* cu_seqlens [[buffer(8)]],
    constant int& batch [[buffer(9)]],
    constant int& num_heads [[buffer(10)]],
    constant int& k_dim [[buffer(11)]],
    constant int& v_dim [[buffer(12)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint tid = tid3.x;
    const uint v_idx = tgid.x * GDN_REC_THREADS + tid;
    const uint seq_head = tgid.y;
    if (seq_head >= static_cast<uint>(batch * num_heads) || v_idx >= static_cast<uint>(v_dim)) {
        return;
    }

    const uint seq_idx = seq_head / num_heads;
    const uint head_idx = seq_head % num_heads;
    const int64_t slot = slots[seq_idx];
    if (slot < 0) {
        return;
    }

    const uint start = cu_seqlens[seq_idx];
    const uint end = cu_seqlens[seq_idx + 1];
    const uint seq_len = end - start;
    if (seq_len == 0) {
        return;
    }

    const uint token_stride_k = num_heads * k_dim;
    const uint token_stride_v = num_heads * v_dim;
    const uint token_stride_g = num_heads;

    threadgroup float k_shared[BK];
    threadgroup float q_shared[BK];
    threadgroup float scalars[2];

    // State layout: [V, K] — contiguous K-row per v_idx
    device float* state_bh = state + (((static_cast<uint>(slot) * num_heads + head_idx) * v_dim + v_idx) * k_dim);
    float s[BK];
    for (uint j = 0; j < BK; ++j) {
        s[j] = state_bh[j];
    }

    for (uint t = 0; t < seq_len; ++t) {
        const uint token_idx = start + t;
        const uint qk_base = token_idx * token_stride_k + head_idx * k_dim;
        const uint v_base = token_idx * token_stride_v + head_idx * v_dim;
        const uint g_base = token_idx * token_stride_g + head_idx;

        for (uint j = tid; j < static_cast<uint>(k_dim); j += GDN_REC_THREADS) {
            k_shared[j] = gdn_to_float(k[qk_base + j]);
        }
        if (tid == 0) {
            // g is pre-computed decay = exp(g) by caller
            scalars[0] = gdn_to_float(g[g_base]);
            scalars[1] = gdn_to_float(beta[g_base]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float decay = scalars[0];
        const float beta_t = scalars[1];
        float kv_mem = 0.0f;
        for (uint j = 0; j < BK; ++j) {
            s[j] *= decay;
            kv_mem += s[j] * k_shared[j];
        }
        const float delta = (gdn_to_float(v[v_base + v_idx]) - kv_mem) * beta_t;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = tid; j < static_cast<uint>(k_dim); j += GDN_REC_THREADS) {
            q_shared[j] = gdn_to_float(q[qk_base + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float y = 0.0f;
        for (uint j = 0; j < BK; ++j) {
            s[j] += k_shared[j] * delta;
            y += s[j] * q_shared[j];
        }
        out[v_base + v_idx] = gdn_from_float<T>(y);
    }

    for (uint j = 0; j < BK; ++j) {
        state_bh[j] = s[j];
    }
}

template <typename T>
kernel void mamba_scatter_rows_kernel(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    const device int64_t* slots [[buffer(2)]],
    constant int& num_rows [[buffer(3)]],
    constant int& row_elems [[buffer(4)]],
    constant int64_t& src_row_stride [[buffer(5)]],
    constant int64_t& dst_row_stride [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = static_cast<uint>(num_rows * row_elems);
    if (gid >= total) {
        return;
    }
    const uint row = gid / static_cast<uint>(row_elems);
    const uint elem = gid % static_cast<uint>(row_elems);
    const int64_t slot = slots[row];
    if (slot < 0) {
        return;
    }
    dst[static_cast<uint>(slot * dst_row_stride) + elem] = src[row * src_row_stride + elem];
}

#define INSTANTIATE_CONV_FWD(type, ksize) \
template [[host_name("gdn_causal_conv1d_fwd_" #type "_k" #ksize)]] \
[[kernel]] void causal_conv1d_fwd_varlen_kernel<type, ksize>( \
    const device type* x [[buffer(0)]], \
    const device type* weight [[buffer(1)]], \
    const device type* bias [[buffer(2)]], \
    device type* conv_state [[buffer(3)]], \
    device type* out [[buffer(4)]], \
    const device uint* cu_seqlens [[buffer(5)]], \
    constant int& batch_size [[buffer(6)]], \
    constant int& d_conv [[buffer(7)]], \
    constant bool& activation_silu [[buffer(8)]], \
    uint2 gid [[thread_position_in_grid]]);

#define INSTANTIATE_CONV_UPDATE(type, ksize) \
template [[host_name("gdn_causal_conv1d_update_" #type "_k" #ksize)]] \
[[kernel]] void causal_conv1d_update_kernel<type, ksize>( \
    const device type* x [[buffer(0)]], \
    const device type* weight [[buffer(1)]], \
    const device type* bias [[buffer(2)]], \
    device type* conv_state [[buffer(3)]], \
    device type* out [[buffer(4)]], \
    constant int& batch_size [[buffer(5)]], \
    constant int& d_conv [[buffer(6)]], \
    constant bool& activation_silu [[buffer(7)]], \
    uint2 gid [[thread_position_in_grid]]);

#define INSTANTIATE_CONV_UPDATE_SLOTS(type, ksize) \
template [[host_name("gdn_causal_conv1d_update_slots_" #type "_k" #ksize)]] \
[[kernel]] void causal_conv1d_update_slots_kernel<type, ksize>( \
    const device type* x [[buffer(0)]], \
    const device type* weight [[buffer(1)]], \
    const device type* bias [[buffer(2)]], \
    device type* conv_state [[buffer(3)]], \
    const device int64_t* slots [[buffer(4)]], \
    device type* out [[buffer(5)]], \
    constant int& batch_size [[buffer(6)]], \
    constant int& d_conv [[buffer(7)]], \
    constant bool& activation_silu [[buffer(8)]], \
    uint2 gid [[thread_position_in_grid]]);

#define INSTANTIATE_GATING(type, alog_type, name) \
template [[host_name(name)]] \
[[kernel]] void fused_gdn_gating_kernel<type, alog_type>( \
    const device alog_type* a_log [[buffer(0)]], \
    const device type* a [[buffer(1)]], \
    const device type* b [[buffer(2)]], \
    const device type* dt_bias [[buffer(3)]], \
    device type* g [[buffer(4)]], \
    device type* beta [[buffer(5)]], \
    constant int& total_elements [[buffer(6)]], \
    constant int& num_heads [[buffer(7)]], \
    uint gid [[thread_position_in_grid]]);

#define INSTANTIATE_L2(type) \
template [[host_name("gdn_l2_norm_last_dim_" #type)]] \
[[kernel]] void l2_norm_last_dim_kernel<type>( \
    const device type* input [[buffer(0)]], \
    device type* output [[buffer(1)]], \
    constant int& rows [[buffer(2)]], \
    constant int& dim [[buffer(3)]], \
    constant float& eps [[buffer(4)]], \
    uint3 tid3 [[thread_position_in_threadgroup]], \
    uint3 threads_per_threadgroup3 [[threads_per_threadgroup]], \
    uint3 tgid [[threadgroup_position_in_grid]]);

#define INSTANTIATE_RMSNORM(type, wtype, name) \
template [[host_name(name)]] \
[[kernel]] void gated_rmsnorm_silu_mul_kernel<type, wtype>( \
    const device type* x [[buffer(0)]], \
    const device type* z [[buffer(1)]], \
    const device wtype* gamma [[buffer(2)]], \
    const device wtype* bias [[buffer(3)]], \
    device type* out [[buffer(4)]], \
    constant int& rows [[buffer(5)]], \
    constant int& value_dim [[buffer(6)]], \
    constant int& group_size [[buffer(7)]], \
    constant float& eps [[buffer(8)]], \
    constant bool& per_group_weights [[buffer(9)]], \
    constant bool& has_bias [[buffer(10)]], \
    uint3 tid3 [[thread_position_in_threadgroup]], \
    uint3 threads_per_threadgroup3 [[threads_per_threadgroup]], \
    uint3 tgid [[threadgroup_position_in_grid]]);

#define INSTANTIATE_RECURRENCE(type, bk) \
template [[host_name("gdn_gated_delta_rule_recurrence_" #type "_k" #bk)]] \
[[kernel]] void gated_delta_rule_recurrence_kernel<type, bk>( \
    const device type* q [[buffer(0)]], \
    const device type* k [[buffer(1)]], \
    const device type* v [[buffer(2)]], \
    const device float* g [[buffer(3)]], \
    const device float* beta [[buffer(4)]], \
    device float* state [[buffer(5)]], \
    device type* out [[buffer(6)]], \
    constant int& bh [[buffer(7)]], \
    constant int& seq_len [[buffer(8)]], \
    constant int& v_dim [[buffer(9)]], \
    uint3 tid3 [[thread_position_in_threadgroup]], \
    uint3 tgid [[threadgroup_position_in_grid]]);

#define INSTANTIATE_DECODE(type, bk) \
template [[host_name("gdn_gated_delta_rule_decode_slots_" #type "_k" #bk)]] \
[[kernel]] void gated_delta_rule_decode_slots_kernel<type, bk>( \
    const device type* q [[buffer(0)]], \
    const device type* k [[buffer(1)]], \
    const device type* v [[buffer(2)]], \
    const device type* g [[buffer(3)]], \
    const device type* beta [[buffer(4)]], \
    device float* state [[buffer(5)]], \
    const device int64_t* slots [[buffer(6)]], \
    device type* out [[buffer(7)]], \
    constant int& batch [[buffer(8)]], \
    constant int& heads [[buffer(9)]], \
    constant int& k_dim [[buffer(10)]], \
    constant int& v_dim [[buffer(11)]], \
    uint3 tid3 [[thread_position_in_threadgroup]], \
    uint3 tgid [[threadgroup_position_in_grid]]);

#define INSTANTIATE_VARLEN(type, bk) \
template [[host_name("gdn_gated_delta_rule_recurrence_varlen_" #type "_k" #bk)]] \
[[kernel]] void gated_delta_rule_recurrence_varlen_kernel<type, bk>( \
    const device type* q [[buffer(0)]], \
    const device type* k [[buffer(1)]], \
    const device type* v [[buffer(2)]], \
    const device type* g [[buffer(3)]], \
    const device type* beta [[buffer(4)]], \
    device float* state [[buffer(5)]], \
    const device int64_t* slots [[buffer(6)]], \
    device type* out [[buffer(7)]], \
    const device uint* cu_seqlens [[buffer(8)]], \
    constant int& batch [[buffer(9)]], \
    constant int& num_heads [[buffer(10)]], \
    constant int& k_dim [[buffer(11)]], \
    constant int& v_dim [[buffer(12)]], \
    uint3 tid3 [[thread_position_in_threadgroup]], \
    uint3 tgid [[threadgroup_position_in_grid]]);

#define INSTANTIATE_SCATTER(type) \
template [[host_name("gdn_mamba_scatter_rows_" #type)]] \
[[kernel]] void mamba_scatter_rows_kernel<type>( \
    const device type* src [[buffer(0)]], \
    device type* dst [[buffer(1)]], \
    const device int64_t* slots [[buffer(2)]], \
    constant int& num_rows [[buffer(3)]], \
    constant int& row_elems [[buffer(4)]], \
    constant int64_t& src_row_stride [[buffer(5)]], \
    constant int64_t& dst_row_stride [[buffer(6)]], \
    uint gid [[thread_position_in_grid]]);

INSTANTIATE_CONV_FWD(float, 2)
INSTANTIATE_CONV_FWD(float, 3)
INSTANTIATE_CONV_FWD(float, 4)
INSTANTIATE_CONV_FWD(half, 2)
INSTANTIATE_CONV_FWD(half, 3)
INSTANTIATE_CONV_FWD(half, 4)
INSTANTIATE_CONV_FWD(bfloat16_t, 2)
INSTANTIATE_CONV_FWD(bfloat16_t, 3)
INSTANTIATE_CONV_FWD(bfloat16_t, 4)

INSTANTIATE_CONV_UPDATE(float, 2)
INSTANTIATE_CONV_UPDATE(float, 3)
INSTANTIATE_CONV_UPDATE(float, 4)
INSTANTIATE_CONV_UPDATE(half, 2)
INSTANTIATE_CONV_UPDATE(half, 3)
INSTANTIATE_CONV_UPDATE(half, 4)
INSTANTIATE_CONV_UPDATE(bfloat16_t, 2)
INSTANTIATE_CONV_UPDATE(bfloat16_t, 3)
INSTANTIATE_CONV_UPDATE(bfloat16_t, 4)

INSTANTIATE_CONV_UPDATE_SLOTS(float, 2)
INSTANTIATE_CONV_UPDATE_SLOTS(float, 3)
INSTANTIATE_CONV_UPDATE_SLOTS(float, 4)
INSTANTIATE_CONV_UPDATE_SLOTS(half, 2)
INSTANTIATE_CONV_UPDATE_SLOTS(half, 3)
INSTANTIATE_CONV_UPDATE_SLOTS(half, 4)
INSTANTIATE_CONV_UPDATE_SLOTS(bfloat16_t, 2)
INSTANTIATE_CONV_UPDATE_SLOTS(bfloat16_t, 3)
INSTANTIATE_CONV_UPDATE_SLOTS(bfloat16_t, 4)

INSTANTIATE_GATING(float, float, "gdn_fused_gating_float")
INSTANTIATE_GATING(half, half, "gdn_fused_gating_half")
INSTANTIATE_GATING(bfloat16_t, bfloat16_t, "gdn_fused_gating_bfloat16_t")
INSTANTIATE_GATING(half, float, "gdn_fused_gating_half_alog_f32")
INSTANTIATE_GATING(bfloat16_t, float, "gdn_fused_gating_bfloat16_t_alog_f32")

INSTANTIATE_L2(float)
INSTANTIATE_L2(half)
INSTANTIATE_L2(bfloat16_t)

INSTANTIATE_RMSNORM(float, float, "gdn_gated_rmsnorm_silu_mul_float")
INSTANTIATE_RMSNORM(half, half, "gdn_gated_rmsnorm_silu_mul_half")
INSTANTIATE_RMSNORM(bfloat16_t, bfloat16_t, "gdn_gated_rmsnorm_silu_mul_bfloat16_t")
INSTANTIATE_RMSNORM(half, float, "gdn_gated_rmsnorm_silu_mul_half_wf32")
INSTANTIATE_RMSNORM(bfloat16_t, float, "gdn_gated_rmsnorm_silu_mul_bfloat16_t_wf32")

INSTANTIATE_RECURRENCE(float, 64)
INSTANTIATE_RECURRENCE(float, 128)
INSTANTIATE_RECURRENCE(half, 64)
INSTANTIATE_RECURRENCE(half, 128)
INSTANTIATE_RECURRENCE(bfloat16_t, 64)
INSTANTIATE_RECURRENCE(bfloat16_t, 128)

INSTANTIATE_DECODE(float, 64)
INSTANTIATE_DECODE(float, 128)
INSTANTIATE_DECODE(half, 64)
INSTANTIATE_DECODE(half, 128)
INSTANTIATE_DECODE(bfloat16_t, 64)
INSTANTIATE_DECODE(bfloat16_t, 128)

INSTANTIATE_VARLEN(float, 64)
INSTANTIATE_VARLEN(float, 128)
INSTANTIATE_VARLEN(half, 64)
INSTANTIATE_VARLEN(half, 128)
INSTANTIATE_VARLEN(bfloat16_t, 64)
INSTANTIATE_VARLEN(bfloat16_t, 128)

INSTANTIATE_SCATTER(float)
INSTANTIATE_SCATTER(half)
INSTANTIATE_SCATTER(bfloat16_t)
