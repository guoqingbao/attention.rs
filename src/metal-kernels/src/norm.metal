#include "metal_dtype.metal"
#include <metal_stdlib>

using namespace metal;

template <typename T>
inline float norm_to_float(T value) {
    return static_cast<float>(value);
}

template <>
inline float norm_to_float<float>(float value) {
    return value;
}

template <typename T>
kernel void add_rms_norm_kernel(
    const device T* x [[buffer(0)]],
    const device T* residual [[buffer(1)]],
    const device float* weight [[buffer(2)]],
    device T* output [[buffer(3)]],
    device T* sum_output [[buffer(4)]],
    constant int& rows [[buffer(5)]],
    constant int& dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]],
    uint3 row3 [[threadgroup_position_in_grid]]) {
    const uint tid = tid3.x;
    const uint row = row3.x;
    if (row >= static_cast<uint>(rows)) {
        return;
    }
    const uint threads = threads3.x;
    threadgroup float partial[256];
    const uint base = row * static_cast<uint>(dim);
    float sumsq = 0.0f;
    for (uint i = tid; i < static_cast<uint>(dim); i += threads) {
        const T sum = T(norm_to_float(x[base + i]) + norm_to_float(residual[base + i]));
        sum_output[base + i] = sum;
        const float value = norm_to_float(sum);
        sumsq += value * value;
    }
    partial[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(partial[0] / static_cast<float>(dim) + eps);
    for (uint i = tid; i < static_cast<uint>(dim); i += threads) {
        output[base + i] = T(norm_to_float(sum_output[base + i]) * scale * weight[i]);
    }
}

#define INSTANTIATE_ADD_RMS_NORM(type, name) \
    template [[host_name("attention_add_rms_norm_" #name)]] [[kernel]] \
    void add_rms_norm_kernel<type>( \
        const device type* x [[buffer(0)]], \
        const device type* residual [[buffer(1)]], \
        const device float* weight [[buffer(2)]], \
        device type* output [[buffer(3)]], \
        device type* sum_output [[buffer(4)]], \
        constant int& rows [[buffer(5)]], \
        constant int& dim [[buffer(6)]], \
        constant float& eps [[buffer(7)]], \
        uint3 tid3 [[thread_position_in_threadgroup]], \
        uint3 threads3 [[threads_per_threadgroup]], \
        uint3 row3 [[threadgroup_position_in_grid]]);

INSTANTIATE_ADD_RMS_NORM(float, float)
INSTANTIATE_ADD_RMS_NORM(half, half)
#if defined(__HAVE_BFLOAT__)
INSTANTIATE_ADD_RMS_NORM(bfloat, bfloat16_t)
#endif
