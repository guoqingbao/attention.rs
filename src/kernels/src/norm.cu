// Fused RMSNorm kernels.
//
// The generic path in Candle requires matching dtypes for x and weight, so
// xinfer's compressed-tensors/NVFP4 models run: cast BF16 -> F32, rmsnorm_f32,
// cast F32 -> BF16 (three launches per norm).  These kernels keep the
// reduction and the weight in F32 while reading/writing activations in their
// native dtype, matching the F32 reduction numerics in a single launch.
//
// A fused residual-add variant also writes the BF16-rounded residual sum,
// replacing a separate broadcast add kernel.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define NORM_CHECK_CUDA(x)                                                      \
    do {                                                                        \
        cudaError_t err__ = (x);                                                \
        if (err__ != cudaSuccess) {                                             \
            printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(err__));                                  \
        }                                                                       \
    } while (0)

namespace {

template <typename T>
__device__ __forceinline__ float nrm_to_float(T x);

template <>
__device__ __forceinline__ float nrm_to_float<float>(float x) {
    return x;
}
template <>
__device__ __forceinline__ float nrm_to_float<__half>(__half x) {
    return __half2float(x);
}
template <>
__device__ __forceinline__ float nrm_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T nrm_from_float(float x);

template <>
__device__ __forceinline__ float nrm_from_float<float>(float x) {
    return x;
}
template <>
__device__ __forceinline__ __half nrm_from_float<__half>(float x) {
    return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 nrm_from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <typename T, int VEC>
union NrmVec {
    uint4 raw;
    T val[VEC];
};

template <typename T, int THREADS>
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[THREADS / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (threadIdx.x < THREADS / 32) ? shared[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
    }
    __shared__ float result;
    if (threadIdx.x == 0) result = val;
    __syncthreads();
    return result;
}

template <typename T, int THREADS, int VEC, bool ADD>
__global__ void rms_norm_kernel(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    const float* __restrict__ weight,
    T* __restrict__ out,
    T* __restrict__ residual_out,
    int rows,
    int dim,
    float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = static_cast<size_t>(row) * dim;
    const T* xr = x + base;
    T* orow = out + base;

    // Vector accesses need both an aligned row stride and full vectors.
    const bool vec_ok = (dim % VEC) == 0;
    float sum_sq = 0.0f;
    if (vec_ok) {
    for (int i = threadIdx.x * VEC; i < dim; i += THREADS * VEC) {
        if (ADD) {
            NrmVec<T, VEC> a;
            NrmVec<T, VEC> b;
            a.raw = *reinterpret_cast<const uint4*>(xr + i);
            b.raw = *reinterpret_cast<const uint4*>(residual + base + i);
            NrmVec<T, VEC> s;
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const float v = nrm_to_float(a.val[j]) + nrm_to_float(b.val[j]);
                s.val[j] = nrm_from_float<T>(v);
                // Use the value that was actually stored (T-rounded) for the
                // statistics, matching add-then-cast reference semantics.
                const float sv = nrm_to_float(s.val[j]);
                sum_sq += sv * sv;
            }
            *reinterpret_cast<uint4*>(residual_out + base + i) = s.raw;
        } else {
            NrmVec<T, VEC> a;
            a.raw = *reinterpret_cast<const uint4*>(xr + i);
#pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const float v = nrm_to_float(a.val[j]);
                sum_sq += v * v;
            }
        }
    }
    }
    // When the row stride is not vector aligned, everything is handled here;
    // otherwise this covers the (normally empty) scalar tail.
    const int tail_start = vec_ok ? dim : 0;
    for (int i = tail_start + threadIdx.x; i < dim; i += THREADS) {
        float v = nrm_to_float(xr[i]);
        if (ADD) {
            const T s = nrm_from_float<T>(v + nrm_to_float(residual[base + i]));
            residual_out[base + i] = s;
            v = nrm_to_float(s);
        }
        sum_sq += v * v;
    }

    const float total = block_reduce_sum<T, THREADS>(sum_sq);
    const float rstd = rsqrtf(total / static_cast<float>(dim) + eps);

    const T* src = ADD ? residual_out + base : xr;
    if (vec_ok) {
    for (int i = threadIdx.x * VEC; i < dim; i += THREADS * VEC) {
        NrmVec<T, VEC> a;
        a.raw = *reinterpret_cast<const uint4*>(src + i);
        const float4* w0 = reinterpret_cast<const float4*>(weight + i);
        NrmVec<T, VEC> y;
        if constexpr (VEC == 8) {
            const float4 w1 = w0[0];
            const float4 w2 = w0[1];
            y.val[0] = nrm_from_float<T>(nrm_to_float(a.val[0]) * rstd * w1.x);
            y.val[1] = nrm_from_float<T>(nrm_to_float(a.val[1]) * rstd * w1.y);
            y.val[2] = nrm_from_float<T>(nrm_to_float(a.val[2]) * rstd * w1.z);
            y.val[3] = nrm_from_float<T>(nrm_to_float(a.val[3]) * rstd * w1.w);
            y.val[4] = nrm_from_float<T>(nrm_to_float(a.val[4]) * rstd * w2.x);
            y.val[5] = nrm_from_float<T>(nrm_to_float(a.val[5]) * rstd * w2.y);
            y.val[6] = nrm_from_float<T>(nrm_to_float(a.val[6]) * rstd * w2.z);
            y.val[7] = nrm_from_float<T>(nrm_to_float(a.val[7]) * rstd * w2.w);
        } else {
            const float4 w1 = w0[0];
            y.val[0] = nrm_from_float<T>(nrm_to_float(a.val[0]) * rstd * w1.x);
            y.val[1] = nrm_from_float<T>(nrm_to_float(a.val[1]) * rstd * w1.y);
            y.val[2] = nrm_from_float<T>(nrm_to_float(a.val[2]) * rstd * w1.z);
            y.val[3] = nrm_from_float<T>(nrm_to_float(a.val[3]) * rstd * w1.w);
        }
        *reinterpret_cast<uint4*>(orow + i) = y.raw;
    }
    }
    for (int i = tail_start + threadIdx.x; i < dim; i += THREADS) {
        const float v = nrm_to_float(src[i]) * rstd * weight[i];
        orow[i] = nrm_from_float<T>(v);
    }
}

template <typename T, int VEC>
static void launch_rms_norm(
    const T* x,
    const T* residual,
    const float* weight,
    T* out,
    T* residual_out,
    int rows,
    int dim,
    float eps,
    bool add,
    cudaStream_t stream) {
    constexpr int THREADS = 256;
    if (rows <= 0 || dim <= 0) return;
    if (add) {
        rms_norm_kernel<T, THREADS, VEC, true><<<rows, THREADS, 0, stream>>>(
            x, residual, weight, out, residual_out, rows, dim, eps);
    } else {
        rms_norm_kernel<T, THREADS, VEC, false><<<rows, THREADS, 0, stream>>>(
            x, residual, weight, out, residual_out, rows, dim, eps);
    }
    NORM_CHECK_CUDA(cudaGetLastError());
}

}  // namespace

extern "C" void attention_rms_norm_f32(
    const float* x, const float* weight, float* out, int rows, int dim, float eps,
    cudaStream_t stream) {
    launch_rms_norm<float, 4>(x, nullptr, weight, out, nullptr, rows, dim, eps, false, stream);
}

extern "C" void attention_rms_norm_f16(
    const __half* x, const float* weight, __half* out, int rows, int dim, float eps,
    cudaStream_t stream) {
    launch_rms_norm<__half, 8>(x, nullptr, weight, out, nullptr, rows, dim, eps, false, stream);
}

extern "C" void attention_rms_norm_bf16(
    const __nv_bfloat16* x, const float* weight, __nv_bfloat16* out, int rows, int dim,
    float eps, cudaStream_t stream) {
    launch_rms_norm<__nv_bfloat16, 8>(
        x, nullptr, weight, out, nullptr, rows, dim, eps, false, stream);
}

extern "C" void attention_add_rms_norm_f32(
    const float* x, const float* residual, const float* weight, float* out,
    float* residual_out, int rows, int dim, float eps, cudaStream_t stream) {
    launch_rms_norm<float, 4>(
        x, residual, weight, out, residual_out, rows, dim, eps, true, stream);
}

extern "C" void attention_add_rms_norm_f16(
    const __half* x, const __half* residual, const float* weight, __half* out,
    __half* residual_out, int rows, int dim, float eps, cudaStream_t stream) {
    launch_rms_norm<__half, 8>(
        x, residual, weight, out, residual_out, rows, dim, eps, true, stream);
}

extern "C" void attention_add_rms_norm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* residual, const float* weight,
    __nv_bfloat16* out, __nv_bfloat16* residual_out, int rows, int dim, float eps,
    cudaStream_t stream) {
    launch_rms_norm<__nv_bfloat16, 8>(
        x, residual, weight, out, residual_out, rows, dim, eps, true, stream);
}
