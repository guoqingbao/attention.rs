#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Fused SiLU-and-Mul kernel for MoE gate/up activation.
//
// Given a contiguous [M, 2*N] tensor (gate_up), computes:
//   out[i, j] = silu(gate_up[i, j]) * gate_up[i, j + N]
// where silu(x) = x / (1 + exp(-x))
//
// This replaces three separate ops: narrow+contiguous (gate), narrow+contiguous (up),
// silu(gate), gate*up — saving 3 kernel launches and 3 intermediate allocations.

__device__ __forceinline__ float silu_f(float x) {
  return x / (1.0f + expf(-x));
}

template <typename T>
__device__ __forceinline__ float to_float_val(T x);

template <>
__device__ __forceinline__ float to_float_val<half>(half x) {
  return __half2float(x);
}

#ifndef NO_BF16_KERNEL
template <>
__device__ __forceinline__ float to_float_val<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
#endif

template <typename T>
__device__ __forceinline__ T from_float_val(float x);

template <>
__device__ __forceinline__ half from_float_val<half>(float x) {
  return __float2half(x);
}

#ifndef NO_BF16_KERNEL
template <>
__device__ __forceinline__ __nv_bfloat16 from_float_val<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
#endif

template <typename T>
__global__ void silu_and_mul_kernel(
    const T *__restrict__ gate_up,  // [total_elems * 2] (gate half then up half)
    T *__restrict__ output,         // [total_elems]
    const int64_t total_elems,
    const int64_t N) {              // half-width of last dim

  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems)
    return;

  // For a [M, 2*N] layout, element idx in the output maps to row = idx / N, col = idx % N
  // gate value is at gate_up[row * 2*N + col]
  // up value is at gate_up[row * 2*N + N + col]
  const int64_t row = idx / N;
  const int64_t col = idx % N;
  const int64_t gate_idx = row * 2 * N + col;
  const int64_t up_idx = gate_idx + N;

  float gate_val = to_float_val(gate_up[gate_idx]);
  float up_val = to_float_val(gate_up[up_idx]);

  float result = silu_f(gate_val) * up_val;
  output[idx] = from_float_val<T>(result);
}

// Vectorized path: process 8 elements per thread using float4 (16 bytes = 8 half/bf16)
template <typename T>
__global__ void silu_and_mul_vec_kernel(
    const T *__restrict__ gate_up,
    T *__restrict__ output,
    const int64_t total_elems,
    const int64_t N) {

  using Vec4 = float4;
  constexpr int VEC = 8; // 8 elements per float4 (each element is 2 bytes)

  const int64_t vec_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t elem_start = vec_idx * VEC;
  if (elem_start >= total_elems)
    return;

  // All VEC elements share the same row since N is assumed to be a multiple of VEC
  const int64_t row = elem_start / N;
  const int64_t col = elem_start % N;
  const int64_t gate_start = row * 2 * N + col;
  const int64_t up_start = gate_start + N;

  Vec4 gate_packed = *reinterpret_cast<const Vec4 *>(&gate_up[gate_start]);
  Vec4 up_packed = *reinterpret_cast<const Vec4 *>(&gate_up[up_start]);

  // Reinterpret as pairs
  using Vec2T = typename std::conditional<std::is_same<T, half>::value, half2, __nv_bfloat162>::type;
  const Vec2T *gate_v2 = reinterpret_cast<const Vec2T *>(&gate_packed);
  const Vec2T *up_v2 = reinterpret_cast<const Vec2T *>(&up_packed);

  float results[VEC];

#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 gf, uf;
    if constexpr (std::is_same<T, half>::value) {
      gf = __half22float2(gate_v2[i]);
      uf = __half22float2(up_v2[i]);
    } else {
#ifndef NO_BF16_KERNEL
      gf = {__bfloat162float(gate_v2[i].x), __bfloat162float(gate_v2[i].y)};
      uf = {__bfloat162float(up_v2[i].x), __bfloat162float(up_v2[i].y)};
#endif
    }
    results[i * 2] = silu_f(gf.x) * uf.x;
    results[i * 2 + 1] = silu_f(gf.y) * uf.y;
  }

  // Pack results back
  Vec2T out_v2[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    if constexpr (std::is_same<T, half>::value) {
      out_v2[i] = __floats2half2_rn(results[i * 2], results[i * 2 + 1]);
    } else {
#ifndef NO_BF16_KERNEL
      out_v2[i].x = __float2bfloat16(results[i * 2]);
      out_v2[i].y = __float2bfloat16(results[i * 2 + 1]);
#endif
    }
  }

  *reinterpret_cast<Vec4 *>(&output[elem_start]) = *reinterpret_cast<const Vec4 *>(out_v2);
}

extern "C" void silu_and_mul_f16(
    const void *gate_up,
    void *output,
    int64_t total_elems,
    int64_t N,
    int64_t stream) {

  if (N % 8 == 0 && total_elems % 8 == 0) {
    constexpr int BLOCK = 256;
    int64_t vec_count = total_elems / 8;
    int64_t grid = (vec_count + BLOCK - 1) / BLOCK;
    silu_and_mul_vec_kernel<half><<<grid, BLOCK, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const half *>(gate_up),
        reinterpret_cast<half *>(output),
        total_elems, N);
  } else {
    constexpr int BLOCK = 256;
    int64_t grid = (total_elems + BLOCK - 1) / BLOCK;
    silu_and_mul_kernel<half><<<grid, BLOCK, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const half *>(gate_up),
        reinterpret_cast<half *>(output),
        total_elems, N);
  }
}

extern "C" void silu_and_mul_bf16(
    const void *gate_up,
    void *output,
    int64_t total_elems,
    int64_t N,
    int64_t stream) {

#ifndef NO_BF16_KERNEL
  if (N % 8 == 0 && total_elems % 8 == 0) {
    constexpr int BLOCK = 256;
    int64_t vec_count = total_elems / 8;
    int64_t grid = (vec_count + BLOCK - 1) / BLOCK;
    silu_and_mul_vec_kernel<__nv_bfloat16><<<grid, BLOCK, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(gate_up),
        reinterpret_cast<__nv_bfloat16 *>(output),
        total_elems, N);
  } else {
    constexpr int BLOCK = 256;
    int64_t grid = (total_elems + BLOCK - 1) / BLOCK;
    silu_and_mul_kernel<__nv_bfloat16><<<grid, BLOCK, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(gate_up),
        reinterpret_cast<__nv_bfloat16 *>(output),
        total_elems, N);
  }
#endif
}
