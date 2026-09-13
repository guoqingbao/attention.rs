#pragma once
// FlashInfer TRT-LLM quantization.cu uses `cuda::maximum<>` without including it.
//
// CUDA 13: nvcc usually finds <cuda/functional> and CUB pulls the same header.
// CUDA 12.8: libcudacxx lives under include/cccl, which is often *not* on the
// nvcc include path when Cutlass's bundled CUB is used. That CUB no longer
// injects `cuda::maximum`, so SM120 builds fail unless we provide it.
#if __has_include(<cuda/functional>)
#include <cuda/functional>
#endif

#ifndef _CUDA_FUNCTIONAL_MAXIMUM_H
#define _CUDA_FUNCTIONAL_MAXIMUM_H
namespace cuda {
template <class T = void>
struct maximum {
  __host__ __device__ constexpr T operator()(T const& a, T const& b) const {
    return a < b ? b : a;
  }
};

template <>
struct maximum<void> {
  template <class T1, class T2>
  __host__ __device__ constexpr auto operator()(T1 const& a, T2 const& b) const
      -> decltype(a < b ? b : a) {
    return a < b ? b : a;
  }
};
}  // namespace cuda
#endif

// FlashInfer fastdiv.cuh (commit 2bfb9334+) uses cuda::fast_mod_div<uint32_t>
// from CCCL. On CUDA 13+ (CCCL 3.0) this type is always on the include path
// via <cuda/std/...>, so a polyfill would create an ambiguous overload.
// On CUDA 12.x (CCCL 2.x) the type is internal and not found when CUTLASS's
// bundled CUB shadows the system CCCL headers.
//
// CCCL timeline:
//   CUDA 12.0 (CCCL 2.0): cuda::fast_mod_div introduced as internal type
//   CUDA 13.0 (CCCL 3.0): promoted to public <cuda/std> header
#if __CUDACC_VER_MAJOR__ < 13
namespace cuda {
template <typename T>
struct fast_mod_div {
  T divisor;
  __host__ __device__ explicit fast_mod_div(T d) : divisor(d ? d : 1) {}
};
template <typename T>
__host__ __device__ inline T operator/(T n, const fast_mod_div<T>& f) {
  return n / f.divisor;
}
template <typename T>
__host__ __device__ inline T operator%(T n, const fast_mod_div<T>& f) {
  return n % f.divisor;
}
}  // namespace cuda
#endif  // __CUDACC_VER_MAJOR__ < 13
