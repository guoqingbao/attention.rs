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
