#pragma once

#ifndef USE_ROCM
  #include <cub/cub.cuh>
  #if CUB_VERSION >= 200800 && __has_include(<cuda/functional>)
    #include <cuda/functional>
    #include <cuda/std/functional>
using CubAddOp = cuda::std::plus<>;
using CubMaxOp = cuda::maximum<>;
  #elif CUB_VERSION >= 200800 && __has_include(<cuda/std/functional>)
    #include <cuda/std/functional>
using CubAddOp = cuda::std::plus<>;
using CubMaxOp = cub::Max;
  #else   // if CUB_VERSION < 200800
using CubAddOp = cub::Sum;
using CubMaxOp = cub::Max;
  #endif  // CUB_VERSION
#else
  #include <hipcub/hipcub.hpp>
namespace cub = hipcub;
using CubAddOp = hipcub::Sum;
using CubMaxOp = hipcub::Max;
#endif  // USE_ROCM
