#pragma once
#ifndef NO_HARDWARE_FP8
  #include "cuda_fp8.h"
#endif
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <type_traits>

#include <cstring>
#include <cassert>

#include "attention_generic.cuh"
#include "dtype_bfloat16.cuh"
#include <stdio.h>

#include <cmath>
#include <limits>
#include <cuda_fp16.h>

namespace vllm {

template<>
struct Vec<uint8_t, 1> {
  using Type = uint8_t;
};
template<>
struct Vec<uint8_t, 2> {
  using Type = uint16_t;
};
template<>
struct Vec<uint8_t, 4> {
  using Type = uint32_t;
};
template<>
struct Vec<uint8_t, 8> {
  using Type = uint2;
};


namespace fp8 {

#define MIN_FP8_VALUE (-448.0f) // min finite fp8 E4M3FN
#define MAX_FP8_VALUE (448.0f)  // max finite fp8 E4M3FN

  // --- softmax FP8 (E4M3) software fast conversions on GPU Arch < 900 ---
// Author: Guoqing Bao,
// Part of the vllm.rs project
static inline __device__ uint8_t softmax_float_to_fp8_e4m3(float f) {
  // bit-level access
  uint32_t bits = __float_as_uint(f);
  int sign = (bits >> 31) & 1;
  int exp = (bits >> 23) & 0xFF;
  uint32_t mant = bits & 0x7FFFFF;

  // special cases
  if (exp == 0xFF) { // Inf or NaN
    if (mant != 0) {
      // NaN: exp=1111, mant=111 (E4M3FN reserves only this for NaN)
      return (uint8_t)((0 << 7) | (0xF << 3) | 0x7);
    } else {
      // Inf -> saturate to max finite
      return (uint8_t)((sign << 7) | (0xF << 3) | 0x6);
    }
  }

  // zero
  if (exp == 0 && mant == 0) {
    return (uint8_t)(sign << 7);
  }

  // normalized float: compute new exponent with bias differences
  const int FP32_BIAS = 127;
  const int FP8_BIAS = 7;
  int new_exp = exp - FP32_BIAS + FP8_BIAS;

  // Prepare mantissa including hidden 1 for normals
  uint32_t mant_with_hidden = mant;
  if (exp != 0) {
    mant_with_hidden |= (1u << 23);
  } else {
    // subnormal float -- handled below
  }

  // Overflow -> saturate to max finite (E4M3FN)
  if (new_exp > 0xF) {
    return (uint8_t)((sign << 7) | (0xF << 3) | 0x6);
  }

  // Underflow -> could become subnormal or zero in FP8
  if (new_exp <= 0) {
    // shift needed to form subnormal in FP8
    // effective shift: number of bits to right-shift mantissa to align to 3-bit mantissa
    int shift = (1 - new_exp) + (23 - 3); // (1-new_exp) accounts for denorm scaling
    if (shift >= 32) {
      // too small -> underflow to zero
      return (uint8_t)(sign << 7);
    }
    // build a rounding window
    uint32_t abs_mant = mant_with_hidden;
    // Round-to-nearest-even for subnormal:
    uint32_t truncated = abs_mant >> shift;
    uint32_t remainder = abs_mant & ((1u << shift) - 1u);

    // rounding decision: look at the highest discarded bit and lower bits
    uint32_t half = 1u << (shift - 1);
    if ( (remainder > half) || (remainder == half && (truncated & 1u)) ) {
      truncated += 1u;
    }

    // truncated now contains the 3-bit (or fewer) significand; clamp to 3 bits
    uint32_t mant8 = truncated & 0x7u;
    if (mant8 == 0) {
      // rounds to zero
      return (uint8_t)(sign << 7);
    } else {
      // exp = 0 (subnormal)
      return (uint8_t)((sign << 7) | (0 << 3) | (mant8));
    }
  }

  // Normal case: need to round mantissa to 3 bits
  // shift mantissa (23 -> 3): keep top 3 bits of mant_with_hidden (which has 24 bits)
  int shift_norm = 23 - 3;
  uint32_t truncated = (mant_with_hidden >> shift_norm) & 0x7u; // 3 bits
  // gather bits used for rounding
  uint32_t round_bit_pos = shift_norm - 1;
  uint32_t round_bit = (mant_with_hidden >> round_bit_pos) & 1u;
  uint32_t sticky_mask = (1u << round_bit_pos) - 1u;
  uint32_t sticky = (mant_with_hidden & sticky_mask) ? 1u : 0u;

  // Round-to-nearest-even
  if (round_bit && (sticky || (truncated & 1u))) {
    truncated += 1u;
    if (truncated == (1u << 3)) {
      // mantissa overflow -> increment exponent
      truncated = 0;
      new_exp += 1;
      if (new_exp > 0xF) {
        // overflow to max finite
        return (uint8_t)((sign << 7) | (0xF << 3) | 0x6);
      }
    }
  }

  // exp=1111 with mant=111 is NaN in E4M3FN, clamp to max finite instead
  if (new_exp == 0xF && truncated == 0x7) {
    truncated = 0x6;
  }

  uint8_t out = (uint8_t)((sign << 7) | ((new_exp & 0xF) << 3) | (truncated & 0x7));
  return out;
}

// Author: Guoqing Bao,
// Part of the vllm.rs project
static inline __device__ float softmax_fp8_to_float_e4m3(uint8_t x) {
  int sign = (x >> 7) & 1;
  int exp = (x >> 3) & 0xF;
  int mant = x & 0x7;

  const int FP32_BIAS = 127;
  const int FP8_BIAS = 7;

  if (exp == 0) {
    if (mant == 0) {
      // zero
      uint32_t bits = (sign << 31);
      return __uint_as_float(bits);
    } else {
      // subnormal: value = (-1)^s * 2^(1-bias8 - (mantissa bits)) * (mant / 2^3)
      // We'll reconstruct as float by shifting mant into float mantissa position
      // compute exponent for float
      int e = (1 - FP8_BIAS) + FP32_BIAS; // unbiased exponent + fp32 bias
      uint32_t mant32 = (uint32_t)mant << (23 - 3);
      uint32_t bits = (sign << 31) | ((uint32_t)e << 23) | mant32;
      return __uint_as_float(bits);
    }
  } else if (exp == 0xF && mant == 0x7) {
    // NaN - produce a quiet NaN
    uint32_t bits = (0u << 31) | (0xFFu << 23) | (1u << 22);
    return __uint_as_float(bits);
  } else {
    int new_exp = exp - FP8_BIAS + FP32_BIAS;
    uint32_t mant32 = (uint32_t)mant << (23 - 3);
    uint32_t bits = (sign << 31) | ((uint32_t)new_exp << 23) | mant32;
    return __uint_as_float(bits);
  }
}

// --- Dispatch wrappers that prefer NV intrinsics when available ---

  // If native fp8 intrinsics exist, use them.
  static inline __device__ uint8_t dispatch_float_to_fp8(float f) {
    // use NV intrinsic that returns __nv_fp8_storage_t (uint8-ish)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && !defined(NO_HARDWARE_FP8)
    __nv_fp8_storage_t r = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
    return (uint8_t)r;
#else
    return softmax_float_to_fp8_e4m3(f);
#endif
  }
  static inline __device__ float dispatch_fp8_to_float(uint8_t a) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && !defined(NO_HARDWARE_FP8)
    return __half2float(__nv_cvt_fp8_to_halfraw(a, __NV_E4M3));
#else
    return softmax_fp8_to_float_e4m3(a);
#endif
  }


template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(
    const Tin& x, const float scale) {
  return x;
}

// fp8 -> half
template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, uint8_t>(
    const uint8_t& a, const float scale) {
  float f = dispatch_fp8_to_float(a);
  return __float2half(f * scale);
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, uint16_t>(
    const uint16_t& a, const float scale) {
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;
  uint8_t b0 = (uint8_t)(a & 0xFFu);
  uint8_t b1 = (uint8_t)((a >> 8u) & 0xFFu);
  tmp.u16[0] = __float2half(dispatch_fp8_to_float(b0) * scale);
  tmp.u16[1] = __float2half(dispatch_fp8_to_float(b1) * scale);
  return tmp.u32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint32_t>(
    const uint32_t& a, const float scale) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a & 0xFFFFu), scale);
  tmp.u32[1] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)((a >> 16u) & 0xFFFFu), scale);
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4 scaled_vec_conversion<uint4, uint2>(
    const uint2& a, const float scale) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale);
  tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale);
  return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16 scaled_vec_conversion<__nv_bfloat16, uint8_t>(
    const uint8_t& a, const float scale) {
  float f = dispatch_fp8_to_float(a);
  return __float2bfloat16(f * scale);
}

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162 scaled_vec_conversion<__nv_bfloat162, uint16_t>(
    const uint16_t& a, const float scale) {
  __nv_bfloat162 res;
  uint8_t b0 = (uint8_t)(a & 0xFFu);
  uint8_t b1 = (uint8_t)((a >> 8u) & 0xFFu);
  res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>(b0, scale);
  res.y = scaled_vec_conversion<__nv_bfloat16, uint8_t>(b1, scale);
  return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t scaled_vec_conversion<bf16_4_t, uint32_t>(
    const uint32_t& a, const float scale) {
  bf16_4_t res;
  res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a & 0xFFFFu), scale);
  res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)((a >> 16u) & 0xFFFFu), scale);
  return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t scaled_vec_conversion<bf16_8_t, uint2>(
    const uint2& a, const float scale) {
  bf16_4_t tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale);
  bf16_4_t tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// fp8 -> float
template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
    const uint8_t& a, const float scale) {
  return dispatch_fp8_to_float(a) * scale;
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2 scaled_vec_conversion<float2, uint16_t>(
    const uint16_t& a, const float scale) {
  uint8_t b0 = (uint8_t)(a & 0xFFu);
  uint8_t b1 = (uint8_t)((a >> 8u) & 0xFFu);
  return make_float2(dispatch_fp8_to_float(b0) * scale,
                     dispatch_fp8_to_float(b1) * scale);
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_ scaled_vec_conversion<Float4_, uint32_t>(
    const uint32_t& a, const float scale) {
  uint8_t b0 = (uint8_t)(a & 0xFFu);
  uint8_t b1 = (uint8_t)((a >> 8u) & 0xFFu);
  uint8_t b2 = (uint8_t)((a >> 16u) & 0xFFu);
  uint8_t b3 = (uint8_t)((a >> 24u) & 0xFFu);
  Float4_ res;
  res.x = make_float2(dispatch_fp8_to_float(b0) * scale,
                      dispatch_fp8_to_float(b1) * scale);
  res.y = make_float2(dispatch_fp8_to_float(b2) * scale,
                      dispatch_fp8_to_float(b3) * scale);
  return res;
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_ scaled_vec_conversion<Float8_, uint2>(
    const uint2& a, const float scale) {
  Float4_ tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale);
  Float4_ tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// half -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, uint16_t>(
    const uint16_t& a, const float scale) {
  // float f = fminf(fmaxf(__half2float(a) / scale, MIN_FP8_VALUE), MAX_FP8_VALUE);
  float f = __half2float(a) / scale;
  return dispatch_float_to_fp8(f);
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, const float scale) {
  // float f = fminf(fmaxf(__bfloat162float(a) / scale, MIN_FP8_VALUE), MAX_FP8_VALUE);
  float f = __bfloat162float(a) / scale;
  return dispatch_float_to_fp8(f);
}

// float -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(
    const float& a, const float scale) {
  // float f = fminf(fmaxf(a / scale, MIN_FP8_VALUE), MAX_FP8_VALUE);
  float f = a / scale;
  return dispatch_float_to_fp8(f);
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4 scaled_vec_conversion<float4, uint32_t>(
    const uint32_t& a, const float scale) {
  Float4_ tmp = scaled_vec_conversion<Float4_, uint32_t>(a, scale);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
    return scaled_vec_conversion<Tout, Tin>(x, scale);
}

}  // namespace fp8

}  // namespace vllm
