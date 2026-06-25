#pragma once
#include <cstdint>

namespace attention_rs {
namespace e8m0 {

// Decode E8M0 byte to float. 0xFF → NaN (quiet NaN).
__device__ __forceinline__ float to_float(uint8_t e) {
    return (e == 0xFF)
        ? __uint_as_float(0x7FC00000u)  // quiet NaN
        : __uint_as_float((unsigned int)e << 23);
}

// Scale reader: reads a scale value and returns float.
// Specialized for float (identity) and uint8_t (e8m0 decode).
template <typename ScaleT>
__device__ __forceinline__ float read_scale(const ScaleT* ptr, int idx);

template <>
__device__ __forceinline__ float read_scale<float>(const float* ptr, int idx) {
    return __ldg(&ptr[idx]);
}

template <>
__device__ __forceinline__ float read_scale<uint8_t>(const uint8_t* ptr, int idx) {
    return to_float(__ldg(&ptr[idx]));
}

} // namespace e8m0
} // namespace attention_rs
