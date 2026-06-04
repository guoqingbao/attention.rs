/*
 * MXFP4 activation quantization kernel.
 * Quantizes BF16/FP16 activations to packed FP4 E2M1 with E8M0 block scales.
 * Block size = 32 (MXFP4 / OCP Microscaling standard).
 *
 * Key differences from NVFP4 quantization:
 *   - Block size: 32 elements (vs 16 for NVFP4)
 *   - Scale format: E8M0 (8-bit exponent-only, bias 127) instead of FP8 E4M3
 *   - No global scale factor
 *
 * Required for hardware MXFP4 GEMM path on Blackwell (SM100+).
 * The CUTLASS Mxf8f6f4 block-scaled tensor ops expect both A and B in FP4 format
 * with E8M0 block scales.
 */

#ifdef ENABLE_FP4

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

static constexpr int MXFP4_BLOCK_SIZE = 32;

// FP4 E2M1 quantization: maps float values to 4-bit E2M1 codes
// E2M1 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
__device__ __forceinline__ uint8_t mxfp4_float_to_fp4_e2m1(float val) {
  float abs_val = fabsf(val);
  uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;

  uint8_t code;
  if (abs_val < 0.25f) {
    code = 0x0;  // 0.0
  } else if (abs_val < 0.75f) {
    code = 0x1;  // 0.5
  } else if (abs_val < 1.25f) {
    code = 0x2;  // 1.0
  } else if (abs_val < 1.75f) {
    code = 0x3;  // 1.5
  } else if (abs_val < 2.5f) {
    code = 0x4;  // 2.0
  } else if (abs_val < 3.5f) {
    code = 0x5;  // 3.0
  } else if (abs_val < 5.0f) {
    code = 0x6;  // 4.0
  } else {
    code = 0x7;  // 6.0
  }

  return sign | code;
}

// Convert float to E8M0 scale format.
// E8M0 is an 8-bit exponent-only format with bias 127.
// The scale is the power-of-2 that best represents the max abs value in the block.
// E8M0 value = 2^(e - 127), stored as uint8_t e.
__device__ __forceinline__ uint8_t float_to_e8m0(float val) {
  if (val <= 0.0f) return 0;
  uint32_t bits = __float_as_uint(val);
  uint8_t biased_exp = (bits >> 23) & 0xFF;
  // Round up if mantissa is nonzero (ceil to next power of 2)
  if (bits & 0x7FFFFF) {
    if (biased_exp < 254) biased_exp++;
  }
  return biased_exp;
}

// Convert E8M0 to float: 2^(e - 127)
__device__ __forceinline__ float e8m0_to_float(uint8_t e) {
  return __uint_as_float((uint32_t)e << 23);
}

// Quantize a block of 32 floats to FP4 E2M1 + compute E8M0 block scale
__device__ __forceinline__ uint8_t mxfp4_quantize_block(
    const float* vals, uint8_t* packed_out, int valid_count)
{
  float amax = 0.0f;
  for (int i = 0; i < valid_count; i++) {
    amax = fmaxf(amax, fabsf(vals[i]));
  }

  // E8M0 scale: power-of-2 that covers amax / 6.0 (max FP4 E2M1 value)
  float scale_val = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
  uint8_t e8m0_scale = float_to_e8m0(scale_val);
  float scale = e8m0_to_float(e8m0_scale);
  float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

  uint8_t codes[32];
  for (int i = 0; i < valid_count; i++) {
    codes[i] = mxfp4_float_to_fp4_e2m1(vals[i] * inv_scale);
  }
  for (int i = valid_count; i < 32; i++) {
    codes[i] = 0;
  }

  // Pack pairs of 4-bit values into bytes (low nibble first)
  for (int i = 0; i < 16; i++) {
    packed_out[i] = (codes[2 * i + 1] << 4) | codes[2 * i];
  }

  return e8m0_scale;
}

// ============================================================================
// Activation quantization kernel: BF16/F16 -> packed FP4 + E8M0 block scales
// ============================================================================

template <typename InType>
__global__ void mxfp4_quantize_activation_kernel(
    const InType* __restrict__ input,   // [M, K]
    uint8_t* __restrict__ output,       // [M, K/2] packed FP4
    uint8_t* __restrict__ scales,       // [M, K/32] E8M0 block scales
    int M, int K)
{
  int row = blockIdx.x;
  int block_idx = threadIdx.x;
  int num_blocks = K / MXFP4_BLOCK_SIZE;

  if (row >= M || block_idx >= num_blocks) return;

  int k_start = block_idx * MXFP4_BLOCK_SIZE;

  float vals[32];
  for (int i = 0; i < MXFP4_BLOCK_SIZE; i++) {
    int k_idx = k_start + i;
    if (k_idx < K) {
      vals[i] = static_cast<float>(input[row * K + k_idx]);
    } else {
      vals[i] = 0.0f;
    }
  }

  uint8_t packed[16];
  int valid = min(MXFP4_BLOCK_SIZE, K - k_start);
  uint8_t block_scale = mxfp4_quantize_block(vals, packed, valid);

  int out_offset = row * (K / 2) + k_start / 2;
  for (int i = 0; i < 16; i++) {
    output[out_offset + i] = packed[i];
  }

  scales[row * num_blocks + block_idx] = block_scale;
}

// ============================================================================
// Scale factor swizzling kernel for CUTLASS block-scaled layout
// Same 128x4 swizzled layout as NVFP4, but scale columns = K/32 (not K/16)
// ============================================================================

__global__ void mxfp4_swizzle_scales_kernel(
    const uint8_t* __restrict__ linear_scales,
    uint8_t* __restrict__ swizzled_scales,
    int rows, int cols,
    int rows_padded, int cols_padded)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows_padded * cols_padded;
  if (idx >= total) return;

  int dst_row = idx / cols_padded;
  int dst_col = idx % cols_padded;

  // CUTLASS 128x4 swizzled layout inverse mapping
  int block_128 = dst_row / 128;
  int within_128 = dst_row % 128;
  int src_within = (within_128 % 32) * 4 + (within_128 / 32);
  int src_row = block_128 * 128 + src_within;

  uint8_t val = 0;
  if (src_row < rows && dst_col < cols) {
    val = linear_scales[src_row * cols + dst_col];
  }
  swizzled_scales[idx] = val;
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

void mxfp4_quantize_activation_f16(
    const void* input,      // [M, K] FP16
    void* output,           // [M, K/2] packed FP4 uint8
    void* scales,           // [M, K/32] E8M0 block scales
    void* swizzled_scales,  // [M_padded, K_scale_padded] swizzled scales
    int M, int K,
    int M_padded, int K_scale_padded,
    int64_t stream)
{
  int num_blocks_k = K / MXFP4_BLOCK_SIZE;
  dim3 grid(M);
  dim3 block(num_blocks_k);

  mxfp4_quantize_activation_kernel<half><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const half*>(input),
      static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(scales),
      M, K);

  int total_swizzled = M_padded * K_scale_padded;
  int threads = 256;
  int blocks_sw = (total_swizzled + threads - 1) / threads;
  mxfp4_swizzle_scales_kernel<<<blocks_sw, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(scales),
      static_cast<uint8_t*>(swizzled_scales),
      M, num_blocks_k,
      M_padded, K_scale_padded);
}

void mxfp4_quantize_activation_bf16(
    const void* input,
    void* output,
    void* scales,
    void* swizzled_scales,
    int M, int K,
    int M_padded, int K_scale_padded,
    int64_t stream)
{
  int num_blocks_k = K / MXFP4_BLOCK_SIZE;
  dim3 grid(M);
  dim3 block(num_blocks_k);

  mxfp4_quantize_activation_kernel<nv_bfloat16><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const nv_bfloat16*>(input),
      static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(scales),
      M, K);

  int total_swizzled = M_padded * K_scale_padded;
  int threads = 256;
  int blocks_sw = (total_swizzled + threads - 1) / threads;
  mxfp4_swizzle_scales_kernel<<<blocks_sw, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(scales),
      static_cast<uint8_t*>(swizzled_scales),
      M, num_blocks_k,
      M_padded, K_scale_padded);
}

void mxfp4_swizzle_weight_scales_e8m0(
    const void* linear_scales,
    void* swizzled_scales,
    int rows, int cols,
    int rows_padded, int cols_padded,
    int64_t stream)
{
  int total = rows_padded * cols_padded;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  mxfp4_swizzle_scales_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(linear_scales),
      static_cast<uint8_t*>(swizzled_scales),
      rows, cols,
      rows_padded, cols_padded);
}

}  // extern "C"

#else  // !ENABLE_FP4

extern "C" {

void mxfp4_quantize_activation_f16(
    const void*, void*, void*, void*, int, int, int, int, int64_t) {}

void mxfp4_quantize_activation_bf16(
    const void*, void*, void*, void*, int, int, int, int, int64_t) {}

void mxfp4_swizzle_weight_scales_e8m0(
    const void*, void*, int, int, int, int, int64_t) {}

}  // extern "C"

#endif  // ENABLE_FP4
