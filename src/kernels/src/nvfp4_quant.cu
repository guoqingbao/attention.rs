/*
 * NVFP4 activation quantization kernel.
 * Quantizes BF16/FP16 activations to packed FP4 E2M1 with FP8 E4M3 block scales.
 * Block size = 16 (NVFP4 standard).
 *
 * Required for hardware FP4 GEMM path on Blackwell (SM100+).
 * The CUTLASS block-scaled tensor ops expect both A and B in FP4 format.
 */

#ifdef ENABLE_FP4

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

static constexpr int NVFP4_BLOCK_SIZE = 16;

// FP4 E2M1 quantization LUT: maps float values to 4-bit E2M1 codes
// E2M1 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
  // Clamp to FP4 E2M1 range: [-6.0, 6.0]
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

// FP4 E2M1 dequantization LUT
__device__ __constant__ float fp4_e2m1_lut[16] = {
  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Convert float to FP8 E4M3 (for block scales)
__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(val);
  return *reinterpret_cast<uint8_t*>(&fp8_val);
#else
  // Software fallback
  if (val == 0.0f) return 0;
  uint32_t bits = __float_as_uint(val);
  uint32_t sign = (bits >> 31) & 1;
  int exp = ((bits >> 23) & 0xFF) - 127;
  uint32_t mantissa = bits & 0x7FFFFF;

  // Clamp to FP8 E4M3 range
  if (exp > 8) { exp = 8; mantissa = 0x600000; }
  if (exp < -9) return 0;

  int biased_exp = exp + 7;
  if (biased_exp < 0) biased_exp = 0;
  if (biased_exp > 15) biased_exp = 15;

  uint8_t mant3 = (mantissa >> 20) & 0x7;
  return (sign << 7) | (biased_exp << 3) | mant3;
#endif
}

// Convert FP8 E4M3 to float (for dequant verification)
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  __nv_fp8_e4m3 fp8_val = *reinterpret_cast<__nv_fp8_e4m3*>(&val);
  return float(fp8_val);
#else
  if (val == 0) return 0.0f;
  uint32_t sign = (val >> 7) & 1;
  int biased_exp = (val >> 3) & 0xF;
  uint32_t mantissa = val & 0x7;
  int exp = biased_exp - 7;
  float result = (1.0f + mantissa / 8.0f) * powf(2.0f, exp);
  return sign ? -result : result;
#endif
}

// Quantize a block of 16 floats to FP4 E2M1 + compute block scale
// Returns the block scale as FP8 E4M3
//
// input_scale_inv: reciprocal of the per-tensor activation scale from the checkpoint.
//   The block scale is computed as: sf = input_scale_inv * (amax / 6.0)
//   This pre-bakes the input scaling into the block scales so that the CUTLASS GEMM
//   epilogue alpha = (input_scale * weight_global_scale) produces correctly scaled output.
//   When input_scale = 1.0, this has no effect.
__device__ __forceinline__ uint8_t quantize_block_fp4(
    const float* vals, uint8_t* packed_out, int valid_count,
    float input_scale_inv)
{
  float amax = 0.0f;
  for (int i = 0; i < valid_count; i++) {
    amax = fmaxf(amax, fabsf(vals[i]));
  }

  // Block scale = input_scale_inv * (amax / 6.0)
  // Following SGLang/TRT-LLM convention: SFValue = SFScaleVal * (vecMax / max_fp4)
  float raw_scale = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
  float scale = input_scale_inv * raw_scale;
  uint8_t fp8_scale = float_to_fp8_e4m3(scale);

  // Reconstruct quantized scale for accurate inverse
  float quant_scale = fp8_e4m3_to_float(fp8_scale);
  float inv_scale = (quant_scale > 0.0f)
      ? (1.0f / (quant_scale / input_scale_inv))
      : 0.0f;

  uint8_t codes[16];
  for (int i = 0; i < valid_count; i++) {
    codes[i] = float_to_fp4_e2m1(vals[i] * inv_scale);
  }
  for (int i = valid_count; i < 16; i++) {
    codes[i] = 0;
  }

  for (int i = 0; i < 8; i++) {
    packed_out[i] = (codes[2 * i + 1] << 4) | codes[2 * i];
  }

  return fp8_scale;
}

// ============================================================================
// Activation quantization kernel: BF16/F16 -> packed FP4 + FP8 block scales
// ============================================================================

template <typename InType>
__global__ void nvfp4_quantize_activation_kernel(
    const InType* __restrict__ input,   // [M, K]
    uint8_t* __restrict__ output,       // [M, K/2] packed FP4
    uint8_t* __restrict__ scales,       // [M_padded, K/16] FP8 E4M3 block scales
    float input_scale_inv,              // 1.0 / input_scale (pre-baked into block scales)
    int M, int K, int M_padded)
{
  int row = blockIdx.x;
  int block_idx = threadIdx.x;
  int num_blocks = K / NVFP4_BLOCK_SIZE;

  if (row >= M || block_idx >= num_blocks) return;

  int k_start = block_idx * NVFP4_BLOCK_SIZE;

  float vals[16];
  for (int i = 0; i < NVFP4_BLOCK_SIZE; i++) {
    int k_idx = k_start + i;
    if (k_idx < K) {
      vals[i] = static_cast<float>(input[row * K + k_idx]);
    } else {
      vals[i] = 0.0f;
    }
  }

  uint8_t packed[8];
  uint8_t block_scale = quantize_block_fp4(
      vals, packed, min(NVFP4_BLOCK_SIZE, K - k_start), input_scale_inv);

  int out_offset = row * (K / 2) + k_start / 2;
  for (int i = 0; i < 8; i++) {
    output[out_offset + i] = packed[i];
  }

  scales[row * num_blocks + block_idx] = block_scale;
}

// ============================================================================
// Scale factor swizzling kernel for CUTLASS block-scaled layout
// Converts linear scale layout to the swizzled 128x4 layout expected by CUTLASS
// ============================================================================

__global__ void nvfp4_swizzle_scales_kernel(
    const uint8_t* __restrict__ linear_scales,  // [rows, cols] linear layout
    uint8_t* __restrict__ swizzled_scales,       // [rows_padded, cols_padded] swizzled
    int rows, int cols,
    int rows_padded, int cols_padded)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows_padded * cols_padded;
  if (idx >= total) return;

  int dst_row = idx / cols_padded;
  int dst_col = idx % cols_padded;

  // Swizzle mapping: for scale factor row 'i', it maps to data block row:
  // (i % 4) * 32 + (i / 4)
  // Inverse: given dst_row, find src_row such that (src_row % 4) * 32 + (src_row / 4) == dst_row
  // This is the CUTLASS 128x4 swizzled layout

  // For the 128x4 block: rows are interleaved in groups of 4
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
// C API: Quantize activations to NVFP4 format for CUTLASS GEMM
// ============================================================================

extern "C" {

void nvfp4_quantize_activation_f16(
    const void* input,      // [M, K] FP16
    void* output,           // [M, K/2] packed FP4 uint8
    void* scales,           // [M_padded, K/16] FP8 block scales
    void* swizzled_scales,  // [M_padded, K_scale_padded] swizzled scales for CUTLASS
    float input_scale_inv,  // 1.0 / input_scale (from checkpoint, default 1.0)
    int M, int K,
    int M_padded, int K_scale_padded,
    int64_t stream)
{
  int num_blocks_k = K / NVFP4_BLOCK_SIZE;
  dim3 grid(M);
  dim3 block(num_blocks_k);

  nvfp4_quantize_activation_kernel<half><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const half*>(input),
      static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(scales),
      input_scale_inv,
      M, K, M_padded);

  int total_swizzled = M_padded * K_scale_padded;
  int threads = 256;
  int blocks = (total_swizzled + threads - 1) / threads;
  nvfp4_swizzle_scales_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(scales),
      static_cast<uint8_t*>(swizzled_scales),
      M, num_blocks_k,
      M_padded, K_scale_padded);
}

void nvfp4_quantize_activation_bf16(
    const void* input,
    void* output,
    void* scales,
    void* swizzled_scales,
    float input_scale_inv,
    int M, int K,
    int M_padded, int K_scale_padded,
    int64_t stream)
{
  int num_blocks_k = K / NVFP4_BLOCK_SIZE;
  dim3 grid(M);
  dim3 block(num_blocks_k);

  nvfp4_quantize_activation_kernel<nv_bfloat16><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const nv_bfloat16*>(input),
      static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(scales),
      input_scale_inv,
      M, K, M_padded);

  int total_swizzled = M_padded * K_scale_padded;
  int threads = 256;
  int blocks = (total_swizzled + threads - 1) / threads;
  nvfp4_swizzle_scales_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(scales),
      static_cast<uint8_t*>(swizzled_scales),
      M, num_blocks_k,
      M_padded, K_scale_padded);
}

// Swizzle weight scales from linear to CUTLASS 128x4 layout
void nvfp4_swizzle_weight_scales(
    const void* linear_scales,
    void* swizzled_scales,
    int rows, int cols,
    int rows_padded, int cols_padded,
    int64_t stream)
{
  int total = rows_padded * cols_padded;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  nvfp4_swizzle_scales_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(linear_scales),
      static_cast<uint8_t*>(swizzled_scales),
      rows, cols,
      rows_padded, cols_padded);
}

}  // extern "C"

// ============================================================================
// MoE helper kernels (C++ templates, outside extern "C")
// ============================================================================

template <typename T>
__global__ void nvfp4_moe_gather_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int32_t* __restrict__ sorted_token_ids,
    int K, int total_expanded)
{
  int row = blockIdx.x;
  int col = threadIdx.x + blockIdx.y * blockDim.x;
  if (row >= total_expanded || col >= K) return;

  int src_token = sorted_token_ids[row];
  output[row * K + col] = input[src_token * K + col];
}

template <typename T>
__global__ void nvfp4_moe_scatter_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int32_t* __restrict__ scatter_ids,
    int N, int total_expanded)
{
  int row = blockIdx.x;
  int col = threadIdx.x + blockIdx.y * blockDim.x;
  if (row >= total_expanded || col >= N) return;

  int dst_row = scatter_ids[row];
  output[dst_row * N + col] = input[row * N + col];
}

extern "C" {

void nvfp4_moe_gather_f16(
    const void* input, void* output,
    const int32_t* sorted_token_ids,
    int total_expanded, int K, int64_t stream)
{
  int threads = min(K, 1024);
  dim3 grid(total_expanded, (K + threads - 1) / threads);
  nvfp4_moe_gather_kernel<half><<<grid, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const half*>(input),
      static_cast<half*>(output),
      sorted_token_ids, K, total_expanded);
}

void nvfp4_moe_gather_bf16(
    const void* input, void* output,
    const int32_t* sorted_token_ids,
    int total_expanded, int K, int64_t stream)
{
  int threads = min(K, 1024);
  dim3 grid(total_expanded, (K + threads - 1) / threads);
  nvfp4_moe_gather_kernel<nv_bfloat16><<<grid, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const nv_bfloat16*>(input),
      static_cast<nv_bfloat16*>(output),
      sorted_token_ids, K, total_expanded);
}

void nvfp4_moe_scatter_f16(
    const void* input, void* output,
    const int32_t* scatter_ids,
    int total_expanded, int N, int64_t stream)
{
  int threads = min(N, 1024);
  dim3 grid(total_expanded, (N + threads - 1) / threads);
  nvfp4_moe_scatter_kernel<half><<<grid, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const half*>(input),
      static_cast<half*>(output),
      scatter_ids, N, total_expanded);
}

void nvfp4_moe_scatter_bf16(
    const void* input, void* output,
    const int32_t* scatter_ids,
    int total_expanded, int N, int64_t stream)
{
  int threads = min(N, 1024);
  dim3 grid(total_expanded, (N + threads - 1) / threads);
  nvfp4_moe_scatter_kernel<nv_bfloat16><<<grid, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const nv_bfloat16*>(input),
      static_cast<nv_bfloat16*>(output),
      scatter_ids, N, total_expanded);
}

}  // extern "C"

#else  // !ENABLE_FP4

extern "C" {

void nvfp4_quantize_activation_f16(
    const void*, void*, void*, void*, float, int, int, int, int, int64_t) {}

void nvfp4_quantize_activation_bf16(
    const void*, void*, void*, void*, float, int, int, int, int, int64_t) {}

void nvfp4_swizzle_weight_scales(
    const void*, void*, int, int, int, int, int64_t) {}

void nvfp4_moe_gather_f16(
    const void*, void*, const int32_t*, int, int, int64_t) {}

void nvfp4_moe_gather_bf16(
    const void*, void*, const int32_t*, int, int, int64_t) {}

void nvfp4_moe_scatter_f16(
    const void*, void*, const int32_t*, int, int, int64_t) {}

void nvfp4_moe_scatter_bf16(
    const void*, void*, const int32_t*, int, int, int64_t) {}

}  // extern "C"

#endif  // ENABLE_FP4
