/*
 * NVFP4 activation quantization kernel.
 * Quantizes BF16/FP16 activations to packed FP4 E2M1 with FP8 E4M3 block scales.
 * Block size = 16 (NVFP4 standard).
 *
 * Required for hardware FP4 GEMM path on Blackwell (SM100+).
 * The CUTLASS block-scaled tensor ops expect both A and B in FP4 format.
 *
 * On SM100+ (Blackwell): uses hardware PTX cvt.rn.satfinite.e2m1x2.f32 for
 * precise FP4 conversion and __nv_fp8_e4m3 for scale factor encoding.
 * On older GPUs: uses software fallback with LUT-based conversion.
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

// ============================================================================
// Hardware FP4 conversion (SM100+ / Blackwell)
// Uses PTX cvt.rn.satfinite.e2m1x2.f32 for precise round-to-nearest-even
// ============================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

__device__ __forceinline__ uint32_t fp32x8_to_e2m1x8(float (&vals)[8]) {
  uint32_t result;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(result)
      : "f"(vals[0]), "f"(vals[1]), "f"(vals[2]), "f"(vals[3]),
        "f"(vals[4]), "f"(vals[5]), "f"(vals[6]), "f"(vals[7]));
  return result;
}

__device__ __forceinline__ float fast_rcp_ftz(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

#endif // __CUDA_ARCH__ >= 1000

// ============================================================================
// Software FP4 conversion fallback (pre-SM100)
// ============================================================================

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
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

// ============================================================================
// Activation quantization kernel (SM100+ hardware path)
// Matches TRT-LLM/FlashInfer quantize_with_block_size exactly:
//   SFValue = SFScaleVal * (vecMax / 6.0)
//   fp8_scale = fp8_e4m3(SFValue)
//   outputScale = 1 / (float(fp8_scale) / SFScaleVal)
//   quantized_val = val * outputScale
// ============================================================================

template <typename InType>
__global__ void nvfp4_quantize_activation_hw_kernel(
    const InType* __restrict__ input,   // [M, K]
    uint8_t* __restrict__ output,       // [M, K/2] packed FP4
    uint8_t* __restrict__ scales,       // [M_padded, K/16] FP8 E4M3 block scales
    float SFScaleVal,                   // globalScale = (448*6)/amax or 1/input_scale
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

  // Find absolute maximum of the block (matching TRT-LLM's warp reduction)
  float vecMax = 0.0f;
  for (int i = 0; i < NVFP4_BLOCK_SIZE; i++) {
    vecMax = fmaxf(vecMax, fabsf(vals[i]));
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Hardware path: matches TRT-LLM exactly
  // SFValue = SFScaleVal * (vecMax / 6.0)
  float SFValue = SFScaleVal * (vecMax * fast_rcp_ftz(6.0f));

  __nv_fp8_e4m3 fp8_sf = __nv_fp8_e4m3(SFValue);
  uint8_t fp8_scale_bits = fp8_sf.__x;
  SFValue = static_cast<float>(fp8_sf);

  // outputScale = 1 / (float(fp8(SFValue)) / SFScaleVal)
  float outputScale = vecMax != 0.0f
      ? fast_rcp_ftz(SFValue * fast_rcp_ftz(SFScaleVal))
      : 0.0f;

  // Scale values and convert to FP4 using hardware intrinsics
  float scaled_vals_0[8], scaled_vals_1[8];
  for (int i = 0; i < 8; i++) {
    scaled_vals_0[i] = vals[i] * outputScale;
  }
  for (int i = 0; i < 8; i++) {
    scaled_vals_1[i] = vals[8 + i] * outputScale;
  }

  uint32_t packed_lo = fp32x8_to_e2m1x8(scaled_vals_0);
  uint32_t packed_hi = fp32x8_to_e2m1x8(scaled_vals_1);

  // Write packed FP4 output (8 bytes = 16 nibbles)
  int out_offset = row * (K / 2) + k_start / 2;
  reinterpret_cast<uint32_t*>(output + out_offset)[0] = packed_lo;
  reinterpret_cast<uint32_t*>(output + out_offset)[1] = packed_hi;

  scales[row * num_blocks + block_idx] = fp8_scale_bits;

#else
  // Software fallback for pre-SM100
  float SFValue = SFScaleVal * (vecMax / 6.0f);

  uint8_t fp8_scale_bits = 0;
  float outputScale = 0.0f;

  if (vecMax > 0.0f && SFValue > 0.0f) {
    // Software FP8 E4M3 conversion
    uint32_t bits = __float_as_uint(SFValue);
    uint32_t sign = (bits >> 31) & 1;
    int exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exp > 8) { exp = 8; mantissa = 0x600000; }
    if (exp >= -9) {
      int biased_exp = exp + 7;
      if (biased_exp < 0) biased_exp = 0;
      if (biased_exp > 15) biased_exp = 15;

      uint8_t mant3 = (mantissa >> 20) & 0x7;
      fp8_scale_bits = (sign << 7) | (biased_exp << 3) | mant3;

      float recon_mantissa = 1.0f + mant3 / 8.0f;
      float quant_scale = recon_mantissa * powf(2.0f, biased_exp - 7);
      if (sign) quant_scale = -quant_scale;

      outputScale = (quant_scale != 0.0f && SFScaleVal != 0.0f)
          ? (1.0f / (quant_scale / SFScaleVal))
          : 0.0f;
    }
  }

  uint8_t codes[16];
  for (int i = 0; i < NVFP4_BLOCK_SIZE; i++) {
    codes[i] = float_to_fp4_e2m1(vals[i] * outputScale);
  }

  int out_offset = row * (K / 2) + k_start / 2;
  for (int i = 0; i < 8; i++) {
    output[out_offset + i] = (codes[2 * i + 1] << 4) | codes[2 * i];
  }

  scales[row * num_blocks + block_idx] = fp8_scale_bits;
#endif
}

// ============================================================================
// Scale factor swizzling kernel for CUTLASS block-scaled 128x4 layout
// Matches TRT-LLM's get_sf_out_offset_128x4 exactly:
//   SF layout [numMTiles, numKTiles, 32 (outerM), 4 (innerM), 4 (innerK)]
//   Total tile size = 32 * 4 * 4 = 512
// Input:  linear_scales[rows, cols] in row-major
// Output: swizzled_scales[total_swizzled_size] in CUTLASS 128x4 layout
// ============================================================================

__global__ void nvfp4_swizzle_scales_kernel(
    const uint8_t* __restrict__ linear_scales,  // [rows, cols] linear layout
    uint8_t* __restrict__ swizzled_scales,       // flat swizzled output
    int rows, int cols,
    int rows_padded, int cols_padded)
{
  // Each thread processes one (mIdx, kIdx) pair from the linear layout
  // and writes to the swizzled destination
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows_padded * cols_padded;
  if (idx >= total) return;

  int mIdx = idx / cols_padded;
  int kIdx = idx % cols_padded;

  // Read from linear layout (zero for padding)
  uint8_t val = 0;
  if (mIdx < rows && kIdx < cols) {
    val = linear_scales[mIdx * cols + kIdx];
  }

  // Compute swizzled destination offset matching TRT-LLM's get_sf_out_offset_128x4
  int innerKIdx = kIdx % 4;
  int innerMIdx = (mIdx % 128) / 32;
  int outerMIdx = mIdx % 32;
  int kTileIdx = kIdx / 4;
  int mTileIdx = mIdx / 128;

  int numKTiles = (cols_padded + 3) / 4;

  int64_t kTileStride = 512;  // 32 * 4 * 4
  int64_t mTileStride = (int64_t)numKTiles * kTileStride;

  int64_t dstOffset = (int64_t)mTileIdx * mTileStride
                    + (int64_t)kTileIdx * kTileStride
                    + outerMIdx * 16
                    + innerMIdx * 4
                    + innerKIdx;

  swizzled_scales[dstOffset] = val;
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
    float input_scale_inv,  // SFScaleVal = 1.0 / input_scale (from checkpoint, default 1.0)
    int M, int K,
    int M_padded, int K_scale_padded,
    int64_t stream)
{
  int num_blocks_k = K / NVFP4_BLOCK_SIZE;
  dim3 grid(M);
  dim3 block(num_blocks_k);

  nvfp4_quantize_activation_hw_kernel<half><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
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

  nvfp4_quantize_activation_hw_kernel<nv_bfloat16><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
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
    int K, int total_expanded, int map_divisor)
{
  int row = blockIdx.x;
  int col = threadIdx.x + blockIdx.y * blockDim.x;
  if (row >= total_expanded || col >= K) return;

  int src_token = sorted_token_ids[row] / map_divisor;
  output[row * K + col] = input[src_token * K + col];
}

__device__ __forceinline__ int nvfp4_find_expert_for_row(
    const int32_t* __restrict__ expert_offsets,
    int num_experts,
    int row) {
  int lo = 0;
  int hi = num_experts;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    if (expert_offsets[mid] <= row) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__global__ void nvfp4_moe_build_metadata_kernel(
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ weight_global_scales,
    const float* __restrict__ input_scales,
    int32_t* __restrict__ sf_offsets,
    int32_t* __restrict__ problem_sizes,
    float* __restrict__ alphas,
    float* __restrict__ input_scale_invs,
    int num_experts,
    int N,
    int K) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  int running_sf_offset = 0;
  for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
    int rows = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
    float input_scale = input_scales != nullptr ? input_scales[expert_id] : 1.0f;
    float input_scale_inv = input_scale != 0.0f ? 1.0f / input_scale : 1.0f;

    sf_offsets[expert_id] = running_sf_offset;
    problem_sizes[expert_id * 3 + 0] = rows;
    problem_sizes[expert_id * 3 + 1] = N;
    problem_sizes[expert_id * 3 + 2] = K;
    alphas[expert_id] = input_scale * weight_global_scales[expert_id];
    input_scale_invs[expert_id] = input_scale_inv;

    running_sf_offset += ((rows + 127) / 128) * 128;
  }
}

template <typename InType>
__global__ void nvfp4_quantize_activation_hw_grouped_kernel(
    const InType* __restrict__ input,        // [total_rows, K]
    uint8_t* __restrict__ output,            // [total_rows, K/2]
    uint8_t* __restrict__ swizzled_scales,   // flat swizzled buffer
    const float* __restrict__ input_scale_invs,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ sf_offsets,
    int total_rows,
    int num_experts,
    int K,
    int K_scale_padded) {
  int row = blockIdx.x;
  int block_idx = threadIdx.x;
  int num_blocks = K / NVFP4_BLOCK_SIZE;
  if (row >= total_rows || block_idx >= num_blocks) {
    return;
  }

  int expert_id = nvfp4_find_expert_for_row(expert_offsets, num_experts, row);
  int local_row = row - expert_offsets[expert_id];
  int local_row_padded_base = sf_offsets[expert_id];
  int rows = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
  int rows_padded = ((rows + 127) / 128) * 128;
  float SFScaleVal = input_scale_invs[expert_id];

  int k_start = block_idx * NVFP4_BLOCK_SIZE;
  float vals[16];
  for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i) {
    vals[i] = static_cast<float>(input[row * K + k_start + i]);
  }

  float vecMax = 0.0f;
  for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i) {
    vecMax = fmaxf(vecMax, fabsf(vals[i]));
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  float SFValue = SFScaleVal * (vecMax * fast_rcp_ftz(6.0f));
  __nv_fp8_e4m3 fp8_sf = __nv_fp8_e4m3(SFValue);
  uint8_t fp8_scale_bits = fp8_sf.__x;
  SFValue = static_cast<float>(fp8_sf);
  float outputScale = vecMax != 0.0f ? fast_rcp_ftz(SFValue * fast_rcp_ftz(SFScaleVal)) : 0.0f;

  float scaled_vals_0[8], scaled_vals_1[8];
  for (int i = 0; i < 8; ++i) {
    scaled_vals_0[i] = vals[i] * outputScale;
    scaled_vals_1[i] = vals[8 + i] * outputScale;
  }

  uint32_t packed_lo = fp32x8_to_e2m1x8(scaled_vals_0);
  uint32_t packed_hi = fp32x8_to_e2m1x8(scaled_vals_1);
  int out_offset = row * (K / 2) + k_start / 2;
  reinterpret_cast<uint32_t*>(output + out_offset)[0] = packed_lo;
  reinterpret_cast<uint32_t*>(output + out_offset)[1] = packed_hi;
#else
  float SFValue = SFScaleVal * (vecMax / 6.0f);
  uint8_t fp8_scale_bits = 0;
  float outputScale = 0.0f;

  if (vecMax > 0.0f && SFValue > 0.0f) {
    uint32_t bits = __float_as_uint(SFValue);
    uint32_t sign = (bits >> 31) & 1;
    int exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (exp > 8) { exp = 8; mantissa = 0x600000; }
    if (exp >= -9) {
      int biased_exp = exp + 7;
      if (biased_exp < 0) biased_exp = 0;
      if (biased_exp > 15) biased_exp = 15;
      uint8_t mant3 = (mantissa >> 20) & 0x7;
      fp8_scale_bits = (sign << 7) | (biased_exp << 3) | mant3;
      float recon_mantissa = 1.0f + mant3 / 8.0f;
      float quant_scale = recon_mantissa * powf(2.0f, biased_exp - 7);
      if (sign) quant_scale = -quant_scale;
      outputScale = (quant_scale != 0.0f && SFScaleVal != 0.0f)
          ? (1.0f / (quant_scale / SFScaleVal))
          : 0.0f;
    }
  }

  uint8_t codes[16];
  for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i) {
    codes[i] = float_to_fp4_e2m1(vals[i] * outputScale);
  }

  int out_offset = row * (K / 2) + k_start / 2;
  for (int i = 0; i < 8; ++i) {
    output[out_offset + i] = (codes[2 * i + 1] << 4) | codes[2 * i];
  }
#endif

  int innerKIdx = block_idx % 4;
  int innerMIdx = (local_row % 128) / 32;
  int outerMIdx = local_row % 32;
  int kTileIdx = block_idx / 4;
  int mTileIdx = local_row / 128;
  int numKTiles = K_scale_padded / 4;
  int64_t kTileStride = 512;
  int64_t mTileStride = static_cast<int64_t>(numKTiles) * kTileStride;
  int64_t chunk_base = static_cast<int64_t>(local_row_padded_base) * K_scale_padded;
  int64_t dstOffset = chunk_base +
      static_cast<int64_t>(mTileIdx) * mTileStride +
      static_cast<int64_t>(kTileIdx) * kTileStride +
      outerMIdx * 16 +
      innerMIdx * 4 +
      innerKIdx;

  if (local_row < rows && local_row < rows_padded) {
    swizzled_scales[dstOffset] = fp8_scale_bits;
  }
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
    int total_expanded, int K, int map_divisor, int64_t stream)
{
  int threads = min(K, 1024);
  dim3 grid(total_expanded, (K + threads - 1) / threads);
  nvfp4_moe_gather_kernel<half><<<grid, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const half*>(input),
      static_cast<half*>(output),
      sorted_token_ids, K, total_expanded, map_divisor);
}

void nvfp4_moe_gather_bf16(
    const void* input, void* output,
    const int32_t* sorted_token_ids,
    int total_expanded, int K, int map_divisor, int64_t stream)
{
  int threads = min(K, 1024);
  dim3 grid(total_expanded, (K + threads - 1) / threads);
  nvfp4_moe_gather_kernel<nv_bfloat16><<<grid, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const nv_bfloat16*>(input),
      static_cast<nv_bfloat16*>(output),
      sorted_token_ids, K, total_expanded, map_divisor);
}

void nvfp4_moe_build_metadata(
    const int32_t* expert_offsets,
    const float* weight_global_scales,
    const float* input_scales,
    int32_t* sf_offsets,
    int32_t* problem_sizes,
    float* alphas,
    float* input_scale_invs,
    int num_experts,
    int N,
    int K,
    int64_t stream)
{
  nvfp4_moe_build_metadata_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      expert_offsets,
      weight_global_scales,
      input_scales,
      sf_offsets,
      problem_sizes,
      alphas,
      input_scale_invs,
      num_experts,
      N,
      K);
}

void nvfp4_quantize_activation_grouped_f16(
    const void* input,
    void* output,
    void* swizzled_scales,
    const float* input_scale_invs,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    int total_rows,
    int num_experts,
    int K,
    int K_scale_padded,
    int64_t stream)
{
  dim3 grid(total_rows);
  dim3 block(K / NVFP4_BLOCK_SIZE);
  nvfp4_quantize_activation_hw_grouped_kernel<half><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const half*>(input),
      static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(swizzled_scales),
      input_scale_invs,
      expert_offsets,
      sf_offsets,
      total_rows,
      num_experts,
      K,
      K_scale_padded);
}

void nvfp4_quantize_activation_grouped_bf16(
    const void* input,
    void* output,
    void* swizzled_scales,
    const float* input_scale_invs,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    int total_rows,
    int num_experts,
    int K,
    int K_scale_padded,
    int64_t stream)
{
  dim3 grid(total_rows);
  dim3 block(K / NVFP4_BLOCK_SIZE);
  nvfp4_quantize_activation_hw_grouped_kernel<nv_bfloat16><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const nv_bfloat16*>(input),
      static_cast<uint8_t*>(output),
      static_cast<uint8_t*>(swizzled_scales),
      input_scale_invs,
      expert_offsets,
      sf_offsets,
      total_rows,
      num_experts,
      K,
      K_scale_padded);
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

void nvfp4_quantize_activation_grouped_f16(
    const void*, void*, void*, const float*, const int32_t*, const int32_t*, int, int, int, int, int64_t) {}

void nvfp4_quantize_activation_grouped_bf16(
    const void*, void*, void*, const float*, const int32_t*, const int32_t*, int, int, int, int, int64_t) {}

void nvfp4_moe_build_metadata(
    const int32_t*, const float*, const float*, int32_t*, int32_t*, float*, float*, int, int, int, int64_t) {}

void nvfp4_swizzle_weight_scales(
    const void*, void*, int, int, int, int, int64_t) {}

void nvfp4_moe_gather_f16(
    const void*, void*, const int32_t*, int, int, int, int64_t) {}

void nvfp4_moe_gather_bf16(
    const void*, void*, const int32_t*, int, int, int, int64_t) {}

void nvfp4_moe_scatter_f16(
    const void*, void*, const int32_t*, int, int, int64_t) {}

void nvfp4_moe_scatter_bf16(
    const void*, void*, const int32_t*, int, int, int64_t) {}

}  // extern "C"

#endif  // ENABLE_FP4
