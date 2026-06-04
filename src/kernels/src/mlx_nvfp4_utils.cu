/**
 * @brief CUDA kernels for MLX NVFP4 format support.
 *
 * Two kernels:
 *   1. mlx_nvfp4_repack_u32_to_u8 - Reinterpret U32 packed weights as U8 on GPU.
 *      MLX stores FP4 E2M1 weights as U32 (8 nibbles per U32). Our NVFP4 GEMM
 *      kernels expect U8 (2 nibbles per byte). This kernel copies the raw bytes
 *      from the U32 layout to a contiguous U8 tensor with shape [rows, cols*4].
 *      Little-endian byte order means the nibble ordering is already correct.
 *
 *   2. mlx_nvfp4_dequant_embedding - Dequantize MLX NVFP4 embeddings on GPU.
 *      Takes U32 packed weights [vocab, hidden/8] and U8 FP8 E4M3 scales
 *      [vocab, hidden/16], produces F16 or BF16 [vocab, hidden] embeddings.
 *      Uses FP4 E2M1 LUT + FP8 E4M3 scale decode (no global scale needed for MLX).
 */

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "attention/dtype_fp8.cuh"

namespace mlx_nvfp4 {

using vllm::fp8::dispatch_fp8_to_float;

// FP4 E2M1 dequantization LUT (direct float values)
__device__ __constant__ float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// ============================================================================
// Kernel 1: Repack U32 → U8 (zero-copy reinterpret on GPU)
// ============================================================================

// Each thread copies one U32 value → 4 U8 bytes (little-endian).
// Input:  [num_rows, num_u32_cols] as uint32_t
// Output: [num_rows, num_u32_cols * 4] as uint8_t
__global__ void mlx_nvfp4_repack_u32_to_u8_kernel(
    const uint32_t *__restrict__ input, uint8_t *__restrict__ output,
    int num_rows, int num_u32_cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_rows * num_u32_cols;
  if (idx >= total) return;

  uint32_t val = input[idx];
  int out_base = idx * 4;
  output[out_base + 0] = (uint8_t)(val & 0xFF);
  output[out_base + 1] = (uint8_t)((val >> 8) & 0xFF);
  output[out_base + 2] = (uint8_t)((val >> 16) & 0xFF);
  output[out_base + 3] = (uint8_t)((val >> 24) & 0xFF);
}

// ============================================================================
// Kernel 2: Dequantize MLX NVFP4 embeddings
// ============================================================================
// Each thread dequantizes 2 FP4 nibbles (1 byte of packed weight).
// Input weights:  [vocab_size, hidden_size/8] as uint32_t
// Input scales:   [vocab_size, hidden_size/16] as uint8_t (FP8 E4M3)
// Output:         [vocab_size, hidden_size] as T (half or __nv_bfloat16)
//
// Thread mapping: one thread per byte of packed data = 2 output elements.
// Total bytes per row = hidden_size / 2.

template <typename T>
__global__ void mlx_nvfp4_dequant_embedding_kernel(
    const uint32_t *__restrict__ weight_u32,
    const uint8_t *__restrict__ scales, T *__restrict__ output, int vocab_size,
    int hidden_size) {
  int bytes_per_row = hidden_size / 2;
  int u32_per_row = hidden_size / 8;
  int scale_per_row = hidden_size / 16;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_bytes = vocab_size * bytes_per_row;
  if (idx >= total_bytes) return;

  int row = idx / bytes_per_row;
  int byte_in_row = idx % bytes_per_row;

  // Locate the U32 that contains this byte
  int u32_idx = byte_in_row / 4;
  int byte_within_u32 = byte_in_row % 4;

  uint32_t packed = __ldg(&weight_u32[row * u32_per_row + u32_idx]);
  uint8_t byte_val = (packed >> (byte_within_u32 * 8)) & 0xFF;

  uint8_t lo_nibble = byte_val & 0xF;
  uint8_t hi_nibble = (byte_val >> 4) & 0xF;

  // Each byte covers 2 elements at col positions byte_in_row*2 and byte_in_row*2+1
  int col = byte_in_row * 2;

  // Block scale: one FP8 E4M3 scale per 16 elements
  int lo_scale_idx = col / 16;
  int hi_scale_idx = (col + 1) / 16;
  float lo_scale = dispatch_fp8_to_float(__ldg(&scales[row * scale_per_row + lo_scale_idx]));
  float hi_scale = dispatch_fp8_to_float(__ldg(&scales[row * scale_per_row + hi_scale_idx]));

  float lo_val = kFp4Lut[lo_nibble] * lo_scale;
  float hi_val = kFp4Lut[hi_nibble] * hi_scale;

  int out_offset = row * hidden_size + col;
  if constexpr (sizeof(T) == 2) {
    if constexpr (std::is_same_v<T, half>) {
      output[out_offset] = __float2half(lo_val);
      output[out_offset + 1] = __float2half(hi_val);
    } else {
      output[out_offset] = __float2bfloat16(lo_val);
      output[out_offset + 1] = __float2bfloat16(hi_val);
    }
  }
}

}  // namespace mlx_nvfp4

// ============================================================================
// C-linkage entry points
// ============================================================================

extern "C" {

void mlx_nvfp4_repack_u32_to_u8(const void *input, void *output, int num_rows,
                                 int num_u32_cols, int64_t stream) {
  int total = num_rows * num_u32_cols;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  mlx_nvfp4::mlx_nvfp4_repack_u32_to_u8_kernel<<<blocks, threads, 0,
                                                   (cudaStream_t)stream>>>(
      (const uint32_t *)input, (uint8_t *)output, num_rows, num_u32_cols);
}

void mlx_nvfp4_dequant_embedding_f16(const void *weight_u32,
                                     const void *scales, void *output,
                                     int vocab_size, int hidden_size,
                                     int64_t stream) {
  int bytes_per_row = hidden_size / 2;
  int total = vocab_size * bytes_per_row;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  mlx_nvfp4::mlx_nvfp4_dequant_embedding_kernel<half>
      <<<blocks, threads, 0, (cudaStream_t)stream>>>(
          (const uint32_t *)weight_u32, (const uint8_t *)scales,
          (half *)output, vocab_size, hidden_size);
}

void mlx_nvfp4_dequant_embedding_bf16(const void *weight_u32,
                                      const void *scales, void *output,
                                      int vocab_size, int hidden_size,
                                      int64_t stream) {
  int bytes_per_row = hidden_size / 2;
  int total = vocab_size * bytes_per_row;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  mlx_nvfp4::mlx_nvfp4_dequant_embedding_kernel<__nv_bfloat16>
      <<<blocks, threads, 0, (cudaStream_t)stream>>>(
          (const uint32_t *)weight_u32, (const uint8_t *)scales,
          (__nv_bfloat16 *)output, vocab_size, hidden_size);
}

}  // extern "C"
