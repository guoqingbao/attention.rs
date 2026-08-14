// DeepSeek V4 / FlashMLA MODEL1 FP8 FOOTER KV pack.
// Physical layout per page (page_block_size tokens):
//   [0, page*576):  per-token [448B FP8 nope | 128B BF16 rope]
//   [page*576, page*584): per-token [7×UE8M0 scale | 1 pad]
// Logical view: [num_pages, page_block_size, 1, 584]

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kDNope = 448;
constexpr int kDRope = 64;
constexpr int kTile = 64;
constexpr int kNumTiles = 7;  // 448/64
constexpr int kScaleBytes = 8;  // 7 UE8M0 + 1 pad
constexpr int kDataStride = kDNope + kDRope * 2;  // 576
constexpr int kBytesPerToken = kDataStride + kScaleBytes;  // 584

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ uint8_t fp32_to_ue8m0(float scale) {
  // scale is a power-of-two; extract IEEE exponent byte.
  uint32_t bits = __float_as_uint(scale);
  return static_cast<uint8_t>((bits >> 23) & 0xFFu);
}

__device__ __forceinline__ float ue8m0_to_fp32(uint8_t e) {
  return __uint_as_float(static_cast<uint32_t>(e) << 23);
}

// One block per token. Threads cooperatively handle tiles + rope.
__global__ void ds_fp8_kv_pack_footer_kernel(
    const __nv_bfloat16* __restrict__ src,  // [num_tokens, 512]
    uint8_t* __restrict__ dst,              // FOOTER packed pages
    int num_tokens,
    int page_block_size,
    int src_stride) {
  int tok = blockIdx.x;
  if (tok >= num_tokens) return;

  int page = tok / page_block_size;
  int row = tok % page_block_size;
  uint8_t* page_base = dst + static_cast<size_t>(page) * page_block_size * kBytesPerToken;
  uint8_t* data_base = page_base + static_cast<size_t>(row) * kDataStride;
  uint8_t* scale_base =
      page_base + static_cast<size_t>(page_block_size) * kDataStride +
      static_cast<size_t>(row) * kScaleBytes;

  const __nv_bfloat16* row_src = src + static_cast<size_t>(tok) * src_stride;

  // Quantize nope tiles.
  for (int ti = threadIdx.x; ti < kNumTiles; ti += blockDim.x) {
    float amax = 1e-4f;
    for (int i = 0; i < kTile; ++i) {
      float v = fabsf(bf16_to_f32(row_src[ti * kTile + i]));
      amax = fmaxf(amax, v);
    }
    // Match FlashMLA/FlashInfer: round inverse scale to power-of-2 via ceil(log2).
    float inv = amax / 448.0f;
    float scale = exp2f(ceilf(log2f(fmaxf(inv, 1e-4f))));
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < kTile; ++i) {
      float q = bf16_to_f32(row_src[ti * kTile + i]) * inv_scale;
      q = fminf(fmaxf(q, -448.0f), 448.0f);
      __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(q);
      data_base[ti * kTile + i] = *reinterpret_cast<uint8_t*>(&fp8);
    }
    scale_base[ti] = fp32_to_ue8m0(scale);
  }
  if (threadIdx.x == 0) {
    scale_base[7] = 0;  // pad
  }

  // Copy rope as BF16 bytes.
  const uint8_t* rope_src =
      reinterpret_cast<const uint8_t*>(row_src + kDNope);
  uint8_t* rope_dst = data_base + kDNope;
  for (int i = threadIdx.x; i < kDRope * 2; i += blockDim.x) {
    rope_dst[i] = rope_src[i];
  }
}

// Pack into a single contiguous FOOTER buffer with page_block_size == num_tokens
// (or caller-chosen page size). Unused trailing rows left untouched.
__global__ void ds_fp8_kv_pack_rows_kernel(
    const __nv_bfloat16* __restrict__ src,
    uint8_t* __restrict__ dst_page,  // one page buffer of page_block_size tokens
    int num_tokens,
    int page_block_size,
    int src_stride) {
  int tok = blockIdx.x;
  if (tok >= num_tokens || tok >= page_block_size) return;

  uint8_t* data_base = dst_page + static_cast<size_t>(tok) * kDataStride;
  uint8_t* scale_base =
      dst_page + static_cast<size_t>(page_block_size) * kDataStride +
      static_cast<size_t>(tok) * kScaleBytes;
  const __nv_bfloat16* row_src = src + static_cast<size_t>(tok) * src_stride;

  for (int ti = threadIdx.x; ti < kNumTiles; ti += blockDim.x) {
    float amax = 1e-4f;
    for (int i = 0; i < kTile; ++i) {
      float v = fabsf(bf16_to_f32(row_src[ti * kTile + i]));
      amax = fmaxf(amax, v);
    }
    float inv = amax / 448.0f;
    float scale = exp2f(ceilf(log2f(fmaxf(inv, 1e-4f))));
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < kTile; ++i) {
      float q = bf16_to_f32(row_src[ti * kTile + i]) * inv_scale;
      q = fminf(fmaxf(q, -448.0f), 448.0f);
      __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(q);
      data_base[ti * kTile + i] = *reinterpret_cast<uint8_t*>(&fp8);
    }
    scale_base[ti] = fp32_to_ue8m0(scale);
  }
  if (threadIdx.x == 0) scale_base[7] = 0;

  const uint8_t* rope_src =
      reinterpret_cast<const uint8_t*>(row_src + kDNope);
  uint8_t* rope_dst = data_base + kDNope;
  for (int i = threadIdx.x; i < kDRope * 2; i += blockDim.x) {
    rope_dst[i] = rope_src[i];
  }
}

}  // namespace

extern "C" int ds_fp8_kv_pack_footer(
    const void* src_bf16,
    void* dst_u8,
    int num_tokens,
    int page_block_size,
    int src_stride,
    cudaStream_t stream) {
  if (!src_bf16 || !dst_u8 || num_tokens <= 0 || page_block_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (src_stride <= 0) src_stride = 512;
  dim3 grid(num_tokens);
  dim3 block(32);
  ds_fp8_kv_pack_footer_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(src_bf16),
      reinterpret_cast<uint8_t*>(dst_u8),
      num_tokens,
      page_block_size,
      src_stride);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int ds_fp8_kv_pack_rows(
    const void* src_bf16,
    void* dst_page_u8,
    int num_tokens,
    int page_block_size,
    int src_stride,
    cudaStream_t stream) {
  if (!src_bf16 || !dst_page_u8 || num_tokens <= 0 || page_block_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (src_stride <= 0) src_stride = 512;
  dim3 grid(num_tokens);
  dim3 block(32);
  ds_fp8_kv_pack_rows_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(src_bf16),
      reinterpret_cast<uint8_t*>(dst_page_u8),
      num_tokens,
      page_block_size,
      src_stride);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int ds_fp8_kv_bytes_per_token() { return kBytesPerToken; }
