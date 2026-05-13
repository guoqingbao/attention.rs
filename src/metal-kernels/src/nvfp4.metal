#include "metal_dtype.metal"
#include <metal_stdlib>

using namespace metal;

// NVFP4 (FP4 E2M1) kernels for linear GEMM on Metal.
//
// Weight format: packed U8, 2 FP4 E2M1 nibbles per byte (low nibble = even k, high = k+1)
// Scale format:  U8 FP8 E4M3 block scales, one per 16 weights (NVFP4 block size)
// Weight layout: [N, K/2]
// Scale layout:  [N, K/16]
//
// Dequantization: value = LUT[nibble] * fp8e4m3_to_float(block_scale) * global_scale * 0.5
// The LUT produces integer magnitudes 2x the actual FP4 E2M1 float values,
// so the 0.5 factor compensates (matching the CUDA LUT-based path exactly).

namespace nvfp4 {

constexpr constant int kWarpSize = 32;
constexpr constant int kElemsPerLane = 32;
constexpr constant int kBlockN = 8;
constexpr constant int kThreadsPerThreadgroup = kBlockN * kWarpSize;
constexpr constant int kBlockSize = 16;

constexpr constant int kKTile = kWarpSize * kElemsPerLane;
constexpr constant int kKTilePadded = kKTile + (kKTile / kWarpSize);

// FP4 E2M1 dequant LUT: maps 3-bit magnitude (nibble & 0x7) to integer 2x the
// actual float value. {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} * 2 = below.
// The caller applies * 0.5 via the block scale to get correct values.
constant char kMagLut[8] = {0, 1, 2, 3, 4, 6, 8, 12};

// Decode FP8 E4M3 to float.
// FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7
METAL_FUNC float fp8e4m3_to_float(uchar bits) {
  uint sign = (bits >> 7) & 1;
  uint exp  = (bits >> 3) & 0xF;
  uint mant = bits & 0x7;

  float result;
  if (exp == 0) {
    // Subnormal: value = 2^(-6) * (mantissa/8)
    result = float(mant) / 8.0f * exp2(-6.0f);
  } else if (exp == 15 && mant == 7) {
    // NaN → treat as 0
    result = 0.0f;
  } else {
    // Normal: value = 2^(exp-7) * (1 + mantissa/8)
    result = (1.0f + float(mant) / 8.0f) * exp2(float(exp) - 7.0f);
  }
  return sign ? -result : result;
}

METAL_FUNC char fp4_to_i8x2(uchar nibble) {
  char v = kMagLut[nibble & 7];
  return (nibble & 8) ? char(-v) : v;
}

METAL_FUNC float simdgroup_reduce_sum(float v) {
  v += simd_shuffle_xor(v, ushort(16));
  v += simd_shuffle_xor(v, ushort(8));
  v += simd_shuffle_xor(v, ushort(4));
  v += simd_shuffle_xor(v, ushort(2));
  v += simd_shuffle_xor(v, ushort(1));
  return v;
}

// Process one tile of 32 elements per lane (2 NVFP4 blocks of 16).
// Each lane handles kElemsPerLane = 32 contiguous K elements.
template <typename T>
METAL_FUNC void nvfp4_dot_k1024_tiles(const device uchar *w_row,
                                      const device uchar *s_row,
                                      const threadgroup float *x_tile_padded,
                                      thread float &acc,
                                      int K, int k_base, ushort lane_id,
                                      float global_scale) {
  const int k_lane = k_base + int(lane_id) * kElemsPerLane;
  if (k_lane >= K) {
    return;
  }

  const device uint4 *w_ptr =
      reinterpret_cast<const device uint4 *>(w_row + (k_lane / 2));
  const uint4 packed = *w_ptr;

  const threadgroup float *in =
      x_tile_padded + int(lane_id) * (kElemsPerLane + 1);

  int in_idx = 0;

  // Process 16 bytes = 32 FP4 values (4 x uint32, each containing 8 nibbles)
  // Every 16 elements uses a new FP8 E4M3 block scale.
  // First 16 elements: scale from s_row[k_lane / kBlockSize]
  // Second 16 elements: scale from s_row[(k_lane + 16) / kBlockSize]

  // Block 0: elements [0..15]
  {
    const int scale_idx = k_lane / kBlockSize;
    const float w_scale = fp8e4m3_to_float(s_row[scale_idx]) * global_scale * 0.5f;

    float partial = 0.0f;

    // packed.x = bytes [0..3] = 8 nibbles = elements [0..7]
    {
      uint vv = packed.x;
      for (int j = 0; j < 4; ++j) {
        const uchar b = uchar(vv & 0xff);
        vv >>= 8;
        partial = fma(in[in_idx + 0], float(fp4_to_i8x2(b & 0x0f)), partial);
        partial = fma(in[in_idx + 1], float(fp4_to_i8x2((b >> 4) & 0x0f)), partial);
        in_idx += 2;
      }
    }

    // packed.y = bytes [4..7] = 8 nibbles = elements [8..15]
    {
      uint vv = packed.y;
      for (int j = 0; j < 4; ++j) {
        const uchar b = uchar(vv & 0xff);
        vv >>= 8;
        partial = fma(in[in_idx + 0], float(fp4_to_i8x2(b & 0x0f)), partial);
        partial = fma(in[in_idx + 1], float(fp4_to_i8x2((b >> 4) & 0x0f)), partial);
        in_idx += 2;
      }
    }

    acc = fma(partial, w_scale, acc);
  }

  // Block 1: elements [16..31]
  {
    const int k2 = k_lane + kBlockSize;
    if (k2 < K) {
      const int scale_idx2 = k2 / kBlockSize;
      const float w_scale2 = fp8e4m3_to_float(s_row[scale_idx2]) * global_scale * 0.5f;

      float partial2 = 0.0f;

      // packed.z = bytes [8..11] = elements [16..23]
      {
        uint vv = packed.z;
        for (int j = 0; j < 4; ++j) {
          const uchar b = uchar(vv & 0xff);
          vv >>= 8;
          partial2 = fma(in[in_idx + 0], float(fp4_to_i8x2(b & 0x0f)), partial2);
          partial2 = fma(in[in_idx + 1], float(fp4_to_i8x2((b >> 4) & 0x0f)), partial2);
          in_idx += 2;
        }
      }

      // packed.w = bytes [12..15] = elements [24..31]
      {
        uint vv = packed.w;
        for (int j = 0; j < 4; ++j) {
          const uchar b = uchar(vv & 0xff);
          vv >>= 8;
          partial2 = fma(in[in_idx + 0], float(fp4_to_i8x2(b & 0x0f)), partial2);
          partial2 = fma(in[in_idx + 1], float(fp4_to_i8x2((b >> 4) & 0x0f)), partial2);
          in_idx += 2;
        }
      }

      acc = fma(partial2, w_scale2, acc);
    }
  }
}

// ---- Linear matmul: output[m,n] = input[m,:] @ weight[n,:]^T ----

template <typename T>
METAL_FUNC void
nvfp4_matmul_impl(const device T *x, const device uchar *w,
                  const device uchar *scales, const device T *bias, device T *y,
                  int M, int N, int K, float global_scale, int has_bias,
                  threadgroup float *x_tile,
                  uint tid, ushort simd_gid, ushort lane_id, uint3 gid) {
  (void)tid;

  const int row = int(gid.y);
  const int n_base = int(gid.x) * kBlockN;
  const int n = n_base + int(simd_gid);

  if (row >= M || n >= N) {
    return;
  }

  const device T *x_row = x + row * K;
  const device uchar *w_row = w + n * (K / 2);
  const device uchar *s_row = scales + n * (K / kBlockSize);

  float acc = 0.0f;

  for (int k_base = 0; k_base < K; k_base += kKTile) {
    const int local_base = int(tid) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k_local = local_base + i;
      const int k = k_base + k_local;
      const float v = (k < K) ? float(x_row[k]) : 0.0f;
      x_tile[k_local + (k_local / kWarpSize)] = v;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    nvfp4_dot_k1024_tiles<T>(w_row, s_row, x_tile, acc, K, k_base,
                             lane_id, global_scale);

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  acc = simdgroup_reduce_sum(acc);

  if (lane_id == 0) {
    if (has_bias != 0) {
      acc += float(bias[n]);
    }
    y[row * N + n] = T(acc);
  }
}

// ---- MoE GEMM split: one threadgroup per (token, expert_slot, n_block) ----

template <typename T>
METAL_FUNC void nvfp4_moe_gemm_split_impl(
    const device T *x, const device uchar *w, const device uchar *scales,
    const device float *global_scales, const device uint *indices, device T *y,
    int num_tokens, int topk, int num_experts, int N, int K,
    int input_has_topk_dim, threadgroup float *x_tile, uint tid,
    ushort simd_gid, ushort lane_id, uint3 gid) {
  (void)tid;

  const int n_base = int(gid.x) * kBlockN;
  const int token_idx = int(gid.y);
  const int expert_slot = int(gid.z);
  const int n = n_base + int(simd_gid);

  if (token_idx >= num_tokens || expert_slot >= topk || n >= N) {
    return;
  }

  const uint expert_idx = indices[token_idx * topk + expert_slot];
  if (expert_idx >= uint(num_experts)) {
    if (lane_id == 0) {
      y[(token_idx * topk + expert_slot) * N + n] = T(0.0f);
    }
    return;
  }

  const float gscale = global_scales[expert_idx];

  const device T *x_row = input_has_topk_dim != 0
                              ? (x + (token_idx * topk + expert_slot) * K)
                              : (x + token_idx * K);

  const int weight_row_stride = K / 2;
  const int scale_stride = K / kBlockSize;

  const device uchar *w_row =
      w + (size_t(expert_idx) * N + size_t(n)) * weight_row_stride;
  const device uchar *s_row =
      scales + (size_t(expert_idx) * N + size_t(n)) * scale_stride;

  float acc = 0.0f;

  for (int k_base = 0; k_base < K; k_base += kKTile) {
    const int local_base = int(tid) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k_local = local_base + i;
      const int k = k_base + k_local;
      const float v = (k < K) ? float(x_row[k]) : 0.0f;
      x_tile[k_local + (k_local / kWarpSize)] = v;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    nvfp4_dot_k1024_tiles<T>(w_row, s_row, x_tile, acc, K, k_base,
                             lane_id, gscale);

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  acc = simdgroup_reduce_sum(acc);

  if (lane_id == 0) {
    y[(token_idx * topk + expert_slot) * N + n] = T(acc);
  }
}

// ---- MoE GEMM reuse: reuse input across expert slots when topk <= 8 ----

template <typename T, int MAX_TOPK>
METAL_FUNC void nvfp4_moe_gemm_reuse_impl(
    const device T *x, const device uchar *w, const device uchar *scales,
    const device float *global_scales, const device uint *indices, device T *y,
    int num_tokens, int topk, int num_experts, int N, int K,
    uint tid, ushort simd_gid, threadgroup float *x_tile, ushort lane_id,
    uint3 gid) {
  (void)tid;

  const int n_base = int(gid.x) * kBlockN;
  const int token_idx = int(gid.y);
  const int n = n_base + int(simd_gid);

  if (token_idx >= num_tokens || n >= N) {
    return;
  }
  if (topk > MAX_TOPK) {
    return;
  }

  thread uint expert_idx[MAX_TOPK];
  thread float gscale[MAX_TOPK];
#pragma unroll
  for (int s = 0; s < MAX_TOPK; ++s) {
    expert_idx[s] = (s < topk) ? indices[token_idx * topk + s] : 0u;
    gscale[s] = (s < topk && expert_idx[s] < uint(num_experts))
                    ? global_scales[expert_idx[s]] : 0.0f;
  }

  const device T *x_row = x + token_idx * K;

  const int weight_row_stride = K / 2;
  const int scale_stride = K / kBlockSize;

  float acc[MAX_TOPK];
#pragma unroll
  for (int s = 0; s < MAX_TOPK; ++s) {
    acc[s] = 0.0f;
  }

  for (int k_base = 0; k_base < K; k_base += kKTile) {
    const int local_base = int(tid) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k_local = local_base + i;
      const int k = k_base + k_local;
      const float v = (k < K) ? float(x_row[k]) : 0.0f;
      x_tile[k_local + (k_local / kWarpSize)] = v;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
    for (int s = 0; s < MAX_TOPK; ++s) {
      if (s >= topk) {
        continue;
      }
      const uint e = expert_idx[s];
      if (e >= uint(num_experts)) {
        continue;
      }
      const device uchar *w_row =
          w + (size_t(e) * N + size_t(n)) * weight_row_stride;
      const device uchar *s_row =
          scales + (size_t(e) * N + size_t(n)) * scale_stride;
      nvfp4_dot_k1024_tiles<T>(w_row, s_row, x_tile, acc[s], K,
                               k_base, lane_id, gscale[s]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

#pragma unroll
  for (int s = 0; s < MAX_TOPK; ++s) {
    if (s >= topk) {
      continue;
    }
    float a = acc[s];
    a = simdgroup_reduce_sum(a);
    if (lane_id == 0) {
      const uint e = expert_idx[s];
      if (e >= uint(num_experts)) {
        y[(token_idx * topk + s) * N + n] = T(0.0f);
        continue;
      }
      y[(token_idx * topk + s) * N + n] = T(a);
    }
  }
}

} // namespace nvfp4

// ---- Kernel instantiations ----

[[kernel]] void nvfp4_matmul_f16(
    const device half *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
    const device uchar *scales [[buffer(2)]],
    const device half *bias [[buffer(3)]], device half *y [[buffer(4)]],
    const constant int &M [[buffer(5)]], const constant int &N [[buffer(6)]],
    const constant int &K [[buffer(7)]],
    const constant float &global_scale [[buffer(8)]],
    const constant int &has_bias [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]]) {
  threadgroup float x_tile[nvfp4::kKTilePadded];
  nvfp4::nvfp4_matmul_impl<half>(x, w, scales, bias, y, M, N, K, global_scale,
                                 has_bias, x_tile, tid, simd_gid, lane_id, gid);
}

#if defined(__HAVE_BFLOAT__)
[[kernel]] void nvfp4_matmul_bf16(const device bfloat16_t *x [[buffer(0)]],
                                  const device uchar *w [[buffer(1)]],
                                  const device uchar *scales [[buffer(2)]],
                                  const device bfloat16_t *bias [[buffer(3)]],
                                  device bfloat16_t *y [[buffer(4)]],
                                  const constant int &M [[buffer(5)]],
                                  const constant int &N [[buffer(6)]],
                                  const constant int &K [[buffer(7)]],
                                  const constant float &global_scale [[buffer(8)]],
                                  const constant int &has_bias [[buffer(9)]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  ushort simd_gid
                                  [[simdgroup_index_in_threadgroup]],
                                  ushort lane_id [[thread_index_in_simdgroup]],
                                  uint3 gid [[threadgroup_position_in_grid]]) {
  threadgroup float x_tile[nvfp4::kKTilePadded];
  nvfp4::nvfp4_matmul_impl<bfloat16_t>(x, w, scales, bias, y, M, N, K,
                                       global_scale, has_bias, x_tile, tid,
                                       simd_gid, lane_id, gid);
}
#endif

// ---- NVFP4 MoE split kernel instantiations ----

[[kernel]] void nvfp4_moe_gemm_split_f16(
    const device half *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
    const device uchar *scales [[buffer(2)]],
    const device float *global_scales [[buffer(3)]],
    const device uint *indices [[buffer(4)]], device half *y [[buffer(5)]],
    const constant int &num_tokens [[buffer(6)]],
    const constant int &topk [[buffer(7)]],
    const constant int &num_experts [[buffer(8)]],
    const constant int &N [[buffer(9)]], const constant int &K [[buffer(10)]],
    const constant int &input_has_topk_dim [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]]) {
  threadgroup float x_tile[nvfp4::kKTilePadded];
  nvfp4::nvfp4_moe_gemm_split_impl<half>(
      x, w, scales, global_scales, indices, y, num_tokens, topk, num_experts, N, K,
      input_has_topk_dim, x_tile, tid, simd_gid, lane_id, gid);
}

#if defined(__HAVE_BFLOAT__)
[[kernel]] void
nvfp4_moe_gemm_split_bf16(const device bfloat16_t *x [[buffer(0)]],
                          const device uchar *w [[buffer(1)]],
                          const device uchar *scales [[buffer(2)]],
                          const device float *global_scales [[buffer(3)]],
                          const device uint *indices [[buffer(4)]],
                          device bfloat16_t *y [[buffer(5)]],
                          const constant int &num_tokens [[buffer(6)]],
                          const constant int &topk [[buffer(7)]],
                          const constant int &num_experts [[buffer(8)]],
                          const constant int &N [[buffer(9)]],
                          const constant int &K [[buffer(10)]],
                          const constant int &input_has_topk_dim [[buffer(11)]],
                          uint tid [[thread_index_in_threadgroup]],
                          ushort simd_gid [[simdgroup_index_in_threadgroup]],
                          ushort lane_id [[thread_index_in_simdgroup]],
                          uint3 gid [[threadgroup_position_in_grid]]) {
  threadgroup float x_tile[nvfp4::kKTilePadded];
  nvfp4::nvfp4_moe_gemm_split_impl<bfloat16_t>(
      x, w, scales, global_scales, indices, y, num_tokens, topk, num_experts, N, K,
      input_has_topk_dim, x_tile, tid, simd_gid, lane_id, gid);
}
#endif

// ---- NVFP4 MoE reuse kernel instantiations ----

[[kernel]] void nvfp4_moe_gemm_reuse_f16(
    const device half *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
    const device uchar *scales [[buffer(2)]],
    const device float *global_scales [[buffer(3)]],
    const device uint *indices [[buffer(4)]], device half *y [[buffer(5)]],
    const constant int &num_tokens [[buffer(6)]],
    const constant int &topk [[buffer(7)]],
    const constant int &num_experts [[buffer(8)]],
    const constant int &N [[buffer(9)]], const constant int &K [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]]) {
  threadgroup float x_tile[nvfp4::kKTilePadded];
  nvfp4::nvfp4_moe_gemm_reuse_impl<half, 8>(
      x, w, scales, global_scales, indices, y, num_tokens, topk, num_experts, N, K,
      tid, simd_gid, x_tile, lane_id, gid);
}

#if defined(__HAVE_BFLOAT__)
[[kernel]] void
nvfp4_moe_gemm_reuse_bf16(const device bfloat16_t *x [[buffer(0)]],
                          const device uchar *w [[buffer(1)]],
                          const device uchar *scales [[buffer(2)]],
                          const device float *global_scales [[buffer(3)]],
                          const device uint *indices [[buffer(4)]],
                          device bfloat16_t *y [[buffer(5)]],
                          const constant int &num_tokens [[buffer(6)]],
                          const constant int &topk [[buffer(7)]],
                          const constant int &num_experts [[buffer(8)]],
                          const constant int &N [[buffer(9)]],
                          const constant int &K [[buffer(10)]],
                          uint tid [[thread_index_in_threadgroup]],
                          ushort simd_gid [[simdgroup_index_in_threadgroup]],
                          ushort lane_id [[thread_index_in_simdgroup]],
                          uint3 gid [[threadgroup_position_in_grid]]) {
  threadgroup float x_tile[nvfp4::kKTilePadded];
  nvfp4::nvfp4_moe_gemm_reuse_impl<bfloat16_t, 8>(
      x, w, scales, global_scales, indices, y, num_tokens, topk, num_experts, N, K,
      tid, simd_gid, x_tile, lane_id, gid);
}
#endif
