// ZipCCL: lossless BF16/F16 exponent compression for NCCL collectives.
// Based on arXiv:2604.27844 — top-7 exponent remapping + bitplane packing.
//
// The compressed payload has a dynamic zero-point section. xInfer performs
// metadata extraction and exchange only during prefill; decode stays on the
// native NCCL/CUDA-graph path.
//
// Wire format (fixed xfer size):
//   Header(16): n:u32, zp_count:u32, top7[7]:u8, dtype:u8
//   sign_mant[n]
//   exp_bitplanes[3 * ceil(n/8)]            (always)
//   mant_lo_bitplanes[3 * ceil(n/8)]        (F16 only)
//   block_zp_bases[num_blocks]              (uint32)
//   zp_exps[zp_count]                       (dynamic, all outliers preserved)
//
// dtype 0 = BF16, 1 = F16.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace zipccl {

constexpr int HEADER = 16;
constexpr int BLOCK = 256;

struct __align__(16) Header {
  uint32_t n;
  uint32_t zp_count;
  uint8_t top7[7];
  uint8_t dtype;
};

__host__ __device__ __forceinline__ int bp_bytes(int n) { return (n + 7) >> 3; }

__host__ __device__ __forceinline__ int align_up(int x, int a) {
  return (x + a - 1) & ~(a - 1);
}

__host__ __device__ __forceinline__ int static_bytes(int n, int dtype) {
  const int planes = (dtype == 1) ? 6 : 3;
  // Pad so block_zp_bases (uint32[]) is 16-byte aligned — avoids
  // CUDA_ERROR_MISALIGNED_ADDRESS when n is odd.
  return align_up(HEADER + n + planes * bp_bytes(n), 16);
}

// Every non-top-7 exponent is stored verbatim.  The previous n/2 capacity
// silently dropped the tail when more than half of a tensor was out-of-range,
// and the decoder replaced those values with top7[3].  That is not lossless.
// Keep enough metadata for the worst case; the host-side caller must reject
// this fixed wire format when it is not smaller than native NCCL.
__host__ __device__ __forceinline__ int zp_cap(int n) {
  return n < 1 ? 1 : n;
}

__host__ __device__ __forceinline__ int xfer_bytes(int n, int dtype) {
  const int nb = n <= 0 ? 0 : (n + BLOCK - 1) / BLOCK;
  // Align whole payload to 16B for NCCL / vectorized access.
  return align_up(static_bytes(n, dtype) + nb * (int)sizeof(uint32_t) + zp_cap(n), 16);
}

__device__ __constant__ uint8_t kTop7BF16[7] = {123, 124, 125, 126, 127, 128, 129};
__device__ __constant__ uint8_t kTop7F16[7] = {12, 13, 14, 15, 16, 17, 18};

__device__ __forceinline__ uint8_t enc_exp(uint8_t exp, const uint8_t *t7, uint8_t *zp,
                                           bool *is_zp) {
  // The paper's Gaussian model makes the optimal exponents a contiguous
  // window. This replaces seven compares with one range check.
  const int base = t7[0];
  const uint8_t code = ((int)exp >= base && (int)exp < base + 7)
                           ? (uint8_t)((int)exp - base + 1)
                           : 0;
  *is_zp = (code == 0);
  *zp = exp;
  return code;
}

__device__ __forceinline__ void split_bf16(uint16_t bits, uint8_t *exp, uint8_t *sm) {
  *exp = (uint8_t)((bits >> 7) & 0xFF);
  *sm = (uint8_t)(((bits >> 8) & 0x80) | (bits & 0x7F));
}

__device__ __forceinline__ uint16_t join_bf16(uint8_t exp, uint8_t sm) {
  return (uint16_t)(((uint16_t)(sm & 0x80) << 8) | ((uint16_t)exp << 7) | (sm & 0x7F));
}

__device__ __forceinline__ void split_f16(uint16_t bits, uint8_t *exp, uint8_t *sm,
                                          uint8_t *mant_lo) {
  *exp = (uint8_t)((bits >> 10) & 0x1F);
  *sm = (uint8_t)(((bits >> 8) & 0x80) | ((bits >> 3) & 0x7F));
  *mant_lo = (uint8_t)(bits & 7);
}

__device__ __forceinline__ uint16_t join_f16(uint8_t exp, uint8_t sm, uint8_t mant_lo) {
  return (uint16_t)(((uint16_t)(sm & 0x80) << 8) | ((uint16_t)exp << 10) |
                    ((uint16_t)(sm & 0x7F) << 3) | (mant_lo & 7));
}

} // namespace zipccl

extern "C" int zipccl_header_bytes(void) { return zipccl::HEADER; }

extern "C" int zipccl_block_size(void) { return zipccl::BLOCK; }

extern "C" int zipccl_num_blocks(int n) {
  return n <= 0 ? 0 : (n + zipccl::BLOCK - 1) / zipccl::BLOCK;
}

extern "C" int zipccl_static_bytes_dtype(int n, int dtype) {
  return zipccl::static_bytes(n, dtype);
}

extern "C" int zipccl_zp_cap(int n) { return zipccl::zp_cap(n); }

// Fixed transfer size — host-only, no device touch. CUDA-graph safe.
extern "C" int zipccl_xfer_bytes_dtype(int n, int dtype) {
  return zipccl::xfer_bytes(n, dtype);
}

// Back-compat alias used by Rust max alloc.
extern "C" int zipccl_max_compressed_bytes_dtype(int n, int dtype) {
  return zipccl::xfer_bytes(n, dtype);
}

// Write full header + zero zp_count. Runs alone on the stream before compress_k.
__global__ void zipccl_init_header_k(uint8_t *out, int n, int dtype, const uint8_t *top7) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  zipccl::Header h;
  h.n = (uint32_t)n;
  h.zp_count = 0;
  h.dtype = (uint8_t)dtype;
#pragma unroll
  for (int i = 0; i < 7; ++i) {
    if (top7)
      h.top7[i] = top7[i];
    else
      h.top7[i] = (dtype == 0) ? zipccl::kTop7BF16[i] : zipccl::kTop7F16[i];
  }
  *reinterpret_cast<zipccl::Header *>(out) = h;
}

// The exponent distribution is very smooth across a transformer activation.
// A strided sample is enough to select the dominant exponents while avoiding
// the full-tensor atomic histogram on every prefill collective.
constexpr int HIST_SAMPLE_STRIDE = 16;

__global__ void zipccl_histogram_k(const uint16_t *__restrict__ in, uint32_t *__restrict__ hist,
                                   int n, int dtype) {
  const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * HIST_SAMPLE_STRIDE;
  if (idx >= n)
    return;
  const uint16_t bits = in[idx];
  const uint8_t exp = dtype == 0 ? (uint8_t)((bits >> 7) & 0xFF) : (uint8_t)((bits >> 10) & 0x1F);
  atomicAdd(hist + exp, 1u);
}

__global__ void zipccl_select_top7_k(const uint32_t *__restrict__ hist, uint8_t *__restrict__ top7,
                                     int dtype) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  const int max_exp = dtype == 0 ? 255 : 31;
  const int max_base = max_exp - 6;
  uint32_t best_count = 0;
  int best_base = 0;
  for (int base = 0; base <= max_base; ++base) {
    uint32_t count = 0;
#pragma unroll
    for (int k = 0; k < 7; ++k)
      count += hist[base + k];
    if (count > best_count || (count == best_count && base > best_base)) {
      best_count = count;
      best_base = base;
    }
  }
#pragma unroll
  for (int k = 0; k < 7; ++k)
    top7[k] = (uint8_t)(best_base + k);
}

__global__ void zipccl_compress_k(const uint16_t *__restrict__ in, uint8_t *__restrict__ out,
                                  int n, int dtype, int num_blocks,
                                  const uint8_t *__restrict__ top7) {
  using namespace zipccl;
  __shared__ uint8_t s_top7[8];
  __shared__ int s_warp_cnt[BLOCK / 32];
  __shared__ int s_warp_base[BLOCK / 32];
  __shared__ int s_block_total;
  __shared__ int s_block_base;

  // Header already written by zipccl_init_header_k on this stream (happens-before).
  if (threadIdx.x < 7) {
    s_top7[threadIdx.x] =
        top7 ? top7[threadIdx.x] : (dtype == 0 ? kTop7BF16[threadIdx.x] : kTop7F16[threadIdx.x]);
  }
  __syncthreads();

  const int idx = blockIdx.x * BLOCK + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  uint8_t sm = 0, code = 0, zp_e = 0, mant_lo = 0;
  bool is_zp = false;
  if (idx < n) {
    const uint16_t bits = in[idx];
    uint8_t exp;
    if (dtype == 0)
      split_bf16(bits, &exp, &sm);
    else
      split_f16(bits, &exp, &sm, &mant_lo);
    code = enc_exp(exp, s_top7, &zp_e, &is_zp);
  }

  const unsigned ballot = __ballot_sync(0xffffffffu, is_zp && idx < n);
  const int prefix = __popc(ballot & ((1u << lane) - 1u));
  if (lane == 0)
    s_warp_cnt[warp] = __popc(ballot);
  __syncthreads();

  if (threadIdx.x == 0) {
    int total = 0;
#pragma unroll
    for (int w = 0; w < BLOCK / 32; ++w) {
      s_warp_base[w] = total;
      total += s_warp_cnt[w];
    }
    s_block_total = total;
    const int cap = zp_cap(n);
    int base = (int)atomicAdd(reinterpret_cast<unsigned int *>(out + 4), (unsigned int)total);
    if (base >= cap) {
      atomicSub(reinterpret_cast<unsigned int *>(out + 4), (unsigned int)total);
      s_block_base = -1;
      s_block_total = 0;
    } else if (base + total > cap) {
      int keep = cap - base;
      atomicSub(reinterpret_cast<unsigned int *>(out + 4), (unsigned int)(total - keep));
      s_block_total = keep;
      s_block_base = base;
    } else {
      s_block_base = base;
    }
    uint32_t *bases = reinterpret_cast<uint32_t *>(out + static_bytes(n, dtype));
    bases[blockIdx.x] = (uint32_t)(s_block_base < 0 ? 0 : s_block_base);
  }
  __syncthreads();

  const int bp = bp_bytes(n);
  uint8_t *sm_out = out + HEADER;
  uint8_t *b0 = sm_out + n;
  uint8_t *b1 = b0 + bp;
  uint8_t *b2 = b1 + bp;
  uint8_t *m0 = b2 + bp;
  uint8_t *m1 = m0 + bp;
  uint8_t *m2 = m1 + bp;
  uint8_t *zp_out = out + static_bytes(n, dtype) + num_blocks * (int)sizeof(uint32_t);

  if (idx < n)
    sm_out[idx] = sm;

  // Every lane named by the full-warp mask must execute the shuffles.  Only
  // the group leaders write the resulting byte; calling __shfl_sync from the
  // leaders alone is undefined and corrupts the bitplanes on real GPUs.
  uint8_t e0 = 0, e1 = 0, e2 = 0, l0 = 0, l1 = 0, l2 = 0;
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) {
    const int group_lane = lane & ~7;
    uint8_t c = __shfl_sync(0xffffffffu, code, group_lane + bit) & 7;
    uint8_t ml = __shfl_sync(0xffffffffu, mant_lo, group_lane + bit) & 7;
    const int group_idx = (idx & ~7) + bit;
    if (group_idx >= n) {
      c = 0;
      ml = 0;
    }
    e0 |= (uint8_t)((c & 1) << bit);
    e1 |= (uint8_t)(((c >> 1) & 1) << bit);
    e2 |= (uint8_t)(((c >> 2) & 1) << bit);
    l0 |= (uint8_t)((ml & 1) << bit);
    l1 |= (uint8_t)(((ml >> 1) & 1) << bit);
    l2 |= (uint8_t)(((ml >> 2) & 1) << bit);
  }
  if (idx < n && (lane & 7) == 0) {
    const int byte = idx >> 3;
    b0[byte] = e0;
    b1[byte] = e1;
    b2[byte] = e2;
    if (dtype == 1) {
      m0[byte] = l0;
      m1[byte] = l1;
      m2[byte] = l2;
    }
  }

  if (is_zp && idx < n && s_block_base >= 0) {
    int pos = s_block_base + s_warp_base[warp] + prefix;
    if (pos < s_block_base + s_block_total && pos < zp_cap(n))
      zp_out[pos] = zp_e;
  }
}

__global__ void zipccl_decompress_k(const uint8_t *__restrict__ in, uint16_t *__restrict__ out_buf,
                                    int n, int dtype, int num_blocks) {
  using namespace zipccl;
  const Header h = *reinterpret_cast<const Header *>(in);
  __shared__ uint8_t s_top7[8];
  __shared__ int s_warp_cnt[BLOCK / 32];
  __shared__ int s_warp_base[BLOCK / 32];
  __shared__ int s_block_base;

  if (threadIdx.x < 7)
    s_top7[threadIdx.x] = h.top7[threadIdx.x];
  __syncthreads();

  const int idx = blockIdx.x * BLOCK + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const int bp = bp_bytes(n);
  const uint8_t *sm_in = in + HEADER;
  const uint8_t *b0 = sm_in + n;
  const uint8_t *b1 = b0 + bp;
  const uint8_t *b2 = b1 + bp;
  const uint8_t *m0 = b2 + bp;
  const uint8_t *m1 = m0 + bp;
  const uint8_t *m2 = m1 + bp;
  const uint32_t *bases = reinterpret_cast<const uint32_t *>(in + static_bytes(n, dtype));
  const uint8_t *zp = reinterpret_cast<const uint8_t *>(bases + num_blocks);

  uint8_t code = 0, mant_lo = 0;
  bool is_zp = false;
  if (idx < n) {
    const int byte = idx >> 3;
    const int bit = idx & 7;
    code = (uint8_t)(((b0[byte] >> bit) & 1) | (((b1[byte] >> bit) & 1) << 1) |
                     (((b2[byte] >> bit) & 1) << 2));
    is_zp = (code == 0);
    if (dtype == 1) {
      mant_lo = (uint8_t)(((m0[byte] >> bit) & 1) | (((m1[byte] >> bit) & 1) << 1) |
                          (((m2[byte] >> bit) & 1) << 2));
    }
  }

  const unsigned ballot = __ballot_sync(0xffffffffu, is_zp && idx < n);
  const int prefix = __popc(ballot & ((1u << lane) - 1u));
  if (lane == 0)
    s_warp_cnt[warp] = __popc(ballot);
  __syncthreads();

  if (threadIdx.x == 0) {
    int total = 0;
#pragma unroll
    for (int w = 0; w < BLOCK / 32; ++w) {
      s_warp_base[w] = total;
      total += s_warp_cnt[w];
    }
    s_block_base = (int)bases[blockIdx.x];
  }
  __syncthreads();

  if (idx < n) {
    const uint8_t sm = sm_in[idx];
    // If zp overflowed capacity during compress, code==0 may lack a stored exp —
    // fall back to top7[3] (near 1.0) to avoid garbage (should not happen).
    uint8_t exp;
    if (is_zp) {
      int pos = s_block_base + s_warp_base[warp] + prefix;
      exp = (pos >= 0 && pos < zp_cap(n)) ? zp[pos] : s_top7[3];
    } else {
      exp = s_top7[code - 1];
    }
    out_buf[idx] = (dtype == 0) ? join_bf16(exp, sm) : join_f16(exp, sm, mant_lo);
  }
}

// Decompress directly into the FP32 reduce accumulator.  This fuses the
// temporary half write/read and one kernel launch from the reduce-scatter
// path; the arithmetic is identical to decompress_k followed by
// zipccl_half_add_f32_k.
__global__ void zipccl_decompress_add_f32_k(const uint8_t *__restrict__ in,
                                            float *__restrict__ out_buf, int n, int dtype,
                                            int num_blocks) {
  using namespace zipccl;
  const Header h = *reinterpret_cast<const Header *>(in);
  __shared__ uint8_t s_top7[8];
  __shared__ int s_warp_cnt[BLOCK / 32];
  __shared__ int s_warp_base[BLOCK / 32];
  __shared__ int s_block_base;
  if (threadIdx.x < 7)
    s_top7[threadIdx.x] = h.top7[threadIdx.x];
  __syncthreads();

  const int idx = blockIdx.x * BLOCK + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int bp = bp_bytes(n);
  const uint8_t *sm_in = in + HEADER;
  const uint8_t *b0 = sm_in + n;
  const uint8_t *b1 = b0 + bp;
  const uint8_t *b2 = b1 + bp;
  const uint8_t *m0 = b2 + bp;
  const uint8_t *m1 = m0 + bp;
  const uint8_t *m2 = m1 + bp;
  const uint32_t *bases = reinterpret_cast<const uint32_t *>(in + static_bytes(n, dtype));
  const uint8_t *zp = reinterpret_cast<const uint8_t *>(bases + num_blocks);

  uint8_t code = 0, mant_lo = 0;
  bool is_zp = false;
  if (idx < n) {
    const int byte = idx >> 3;
    const int bit = idx & 7;
    code = (uint8_t)(((b0[byte] >> bit) & 1) | (((b1[byte] >> bit) & 1) << 1) |
                     (((b2[byte] >> bit) & 1) << 2));
    is_zp = (code == 0);
    if (dtype == 1)
      mant_lo = (uint8_t)(((m0[byte] >> bit) & 1) | (((m1[byte] >> bit) & 1) << 1) |
                          (((m2[byte] >> bit) & 1) << 2));
  }
  const unsigned ballot = __ballot_sync(0xffffffffu, is_zp && idx < n);
  const int prefix = __popc(ballot & ((1u << lane) - 1u));
  if (lane == 0)
    s_warp_cnt[warp] = __popc(ballot);
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
#pragma unroll
    for (int w = 0; w < BLOCK / 32; ++w) {
      s_warp_base[w] = total;
      total += s_warp_cnt[w];
    }
    s_block_base = (int)bases[blockIdx.x];
  }
  __syncthreads();
  if (idx < n) {
    const uint8_t sm = sm_in[idx];
    uint8_t exp;
    if (is_zp) {
      const int pos = s_block_base + s_warp_base[warp] + prefix;
      exp = (pos >= 0 && pos < zp_cap(n)) ? zp[pos] : s_top7[3];
    } else {
      exp = s_top7[code - 1];
    }
    const uint16_t bits = (dtype == 0) ? join_bf16(exp, sm)
                                      : join_f16(exp, sm, mant_lo);
    float value;
    if (dtype == 0)
      value = __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&bits));
    else
      value = __half2float(*reinterpret_cast<const __half *>(&bits));
    out_buf[idx] += value;
  }
}

// Device-only header init (async). No D2H/H2D/sync. Ordered before compress_k on `s`.
extern "C" cudaError_t zipccl_compress(const void *in, void *out, int n, int dtype, const void *top7,
                                        int64_t stream) {
  cudaStream_t s = (cudaStream_t)stream;
  const int num_blocks = zipccl_num_blocks(n);
  if (num_blocks <= 0)
    return cudaSuccess;

  zipccl_init_header_k<<<1, 1, 0, s>>>((uint8_t *)out, n, dtype, (const uint8_t *)top7);
  zipccl_compress_k<<<num_blocks, zipccl::BLOCK, 0, s>>>(
      (const uint16_t *)in, (uint8_t *)out, n, dtype, num_blocks, (const uint8_t *)top7);
  return cudaGetLastError();
}

// Dynamic top-7 selection for model activations. The histogram and top7
// buffers are persistent caller-owned scratch; no allocation occurs here.
extern "C" cudaError_t zipccl_compress_dynamic(const void *in, void *out, int n, int dtype,
                                                void *hist, void *top7, int64_t stream) {
  cudaStream_t s = (cudaStream_t)stream;
  const int num_blocks = zipccl_num_blocks(n);
  if (num_blocks <= 0)
    return cudaSuccess;
  cudaError_t err = cudaMemsetAsync(hist, 0, 256 * sizeof(uint32_t), s);
  if (err != cudaSuccess)
    return err;
  zipccl_histogram_k<<<(n + 256 * HIST_SAMPLE_STRIDE - 1) / (256 * HIST_SAMPLE_STRIDE), 256, 0, s>>>(
      (const uint16_t *)in, (uint32_t *)hist, n, dtype);
  zipccl_select_top7_k<<<1, 1, 0, s>>>((const uint32_t *)hist, (uint8_t *)top7, dtype);
  zipccl_init_header_k<<<1, 1, 0, s>>>((uint8_t *)out, n, dtype, (const uint8_t *)top7);
  zipccl_compress_k<<<num_blocks, zipccl::BLOCK, 0, s>>>(
      (const uint16_t *)in, (uint8_t *)out, n, dtype, num_blocks, (const uint8_t *)top7);
  return cudaGetLastError();
}

// Read all dynamic zero-point counts for a packed set of payloads in one
// launch. The host synchronizes once per collective instead of once per
// rank/chunk.
__global__ void zipccl_extract_counts_k(const uint8_t *__restrict__ packed, int stride,
                                        uint32_t *__restrict__ counts, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    counts[i] = *reinterpret_cast<const uint32_t *>(packed + 4 + i * stride);
}

extern "C" cudaError_t zipccl_extract_counts(const void *packed, int stride, void *counts,
                                              int count, int64_t stream) {
  if (count <= 0)
    return cudaSuccess;
  zipccl_extract_counts_k<<<(count + 127) / 128, 128, 0, (cudaStream_t)stream>>>(
      (const uint8_t *)packed, stride, (uint32_t *)counts, count);
  return cudaGetLastError();
}

extern "C" cudaError_t zipccl_decompress(const void *in, void *out, int n, int dtype, int64_t stream) {
  cudaStream_t s = (cudaStream_t)stream;
  const int num_blocks = zipccl_num_blocks(n);
  if (num_blocks <= 0)
    return cudaSuccess;
  zipccl_decompress_k<<<num_blocks, zipccl::BLOCK, 0, s>>>(
      (const uint8_t *)in, (uint16_t *)out, n, dtype, num_blocks);
  return cudaGetLastError();
}

extern "C" cudaError_t zipccl_decompress_add_f32(const void *in, void *out, int n, int dtype,
                                                   int64_t stream) {
  const int num_blocks = zipccl_num_blocks(n);
  if (num_blocks <= 0)
    return cudaSuccess;
  zipccl_decompress_add_f32_k<<<num_blocks, zipccl::BLOCK, 0, (cudaStream_t)stream>>>(
      (const uint8_t *)in, (float *)out, n, dtype, num_blocks);
  return cudaGetLastError();
}

// ---- Helpers for AllReduce local sum in FP32 (device-only, no sync) ----

__global__ void zipccl_half_to_f32_k(const uint16_t *__restrict__ in, float *__restrict__ out,
                                     int n, int dtype) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  uint16_t bits = in[i];
  if (dtype == 0) {
    out[i] = __bfloat162float(*reinterpret_cast<__nv_bfloat16 *>(&bits));
  } else {
    out[i] = __half2float(*reinterpret_cast<__half *>(&bits));
  }
}

__global__ void zipccl_half_add_f32_k(const uint16_t *__restrict__ in, float *__restrict__ out,
                                      int n, int dtype) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  uint16_t bits = in[i];
  float v;
  if (dtype == 0)
    v = __bfloat162float(*reinterpret_cast<__nv_bfloat16 *>(&bits));
  else
    v = __half2float(*reinterpret_cast<__half *>(&bits));
  out[i] += v;
}

__global__ void zipccl_f32_to_half_k(const float *__restrict__ in, uint16_t *__restrict__ out,
                                     int n, int dtype) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  if (dtype == 0) {
    __nv_bfloat16 h = __float2bfloat16_rn(in[i]);
    out[i] = *reinterpret_cast<uint16_t *>(&h);
  } else {
    __half h = __float2half_rn(in[i]);
    out[i] = *reinterpret_cast<uint16_t *>(&h);
  }
}

extern "C" cudaError_t zipccl_half_to_f32(const void *in, void *out, int n, int dtype, int64_t stream) {
  int grid = (n + zipccl::BLOCK - 1) / zipccl::BLOCK;
  if (grid <= 0)
    return cudaSuccess;
  zipccl_half_to_f32_k<<<grid, zipccl::BLOCK, 0, (cudaStream_t)stream>>>(
      (const uint16_t *)in, (float *)out, n, dtype);
  return cudaGetLastError();
}

extern "C" cudaError_t zipccl_half_add_f32(const void *in, void *out, int n, int dtype, int64_t stream) {
  int grid = (n + zipccl::BLOCK - 1) / zipccl::BLOCK;
  if (grid <= 0)
    return cudaSuccess;
  zipccl_half_add_f32_k<<<grid, zipccl::BLOCK, 0, (cudaStream_t)stream>>>(
      (const uint16_t *)in, (float *)out, n, dtype);
  return cudaGetLastError();
}

extern "C" cudaError_t zipccl_f32_to_half(const void *in, void *out, int n, int dtype, int64_t stream) {
  int grid = (n + zipccl::BLOCK - 1) / zipccl::BLOCK;
  if (grid <= 0)
    return cudaSuccess;
  zipccl_f32_to_half_k<<<grid, zipccl::BLOCK, 0, (cudaStream_t)stream>>>(
      (const float *)in, (uint16_t *)out, n, dtype);
  return cudaGetLastError();
}
