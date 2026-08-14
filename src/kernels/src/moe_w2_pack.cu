// Fused MXFP4/e2m1 → Moet sign-symmetric 2-bit plane pack (GPU, no host round-trip).
//
// Input packed:  [E, N, K/2] U8  (two e2m1 nibbles per byte)
// Input scales:  [E, N, K/32] U8 (UE8M0) — copied through unchanged
// Output planes: [E, N*K/4] U8   (four 2-bit codes per byte, row-major)
//
// Codebook: {0:-4, 1:-1, 2:+1, 3:+4} via Moet NIBBLE_TO_CODE LUT.

#include <cstdint>
#include <cuda_runtime.h>

__device__ __constant__ uint8_t kNibbleToCode[16] = {
    2, 2, 2, 2, 2, 3, 3, 3, // +0,.5,1,1.5,2,3,4,6
    1, 1, 1, 1, 1, 0, 0, 0, // negatives
};

// One thread emits one plane byte from two consecutive packed e2m1 bytes.
__global__ void moe_w2_pack_mxfp4_kernel(
    const uint8_t* __restrict__ packed, // [E, N, K/2]
    uint8_t* __restrict__ planes,       // [E, N*K/4]
    int E, int N, int K) {
  const int k_half = K / 2;
  const int packed_stride = N * k_half;
  const int plane_stride = N * K / 4;
  const int plane_elems = plane_stride;

  for (int e = blockIdx.y; e < E; e += gridDim.y) {
    const uint8_t* src = packed + static_cast<int64_t>(e) * packed_stride;
    uint8_t* dst = planes + static_cast<int64_t>(e) * plane_stride;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < plane_elems;
         i += blockDim.x * gridDim.x) {
      // Plane byte i packs codes for packed bytes 2*i and 2*i+1.
      const int p0 = 2 * i;
      const uint8_t b0 = src[p0];
      const uint8_t b1 = src[p0 + 1];
      const uint8_t c0 = kNibbleToCode[b0 & 0x0Fu];
      const uint8_t c1 = kNibbleToCode[b0 >> 4];
      const uint8_t c2 = kNibbleToCode[b1 & 0x0Fu];
      const uint8_t c3 = kNibbleToCode[b1 >> 4];
      dst[i] = static_cast<uint8_t>(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
    }
  }
}

__global__ void moe_w2_copy_scales_kernel(
    const uint8_t* __restrict__ in_scales,
    uint8_t* __restrict__ out_scales,
    int n_elems) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elems;
       i += blockDim.x * gridDim.x) {
    out_scales[i] = in_scales[i];
  }
}

extern "C" void moe_w2_pack_from_mxfp4(
    const uint8_t* packed,
    const uint8_t* scales_in,
    uint8_t* planes,
    uint8_t* scales_out,
    int E, int N, int K,
    int64_t stream_i64) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_i64);
  if (K % 4 != 0 || K % 32 != 0) {
    return;
  }
  const int plane_elems = N * K / 4;
  dim3 block(256);
  dim3 grid((plane_elems + 255) / 256, E);
  if (grid.x == 0) grid.x = 1;
  moe_w2_pack_mxfp4_kernel<<<grid, block, 0, stream>>>(packed, planes, E, N, K);

  const int scale_elems = E * N * (K / 32);
  dim3 sgrid((scale_elems + 255) / 256);
  if (sgrid.x == 0) sgrid.x = 1;
  moe_w2_copy_scales_kernel<<<sgrid, block, 0, stream>>>(
      scales_in, scales_out, scale_elems);
}
