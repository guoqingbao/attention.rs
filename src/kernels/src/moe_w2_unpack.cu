// Moet-style 2-bit expert plane → e4m3 + F32 UE8M0 scales (row-major packing).
#include <cstdint>
#include <cuda_runtime.h>
#include "attention/dtype_fp8.cuh"

static __device__ __forceinline__ uint8_t code_to_e4m3(uint8_t code) {
  // PRMT LUT 0x4838B8C8 → bytes {-4,-1,+1,+4} as e4m3
  constexpr uint32_t LUT = 0x4838B8C8u;
  return static_cast<uint8_t>((LUT >> (8u * code)) & 0xFFu);
}

static __device__ __forceinline__ float ue8m0_to_f32(uint8_t s) {
  if (s == 0) return 0.f;
  return ldexpf(1.f, static_cast<int>(s) - 127);
}

__global__ void moe_w2_unpack_kernel(
    const uint8_t* __restrict__ planes,
    const uint8_t* __restrict__ scales,
    uint8_t* __restrict__ out_w,
    float* __restrict__ out_s,
    int E, int N, int K) {
  const int plane_stride = N * K / 4;
  const int scale_stride = N * (K / 32);
  const int w_stride = N * K;

  for (int e = blockIdx.y; e < E; e += gridDim.y) {
    const uint8_t* plane = planes + e * plane_stride;
    const uint8_t* scale = scales + e * scale_stride;
    uint8_t* w = out_w + e * w_stride;
    float* s = out_s + e * scale_stride;

    // Unpack weights
    const int nbytes = plane_stride;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nbytes;
         i += blockDim.x * gridDim.x) {
      const uint8_t b = plane[i];
      w[4 * i + 0] = code_to_e4m3(b & 3u);
      w[4 * i + 1] = code_to_e4m3((b >> 2) & 3u);
      w[4 * i + 2] = code_to_e4m3((b >> 4) & 3u);
      w[4 * i + 3] = code_to_e4m3((b >> 6) & 3u);
    }

    // Unpack scales
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < scale_stride;
         i += blockDim.x * gridDim.x) {
      s[i] = ue8m0_to_f32(scale[i]);
    }
  }
}

extern "C" void moe_w2_unpack_to_fp8(
    const uint8_t* planes,
    const uint8_t* scales,
    uint8_t* out_w,
    float* out_s,
    int E, int N, int K,
    int64_t stream_i64) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_i64);
  dim3 block(256);
  dim3 grid((N * K / 4 + 255) / 256, E);
  if (grid.x == 0) grid.x = 1;
  moe_w2_unpack_kernel<<<grid, block, 0, stream>>>(
      planes, scales, out_w, out_s, E, N, K);
}

// Unpack only the experts listed in `expert_ids` [U] → contiguous [U, N, K].
// Used by decode (U ≪ E) to avoid materializing all experts every step.
__global__ void moe_w2_unpack_by_ids_kernel(
    const uint8_t* __restrict__ planes,
    const uint8_t* __restrict__ scales,
    const int* __restrict__ expert_ids,
    uint8_t* __restrict__ out_w,
    float* __restrict__ out_s,
    int U, int N, int K) {
  const int plane_stride = N * K / 4;
  const int scale_stride = N * (K / 32);
  const int w_stride = N * K;

  for (int u = blockIdx.y; u < U; u += gridDim.y) {
    const int e = expert_ids[u];
    if (e < 0) continue;
    const uint8_t* plane = planes + e * plane_stride;
    const uint8_t* scale = scales + e * scale_stride;
    uint8_t* w = out_w + u * w_stride;
    float* s = out_s + u * scale_stride;

    const int nbytes = plane_stride;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nbytes;
         i += blockDim.x * gridDim.x) {
      const uint8_t b = plane[i];
      w[4 * i + 0] = code_to_e4m3(b & 3u);
      w[4 * i + 1] = code_to_e4m3((b >> 2) & 3u);
      w[4 * i + 2] = code_to_e4m3((b >> 4) & 3u);
      w[4 * i + 3] = code_to_e4m3((b >> 6) & 3u);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < scale_stride;
         i += blockDim.x * gridDim.x) {
      s[i] = ue8m0_to_f32(scale[i]);
    }
  }
}

extern "C" void moe_w2_unpack_by_ids_to_fp8(
    const uint8_t* planes,
    const uint8_t* scales,
    const int* expert_ids,
    uint8_t* out_w,
    float* out_s,
    int U, int N, int K,
    int64_t stream_i64) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_i64);
  dim3 block(256);
  dim3 grid((N * K / 4 + 255) / 256, U);
  if (grid.x == 0) grid.x = 1;
  if (U <= 0) return;
  moe_w2_unpack_by_ids_kernel<<<grid, block, 0, stream>>>(
      planes, scales, expert_ids, out_w, out_s, U, N, K);
}

// W2 uses the same per-token group-128 FP8 activation quantization as the
// reference Moet path.  The generic WMMA fallback consumes BF16 activations,
// so materialize the quantize/dequantize result before that GEMM.
__global__ void moe_w2_dequantize_activation_fp8_kernel(
    const uint8_t* __restrict__ input_q,
    const float* __restrict__ input_scales,
    __nv_bfloat16* __restrict__ output,
    int rows, int K) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * K;
  if (idx >= total) return;
  const int group = (idx % K) / 128;
  const float scale = input_scales[(idx / K) * (K / 128) + group];
  output[idx] = __float2bfloat16(
      vllm::fp8::dispatch_fp8_to_float(input_q[idx]) * scale);
}

extern "C" void moe_w2_dequantize_activation_fp8(
    const uint8_t* input_q,
    const float* input_scales,
    void* output,
    int rows,
    int K,
    int64_t stream_i64) {
  if (rows <= 0 || K <= 0 || (K % 128) != 0) return;
  const int total = rows * K;
  moe_w2_dequantize_activation_fp8_kernel<<<(total + 255) / 256, 256,
                                             0,
                                             reinterpret_cast<cudaStream_t>(stream_i64)>>>(
      input_q, input_scales, reinterpret_cast<__nv_bfloat16*>(output), rows, K);
}

// DeepSeek V4's W2 activation contract.  This is deliberately separate from
// the generic SiLU path: V4 clamps gate only on the upper side, clamps up
// symmetrically, rounds the product to BF16, and only then performs the W2
// activation quantization / GEMM step.
//
// SiLU must be NaN-safe: the naive `x/(1+exp(-x))` form yields NaN for
// x=-Inf (and can poison router logits → garbage expert ids on decode).
__global__ void moe_w2_swiglu_clamp_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ output,
    int rows, int hidden, float limit) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * hidden;
  if (idx >= total) return;

  const int row = idx / hidden;
  const int col = idx - row * hidden;
  const int stride = hidden * 2;
  float gate = __bfloat162float(gate_up[row * stride + col]);
  float up = __bfloat162float(gate_up[row * stride + hidden + col]);
  if (limit > 0.0f) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
  }
  // Stable SiLU: clamp the exp argument so we never hit Inf/NaN edge cases.
  float gate_exp = fminf(fmaxf(gate, -80.0f), 80.0f);
  float silu_gate = gate / (1.0f + expf(-gate_exp));
  float out = silu_gate * up;
  if (!isfinite(out)) {
    out = 0.0f;
  }
  output[idx] = __float2bfloat16(out);
}

extern "C" void moe_w2_swiglu_clamp_bf16(
    const void* gate_up,
    void* output,
    int rows,
    int hidden,
    float limit,
    int64_t stream_i64) {
  if (rows <= 0 || hidden <= 0) return;
  const int total = rows * hidden;
  moe_w2_swiglu_clamp_bf16_kernel<<<(total + 255) / 256, 256,
                                    0,
                                    reinterpret_cast<cudaStream_t>(stream_i64)>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate_up),
      reinterpret_cast<__nv_bfloat16*>(output), rows, hidden, limit);
}
