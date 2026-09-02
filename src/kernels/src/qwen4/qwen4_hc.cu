// Qwen4 Gated Residual (Hyper-Connection) CUDA kernels.
// Reference: HuggingFace Qwen4ExpTextGatedResidual in modeling_qwen4_exp.py

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
  return __float2bfloat16(x);
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

// Grouped RMSNorm: input [hc, hidden], weight [hc*hidden], group_size=hidden.
__device__ void grouped_rms_norm(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int hc,
    int hidden,
    float eps) {
  for (int h = 0; h < hc; ++h) {
    const __nv_bfloat16* in_group = input + h * hidden;
    __nv_bfloat16* out_group = output + h * hidden;
    float sum_sq = 0.0f;
    for (int d = 0; d < hidden; ++d) {
      float v = bf16_to_f32(in_group[d]);
      sum_sq += v * v;
    }
    float rms = rsqrtf(sum_sq / (float)hidden + eps);
    for (int d = 0; d < hidden; ++d) {
      float w = 1.0f + bf16_to_f32(weight[h * hidden + d]);
      float v = bf16_to_f32(in_group[d]);
      out_group[d] = f32_to_bf16(w * v * rms);
    }
  }
}

// Read: mixed_input [hidden], injection_weights [hc]
__global__ void qwen4_hc_read_kernel(
    const __nv_bfloat16* hyper_input,
    const __nv_bfloat16* hc_norm_weight,
    const __nv_bfloat16* mix_down_weight,
    const __nv_bfloat16* mix_up_weight,
    const __nv_bfloat16* inject_weight,
    __nv_bfloat16* mixed_out,
    __nv_bfloat16* inject_out,
    __nv_bfloat16* normed_scratch,
    int hc,
    int hidden,
    int lowrank,
    float eps,
    int use_combine) {
  const int token = blockIdx.x;
  const int hc_hidden = hc * hidden;

  const __nv_bfloat16* x = hyper_input + token * hc_hidden;
  __nv_bfloat16* normed = normed_scratch + token * hc_hidden;

  grouped_rms_norm(x, hc_norm_weight, normed, hc, hidden, eps);

  // mix_down = silu(down @ normed / hc) -> [lowrank]
  float mix_down[512];
  for (int r = 0; r < lowrank; ++r) {
    float acc = 0.0f;
    for (int i = 0; i < hc_hidden; ++i) {
      acc += bf16_to_f32(mix_down_weight[r * hc_hidden + i]) *
             bf16_to_f32(normed[i]);
    }
    mix_down[r] = silu(acc / (float)hc);
  }

  // mix_up = sigmoid(up @ mix_down) -> [hc, hidden], fused into mixed output
  __nv_bfloat16* mixed = mixed_out + token * hidden;
  for (int d = 0; d < hidden; ++d) {
    float acc = 0.0f;
    for (int h = 0; h < hc; ++h) {
      const int idx = h * hidden + d;
      float mix_acc = 0.0f;
      for (int r = 0; r < lowrank; ++r) {
        mix_acc += bf16_to_f32(mix_up_weight[idx * lowrank + r]) * mix_down[r];
      }
      const float mix_val = 1.0f / (1.0f + expf(-mix_acc));
      acc += mix_val * bf16_to_f32(normed[idx]);
    }
    mixed[d] = f32_to_bf16(acc / (float)hc);
  }

  if (use_combine) {
    __nv_bfloat16* inject = inject_out + token * hc;
    for (int h = 0; h < hc; ++h) {
      float acc = 0.0f;
      for (int i = 0; i < hc_hidden; ++i) {
        acc += bf16_to_f32(inject_weight[h * hc_hidden + i]) *
               bf16_to_f32(normed[i]);
      }
      inject[h] = f32_to_bf16(2.0f / (1.0f + expf(-acc / (float)hc)));
    }
  }
}

// Write: out = hyper_input + block_out * injection_weights
__global__ void qwen4_hc_write_kernel(
    const __nv_bfloat16* hyper_input,
    const __nv_bfloat16* block_out,
    const __nv_bfloat16* inject_weights,
    __nv_bfloat16* out,
    int hc,
    int hidden) {
  const int token = blockIdx.x;
  const int hc_hidden = hc * hidden;
  const __nv_bfloat16* x_in = hyper_input + token * hc_hidden;
  const __nv_bfloat16* b_out = block_out + token * hidden;
  const __nv_bfloat16* inj = inject_weights + token * hc;
  __nv_bfloat16* x_out = out + token * hc_hidden;

  for (int h = 0; h < hc; ++h) {
    float gate = bf16_to_f32(inj[h]);
    for (int d = 0; d < hidden; ++d) {
      float v = bf16_to_f32(x_in[h * hidden + d]) +
                gate * bf16_to_f32(b_out[d]);
      x_out[h * hidden + d] = f32_to_bf16(v);
    }
  }
}

}  // namespace

extern "C" int qwen4_hc_read(
    const void* hyper_input,
    const void* hc_norm_weight,
    const void* mix_down_weight,
    const void* mix_up_weight,
    const void* inject_weight,
    void* mixed_out,
    void* inject_out,
    void* normed_scratch,
    int seq_len,
    int hc,
    int hidden,
    int lowrank,
    float eps,
    int use_combine,
    int64_t stream_) {
  if (hc > 16 || hidden > 8192 || lowrank > 512) return -1;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  qwen4_hc_read_kernel<<<seq_len, 1, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(hyper_input),
      static_cast<const __nv_bfloat16*>(hc_norm_weight),
      static_cast<const __nv_bfloat16*>(mix_down_weight),
      static_cast<const __nv_bfloat16*>(mix_up_weight),
      static_cast<const __nv_bfloat16*>(inject_weight),
      static_cast<__nv_bfloat16*>(mixed_out),
      static_cast<__nv_bfloat16*>(inject_out),
      static_cast<__nv_bfloat16*>(normed_scratch),
      hc, hidden, lowrank, eps, use_combine);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int qwen4_hc_write(
    const void* hyper_input,
    const void* block_out,
    const void* inject_weights,
    void* out,
    int seq_len,
    int hc,
    int hidden,
    int64_t stream_) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  qwen4_hc_write_kernel<<<seq_len, 1, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(hyper_input),
      static_cast<const __nv_bfloat16*>(block_out),
      static_cast<const __nv_bfloat16*>(inject_weights),
      static_cast<__nv_bfloat16*>(out),
      hc, hidden);
  return static_cast<int>(cudaGetLastError());
}
