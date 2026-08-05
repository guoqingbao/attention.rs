#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mutex>

namespace {

constexpr int kDsCompressorMaxDevices = 16;

__device__ __forceinline__ bool ds_compressor_should_emit(int start_pos, int ratio) {
  return ratio > 0 && ((start_pos + 1) % ratio) == 0;
}

struct DsCompressorGemmScratch {
  cublasHandle_t handle = nullptr;
  std::mutex mutex;
};

DsCompressorGemmScratch g_ds_compressor_gemm_scratch[kDsCompressorMaxDevices];

cudaError_t ds_compressor_scratch_for_device(DsCompressorGemmScratch **out) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  if (device < 0 || device >= kDsCompressorMaxDevices) return cudaErrorInvalidDevice;
  *out = &g_ds_compressor_gemm_scratch[device];
  return cudaSuccess;
}

cudaError_t ds_compressor_ensure_handle(DsCompressorGemmScratch &scratch) {
  if (scratch.handle != nullptr) return cudaSuccess;
  cublasStatus_t status = cublasCreate(&scratch.handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    scratch.handle = nullptr;
    return cudaErrorUnknown;
  }
  // BF16 inputs are accumulated into F32.  Tensor-op math is safe here and
  // matches OpenInfer; unlike the old Candle path, no BF16->F32 host-side
  // expansion is needed and no TF32 conversion is involved.
  status = cublasSetMathMode(scratch.handle, CUBLAS_TENSOR_OP_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(scratch.handle);
    scratch.handle = nullptr;
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

}  // namespace

static __device__ __forceinline__ void ds_apply_rope_pair(
    __nv_bfloat16 *x,
    int offset,
    float cos_value,
    float sin_value,
    bool inverse) {
  float x0 = __bfloat162float(x[offset]);
  float x1 = __bfloat162float(x[offset + 1]);
  float c = cos_value;
  float s = sin_value;
  if (inverse) s = -s;
  float out0 = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
  float out1 = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
  x[offset] = __float2bfloat16(out0);
  x[offset + 1] = __float2bfloat16(out1);
}

__global__ void ds_apply_rope_hidden_from_pos_kernel(
    __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ cos_cache,
    const float *__restrict__ sin_cache,
    const int64_t *__restrict__ positions,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int position_offset,
    int inverse) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  if (pair >= total_pairs) return;

  int rotary_pair = pair % (rotary_dim / 2);
  int tmp = pair / (rotary_dim / 2);
  int head = tmp % local_heads;
  int token = tmp / local_heads;
  int nope_dim = head_dim - rotary_dim;
  int start_pos = static_cast<int>(positions[token]) + position_offset;
  int pos = start_pos + token;
  int offset =
      token * local_heads * head_dim + head * head_dim + nope_dim + 2 * rotary_pair;
  ds_apply_rope_pair(
      x, offset, cos_cache[pos * (rotary_dim / 2) + rotary_pair],
      sin_cache[pos * (rotary_dim / 2) + rotary_pair], inverse != 0);
}

__global__ void ds_compressor_decode_project_from_pos_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ wkv,
    const __nv_bfloat16 *__restrict__ wgate,
    const float *__restrict__ ape,
    float *__restrict__ kv_state,
    float *__restrict__ score_state,
    const int64_t *__restrict__ positions,
    int hidden_dim,
    int out_dim,
    int ratio,
    int state_offset) {
  int dim = blockIdx.x;
  int tid = threadIdx.x;
  if (dim >= out_dim) return;

  extern __shared__ float scratch[];
  float *kv_scratch = scratch;
  float *score_scratch = scratch + blockDim.x;
  float kv_partial = 0.0f;
  float score_partial = 0.0f;
  for (int k = tid; k < hidden_dim; k += blockDim.x) {
    float xv = __bfloat162float(x[k]);
    kv_partial += xv * __bfloat162float(wkv[dim * hidden_dim + k]);
    score_partial += xv * __bfloat162float(wgate[dim * hidden_dim + k]);
  }
  kv_scratch[tid] = kv_partial;
  score_scratch[tid] = score_partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      kv_scratch[tid] += kv_scratch[tid + stride];
      score_scratch[tid] += score_scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    int start_pos = static_cast<int>(positions[0]);
    int local_pos = start_pos % ratio;
    int state_row = state_offset + local_pos;
    kv_state[state_row * out_dim + dim] = kv_scratch[0];
    score_state[state_row * out_dim + dim] =
        score_scratch[0] + ape[local_pos * out_dim + dim];
  }
}

__global__ void ds_apply_rope_hidden_kernel(
    __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ cos_cache,
    const float *__restrict__ sin_cache,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int start_pos,
    int inverse) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  if (pair >= total_pairs) return;

  int rotary_pair = pair % (rotary_dim / 2);
  int tmp = pair / (rotary_dim / 2);
  int head = tmp % local_heads;
  int token = tmp / local_heads;
  int nope_dim = head_dim - rotary_dim;
  int pos = start_pos + token;
  int offset =
      token * local_heads * head_dim + head * head_dim + nope_dim + 2 * rotary_pair;
  ds_apply_rope_pair(
      x, offset, cos_cache[pos * (rotary_dim / 2) + rotary_pair],
      sin_cache[pos * (rotary_dim / 2) + rotary_pair], inverse != 0);
}

__global__ void ds_apply_rope_hidden_strided_kernel(
    __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ cos_cache,
    const float *__restrict__ sin_cache,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int start_pos,
    int position_stride,
    int inverse) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  if (pair >= total_pairs) return;

  int rotary_pair = pair % (rotary_dim / 2);
  int tmp = pair / (rotary_dim / 2);
  int head = tmp % local_heads;
  int token = tmp / local_heads;
  int nope_dim = head_dim - rotary_dim;
  int pos = start_pos + token * position_stride;
  int offset =
      token * local_heads * head_dim + head * head_dim + nope_dim + 2 * rotary_pair;
  ds_apply_rope_pair(
      x, offset, cos_cache[pos * (rotary_dim / 2) + rotary_pair],
      sin_cache[pos * (rotary_dim / 2) + rotary_pair], inverse != 0);
}

__global__ void ds_compressor_norm_serial_kernel(
    const float *__restrict__ weighted,
    const __nv_bfloat16 *__restrict__ norm,
    __nv_bfloat16 *__restrict__ out,
    int compressed_len,
    int head_dim,
    float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = compressed_len * head_dim;
  if (idx >= total) return;

  int dim = idx % head_dim;
  int compressed = idx / head_dim;
  float sum_sq = 0.0f;
  for (int k = 0; k < head_dim; ++k) {
    // Official Compressor.forward casts pooled FP32 KV to the model dtype
    // (BF16) before RMSNorm. Preserve weighted[] in FP32 for callers/state,
    // but reproduce that explicit rounding boundary for normalization.
    float value = __bfloat162float(
        __float2bfloat16(weighted[compressed * head_dim + k]));
    sum_sq += value * value;
  }
  float inv_rms = rsqrtf(sum_sq / head_dim + eps);
  float value = __bfloat162float(
                    __float2bfloat16(weighted[compressed * head_dim + dim])) *
                inv_rms * __bfloat162float(norm[dim]);
  out[compressed * head_dim + dim] = __float2bfloat16(value);
}

__global__ void ds_compressor_norm_serial_gated_kernel(
    const float *__restrict__ weighted,
    const __nv_bfloat16 *__restrict__ norm,
    __nv_bfloat16 *__restrict__ out,
    int compressed_len,
    int head_dim,
    float eps,
    const int64_t *__restrict__ positions,
    int ratio) {
  int start_pos = static_cast<int>(positions[0]);
  if (!ds_compressor_should_emit(start_pos, ratio)) return;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = compressed_len * head_dim;
  if (idx >= total) return;

  int dim = idx % head_dim;
  int compressed = idx / head_dim;
  float sum_sq = 0.0f;
  for (int k = 0; k < head_dim; ++k) {
    float value = __bfloat162float(
        __float2bfloat16(weighted[compressed * head_dim + k]));
    sum_sq += value * value;
  }
  float inv_rms = rsqrtf(sum_sq / head_dim + eps);
  float value = __bfloat162float(
                    __float2bfloat16(weighted[compressed * head_dim + dim])) *
                inv_rms * __bfloat162float(norm[dim]);
  out[compressed * head_dim + dim] = __float2bfloat16(value);
}

// Fused non-overlap-compressor prefill epilogue. Consumes per-token
// (score, value) FP32 tensors produced by upstream X @ W^T projections
// (gate scores, kv values), and for each compressed row:
//   1) streams over the `ratio` non-overlap routes per (compressed, dim) cell
//      with online softmax (two passes per dim: max, then denom + weighted
//      sum together) so `ratio` is unbounded,
//   2) adds the per-route APE bias and runs numerically-stable softmax,
//   3) writes the FP32 weighted output,
//   4) reduces the row's sum-of-squares to inv_rms and emits the BF16 RMSNormed
//      output -- all in one launch, without round-tripping `weighted` through
//      HBM between softmax and RMSNorm.
//
// scores_in / values_in are row-major (seq_len, head_dim) FP32 tensors.
// sv_n_stride is the row stride, equal to head_dim.
//
// Launch: 1 block per compressed position, blockDim.x covers head_dim with a
// strided loop. Block must be a multiple of warpSize and at most 1024 threads.
__global__ void ds_compressor_nonoverlap_fused_epilogue_kernel(
    const float *__restrict__ scores_in,
    const float *__restrict__ values_in,
    const float *__restrict__ ape,
    const __nv_bfloat16 *__restrict__ norm,
    float *__restrict__ weighted,
    __nv_bfloat16 *__restrict__ out,
    int compressed_len,
    int head_dim,
    int ratio,
    int sv_n_stride,
    float eps) {
  int c = blockIdx.x;
  if (c >= compressed_len) return;
  int tid = threadIdx.x;
  int n_block = blockDim.x;

  constexpr float neg_inf = -3.4028234663852886e38f;

  float sum_sq_local = 0.0f;

  // Streaming online softmax + weighted sum so ratio is unbounded (DSV4 uses
  // ratio up to 128). Two passes over the routes per d:
  //   1) find max(score) for numerical stability.
  //   2) accumulate softmax denominator + weighted-value numerator together.
  // Each thread handles its own d strides; routes are read from L2-cached
  // FP32 sv_buf so the double read is bandwidth-cheap.
  for (int d = tid; d < head_dim; d += n_block) {
    float m = neg_inf;
    for (int r = 0; r < ratio; ++r) {
      int token = c * ratio + r;
      int offset = token * sv_n_stride + d;
      float s = scores_in[offset] + ape[r * head_dim + d];
      m = fmaxf(m, s);
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int r = 0; r < ratio; ++r) {
      int token = c * ratio + r;
      int offset = token * sv_n_stride + d;
      float s = scores_in[offset] + ape[r * head_dim + d];
      float v = values_in[offset];
      float p = __expf(s - m);
      denom += p;
      acc += p * v;
    }
    float w = acc / denom;
    weighted[c * head_dim + d] = w;
    float rounded = __bfloat162float(__float2bfloat16(w));
    sum_sq_local += rounded * rounded;
  }

  // Block-wide reduction of sum_sq via warp-shfl + smem (mirrors overlap epilogue).
  __shared__ float warp_sums[32];
  int lane = tid & 31;
  int warp = tid >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sum_sq_local += __shfl_down_sync(0xffffffffu, sum_sq_local, off);
  }
  if (lane == 0) warp_sums[warp] = sum_sq_local;
  __syncthreads();

  int n_warps = (n_block + 31) >> 5;
  float total = (tid < n_warps) ? warp_sums[tid] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, off);
    }
  }
  __shared__ float total_sum;
  if (tid == 0) total_sum = total;
  __syncthreads();

  float inv_rms = rsqrtf(total_sum / static_cast<float>(head_dim) + eps);

  for (int d = tid; d < head_dim; d += n_block) {
    float w = __bfloat162float(
        __float2bfloat16(weighted[c * head_dim + d]));
    float ns = __bfloat162float(norm[d]);
    out[c * head_dim + d] = __float2bfloat16(w * inv_rms * ns);
  }
}

__global__ void ds_compressor_overlap_fused_epilogue_kernel(
    const float *__restrict__ scores_in,
    const float *__restrict__ values_in,
    const float *__restrict__ ape,
    const __nv_bfloat16 *__restrict__ norm,
    float *__restrict__ weighted,
    __nv_bfloat16 *__restrict__ out,
    int compressed_len,
    int head_dim,
    int sv_n_stride,
    float eps) {
  int c = blockIdx.x;
  if (c >= compressed_len) return;
  int tid = threadIdx.x;
  int n_block = blockDim.x;

  constexpr int ratio = 4;
  constexpr int routes = 8;
  constexpr float neg_inf = -3.4028234663852886e38f;

  float sum_sq_local = 0.0f;

  for (int d = tid; d < head_dim; d += n_block) {
    float scores[routes];
    float values[routes];
#pragma unroll
    for (int r = 0; r < routes; ++r) {
      bool valid;
      int token;
      int out_dim;
      int ape_dim;
      if (r < ratio) {
        valid = c > 0;
        token = (c - 1) * ratio + r;
        out_dim = d;
        ape_dim = r * (2 * head_dim) + d;
      } else {
        int lr = r - ratio;
        valid = true;
        token = c * ratio + lr;
        out_dim = head_dim + d;
        ape_dim = lr * (2 * head_dim) + head_dim + d;
      }
      if (valid) {
        int offset = token * sv_n_stride + out_dim;
        scores[r] = scores_in[offset] + ape[ape_dim];
        values[r] = values_in[offset];
      } else {
        scores[r] = neg_inf;
        values[r] = 0.0f;
      }
    }

    float m = scores[0];
#pragma unroll
    for (int r = 1; r < routes; ++r) m = fmaxf(m, scores[r]);

    float denom = 0.0f;
    float acc = 0.0f;
#pragma unroll
    for (int r = 0; r < routes; ++r) {
      float p = __expf(scores[r] - m);
      denom += p;
      acc += p * values[r];
    }
    float w = acc / denom;
    weighted[c * head_dim + d] = w;
    float rounded = __bfloat162float(__float2bfloat16(w));
    sum_sq_local += rounded * rounded;
  }

  // Block-wide reduction of sum_sq via warp-shfl + smem.
  __shared__ float warp_sums[32];
  int lane = tid & 31;
  int warp = tid >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sum_sq_local += __shfl_down_sync(0xffffffffu, sum_sq_local, off);
  }
  if (lane == 0) warp_sums[warp] = sum_sq_local;
  __syncthreads();

  int n_warps = (n_block + 31) >> 5;
  float total = (tid < n_warps) ? warp_sums[tid] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, off);
    }
  }
  __shared__ float total_sum;
  if (tid == 0) total_sum = total;
  __syncthreads();

  float inv_rms = rsqrtf(total_sum / static_cast<float>(head_dim) + eps);

  for (int d = tid; d < head_dim; d += n_block) {
    float w = __bfloat162float(
        __float2bfloat16(weighted[c * head_dim + d]));
    float ns = __bfloat162float(norm[d]);
    out[c * head_dim + d] = __float2bfloat16(w * inv_rms * ns);
  }
}

__global__ void ds_compressor_decode_project_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ wkv,
    const __nv_bfloat16 *__restrict__ wgate,
    const float *__restrict__ ape,
    float *__restrict__ kv_state,
    float *__restrict__ score_state,
    int start_pos,
    int hidden_dim,
    int out_dim,
    int ratio,
    int state_offset) {
  int dim = blockIdx.x;
  int tid = threadIdx.x;
  if (dim >= out_dim) return;

  extern __shared__ float scratch[];
  float *kv_scratch = scratch;
  float *score_scratch = scratch + blockDim.x;
  float kv_partial = 0.0f;
  float score_partial = 0.0f;
  for (int k = tid; k < hidden_dim; k += blockDim.x) {
    float xv = __bfloat162float(x[k]);
    kv_partial += xv * __bfloat162float(wkv[dim * hidden_dim + k]);
    score_partial += xv * __bfloat162float(wgate[dim * hidden_dim + k]);
  }
  kv_scratch[tid] = kv_partial;
  score_scratch[tid] = score_partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      kv_scratch[tid] += kv_scratch[tid + stride];
      score_scratch[tid] += score_scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    int local_pos = start_pos % ratio;
    int state_row = state_offset + local_pos;
    kv_state[state_row * out_dim + dim] = kv_scratch[0];
    score_state[state_row * out_dim + dim] =
        score_scratch[0] + ape[local_pos * out_dim + dim];
  }
}

__global__ void ds_compressor_decode_project_serial_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ wkv,
    const __nv_bfloat16 *__restrict__ wgate,
    const float *__restrict__ ape,
    float *__restrict__ kv_state,
    float *__restrict__ score_state,
    int start_pos,
    int hidden_dim,
    int out_dim,
    int ratio,
    int state_offset) {
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= out_dim) return;

  float kv_sum = 0.0f;
  float score_sum = 0.0f;
  for (int k = 0; k < hidden_dim; ++k) {
    float xv = __bfloat162float(x[k]);
    kv_sum += xv * __bfloat162float(wkv[dim * hidden_dim + k]);
    score_sum += xv * __bfloat162float(wgate[dim * hidden_dim + k]);
  }

  int local_pos = start_pos % ratio;
  int state_row = state_offset + local_pos;
  kv_state[state_row * out_dim + dim] = kv_sum;
  score_state[state_row * out_dim + dim] = score_sum + ape[local_pos * out_dim + dim];
}

__global__ void ds_compressor_nonoverlap_decode_weighted_kernel(
    const float *__restrict__ kv_state,
    const float *__restrict__ score_state,
    float *__restrict__ weighted,
    int head_dim,
    int ratio) {
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= head_dim) return;

  float max_score = -3.4028234663852886e38f;
  for (int route = 0; route < ratio; ++route) {
    max_score = fmaxf(max_score, score_state[route * head_dim + dim]);
  }
  float denom = 0.0f;
  float acc = 0.0f;
  for (int route = 0; route < ratio; ++route) {
    float prob = expf(score_state[route * head_dim + dim] - max_score);
    denom += prob;
    acc += prob * kv_state[route * head_dim + dim];
  }
  weighted[dim] = acc / denom;
}

__global__ void ds_compressor_nonoverlap_decode_weighted_gated_kernel(
    const float *__restrict__ kv_state,
    const float *__restrict__ score_state,
    float *__restrict__ weighted,
    const int64_t *__restrict__ positions,
    int head_dim,
    int ratio) {
  int start_pos = static_cast<int>(positions[0]);
  if (!ds_compressor_should_emit(start_pos, ratio)) return;
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= head_dim) return;

  float max_score = -3.4028234663852886e38f;
  for (int route = 0; route < ratio; ++route) {
    max_score = fmaxf(max_score, score_state[route * head_dim + dim]);
  }
  float denom = 0.0f;
  float acc = 0.0f;
  for (int route = 0; route < ratio; ++route) {
    float prob = expf(score_state[route * head_dim + dim] - max_score);
    denom += prob;
    acc += prob * kv_state[route * head_dim + dim];
  }
  weighted[dim] = acc / denom;
}

__global__ void ds_compressor_overlap_decode_weighted_kernel(
    const float *__restrict__ kv_state,
    const float *__restrict__ score_state,
    float *__restrict__ weighted,
    int head_dim) {
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= head_dim) return;

  constexpr int ratio = 4;
  constexpr int routes = 8;
  int state_dim = 2 * head_dim;
  float route_scores[routes];
  float route_values[routes];
  for (int route = 0; route < routes; ++route) {
    if (route < ratio) {
      route_scores[route] = score_state[route * state_dim + dim];
      route_values[route] = kv_state[route * state_dim + dim];
    } else {
      int local = route - ratio;
      route_scores[route] = score_state[(ratio + local) * state_dim + head_dim + dim];
      route_values[route] = kv_state[(ratio + local) * state_dim + head_dim + dim];
    }
  }

  float max_score = -3.4028234663852886e38f;
  for (int route = 0; route < routes; ++route) {
    max_score = fmaxf(max_score, route_scores[route]);
  }
  float denom = 0.0f;
  float acc = 0.0f;
  for (int route = 0; route < routes; ++route) {
    float prob = expf(route_scores[route] - max_score);
    denom += prob;
    acc += prob * route_values[route];
  }
  weighted[dim] = acc / denom;
}

__global__ void ds_compressor_overlap_decode_weighted_gated_kernel(
    const float *__restrict__ kv_state,
    const float *__restrict__ score_state,
    float *__restrict__ weighted,
    const int64_t *__restrict__ positions,
    int head_dim) {
  int start_pos = static_cast<int>(positions[0]);
  constexpr int ratio = 4;
  if (!ds_compressor_should_emit(start_pos, ratio)) return;
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= head_dim) return;

  constexpr int routes = 8;
  int state_dim = 2 * head_dim;
  float route_scores[routes];
  float route_values[routes];
  for (int route = 0; route < routes; ++route) {
    if (route < ratio) {
      route_scores[route] = score_state[route * state_dim + dim];
      route_values[route] = kv_state[route * state_dim + dim];
    } else {
      int local = route - ratio;
      route_scores[route] = score_state[(ratio + local) * state_dim + head_dim + dim];
      route_values[route] = kv_state[(ratio + local) * state_dim + head_dim + dim];
    }
  }

  float max_score = -3.4028234663852886e38f;
  for (int route = 0; route < routes; ++route) {
    max_score = fmaxf(max_score, route_scores[route]);
  }
  float denom = 0.0f;
  float acc = 0.0f;
  for (int route = 0; route < routes; ++route) {
    float prob = expf(route_scores[route] - max_score);
    denom += prob;
    acc += prob * route_values[route];
  }
  weighted[dim] = acc / denom;
}

__global__ void ds_compressor_overlap_shift_kernel(
    float *__restrict__ kv_state,
    float *__restrict__ score_state,
    int state_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = 4 * state_dim;
  if (idx >= total) return;
  kv_state[idx] = kv_state[total + idx];
  score_state[idx] = score_state[total + idx];
}

__global__ void ds_compressor_overlap_shift_gated_kernel(
    float *__restrict__ kv_state,
    float *__restrict__ score_state,
    const int64_t *__restrict__ positions,
    int state_dim,
    int ratio) {
  int start_pos = static_cast<int>(positions[0]);
  if (((start_pos + 1) % ratio) != 0) return;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = ratio * state_dim;
  if (idx >= total) return;
  kv_state[idx] = kv_state[total + idx];
  score_state[idx] = score_state[total + idx];
}

extern "C" {

cudaError_t ds_apply_rope_hidden(
    __nv_bfloat16 *x,
    const float *cos_cache,
    const float *sin_cache,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int start_pos,
    int inverse,
    cudaStream_t stream) {
  if (x == nullptr || cos_cache == nullptr || sin_cache == nullptr || seq_len <= 0 ||
      local_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0 || start_pos < 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  int blocks = (total_pairs + threads - 1) / threads;
  ds_apply_rope_hidden_kernel<<<blocks, threads, 0, stream>>>(
      x, cos_cache, sin_cache, seq_len, local_heads, head_dim, rotary_dim, start_pos,
      inverse);
  return cudaGetLastError();
}

cudaError_t ds_apply_rope_hidden_strided(
    __nv_bfloat16 *x,
    const float *cos_cache,
    const float *sin_cache,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int start_pos,
    int position_stride,
    int inverse,
    cudaStream_t stream) {
  if (x == nullptr || cos_cache == nullptr || sin_cache == nullptr || seq_len <= 0 ||
      local_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0 || start_pos < 0 || position_stride <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  int blocks = (total_pairs + threads - 1) / threads;
  ds_apply_rope_hidden_strided_kernel<<<blocks, threads, 0, stream>>>(
      x, cos_cache, sin_cache, seq_len, local_heads, head_dim, rotary_dim, start_pos,
      position_stride, inverse);
  return cudaGetLastError();
}

// Row-major BF16 X @ W^T -> F32.  cuBLAS is invoked with the standard
// column-major swap-and-transpose layout so the output buffer can be consumed
// directly by the compressor epilogue as [seq_len, out_dim].
cudaError_t ds_compressor_bf16_linear_f32(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *weight,
    float *out,
    int seq_len,
    int in_dim,
    int out_dim,
    cudaStream_t stream) {
  if (x == nullptr || weight == nullptr || out == nullptr || seq_len <= 0 ||
      in_dim <= 0 || out_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  DsCompressorGemmScratch *scratch_ptr = nullptr;
  cudaError_t err = ds_compressor_scratch_for_device(&scratch_ptr);
  if (err != cudaSuccess) return err;
  DsCompressorGemmScratch &scratch = *scratch_ptr;
  std::lock_guard<std::mutex> lock(scratch.mutex);
  err = ds_compressor_ensure_handle(scratch);
  if (err != cudaSuccess) return err;
  if (cublasSetStream(scratch.handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasGemmEx(
      scratch.handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      out_dim,
      seq_len,
      in_dim,
      &alpha,
      weight,
      CUDA_R_16BF,
      in_dim,
      x,
      CUDA_R_16BF,
      in_dim,
      &beta,
      out,
      CUDA_R_32F,
      out_dim,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

// Non-overlap prefill compressor epilogue: consumes pre-computed FP32 score/value
// projections (X @ Wgate^T, X @ Wkv^T) and runs the fused epilogue that gathers
// the `ratio` routes per compressed token, softmaxes, writes `weighted`, and
// RMSNorms in place to BF16 `out`.
cudaError_t ds_compressor_nonoverlap_prefill_epilogue(
    const float *scores,
    const float *values,
    const float *ape,
    const __nv_bfloat16 *norm,
    float *weighted,
    __nv_bfloat16 *out,
    int seq_len,
    int head_dim,
    int ratio,
    float eps,
    cudaStream_t stream) {
  if (scores == nullptr || values == nullptr || ape == nullptr || norm == nullptr ||
      weighted == nullptr || out == nullptr) {
    return cudaErrorInvalidValue;
  }
  // ratio upper bound 128 matches DSV4's actual layer configuration
  // (`config.compress_ratios` reaches 128). The fused epilogue uses streaming
  // softmax so it has no compile-time route ceiling.
  // `seq_len` does NOT need to be a multiple of `ratio`: the epilogue reads
  // only the first `compressed_len * ratio` tokens, and any trailing partial
  // group is ignored. Required for online prompts whose prefill length is not
  // aligned to `ratio` (e.g. ratio=2 with seq_len=21).
  if (ratio <= 1 || ratio > 128 || seq_len < ratio || head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  const int compressed_len = seq_len / ratio;

  // Epilogue: one block per compressed token; threads collaborate over head_dim
  // with a block-wide warp-shuffle reduction for the RMS-norm sum_sq.
  int epilogue_threads = head_dim < 256 ? head_dim : 256;
  if (epilogue_threads > 1024) epilogue_threads = 1024;
  epilogue_threads = (epilogue_threads + 31) & ~31;
  if (epilogue_threads <= 0) epilogue_threads = 32;
  ds_compressor_nonoverlap_fused_epilogue_kernel<<<compressed_len, epilogue_threads, 0, stream>>>(
      scores, values, ape, norm, weighted, out,
      compressed_len, head_dim, ratio, /*sv_n_stride=*/head_dim, eps);
  return cudaGetLastError();
}

// Overlap prefill compressor epilogue: consumes pre-computed FP32 score/value
// projections (X @ Wgate^T, X @ Wkv^T at 2*head_dim) and runs the fused epilogue
// that gathers the 8 overlap routes, softmaxes, writes `weighted`, and RMSNorms
// in place to BF16 `out`.
cudaError_t ds_compressor_overlap_prefill_epilogue(
    const float *scores,
    const float *values,
    const float *ape,
    const __nv_bfloat16 *norm,
    float *weighted,
    __nv_bfloat16 *out,
    int seq_len,
    int head_dim,
    float eps,
    cudaStream_t stream) {
  if (scores == nullptr || values == nullptr || ape == nullptr || norm == nullptr ||
      weighted == nullptr || out == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (seq_len < 4 || head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  const int compressed_len = seq_len / 4;
  const int n = 2 * head_dim;

  // Block size: round head_dim up to a multiple of warpSize, capped at 1024.
  int epilogue_threads = head_dim;
  if (epilogue_threads < 32) epilogue_threads = 32;
  if (epilogue_threads > 1024) epilogue_threads = 1024;
  epilogue_threads = (epilogue_threads + 31) & ~31;
  ds_compressor_overlap_fused_epilogue_kernel<<<compressed_len, epilogue_threads, 0, stream>>>(
      scores, values, ape, norm, weighted, out,
      compressed_len, head_dim, n, eps);
  return cudaGetLastError();
}

cudaError_t ds_compressor_nonoverlap_decode_at(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *wkv,
    const __nv_bfloat16 *wgate,
    const float *ape,
    const __nv_bfloat16 *norm,
    float *kv_state,
    float *score_state,
    float *weighted,
    __nv_bfloat16 *out,
    int start_pos,
    int hidden_dim,
    int head_dim,
    int ratio,
    int state_offset,
    float eps,
    cudaStream_t stream) {
  if (start_pos < 0 || hidden_dim <= 0 || head_dim <= 0 || ratio <= 1 || ratio > 128 ||
      state_offset < 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  size_t project_shared = 2 * threads * sizeof(float);
  ds_compressor_decode_project_kernel<<<head_dim, threads, project_shared, stream>>>(
      x, wkv, wgate, ape, kv_state, score_state, start_pos, hidden_dim, head_dim, ratio,
      state_offset);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  bool should_compress = ((start_pos + 1) % ratio) == 0;
  if (!should_compress) return cudaSuccess;
  if (weighted == nullptr || out == nullptr) return cudaErrorInvalidValue;

  int blocks = (head_dim + threads - 1) / threads;
  float *kv_slot = kv_state + state_offset * head_dim;
  float *score_slot = score_state + state_offset * head_dim;
  ds_compressor_nonoverlap_decode_weighted_kernel<<<blocks, threads, 0, stream>>>(
      kv_slot, score_slot, weighted, head_dim, ratio);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int norm_blocks = (head_dim + threads - 1) / threads;
  ds_compressor_norm_serial_kernel<<<norm_blocks, threads, 0, stream>>>(
      weighted, norm, out, 1, head_dim, eps);
  return cudaGetLastError();
}

cudaError_t ds_compressor_overlap_decode_at(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *wkv,
    const __nv_bfloat16 *wgate,
    const float *ape,
    const __nv_bfloat16 *norm,
    float *kv_state,
    float *score_state,
    float *weighted,
    __nv_bfloat16 *out,
    int start_pos,
    int hidden_dim,
    int head_dim,
    int state_offset,
    float eps,
    cudaStream_t stream) {
  if (start_pos < 0 || hidden_dim <= 0 || head_dim <= 0 || state_offset < 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int ratio = 4;
  constexpr int threads = 256;
  int state_dim = 2 * head_dim;
  size_t project_shared = 2 * threads * sizeof(float);
  ds_compressor_decode_project_kernel<<<state_dim, threads, project_shared, stream>>>(
      x, wkv, wgate, ape, kv_state, score_state, start_pos, hidden_dim, state_dim, ratio,
      state_offset + ratio);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  bool should_compress = ((start_pos + 1) % ratio) == 0;
  if (!should_compress) return cudaSuccess;
  if (weighted == nullptr || out == nullptr) return cudaErrorInvalidValue;

  int blocks = (head_dim + threads - 1) / threads;
  float *kv_slot = kv_state + state_offset * state_dim;
  float *score_slot = score_state + state_offset * state_dim;
  ds_compressor_overlap_decode_weighted_kernel<<<blocks, threads, 0, stream>>>(
      kv_slot, score_slot, weighted, head_dim);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int norm_blocks = (head_dim + threads - 1) / threads;
  ds_compressor_norm_serial_kernel<<<norm_blocks, threads, 0, stream>>>(
      weighted, norm, out, 1, head_dim, eps);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int shift_total = ratio * state_dim;
  int shift_blocks = (shift_total + threads - 1) / threads;
  ds_compressor_overlap_shift_kernel<<<shift_blocks, threads, 0, stream>>>(
      kv_slot, score_slot, state_dim);
  return cudaGetLastError();
}

cudaError_t ds_apply_rope_hidden_from_pos(
    __nv_bfloat16 *x,
    const float *cos_cache,
    const float *sin_cache,
    const int64_t *positions,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int position_offset,
    int inverse,
    cudaStream_t stream) {
  if (x == nullptr || cos_cache == nullptr || sin_cache == nullptr ||
      positions == nullptr || seq_len <= 0 || local_heads <= 0 ||
      head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      (rotary_dim % 2) != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int total_pairs = seq_len * local_heads * (rotary_dim / 2);
  int blocks = (total_pairs + threads - 1) / threads;
  ds_apply_rope_hidden_from_pos_kernel<<<blocks, threads, 0, stream>>>(
      x, cos_cache, sin_cache, positions, seq_len, local_heads, head_dim,
      rotary_dim, position_offset, inverse);
  return cudaGetLastError();
}

// CUDA-graph decode: fixed topology, position read from device `positions[0]`.
cudaError_t ds_compressor_nonoverlap_decode_at_graph(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *wkv,
    const __nv_bfloat16 *wgate,
    const float *ape,
    const __nv_bfloat16 *norm,
    float *kv_state,
    float *score_state,
    float *weighted,
    __nv_bfloat16 *out,
    const int64_t *positions,
    int hidden_dim,
    int head_dim,
    int ratio,
    int state_offset,
    float eps,
    cudaStream_t stream) {
  if (positions == nullptr || weighted == nullptr || out == nullptr ||
      hidden_dim <= 0 || head_dim <= 0 || ratio <= 1 || ratio > 128 ||
      state_offset < 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  size_t project_shared = 2 * threads * sizeof(float);
  ds_compressor_decode_project_from_pos_kernel<<<head_dim, threads, project_shared, stream>>>(
      x, wkv, wgate, ape, kv_state, score_state, positions, hidden_dim, head_dim,
      ratio, state_offset);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int blocks = (head_dim + threads - 1) / threads;
  float *kv_slot = kv_state + state_offset * head_dim;
  float *score_slot = score_state + state_offset * head_dim;
  ds_compressor_nonoverlap_decode_weighted_gated_kernel<<<blocks, threads, 0, stream>>>(
      kv_slot, score_slot, weighted, positions, head_dim, ratio);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int norm_blocks = (head_dim + threads - 1) / threads;
  ds_compressor_norm_serial_gated_kernel<<<norm_blocks, threads, 0, stream>>>(
      weighted, norm, out, 1, head_dim, eps, positions, ratio);
  return cudaGetLastError();
}

cudaError_t ds_compressor_overlap_decode_at_graph(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *wkv,
    const __nv_bfloat16 *wgate,
    const float *ape,
    const __nv_bfloat16 *norm,
    float *kv_state,
    float *score_state,
    float *weighted,
    __nv_bfloat16 *out,
    const int64_t *positions,
    int hidden_dim,
    int head_dim,
    int state_offset,
    float eps,
    cudaStream_t stream) {
  if (positions == nullptr || weighted == nullptr || out == nullptr ||
      hidden_dim <= 0 || head_dim <= 0 || state_offset < 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int ratio = 4;
  constexpr int threads = 256;
  int state_dim = 2 * head_dim;
  size_t project_shared = 2 * threads * sizeof(float);
  ds_compressor_decode_project_from_pos_kernel<<<state_dim, threads, project_shared, stream>>>(
      x, wkv, wgate, ape, kv_state, score_state, positions, hidden_dim, state_dim,
      ratio, state_offset + ratio);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int blocks = (head_dim + threads - 1) / threads;
  float *kv_slot = kv_state + state_offset * state_dim;
  float *score_slot = score_state + state_offset * state_dim;
  ds_compressor_overlap_decode_weighted_gated_kernel<<<blocks, threads, 0, stream>>>(
      kv_slot, score_slot, weighted, positions, head_dim);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int norm_blocks = (head_dim + threads - 1) / threads;
  ds_compressor_norm_serial_gated_kernel<<<norm_blocks, threads, 0, stream>>>(
      weighted, norm, out, 1, head_dim, eps, positions, ratio);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int shift_total = ratio * state_dim;
  int shift_blocks = (shift_total + threads - 1) / threads;
  ds_compressor_overlap_shift_gated_kernel<<<shift_blocks, threads, 0, stream>>>(
      kv_slot, score_slot, positions, state_dim, ratio);
  return cudaGetLastError();
}

cudaError_t ds_v4_compressor_prewarm(int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  DsCompressorGemmScratch *scratch_ptr = nullptr;
  cudaError_t err = ds_compressor_scratch_for_device(&scratch_ptr);
  if (err != cudaSuccess) return err;
  std::lock_guard<std::mutex> lock(scratch_ptr->mutex);
  return ds_compressor_ensure_handle(*scratch_ptr);
}

}  // extern "C"
