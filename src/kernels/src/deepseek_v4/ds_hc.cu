// DeepSeek V4 Hyper-Connection (HC) CUDA kernels
// Provides: hc_expand, hc_pre_from_mixes (fused sinkhorn + pre-output),
//           hc_pre_output, hc_post, hc_head_pre, hc_scale_mixes

#include <cuda_bf16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float round_to_bf16_float(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

static __device__ __forceinline__ float ds_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// ============ Kernels ============

__global__ void ds_hc_expand_kernel(
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ out,
    int seq_len, int hc, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * hc * dim;
  if (idx >= total) return;
  int dim_idx = idx % dim;
  int token = idx / (hc * dim);
  out[idx] = x[token * dim + dim_idx];
}

__global__ void ds_hc_scale_mixes_block_kernel(
    const __nv_bfloat16 *__restrict__ x,
    float *__restrict__ mixes,
    float *__restrict__ rms_scales,
    int seq_len, int hc_dim, int mix_hc, float eps) {
  int token = blockIdx.x;
  int tid = threadIdx.x;
  if (token >= seq_len) return;

  extern __shared__ float scratch[];
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int vec_end = hc_dim / 4;
  for (int vec = tid; vec < vec_end; vec += blockDim.x) {
    int base = token * hc_dim + vec * 4;
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      float value = __bfloat162float(x[base + lane]);
      float sq = __fmul_rn(value, value);
      sums[lane] = __fadd_rn(sums[lane], sq);
    }
  }
  float sumsq = sums[0];
  sumsq = __fadd_rn(sumsq, sums[1]);
  sumsq = __fadd_rn(sumsq, sums[2]);
  sumsq = __fadd_rn(sumsq, sums[3]);
  if (tid == 0) {
    for (int k = vec_end * 4; k < hc_dim; ++k) {
      float value = __bfloat162float(x[token * hc_dim + k]);
      sumsq = __fadd_rn(sumsq, __fmul_rn(value, value));
    }
  }
  scratch[tid] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] = __fadd_rn(scratch[tid], scratch[tid + stride]);
    __syncthreads();
  }
  float mean = __fmul_rn(scratch[0], 1.0f / static_cast<float>(hc_dim));
  float scale = rsqrtf(__fadd_rn(mean, eps));
  if (tid == 0 && rms_scales != nullptr) rms_scales[token] = scale;
  for (int mix = tid; mix < mix_hc; mix += blockDim.x) {
    mixes[token * mix_hc + mix] *= scale;
  }
}

__global__ void ds_hc_pre_output_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ pre,
    __nv_bfloat16 *__restrict__ out,
    int seq_len, int hc, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * dim;
  if (idx >= total) return;
  int dim_idx = idx % dim;
  int token = idx / dim;
  float sum = 0.0f;
  for (int h = 0; h < hc; ++h) {
    sum += pre[token * hc + h] * __bfloat162float(x[(token * hc + h) * dim + dim_idx]);
  }
  out[idx] = __float2bfloat16(sum);
}

__global__ void ds_hc_pre_from_mixes_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ mixes,
    const float *__restrict__ hc_scale,
    const float *__restrict__ hc_base,
    float *__restrict__ post,
    float *__restrict__ comb,
    __nv_bfloat16 *__restrict__ out,
    int seq_len, int dim, int sinkhorn_iters, float eps) {
  constexpr int hc = 4;
  constexpr int mix_hc = (2 + hc) * hc; // 24
  int token = blockIdx.x;
  if (token >= seq_len) return;

  __shared__ float pre_shared[hc];

  if (threadIdx.x == 0) {
    float comb_frag[hc * hc];
    const float* mix = mixes + token * mix_hc;

    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      pre_shared[j] = ds_sigmoid(mix[j] * hc_scale[0] + hc_base[j]) + eps;
      post[token * hc + j] = 2.0f * ds_sigmoid(mix[j + hc] * hc_scale[1] + hc_base[j + hc]);
    }

    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      #pragma unroll
      for (int k = 0; k < hc; ++k) {
        int offset = j * hc + k + hc * 2;
        comb_frag[j * hc + k] = mix[offset] * hc_scale[2] + hc_base[offset];
      }
    }

    // Numerically stable exp + sinkhorn
    float row_sum[hc], col_sum[hc], row_max[hc];
    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      float mx = comb_frag[j * hc];
      #pragma unroll
      for (int k = 1; k < hc; ++k) mx = fmaxf(mx, comb_frag[j * hc + k]);
      row_max[j] = mx;
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      float s = 0.0f;
      #pragma unroll
      for (int k = 0; k < hc; ++k) {
        float v = expf(comb_frag[j * hc + k] - row_max[j]);
        comb_frag[j * hc + k] = v;
        s += v;
      }
      row_sum[j] = s;
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      #pragma unroll
      for (int k = 0; k < hc; ++k)
        comb_frag[j * hc + k] = comb_frag[j * hc + k] / row_sum[j] + eps;
    }

    // Column normalization
    #pragma unroll
    for (int k = 0; k < hc; ++k) {
      float s = 0.0f;
      #pragma unroll
      for (int j = 0; j < hc; ++j) s += comb_frag[j * hc + k];
      col_sum[k] = s;
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      #pragma unroll
      for (int k = 0; k < hc; ++k)
        comb_frag[j * hc + k] = comb_frag[j * hc + k] / (col_sum[k] + eps);
    }

    // Remaining sinkhorn iterations
    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
      #pragma unroll
      for (int j = 0; j < hc; ++j) {
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < hc; ++k) s += comb_frag[j * hc + k];
        row_sum[j] = s;
      }
      #pragma unroll
      for (int j = 0; j < hc; ++j) {
        #pragma unroll
        for (int k = 0; k < hc; ++k)
          comb_frag[j * hc + k] = comb_frag[j * hc + k] / (row_sum[j] + eps);
      }
      #pragma unroll
      for (int k = 0; k < hc; ++k) {
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < hc; ++j) s += comb_frag[j * hc + k];
        col_sum[k] = s;
      }
      #pragma unroll
      for (int j = 0; j < hc; ++j) {
        #pragma unroll
        for (int k = 0; k < hc; ++k)
          comb_frag[j * hc + k] = comb_frag[j * hc + k] / (col_sum[k] + eps);
      }
    }

    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      #pragma unroll
      for (int k = 0; k < hc; ++k)
        comb[token * hc * hc + j * hc + k] = comb_frag[j * hc + k];
    }
  }
  __syncthreads();

  for (int dim_idx = threadIdx.x; dim_idx < dim; dim_idx += blockDim.x) {
    float sum = 0.0f;
    #pragma unroll
    for (int h = 0; h < hc; ++h) {
      sum += pre_shared[h] * __bfloat162float(x[(token * hc + h) * dim + dim_idx]);
    }
    out[token * dim + dim_idx] = __float2bfloat16(sum);
  }
}

__global__ void ds_hc_pre_norm_from_mixes_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ mixes,
    const float *__restrict__ hc_scale,
    const float *__restrict__ hc_base,
    const __nv_bfloat16 *__restrict__ norm_weight,
    float *__restrict__ post,
    float *__restrict__ comb,
    __nv_bfloat16 *__restrict__ out,
    int seq_len, int dim, int sinkhorn_iters, float hc_eps, float norm_eps) {
  constexpr int hc = 4;
  constexpr int mix_hc = (2 + hc) * hc;
  int token = blockIdx.x;
  if (token >= seq_len) return;

  extern __shared__ float shared[];
  float* pre_values = shared;
  float* reduction = shared + dim;
  __shared__ float pre_shared[hc];

  if (threadIdx.x == 0) {
    float comb_frag[hc * hc];
    const float* mix = mixes + token * mix_hc;
    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      pre_shared[j] = ds_sigmoid(mix[j] * hc_scale[0] + hc_base[j]) + hc_eps;
      post[token * hc + j] = 2.0f * ds_sigmoid(mix[j + hc] * hc_scale[1] + hc_base[j + hc]);
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j)
      #pragma unroll
      for (int k = 0; k < hc; ++k) {
        int off = j * hc + k + hc * 2;
        comb_frag[j * hc + k] = mix[off] * hc_scale[2] + hc_base[off];
      }

    float row_sum[hc], col_sum[hc], row_max[hc];
    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      float mx = comb_frag[j * hc];
      #pragma unroll
      for (int k = 1; k < hc; ++k) mx = fmaxf(mx, comb_frag[j * hc + k]);
      row_max[j] = mx;
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      float s = 0.0f;
      #pragma unroll
      for (int k = 0; k < hc; ++k) {
        float v = expf(comb_frag[j * hc + k] - row_max[j]);
        comb_frag[j * hc + k] = v; s += v;
      }
      row_sum[j] = s;
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j)
      #pragma unroll
      for (int k = 0; k < hc; ++k)
        comb_frag[j * hc + k] = comb_frag[j * hc + k] / row_sum[j] + hc_eps;
    #pragma unroll
    for (int k = 0; k < hc; ++k) {
      float s = 0.0f;
      #pragma unroll
      for (int j = 0; j < hc; ++j) s += comb_frag[j * hc + k];
      col_sum[k] = s;
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j)
      #pragma unroll
      for (int k = 0; k < hc; ++k)
        comb_frag[j * hc + k] = comb_frag[j * hc + k] / (col_sum[k] + hc_eps);

    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
      #pragma unroll
      for (int j = 0; j < hc; ++j) {
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < hc; ++k) s += comb_frag[j * hc + k];
        row_sum[j] = s;
      }
      #pragma unroll
      for (int j = 0; j < hc; ++j)
        #pragma unroll
        for (int k = 0; k < hc; ++k)
          comb_frag[j * hc + k] = comb_frag[j * hc + k] / (row_sum[j] + hc_eps);
      #pragma unroll
      for (int k = 0; k < hc; ++k) {
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < hc; ++j) s += comb_frag[j * hc + k];
        col_sum[k] = s;
      }
      #pragma unroll
      for (int j = 0; j < hc; ++j)
        #pragma unroll
        for (int k = 0; k < hc; ++k)
          comb_frag[j * hc + k] = comb_frag[j * hc + k] / (col_sum[k] + hc_eps);
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j)
      #pragma unroll
      for (int k = 0; k < hc; ++k)
        comb[token * hc * hc + j * hc + k] = comb_frag[j * hc + k];
  }
  __syncthreads();

  float sumsq = 0.0f;
  for (int dim_idx = threadIdx.x; dim_idx < dim; dim_idx += blockDim.x) {
    float sum = 0.0f;
    #pragma unroll
    for (int h = 0; h < hc; ++h)
      sum += pre_shared[h] * __bfloat162float(x[(token * hc + h) * dim + dim_idx]);
    float rounded = round_to_bf16_float(sum);
    pre_values[dim_idx] = rounded;
    sumsq += rounded * rounded;
  }
  reduction[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  float inv_rms = rsqrtf(reduction[0] / static_cast<float>(dim) + norm_eps);
  for (int dim_idx = threadIdx.x; dim_idx < dim; dim_idx += blockDim.x) {
    float value = pre_values[dim_idx] * inv_rms * __bfloat162float(norm_weight[dim_idx]);
    out[token * dim + dim_idx] = __float2bfloat16(value);
  }
}

__global__ void ds_hc_head_pre_kernel(
    const float *__restrict__ mixes,
    const float *__restrict__ hc_scale,
    const float *__restrict__ hc_base,
    float *__restrict__ pre,
    int seq_len, int hc, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * hc;
  if (idx >= total) return;
  int h = idx % hc;
  pre[idx] = ds_sigmoid(mixes[idx] * hc_scale[0] + hc_base[h]) + eps;
}

__global__ void ds_hc_post_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ residual,
    const float *__restrict__ post,
    const float *__restrict__ comb,
    __nv_bfloat16 *__restrict__ out,
    int seq_len, int hc, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = seq_len * hc * dim;
  if (idx >= total) return;
  int dim_idx = idx % dim;
  int h_out = (idx / dim) % hc;
  int token = idx / (hc * dim);
  float residual_sum = 0.0f;
  if (hc == 4) {
    float term0 = __fmul_rn(comb[(token * hc + 0) * hc + h_out],
        __bfloat162float(residual[(token * hc + 0) * dim + dim_idx]));
    float term1 = __fmul_rn(comb[(token * hc + 1) * hc + h_out],
        __bfloat162float(residual[(token * hc + 1) * dim + dim_idx]));
    float term2 = __fmul_rn(comb[(token * hc + 2) * hc + h_out],
        __bfloat162float(residual[(token * hc + 2) * dim + dim_idx]));
    float term3 = __fmul_rn(comb[(token * hc + 3) * hc + h_out],
        __bfloat162float(residual[(token * hc + 3) * dim + dim_idx]));
    residual_sum = __fadd_rn(__fadd_rn(__fadd_rn(term0, term1), term2), term3);
  } else {
    for (int h_in = 0; h_in < hc; ++h_in) {
      float term = __fmul_rn(comb[(token * hc + h_in) * hc + h_out],
          __bfloat162float(residual[(token * hc + h_in) * dim + dim_idx]));
      residual_sum = __fadd_rn(residual_sum, term);
    }
  }
  float post_term = __fmul_rn(post[token * hc + h_out], __bfloat162float(x[token * dim + dim_idx]));
  out[idx] = __float2bfloat16(__fadd_rn(post_term, residual_sum));
}

// Per-head RMSNorm for V4 Q projection
__global__ void ds_head_rms_norm_kernel(
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ out,
    int seq_len, int num_heads, int head_dim, float eps) {
  int token = blockIdx.x;
  int head = blockIdx.y;
  int tid = threadIdx.x;
  if (token >= seq_len || head >= num_heads) return;

  extern __shared__ float scratch[];
  int base = token * num_heads * head_dim + head * head_dim;
  float partial = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float value = __bfloat162float(x[base + d]);
    partial += round_to_bf16_float(value * value);
  }
  scratch[tid] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  float mean_sq = round_to_bf16_float(scratch[0] / head_dim);
  float inv_rms = round_to_bf16_float(rsqrtf(round_to_bf16_float(mean_sq + eps)));
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float value = __bfloat162float(x[base + d]);
    out[base + d] = __float2bfloat16(value * inv_rms);
  }
}

// ============ Extern "C" entry points ============

extern "C" {

cudaError_t ds_v4_hc_expand(
    const void *x, void *out,
    int seq_len, int hc, int dim, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  constexpr int threads = 256;
  int total = seq_len * hc * dim;
  int blocks = (total + threads - 1) / threads;
  ds_hc_expand_kernel<<<blocks, threads, 0, stream>>>(
      (const __nv_bfloat16*)x, (__nv_bfloat16*)out, seq_len, hc, dim);
  return cudaGetLastError();
}

cudaError_t ds_v4_hc_scale_mixes(
    const void *x, void *mixes,
    int seq_len, int hc, int dim, int mix_hc,
    float eps, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  constexpr int scale_threads = 512;
  int hc_dim = hc * dim;
  ds_hc_scale_mixes_block_kernel<<<seq_len, scale_threads, scale_threads * sizeof(float), stream>>>(
      (const __nv_bfloat16*)x, (float*)mixes, nullptr, seq_len, hc_dim, mix_hc, eps);
  return cudaGetLastError();
}

cudaError_t ds_v4_hc_pre_from_mixes(
    const void *x, const void *mixes,
    const void *hc_scale, const void *hc_base,
    void *post, void *comb, void *out,
    int seq_len, int hc, int dim, int sinkhorn_iters, float eps,
    int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  if (hc != 4) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  ds_hc_pre_from_mixes_kernel<<<seq_len, threads, 0, stream>>>(
      (const __nv_bfloat16*)x, (const float*)mixes,
      (const float*)hc_scale, (const float*)hc_base,
      (float*)post, (float*)comb, (__nv_bfloat16*)out,
      seq_len, dim, sinkhorn_iters, eps);
  return cudaGetLastError();
}

cudaError_t ds_v4_hc_pre_norm_from_mixes(
    const void *x, const void *mixes,
    const void *hc_scale, const void *hc_base,
    const void *norm_weight,
    void *post, void *comb, void *out,
    int seq_len, int hc, int dim, int sinkhorn_iters,
    float hc_eps, float norm_eps, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  if (hc != 4) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  size_t shared_bytes = ((size_t)dim + threads) * sizeof(float);
  ds_hc_pre_norm_from_mixes_kernel<<<seq_len, threads, shared_bytes, stream>>>(
      (const __nv_bfloat16*)x, (const float*)mixes,
      (const float*)hc_scale, (const float*)hc_base,
      (const __nv_bfloat16*)norm_weight,
      (float*)post, (float*)comb, (__nv_bfloat16*)out,
      seq_len, dim, sinkhorn_iters, hc_eps, norm_eps);
  return cudaGetLastError();
}

cudaError_t ds_v4_hc_pre_output(
    const void *x, const void *pre, void *out,
    int seq_len, int hc, int dim, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  constexpr int threads = 256;
  int total = seq_len * dim;
  int blocks = (total + threads - 1) / threads;
  ds_hc_pre_output_kernel<<<blocks, threads, 0, stream>>>(
      (const __nv_bfloat16*)x, (const float*)pre, (__nv_bfloat16*)out,
      seq_len, hc, dim);
  return cudaGetLastError();
}

cudaError_t ds_v4_hc_head_pre(
    const void *mixes, const void *hc_scale, const void *hc_base,
    void *pre, int seq_len, int hc, float eps, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  constexpr int threads = 256;
  int total = seq_len * hc;
  int blocks = (total + threads - 1) / threads;
  ds_hc_head_pre_kernel<<<blocks, threads, 0, stream>>>(
      (const float*)mixes, (const float*)hc_scale, (const float*)hc_base,
      (float*)pre, seq_len, hc, eps);
  return cudaGetLastError();
}

cudaError_t ds_v4_hc_post(
    const void *x, const void *residual,
    const void *post, const void *comb, void *out,
    int seq_len, int hc, int dim, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  constexpr int threads = 256;
  int total = seq_len * hc * dim;
  int blocks = (total + threads - 1) / threads;
  ds_hc_post_kernel<<<blocks, threads, 0, stream>>>(
      (const __nv_bfloat16*)x, (const __nv_bfloat16*)residual,
      (const float*)post, (const float*)comb, (__nv_bfloat16*)out,
      seq_len, hc, dim);
  return cudaGetLastError();
}

cudaError_t ds_v4_head_rms_norm(
    const void *x, void *out,
    int seq_len, int num_heads, int head_dim, float eps, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  constexpr int threads = 256;
  dim3 grid(seq_len, num_heads);
  size_t shared_bytes = threads * sizeof(float);
  ds_head_rms_norm_kernel<<<grid, threads, shared_bytes, stream>>>(
      (const __nv_bfloat16*)x, (__nv_bfloat16*)out,
      seq_len, num_heads, head_dim, eps);
  return cudaGetLastError();
}

} // extern "C"
