// DeepSeek V4 Hyper-Connection (HC) CUDA kernels
// Provides: hc_expand, hc_pre_from_mixes (fused sinkhorn + pre-output),
//           hc_pre_output, hc_post, hc_head_pre, hc_scale_mixes

#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mutex>

namespace {

constexpr int kDsHcMaxDevices = 16;

struct DsHcGemmScratch {
  cublasHandle_t handle = nullptr;
  float *x_f32 = nullptr;
  size_t x_capacity = 0;
  std::mutex mutex;
};

DsHcGemmScratch g_ds_hc_gemm_scratch[kDsHcMaxDevices];

cudaError_t ds_hc_scratch_for_device(DsHcGemmScratch **out) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  if (device < 0 || device >= kDsHcMaxDevices) return cudaErrorInvalidDevice;
  *out = &g_ds_hc_gemm_scratch[device];
  return cudaSuccess;
}

cudaError_t ds_hc_ensure_scratch(DsHcGemmScratch &scratch, size_t elements) {
  if (elements <= scratch.x_capacity) return cudaSuccess;
  if (scratch.x_f32 != nullptr) {
    // Wait for in-flight kernels/cublas that may still use x_f32 before free.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return err;
    err = cudaFree(scratch.x_f32);
    if (err != cudaSuccess) return err;
    scratch.x_f32 = nullptr;
    scratch.x_capacity = 0;
  }
  cudaError_t err = cudaMalloc(
      reinterpret_cast<void **>(&scratch.x_f32), elements * sizeof(float));
  if (err != cudaSuccess) return err;
  scratch.x_capacity = elements;
  return cudaSuccess;
}

cudaError_t ds_hc_ensure_handle(DsHcGemmScratch &scratch) {
  if (scratch.handle != nullptr) return cudaSuccess;
  cublasStatus_t status = cublasCreate(&scratch.handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    scratch.handle = nullptr;
    return cudaErrorUnknown;
  }
  // HC mixes are especially sensitive: the reference uses a pedantic F32
  // GEMM after converting the BF16 hidden state to F32.  This also prevents
  // the SM90 default from silently selecting TF32 for this reduction.
  status = cublasSetMathMode(scratch.handle, CUBLAS_PEDANTIC_MATH);
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(scratch.handle);
    scratch.handle = nullptr;
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

}  // namespace

static __device__ __forceinline__ float round_to_bf16_float(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

static __device__ __forceinline__ float ds_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// TileLang AllReduce<..., 4, 1> and AllReduce<..., 16, 4> both use an
// XOR-butterfly: pair lanes (0,2)/(1,3), then combine those pairs.  Preserve
// that F32 order instead of a sequential four-value accumulation.
static __device__ __forceinline__ float ds_tl_sum4(
    float v0, float v1, float v2, float v3) {
  return __fadd_rn(__fadd_rn(v0, v2), __fadd_rn(v1, v3));
}

static __device__ __forceinline__ float ds_tl_max4(
    float v0, float v1, float v2, float v3) {
  return fmaxf(fmaxf(v0, v2), fmaxf(v1, v3));
}

// ============ Kernels ============

__global__ void ds_hc_bf16_to_f32_kernel(
    const __nv_bfloat16 *__restrict__ input,
    float *__restrict__ output,
    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) output[idx] = __bfloat162float(input[idx]);
}

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
  // Match PyTorch's contiguous F32 mean reduction for hc_dim=8192.  ATen's
  // ReduceConfig vectorizes four inputs per load and launches (128, 4), then
  // reduces block.x before block.y.  A flat 512-thread binary tree changes
  // the last few F32 bits of the RMS scale; HC keeps post/comb in F32, so that
  // otherwise tiny error compounds through all 86 hyper-connection updates.
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (token >= seq_len) return;

  extern __shared__ float scratch[];
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int vec_end = hc_dim / 4;
  for (int vec = tid; vec < vec_end; vec += blockDim.x * blockDim.y) {
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

  // ATen Reduce.cuh::block_x_reduce for blockDim.x=128.
  for (int offset = blockDim.x / 2; offset >= warpSize; offset >>= 1) {
    __syncthreads();
    if (threadIdx.x < offset) {
      sumsq = __fadd_rn(sumsq, scratch[tid + offset]);
      scratch[tid] = sumsq;
    }
  }
  __syncthreads();
  if (threadIdx.x < warpSize) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      sumsq = __fadd_rn(sumsq, __shfl_down_sync(0xffffffff, sumsq, offset));
    }
  }

  // ATen Reduce.cuh::block_y_reduce for blockDim.y=4.
  scratch[tid] = sumsq;
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.y < offset) {
      sumsq = __fadd_rn(sumsq, scratch[tid + offset * blockDim.x]);
      scratch[tid] = sumsq;
    }
  }
  __syncthreads();
  float mean = __fmul_rn(scratch[0], 1.0f / static_cast<float>(hc_dim));
  float scale = rsqrtf(__fadd_rn(mean, eps));
  if (tid == 0 && rms_scales != nullptr) rms_scales[token] = scale;
  for (int mix = tid; mix < mix_hc; mix += blockDim.x) {
    mixes[token * mix_hc + mix] *= scale;
  }
}

// ATen-order RMSNorm for DeepSeek V4 attn/ffn/q/kv norms.
// Golden: x.float(); var = x.square().mean(-1); (weight.float() * x * rsqrt(var+eps)).to(dtype).
// ATen mean over the last dim uses vectorized 4-wide loads with block (128, 4):
// each thread accumulates 4 separate sums over its vec groups, folds them in
// order, block.x tree-reduces (128), then block.y reduces (4).  Candle's
// generic rmsnorm kernel uses a strided + XOR-butterfly order which diverges
// in the last F32 bits and compounds through the 86 HC updates.
__global__ void ds_v4_rms_norm_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    int rows, int dim, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;

  extern __shared__ float scratch[];
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int vec_end = dim / 4;
  for (int vec = tid; vec < vec_end; vec += blockDim.x * blockDim.y) {
    const int base = row * dim + vec * 4;
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      float value = __bfloat162float(x[base + lane]);
      sums[lane] = __fadd_rn(sums[lane], __fmul_rn(value, value));
    }
  }
  float sumsq = sums[0];
  sumsq = __fadd_rn(sumsq, sums[1]);
  sumsq = __fadd_rn(sumsq, sums[2]);
  sumsq = __fadd_rn(sumsq, sums[3]);
  if (tid == 0) {
    for (int k = vec_end * 4; k < dim; ++k) {
      float value = __bfloat162float(x[row * dim + k]);
      sumsq = __fadd_rn(sumsq, __fmul_rn(value, value));
    }
  }
  scratch[tid] = sumsq;

  // ATen Reduce.cuh::block_x_reduce for blockDim.x=128.
  for (int offset = blockDim.x / 2; offset >= warpSize; offset >>= 1) {
    __syncthreads();
    if (threadIdx.x < offset) {
      sumsq = __fadd_rn(sumsq, scratch[tid + offset]);
      scratch[tid] = sumsq;
    }
  }
  __syncthreads();
  if (threadIdx.x < warpSize) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      sumsq = __fadd_rn(sumsq, __shfl_down_sync(0xffffffff, sumsq, offset));
    }
  }

  // ATen Reduce.cuh::block_y_reduce for blockDim.y=4.
  scratch[tid] = sumsq;
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.y < offset) {
      sumsq = __fadd_rn(sumsq, scratch[tid + offset * blockDim.x]);
      scratch[tid] = sumsq;
    }
  }
  __syncthreads();
  const float mean = __fmul_rn(scratch[0], 1.0f / static_cast<float>(dim));
  const float scale = rsqrtf(__fadd_rn(mean, eps));
  for (int d = tid; d < dim; d += blockDim.x * blockDim.y) {
    float value = __bfloat162float(x[row * dim + d]);
    float w = (weight != nullptr) ? weight[d] : 1.0f;
    out[row * dim + d] = __float2bfloat16(__fmul_rn(__fmul_rn(value, scale), w));
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
      row_max[j] = ds_tl_max4(
          comb_frag[j * hc], comb_frag[j * hc + 1],
          comb_frag[j * hc + 2], comb_frag[j * hc + 3]);
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      #pragma unroll
      for (int k = 0; k < hc; ++k) {
        float v = expf(comb_frag[j * hc + k] - row_max[j]);
        comb_frag[j * hc + k] = v;
      }
      row_sum[j] = ds_tl_sum4(
          comb_frag[j * hc], comb_frag[j * hc + 1],
          comb_frag[j * hc + 2], comb_frag[j * hc + 3]);
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
      col_sum[k] = ds_tl_sum4(
          comb_frag[k], comb_frag[hc + k],
          comb_frag[2 * hc + k], comb_frag[3 * hc + k]);
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
        row_sum[j] = ds_tl_sum4(
            comb_frag[j * hc], comb_frag[j * hc + 1],
            comb_frag[j * hc + 2], comb_frag[j * hc + 3]);
      }
      #pragma unroll
      for (int j = 0; j < hc; ++j) {
        #pragma unroll
        for (int k = 0; k < hc; ++k)
          comb_frag[j * hc + k] = comb_frag[j * hc + k] / (row_sum[j] + eps);
      }
      #pragma unroll
      for (int k = 0; k < hc; ++k) {
        col_sum[k] = ds_tl_sum4(
            comb_frag[k], comb_frag[hc + k],
            comb_frag[2 * hc + k], comb_frag[3 * hc + k]);
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
      row_max[j] = ds_tl_max4(
          comb_frag[j * hc], comb_frag[j * hc + 1],
          comb_frag[j * hc + 2], comb_frag[j * hc + 3]);
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j) {
      #pragma unroll
      for (int k = 0; k < hc; ++k) {
        float v = expf(comb_frag[j * hc + k] - row_max[j]);
        comb_frag[j * hc + k] = v;
      }
      row_sum[j] = ds_tl_sum4(
          comb_frag[j * hc], comb_frag[j * hc + 1],
          comb_frag[j * hc + 2], comb_frag[j * hc + 3]);
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j)
      #pragma unroll
      for (int k = 0; k < hc; ++k)
        comb_frag[j * hc + k] = comb_frag[j * hc + k] / row_sum[j] + hc_eps;
    #pragma unroll
    for (int k = 0; k < hc; ++k) {
      col_sum[k] = ds_tl_sum4(
          comb_frag[k], comb_frag[hc + k],
          comb_frag[2 * hc + k], comb_frag[3 * hc + k]);
    }
    #pragma unroll
    for (int j = 0; j < hc; ++j)
      #pragma unroll
      for (int k = 0; k < hc; ++k)
        comb_frag[j * hc + k] = comb_frag[j * hc + k] / (col_sum[k] + hc_eps);

    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
      #pragma unroll
      for (int j = 0; j < hc; ++j) {
        row_sum[j] = ds_tl_sum4(
            comb_frag[j * hc], comb_frag[j * hc + 1],
            comb_frag[j * hc + 2], comb_frag[j * hc + 3]);
      }
      #pragma unroll
      for (int j = 0; j < hc; ++j)
        #pragma unroll
        for (int k = 0; k < hc; ++k)
          comb_frag[j * hc + k] = comb_frag[j * hc + k] / (row_sum[j] + hc_eps);
      #pragma unroll
      for (int k = 0; k < hc; ++k) {
        col_sum[k] = ds_tl_sum4(
            comb_frag[k], comb_frag[hc + k],
            comb_frag[2 * hc + k], comb_frag[3 * hc + k]);
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

__global__ void ds_hc_post_f32_branch_kernel(
    const float *__restrict__ x,
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
  // The attention/FFN branch is BF16 at the Python model boundary.  Decode
  // keeps the row-parallel all-reduce in F32, so reproduce that BF16 boundary
  // here before feeding the branch back through the hyperconnection.
  float branch = __bfloat162float(__float2bfloat16(x[token * dim + dim_idx]));
  float post_term = __fmul_rn(post[token * hc + h_out], branch);
  out[idx] = __float2bfloat16(__fadd_rn(post_term, residual_sum));
}

// Per-head RMSNorm for V4 Q projection
__global__ void ds_head_rms_norm_kernel(
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ out,
    int seq_len, int num_heads, int head_dim, float eps) {
  // Match the ATen contiguous F32 mean reduction selected by the official
  // implementation for rows of 512 values:
  //   reduce_kernel<512, 1, ..., 4, 4>, grid=(ceil(rows/16),1,1),
  //   block=(32,16,1).
  // Each warp owns one head row.  Its lanes each accumulate four contiguous
  // vector lanes over four iterations, fold those four accumulators in order,
  // then perform the warp-x reduction.
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int rows = seq_len * num_heads;
  const int lane = threadIdx.x;
  if (row >= rows) return;

  const int base = row * head_dim;
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int vec_end = head_dim / 4;
  for (int vec = lane; vec < vec_end; vec += blockDim.x) {
    const int vec_base = base + vec * 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float value = __bfloat162float(x[vec_base + i]);
      sums[i] = __fadd_rn(sums[i], __fmul_rn(value, value));
    }
  }
  float partial = sums[0];
  partial = __fadd_rn(partial, sums[1]);
  partial = __fadd_rn(partial, sums[2]);
  partial = __fadd_rn(partial, sums[3]);
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    partial = __fadd_rn(
        partial, __shfl_down_sync(0xffffffff, partial, offset));
  partial = __shfl_sync(0xffffffff, partial, 0);

  float mean_sq = __fmul_rn(partial, 1.0f / static_cast<float>(head_dim));
  float inv_rms = rsqrtf(__fadd_rn(mean_sq, eps));
  for (int d = lane; d < head_dim; d += blockDim.x) {
    float value = __bfloat162float(x[base + d]);
    out[base + d] = __float2bfloat16(__fmul_rn(value, inv_rms));
  }
}

// ============ Extern "C" entry points ============

extern "C" {

cudaError_t ds_v4_rms_norm(
    const void *x, const void *weight, void *out,
    int rows, int dim, float eps, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  if (rows <= 0 || dim <= 0 || dim % 4 != 0) return cudaErrorInvalidValue;
  const dim3 threads(128, 4);
  constexpr int thread_count = 512;
  ds_v4_rms_norm_kernel<<<rows, threads, thread_count * sizeof(float), stream>>>(
      (const __nv_bfloat16*)x, (const float*)weight, (__nv_bfloat16*)out,
      rows, dim, eps);
  return cudaGetLastError();
}

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

cudaError_t ds_v4_hc_mixes(
    const void *x, const void *hc_fn, void *mixes,
    int seq_len, int hc, int dim, int mix_hc, float eps, int64_t stream_) {
  if (x == nullptr || hc_fn == nullptr || mixes == nullptr || seq_len <= 0 ||
      hc <= 0 || dim <= 0 || mix_hc <= 0) {
    return cudaErrorInvalidValue;
  }
  const cudaStream_t stream = (cudaStream_t)stream_;
  const int hc_dim = hc * dim;
  const int total = seq_len * hc_dim;
  constexpr int threads = 256;

  DsHcGemmScratch *scratch_ptr = nullptr;
  cudaError_t err = ds_hc_scratch_for_device(&scratch_ptr);
  if (err != cudaSuccess) return err;
  DsHcGemmScratch &scratch = *scratch_ptr;
  std::lock_guard<std::mutex> lock(scratch.mutex);

  err = ds_hc_ensure_scratch(scratch, static_cast<size_t>(total));
  if (err != cudaSuccess) return err;
  const int blocks = (total + threads - 1) / threads;
  ds_hc_bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
      (const __nv_bfloat16 *)x, scratch.x_f32, total);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = ds_hc_ensure_handle(scratch);
  if (err != cudaSuccess) return err;
  if (cublasSetStream(scratch.handle, stream) != CUBLAS_STATUS_SUCCESS) {
    return cudaErrorUnknown;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status;
  if (seq_len == 1) {
    status = cublasSgemv(
        scratch.handle, CUBLAS_OP_T, hc_dim, mix_hc, &alpha,
        (const float *)hc_fn, hc_dim, scratch.x_f32, 1, &beta,
        (float *)mixes, 1);
  } else {
    status = cublasGemmEx(
        scratch.handle, CUBLAS_OP_T, CUBLAS_OP_N,
        mix_hc, seq_len, hc_dim, &alpha,
        hc_fn, CUDA_R_32F, hc_dim,
        scratch.x_f32, CUDA_R_32F, hc_dim,
        &beta, mixes, CUDA_R_32F, mix_hc,
        CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
  }
  if (status != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

  // Keep the exact reference ordering: normalize the source hidden state
  // after the GEMM, on the same stream, without a host-visible fence.
  const dim3 scale_threads(128, 4);
  constexpr int scale_thread_count = 512;
  ds_hc_scale_mixes_block_kernel<<<
      seq_len, scale_threads, scale_thread_count * sizeof(float), stream>>>(
          (const __nv_bfloat16 *)x, (float *)mixes, nullptr,
          seq_len, hc_dim, mix_hc, eps);
  return cudaGetLastError();
}

cudaError_t ds_v4_hc_scale_mixes(
    const void *x, void *mixes,
    int seq_len, int hc, int dim, int mix_hc,
    float eps, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  const dim3 scale_threads(128, 4);
  constexpr int scale_thread_count = 512;
  int hc_dim = hc * dim;
  ds_hc_scale_mixes_block_kernel<<<seq_len, scale_threads, scale_thread_count * sizeof(float), stream>>>(
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

cudaError_t ds_v4_hc_post_f32_branch(
    const void *x, const void *residual,
    const void *post, const void *comb, void *out,
    int seq_len, int hc, int dim, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  constexpr int threads = 256;
  int total = seq_len * hc * dim;
  int blocks = (total + threads - 1) / threads;
  ds_hc_post_f32_branch_kernel<<<blocks, threads, 0, stream>>>(
      (const float*)x, (const __nv_bfloat16*)residual,
      (const float*)post, (const float*)comb, (__nv_bfloat16*)out,
      seq_len, hc, dim);
  return cudaGetLastError();
}

cudaError_t ds_v4_head_rms_norm(
    const void *x, void *out,
    int seq_len, int num_heads, int head_dim, float eps, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  if (head_dim != 512) return cudaErrorInvalidValue;
  dim3 threads(32, 16);
  dim3 grid((seq_len * num_heads + threads.y - 1) / threads.y);
  ds_head_rms_norm_kernel<<<grid, threads, 0, stream>>>(
      (const __nv_bfloat16*)x, (__nv_bfloat16*)out,
      seq_len, num_heads, head_dim, eps);
  return cudaGetLastError();
}

} // extern "C"
