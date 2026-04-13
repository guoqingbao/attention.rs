// Simple per-tensor FP8 quantization for Q (E4M3)
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "attention/dtype_fp8.cuh"

__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
  int* addr_i = reinterpret_cast<int*>(address);
  int old_i = *addr_i;
  float old_f = __int_as_float(old_i);
  while (old_f < val) {
    int assumed = old_i;
    int new_i = __float_as_int(val);
    int prev = atomicCAS(addr_i, assumed, new_i);
    if (prev == assumed) {
      return __int_as_float(prev);
    }
    old_i = prev;
    old_f = __int_as_float(old_i);
  }
  return old_f;
}

template <typename T>
__device__ __forceinline__ float to_float_abs(T x);

template <>
__device__ __forceinline__ float to_float_abs<__half>(__half x) {
  return fabsf(__half2float(x));
}

template <>
__device__ __forceinline__ float to_float_abs<__nv_bfloat16>(__nv_bfloat16 x) {
  return fabsf(__bfloat162float(x));
}

template <typename T>
__global__ void q_fp8_compute_scale_per_head_kernel(const T* __restrict__ input,
                                                    int64_t num_tokens, int num_heads,
                                                    int head_dim, float* __restrict__ scale_out) {
  extern __shared__ float sdata[];
  float* s = sdata;

  const int head_idx = blockIdx.y;
  const int64_t numel_per_head = num_tokens * (int64_t)head_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  float local_max = 0.0f;
  for (int64_t i = idx; i < numel_per_head; i += stride) {
    const int64_t token = i / head_dim;
    const int d = (int)(i % head_dim);
    const int64_t base = token * (int64_t)(num_heads * head_dim) + (int64_t)head_idx * head_dim + d;
    float v = to_float_abs<T>(input[base]);
    if (v > local_max) {
      local_max = v;
    }
  }
  s[threadIdx.x] = local_max;
  __syncthreads();
  for (int sstep = blockDim.x >> 1; sstep > 0; sstep >>= 1) {
    if (threadIdx.x < sstep) {
      float other = s[threadIdx.x + sstep];
      if (other > s[threadIdx.x]) s[threadIdx.x] = other;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float candidate = fmaxf(s[0] / 448.0f, 1e-6f);
    if (candidate > 0.0f) {
      atomicMaxFloat(&scale_out[head_idx], candidate);
    }
  }
}

template <typename T>
__global__ void q_fp8_quantize_per_head_kernel(const T* __restrict__ input,
                                               uint8_t* __restrict__ output, int64_t numel,
                                               int num_heads, int head_dim,
                                               const float* __restrict__ scale_ptr) {
  const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = start; idx < numel; idx += stride) {
    const int64_t head_idx = (idx / head_dim) % num_heads;
    float scale = scale_ptr[head_idx];
    if (scale == 0.0f) {
      output[idx] = 0;
      continue;
    }
    float f = 0.0f;
    if constexpr (std::is_same_v<T, __half>) {
      f = __half2float(input[idx]);
    } else {
      f = __bfloat162float(input[idx]);
    }
    f = f / scale;
    output[idx] = vllm::fp8::dispatch_float_to_fp8(f);
  }
}

extern "C" void flashinfer_fp8_quantize_q_per_head(const void* input, void* output_q,
                                                   float* output_scale, int64_t numel,
                                                   int num_heads, int head_dim,
                                                   bool is_input_f16, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  if (numel <= 0) {
    return;
  }
  // Ensure scales start from 0 for atomicMax-based reduction.
  cudaMemsetAsync(output_scale, 0, (size_t)num_heads * sizeof(float), stream);
  const int threads = 256;
  int blocks = static_cast<int>((numel + threads - 1) / threads);
  if (blocks > 65535) {
    blocks = 65535;
  }
  int64_t num_tokens = numel / (int64_t)(num_heads * head_dim);
  if (num_tokens <= 0) {
    return;
  }
  dim3 grid_scale((int)blocks, num_heads, 1);
  size_t shared_bytes = threads * sizeof(float);
  if (is_input_f16) {
    q_fp8_compute_scale_per_head_kernel<<<grid_scale, threads, shared_bytes, stream>>>(
        static_cast<const __half*>(input), num_tokens, num_heads, head_dim, output_scale);
    q_fp8_quantize_per_head_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<uint8_t*>(output_q), numel, num_heads,
        head_dim, output_scale);
  } else {
    q_fp8_compute_scale_per_head_kernel<<<grid_scale, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(input), num_tokens, num_heads, head_dim, output_scale);
    q_fp8_quantize_per_head_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input), static_cast<uint8_t*>(output_q), numel,
        num_heads, head_dim, output_scale);
  }
}

template <typename T>
__global__ void q_fp8_quantize_scalar_kernel(const T* __restrict__ input,
                                             uint8_t* __restrict__ output, int64_t numel,
                                             const float* __restrict__ scale_ptr) {
  const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  float scale = scale_ptr[0];
  for (int64_t idx = start; idx < numel; idx += stride) {
    if (scale == 0.0f) {
      output[idx] = 0;
      continue;
    }
    float f = 0.0f;
    if constexpr (std::is_same_v<T, __half>) {
      f = __half2float(input[idx]);
    } else {
      f = __bfloat162float(input[idx]);
    }
    f = f / scale;
    output[idx] = vllm::fp8::dispatch_float_to_fp8(f);
  }
}

extern "C" void flashinfer_fp8_quantize_q_scalar(const void* input, void* output_q, int64_t numel,
                                                  const float* q_scale, bool is_input_f16,
                                                  int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  if (numel <= 0) {
    return;
  }
  const int threads = 256;
  int blocks = static_cast<int>((numel + threads - 1) / threads);
  if (blocks > 65535) {
    blocks = 65535;
  }
  if (is_input_f16) {
    q_fp8_quantize_scalar_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<uint8_t*>(output_q), numel, q_scale);
  } else {
    q_fp8_quantize_scalar_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input), static_cast<uint8_t*>(output_q), numel, q_scale);
  }
}

template <typename T>
__global__ void kv_fp8_quantize_per_head_kernel(const T* __restrict__ input,
                                                uint8_t* __restrict__ output, int64_t numel,
                                                int num_heads, int head_dim,
                                                const float* __restrict__ scale_ptr) {
  const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = start; idx < numel; idx += stride) {
    const int64_t head_idx = (idx / head_dim) % num_heads;
    float scale = scale_ptr[head_idx];
    if (scale == 0.0f) {
      output[idx] = 0;
      continue;
    }
    float f = 0.0f;
    if constexpr (std::is_same_v<T, __half>) {
      f = __half2float(input[idx]);
    } else {
      f = __bfloat162float(input[idx]);
    }
    f = f / scale;
    output[idx] = vllm::fp8::dispatch_float_to_fp8(f);
  }
}

template <typename T>
__global__ void kv_fp8_quantize_scalar_kernel(const T* __restrict__ input,
                                              uint8_t* __restrict__ output, int64_t numel,
                                              const float* __restrict__ scale_ptr) {
  const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  float scale = scale_ptr[0];
  for (int64_t idx = start; idx < numel; idx += stride) {
    if (scale == 0.0f) {
      output[idx] = 0;
      continue;
    }
    float f = 0.0f;
    if constexpr (std::is_same_v<T, __half>) {
      f = __half2float(input[idx]);
    } else {
      f = __bfloat162float(input[idx]);
    }
    f = f / scale;
    output[idx] = vllm::fp8::dispatch_float_to_fp8(f);
  }
}

extern "C" void flashinfer_fp8_quantize_kv_scalar(const void* k_in, const void* v_in,
                                                  void* k_out, void* v_out, int64_t numel,
                                                  const float* k_scale, const float* v_scale,
                                                  bool is_input_f16, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  if (numel <= 0) {
    return;
  }
  const int threads = 256;
  int blocks = static_cast<int>((numel + threads - 1) / threads);
  if (blocks > 65535) {
    blocks = 65535;
  }
  if (is_input_f16) {
    kv_fp8_quantize_scalar_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(k_in), static_cast<uint8_t*>(k_out), numel, k_scale);
    kv_fp8_quantize_scalar_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(v_in), static_cast<uint8_t*>(v_out), numel, v_scale);
  } else {
    kv_fp8_quantize_scalar_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k_in), static_cast<uint8_t*>(k_out), numel, k_scale);
    kv_fp8_quantize_scalar_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(v_in), static_cast<uint8_t*>(v_out), numel, v_scale);
  }
}

extern "C" void flashinfer_fp8_quantize_kv_per_head(const void* k_in, const void* v_in,
                                                    void* k_out, void* v_out, int64_t numel,
                                                    int num_heads, int head_dim,
                                                    const float* k_scale, const float* v_scale,
                                                    bool is_input_f16, int64_t stream_) {
  const cudaStream_t stream = (cudaStream_t)stream_;
  if (numel <= 0) {
    return;
  }
  const int threads = 256;
  int blocks = static_cast<int>((numel + threads - 1) / threads);
  if (blocks > 65535) {
    blocks = 65535;
  }
  if (is_input_f16) {
    kv_fp8_quantize_per_head_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(k_in), static_cast<uint8_t*>(k_out), numel, num_heads, head_dim, k_scale);
    kv_fp8_quantize_per_head_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __half*>(v_in), static_cast<uint8_t*>(v_out), numel, num_heads, head_dim, v_scale);
  } else {
    kv_fp8_quantize_per_head_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k_in), static_cast<uint8_t*>(k_out), numel, num_heads, head_dim, k_scale);
    kv_fp8_quantize_per_head_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(v_in), static_cast<uint8_t*>(v_out), numel, num_heads, head_dim, v_scale);
  }
}
