// DeepSeek-V4 QAT: FP8 E4M3+UE8M0 act-quant on non-RoPE (nope) head dims.
// Matches official tilelang act_quant(inplace=True, scale_fmt=ue8m0, block=64).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

extern "C" int ds_copy_device_bytes(
    const void *src,
    void *dst,
    size_t bytes,
    size_t dst_byte_offset,
    int64_t stream_i64) {
  if (src == nullptr || dst == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (bytes == 0) return static_cast<int>(cudaSuccess);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_i64);
  return static_cast<int>(cudaMemcpyAsync(
      static_cast<unsigned char *>(dst) + dst_byte_offset,
      src,
      bytes,
      cudaMemcpyDeviceToDevice,
      stream));
}

static __device__ __forceinline__ float e8m0_to_float(unsigned char value) {
  __nv_bfloat16_raw raw = __nv_cvt_e8m0_to_bf16raw(value);
  __nv_bfloat16 bf16_value(raw);
  return __bfloat162float(bf16_value);
}

static __device__ __forceinline__ unsigned char float_to_e8m0(float value) {
  return __nv_cvt_float_to_e8m0(value, __NV_SATFINITE, cudaRoundPosInf);
}

static __device__ __forceinline__ float fp8_e4m3_to_float(unsigned char value) {
  __half_raw raw = __nv_cvt_fp8_to_halfraw(value, __NV_E4M3);
  return __half2float(__half(raw));
}

static __device__ __forceinline__ unsigned char float_to_fp8_e4m3(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

__global__ void ds_fp8_act_quant_nope_bf16_kernel(
    const __nv_bfloat16 *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int block_size) {
  int token = blockIdx.x;
  int head = blockIdx.y;
  int group = blockIdx.z;
  int tid = threadIdx.x;
  int nope_dim = head_dim - rotary_dim;
  if (token >= seq_len || head >= local_heads || group * block_size >= nope_dim) return;

  int start = group * block_size;
  int end = min(start + block_size, nope_dim);
  int base = token * local_heads * head_dim + head * head_dim;

  extern __shared__ float scratch[];
  float amax = 0.0f;
  for (int dim = start + tid; dim < end; dim += blockDim.x) {
    amax = fmaxf(amax, fabsf(__bfloat162float(input[base + dim])));
  }
  scratch[tid] = amax;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }

  float scale_float = fmaxf(scratch[0], 1.0e-4f) * (1.0f / 448.0f);
  unsigned char scale_e8m0 = float_to_e8m0(scale_float);
  float scale = e8m0_to_float(scale_e8m0);
  for (int dim = start + tid; dim < end; dim += blockDim.x) {
    float value = __bfloat162float(input[base + dim]);
    // Guard zero / denormal UE8M0 scales — value/0 → Inf → NaN after *scale.
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    float clamped = fminf(fmaxf(value * inv_scale, -448.0f), 448.0f);
    // Match official tilelang act_quant inplace: Cast(FP8_E4M3) then dequant * scale.
    float quantized = (scale > 0.0f)
                          ? fp8_e4m3_to_float(float_to_fp8_e4m3(clamped)) * scale
                          : 0.0f;
    output[base + dim] = __float2bfloat16(quantized);
  }
}

extern "C" int ds_fp8_act_quant_nope_bf16(
    const void *input,
    void *output,
    int seq_len,
    int local_heads,
    int head_dim,
    int rotary_dim,
    int block_size,
    int64_t stream_i64) {
  if (input == nullptr || output == nullptr || seq_len <= 0 || local_heads <= 0 || head_dim <= 0 ||
      rotary_dim < 0 || rotary_dim >= head_dim || block_size <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int nope_dim = head_dim - rotary_dim;
  if (nope_dim % block_size != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_i64);
  constexpr int threads = 128;
  dim3 grid(seq_len, local_heads, nope_dim / block_size);
  size_t shared_bytes = threads * sizeof(float);
  ds_fp8_act_quant_nope_bf16_kernel<<<grid, threads, shared_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(input),
      reinterpret_cast<__nv_bfloat16 *>(output),
      seq_len,
      local_heads,
      head_dim,
      rotary_dim,
      block_size);
  return static_cast<int>(cudaGetLastError());
}

__global__ void ds_write_kv_row_from_pos_kernel(
    __nv_bfloat16 *__restrict__ cache,
    const __nv_bfloat16 *__restrict__ token,
    const int64_t *__restrict__ positions,
    int window_size,
    int head_dim) {
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= head_dim) return;
  int start_pos = static_cast<int>(positions[0]);
  int slot = start_pos % window_size;
  cache[slot * head_dim + dim] = token[dim];
}

__global__ void ds_write_compressed_row_from_pos_kernel(
    __nv_bfloat16 *__restrict__ cache,
    const __nv_bfloat16 *__restrict__ row,
    const int64_t *__restrict__ positions,
    int window_size,
    int head_dim,
    int ratio) {
  int start_pos = static_cast<int>(positions[0]);
  if (((start_pos + 1) % ratio) != 0) return;
  int row_idx = start_pos / ratio;
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= head_dim) return;
  cache[(window_size + row_idx) * head_dim + dim] = row[dim];
}

__global__ void ds_write_indexer_row_from_pos_kernel(
    __nv_bfloat16 *__restrict__ cache,
    const __nv_bfloat16 *__restrict__ row,
    const int64_t *__restrict__ positions,
    int head_dim,
    int ratio) {
  int start_pos = static_cast<int>(positions[0]);
  if (((start_pos + 1) % ratio) != 0) return;
  int row_idx = start_pos / ratio;
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  if (dim >= head_dim) return;
  cache[row_idx * head_dim + dim] = row[dim];
}

extern "C" int ds_write_kv_row_from_pos(
    void *cache,
    const void *token,
    const void *positions,
    int window_size,
    int head_dim,
    int64_t stream_i64) {
  if (cache == nullptr || token == nullptr || positions == nullptr ||
      window_size <= 0 || head_dim <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_i64);
  constexpr int threads = 256;
  int blocks = (head_dim + threads - 1) / threads;
  ds_write_kv_row_from_pos_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 *>(cache),
      reinterpret_cast<const __nv_bfloat16 *>(token),
      reinterpret_cast<const int64_t *>(positions),
      window_size, head_dim);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int ds_write_compressed_row_from_pos(
    void *cache,
    const void *row,
    const void *positions,
    int window_size,
    int head_dim,
    int ratio,
    int64_t stream_i64) {
  if (cache == nullptr || row == nullptr || positions == nullptr ||
      window_size <= 0 || head_dim <= 0 || ratio <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_i64);
  constexpr int threads = 256;
  int blocks = (head_dim + threads - 1) / threads;
  ds_write_compressed_row_from_pos_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 *>(cache),
      reinterpret_cast<const __nv_bfloat16 *>(row),
      reinterpret_cast<const int64_t *>(positions),
      window_size, head_dim, ratio);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int ds_write_indexer_row_from_pos(
    void *cache,
    const void *row,
    const void *positions,
    int head_dim,
    int ratio,
    int64_t stream_i64) {
  if (cache == nullptr || row == nullptr || positions == nullptr ||
      head_dim <= 0 || ratio <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_i64);
  constexpr int threads = 256;
  int blocks = (head_dim + threads - 1) / threads;
  ds_write_indexer_row_from_pos_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 *>(cache),
      reinterpret_cast<const __nv_bfloat16 *>(row),
      reinterpret_cast<const int64_t *>(positions),
      head_dim, ratio);
  return static_cast<int>(cudaGetLastError());
}
