#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

// Grid: (seq_len, num_heads), Block: 256 threads.
// Each block handles one (token, head) pair.
// Online softmax over topk gathered KV rows + attention sink denominator.
// Q: [seq_len, num_heads, head_dim] BF16
// KV: [kv_len, head_dim] BF16 (shared across heads)
// attn_sink: [num_heads] F32
// topk_idxs: [seq_len, topk] I32 (-1 = invalid/skip)
// out: [seq_len, num_heads, head_dim] BF16

static constexpr int kSparseThreads = 256;

static __device__ __forceinline__ float ds_sparse_round_bf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

__global__ void ds_sparse_attn_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ kv,
    const float *__restrict__ attn_sink,
    const int *__restrict__ topk_idxs,
    __nv_bfloat16 *__restrict__ out,
    int num_heads,
    int head_dim,
    int kv_len,
    int topk,
    float softmax_scale) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int dims_per_thread = head_dim / kSparseThreads;

  const int q_base = (token * num_heads + head) * head_dim;
  float q_reg[4];
  #pragma unroll
  for (int i = 0; i < dims_per_thread; ++i)
    q_reg[i] = __bfloat162float(q[q_base + tid * dims_per_thread + i]);

  // Invalid/padded indices must not participate in the online softmax.  In
  // particular, using a finite sentinel here makes an invalid score equal to
  // the initial running maximum and contributes exp(0) to the denominator.
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  __shared__ float s_score[kSparseThreads / 32];

  const int *tok_idxs = topk_idxs + token * topk;

  for (int ki = 0; ki < topk; ++ki) {
    int kv_idx = tok_idxs[ki];

    // The top-k builders intentionally pad causal/future slots with -1.  All
    // threads take the same branch, so skipping here is synchronization-safe
    // and keeps both the numerator and denominator unchanged.
    if (kv_idx < 0 || kv_idx >= kv_len) continue;

    float partial = 0.0f;
    const __nv_bfloat16 *krow = kv + kv_idx * head_dim;
    #pragma unroll
    for (int i = 0; i < dims_per_thread; ++i)
      partial += q_reg[i] * __bfloat162float(krow[tid * dims_per_thread + i]);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_down_sync(0xffffffff, partial, off);

    if ((tid & 31) == 0)
      s_score[tid >> 5] = partial;
    __syncthreads();

    float score;
    if (tid == 0) {
      float s = 0.0f;
      for (int w = 0; w < kSparseThreads / 32; ++w)
        s += s_score[w];
      score = s * softmax_scale;
      s_score[0] = score;
    }
    __syncthreads();
    score = s_score[0];
    // All threads must consume s_score[0] before warp leaders overwrite the
    // shared buffer in the next iteration.
    __syncthreads();

    float new_max = fmaxf(running_max, score);
    float old_scale = expf(running_max - new_max);
    float exp_score = expf(score - new_max);

    #pragma unroll
    for (int i = 0; i < dims_per_thread; ++i)
      acc[i] *= old_scale;
    running_sum = running_sum * old_scale + exp_score;
    running_max = new_max;

    // The official TileLang kernel keeps the softmax denominator in FP32 but
    // casts the unnormalized probabilities to BF16 before the P@V GEMM.
    // Retaining FP32 here looks more accurate in isolation, yet crosses a QAT
    // precision boundary and measurably changes every sparse layer.
    float value_weight = ds_sparse_round_bf16(exp_score);
    #pragma unroll
    for (int i = 0; i < dims_per_thread; ++i)
      acc[i] += value_weight * __bfloat162float(krow[tid * dims_per_thread + i]);
  }

  float sink_val = attn_sink[head];
  float sink_max = fmaxf(running_max, sink_val);
  float old_sc = expf(running_max - sink_max);
  float exp_sink = expf(sink_val - sink_max);
  #pragma unroll
  for (int i = 0; i < dims_per_thread; ++i)
    acc[i] *= old_sc;
  running_sum = running_sum * old_sc + exp_sink;

  float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
  const int out_base = (token * num_heads + head) * head_dim;
  #pragma unroll
  for (int i = 0; i < dims_per_thread; ++i)
    out[out_base + tid * dims_per_thread + i] = __float2bfloat16(acc[i] * inv_sum);
}

extern "C" int ds_sparse_attn_dispatch(
    const void* q, const void* kv, const void* attn_sink,
    const int* topk_idxs, void* out,
    int seq_len, int num_heads, int head_dim, int kv_len, int topk,
    float softmax_scale, cudaStream_t stream) {
  if (seq_len <= 0 || kv_len <= 0 || topk <= 0 || num_heads <= 0) return (int)cudaSuccess;
  // Kernel packs head_dim / 256 dims per thread into a fixed float[4] register
  // file. DeepSeek V4 Flash uses head_dim=512 (exactly 2 dims/thread).
  if (head_dim <= 0 || head_dim % kSparseThreads != 0 ||
      (head_dim / kSparseThreads) > 4) {
    return (int)cudaErrorInvalidValue;
  }
  dim3 grid(seq_len, num_heads);
  ds_sparse_attn_kernel<<<grid, kSparseThreads, 0, stream>>>(
      (const __nv_bfloat16*)q, (const __nv_bfloat16*)kv, (const float*)attn_sink,
      topk_idxs, (__nv_bfloat16*)out, num_heads, head_dim, kv_len, topk, softmax_scale);
  return (int)cudaGetLastError();
}
