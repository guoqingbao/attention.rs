// Qwen4 QSA (Qwen Sparse Attention) indexer mask builder.
// Reference: Qwen4ExpTextQSAIndexer in HuggingFace modeling_qwen4_exp.py

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float relu(float x) { return x > 0.0f ? x : 0.0f; }

__device__ void apply_rope_partial(
    float* vec,
    int head_dim,
    int rotary_dim,
    const float* cos,
    const float* sin) {
  for (int i = 0; i < rotary_dim / 2; ++i) {
    float x0 = vec[i];
    float x1 = vec[i + rotary_dim / 2];
    float c = cos[i];
    float s = sin[i];
    vec[i] = x0 * c - x1 * s;
    vec[i + rotary_dim / 2] = x0 * s + x1 * c;
  }
}

__device__ void rms_norm_vec(float* vec, const __nv_bfloat16* weight, int dim, float eps) {
  float sum_sq = 0.0f;
  for (int i = 0; i < dim; ++i) sum_sq += vec[i] * vec[i];
  float rms = rsqrtf(sum_sq / (float)dim + eps);
  for (int i = 0; i < dim; ++i) {
    float w = 1.0f + bf16_to_f32(weight[i]);
    vec[i] = w * vec[i] * rms;
  }
}

// One block per query position.
__global__ void qwen4_qsa_indexer_mask_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* raw_keys,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const float* cos_table,
    const float* sin_table,
    float* mask_out,
    int seq_len,
    int kv_len,
    int n_heads,
    int head_dim,
    int rotary_dim,
    int compress_ratio,
    int block_topk,
    float score_scale,
    float mask_min,
    float eps) {
  const int q_idx = blockIdx.x;
  if (q_idx >= seq_len) return;

  const int num_visible = min(q_idx + 1, kv_len);
  const int num_complete_blocks = num_visible / compress_ratio;

  const int half_dim = rotary_dim / 2;

  // Prepare query with q_layernorm + partial RoPE at q_idx
  float q_heads[4][128];
  for (int h = 0; h < n_heads; ++h) {
    for (int d = 0; d < head_dim; ++d) {
      q_heads[h][d] = bf16_to_f32(q[(q_idx * n_heads + h) * head_dim + d]);
    }
    rms_norm_vec(q_heads[h], q_norm_weight, head_dim, eps);
    apply_rope_partial(q_heads[h], head_dim, rotary_dim,
                       cos_table + q_idx * half_dim,
                       sin_table + q_idx * half_dim);
  }

  // Score complete blocks
  const int max_blocks = 512;
  float block_scores[max_blocks];
  int num_blocks = min(num_complete_blocks, max_blocks);
  for (int b = 0; b < num_blocks; ++b) {
    float pooled[128];
    for (int d = 0; d < head_dim; ++d) pooled[d] = 0.0f;
    for (int t = 0; t < compress_ratio; ++t) {
      int pos = b * compress_ratio + t;
      for (int d = 0; d < head_dim; ++d) {
        pooled[d] += bf16_to_f32(raw_keys[pos * head_dim + d]);
      }
    }
    for (int d = 0; d < head_dim; ++d) pooled[d] /= (float)compress_ratio;
    rms_norm_vec(pooled, k_norm_weight, head_dim, eps);
    const int block_start = b * compress_ratio;
    apply_rope_partial(pooled, head_dim, rotary_dim,
                       cos_table + block_start * half_dim,
                       sin_table + block_start * half_dim);

    float score = 0.0f;
    for (int h = 0; h < n_heads; ++h) {
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) dot += q_heads[h][d] * pooled[d];
      score += relu(dot);
    }
    block_scores[b] = score * score_scale;
  }

  // Top-k block selection (simple insertion sort for small k)
  int selected[max_blocks];
  int num_selected = min(block_topk, num_blocks);
  for (int i = 0; i < num_selected; ++i) selected[i] = -1;
  for (int k = 0; k < num_selected; ++k) {
    int best = -1;
    float best_score = -1e30f;
    for (int b = 0; b < num_blocks; ++b) {
      bool used = false;
      for (int j = 0; j < k; ++j)
        if (selected[j] == b) used = true;
      if (!used && block_scores[b] > best_score) {
        best_score = block_scores[b];
        best = b;
      }
    }
    selected[k] = best;
  }

  // Build mask: default mask_min (blocked), 0.0 for selected + tail
  float* row = mask_out + q_idx * kv_len;
  for (int i = 0; i < kv_len; ++i) {
    row[i] = (i <= q_idx) ? mask_min : mask_min;
  }
  for (int k = 0; k < num_selected; ++k) {
    int b = selected[k];
    if (b < 0) continue;
    for (int t = 0; t < compress_ratio; ++t) {
      int pos = b * compress_ratio + t;
      if (pos < num_visible) row[pos] = 0.0f;
    }
  }
  // Tail tokens (incomplete block) always visible
  for (int pos = num_complete_blocks * compress_ratio; pos < num_visible; ++pos) {
    row[pos] = 0.0f;
  }
}

}  // namespace

extern "C" int qwen4_qsa_indexer_mask(
    const void* q,
    const void* raw_keys,
    const void* q_norm_weight,
    const void* k_norm_weight,
    const void* cos_table,
    const void* sin_table,
    void* mask_out,
    int seq_len,
    int kv_len,
    int n_heads,
    int head_dim,
    int rotary_dim,
    int compress_ratio,
    int block_topk,
    float score_scale,
    float mask_min,
    float eps,
    int64_t stream_) {
  if (n_heads > 4 || head_dim > 128 || block_topk > 512) return -2;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
  qwen4_qsa_indexer_mask_kernel<<<seq_len, 1, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(raw_keys),
      static_cast<const __nv_bfloat16*>(q_norm_weight),
      static_cast<const __nv_bfloat16*>(k_norm_weight),
      static_cast<const float*>(cos_table),
      static_cast<const float*>(sin_table),
      static_cast<float*>(mask_out),
      seq_len, kv_len, n_heads, head_dim, rotary_dim,
      compress_ratio, block_topk, score_scale, mask_min, eps);
  return static_cast<int>(cudaGetLastError());
}
