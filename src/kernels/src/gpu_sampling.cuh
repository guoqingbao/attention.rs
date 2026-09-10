#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
  abort(); } } while(0)
#endif

// Ensure this matches the Rust struct layout
struct SamplerParams {
  int B;            // batch size: 1..64
  int V;            // vocab size
  float temperature; // 0 => greedy-like behavior (handled as large invT)
  float top_p;      // <=0 or >=1 => disabled; else top-p within top-k
  int top_k;        // requested top-k (<=0 means no top-k cap)
  uint64_t seed;    // base seed
  uint64_t token_pos; // monotonically increasing per generated token (for determinism)
};

// Per-sequence variant (the additive path, the QoS-gated): the temperature / top_p /
// top_k are per-batch-row tensors (the [B] device pointers), so each sequence in the
// batch samples with its own strategy. The existing SamplerParams (the single shared
// strategy) is unchanged for the other inference engines.
struct SamplerParamsPerSeq {
  int B;
  int V;
  const float* temperature_d; // [B] per-seq temperature
  const float* top_p_d;       // [B] per-seq top_p
  const unsigned int* top_k_d; // [B] per-seq top_k
  uint64_t seed;
  uint64_t token_pos;
};

// Runtime entrypoint (supports K=32, 64, 128, or 256 via template instantiation)
template<int K>
void gpu_topk_topp_sample(
    const float* logits_d,   // [B,V] row-major
    int* out_tokens_d,       // [B]
    const SamplerParams& p,
    cudaStream_t stream);

// Per-sequence entrypoint (the additive path, the QoS-gated): the temperature / top_p /
// top_k are per-batch-row tensors (the SamplerParamsPerSeq), so each sequence samples
// with its own strategy.
template<int K>
void gpu_topk_topp_sample_perseq(
    const float* logits_d,   // [B,V] row-major
    const float* mask_d,     // [B,V] 1.0=legal / 0.0=illegal, or nullptr
    int* out_tokens_d,       // [B]
    const SamplerParamsPerSeq& p,
    cudaStream_t stream);
