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

// Runtime entrypoint (supports K=32, 64, 128, or 256 via template instantiation)
template<int K>
void gpu_topk_topp_sample(
    const float* logits_d,   // [B,V] row-major
    int* out_tokens_d,       // [B]
    const SamplerParams& p,
    cudaStream_t stream);
