#include <cstdint>

#if defined(USE_FLASHINFER) && __has_include("trtllm/gen/CudaRunner.h") && \
    __has_include("tensorrt_llm/common/logger.h")

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "flashinfer/trtllm/fused_moe/runner.h"

namespace trtllm_moe = tensorrt_llm::kernels::trtllmgen_moe;
namespace btg = batchedGemm::trtllm::gen;

namespace {

__global__ void pack_topk_to_bf16_packed(const int32_t* topk_ids, const float* topk_weights,
                                         int32_t* packed, int64_t numel, int32_t num_experts) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= numel) {
    return;
  }

  int32_t expert_id = topk_ids[i];
  if (expert_id < 0 || expert_id >= num_experts) {
    expert_id = 0;
  }

  __nv_bfloat16 score_bf16 = __float2bfloat16(topk_weights[i]);
  uint16_t score_bits = *reinterpret_cast<uint16_t*>(&score_bf16);
  uint16_t expert_bits = static_cast<uint16_t>(expert_id);
  packed[i] = (static_cast<int32_t>(expert_bits) << 16) | static_cast<int32_t>(score_bits);
}

int32_t choose_tile_tokens_dim(int32_t num_tokens, int32_t top_k, int32_t num_experts) {
  float avg_tokens_per_expert =
      static_cast<float>(num_tokens) * static_cast<float>(top_k) / std::max(num_experts, 1);
  int32_t tile_tokens_dim = 1;
  while (tile_tokens_dim < static_cast<int32_t>(avg_tokens_per_expert) && tile_tokens_dim < 128) {
    tile_tokens_dim <<= 1;
  }
  tile_tokens_dim = std::clamp(tile_tokens_dim, 8, 128);
  return tile_tokens_dim;
}

btg::Dtype parse_dtype(int32_t dtype_code) {
  switch (dtype_code) {
    case 0:
      return btg::Dtype::Fp16;
    case 1:
      return btg::Dtype::Bfloat16;
    default:
      throw std::runtime_error("Unsupported dtype code for fused moe");
  }
}

size_t dtype_size(btg::Dtype dtype) {
  switch (dtype) {
    case btg::Dtype::Fp16:
      return sizeof(__half);
    case btg::Dtype::Bfloat16:
      return sizeof(__nv_bfloat16);
    case btg::Dtype::E4m3:
      return sizeof(uint8_t);
    default:
      throw std::runtime_error("Unsupported dtype size query");
  }
}

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  void* ensure(size_t required_bytes, cudaStream_t stream) {
    if (required_bytes == 0) {
      return nullptr;
    }
    if (ptr != nullptr && bytes >= required_bytes) {
      return ptr;
    }
    if (ptr != nullptr) {
      cudaError_t free_err = cudaFreeAsync(ptr, stream);
      if (free_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaFreeAsync failed: ") +
                                 cudaGetErrorString(free_err));
      }
      ptr = nullptr;
      bytes = 0;
    }
    cudaError_t err = cudaMallocAsync(&ptr, required_bytes, stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMallocAsync failed: ") +
                               cudaGetErrorString(err));
    }
    bytes = required_bytes;
    return ptr;
  }

  template <typename T>
  T* ensure_typed(size_t count, cudaStream_t stream) {
    return static_cast<T*>(ensure(count * sizeof(T), stream));
  }
};

struct StreamMoECache {
  int device = -1;
  cudaStream_t stream = nullptr;

  int routing_tile_tokens_dim = -1;
  std::unique_ptr<trtllm_moe::Routing::Runner> routing_runner;

  int bf16_tile_tokens_dim = -1;
  btg::Dtype bf16_input_dtype = btg::Dtype::Fp16;
  btg::Dtype bf16_weight_dtype = btg::Dtype::Fp16;
  std::unique_ptr<trtllm_moe::MoE::Runner> bf16_runner;

  int fp8_tile_tokens_dim = -1;
  std::unique_ptr<trtllm_moe::MoE::Runner> fp8_runner;

  DeviceBuffer packed_topk;
  DeviceBuffer num_tokens_per_expert;
  DeviceBuffer expert_count_histogram;
  DeviceBuffer permuted_idx_size;
  DeviceBuffer expanded_idx_to_permuted_idx;
  DeviceBuffer permuted_idx_to_token_idx;
  DeviceBuffer expert_weights;
  DeviceBuffer cta_idx_xy_to_batch_idx;
  DeviceBuffer cta_idx_xy_to_mn_limit;
  DeviceBuffer num_non_exiting_ctas;

  DeviceBuffer bmm1_workspace;
  DeviceBuffer bmm2_workspace;
  DeviceBuffer gemm1_output;
  DeviceBuffer gemm1_output_scale;
  DeviceBuffer activation_output;
  DeviceBuffer activation_output_scale;
  DeviceBuffer gemm2_output;
};

uint64_t make_cache_key(int device, cudaStream_t stream) {
  uint64_t s = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
  return (static_cast<uint64_t>(static_cast<uint32_t>(device)) << 32) ^ s;
}

StreamMoECache& get_stream_cache(cudaStream_t stream) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGetDevice failed: ") +
                             cudaGetErrorString(err));
  }

  static thread_local std::unordered_map<uint64_t, std::unique_ptr<StreamMoECache>> caches;
  uint64_t key = make_cache_key(device, stream);
  auto it = caches.find(key);
  if (it == caches.end()) {
    auto cache = std::make_unique<StreamMoECache>();
    cache->device = device;
    cache->stream = stream;
    it = caches.emplace(key, std::move(cache)).first;
  }
  return *it->second;
}

void run_routing_from_precomputed_topk(
    const int32_t* topk_ids, const float* topk_weights, int32_t num_tokens, int32_t num_experts,
    int32_t top_k, int32_t tile_tokens_dim, btg::Dtype dtype_elt,
    trtllm_moe::MoE::MoEWorkspace& workspace, StreamMoECache& cache, cudaStream_t stream) {
  int64_t expanded_elems = static_cast<int64_t>(num_tokens) * top_k;

  int32_t* packed_topk = cache.packed_topk.ensure_typed<int32_t>(expanded_elems, stream);

  int threads = 256;
  int blocks = static_cast<int>((expanded_elems + threads - 1) / threads);
  if (blocks > 0) {
    pack_topk_to_bf16_packed<<<blocks, threads, 0, stream>>>(topk_ids, topk_weights, packed_topk,
                                                              expanded_elems, num_experts);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      throw std::runtime_error(std::string("pack_topk_to_bf16_packed launch failed: ") +
                               cudaGetErrorString(launch_err));
    }
  }

  int32_t max_num_ctas = trtllm_moe::Routing::getMaxNumCtasInBatchDim(num_tokens, top_k,
                                                                       num_experts, tile_tokens_dim);
  int32_t total_max_padded_tokens = trtllm_moe::Routing::getMaxPermutedPaddedCount(
      num_tokens, top_k, num_experts, tile_tokens_dim);

  workspace.total_max_padded_tokens = total_max_padded_tokens;
  workspace.ProjUpTileN = tile_tokens_dim;

  int32_t* num_tokens_per_expert =
      cache.num_tokens_per_expert.ensure_typed<int32_t>(num_experts, stream);
  int32_t* expert_count_histogram =
      cache.expert_count_histogram.ensure_typed<int32_t>(std::max(num_experts * 2, 256 * 2),
                                                         stream);

  workspace.routing_expert_indexes = packed_topk;
  workspace.permuted_idx_size = cache.permuted_idx_size.ensure_typed<int32_t>(1, stream);
  workspace.total_num_padded_tokens = workspace.permuted_idx_size;
  workspace.expanded_idx_to_permuted_idx =
      cache.expanded_idx_to_permuted_idx.ensure_typed<int32_t>(expanded_elems, stream);
  workspace.permuted_idx_to_expanded_idx = nullptr;
  workspace.permuted_idx_to_token_idx =
      cache.permuted_idx_to_token_idx.ensure_typed<int32_t>(total_max_padded_tokens, stream);
  workspace.expert_weights =
      cache.expert_weights.ensure_typed<__nv_bfloat16>(expanded_elems, stream);
  workspace.token_scales = nullptr;
  workspace.cta_idx_xy_to_batch_idx =
      cache.cta_idx_xy_to_batch_idx.ensure_typed<int32_t>(max_num_ctas, stream);
  workspace.cta_idx_xy_to_mn_limit =
      cache.cta_idx_xy_to_mn_limit.ensure_typed<int32_t>(max_num_ctas, stream);
  workspace.num_non_exiting_ctas =
      cache.num_non_exiting_ctas.ensure_typed<int32_t>(1, stream);

  if (!cache.routing_runner || cache.routing_tile_tokens_dim != tile_tokens_dim) {
    cache.routing_runner = std::make_unique<trtllm_moe::Routing::Runner>(tile_tokens_dim);
    cache.routing_tile_tokens_dim = tile_tokens_dim;
  }
  cache.routing_runner->run(
      /*routingLogits=*/nullptr,
      /*routingBias=*/nullptr,
      num_tokens,
      num_experts,
      top_k,
      /*nGroups=*/0,
      /*topkGroups=*/0,
      /*localExpertOffset=*/0,
      /*localNumExperts=*/num_experts,
      /*routedScalingFactor=*/1.f,
      workspace.routing_expert_indexes,
      expert_count_histogram,
      workspace.permuted_idx_size,
      workspace.expanded_idx_to_permuted_idx,
      workspace.permuted_idx_to_expanded_idx,
      workspace.permuted_idx_to_token_idx,
      workspace.expert_weights,
      num_tokens_per_expert,
      workspace.cta_idx_xy_to_batch_idx,
      workspace.cta_idx_xy_to_mn_limit,
      workspace.num_non_exiting_ctas,
      dtype_elt,
      btg::Dtype::Bfloat16,
      /*useRoutingScalesOnInput=*/false,
      /*useDeepSeekFp8=*/false,
      trtllm_moe::Routing::RoutingMethodType::Renormalize,
      stream);
}

int run_fused_moe_bf16(const void* input, const int32_t* topk_ids, const float* topk_weights,
                       const void* gate_up_weights, const void* down_weights, void* output,
                       int32_t num_tokens, int32_t hidden_size, int32_t intermediate_size,
                       int32_t num_experts, int32_t top_k, int32_t input_dtype_code,
                       int32_t weight_dtype_code, cudaStream_t stream) {
  btg::Dtype input_dtype = parse_dtype(input_dtype_code);
  btg::Dtype weight_dtype = parse_dtype(weight_dtype_code);
  int32_t tile_tokens_dim = choose_tile_tokens_dim(num_tokens, top_k, num_experts);

  StreamMoECache& cache = get_stream_cache(stream);
  trtllm_moe::MoE::MoEWorkspace workspace{};

  run_routing_from_precomputed_topk(topk_ids, topk_weights, num_tokens, num_experts, top_k,
                                    tile_tokens_dim, input_dtype, workspace, cache, stream);

  if (!cache.bf16_runner || cache.bf16_tile_tokens_dim != tile_tokens_dim ||
      cache.bf16_input_dtype != input_dtype || cache.bf16_weight_dtype != weight_dtype) {
    cache.bf16_runner = std::make_unique<trtllm_moe::MoE::Runner>(
        input_dtype, weight_dtype, /*useDeepSeekFp8=*/false, tile_tokens_dim,
        trtllm_moe::MoE::GatedActType::SwiGlu, /*useShuffledMatrixA=*/false,
        batchedGemm::gemm::MatrixLayout::MajorK);
    cache.bf16_tile_tokens_dim = tile_tokens_dim;
    cache.bf16_input_dtype = input_dtype;
    cache.bf16_weight_dtype = weight_dtype;
  }

  trtllm_moe::MoE::MoERunnerArgs args{};
  args.hidden_states = const_cast<void*>(input);
  args.gemm1_weights = const_cast<void*>(gate_up_weights);
  args.gemm2_weights = const_cast<void*>(down_weights);
  args.output = output;

  args.num_tokens = num_tokens;
  args.num_experts = num_experts;
  args.hidden_size = hidden_size;
  args.intermediate_size = intermediate_size;
  args.top_k = top_k;
  args.local_expert_offset = 0;
  args.local_num_experts = num_experts;
  args.mDtypeElt = input_dtype;
  args.mDtypeExpW = btg::Dtype::Bfloat16;
  args.mDtypeOut = input_dtype;
  args.mUseRoutingScalesOnInput = false;
  args.mUseDeepSeekFp8 = false;
  args.output_scale = nullptr;
  args.do_finalize = true;

  int64_t config_idx =
      cache.bf16_runner->getDefaultValidConfigIndex(top_k, hidden_size, intermediate_size,
                                                    num_experts, num_tokens);

  auto workspace_sizes = cache.bf16_runner->getWorkspaceSizeInBytes(args, config_idx);
  workspace.bmm1_workspace =
      cache.bmm1_workspace.ensure(static_cast<size_t>(std::get<0>(workspace_sizes)), stream);
  workspace.bmm2_workspace =
      cache.bmm2_workspace.ensure(static_cast<size_t>(std::get<1>(workspace_sizes)), stream);

  int32_t max_num_padded_tokens = workspace.total_max_padded_tokens;
  size_t gemm1_elem_size = dtype_size(input_dtype);
  size_t gemm2_elem_size = dtype_size(input_dtype);

  workspace.hidden_states_scale_linear = nullptr;
  workspace.gemm1_output = cache.gemm1_output.ensure(
      static_cast<size_t>(max_num_padded_tokens) * intermediate_size * gemm1_elem_size, stream);
  workspace.gemm1_output_scale = nullptr;
  workspace.activation_output = cache.activation_output.ensure(
      static_cast<size_t>(max_num_padded_tokens) * intermediate_size * gemm1_elem_size, stream);
  workspace.activation_output_scale = nullptr;
  workspace.gemm2_output = cache.gemm2_output.ensure(
      static_cast<size_t>(max_num_padded_tokens) * hidden_size * gemm2_elem_size, stream);
  workspace.gemm2_output_scale = nullptr;
  cache.bf16_runner->run(args, workspace, cache.device, stream, config_idx, /*enable_pdl=*/true);
  return 0;
}

int run_fused_moe_fp8(const void* input, const int32_t* topk_ids, const float* topk_weights,
                      const uint8_t* gate_up_weights, const float* gate_up_scales,
                      const uint8_t* down_weights, const float* down_scales, void* output,
                      int32_t num_tokens, int32_t hidden_size, int32_t intermediate_size,
                      int32_t num_experts, int32_t top_k, int32_t input_dtype_code,
                      cudaStream_t stream) {
  btg::Dtype input_dtype = parse_dtype(input_dtype_code);
  int32_t tile_tokens_dim = choose_tile_tokens_dim(num_tokens, top_k, num_experts);

  if (hidden_size % 128 != 0 || intermediate_size % 128 != 0) {
    throw std::runtime_error("FP8 fused moe requires hidden/intermediate dims divisible by 128");
  }

  StreamMoECache& cache = get_stream_cache(stream);
  trtllm_moe::MoE::MoEWorkspace workspace{};

  run_routing_from_precomputed_topk(topk_ids, topk_weights, num_tokens, num_experts, top_k,
                                    tile_tokens_dim, input_dtype, workspace, cache, stream);

  if (!cache.fp8_runner || cache.fp8_tile_tokens_dim != tile_tokens_dim) {
    cache.fp8_runner = std::make_unique<trtllm_moe::MoE::Runner>(
        btg::Dtype::E4m3, /*useDeepSeekFp8=*/true, tile_tokens_dim,
        /*useShuffledMatrixA=*/false, batchedGemm::gemm::MatrixLayout::MajorK);
    cache.fp8_tile_tokens_dim = tile_tokens_dim;
  }

  trtllm_moe::MoE::MoERunnerArgs args{};
  args.hidden_states = const_cast<void*>(input);
  args.hidden_states_scale = nullptr;
  args.gemm1_weights = const_cast<uint8_t*>(gate_up_weights);
  args.gemm1_weights_scale = const_cast<float*>(gate_up_scales);
  args.gemm2_weights = const_cast<uint8_t*>(down_weights);
  args.gemm2_weights_scale = const_cast<float*>(down_scales);
  args.output = output;

  args.num_tokens = num_tokens;
  args.num_experts = num_experts;
  args.hidden_size = hidden_size;
  args.intermediate_size = intermediate_size;
  args.top_k = top_k;
  args.local_expert_offset = 0;
  args.local_num_experts = num_experts;
  args.mDtypeElt = input_dtype;
  args.mDtypeExpW = btg::Dtype::Bfloat16;
  args.mDtypeOut = btg::Dtype::Bfloat16;
  args.mUseRoutingScalesOnInput = false;
  args.mUseDeepSeekFp8 = true;
  args.output_scale = nullptr;
  args.do_finalize = true;

  int64_t config_idx =
      cache.fp8_runner->getDefaultValidConfigIndex(top_k, hidden_size, intermediate_size,
                                                   num_experts, num_tokens);

  auto workspace_sizes = cache.fp8_runner->getWorkspaceSizeInBytes(args, config_idx);
  workspace.bmm1_workspace =
      cache.bmm1_workspace.ensure(static_cast<size_t>(std::get<0>(workspace_sizes)), stream);
  workspace.bmm2_workspace =
      cache.bmm2_workspace.ensure(static_cast<size_t>(std::get<1>(workspace_sizes)), stream);

  int32_t max_num_padded_tokens_gemm1 = workspace.total_max_padded_tokens + num_experts;
  int32_t max_num_padded_tokens_gemm2 = workspace.total_max_padded_tokens;

  workspace.hidden_states_scale_linear = nullptr;
  workspace.gemm1_output = cache.gemm1_output.ensure(
      static_cast<size_t>(max_num_padded_tokens_gemm1) * 2 * intermediate_size, stream);
  workspace.gemm1_output_scale = cache.gemm1_output_scale.ensure(
      static_cast<size_t>(2 * intermediate_size / 128) * max_num_padded_tokens_gemm1 *
          sizeof(float),
      stream);
  workspace.activation_output = cache.activation_output.ensure(
      static_cast<size_t>(max_num_padded_tokens_gemm1) * intermediate_size, stream);
  workspace.activation_output_scale = cache.activation_output_scale.ensure(
      static_cast<size_t>(intermediate_size / 128) * max_num_padded_tokens_gemm1 *
          sizeof(float),
      stream);
  workspace.gemm2_output = cache.gemm2_output.ensure(
      static_cast<size_t>(max_num_padded_tokens_gemm2) * hidden_size * sizeof(__nv_bfloat16),
      stream);
  workspace.gemm2_output_scale = nullptr;
  cache.fp8_runner->run(args, workspace, cache.device, stream, config_idx, /*enable_pdl=*/true);
  return 0;
}

#if defined(ENABLE_FP4) && defined(FLASHINFER_ENABLE_FP4_E2M1)
int run_fused_moe_mxfp4(const void* input, const int32_t* topk_ids, const float* topk_weights,
                         const uint8_t* gate_up_weights, const uint8_t* gate_up_scales,
                         const uint8_t* down_weights, const uint8_t* down_scales, void* output,
                         int32_t num_tokens, int32_t hidden_size, int32_t intermediate_size,
                         int32_t num_experts, int32_t top_k, int32_t input_dtype_code,
                         cudaStream_t stream) {
  btg::Dtype input_dtype = parse_dtype(input_dtype_code);
  int32_t tile_tokens_dim = choose_tile_tokens_dim(num_tokens, top_k, num_experts);

  if (hidden_size % 32 != 0 || intermediate_size % 32 != 0) {
    throw std::runtime_error("MXFP4 fused moe requires hidden/intermediate dims divisible by 32");
  }

  StreamMoECache& cache = get_stream_cache(stream);
  trtllm_moe::MoE::MoEWorkspace workspace{};

  run_routing_from_precomputed_topk(topk_ids, topk_weights, num_tokens, num_experts, top_k,
                                    tile_tokens_dim, input_dtype, workspace, cache, stream);

  if (!cache.fp8_runner || cache.fp8_tile_tokens_dim != tile_tokens_dim) {
    cache.fp8_runner = std::make_unique<trtllm_moe::MoE::Runner>(
        input_dtype, btg::Dtype::MxE2m1,
        /*useDeepSeekFp8=*/false, tile_tokens_dim,
        trtllm_moe::MoE::GatedActType::SwiGlu,
        /*useShuffledMatrixA=*/false, batchedGemm::gemm::MatrixLayout::MajorK);
    cache.fp8_tile_tokens_dim = tile_tokens_dim;
  }

  trtllm_moe::MoE::MoERunnerArgs args{};
  args.hidden_states = const_cast<void*>(input);
  args.hidden_states_scale = nullptr;
  args.gemm1_weights = const_cast<uint8_t*>(gate_up_weights);
  args.gemm1_weights_scale = const_cast<uint8_t*>(gate_up_scales);
  args.gemm2_weights = const_cast<uint8_t*>(down_weights);
  args.gemm2_weights_scale = const_cast<uint8_t*>(down_scales);
  args.output = output;

  args.num_tokens = num_tokens;
  args.num_experts = num_experts;
  args.hidden_size = hidden_size;
  args.intermediate_size = intermediate_size;
  args.top_k = top_k;
  args.local_expert_offset = 0;
  args.local_num_experts = num_experts;
  args.mDtypeElt = input_dtype;
  args.mDtypeExpW = btg::Dtype::Bfloat16;
  args.mDtypeOut = input_dtype;
  args.mUseRoutingScalesOnInput = false;
  args.mUseDeepSeekFp8 = false;
  args.output_scale = nullptr;
  args.do_finalize = true;

  int64_t config_idx =
      cache.fp8_runner->getDefaultValidConfigIndex(top_k, hidden_size, intermediate_size,
                                                   num_experts, num_tokens);

  auto workspace_sizes = cache.fp8_runner->getWorkspaceSizeInBytes(args, config_idx);
  workspace.bmm1_workspace =
      cache.bmm1_workspace.ensure(static_cast<size_t>(std::get<0>(workspace_sizes)), stream);
  workspace.bmm2_workspace =
      cache.bmm2_workspace.ensure(static_cast<size_t>(std::get<1>(workspace_sizes)), stream);

  int32_t max_num_padded_tokens = workspace.total_max_padded_tokens;
  size_t elt_size = dtype_size(input_dtype);

  workspace.hidden_states_scale_linear = nullptr;
  workspace.gemm1_output = cache.gemm1_output.ensure(
      static_cast<size_t>(max_num_padded_tokens) * 2 * intermediate_size * elt_size, stream);
  workspace.gemm1_output_scale = nullptr;
  workspace.activation_output = cache.activation_output.ensure(
      static_cast<size_t>(max_num_padded_tokens) * intermediate_size * elt_size, stream);
  workspace.activation_output_scale = nullptr;
  workspace.gemm2_output = cache.gemm2_output.ensure(
      static_cast<size_t>(max_num_padded_tokens) * hidden_size * elt_size, stream);
  workspace.gemm2_output_scale = nullptr;
  cache.fp8_runner->run(args, workspace, cache.device, stream, config_idx, /*enable_pdl=*/true);
  return 0;
}
#endif  // ENABLE_FP4 && FLASHINFER_ENABLE_FP4_E2M1

}  // namespace

extern "C" int flashinfer_fused_moe_bf16(const void* input, const int32_t* topk_ids,
                                          const float* topk_weights,
                                          const void* gate_up_weights,
                                          const void* down_weights, void* output,
                                          int32_t num_tokens, int32_t hidden_size,
                                          int32_t intermediate_size, int32_t num_experts,
                                          int32_t top_k, int32_t input_dtype_code,
                                          int32_t weight_dtype_code, int64_t stream) {
  try {
    return run_fused_moe_bf16(input, topk_ids, topk_weights, gate_up_weights, down_weights,
                              output, num_tokens, hidden_size, intermediate_size, num_experts,
                              top_k, input_dtype_code, weight_dtype_code,
                              reinterpret_cast<cudaStream_t>(stream));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "flashinfer_fused_moe_bf16 failed: %s\n", e.what());
    return -1;
  }
}

extern "C" int flashinfer_fused_moe_fp8(
    const void* input, const int32_t* topk_ids, const float* topk_weights,
    const uint8_t* gate_up_weights, const float* gate_up_scales, const uint8_t* down_weights,
    const float* down_scales, void* output, int32_t num_tokens, int32_t hidden_size,
    int32_t intermediate_size, int32_t num_experts, int32_t top_k, int32_t input_dtype_code,
    int64_t stream) {
  try {
    return run_fused_moe_fp8(input, topk_ids, topk_weights, gate_up_weights, gate_up_scales,
                             down_weights, down_scales, output, num_tokens, hidden_size,
                             intermediate_size, num_experts, top_k, input_dtype_code,
                             reinterpret_cast<cudaStream_t>(stream));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "flashinfer_fused_moe_fp8 failed: %s\n", e.what());
    return -1;
  }
}

extern "C" int flashinfer_fused_moe_mxfp4(
    const void* input, const int32_t* topk_ids, const float* topk_weights,
    const uint8_t* gate_up_weights, const uint8_t* gate_up_scales, const uint8_t* down_weights,
    const uint8_t* down_scales, void* output, int32_t num_tokens, int32_t hidden_size,
    int32_t intermediate_size, int32_t num_experts, int32_t top_k, int32_t input_dtype_code,
    int64_t stream) {
#if defined(ENABLE_FP4) && defined(FLASHINFER_ENABLE_FP4_E2M1)
  try {
    return run_fused_moe_mxfp4(input, topk_ids, topk_weights, gate_up_weights, gate_up_scales,
                                down_weights, down_scales, output, num_tokens, hidden_size,
                                intermediate_size, num_experts, top_k, input_dtype_code,
                                reinterpret_cast<cudaStream_t>(stream));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "flashinfer_fused_moe_mxfp4 failed: %s\n", e.what());
    return -1;
  }
#else
  return -1;
#endif
}

#else

extern "C" int flashinfer_fused_moe_bf16(
    const void*,
    const int32_t*,
    const float*,
    const void*,
    const void*,
    void*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int64_t) {
  return -1;
}

extern "C" int flashinfer_fused_moe_fp8(
    const void*,
    const int32_t*,
    const float*,
    const uint8_t*,
    const float*,
    const uint8_t*,
    const float*,
    void*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int64_t) {
  return -1;
}

extern "C" int flashinfer_fused_moe_mxfp4(
    const void*,
    const int32_t*,
    const float*,
    const uint8_t*,
    const uint8_t*,
    const uint8_t*,
    const uint8_t*,
    void*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int64_t) {
  return -1;
}

#endif  // USE_FLASHINFER && required TRT-LLM headers
