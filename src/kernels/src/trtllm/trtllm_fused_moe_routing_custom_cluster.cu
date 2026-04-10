/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// routingCustom cluster kernel — single-cluster fused kernel for ≤256 tokens (SM90+).
// Split out for parallel compilation (LAUNCH_ROUTING_CUSTOM causes heavy template explosion).

#if defined(USE_FLASHINFER) && defined(USE_TRTLLM)

#include "flashinfer/trtllm/common/cudaUtils.h"
#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"

namespace moe::dev::routing {
namespace routingCustom {

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  using TypePacked = PackedScoreIdx<BaseType>;
  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

  __shared__ TypePacked
      __attribute((aligned(128))) smemPackedScoreIdx[NumWarps * KernelParams::MaxNumTopExperts];

  uint32_t const clusterBlockRank = blockIdx.x;
  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();
  auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
  auto scoreOffset = warpTokenIdx * params.mNumExperts;
  bool validToken = warpTokenIdx < params.mNumTokens;
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }

  if (params.mPtrScores != nullptr) {
    BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
    int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];
    if (validToken) {
      KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize,
                                                       KernelParams::MaxNumTopExperts>(
          warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
          params.mPtrScores + scoreOffset, params);
      if (laneIdx < params.mTopK) {
        smemPackedScoreIdx[warpIdx * params.mTopK + laneIdx] =
            TypePacked{warpTopKScore[laneIdx], static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
      }
    }
  }

  __cluster_barrier_arrive();
  __cluster_barrier_wait();

  if (params.mPtrScores != nullptr) {
    routingPermutation<KernelParams, BaseType, NumThreads, NumWarps, KernelParams::MaxNumTopExperts,
                       /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx,
                                                          clusterBlockRank);
  } else {
    routingPermutation<KernelParams, BaseType, NumThreads, NumWarps, KernelParams::MaxNumTopExperts,
                       /*LoadExpertIdxFromGlobal=*/true>(params, smemPackedScoreIdx, warpIdx,
                                                         clusterBlockRank);
  }
}
#else
__global__ void __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams /* params */) {
  assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

void launchClusterKernel(Data const& data, void* stream) {
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
                        /*smemSize=*/0, stream);
}

}  // namespace routingCustom
}  // namespace moe::dev::routing

#endif  // USE_FLASHINFER
