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

// routingCustom entry point + histogramScores kernel + coop/utility wrappers.
//
// The block and cluster kernels (which each expand LAUNCH_ROUTING_CUSTOM with
// heavy template instantiation) are split into separate TUs for parallel compilation:
//   - trtllm_fused_moe_routing_custom_block.cu
//   - trtllm_fused_moe_routing_custom_cluster.cu

#if defined(USE_FLASHINFER) && defined(USE_TRTLLM)

#include "flashinfer/trtllm/common/cudaUtils.h"
#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"

namespace moe::dev::routing {
namespace routingCustom {

// Forward declarations for functions defined in split TUs
void launchBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream);
void launchClusterKernel(Data const& data, void* stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
// HistogramScores kernel — computes TopK from raw scores and initializes expert counts.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts
                                                                      : 1024)
    routingIndicesHistogramScoresKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  static constexpr int NumThreadsBlock =
      KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024;
  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

  int32_t const laneIdx = cutlass::arch::LaneId();
  int32_t const warpIdx = threadIdx.x / WarpSize;
  int32_t const globalWarpIdx = blockIdx.x * NumThreadsBlock / WarpSize + warpIdx;
  int32_t const globalWarpStride = gridDim.x * NumThreadsBlock / WarpSize;
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  int32_t expertCountsNum = 2 * params.mNumExperts;
  int32_t globalThreadIdx = blockIdx.x * NumThreadsBlock + threadIdx.x;
  int32_t globalThreadStride = gridDim.x * NumThreadsBlock;
  initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

  BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
  int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];
  for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride) {
    auto scoreOffset = tokenIdx * params.mNumExperts;

    KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize,
                                                     KernelParams::MaxNumTopExperts>(
        warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
        params.mPtrScores + scoreOffset, params);

    if (laneIdx < params.mTopK) {
      PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(warpTopKScore[laneIdx]),
                                          static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
      params.mPtrTopKPacked[tokenIdx * params.mTopK + laneIdx] = packedScore;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.mUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

void launchHistogramScoresKernel(Data const& data, uint32_t maxNumBlocks,
                                 uint32_t numThreadsHist, void* stream) {
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesHistogramScoresKernel, maxNumBlocks,
                        numThreadsHist, /*smemSize=*/0, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Coop kernel + utility kernel launch wrappers (lighter: fixed-policy dispatch only)
////////////////////////////////////////////////////////////////////////////////////////////////////

void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist,
                      void* stream) {
  if (data.mNumExperts <= NumExperts128Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, 0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts128Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts256Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, 0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts256Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts512Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, 0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts512Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts1024Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, 0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts1024Experts, NumTop8Experts);
  } else {
    FLASHINFER_CHECK(false, "Coop kernel does not support numExperts > ", NumExperts1024Experts,
                     ", got ", data.mNumExperts);
  }
}

void launchInitExpertCounts(Data const& data, uint32_t numThreadsHist, void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingInitExpertCounts,
                                  (2 * data.mNumExperts - 1) / numThreadsHist + 1, numThreadsHist,
                                  /*smemSize=*/0, stream);
}

void launchHistogramKernel(Data const& data, int numBlocksHistogram,
                           uint32_t numThreadsHist, void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingIndicesHistogramKernel, numBlocksHistogram,
                                  numThreadsHist, /*smemSize=*/0, stream);
}

void launchOffsetsKernel(Data const& data, int numBlocksOffsets, uint32_t numThreadsHist,
                         void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingIndicesOffsetsKernel, numBlocksOffsets,
                                  numThreadsHist, /*smemSize=*/0, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Entry point
////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  FLASHINFER_CHECK(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr ||
                       data.mPtrTopKIds != nullptr,
                   "Routing kernel requires at least one input parameter");

  if (data.mPtrTopKIds != nullptr ||
      (data.mPtrTopKPacked != nullptr && data.mPtrScores == nullptr)) {
    if (data.mPtrTopKIds != nullptr) {
      FLASHINFER_CHECK(data.mPtrTopKWeights != nullptr,
                       "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided.");
    }
    uint32_t const numThreadsHist =
        std::min(1024u, static_cast<uint32_t>(getMaxNumExperts(data.mNumExperts)));
    moe::dev::routing::runPostTopKPipeline(data, numThreadsHist, stream);
    return;
  }

  FLASHINFER_CHECK(data.mPtrScores != nullptr, "Expected mPtrScores to be non-null.");
  FLASHINFER_CHECK(
      data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr &&
          data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
      "Custom routing kernel expects permuted idx and grouped Gemm launch config buffers");
  FLASHINFER_CHECK(data.mTopK <= static_cast<int32_t>(MaxSupportedTopExperts),
                   "Routing kernel expects topK experts <= ", MaxSupportedTopExperts, ", got ",
                   data.mTopK);
  FLASHINFER_CHECK(data.mNumExperts <= static_cast<int32_t>(MaxSupportedExperts),
                   "Routing kernel expects #experts <= ", MaxSupportedExperts, ", got ",
                   data.mNumExperts);
  FLASHINFER_CHECK(data.mNumExperts % 4 == 0,
                   "Routing kernel expects #experts to be a multiple of 4, got ",
                   data.mNumExperts);

  static int const smMajor = tensorrt_llm::common::getSMVersion() / 10;
  bool const useSingleBlock = data.mNumTokens <= BlockKernelMaxNumTokens;
  bool const useSingleCluster =
      (smMajor >= 9) && (data.mNumTokens <= MaxNumTokensSingleClusterScores);

  if (!useSingleCluster && !useSingleBlock) {
    FLASHINFER_CHECK(data.mPtrTopKPacked != nullptr,
                     "When #tokens is large, `mPtrTopKPacked` is a required input.");
    FLASHINFER_CHECK(data.mPtrExpertCounts != nullptr,
                     "When #tokens is large, `mPtrExpertCounts` is a required input.");
  }

  uint32_t const numThreadsHist =
      std::min(1024u, static_cast<uint32_t>(getMaxNumExperts(data.mNumExperts)));

  Data mutableData = data;
  bool const pdl = data.mUsePdl;

  if (useSingleBlock) {
    mutableData.mPdlOverlapWithNext = false;
    launchBlockKernel(mutableData, numThreadsHist, stream);
  } else if (useSingleCluster) {
    mutableData.mPdlOverlapWithNext = false;
    launchClusterKernel(mutableData, stream);
  } else {
    uint32_t const maxNumBlocks = 1024;

    mutableData.mPdlOverlapWithNext = pdl;
    launchHistogramScoresKernel(mutableData, maxNumBlocks, numThreadsHist, stream);

    bool const canUseCoop =
        (smMajor >= 9) && (data.mNumExperts <= 1024) && (data.mPtrPermutedIdxSize != nullptr);
    bool useCoop = false;
    int numBlocksCoop = 0;

    if (canUseCoop) {
      static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
      numBlocksCoop = smCount - kReservedSMsForOverlapping;
      int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;
      useCoop = (data.mNumTokens <= maxTokensCoop);
    }

    if (useCoop) {
      mutableData.mPdlOverlapWithNext = pdl;
      launchInitExpertCounts(mutableData, numThreadsHist, stream);
      mutableData.mPdlOverlapWithNext = false;
      launchCoopKernel(mutableData, numBlocksCoop, numThreadsHist, stream);
    } else {
      uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
      uint32_t const histogramEltsPerBlock = 8 * numThreadsHist;
      uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;

      int const numBlocksHistogram = std::min(
          (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
      int const numBlocksOffsets =
          std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

      mutableData.mPdlOverlapWithNext = pdl;
      launchHistogramKernel(mutableData, numBlocksHistogram, numThreadsHist, stream);
      mutableData.mPdlOverlapWithNext = false;
      launchOffsetsKernel(mutableData, numBlocksOffsets, numThreadsHist, stream);
    }
  }
}

}  // namespace routingCustom
}  // namespace moe::dev::routing

#endif  // USE_FLASHINFER
