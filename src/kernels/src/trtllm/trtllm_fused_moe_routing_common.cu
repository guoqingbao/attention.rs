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

// Shared post-topK pipeline for all routing methods.
// When topK is already computed (mPtrTopKIds or mPtrTopKPacked), all routing
// methods use the same workflow: single-block, single-cluster, coop, or
// multi-kernel histogram+offsets.

#if defined(USE_FLASHINFER) && defined(USE_TRTLLM)

#include "flashinfer/trtllm/common/cudaUtils.h"
#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"

namespace moe::dev::routing {

namespace routingCustom {
void launchBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream);
void launchClusterKernel(Data const& data, void* stream);
void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist, void* stream);
void launchInitExpertCounts(Data const& data, uint32_t numThreadsHist, void* stream);
void launchHistogramKernel(Data const& data, int numBlocksHistogram, uint32_t numThreadsHist,
                           void* stream);
void launchOffsetsKernel(Data const& data, int numBlocksOffsets, uint32_t numThreadsHist,
                         void* stream);
}  // namespace routingCustom

template <typename DataType>
void runPostTopKPipeline(DataType const& data, uint32_t /*numThreadsHist*/, void* stream) {
  routingCustom::Data customData;
  static_cast<DataBase&>(customData) = static_cast<DataBase const&>(data);
  customData.mDtypeOutput = data.mDtypeOutput;
  customData.mDtypeInput = data.mDtypeOutput;
  customData.mPreprocessType = RoutingPreprocessType::None;
  customData.mPostprocessType = RoutingPostprocessType::Softmax;

  uint32_t const numThreadsHist =
      std::min(1024u, static_cast<uint32_t>(routingCustom::getMaxNumExperts(data.mNumExperts)));

  static int const smMajor = tensorrt_llm::common::getSMVersion() / 10;
  bool const useSingleBlock = data.mNumTokens <= routingCustom::BlockKernelMaxNumTokens;
  bool const useSingleCluster =
      (smMajor >= 9) && (data.mNumTokens <= routingCustom::MaxNumTokensSingleCluster);

  routingCustom::Data lastKernelData = customData;
  lastKernelData.mPdlOverlapWithNext = false;

  if (useSingleBlock) {
    routingCustom::launchBlockKernel(lastKernelData, numThreadsHist, stream);
  } else if (useSingleCluster) {
    routingCustom::launchClusterKernel(lastKernelData, stream);
  } else {
    bool const canUseCoop = (smMajor >= 9) && (data.mNumExperts <= 1024) &&
                            (data.mPtrPermutedIdxSize != nullptr) &&
                            (data.mPtrExpertCounts != nullptr);
    bool useCoop = false;
    int numBlocksCoop = 0;

    if (canUseCoop) {
      static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
      numBlocksCoop = smCount - kReservedSMsForOverlapping;
      int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;
      useCoop = (data.mNumTokens <= maxTokensCoop);
    }

    if (useCoop) {
      routingCustom::launchInitExpertCounts(customData, numThreadsHist, stream);
      routingCustom::launchCoopKernel(lastKernelData, numBlocksCoop, numThreadsHist, stream);
    } else {
      FLASHINFER_CHECK(data.mPtrExpertCounts != nullptr,
                       "When #tokens is large, `mPtrExpertCounts` is a required input.");

      routingCustom::launchInitExpertCounts(customData, numThreadsHist, stream);

      int32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
      int32_t const histogramEltsPerBlock = 8 * numThreadsHist;
      int32_t const offsetEltsPerBlock = routing::NumEltsPerOffsetTilePerThread * numThreadsHist;
      int32_t const maxNumBlocks = 1024;

      int const numBlocksHistogram = std::min(
          (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
      int const numBlocksOffsets =
          std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

      routingCustom::launchHistogramKernel(customData, numBlocksHistogram, numThreadsHist, stream);
      routingCustom::launchOffsetsKernel(lastKernelData, numBlocksOffsets, numThreadsHist, stream);
    }
  }
}

template void runPostTopKPipeline<routingCustom::Data>(routingCustom::Data const&, uint32_t, void*);
template void runPostTopKPipeline<routingDeepSeek::Data>(routingDeepSeek::Data const&, uint32_t,
                                                         void*);
template void runPostTopKPipeline<routingLlama4::Data>(routingLlama4::Data const&, uint32_t, void*);

}  // namespace moe::dev::routing

#endif  // USE_FLASHINFER
