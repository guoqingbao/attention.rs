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

// routingDeepSeek: updated for FlashInfer v0.6.7 API
// Uses LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT from RoutingDevKernel.h

#if defined(USE_FLASHINFER) && defined(USE_TRTLLM)

#include <algorithm>
#include <cmath>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.cuh"

namespace moe::dev::routing {

namespace routingCustom {
void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist, void* stream);
}  // namespace routingCustom

namespace routingDeepSeek {

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumNemotronExperts = 512;
static constexpr int NumKimiK2Experts = 384;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxSupportedExpertCount =
    std::max({NumNemotronExperts, NumKimiK2Experts, NumDeepseekExperts});
static constexpr int NumTopGroupScores = 2;
static constexpr int MaxNumTopGroups = 4;
static constexpr int MaxNumGroups = 8;

static constexpr int NumTop8Experts = 8;
static constexpr int NumTop22Experts = 22;
static constexpr int MaxSupportedTopExperts = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getMaxNumExperts(int32_t numExperts) {
  if (numExperts <= topk::MaxNumExpertsUnit) {
    return topk::MaxNumExpertsUnit;
  } else if (numExperts <= NumDeepseekExperts) {
    return NumDeepseekExperts;
  } else if (numExperts <= NumKimiK2Experts) {
    return NumKimiK2Experts;
  } else if (numExperts <= NumNemotronExperts) {
    return NumNemotronExperts;
  } else {
    FLASHINFER_WARN("Unsupported numExperts");
    return 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, \
                                  stream, extraFlag1, forceFloatInput, numExperts)           \
  if (data.mTopK <= NumTop8Experts) {                                                        \
    LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(                                       \
        data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
        forceFloatInput, numExperts, NumTop8Experts);                                        \
  } else if (data.mTopK <= NumTop22Experts) {                                                \
    LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(                                       \
        data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
        forceFloatInput, numExperts, NumTop22Experts);                                       \
  } else {                                                                                   \
    LAUNCH_ROUTING_WITH_NUM_EXPERTS_FORCE_FLOAT_INPUT(                                       \
        data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, extraFlag1,       \
        forceFloatInput, numExperts, MaxSupportedTopExperts);                                \
  }

#define LAUNCH_ROUTING_DEEPSEEK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream, \
                                extraFlag1, forceFloatInput)                                       \
  if (data.mNumExperts <= topk::MaxNumExpertsUnit) {                                               \
    LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,   \
                              extraFlag1, forceFloatInput, topk::MaxNumExpertsUnit);               \
  } else if (data.mNumExperts <= NumDeepseekExperts) {                                             \
    LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,   \
                              extraFlag1, forceFloatInput, NumDeepseekExperts);                    \
  } else if (data.mNumExperts <= NumKimiK2Experts) {                                               \
    LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,   \
                              extraFlag1, forceFloatInput, NumKimiK2Experts);                      \
  } else if (data.mNumExperts <= NumNemotronExperts) {                                             \
    LAUNCH_DEEPSEEK_WITH_TOPK(data, coopLaunch, kernel, numBlocks, numThreads, smemSize, stream,   \
                              extraFlag1, forceFloatInput, NumNemotronExperts);                    \
  } else {                                                                                         \
    FLASHINFER_WARN("Unsupported numExperts");                                                     \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////
// 1. Main kernel — DeepSeek-specific routing with sigmoid, bias, and group TopK.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void routingMainKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;

  __shared__ float __attribute((aligned(128))) smemScoreSigmoid[KernelParams::MaxNumExperts];
  __shared__ float __attribute((aligned(128))) smemScoreBias[KernelParams::MaxNumExperts];
  __shared__ float __attribute((aligned(128))) smemGroupScores[MaxNumGroups];

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);
  int32_t laneIdx = threadIdx.x % WarpSize;
  int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  static constexpr float invalidScoreFloat = float{-INFINITY};
  const OutputT invalidScore = OutputT{invalidScoreFloat};

  auto threadExpert = threadIdx.x;
  bool expertSelected = threadExpert < params.mNumExperts;
  if constexpr (KernelParams::UseGroups) {
    threadExpert = warpIdx * params.mNumExpertsPerGroup + laneIdx;
    expertSelected = (warpIdx < params.mNumExpertGroups) && (laneIdx < params.mNumExpertsPerGroup);
  }
  auto scoreIdx = int64_t{blockIdx.x} * int64_t{params.mNumExperts} + threadExpert;
  auto biasVal = (expertSelected && params.mPtrRoutingBias != nullptr)
                     ? static_cast<OutputT>(
                           loadScalar(params.mPtrRoutingBias, threadExpert, params.mDtypeBias))
                     : (expertSelected ? OutputT{0} : invalidScore);

  if (params.mPtrExpertCounts) {
    int32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t globalThreadStride = gridDim.x * blockDim.x;
    int32_t expertCountsNum = 2 * params.mNumExperts;
    initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  if (params.mPtrScores != nullptr) {
    float score =
        expertSelected ? static_cast<float>(params.mPtrScores[scoreIdx]) : invalidScoreFloat;
    auto scoreSigmoid = sigmoid_accurate(score);
    if (expertSelected) {
      smemScoreSigmoid[threadExpert] = scoreSigmoid;
    }
    auto scoreBias = float{scoreSigmoid + float{biasVal}};

    if (expertSelected) {
      smemScoreBias[threadExpert] = scoreBias;
    }

    float topExpGroupScores[NumTopGroupScores];
    [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
    float topGroups[MaxNumTopGroups];
    int32_t topGroupIdx[MaxNumTopGroups];
    float expertScoreGroup[MaxNumTopGroups];
    int32_t expertIdxGroup[MaxNumTopGroups];
    float topScores[KernelParams::MaxNumTopExperts];
    int32_t topExperts[KernelParams::MaxNumTopExperts];

    if constexpr (KernelParams::UseGroups) {
      topk::reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
                       invalidScoreFloat);
      if (cute::elect_one_sync()) {
        auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
        smemGroupScores[warpIdx] = groupScore;
      }
    }

    __syncthreads();

    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    if constexpr (KernelParams::UseGroups) {
      if (warpIdx == 0) {
        float groupScore =
            laneIdx < params.mNumExpertGroups ? smemGroupScores[laneIdx] : invalidScoreFloat;
        topk::reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx, invalidScoreFloat);

#pragma unroll
        for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
          auto groupIdx = topGroupIdx[ii];
          expertIdxGroup[ii] = groupIdx * params.mNumExpertsPerGroup + laneIdx;
          expertScoreGroup[ii] = (ii < params.mNumLimitedGroups) &&
                                         (groupIdx < params.mNumExpertGroups) && expertSelected
                                     ? smemScoreBias[expertIdxGroup[ii]]
                                     : invalidScoreFloat;
        }

        topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                         invalidScoreFloat, params.mTopK);
      }
    } else if constexpr (KernelParams::MaxNumExperts > topk::MaxNumExpertsUnit) {
      int constexpr NumExpertWarps =
          (KernelParams::MaxNumExperts - 1) / topk::MaxNumExpertsUnit + 1;
      int constexpr NumInterTopK = NumExpertWarps * KernelParams::MaxNumTopExperts;
      __shared__ float __attribute((aligned(128))) smemInterTopScores[NumInterTopK];
      __shared__ int32_t __attribute((aligned(128))) smemInterTopExperts[NumInterTopK];
      if (warpIdx < NumExpertWarps) {
        int offset = warpIdx * WarpSize * MaxNumTopGroups;
#pragma unroll
        for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
          auto expertIdx = ii * WarpSize + laneIdx;
          expertIdxGroup[ii] = offset + expertIdx;
          expertScoreGroup[ii] = offset + expertIdx < params.mNumExperts
                                     ? smemScoreBias[offset + expertIdx]
                                     : invalidScoreFloat;
        }
        topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                         invalidScoreFloat, params.mTopK);

        if (laneIdx < params.mTopK) {
          smemInterTopScores[warpIdx * KernelParams::MaxNumTopExperts + laneIdx] =
              topScores[laneIdx];
          smemInterTopExperts[warpIdx * KernelParams::MaxNumTopExperts + laneIdx] =
              topExperts[laneIdx];
        } else if (laneIdx >= params.mTopK && laneIdx < KernelParams::MaxNumTopExperts) {
          smemInterTopScores[warpIdx * KernelParams::MaxNumTopExperts + laneIdx] =
              invalidScoreFloat;
          smemInterTopExperts[warpIdx * KernelParams::MaxNumTopExperts + laneIdx] =
              MaxSupportedExpertCount - 1;
        }
      }
      __syncthreads();
      if (warpIdx == 0) {
        int constexpr NumInterTopKPerThread = (NumInterTopK - 1) / WarpSize + 1;
        float intermediateScore[NumInterTopKPerThread];
        int32_t intermediateExpert[NumInterTopKPerThread];
        for (int i = laneIdx; i < NumInterTopKPerThread * WarpSize; i += WarpSize) {
          int ii = i / WarpSize;
          if (i < NumInterTopK) {
            intermediateScore[ii] = smemInterTopScores[i];
            intermediateExpert[ii] = smemInterTopExperts[i];
          } else {
            intermediateScore[ii] = invalidScoreFloat;
            intermediateExpert[ii] = KernelParams::MaxNumExperts - 1;
          }
        }
        topk::reduceTopK(warp, topScores, topExperts, intermediateScore, intermediateExpert,
                         invalidScoreFloat, params.mTopK);
      }
    } else {
      if (warpIdx == 0) {
#pragma unroll
        for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
          auto expertIdx = ii * WarpSize + laneIdx;
          expertIdxGroup[ii] = expertIdx;
          expertScoreGroup[ii] =
              expertIdx < params.mNumExperts ? smemScoreBias[expertIdx] : invalidScoreFloat;
        }
        topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                         invalidScoreFloat, params.mTopK);
      }
    }

    if (warpIdx == 0) {
      int32_t expertIdx = 0;
#pragma unroll
      for (int ii = 0; ii < params.mTopK; ++ii) {
        expertIdx = laneIdx == ii ? topExperts[ii] : expertIdx;
      }
      auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
      auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                           (localExpertIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

      float scoreNorm = laneIdx < params.mTopK ? smemScoreSigmoid[expertIdx] : 0.F;
      auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
      auto finalScore = OutputT{scoreNorm * params.mRouteScale / redNorm};

      auto idxTopK = blockIdx.x * params.mTopK + laneIdx;
      if (laneIdx < params.mTopK && params.mPtrTopKPacked != nullptr) {
        PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(finalScore),
                                            static_cast<int16_t>(expertIdx)};
        params.mPtrTopKPacked[idxTopK] = packedScore;
      }

      if (laneIdx < params.mTopK && params.mPtrTopKWeights != nullptr &&
          params.mPtrTopKIds == nullptr) {
        params.mPtrTopKWeights[idxTopK] = finalScore;
      }
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (params.mUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void launchMainKernel(Data& data, int numBlocks, int numThreadsMain, void* stream) {
  bool const forceFloatInput = (data.mDtypeInput == tg::Dtype::Fp32);
  LAUNCH_ROUTING_DEEPSEEK(data, false, routingMainKernel, numBlocks, numThreadsMain,
                          0, stream, data.mNumExpertGroups > 1, forceFloatInput);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2. Cluster kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1)
    __launch_bounds__(KernelParams::MaxNumExperts)
        routingIndicesClusterKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const clusterBlockRank = blockIdx.x;

  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
  routingPermutation<KernelParams, OutputT, KernelParams::MaxNumExperts,
                     KernelParams::MaxNumExperts / WarpSize, KernelParams::MaxNumTopExperts,
                     /*LoadExpertIdxFromGlobal=*/true>(params, nullptr, warpIdx, clusterBlockRank);
}
#else
__global__ void routingIndicesClusterKernel(KernelParams params) {
  assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

static void launchClusterKernel(Data& data, int numThreadsHist, void* stream) {
  LAUNCH_ROUTING_DEEPSEEK(data, false, routingIndicesClusterKernel, NumBlocksPerCluster,
                          numThreadsHist, 0, stream, data.mNumExpertGroups > 1, true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 3-6. Launch wrappers for shared kernels
////////////////////////////////////////////////////////////////////////////////////////////////////

static void launchCoopKernel(Data& data, int numBlocksCoop, int /*numThreadsHist*/, void* stream) {
  routingCustom::Data customData;
  static_cast<DataBase&>(customData) = static_cast<DataBase const&>(data);
  customData.mDtypeOutput = data.mDtypeOutput;
  customData.mDtypeInput = data.mDtypeOutput;
  customData.mPreprocessType = RoutingPreprocessType::None;
  customData.mPostprocessType = RoutingPostprocessType::Softmax;

  uint32_t const customNumThreadsHist =
      std::min(1024u, static_cast<uint32_t>(routingCustom::getMaxNumExperts(data.mNumExperts)));
  routingCustom::launchCoopKernel(customData, numBlocksCoop, customNumThreadsHist, stream);
}

static void launchHistogramKernel(Data& data, int numBlocksHistogram, int numThreadsHist,
                                  void* stream) {
  LAUNCH_ROUTING_DEEPSEEK(data, false, routingIndicesHistogramKernel, numBlocksHistogram,
                          numThreadsHist, 0, stream, data.mNumExpertGroups > 1, true);
}

static void launchOffsetsKernel(Data& data, int numBlocksOffsets, int numThreadsHist,
                                void* stream) {
  LAUNCH_ROUTING_DEEPSEEK(data, false, routingIndicesOffsetsKernel, numBlocksOffsets,
                          numThreadsHist, 0, stream, data.mNumExpertGroups > 1, true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data& data, void* stream) {
  FLASHINFER_CHECK(
      data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr,
      "Routing kernel requires at least one input parameter");
  FLASHINFER_CHECK(data.mTopK > 0, "mTopK must be positive");

  if (data.mPtrTopKIds != nullptr ||
      (data.mPtrTopKPacked != nullptr && data.mPtrScores == nullptr)) {
    if (data.mPtrTopKIds != nullptr) {
      FLASHINFER_CHECK(data.mPtrTopKWeights != nullptr,
                       "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided.");
    }
    int const numThreadsHist = routingCustom::getMaxNumExperts(data.mNumExperts);
    runPostTopKPipeline(data, numThreadsHist, stream);
    return;
  }

  FLASHINFER_CHECK(!data.mUseRoutingSoftmax, "Routing with softmax not implemented yet");
  FLASHINFER_CHECK(data.mNumExperts >= data.mTopK,
                   "Routing kernel expects topK <= numExperts");
  FLASHINFER_CHECK(data.mNumExperts <= MaxSupportedExpertCount,
                   "Routing kernel expects #experts <= ", MaxSupportedExpertCount);
  FLASHINFER_CHECK(data.mTopK <= MaxSupportedTopExperts,
                   "Routing kernel expects topK <= ", MaxSupportedTopExperts);

  if (data.mPtrExpandedIdxToPermutedIdx != nullptr ||
      data.mPtrPermutedIdxToExpandedIdx != nullptr || data.mPtrPermutedIdxToTokenIdx != nullptr)
    FLASHINFER_CHECK(data.mPtrTopKPacked != nullptr && data.mPtrPermutedIdxSize,
                     "If permuted index is required, `mPtrTopKPacked` is also required");

  if (data.mNumExpertGroups > 1) {
    FLASHINFER_CHECK(data.mNumExpertGroups <= MaxNumGroups,
                     "Routing kernel expects #expert groups <= ", MaxNumGroups);
    FLASHINFER_CHECK(data.mNumExperts % data.mNumExpertGroups == 0,
                     "Routing kernel expects #experts to be a multiple of #expert groups");
    FLASHINFER_CHECK(data.mNumExperts / data.mNumExpertGroups <= WarpSize,
                     "Routing kernel expects #experts per group <= warp size");
    FLASHINFER_CHECK(data.mNumLimitedGroups <= MaxNumTopGroups,
                     "Routing kernel expects <= ", MaxNumTopGroups, " top groups");
    FLASHINFER_CHECK(data.mNumExpertGroups >= data.mNumLimitedGroups,
                     "Routing kernel expects top groups <= #expert groups");
    FLASHINFER_CHECK(data.mNumExperts % 4 == 0,
                     "Routing kernel expects #experts to be a multiple of 4");
  }

  int const numBlocks = data.mNumTokens;
  int const numThreadsHist = getMaxNumExperts(data.mNumExperts);
  bool const pdl = data.mUsePdl;

  int const numThreadsMain =
      std::max(data.mNumExpertGroups * WarpSize, getMaxNumExperts(data.mNumExperts));
  data.mPdlOverlapWithNext = pdl;
  launchMainKernel(data, numBlocks, numThreadsMain, stream);

  if (data.mPtrPermutedIdxSize != nullptr) {
    bool const useSingleCluster = data.mNumTokens <= 1024;
    if (!useSingleCluster) {
      FLASHINFER_CHECK(data.mPtrExpertCounts != nullptr,
                       "When #tokens is large, `mPtrExpertCounts` is a required input.");
    } else {
      data.mPtrExpertCounts = nullptr;
    }

    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int const numBlocksCoop = smCount - 8;
    int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;

    if (useSingleCluster) {
      data.mPdlOverlapWithNext = false;
      launchClusterKernel(data, numThreadsHist, stream);
    } else if (data.mNumTokens <= maxTokensCoop) {
      data.mPdlOverlapWithNext = false;
      launchCoopKernel(data, numBlocksCoop, numThreadsHist, stream);
    } else {
      const int32_t expandedIdxSize = data.mNumTokens * data.mTopK;
      const int32_t histogramEltsPerBlock = 8 * numThreadsHist;
      const int32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;
      const int32_t maxNumBlocks = 1024;

      int const numBlocksHistogram = std::min(
          (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
      int const numBlocksOffsets =
          std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

      data.mPdlOverlapWithNext = pdl;
      launchHistogramKernel(data, numBlocksHistogram, numThreadsHist, stream);
      data.mPdlOverlapWithNext = false;
      launchOffsetsKernel(data, numBlocksOffsets, numThreadsHist, stream);
    }
  }
}

#undef LAUNCH_DEEPSEEK_WITH_TOPK
#undef LAUNCH_ROUTING_DEEPSEEK

}  // namespace routingDeepSeek
}  // namespace moe::dev::routing

#endif  // USE_FLASHINFER
