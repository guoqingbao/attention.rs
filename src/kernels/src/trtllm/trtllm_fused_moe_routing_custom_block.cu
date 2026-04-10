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

// routingCustom block kernel — single-block fused kernel for ≤4 tokens.
// Split out for parallel compilation (LAUNCH_ROUTING_CUSTOM causes heavy template explosion).

#if defined(USE_FLASHINFER) && defined(USE_TRTLLM)

#include "flashinfer/trtllm/common/cudaUtils.h"
#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"

namespace moe::dev::routing {
namespace routingCustom {

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts
                                                                      : 1024)
    routingIndicesBlockKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  using TypePacked = PackedScoreIdx<BaseType>;
  static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
  static constexpr int NumThreadsBlock = MaxNumExperts <= 1024 ? MaxNumExperts : 1024;
  static constexpr int ExpertsPerThread = MaxNumExperts / NumThreadsBlock;
  static_assert(MaxNumExperts % NumThreadsBlock == 0,
                "MaxNumExperts must be a multiple of NumThreadsBlock");

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();
  auto scoreOffset = warpIdx * params.mNumExperts;
  bool validToken = warpIdx < params.mNumTokens;

  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;
  static constexpr int totalExpertCounts = BlockKernelMaxNumTokens * MaxNumExperts;
  __shared__ int8_t __attribute((aligned(128))) smemOffset[totalExpertCounts];
  __shared__ int8_t __attribute((aligned(128))) smemKIdx[totalExpertCounts];

  using Scan = cub::BlockScan<int32_t, NumThreadsBlock>;
  __shared__ typename Scan::TempStorage tempStorage;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  for (int i = threadIdx.x; i < totalExpertCounts; i += blockDim.x) {
    smemOffset[i] = int8_t{-1};
    smemKIdx[i] = int8_t{-1};
  }
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  if (params.mPtrTopKIds != nullptr) {
    if (validToken) {
      if (laneIdx < params.mTopK) {
        auto expertIdx = params.mPtrTopKIds[warpIdx * params.mTopK + laneIdx];
        if (expertIdx > -1 && expertIdx < params.mNumExperts) {
          int offset = warpIdx * MaxNumExperts + expertIdx;
          smemKIdx[offset] = static_cast<int8_t>(laneIdx);
        } else if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[warpIdx * params.mTopK + laneIdx] = int32_t{-1};
        }
      }
    }
  } else if (params.mPtrScores != nullptr) {
    BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
    int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];

    if (validToken) {
      KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize,
                                                       KernelParams::MaxNumTopExperts>(
          warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
          params.mPtrScores + scoreOffset, params);

      if (laneIdx < params.mTopK) {
        int offset = warpIdx * MaxNumExperts + warpTopKExpertIdx[laneIdx];
        smemKIdx[offset] = static_cast<int8_t>(laneIdx);
        if (params.mPtrTopKWeights != nullptr) {
          params.mPtrTopKWeights[warpIdx * params.mTopK + laneIdx] =
              OutputT{warpTopKScore[laneIdx]};
        }
      }
    }
  } else if (params.mPtrTopKPacked != nullptr) {
    if (validToken) {
      if (laneIdx < params.mTopK) {
        auto const expandedIdx = warpIdx * params.mTopK + laneIdx;
        auto const scoreIdx = params.mPtrTopKPacked[expandedIdx];
        int const expertIdx = static_cast<int>(scoreIdx.idx);
        if (expertIdx >= 0 && expertIdx < params.mNumExperts) {
          int const offset = warpIdx * MaxNumExperts + expertIdx;
          smemKIdx[offset] = static_cast<int8_t>(laneIdx);
          if (params.mPtrTopKWeights != nullptr) {
            params.mPtrTopKWeights[expandedIdx] = static_cast<OutputT>(scoreIdx.score);
          }
        } else if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = int32_t{-1};
        }
      }
    }
  }
  __syncthreads();

  int accExpertCount[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    auto localExpIdx = expert - params.mLocalExpertsStartIdx;
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    auto isLocal = localExpIdx >= 0 && localExpIdx < localExpertExtent &&
                   (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

    accExpertCount[e] = 0;
    if (isLocal) {
      int offset = expert;
      for (int j = 0; j < BlockKernelMaxNumTokens; j++) {
        if (smemKIdx[offset] >= 0) {
          smemOffset[offset] = static_cast<int8_t>(accExpertCount[e]);
          accExpertCount[e]++;
        }
        offset += MaxNumExperts;
      }
    }
  }
  __syncthreads();

  int32_t numCtaPerExpert[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    if (params.mIsPow2) {
      numCtaPerExpert[e] = divUpLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
    } else {
      numCtaPerExpert[e] = divUpTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
    }
    numCtaPerExpert[e] *= params.mClusterSizeInBatchDim;
  }
  int32_t ctaOffsetPerExpert[ExpertsPerThread];
  int32_t numNonExitingCtas;
  Scan(tempStorage).ExclusiveSum(numCtaPerExpert, ctaOffsetPerExpert, numNonExitingCtas);
  __syncthreads();

  int32_t tmpCountPerExpert[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    if (params.mIsPow2) {
      tmpCountPerExpert[e] = divUpMulLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
    } else {
      tmpCountPerExpert[e] = divUpMulTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
    }
  }
  int32_t expertScanCountsPerExpert[ExpertsPerThread];
  Scan(tempStorage).ExclusiveSum(tmpCountPerExpert, expertScanCountsPerExpert);
  __syncthreads();

#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    auto localExpIdx = expert - params.mLocalExpertsStartIdx;
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    auto isLocal = localExpIdx >= 0 && localExpIdx < localExpertExtent &&
                   (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

    if (isLocal) {
      for (int cta = 0; cta < numCtaPerExpert[e]; ++cta) {
        int32_t const mappedLocalIdx =
            (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        params.mPtrCtaIdxXyToBatchIdx[ctaOffsetPerExpert[e] + cta] = mappedLocalIdx;
        int32_t mnLimit1;
        int32_t mnLimit2;
        if (params.mIsPow2) {
          int32_t ctaPaddingLog2 = params.mPaddingLog2 - params.mClusterSizeLog2;
          mnLimit1 = mulLog2<int32_t>(ctaOffsetPerExpert[e] + cta + 1, ctaPaddingLog2);
          mnLimit2 = mulLog2<int32_t>(ctaOffsetPerExpert[e], ctaPaddingLog2) + accExpertCount[e];
        } else {
          int32_t ctaTile = params.mTileTokensDim / params.mClusterSizeInBatchDim;
          mnLimit1 = (ctaOffsetPerExpert[e] + cta + 1) * ctaTile;
          mnLimit2 = ctaOffsetPerExpert[e] * ctaTile + accExpertCount[e];
        }
        params.mPtrCtaIdxXyToMnLimit[ctaOffsetPerExpert[e] + cta] = min(mnLimit1, mnLimit2);
      }
    }
  }

  if (threadIdx.x == 0) {
    int32_t permutedIdxSize;
    if (params.mIsPow2) {
      permutedIdxSize =
          mulLog2<int32_t>(numNonExitingCtas >> params.mClusterSizeLog2, params.mPaddingLog2);
    } else {
      permutedIdxSize = (numNonExitingCtas / params.mClusterSizeInBatchDim) * params.mTileTokensDim;
    }
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }

  for (int tokenIdx = 0; tokenIdx < params.mNumTokens; tokenIdx++) {
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      int expert = threadIdx.x * ExpertsPerThread + e;
      int offset = tokenIdx * MaxNumExperts + expert;
      if (smemKIdx[offset] >= 0) {
        auto localExpIdx = expert - params.mLocalExpertsStartIdx;
        auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
        auto isLocal = localExpIdx >= 0 && localExpIdx < localExpertExtent &&
                       (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

        int const expandedIdx = tokenIdx * params.mTopK + smemKIdx[offset];
        int const offsetWithinExpert = static_cast<int>(smemOffset[offset]);
        int const offsetForExpert = expertScanCountsPerExpert[e];
        int const permutedIdx = isLocal ? offsetForExpert + offsetWithinExpert : int32_t{-1};

        if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
        }
        if (params.mPtrPermutedIdxToExpandedIdx != nullptr && isLocal) {
          params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
        }
        if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocal) {
          params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
        }
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.mUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

void launchBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream) {
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesBlockKernel, 1, numThreadsHist,
                        /*smemSize=*/0, stream);
}

}  // namespace routingCustom
}  // namespace moe::dev::routing

#endif  // USE_FLASHINFER
