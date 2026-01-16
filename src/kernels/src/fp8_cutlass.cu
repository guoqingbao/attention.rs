#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

#include "attention/dtype_fp8.cuh"

#if defined(USE_CUTLASS)
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"
#endif

constexpr int kBlockM = 128;
constexpr int kBlockK = 128;
constexpr int kPackThreads = 256;

__device__ __forceinline__ float to_float_half(half v) {
  return __half2float(v);
}

__device__ __forceinline__ float to_float_bf16(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__global__ void quantize_fp8_blockwise_f32(const T* input,
                                           uint8_t* output,
                                           float* scales,
                                           int M_valid,
                                           int M_padded,
                                           int K) {
  const int k_blocks = (K + kBlockK - 1) / kBlockK;
  const int m = blockIdx.y;
  const int k_blk = blockIdx.x;
  const int tid = threadIdx.x;

  float max_val = 0.0f;
  for (int kk = tid; kk < kBlockK; kk += blockDim.x) {
    int k = k_blk * kBlockK + kk;
    if (m < M_padded && k < K) {
      float v = 0.0f;
      if (m < M_valid) {
        if constexpr (std::is_same<T, half>::value) {
          v = to_float_half(input[m * K + k]);
        } else {
          v = to_float_bf16(input[m * K + k]);
        }
      }
      max_val = fmaxf(max_val, fabsf(v));
    }
  }

  __shared__ float sdata[256];
  sdata[tid] = max_val;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
    }
    __syncthreads();
  }

  float block_max = sdata[0];
  float scale = block_max > 0.0f ? block_max / MAX_FP8_VALUE : 1.0f;
  if (tid == 0) {
    scales[m * k_blocks + k_blk] = scale;
    // if (m == 0 && k_blk == 0) {
    //    printf("DEBUG: quantize: m=0 k_blk=0 scale=%f (max_val=%f)\n", scale, block_max);
    // }
  }
  __syncthreads();

  for (int kk = tid; kk < kBlockK; kk += blockDim.x) {
    int k = k_blk * kBlockK + kk;
    if (m < M_padded && k < K) {
      float v = 0.0f;
      if (m < M_valid) {
        if constexpr (std::is_same<T, half>::value) {
          v = to_float_half(input[m * K + k]);
        } else {
          v = to_float_bf16(input[m * K + k]);
        }
      }
      float q = v / scale;
      output[m * K + k] = vllm::fp8::dispatch_float_to_fp8(q);
      // if (m==0 && k==0) {
      //     printf("DEBUG: quantize: m=0 k=0 val=%f q=%f out=%u\n", v, q, (unsigned int)output[m*K+k]);
      // }
    }
  }
}

extern "C" void fp8_matmul_f16_layout(const __half* input,
                                      const uint8_t* weight,
                                      const float* weight_scale,
                                      __half* output,
                                      int M,
                                      int N,
                                      int K,
                                      int scale_row_stride,
                                      int block_size_y,
                                      int block_size_x,
                                      int weight_col_major,
                                      cudaStream_t stream);

extern "C" void fp8_matmul_bf16_layout(const __nv_bfloat16* input,
                                       const uint8_t* weight,
                                       const float* weight_scale,
                                       __nv_bfloat16* output,
                                       int M,
                                       int N,
                                       int K,
                                       int scale_row_stride,
                                       int block_size_y,
                                       int block_size_x,
                                       int weight_col_major,
                                       cudaStream_t stream);

extern "C" void moe_gemm_wmma_fp8_layout(
    const void* input,
    const uint8_t* weights,
    const float* weight_scales,
    const int* sorted_token_ids,
    const int* expert_ids,
    const float* topk_weights,
    void* output,
    int* expert_counts,
    int* expert_offsets,
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int block_size_n,
    int block_size_k,
    int dtype,
    bool is_prefill,
    int weight_col_major,
    cudaStream_t stream);

#if defined(USE_CUTLASS)
using namespace cute;

template <typename GemmKernel>
cutlass::Status cutlass_gemm_caller(typename GemmKernel::Arguments const& args,
                                    cudaStream_t stream) {
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;

  auto can = gemm_op.can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    return can;
  }

  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto init_status = gemm_op.initialize(args, workspace.get(), stream);
  if (init_status != cutlass::Status::kSuccess) {
    return init_status;
  }
  return gemm_op.run(stream);
}

template <typename Layout>
__global__ void pack_sfa_layout(const float* scales_rm,
                                float* scales_packed,
                                Layout layout_sfa,
                                int m_blocks,
                                int k_blocks) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m_blocks * k_blocks;
  if (idx >= total) {
    return;
  }
  int m_blk = idx / k_blocks;
  int k_blk = idx - m_blk * k_blocks;
  // Layout expects element coordinates.
  // Scale A (Activations) has Granularity M=1 (1 scale per row).
  // m_blk is the row index. LayoutSFA expects row index directly if granularity is 1?
  // Wait, layout_sfa maps element coordinate -> scale coordinate -> offset.
  // If ScaleTileShape is <1, 128, 128>. ScaleConfig logic:
  // Tile M=1. Element M maps to Scale M = M / 1 = M.
  // So we pass M.
  // If we pass M*128, we map to scale M*128.
  // So we SHOULD pass m_blk directly.
  int packed_idx = layout_sfa(m_blk, k_blk * 128, 0);
  scales_packed[packed_idx] = scales_rm[idx];
}

template <typename SchedulerType, typename OutType, int GroupSizeM_, int GroupSizeN_, int GroupSizeK_,
          int TileSizeM_ = 128, class ClusterShape = Shape<_1, _2, _1>>
struct cutlass_3x_gemm_fp8_blockwise {
  using GroupSizeM = Int<GroupSizeM_>;
  using GroupSizeN = Int<GroupSizeN_>;
  using GroupSizeK = Int<GroupSizeK_>;
  using TileSizeM = Int<TileSizeM_>;

  using ElementAB = cutlass::float_e4m3_t;
  using ElementA = ElementAB;
  using ElementB = ElementAB;
  using ElementC = void;
  using ElementD = OutType;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentD = AlignmentC;

  using ScaleTileShape = Shape<_1, _128, _128>;
  using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(ScaleTileShape{}));
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ElementAccumulator = float;
  using ElementCompute = float;
  using TileShape = Shape<TileSizeM, GroupSizeN, GroupSizeK>;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      SchedulerType>;
};

template <typename Gemm>
cutlass::Status run_sm90_blockwise(typename Gemm::ElementD* out,
                                   const typename Gemm::ElementAB* a,
                                   const typename Gemm::ElementAB* b,
                                   float* a_scales_packed,
                                   const float* a_scales_rm,
                                   const float* b_scales,
                                   int m,
                                   int n,
                                   int k,
                                   cudaStream_t stream) {
  using GemmKernel = typename Gemm::GemmKernel;
  using ElementD = typename Gemm::ElementD;
  using ElementBlockScale = float;

  using ScaleTileShape = Shape<_1, _128, _128>;
  using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(ScaleTileShape{}));
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  using StrideC = typename GemmKernel::StrideC;

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  {
      int k_blocks = (k + 127) / 128;
      int m_blocks = m;
      int total = m_blocks * k_blocks;
      int threads = 256;
      int blocks = (total + threads - 1) / threads;
      pack_sfa_layout<<<blocks, threads, 0, stream>>>(a_scales_rm, a_scales_packed, layout_sfa, m_blocks, k_blocks);
  }

  typename GemmKernel::MainloopArguments mainloop_args{
      a, a_stride, b, b_stride,
      reinterpret_cast<ElementBlockScale const*>(a_scales_packed), layout_sfa,
      reinterpret_cast<ElementBlockScale const*>(b_scales), layout_sfb};

  auto c_ptr = reinterpret_cast<ElementD*>(out);
  typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::TileSchedulerArguments scheduler{};
  static constexpr bool UsesStreamK =
      cute::is_same_v<typename GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>;
  if constexpr (UsesStreamK) {
    using Params = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams;
    scheduler.decomposition_mode = Params::DecompositionMode::StreamK;
    scheduler.reduction_mode = Params::ReductionMode::Nondeterministic;
  }

  cutlass::KernelHardwareInfo hw_info;
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, device_id);
  hw_info.device_id = device_id;
  hw_info.sm_count = props.multiProcessorCount;

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      mainloop_args,
      epilogue_args,
      hw_info,
      scheduler};

  return cutlass_gemm_caller<GemmKernel>(args, stream);
}

template <typename OutType>
cutlass::Status fp8_blockwise_sm90_dispatch(OutType* out,
                                            const cutlass::float_e4m3_t* a,
                                            const cutlass::float_e4m3_t* b,
                                            float* a_scales_packed,
                                            const float* a_scales_rm,
                                            const float* b_scales,
                                            int m,
                                            int n,
                                            int k,
                                            cudaStream_t stream) {
  if (k > 3 * n) {
    using Gemm = cutlass_3x_gemm_fp8_blockwise<
        cutlass::gemm::StreamKScheduler, OutType, 1, 128, 128>;
    return run_sm90_blockwise<Gemm>(out, a, b, a_scales_packed, a_scales_rm, b_scales, m, n, k, stream);
  }
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      cutlass::gemm::PersistentScheduler, OutType, 1, 128, 128>;
  return run_sm90_blockwise<Gemm>(out, a, b, a_scales_packed, a_scales_rm, b_scales, m, n, k, stream);
}

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
template <typename OutType, typename MmaTileShape, typename PerSmTileShape,
          typename EpilogueTileShape, typename ScalesPerTile,
          int TileSizeM_ = 128, class ClusterShape = Shape<_1, _1, _1>>
cutlass::Status run_sm100_blockwise(OutType* out,
                                    const cutlass::float_e4m3_t* a,
                                    const cutlass::float_e4m3_t* b,
                                    float* a_scales_packed,
                                    const float* a_scales_rm,
                                    const float* b_scales,
                                    int m,
                                    int n,
                                    int k,
                                    cudaStream_t stream) {
  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});

  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = void;
  using ElementD = OutType;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
      cute::UMMA::Major::MN, cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, PerSmTileShape, ClusterShape,
      EpilogueTileShape, ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized1Sm>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
      ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  using StrideC = typename GemmKernel::StrideD;

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  // PACK SCALES for SM100
  {
      int k_blocks = (k + 127) / 128;
      int m_blocks = m;
      int total = m_blocks * k_blocks;
      int threads = 256;
      int blocks = (total + threads - 1) / threads;
      pack_sfa_layout<<<blocks, threads, 0, stream>>>(a_scales_rm, a_scales_packed, layout_SFA, m_blocks, k_blocks);
  }

  typename GemmKernel::MainloopArguments mainloop_args{
      a, a_stride, b, b_stride, reinterpret_cast<float const*>(a_scales_packed), layout_SFA, b_scales, layout_SFB};
  typename GemmKernel::EpilogueArguments epilogue_args{{}, out, c_stride, out, c_stride};
  epilogue_args.thread.alpha = 1.0f;

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
      mainloop_args, epilogue_args};

  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  auto init_status = gemm_op.initialize(args, workspace.get(), stream);
  if (init_status != cutlass::Status::kSuccess) {
    return init_status;
  }
  return gemm_op.run(stream);
}

template <typename OutType>
cutlass::Status fp8_blockwise_sm100_dispatch(OutType* out,
                                             const cutlass::float_e4m3_t* a,
                                             const cutlass::float_e4m3_t* b,
                                             float* a_scales_packed,
                                             const float* a_scales_rm,
                                             const float* b_scales,
                                             int m,
                                             int n,
                                             int k,
                                             cudaStream_t stream) {
  if (m <= 128) {
    using MmaTileShape = Shape<_64, _128, _128>;
    using PerSmTileShape = Shape<_64, _128, _128>;
    using EpilogueTileShape = Shape<_64, _64>;
    using ScalesPerTile = Shape<_64, _1, _1>;
    return run_sm100_blockwise<OutType, MmaTileShape, PerSmTileShape, EpilogueTileShape, ScalesPerTile>(
        out, a, b, a_scales_packed, a_scales_rm, b_scales, m, n, k, stream);
  }
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape = Shape<_128, _128, _128>;
  using EpilogueTileShape = Shape<_128, _64>;
  using ScalesPerTile = Shape<_128, _1, _1>;
  return run_sm100_blockwise<OutType, MmaTileShape, PerSmTileShape, EpilogueTileShape, ScalesPerTile>(
      out, a, b, a_scales_packed, a_scales_rm, b_scales, m, n, k, stream);
}
#endif

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
template <typename OutType, typename MmaTileShape, typename PerSmTileShape,
          typename EpilogueTileShape, typename ScalesPerTile,
          int TileSizeM_ = 128, class ClusterShape = Shape<_1, _1, _1>>
cutlass::Status run_sm120_blockwise(OutType* out,
                                    const cutlass::float_e4m3_t* a,
                                    const cutlass::float_e4m3_t* b,
                                    float* a_scales_packed,
                                    const float* a_scales_rm,
                                    const float* b_scales,
                                    int m,
                                    int n,
                                    int k,
                                    cudaStream_t stream) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = void;
  using ElementD = OutType;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});

  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
      cute::UMMA::Major::MN, cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, PerSmTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
      ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
      ElementAccumulator,
      MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  Gemm gemm_op;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  using StrideC = typename GemmKernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  // PACK SCALES for SM120
  {
      int k_blocks = (k + 127) / 128;
      int m_blocks = m;
      int total = m_blocks * k_blocks;
      int threads = 256;
      int blocks = (total + threads - 1) / threads;
      pack_sfa_layout<<<blocks, threads, 0, stream>>>(a_scales_rm, a_scales_packed, layout_SFA, m_blocks, k_blocks);
  }

  typename GemmKernel::MainloopArguments mainloop_args{
      a, stride_a, b, stride_b, reinterpret_cast<float const*>(a_scales_packed), layout_SFA, b_scales, layout_SFB};
  typename GemmKernel::EpilogueArguments epilogue_args{{}, out, stride_c, out, stride_c};
  epilogue_args.thread.alpha = 1.0f;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
      mainloop_args, epilogue_args};

  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  auto init_status = gemm_op.initialize(args, workspace.get(), stream);
  if (init_status != cutlass::Status::kSuccess) {
    return init_status;
  }
  return gemm_op.run(stream);
}

template <typename OutType>
cutlass::Status fp8_blockwise_sm120_dispatch(OutType* out,
                                             const cutlass::float_e4m3_t* a,
                                             const cutlass::float_e4m3_t* b,
                                             float* a_scales_packed,
                                             const float* a_scales_rm,
                                             const float* b_scales,
                                             int m,
                                             int n,
                                             int k,
                                             cudaStream_t stream) {
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape = Shape<_128, _128, _128>;
  using EpilogueTileShape = Shape<_128, _64>;
  using ScalesPerTile = Shape<_128, _1, _1>;
  return run_sm120_blockwise<OutType, MmaTileShape, PerSmTileShape, EpilogueTileShape, ScalesPerTile>(
      out, a, b, a_scales_packed, a_scales_rm, b_scales, m, n, k, stream);
}
#endif

template <typename OutType>
cutlass::Status fp8_blockwise_dispatch(OutType* out,
                                       const cutlass::float_e4m3_t* a,
                                       const cutlass::float_e4m3_t* b,
                                       float* a_scales_packed,
                                       const float* a_scales_rm,
                                       const float* b_scales,
                                       int m,
                                       int n,
                                       int k,
                                       cudaStream_t stream) {
  cudaDeviceProp props{};
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&props, device_id);

  if (props.major == 9 && props.minor == 0) {
    return fp8_blockwise_sm90_dispatch(out, a, b, a_scales_packed, a_scales_rm, b_scales, m, n, k, stream);
  }
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  if (props.major == 10) {
    return fp8_blockwise_sm100_dispatch(out, a, b, a_scales_packed, a_scales_rm, b_scales, m, n, k, stream);
  }
#endif
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  if (props.major == 12) {
    return fp8_blockwise_sm120_dispatch(out, a, b, a_scales_packed, a_scales_rm, b_scales, m, n, k, stream);
  }
#endif
  return cutlass::Status::kErrorNotSupported;
}
#endif

extern "C" void fp8_matmul_f16_cutlass(const __half* input,
                                       const uint8_t* weight,
                                       const float* weight_scale,
                                       __half* output,
                                       int M,
                                       int N,
                                       int K,
                                       int scale_row_stride,
                                       int block_size_y,
                                       int block_size_x,
                                       int weight_col_major,
                                       cudaStream_t stream) {
#if defined(USE_CUTLASS)
  if (!weight_col_major || block_size_y != kBlockM || block_size_x != kBlockK) {
    fp8_matmul_f16_layout(input, weight, weight_scale, output, M, N, K,
                          scale_row_stride, block_size_y, block_size_x,
                          weight_col_major, stream);
    return;
  }

  const int M_padded = (M + kBlockM - 1) / kBlockM * kBlockM;
  const int k_blocks = (K + kBlockK - 1) / kBlockK;
  const int a_scale_elems = M_padded * k_blocks;

  uint8_t* a_fp8 = nullptr;
  float* a_scales_rm = nullptr;
  float* a_scales_packed = nullptr;
  cudaMallocAsync(&a_fp8, sizeof(uint8_t) * M_padded * K, stream);
  cudaMallocAsync(&a_scales_rm, sizeof(float) * a_scale_elems, stream);
  cudaMallocAsync(&a_scales_packed, sizeof(float) * a_scale_elems, stream);

  dim3 grid(k_blocks, M_padded);
  quantize_fp8_blockwise_f32<<<grid, kPackThreads, 0, stream>>>(
      input, a_fp8, a_scales_rm, M, M_padded, K);

  auto status = cutlass::Status::kErrorNotSupported;
  cudaDeviceProp props{};
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&props, device_id);

  if (props.major == 9 && props.minor == 0) {
    cutlass::half_t* out_ptr = reinterpret_cast<cutlass::half_t*>(output);
    cutlass::half_t* out_tmp = out_ptr;
    if (M_padded != M) {
      cudaMallocAsync(&out_tmp, sizeof(cutlass::half_t) * M_padded * N, stream);
    }
    status = fp8_blockwise_sm90_dispatch(
        out_tmp,
        reinterpret_cast<const cutlass::float_e4m3_t*>(a_fp8),
        reinterpret_cast<const cutlass::float_e4m3_t*>(weight),
        a_scales_packed,
        a_scales_rm,
        weight_scale,
        M_padded, N, K, stream);
    if (status == cutlass::Status::kSuccess && M_padded != M) {
      cudaMemcpyAsync(out_ptr, out_tmp, sizeof(cutlass::half_t) * M * N,
                      cudaMemcpyDeviceToDevice, stream);
      cudaFreeAsync(out_tmp, stream);
    }
  }
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  if (props.major == 10) {
    cutlass::half_t* out_ptr = reinterpret_cast<cutlass::half_t*>(output);
    cutlass::half_t* out_tmp = out_ptr;
    if (M_padded != M) {
      cudaMallocAsync(&out_tmp, sizeof(cutlass::half_t) * M_padded * N, stream);
    }
    status = fp8_blockwise_sm100_dispatch(
        out_tmp,
        reinterpret_cast<const cutlass::float_e4m3_t*>(a_fp8),
        reinterpret_cast<const cutlass::float_e4m3_t*>(weight),
        a_scales_packed,
        a_scales_rm,
        weight_scale,
        M_padded, N, K, stream);
    if (status == cutlass::Status::kSuccess && M_padded != M) {
      cudaMemcpyAsync(out_ptr, out_tmp, sizeof(cutlass::half_t) * M * N,
                      cudaMemcpyDeviceToDevice, stream);
      cudaFreeAsync(out_tmp, stream);
    }
  }
#endif
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  if (props.major == 12) {
    cutlass::half_t* out_ptr = reinterpret_cast<cutlass::half_t*>(output);
    cutlass::half_t* out_tmp = out_ptr;
    if (M_padded != M) {
      cudaMallocAsync(&out_tmp, sizeof(cutlass::half_t) * M_padded * N, stream);
    }
    status = fp8_blockwise_sm120_dispatch(
        out_tmp,
        reinterpret_cast<const cutlass::float_e4m3_t*>(a_fp8),
        reinterpret_cast<const cutlass::float_e4m3_t*>(weight),
        a_scales_packed,
        a_scales_rm,
        weight_scale,
        M_padded, N, K, stream);
    if (status == cutlass::Status::kSuccess && M_padded != M) {
      cudaMemcpyAsync(out_ptr, out_tmp, sizeof(cutlass::half_t) * M * N,
                      cudaMemcpyDeviceToDevice, stream);
      cudaFreeAsync(out_tmp, stream);
    }
  }
#endif

  cudaFreeAsync(a_fp8, stream);
  cudaFreeAsync(a_scales_rm, stream);
  cudaFreeAsync(a_scales_packed, stream);

  if (status != cutlass::Status::kSuccess) {
    fp8_matmul_f16_layout(input, weight, weight_scale, output, M, N, K,
                          scale_row_stride, block_size_y, block_size_x,
                          weight_col_major, stream);
  }
#else
  fp8_matmul_f16_layout(input, weight, weight_scale, output, M, N, K,
                        scale_row_stride, block_size_y, block_size_x,
                        weight_col_major, stream);
#endif
}

extern "C" void fp8_matmul_bf16_cutlass(const __nv_bfloat16* input,
                                        const uint8_t* weight,
                                        const float* weight_scale,
                                        __nv_bfloat16* output,
                                        int M,
                                        int N,
                                        int K,
                                        int scale_row_stride,
                                        int block_size_y,
                                        int block_size_x,
                                        int weight_col_major,
                                        cudaStream_t stream) {
#if defined(USE_CUTLASS)
  if (!weight_col_major || block_size_y != kBlockM || block_size_x != kBlockK) {
    fp8_matmul_bf16_layout(input, weight, weight_scale, output, M, N, K,
                           scale_row_stride, block_size_y, block_size_x,
                           weight_col_major, stream);
    return;
  }

  const int M_padded = (M + kBlockM - 1) / kBlockM * kBlockM;
  const int k_blocks = (K + kBlockK - 1) / kBlockK;
  const int a_scale_elems = M_padded * k_blocks;

  uint8_t* a_fp8 = nullptr;
  float* a_scales_rm = nullptr;
  float* a_scales_packed = nullptr;
  cudaMallocAsync(&a_fp8, sizeof(uint8_t) * M_padded * K, stream);
  cudaMallocAsync(&a_scales_rm, sizeof(float) * a_scale_elems, stream);
  cudaMallocAsync(&a_scales_packed, sizeof(float) * a_scale_elems, stream);

  dim3 grid(k_blocks, M_padded);
  quantize_fp8_blockwise_f32<<<grid, kPackThreads, 0, stream>>>(
      input, a_fp8, a_scales_rm, M, M_padded, K);

  auto status = cutlass::Status::kErrorNotSupported;
  cudaDeviceProp props{};
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&props, device_id);

  if (props.major == 9 && props.minor == 0) {
    cutlass::bfloat16_t* out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(output);
    cutlass::bfloat16_t* out_tmp = out_ptr;
    if (M_padded != M) {
      cudaMallocAsync(&out_tmp, sizeof(cutlass::bfloat16_t) * M_padded * N, stream);
    }
    status = fp8_blockwise_sm90_dispatch(
        out_tmp,
        reinterpret_cast<const cutlass::float_e4m3_t*>(a_fp8),
        reinterpret_cast<const cutlass::float_e4m3_t*>(weight),
        a_scales_packed,
        a_scales_rm,
        weight_scale,
        M_padded, N, K, stream);
    if (status == cutlass::Status::kSuccess && M_padded != M) {
      cudaMemcpyAsync(out_ptr, out_tmp, sizeof(cutlass::bfloat16_t) * M * N,
                      cudaMemcpyDeviceToDevice, stream);
      cudaFreeAsync(out_tmp, stream);
    }
  }
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  if (props.major == 10) {
    cutlass::bfloat16_t* out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(output);
    cutlass::bfloat16_t* out_tmp = out_ptr;
    if (M_padded != M) {
      cudaMallocAsync(&out_tmp, sizeof(cutlass::bfloat16_t) * M_padded * N, stream);
    }
    status = fp8_blockwise_sm100_dispatch(
        out_tmp,
        reinterpret_cast<const cutlass::float_e4m3_t*>(a_fp8),
        reinterpret_cast<const cutlass::float_e4m3_t*>(weight),
        a_scales_packed,
        a_scales_rm,
        weight_scale,
        M_padded, N, K, stream);
    if (status == cutlass::Status::kSuccess && M_padded != M) {
      cudaMemcpyAsync(out_ptr, out_tmp, sizeof(cutlass::bfloat16_t) * M * N,
                      cudaMemcpyDeviceToDevice, stream);
      cudaFreeAsync(out_tmp, stream);
    }
  }
#endif
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  if (props.major == 12) {
    cutlass::bfloat16_t* out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(output);
    cutlass::bfloat16_t* out_tmp = out_ptr;
    if (M_padded != M) {
      cudaMallocAsync(&out_tmp, sizeof(cutlass::bfloat16_t) * M_padded * N, stream);
    }
    status = fp8_blockwise_sm120_dispatch(
        out_tmp,
        reinterpret_cast<const cutlass::float_e4m3_t*>(a_fp8),
        reinterpret_cast<const cutlass::float_e4m3_t*>(weight),
        a_scales_packed,
        a_scales_rm,
        weight_scale,
        M_padded, N, K, stream);
    if (status == cutlass::Status::kSuccess && M_padded != M) {
      cudaMemcpyAsync(out_ptr, out_tmp, sizeof(cutlass::bfloat16_t) * M * N,
                      cudaMemcpyDeviceToDevice, stream);
      cudaFreeAsync(out_tmp, stream);
    }
  }
#endif

  cudaFreeAsync(a_fp8, stream);
  cudaFreeAsync(a_scales_rm, stream);
  cudaFreeAsync(a_scales_packed, stream);

  if (status != cutlass::Status::kSuccess) {
    fp8_matmul_bf16_layout(input, weight, weight_scale, output, M, N, K,
                           scale_row_stride, block_size_y, block_size_x,
                           weight_col_major, stream);
  }
#else
  fp8_matmul_bf16_layout(input, weight, weight_scale, output, M, N, K,
                         scale_row_stride, block_size_y, block_size_x,
                         weight_col_major, stream);
#endif
}

extern "C" void moe_gemm_fp8_cutlass(
    const void* input,
    const uint8_t* weights,
    const float* weight_scales,
    const int* sorted_token_ids,
    const int* expert_ids,
    const float* topk_weights,
    void* output,
    int* expert_counts,
    int* expert_offsets,
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k,
    int block_size_n,
    int block_size_k,
    int dtype,
    bool is_prefill,
    int weight_col_major,
    cudaStream_t stream) {
  moe_gemm_wmma_fp8_layout(input, weights, weight_scales,
                           sorted_token_ids, expert_ids, topk_weights,
                           output, expert_counts, expert_offsets,
                           num_experts, topk, size_m, size_n, size_k,
                           block_size_n, block_size_k, dtype, is_prefill,
                           weight_col_major, stream);
}
