// Adapted from https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu
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

constexpr int kBlockM = 128;
constexpr int kBlockK = 128;
constexpr int kPackThreads = 256;

__device__ __forceinline__ float to_float_half(half v) {
  return __half2float(v);
}

__device__ __forceinline__ float to_float_bf16(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

#ifndef MAX_FP8_VALUE
#define MAX_FP8_VALUE 448.0f
#endif

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float group_reduce_max(float val) {
    unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;
    val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
    val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
    return val;
}

template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR, bool SCALE_UE8M0>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input,
    void* __restrict__ output_q,
    float* __restrict__ output_s,

    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float min_8bit,
    const float max_8bit,
    const int num_groups_per_row = 0,
    const int scale_stride = 0) {
    
    const int threads_per_group = 16;
    const int64_t local_group_id = threadIdx.x / threads_per_group;
    const int lane_id = threadIdx.x % threads_per_group;

    const int64_t block_group_id = blockIdx.x * groups_per_block;
    const int64_t global_group_id = block_group_id + local_group_id;
    
    if (global_group_id >= num_groups) return;

    const int64_t block_group_offset = global_group_id * group_size;

    float local_absmax = eps;

    const T* group_input = input + block_group_offset;
    DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset;
    float* scale_output;

    if constexpr (IS_COLUMN_MAJOR) {
        // const int num_elems_per_pack = 1; // float is 4 bytes
        const int row_idx = global_group_id / num_groups_per_row;
        const int col_idx = global_group_id % num_groups_per_row;
        // Simplified for float scales (no packing needed for f32)
        scale_output = output_s + (col_idx * scale_stride + row_idx);
    } else {
        scale_output = output_s + global_group_id;
    }

    // Vectorized Load Logic replacement for flashinfer
    // Assuming T is half or bfloat16 (2 bytes). 
    // vec_size = 16 / 2 = 8 elements.
    using vec_t = float4; // load 16 bytes
    const int vec_size = 16 / sizeof(T); 
    const int32_t num_vec_elems = group_size / vec_size;

    for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
        // Load 16 bytes
        const int4* ptr_int4 = reinterpret_cast<const int4*>(group_input + i * vec_size);
        int4 loaded = *ptr_int4;
        
        // Unpack to check max
        T* ptr_T = reinterpret_cast<T*>(&loaded);
        #pragma unroll
        for (int j = 0; j < vec_size; ++j) {
            float val;
            if constexpr (std::is_same_v<T, __half>) {
                val = __half2float(ptr_T[j]);
            } else {
                val = __bfloat162float(ptr_T[j]);
            }
            local_absmax = fmaxf(local_absmax, fabsf(val));
        }
    }

    local_absmax = group_reduce_max(local_absmax);

    float y_s = local_absmax / max_8bit;
    // SCALE_UE8M0 is false for us usually

    if (lane_id == 0) {
        *scale_output = y_s;
    }

    // Quantize
    for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
        const int4* ptr_int4 = reinterpret_cast<const int4*>(group_input + i * vec_size);
        int4 loaded = *ptr_int4;
        T* ptr_T = reinterpret_cast<T*>(&loaded);

        #pragma unroll
        for (int j = 0; j < vec_size; ++j) {
             float val;
            if constexpr (std::is_same_v<T, __half>) {
                val = __half2float(ptr_T[j]);
            } else {
                val = __bfloat162float(ptr_T[j]);
            }
            float q_val = fminf(fmaxf(val / y_s, min_8bit), max_8bit);

            group_output[i * vec_size + j] = static_cast<DST_DTYPE>(q_val);
        }
    }
}


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
  void* workspace = nullptr;
  if (workspace_size > 0) {
    auto err = cudaMallocAsync(&workspace, workspace_size, stream);
    if (err != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }
  }

  auto init_status = gemm_op.initialize(args, workspace, stream);
  if (init_status != cutlass::Status::kSuccess) {
    if (workspace != nullptr) {
      cudaFreeAsync(workspace, stream);
    }
    return init_status;
  }
  auto run_status = gemm_op.run(stream);
  if (workspace != nullptr) {
    cudaFreeAsync(workspace, stream);
  }
  return run_status;
}

template <
    typename SchedulerType,
    typename OutType,
    int GroupSizeM_,
    int GroupSizeN_,
    int GroupSizeK_,
    int TileSizeM_ = 128,
    class ClusterShape = Shape<_1, _2, _1>>
struct cutlass_3x_gemm_fp8_blockwise {
  using GroupSizeM = Int<GroupSizeM_>;
  using GroupSizeN = Int<GroupSizeN_>;
  using GroupSizeK = Int<GroupSizeK_>;
  using TileSizeM = Int<TileSizeM_>;

  static_assert(TileSizeM_ % GroupSizeM_ == 0, "TileSizeM must be a multiple of GroupSizeM");

  using ElementAB = cutlass::float_e4m3_t;

  // A matrix configuration
  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  // B matrix configuration
  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  // C/D matrix configuration
  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutType>::value;

  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  using ScaleTileShape = Shape<_1, _128, _128>;
  using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(ScaleTileShape{}));
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  // Multiply-accumulate blocking/pipelining details
  using ElementAccumulator = float;                            // Element type for internal accumulation
  using ElementCompute = float;                                // Element type for compute
  using TileShape = Shape<TileSizeM, GroupSizeN, GroupSizeK>;  // Threadblock-level tile size

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using StoreEpilogueCompute = typename cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>;

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
      EpilogueSchedule,
      StoreEpilogueCompute>::CollectiveOp;

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
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      SchedulerType>;
};

template <typename Gemm>
void cutlass_gemm_caller_blockwise(
                                typename Gemm::ElementD* c_ptr,
                                   const typename Gemm::ElementAB* a_ptr,
                                   const typename Gemm::ElementAB* b_ptr,
                                   float* a_s_ptr,
                                   float* b_s_ptr,
                                   int m,
                                   int n,
                                   int k,
                                   cudaStream_t stream) {
  using GemmKernel = typename Gemm::GemmKernel;
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

  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptr, a_stride, b_ptr, b_stride, a_s_ptr, layout_sfa, b_s_ptr, layout_sfb};
  typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::TileSchedulerArguments scheduler;

  static constexpr bool UsesStreamKScheduler =
      cute::is_same_v<typename GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>;

  if constexpr (UsesStreamKScheduler) {
    using DecompositionMode =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
    using ReductionMode =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::ReductionMode;

    scheduler.decomposition_mode = DecompositionMode::StreamK;
    scheduler.reduction_mode = ReductionMode::Nondeterministic;
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

  cutlass_gemm_caller<GemmKernel>(args, stream);
}


extern "C" void fp8_quantize_per_token_group_launch(
    const void* input,
    void* output_q,
    float* output_s,
    int num_groups,
    int group_size,
    int num_groups_per_row,
    int scale_stride,
    bool is_input_f16,
    bool is_column_major_stats,
    cudaStream_t stream
) {
    constexpr int THREADS_PER_GROUP = 16;
    int groups_per_block = 1;
    if (num_groups % 16 == 0) groups_per_block = 16;
    else if (num_groups % 8 == 0) groups_per_block = 8;
    else if (num_groups % 4 == 0) groups_per_block = 4;
    else if (num_groups % 2 == 0) groups_per_block = 2;

    const int num_blocks = num_groups / groups_per_block;
    const int num_threads = groups_per_block * THREADS_PER_GROUP;

    // Standard E4M3 range
    const float min_8bit = -448.0f;
    const float max_8bit = 448.0f;
    const float eps = 1e-5f;

    if (is_input_f16) {
        if (is_column_major_stats) {
            per_token_group_quant_8bit_kernel<__half, __nv_fp8_e4m3, true, false>
                <<<num_blocks, num_threads, 0, stream>>>(
                    (const __half*)input, output_q, output_s, group_size, num_groups, groups_per_block,
                    eps, min_8bit, max_8bit, num_groups_per_row, scale_stride
                );
        } else {
             per_token_group_quant_8bit_kernel<__half, __nv_fp8_e4m3, false, false>
                <<<num_blocks, num_threads, 0, stream>>>(
                    (const __half*)input, output_q, output_s, group_size, num_groups, groups_per_block,
                    eps, min_8bit, max_8bit
                );
        }
    } else {
        if (is_column_major_stats) {
            per_token_group_quant_8bit_kernel<__nv_bfloat16, __nv_fp8_e4m3, true, false>
                <<<num_blocks, num_threads, 0, stream>>>(
                    (const __nv_bfloat16*)input, output_q, output_s, group_size, num_groups, groups_per_block,
                    eps, min_8bit, max_8bit, num_groups_per_row, scale_stride
                );
        } else {
             per_token_group_quant_8bit_kernel<__nv_bfloat16, __nv_fp8_e4m3, false, false>
                <<<num_blocks, num_threads, 0, stream>>>(
                    (const __nv_bfloat16*)input, output_q, output_s, group_size, num_groups, groups_per_block,
                    eps, min_8bit, max_8bit
                );
        }
    }
}


template <typename T_Out>
void fp8_gemm_launcher_sm90(
                     const uint8_t* a_fp8,
                     const float* a_scales,
                     const uint8_t* b_fp8,
                     const float* b_scales,
                     T_Out* output_ptr,
                     int M, int N, int K,
                     cudaStream_t stream)
{
    if (K > 3 * N) {
        cutlass_gemm_caller_blockwise<cutlass_3x_gemm_fp8_blockwise<cutlass::gemm::StreamKScheduler, T_Out, 1, 128, 128>>(
            output_ptr,
            reinterpret_cast<const cutlass::float_e4m3_t*>(a_fp8),
            reinterpret_cast<const cutlass::float_e4m3_t*>(b_fp8),
            const_cast<float*>(a_scales),
            const_cast<float*>(b_scales),
            M, N, K, stream);
    } else {
        cutlass_gemm_caller_blockwise<
            cutlass_3x_gemm_fp8_blockwise<cutlass::gemm::PersistentScheduler, T_Out, 1, 128, 128>>(
            output_ptr,
            reinterpret_cast<const cutlass::float_e4m3_t*>(a_fp8),
            reinterpret_cast<const cutlass::float_e4m3_t*>(b_fp8),
            const_cast<float*>(a_scales),
            const_cast<float*>(b_scales),
            M, N, K, stream);
    }
}

template <
    typename OutType,
    typename MmaTileShape,
    typename PerSmTileShape,
    typename EpilogueTileShape,
    typename ScalesPerTile,
    int TileSizeM_ = 128,
    class ClusterShape = Shape<_1, _1, _1>>
void launch_sm100_fp8_blockwise_scaled_mm(
    OutType* out,
    const cutlass::float_e4m3_t* a,
    const cutlass::float_e4m3_t* b,
    float* scales_a,
    float* scales_b,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});

  using ElementAB = cutlass::float_e4m3_t;
  using ElementA = ElementAB;
  using ElementB = ElementAB;
  using ElementC = void;
  using ElementD = OutType;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutC = LayoutD;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      cute::UMMA::Major::MN,
      cute::UMMA::Major::K>;
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
      ArchTag,
      OperatorClass,
      PerSmTileShape,
      ClusterShape,
      EpilogueTileShape,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized1Sm>::CollectiveOp;

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
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  using StrideC = typename GemmKernel::StrideD;

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  typename GemmKernel::MainloopArguments mainloop_args{
      a, a_stride, b, b_stride, scales_a, layout_SFA, scales_b, layout_SFB};

  typename GemmKernel::EpilogueArguments epilogue_args{{}, out, c_stride, out, c_stride};
  epilogue_args.thread.alpha = 1.0f;

  typename GemmKernel::Arguments args = {
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1}, mainloop_args, epilogue_args};

  cutlass::Status status = cutlass_gemm_caller<GemmKernel>(args, stream);
  if (status != cutlass::Status::kSuccess) {
    printf("sm100 fp8 gemm failed: %s\n", cutlassGetStatusString(status));
  }
}

template <typename OutType>
void sm100_fp8_blockwise_dispatch_shape(
    OutType* out,
    const cutlass::float_e4m3_t* a,
    const cutlass::float_e4m3_t* b,
    float* scales_a,
    float* scales_b,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  if (m <= 128) {
    using MmaTileShape = Shape<_64, _128, _128>;
    using PerSmTileShape = Shape<_64, _128, _128>;
    using EpilogueTileShape = Shape<_64, _64>;
    using ScalesPerTile = Shape<_64, _1, _1>;
    launch_sm100_fp8_blockwise_scaled_mm<OutType, MmaTileShape, PerSmTileShape, EpilogueTileShape, ScalesPerTile>(
        out, a, b, scales_a, scales_b, m, n, k, stream);
  } else {
    using MmaTileShape = Shape<_128, _128, _128>;
    using PerSmTileShape = Shape<_128, _128, _128>;
    using EpilogueTileShape = Shape<_128, _64>;
    using ScalesPerTile = Shape<_128, _1, _1>;
    launch_sm100_fp8_blockwise_scaled_mm<OutType, MmaTileShape, PerSmTileShape, EpilogueTileShape, ScalesPerTile>(
        out, a, b, scales_a, scales_b, m, n, k, stream);
  }
}

template <
    typename OutType,
    typename MmaTileShape,
    typename PerSmTileShape,
    typename EpilogueTileShape,
    typename ScalesPerTile,
    int TileSizeM_ = 128,
    class ClusterShape = Shape<_1, _1, _1>>
void launch_sm120_fp8_blockwise_scaled_mm(
    OutType* out,
    const cutlass::float_e4m3_t* a,
    const cutlass::float_e4m3_t* b,
    float* scales_a,
    float* scales_b,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  using ElementBlockScale = float;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutATag = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});

  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      cute::UMMA::Major::MN,
      cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutATag, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutBTag, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;
  using StrideC = typename GemmKernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  typename GemmKernel::MainloopArguments mainloop_args{
      a, stride_a, b, stride_b, scales_a, layout_SFA, scales_b, layout_SFB};

  typename GemmKernel::EpilogueArguments epilogue_args{{}, out, stride_c, out, stride_c};
  epilogue_args.thread.alpha = 1.0f;

  typename GemmKernel::Arguments args = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      mainloop_args,
      epilogue_args,
  };

  cutlass::Status status = cutlass_gemm_caller<GemmKernel>(args, stream);
  if (status != cutlass::Status::kSuccess) {
    printf("sm120 fp8 gemm failed: %s\n", cutlassGetStatusString(status));
  }
}

template <typename OutType>
void sm120_fp8_blockwise_dispatch_shape(
    OutType* out,
    const cutlass::float_e4m3_t* a,
    const cutlass::float_e4m3_t* b,
    float* scales_a,
    float* scales_b,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape = Shape<_128, _128, _128>;
  using EpilogueTileShape = Shape<_128, _64>;
  using ScalesPerTile = Shape<_128, _1, _1>;
  launch_sm120_fp8_blockwise_scaled_mm<OutType, MmaTileShape, PerSmTileShape, EpilogueTileShape, ScalesPerTile>(
      out, a, b, scales_a, scales_b, m, n, k, stream);
}

#endif

extern "C" void fp8_matmul_f16_cutlass(const uint8_t* input_q,
                                       const float* input_scale,
                                       const uint8_t* weight,
                                       const float* weight_scale,
                                       __half* output,
                                       int M, int N, int K,
                                       int /*scale_row_stride*/, // Unused args
                                       int /*block_size_y*/,
                                       int /*block_size_x*/,
                                       int sm_version,
                                       cudaStream_t stream) {
#if defined(USE_CUTLASS)
    const auto* a_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(input_q);
    const auto* b_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(weight);
    auto* out_ptr = reinterpret_cast<cutlass::half_t*>(output);
    auto* a_scales = const_cast<float*>(input_scale);
    auto* b_scales = const_cast<float*>(weight_scale);

#ifdef ENABLE_SM120
    sm120_fp8_blockwise_dispatch_shape<cutlass::half_t>(
        out_ptr, a_ptr, b_ptr, a_scales, b_scales, M, N, K, stream);
    return;
#endif

#ifdef ENABLE_SM100
    sm100_fp8_blockwise_dispatch_shape<cutlass::half_t>(
        out_ptr, a_ptr, b_ptr, a_scales, b_scales, M, N, K, stream);
    return;
#endif

    fp8_gemm_launcher_sm90<cutlass::half_t>(
        input_q, input_scale, weight, weight_scale, out_ptr, M, N, K, stream);
#endif
}

extern "C" void fp8_matmul_bf16_cutlass(const uint8_t* input_q,
                                        const float* input_scale,
                                        const uint8_t* weight,
                                        const float* weight_scale,
                                        __nv_bfloat16* output,
                                        int M, int N, int K,
                                        int /*scale_row_stride*/,
                                        int /*block_size_y*/,
                                        int /*block_size_x*/,
                                        int sm_version,
                                        cudaStream_t stream) {
#if defined(USE_CUTLASS)
    const auto* a_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(input_q);
    const auto* b_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(weight);
    auto* out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(output);
    auto* a_scales = const_cast<float*>(input_scale);
    auto* b_scales = const_cast<float*>(weight_scale);

#ifdef ENABLE_SM120
    sm120_fp8_blockwise_dispatch_shape<cutlass::bfloat16_t>(
        out_ptr, a_ptr, b_ptr, a_scales, b_scales, M, N, K, stream);
    return;
#endif

#ifdef ENABLE_SM100
    sm100_fp8_blockwise_dispatch_shape<cutlass::bfloat16_t>(
        out_ptr, a_ptr, b_ptr, a_scales, b_scales, M, N, K, stream);
    return;
#endif

    fp8_gemm_launcher_sm90<cutlass::bfloat16_t>(
        input_q, input_scale, weight, weight_scale, out_ptr, M, N, K, stream);
#endif
}
