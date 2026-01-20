#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <type_traits>

#include "attention/dtype_fp8.cuh"
#include "moe/moe_utils.cuh"

#if defined(USE_CUTLASS)
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

namespace vllm_rs_moe {

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<half>(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ half from_float<half>(float v) {
  return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

template <typename T>
__global__ void gather_rows_kernel(
    const T* input,
    const int32_t* dst2src_map,
    T* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    int32_t map_divisor) {
  int64_t dst_row = blockIdx.x;
  if (dst_row >= num_dst_rows) {
    return;
  }
  int64_t src_row = dst2src_map[dst_row] / map_divisor;
  if (src_row >= num_src_rows) {
    return;
  }

  constexpr int kBytesPerVec = 16;
  constexpr int kElemsPerVec = kBytesPerVec / sizeof(T);
  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(input + src_row * num_cols);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(output + dst_row * num_cols);
  const bool use_vec = (src_addr % kBytesPerVec == 0) && (dst_addr % kBytesPerVec == 0) &&
      (num_cols % kElemsPerVec == 0);

  if (use_vec) {
    int64_t vec_cols = num_cols / kElemsPerVec;
    auto* dst_vec = reinterpret_cast<uint4*>(output + dst_row * num_cols);
    auto* src_vec = reinterpret_cast<const uint4*>(input + src_row * num_cols);
    for (int64_t i = threadIdx.x; i < vec_cols; i += blockDim.x) {
      dst_vec[i] = src_vec[i];
    }
  } else {
    for (int64_t i = threadIdx.x; i < num_cols; i += blockDim.x) {
      output[dst_row * num_cols + i] = input[src_row * num_cols + i];
    }
  }
}

template <typename T>
__global__ void scatter_rows_kernel(
    const T* input,
    const int32_t* src2dst_map,
    T* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    const float* weights) {
  int64_t src_row = blockIdx.x;
  if (src_row >= num_src_rows) {
    return;
  }
  int64_t dst_row = src2dst_map[src_row];
  if (dst_row >= num_dst_rows) {
    return;
  }

  float w = 1.0f;
  if (weights != nullptr) {
    w = weights[dst_row];
  }

  constexpr int kBytesPerVec = 16;
  constexpr int kElemsPerVec = kBytesPerVec / sizeof(T);
  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(input + src_row * num_cols);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(output + dst_row * num_cols);
  const bool use_vec = (src_addr % kBytesPerVec == 0) && (dst_addr % kBytesPerVec == 0) &&
      (num_cols % kElemsPerVec == 0);

  if (use_vec) {
    int64_t vec_cols = num_cols / kElemsPerVec;
    for (int64_t i = threadIdx.x; i < vec_cols; i += blockDim.x) {
      auto* src = reinterpret_cast<const uint4*>(input + src_row * num_cols + i * kElemsPerVec);
      auto* dst = reinterpret_cast<uint4*>(output + dst_row * num_cols + i * kElemsPerVec);
      uint4 v = *src;
      T* vals = reinterpret_cast<T*>(&v);
      for (int j = 0; j < kElemsPerVec; ++j) {
        float val = to_float<T>(vals[j]) * w;
        vals[j] = from_float<T>(val);
      }
      *dst = v;
    }
  } else {
    for (int64_t i = threadIdx.x; i < num_cols; i += blockDim.x) {
      float val = to_float<T>(input[src_row * num_cols + i]) * w;
      output[dst_row * num_cols + i] = from_float<T>(val);
    }
  }
}

template <typename ScaleConfig, typename LayoutSFA, typename LayoutSFB, typename StrideA, typename StrideB, typename StrideC,
          typename UnderlyingProblemShape, typename OutType>
__global__ void build_grouped_gemm_args(
    int num_experts,
    const int32_t* expert_offsets,
    int n,
    int k,
    int n_blocks,
    int k_blocks,
    const cutlass::float_e4m3_t* a,
    const cutlass::float_e4m3_t* b,
    const float* a_scales,
    const float* b_scales,
    const cutlass::float_e4m3_t** a_ptrs,
    const cutlass::float_e4m3_t** b_ptrs,
    OutType** out_ptrs,
    const float** a_scales_ptrs,
    const float** b_scales_ptrs,
    StrideA* stride_a,
    StrideB* stride_b,
    StrideC* stride_c,
    LayoutSFA* layout_sfa,
    LayoutSFB* layout_sfb,
    UnderlyingProblemShape* problem_sizes,
    OutType* out) {
  int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert_id >= num_experts) {
    return;
  }

  int32_t start = expert_offsets[expert_id];
  int32_t end = expert_offsets[expert_id + 1];
  int32_t m = end - start;

  if (m == 0) {
    a_ptrs[expert_id] = nullptr;
    b_ptrs[expert_id] = nullptr;
    out_ptrs[expert_id] = nullptr;
    a_scales_ptrs[expert_id] = nullptr;
    b_scales_ptrs[expert_id] = nullptr;
    stride_a[expert_id] = StrideA{};
    stride_b[expert_id] = StrideB{};
    stride_c[expert_id] = StrideC{};
    layout_sfa[expert_id] = LayoutSFA{};
    layout_sfb[expert_id] = LayoutSFB{};
    problem_sizes[expert_id] = {0, 0, 0};
    return;
  }

  problem_sizes[expert_id] = {m, n, k};

  a_ptrs[expert_id] = a + static_cast<int64_t>(start) * k;
  b_ptrs[expert_id] = b + static_cast<int64_t>(expert_id) * n * k;
  out_ptrs[expert_id] = out + static_cast<int64_t>(start) * n;
  a_scales_ptrs[expert_id] = a_scales + static_cast<int64_t>(start) * k_blocks;
  b_scales_ptrs[expert_id] = b_scales + static_cast<int64_t>(expert_id) * n_blocks * k_blocks;

  stride_a[expert_id] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  stride_b[expert_id] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  stride_c[expert_id] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));

  layout_sfa[expert_id] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  layout_sfb[expert_id] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
}

template <typename OutType, typename ScheduleConfig, typename LayoutD>
cutlass::Status launch_grouped_gemm(
    const cutlass::float_e4m3_t* a,
    const cutlass::float_e4m3_t* b,
    const float* a_scales,
    const float* b_scales,
    const int32_t* expert_offsets,
    int num_experts,
    int n,
    int k,
    int n_blocks,
    int k_blocks,
    OutType* out,
    cudaStream_t stream) {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementAccumulator = float;
  using ElementC = void;
  using ElementD = OutType;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = LayoutD;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = typename ScheduleConfig::ArchTag;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using CustomEVTIdentity = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Identity, ElementD, ElementAccumulator, RoundStyle>,
      cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogueSm90 = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutC*,
      AlignmentC,
      typename ScheduleConfig::EpilogueSchedule,
      CustomEVTIdentity>::CollectiveOp;

  using CollectiveEpilogueDefault = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutC*,
      AlignmentC,
      typename ScheduleConfig::EpilogueSchedule>::CollectiveOp;

  using CollectiveEpilogue = std::conditional_t<
      std::is_same_v<ArchTag, cutlass::arch::Sm90>,
      CollectiveEpilogueSm90,
      CollectiveEpilogueDefault>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA*, typename ScheduleConfig::LayoutSFA*>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB*, typename ScheduleConfig::LayoutSFB*>,
      AlignmentB,
      ElementAccumulator,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  const int threads = 256;
  const int blocks = (num_experts + threads - 1) / threads;

  const cutlass::float_e4m3_t** a_ptrs = nullptr;
  const cutlass::float_e4m3_t** b_ptrs = nullptr;
  OutType** out_ptrs = nullptr;
  const float** a_scales_ptrs = nullptr;
  const float** b_scales_ptrs = nullptr;
  StrideA* stride_a = nullptr;
  StrideB* stride_b = nullptr;
  StrideC* stride_c = nullptr;
  typename ScheduleConfig::LayoutSFA* layout_sfa = nullptr;
  typename ScheduleConfig::LayoutSFB* layout_sfb = nullptr;
  UnderlyingProblemShape* problem_sizes = nullptr;

  cudaMallocAsync(&a_ptrs, sizeof(cutlass::float_e4m3_t const*) * num_experts, stream);
  cudaMallocAsync(&b_ptrs, sizeof(cutlass::float_e4m3_t const*) * num_experts, stream);
  cudaMallocAsync(&out_ptrs, sizeof(OutType*) * num_experts, stream);
  cudaMallocAsync(&a_scales_ptrs, sizeof(float const*) * num_experts, stream);
  cudaMallocAsync(&b_scales_ptrs, sizeof(float const*) * num_experts, stream);
  cudaMallocAsync(&stride_a, sizeof(StrideA) * num_experts, stream);
  cudaMallocAsync(&stride_b, sizeof(StrideB) * num_experts, stream);
  cudaMallocAsync(&stride_c, sizeof(StrideC) * num_experts, stream);
  cudaMallocAsync(&layout_sfa, sizeof(typename ScheduleConfig::LayoutSFA) * num_experts, stream);
  cudaMallocAsync(&layout_sfb, sizeof(typename ScheduleConfig::LayoutSFB) * num_experts, stream);
  cudaMallocAsync(&problem_sizes, sizeof(UnderlyingProblemShape) * num_experts, stream);

  build_grouped_gemm_args<typename ScheduleConfig::ScaleConfig, typename ScheduleConfig::LayoutSFA, typename ScheduleConfig::LayoutSFB, StrideA, StrideB, StrideC, UnderlyingProblemShape, OutType>
      <<<blocks, threads, 0, stream>>>(
          num_experts,
          expert_offsets,
          n,
          k,
          n_blocks,
          k_blocks,
          a,
          b,
          a_scales,
          b_scales,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
          stride_a,
          stride_b,
          stride_c,
          layout_sfa,
          layout_sfb,
          problem_sizes,
          out);

  Gemm gemm_op;
  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptrs,
      stride_a,
      b_ptrs,
      stride_b,
      a_scales_ptrs,
      layout_sfa,
      b_scales_ptrs,
      layout_sfb};

  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      stride_c,
      out_ptrs,
      stride_c};

  cutlass::KernelHardwareInfo hw_info;
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, device_id);
  hw_info.device_id = device_id;
  hw_info.sm_count = props.multiProcessorCount;

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  cutlass::Status status = gemm_op.can_implement(args);

  void* workspace = nullptr;

  if (status == cutlass::Status::kSuccess) {
    size_t workspace_size = gemm_op.get_workspace_size(args);
    if (workspace_size > 0) {
      if (cudaMallocAsync(&workspace, workspace_size, stream) != cudaSuccess) {
        status = cutlass::Status::kErrorInternal;
      }
    }
  }

  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.initialize(args, workspace, stream);
  }

  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.run(stream);
  }

  // ---- cleanup ----
  if (workspace) cudaFreeAsync(workspace, stream);
  cudaFreeAsync(a_ptrs, stream);
  cudaFreeAsync(b_ptrs, stream);
  cudaFreeAsync(out_ptrs, stream);
  cudaFreeAsync(a_scales_ptrs, stream);
  cudaFreeAsync(b_scales_ptrs, stream);
  cudaFreeAsync(stride_a, stream);
  cudaFreeAsync(stride_b, stream);
  cudaFreeAsync(stride_c, stream);
  cudaFreeAsync(layout_sfa, stream);
  cudaFreeAsync(layout_sfb, stream);
  cudaFreeAsync(problem_sizes, stream);

  return status;
}

struct Sm90GroupConfig {
  using ArchTag = cutlass::arch::Sm90;
  using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8Blockwise;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128, cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

#if (defined(CUTLASS_ARCH_MMA_SM100A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)) && \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12080)
struct Sm100GroupConfig {
  using ArchTag = cutlass::arch::Sm100;
  using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};
#endif

#if (defined(CUTLASS_ARCH_MMA_SM120A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)) && \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12080)
struct Sm120GroupConfig {
  using ArchTag = cutlass::arch::Sm120;
  using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelScheduleSm120Blockwise;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};
#endif

}  // namespace vllm_rs_moe

extern "C" void moe_fp8_calculate_expert_offsets(
    const int32_t* expert_ids,
    int32_t* expert_counts,
    int32_t* expert_offsets,
    int num_experts,
    int size_m,
    bool is_prefill,
    cudaStream_t stream) {
  if (is_prefill) {
    calculate_expert_offsets(expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream);
  } else {
    calculate_expert_offsets_light(expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream);
  }
}

extern "C" void moe_fp8_shuffle_rows_u8(
    const uint8_t* input,
    const int32_t* dst2src_map,
    uint8_t* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    int32_t map_divisor,
    cudaStream_t stream) {
  dim3 grid(static_cast<uint32_t>(num_dst_rows));
  dim3 block(256);
  vllm_rs_moe::gather_rows_kernel<uint8_t><<<grid, block, 0, stream>>>(
      input, dst2src_map, output, num_src_rows, num_dst_rows, num_cols, map_divisor);
}

extern "C" void moe_fp8_shuffle_rows_f32(
    const float* input,
    const int32_t* dst2src_map,
    float* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    int32_t map_divisor,
    cudaStream_t stream) {
  dim3 grid(static_cast<uint32_t>(num_dst_rows));
  dim3 block(256);
  vllm_rs_moe::gather_rows_kernel<float><<<grid, block, 0, stream>>>(
      input, dst2src_map, output, num_src_rows, num_dst_rows, num_cols, map_divisor);
}

extern "C" void moe_fp8_scatter_rows_f16(
    const half* input,
    const int32_t* src2dst_map,
    half* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    const float* weights,
    cudaStream_t stream) {
  dim3 grid(static_cast<uint32_t>(num_src_rows));
  dim3 block(256);
  vllm_rs_moe::scatter_rows_kernel<half><<<grid, block, 0, stream>>>(
      input, src2dst_map, output, num_src_rows, num_dst_rows, num_cols, weights);
}

extern "C" void moe_fp8_scatter_rows_bf16(
    const __nv_bfloat16* input,
    const int32_t* src2dst_map,
    __nv_bfloat16* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    const float* weights,
    cudaStream_t stream) {
  dim3 grid(static_cast<uint32_t>(num_src_rows));
  dim3 block(256);
  vllm_rs_moe::scatter_rows_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
      input, src2dst_map, output, num_src_rows, num_dst_rows, num_cols, weights);
}

extern "C" void moe_fp8_grouped_gemm_f16(
    const uint8_t* a,
    const uint8_t* b,
    const float* a_scales,
    const float* b_scales,
    const int32_t* expert_offsets,
    int num_experts,
    int n,
    int k,
    int block_size_n,
    int block_size_k,
    int sm_version,
    half* out,
    cudaStream_t stream) {
#if defined(USE_CUTLASS)
  const auto* a_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(a);
  const auto* b_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(b);
  auto* out_ptr = reinterpret_cast<cutlass::half_t*>(out);

  int n_blocks = (n + block_size_n - 1) / block_size_n;
  int k_blocks = (k + block_size_k - 1) / block_size_k;

#if (defined(CUTLASS_ARCH_MMA_SM100A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)) && \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12080)
#if CUDA_VERSION >= 12090
  if (sm_version == 100 || sm_version == 103) {
#else
  if (sm_version == 100) {
#endif
    auto status = vllm_rs_moe::launch_grouped_gemm<cutlass::half_t, vllm_rs_moe::Sm100GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, n, k, n_blocks, k_blocks, out_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_f16 sm100 failed: %s\n", cutlassGetStatusString(status));
    }
    return;
  }
#endif

#if (defined(CUTLASS_ARCH_MMA_SM120A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)) && \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12080)
  if (sm_version == 120) {
    auto status = vllm_rs_moe::launch_grouped_gemm<cutlass::half_t, vllm_rs_moe::Sm120GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, n, k, n_blocks, k_blocks, out_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_f16 sm120 failed: %s\n", cutlassGetStatusString(status));
    }
    return;
  }
#endif

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  if (sm_version == 90) {
    auto status = vllm_rs_moe::launch_grouped_gemm<cutlass::half_t, vllm_rs_moe::Sm90GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, n, k, n_blocks, k_blocks, out_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_f16 sm90 failed: %s\n", cutlassGetStatusString(status));
    }
    return;
  }
#endif
  printf("moe_fp8_grouped_gemm_f16 unsupported sm_version %d\n", sm_version);
#else
  (void)a;
  (void)b;
  (void)a_scales;
  (void)b_scales;
  (void)expert_offsets;
  (void)num_experts;
  (void)n;
  (void)k;
  (void)block_size_n;
  (void)block_size_k;
  (void)sm_version;
  (void)out;
  (void)stream;
#endif
}

extern "C" void moe_fp8_grouped_gemm_bf16(
    const uint8_t* a,
    const uint8_t* b,
    const float* a_scales,
    const float* b_scales,
    const int32_t* expert_offsets,
    int num_experts,
    int n,
    int k,
    int block_size_n,
    int block_size_k,
    int sm_version,
    __nv_bfloat16* out,
    cudaStream_t stream) {
#if defined(USE_CUTLASS)
  const auto* a_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(a);
  const auto* b_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(b);
  auto* out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(out);

  int n_blocks = (n + block_size_n - 1) / block_size_n;
  int k_blocks = (k + block_size_k - 1) / block_size_k;

#if (defined(CUTLASS_ARCH_MMA_SM100A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)) && \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12080)
#if CUDA_VERSION >= 12090
  if (sm_version == 100 || sm_version == 103) {
#else
  if (sm_version == 100) {
#endif
    auto status = vllm_rs_moe::launch_grouped_gemm<cutlass::bfloat16_t, vllm_rs_moe::Sm100GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, n, k, n_blocks, k_blocks, out_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_bf16 sm100 failed: %s\n", cutlassGetStatusString(status));
    }
    return;
  }
#endif

#if (defined(CUTLASS_ARCH_MMA_SM120A_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)) && \
    (defined(CUDA_VERSION) && CUDA_VERSION >= 12080)
  if (sm_version >= 120) {
    auto status = vllm_rs_moe::launch_grouped_gemm<cutlass::bfloat16_t, vllm_rs_moe::Sm120GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, n, k, n_blocks, k_blocks, out_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_bf16 sm120 failed: %s\n", cutlassGetStatusString(status));
    }
    return;
  }
#endif

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  if (sm_version == 90) {
    auto status = vllm_rs_moe::launch_grouped_gemm<cutlass::bfloat16_t, vllm_rs_moe::Sm90GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, n, k, n_blocks, k_blocks, out_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_bf16 sm90 failed: %s\n", cutlassGetStatusString(status));
    }
    return;
  }
#endif
  printf("moe_fp8_grouped_gemm_bf16 unsupported sm_version %d\n", sm_version);
#else
  (void)a;
  (void)b;
  (void)a_scales;
  (void)b_scales;
  (void)expert_offsets;
  (void)num_experts;
  (void)n;
  (void)k;
  (void)block_size_n;
  (void)block_size_k;
  (void)sm_version;
  (void)out;
  (void)stream;
#endif
}

#endif
