/*
 * Hardware-accelerated NVFP4 MoE Grouped GEMM using CUTLASS block-scaled tensor ops.
 * SM100 path: KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100, 128x128x128 tile
 * SM120 path: KernelScheduleAuto, 128x256x256 tile, StageCount<2>
 * Runtime dispatch via cudaDeviceGetAttribute.
 */

#ifdef ENABLE_FP4

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <type_traits>

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

using namespace cute;

namespace {

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

void* suballocate_workspace(
    char* workspace_base,
    size_t workspace_bytes,
    size_t& workspace_offset,
    size_t allocation_bytes,
    size_t alignment = 256) {
  size_t aligned_offset = align_up(workspace_offset, alignment);
  if (aligned_offset > workspace_bytes || allocation_bytes > workspace_bytes - aligned_offset) {
    return nullptr;
  }
  void* result = workspace_base + aligned_offset;
  workspace_offset = aligned_offset + allocation_bytes;
  return result;
}

}  // namespace

// ============================================================================
// Grouped GEMM for MoE with NVFP4 block-scaled weights
// ============================================================================

#if defined(ENABLE_FP4_SM100)

template <typename OutType>
struct Fp4MoeGemmSm100 {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;

  using ElementType = cutlass::float_e2m1_t;
  using ElementSFType = cutlass::float_ue4m3_t;

  // SM100 requires cute::tuple<Element, SF> for the mainloop builder,
  // matching the FlashInfer SM100 dense GEMM pattern.
  using ElementA = cute::tuple<ElementType, ElementSFType>;
  using ElementB = cute::tuple<ElementType, ElementSFType>;
  using ElementC = OutType;
  using ElementD = OutType;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD_t = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementType>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementType>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm100;
  using EpilogueOperatorClass = cutlass::arch::OpClassTensorOp;
  using MainloopOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using ClusterShape = Shape<_1, _1, _1>;

  using MmaTileShape = Shape<_128, _128, _128>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, EpilogueOperatorClass, MmaTileShape, ClusterShape,
      Shape<_128, _64>,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC,
      ElementD, LayoutC*, AlignmentD,
      EpilogueSchedule
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, MainloopOperatorClass,
      ElementA, LayoutA*, AlignmentA,
      ElementB, LayoutB*, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
};

#endif // ENABLE_FP4_SM100

// ============================================================================
// SM120 (Blackwell) Grouped GEMM for MoE with NVFP4 block-scaled weights
// ============================================================================

#if defined(ENABLE_FP4_SM120)

// SM120 MoE GEMM configuration - using 128x128x128 tile for better compatibility
// with small problem sizes (MoE often has variable M per expert).
// Uses KernelPtrArrayTmaWarpSpecializedPingpong for optimal software pipelining,
// matching SGLang's implementation.
template <typename OutType>
struct Fp4MoeGemmSm120 {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;

  using ElementType = cutlass::float_e2m1_t;
  using ElementSFType = cutlass::float_ue4m3_t;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementC = OutType;
  using ElementD = OutType;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD_t = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using ClusterShape = Shape<_1, _1, _1>;

  using MmaTileShape = Shape<_128, _128, _128>;
  // Use explicit PingPong schedule for optimal software pipelining (matches SGLang)
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC,
      ElementD, LayoutC*, AlignmentD,
      EpilogueSchedule
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA*, AlignmentA,
      ElementB, LayoutB*, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
};

#endif // ENABLE_FP4_SM120

// ============================================================================
// Device kernel to set up per-expert pointer arrays for grouped GEMM
// ============================================================================

template <
    typename ElementAB,
    typename ElementOut,
    typename ElementSF,
    typename ElementAccumulator,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ScaleConfig>
__global__ void setup_moe_group_gemm_args(
    const ElementAB** a_offsets,
    const ElementAB** b_offsets,
    ElementOut** out_offsets,
    const ElementSF** a_scales_offsets,
    const ElementSF** b_scales_offsets,
    const ElementAccumulator** alpha_offsets,
    StrideA* stride_a,
    StrideB* stride_b,
    StrideC* stride_c,
    LayoutSFA* layout_sfa,
    LayoutSFB* layout_sfb,
    const ElementAB* a_base,
    const ElementAB* b_base,
    ElementOut* out_base,
    const ElementSF* a_scales_base,
    const ElementSF* b_scales_base,
    const ElementAccumulator* alphas_base,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    const int32_t* problem_sizes,
    const int num_experts,
    const int K,
    const int N)
{
  int64_t expert_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (expert_id >= num_experts) return;

  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);
  int64_t group_size = 16;
  int64_t row_align = 128;

  int64_t m = static_cast<int64_t>(problem_sizes[expert_id * 3]);
  int64_t n = static_cast<int64_t>(problem_sizes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(problem_sizes[expert_id * 3 + 2]);

  int64_t half_k = k / 2;
  int64_t group_k = k / group_size;
  int64_t group_k_padded = ((group_k + 3) / 4) * 4;
  int64_t n_padded = ((n + row_align - 1) / row_align) * row_align;

  a_offsets[expert_id] = a_base + expert_offset * half_k;
  b_offsets[expert_id] = b_base + expert_id * n * half_k;
  out_offsets[expert_id] = out_base + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base + sf_offset * group_k_padded;
  b_scales_offsets[expert_id] = b_scales_base + expert_id * n_padded * group_k_padded;
  alpha_offsets[expert_id] = alphas_base + expert_id;

  assert((reinterpret_cast<uintptr_t>(a_scales_offsets[expert_id]) % 128) == 0 &&
         "NVFP4 activation scales must be 128-byte aligned");
  assert((reinterpret_cast<uintptr_t>(b_scales_offsets[expert_id]) % 128) == 0 &&
         "NVFP4 weight scales must be 128-byte aligned");

  stride_a[expert_id] = cutlass::make_cute_packed_stride(StrideA{}, {static_cast<int>(m), static_cast<int>(k), 1});
  stride_b[expert_id] = cutlass::make_cute_packed_stride(StrideB{}, {static_cast<int>(n), static_cast<int>(k), 1});
  stride_c[expert_id] = cutlass::make_cute_packed_stride(StrideC{}, {static_cast<int>(m), static_cast<int>(n), 1});

  layout_sfa[expert_id] =
      ScaleConfig::tile_atom_to_shape_SFA(
          cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  layout_sfb[expert_id] =
      ScaleConfig::tile_atom_to_shape_SFB(
          cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
}

// ============================================================================
// MoE Grouped GEMM Launch
// ============================================================================

#if defined(ENABLE_FP4_SM100)

template <typename OutType>
static int run_fp4_moe_grouped_gemm_sm100(
    void* output,
    const void* a,              // [total_tokens, K/2] packed FP4 activations
    const void* b,              // [E, N, K/2] packed FP4 weights
    const void* a_blockscale,   // [total_sf_rows, K/16] FP8 E4M3 activation scales
    const void* b_blockscales,  // [E, N, K/16] FP8 E4M3 weight scales
    const float* alphas,        // [E] per-expert alpha = input_scale * weight_global_scale
    const int32_t* expert_offsets,  // [E] token offsets per expert
    const int32_t* sf_offsets,      // [E] scale factor row offsets per expert
    const int32_t* problem_sizes,   // [E, 3] (M_i, N, K) per expert
    int num_experts,
    int total_tokens,
    int N, int K,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream)
{
  using GemmConfig = Fp4MoeGemmSm100<OutType>;
  using Gemm = typename GemmConfig::Gemm;
  using ElementType = typename GemmConfig::ElementType;
  using ElementSFType = typename GemmConfig::ElementSFType;
  using ElementD = typename GemmConfig::ElementD;
  using StrideA = typename GemmConfig::StrideA;
  using StrideB = typename GemmConfig::StrideB;
  using StrideC = typename GemmConfig::StrideC;
  using StrideD = typename GemmConfig::StrideD;
  using LayoutSFA = typename GemmConfig::LayoutSFA;
  using LayoutSFB = typename GemmConfig::LayoutSFB;
  using ScaleConfig = typename GemmConfig::ScaleConfig;
  using UnderlyingProblemShape = typename GemmConfig::UnderlyingProblemShape;
  using ProblemShape = typename GemmConfig::ProblemShape;

  // Allocate device arrays for pointer arrays
  size_t ptr_array_bytes = num_experts * sizeof(void*);
  size_t stride_a_bytes = num_experts * sizeof(StrideA);
  size_t stride_b_bytes = num_experts * sizeof(StrideB);
  size_t stride_c_bytes = num_experts * sizeof(StrideC);
  size_t layout_sfa_bytes = num_experts * sizeof(LayoutSFA);
  size_t layout_sfb_bytes = num_experts * sizeof(LayoutSFB);
  size_t stride_bytes = stride_a_bytes;  // for workspace suballocation

  // Total workspace: 6 pointer arrays + 3 stride arrays + 2 layout arrays + padding
  size_t total_workspace = 6 * ptr_array_bytes + stride_a_bytes + stride_b_bytes +
                           stride_c_bytes + layout_sfa_bytes + layout_sfb_bytes +
                           16 * 12;  // alignment padding

  if (workspace == nullptr) {
    fprintf(stderr, "[NVFP4 MoE CUTLASS] workspace pointer is null\n");
    return -1;
  }

  char* workspace_buf = static_cast<char*>(workspace);
  size_t workspace_offset = 0;
  auto alloc_from_workspace = [&](size_t bytes, size_t alignment = 256) -> void* {
    return suballocate_workspace(
        workspace_buf, workspace_bytes, workspace_offset, bytes, alignment);
  };

  auto a_ptrs = static_cast<const ElementType**>(alloc_from_workspace(ptr_array_bytes));
  auto b_ptrs = static_cast<const ElementType**>(alloc_from_workspace(ptr_array_bytes));
  auto out_ptrs = static_cast<ElementD**>(alloc_from_workspace(ptr_array_bytes));
  auto a_sf_ptrs = static_cast<const ElementSFType**>(alloc_from_workspace(ptr_array_bytes));
  auto b_sf_ptrs = static_cast<const ElementSFType**>(alloc_from_workspace(ptr_array_bytes));
  auto alpha_ptrs = static_cast<const float**>(alloc_from_workspace(ptr_array_bytes));
  auto layout_sfa_arr = static_cast<LayoutSFA*>(alloc_from_workspace(layout_sfa_bytes));
  auto layout_sfb_arr = static_cast<LayoutSFB*>(alloc_from_workspace(layout_sfb_bytes));

  // Allocate stride arrays
  auto a_strides = static_cast<StrideA*>(alloc_from_workspace(stride_bytes));
  auto b_strides =
      static_cast<StrideB*>(alloc_from_workspace(num_experts * sizeof(StrideB)));
  auto c_strides =
      static_cast<StrideC*>(alloc_from_workspace(num_experts * sizeof(StrideC)));
  if (!a_ptrs || !b_ptrs || !out_ptrs || !a_sf_ptrs || !b_sf_ptrs || !alpha_ptrs ||
      !layout_sfa_arr || !layout_sfb_arr || !a_strides || !b_strides || !c_strides) {
    fprintf(stderr,
            "[NVFP4 MoE CUTLASS] insufficient workspace for metadata: need at least %zu bytes, got %zu\n",
            total_workspace, workspace_bytes);
    return -1;
  }

  setup_moe_group_gemm_args<
      ElementType, ElementD, ElementSFType, float,
      StrideA, StrideB, StrideC,
      LayoutSFA, LayoutSFB, ScaleConfig>
      <<<(num_experts + 255) / 256, 256, 0, stream>>>(
          a_ptrs, b_ptrs, out_ptrs,
          a_sf_ptrs, b_sf_ptrs, alpha_ptrs,
          a_strides, b_strides, c_strides,
          layout_sfa_arr, layout_sfb_arr,
          static_cast<const ElementType*>(a),
          static_cast<const ElementType*>(b),
          static_cast<ElementD*>(output),
          static_cast<const ElementSFType*>(a_blockscale),
          static_cast<const ElementSFType*>(b_blockscales),
          alphas,
          expert_offsets, sf_offsets, problem_sizes,
          num_experts,
          K, N);
  cudaError_t setup_err = cudaPeekAtLastError();
  if (setup_err != cudaSuccess) {
    fprintf(stderr, "[NVFP4 MoE CUTLASS] setup kernel launch failed: %s\n",
            cudaGetErrorString(setup_err));
    return -1;
  }

  Gemm gemm_op;

  cutlass::KernelHardwareInfo hw_info;
  cudaGetDevice(&hw_info.device_id);
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using GemmElementA = typename Gemm::ElementA;
  using GemmElementB = typename Gemm::ElementB;

  typename Gemm::GemmKernel::MainloopArguments mainloop_args{
      reinterpret_cast<GemmElementA const**>(a_ptrs),
      a_strides,
      reinterpret_cast<GemmElementB const**>(b_ptrs),
      b_strides,
      reinterpret_cast<ElementSFType const**>(a_sf_ptrs),
      layout_sfa_arr,
      reinterpret_cast<ElementSFType const**>(b_sf_ptrs),
      layout_sfb_arr
  };

  typename Gemm::GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      c_strides,
      reinterpret_cast<ElementD**>(out_ptrs),
      c_strides
  };
  auto& fusion_args = epilogue_args.thread;
  fusion_args.alpha_ptr_array = reinterpret_cast<float const**>(alpha_ptrs);
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};

  typename Gemm::GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, static_cast<UnderlyingProblemShape*>(
          const_cast<void*>(static_cast<const void*>(problem_sizes))), nullptr},
      mainloop_args,
      epilogue_args,
      hw_info
  };
  // Use AlongM raster order for better MoE workload performance (matches SGLang)
  if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>) {
    using RasterOrderOptions = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = RasterOrderOptions::AlongM;
  }

  size_t gemm_workspace_size = Gemm::get_workspace_size(args);
  void* gemm_workspace = nullptr;
  if (gemm_workspace_size > 0) {
    // Always suballocate from the Rust-provided device workspace buffer.
    gemm_workspace = suballocate_workspace(
        workspace_buf, workspace_bytes, workspace_offset, gemm_workspace_size, 256);
    if (gemm_workspace == nullptr) {
      fprintf(stderr,
              "[NVFP4 MoE CUTLASS] insufficient workspace for CUTLASS GEMM: need %zu additional bytes, got %zu total\n",
              gemm_workspace_size, workspace_bytes);
      return -1;
    }
  }

  auto can_impl = gemm_op.can_implement(args);
  if (can_impl != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 MoE CUTLASS] can_implement failed: %s\n",
            cutlass::cutlassGetStatusString(can_impl));
    return -1;
  }

  auto status = gemm_op.initialize(args, gemm_workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 MoE CUTLASS] initialize failed: %s\n",
            cutlass::cutlassGetStatusString(status));
    return -1;
  }

  status = gemm_op.run(args, gemm_workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 MoE CUTLASS] run failed: %s\n",
            cutlass::cutlassGetStatusString(status));
    return -1;
  }

  return 0;
}

#endif // ENABLE_FP4_SM100

// ============================================================================
// SM120 MoE Grouped GEMM Launch
// ============================================================================

#if defined(ENABLE_FP4_SM120)

template <typename OutType>
static int run_fp4_moe_grouped_gemm_sm120(
    void* output,
    const void* a,
    const void* b,
    const void* a_blockscale,
    const void* b_blockscales,
    const float* alphas,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    const int32_t* problem_sizes,
    int num_experts,
    int total_tokens,
    int N, int K,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream)
{
  using GemmConfig = Fp4MoeGemmSm120<OutType>;
  using Gemm = typename GemmConfig::Gemm;
  using ElementType = typename GemmConfig::ElementType;
  using ElementSFType = typename GemmConfig::ElementSFType;
  using ElementD = typename GemmConfig::ElementD;
  using StrideA = typename GemmConfig::StrideA;
  using StrideB = typename GemmConfig::StrideB;
  using StrideC = typename GemmConfig::StrideC;
  using StrideD = typename GemmConfig::StrideD;
  using LayoutSFA = typename GemmConfig::LayoutSFA;
  using LayoutSFB = typename GemmConfig::LayoutSFB;
  using ScaleConfig = typename GemmConfig::ScaleConfig;
  using UnderlyingProblemShape = typename GemmConfig::UnderlyingProblemShape;
  using ProblemShape = typename GemmConfig::ProblemShape;

  size_t ptr_array_bytes = num_experts * sizeof(void*);
  size_t stride_a_bytes = num_experts * sizeof(StrideA);
  size_t stride_b_bytes = num_experts * sizeof(StrideB);
  size_t stride_c_bytes = num_experts * sizeof(StrideC);
  size_t layout_sfa_bytes = num_experts * sizeof(LayoutSFA);
  size_t layout_sfb_bytes = num_experts * sizeof(LayoutSFB);

  size_t total_workspace = 6 * ptr_array_bytes + stride_a_bytes + stride_b_bytes +
                           stride_c_bytes + layout_sfa_bytes + layout_sfb_bytes +
                           16 * 12;

  if (workspace == nullptr) {
    fprintf(stderr, "[NVFP4 MoE CUTLASS SM120] workspace pointer is null\n");
    return -1;
  }

  char* workspace_buf = static_cast<char*>(workspace);
  size_t workspace_offset = 0;
  auto alloc_from_workspace = [&](size_t bytes, size_t alignment = 256) -> void* {
    return suballocate_workspace(
        workspace_buf, workspace_bytes, workspace_offset, bytes, alignment);
  };

  auto a_ptrs = static_cast<const ElementType**>(alloc_from_workspace(ptr_array_bytes));
  auto b_ptrs = static_cast<const ElementType**>(alloc_from_workspace(ptr_array_bytes));
  auto out_ptrs = static_cast<ElementD**>(alloc_from_workspace(ptr_array_bytes));
  auto a_sf_ptrs = static_cast<const ElementSFType**>(alloc_from_workspace(ptr_array_bytes));
  auto b_sf_ptrs = static_cast<const ElementSFType**>(alloc_from_workspace(ptr_array_bytes));
  auto alpha_ptrs = static_cast<const float**>(alloc_from_workspace(ptr_array_bytes));
  auto layout_sfa_arr = static_cast<LayoutSFA*>(alloc_from_workspace(layout_sfa_bytes));
  auto layout_sfb_arr = static_cast<LayoutSFB*>(alloc_from_workspace(layout_sfb_bytes));
  auto a_strides = static_cast<StrideA*>(alloc_from_workspace(stride_a_bytes));
  auto b_strides = static_cast<StrideB*>(alloc_from_workspace(stride_b_bytes));
  auto c_strides = static_cast<StrideC*>(alloc_from_workspace(stride_c_bytes));
  if (!a_ptrs || !b_ptrs || !out_ptrs || !a_sf_ptrs || !b_sf_ptrs || !alpha_ptrs ||
      !layout_sfa_arr || !layout_sfb_arr || !a_strides || !b_strides || !c_strides) {
    fprintf(stderr,
            "[NVFP4 MoE CUTLASS SM120] insufficient workspace for metadata: need at least %zu bytes, got %zu\n",
            total_workspace, workspace_bytes);
    return -1;
  }

  setup_moe_group_gemm_args<
      ElementType, ElementD, ElementSFType, float,
      StrideA, StrideB, StrideC,
      LayoutSFA, LayoutSFB, ScaleConfig>
      <<<(num_experts + 255) / 256, 256, 0, stream>>>(
          a_ptrs, b_ptrs, out_ptrs,
          a_sf_ptrs, b_sf_ptrs, alpha_ptrs,
          a_strides, b_strides, c_strides,
          layout_sfa_arr, layout_sfb_arr,
          static_cast<const ElementType*>(a),
          static_cast<const ElementType*>(b),
          static_cast<ElementD*>(output),
          static_cast<const ElementSFType*>(a_blockscale),
          static_cast<const ElementSFType*>(b_blockscales),
          alphas,
          expert_offsets, sf_offsets, problem_sizes,
          num_experts,
          K, N);
  cudaError_t setup_err = cudaPeekAtLastError();
  if (setup_err != cudaSuccess) {
    fprintf(stderr, "[NVFP4 MoE CUTLASS SM120] setup kernel launch failed: %s\n",
            cudaGetErrorString(setup_err));
    return -1;
  }

  Gemm gemm_op;

  cutlass::KernelHardwareInfo hw_info;
  cudaGetDevice(&hw_info.device_id);
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using GemmElementA = typename Gemm::ElementA;
  using GemmElementB = typename Gemm::ElementB;

  typename Gemm::GemmKernel::MainloopArguments mainloop_args{
      reinterpret_cast<GemmElementA const**>(a_ptrs),
      a_strides,
      reinterpret_cast<GemmElementB const**>(b_ptrs),
      b_strides,
      reinterpret_cast<ElementSFType const**>(a_sf_ptrs),
      layout_sfa_arr,
      reinterpret_cast<ElementSFType const**>(b_sf_ptrs),
      layout_sfb_arr
  };

  typename Gemm::GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      c_strides,
      reinterpret_cast<ElementD**>(out_ptrs),
      c_strides
  };
  auto& fusion_args = epilogue_args.thread;
  fusion_args.alpha_ptr_array = reinterpret_cast<float const**>(alpha_ptrs);
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};

  typename Gemm::GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, static_cast<UnderlyingProblemShape*>(
          const_cast<void*>(static_cast<const void*>(problem_sizes))), nullptr},
      mainloop_args,
      epilogue_args,
      hw_info
  };
  if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>) {
    args.scheduler.max_swizzle_size = 1;
  }
  // Use AlongM raster order for better MoE workload performance (matches SGLang)
  if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>) {
    using Enum_t = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = Enum_t::AlongM;
  }
  args.hw_info.cluster_shape = dim3(1, 1, 1);
  args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);

  size_t gemm_workspace_size = Gemm::get_workspace_size(args);
  void* gemm_workspace = nullptr;
  if (gemm_workspace_size > 0) {
    // Always suballocate from the Rust-provided device workspace buffer.
    gemm_workspace = suballocate_workspace(
        workspace_buf, workspace_bytes, workspace_offset, gemm_workspace_size, 256);
    if (gemm_workspace == nullptr) {
      fprintf(stderr,
              "[NVFP4 MoE CUTLASS SM120] insufficient workspace for CUTLASS GEMM: need %zu additional bytes, got %zu total\n",
              gemm_workspace_size, workspace_bytes);
      return -1;
    }
  }

  auto can_impl = gemm_op.can_implement(args);
  if (can_impl != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 MoE CUTLASS SM120] can_implement failed: %s\n",
            cutlass::cutlassGetStatusString(can_impl));
    return -1;
  }

  auto status = gemm_op.initialize(args, gemm_workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 MoE CUTLASS SM120] initialize failed: %s\n",
            cutlass::cutlassGetStatusString(status));
    return -1;
  }

  status = gemm_op.run(stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 MoE CUTLASS SM120] run failed: %s\n",
            cutlass::cutlassGetStatusString(status));
    return -1;
  }

  return 0;
}

#endif // ENABLE_FP4_SM120

// ============================================================================
// C API Entry Points (with SM dispatch)
// ============================================================================

extern "C" {

int nvfp4_cutlass_moe_gemm_f16(
    void* output,
    const void* a,
    const void* b,
    const void* a_blockscale,
    const void* b_blockscales,
    const float* alphas,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    const int32_t* problem_sizes,
    int num_experts,
    int total_tokens,
    int N, int K,
    void* workspace,
    int64_t workspace_bytes,
    int64_t stream)
{
  auto s = reinterpret_cast<cudaStream_t>(stream);
#if defined(ENABLE_FP4_SM120)
  return run_fp4_moe_grouped_gemm_sm120<cutlass::half_t>(
      output, a, b, a_blockscale, b_blockscales, alphas,
      expert_offsets, sf_offsets, problem_sizes,
      num_experts, total_tokens, N, K, workspace, static_cast<size_t>(workspace_bytes), s);
#elif defined(ENABLE_FP4_SM100)
  return run_fp4_moe_grouped_gemm_sm100<cutlass::half_t>(
      output, a, b, a_blockscale, b_blockscales, alphas,
      expert_offsets, sf_offsets, problem_sizes,
      num_experts, total_tokens, N, K, workspace, static_cast<size_t>(workspace_bytes), s);
#else
  return -1;
#endif
}

int nvfp4_cutlass_moe_gemm_bf16(
    void* output,
    const void* a,
    const void* b,
    const void* a_blockscale,
    const void* b_blockscales,
    const float* alphas,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    const int32_t* problem_sizes,
    int num_experts,
    int total_tokens,
    int N, int K,
    void* workspace,
    int64_t workspace_bytes,
    int64_t stream)
{
  auto s = reinterpret_cast<cudaStream_t>(stream);
#if defined(ENABLE_FP4_SM120)
  return run_fp4_moe_grouped_gemm_sm120<cutlass::bfloat16_t>(
      output, a, b, a_blockscale, b_blockscales, alphas,
      expert_offsets, sf_offsets, problem_sizes,
      num_experts, total_tokens, N, K, workspace, static_cast<size_t>(workspace_bytes), s);
#elif defined(ENABLE_FP4_SM100)
  return run_fp4_moe_grouped_gemm_sm100<cutlass::bfloat16_t>(
      output, a, b, a_blockscale, b_blockscales, alphas,
      expert_offsets, sf_offsets, problem_sizes,
      num_experts, total_tokens, N, K, workspace, static_cast<size_t>(workspace_bytes), s);
#else
  return -1;
#endif
}

}  // extern "C"

#else  // !ENABLE_FP4

extern "C" {

int nvfp4_cutlass_moe_gemm_f16(
    void*, const void*, const void*, const void*, const void*,
    const float*, const int32_t*, const int32_t*, const int32_t*,
    int, int, int, int, void*, int64_t, int64_t)
{
  return -1;
}

int nvfp4_cutlass_moe_gemm_bf16(
    void*, const void*, const void*, const void*, const void*,
    const float*, const int32_t*, const int32_t*, const int32_t*,
    int, int, int, int, void*, int64_t, int64_t)
{
  return -1;
}

}  // extern "C"

#endif  // ENABLE_FP4
