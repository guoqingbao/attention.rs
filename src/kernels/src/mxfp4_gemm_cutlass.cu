/*
 * Hardware-accelerated MXFP4 GEMM using CUTLASS block-scaled tensor ops.
 * Targets Blackwell (SM100+) with native MX microscaling tensor core support.
 * Falls back gracefully: on SM < 100, the caller uses software dequant kernels instead.
 *
 * Key differences from NVFP4 CUTLASS GEMM:
 *   - Scale type: E8M0 (float_ue8m0_t) instead of E4M3 (float_ue4m3_t)
 *   - Block size: 32 elements per scale (vs 16 for NVFP4)
 *   - Element type: float_e2m1_t (not wrapped in nv_float4_t)
 *   - Kernel schedule: Mxf8f6f4Sm100 (not Nvf4Sm100)
 *   - No global scale factor (alpha = 1.0, not a device pointer)
 *
 * Based on FlashInfer MXFP8 CUTLASS templates (SM100 path).
 * Requires CUTLASS 4.4.2+ with SM100 block-scaled tensor op support.
 */

#ifdef ENABLE_FP4

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>

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
#include "cutlass/util/packed_stride.hpp"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

using namespace cute;

// ============================================================================
// SM100 MXFP4 Dense GEMM Kernel Configurations
// Following FlashInfer mxfp8_gemm_template_sm100.h patterns for CUTLASS 4.4.2
// Only compiled when targeting SM100-SM119 (datacenter Blackwell)
// ============================================================================

#if defined(ENABLE_FP4_SM100)

struct _1SM {};
struct _2SM {};

template <typename T>
struct MxSMTypeAdapter {};

template <>
struct MxSMTypeAdapter<_1SM> {
  static int const Scale = 1;
  using AtomThrShape = cute::Shape<_1, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100;
};

template <>
struct MxSMTypeAdapter<_2SM> {
  static int const Scale = 2;
  using AtomThrShape = cute::Shape<_2, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100;
};

// ============================================================================
// CUTLASS GEMM Instantiation Template
// ============================================================================

template <typename OutType, int CTA_M, int CTA_N, int CTA_K, typename XSM>
struct CutlassMxfp4Gemm {
  using OutElementType = OutType;
  using CTAShape = cute::Shape<cute::Int<CTA_M>, cute::Int<CTA_N>, cute::Int<CTA_K>>;
  using ClusterShape = cute::Shape<int, int, _1>;
  using ElementType = cutlass::float_e2m1_t;
  using Arch = cutlass::arch::Sm100;

  using ElementA = ElementType;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = ElementType;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;

  using SFType = cutlass::float_ue8m0_t;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using EpilogueTileType = std::conditional_t<
      CTA_M == 128 && CTA_N == 256 && CTA_K == 256,
      cute::Shape<cute::_128, cute::_64>,
      cutlass::epilogue::collective::EpilogueTileAuto>;

  using EpilogueSchedule = typename MxSMTypeAdapter<XSM>::EpilogueSchedule;
  using MainloopSchedule = typename MxSMTypeAdapter<XSM>::MainloopSchedule;

  using MmaTileShape = cute::Shape<
      cute::Int<CTA_M * MxSMTypeAdapter<XSM>::Scale>,
      cute::Int<CTA_N>, cute::Int<CTA_K>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OperatorClass, MmaTileShape, ClusterShape, EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      OutElementType, LayoutC, AlignmentC,
      EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<OutElementType, float, void, float>
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassBlockScaledTensorOp,
      cute::tuple<ElementA, SFType>, LayoutA, AlignmentA,
      cute::tuple<ElementB, SFType>, LayoutB, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
  >::CollectiveOp;

  template <typename Base>
  struct Sm10x11xOnly : Base {
    using typename Base::Params;
    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {
#if defined(ENABLE_FP4_SM100)
        this->Base::operator()(params, smem_buf);
#else
        if (cute::thread0()) {
          printf("MXFP4 CUTLASS GEMM: requires SM10x/SM11x\n");
          __trap();
        }
#endif
    }
  };

  using GemmKernel = Sm10x11xOnly<
      cutlass::gemm::kernel::GemmUniversal<
          cute::Shape<int, int, int, int>,
          CollectiveMainloop, CollectiveEpilogue,
          cutlass::gemm::PersistentScheduler>>;

  using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

#endif // ENABLE_FP4_SM100

// ============================================================================
// SM120 (Blackwell consumer) MXFP4 Dense GEMM
// Uses KernelScheduleAuto, EpilogueScheduleAuto, 1x1x1 cluster, StageCount<2>
// Only compiled when targeting SM120+ (RTX 5090 etc.)
// ============================================================================

#if defined(ENABLE_FP4_SM120)

template <typename OutType, int CTA_M, int CTA_N, int CTA_K>
struct CutlassMxfp4GemmSm120 {
  using OutElementType = OutType;
  using CTAShape = cute::Shape<cute::Int<CTA_M>, cute::Int<CTA_N>, cute::Int<CTA_K>>;
  using ClusterShape = cute::Shape<_1, _1, _1>;
  using ElementType = cutlass::float_e2m1_t;
  using Arch = cutlass::arch::Sm120;

  using ElementA = ElementType;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = ElementType;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;

  using SFType = cutlass::float_ue8m0_t;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;

  using MmaTileShape = cute::Shape<cute::Int<CTA_M>, cute::Int<CTA_N>, cute::Int<CTA_K>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape, EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      OutElementType, LayoutC, AlignmentC,
      EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<OutElementType, float, void, float>
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OperatorClass,
      cute::tuple<ElementA, SFType>, LayoutA, AlignmentA,
      cute::tuple<ElementB, SFType>, LayoutB, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<2>,
      MainloopSchedule
  >::CollectiveOp;

  template <typename Base>
  struct Sm12xOnly : Base {
    using typename Base::Params;
    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {
#if defined(ENABLE_FP4_SM120)
        this->Base::operator()(params, smem_buf);
#else
        if (cute::thread0()) {
          printf("MXFP4 CUTLASS GEMM SM120: requires SM12x\n");
          __trap();
        }
#endif
    }
  };

  using GemmKernelPersistent = Sm12xOnly<
      cutlass::gemm::kernel::GemmUniversal<
          cute::Shape<int, int, int, int>,
          CollectiveMainloop, CollectiveEpilogue,
          cutlass::gemm::PersistentScheduler>>;

  using GemmKernelStreamK = Sm12xOnly<
      cutlass::gemm::kernel::GemmUniversal<
          cute::Shape<int, int, int, int>,
          CollectiveMainloop, CollectiveEpilogue,
          cutlass::gemm::StreamKScheduler>>;

  using GemmKernel = GemmKernelPersistent;
  using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using GemmStreamK = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStreamK>;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

#endif // ENABLE_FP4_SM120

// ============================================================================
// Kernel Launch (field-based argument construction for CUTLASS 4.4.2)
// ============================================================================

template <typename GemmOp, typename GemmAdapter = typename GemmOp::Gemm>
static void run_mxfp4_gemm(
    void* D, const void* A, const void* B,
    const void* input_sf, const void* weight_sf,
    int m, int n, int k,
    dim3 preferred_cluster, dim3 fallback_cluster,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream,
    const char* sched_name = "DP")
{
  using Gemm = GemmAdapter;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue8m0_t;
  using ElementSFB = cutlass::float_ue8m0_t;
  using Sm1xxBlkScaledConfig = typename GemmOp::Sm1xxBlkScaledConfig;

  typename Gemm::Arguments operator_args;
  operator_args.mode = cutlass::gemm::GemmUniversalMode::kGemm;

  operator_args.epilogue.thread.alpha_ptr = nullptr;
  operator_args.problem_shape = cute::make_shape(m, n, k, 1);

  operator_args.mainloop.ptr_A = static_cast<cutlass::float_e2m1_t const*>(A);
  operator_args.mainloop.ptr_B = static_cast<cutlass::float_e2m1_t const*>(B);
  operator_args.mainloop.ptr_SFA = static_cast<ElementSFA const*>(input_sf);
  operator_args.mainloop.ptr_SFB = static_cast<ElementSFB const*>(weight_sf);
  operator_args.epilogue.ptr_C = nullptr;
  operator_args.epilogue.ptr_D = static_cast<ElementD*>(D);

  operator_args.mainloop.dA =
      cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideA>(k, 0);
  operator_args.mainloop.dB =
      cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideB>(k, 0);
  operator_args.epilogue.dC =
      cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideC>(n, 0);
  operator_args.epilogue.dD = operator_args.epilogue.dC;

  operator_args.mainloop.layout_SFA =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(operator_args.problem_shape);
  operator_args.mainloop.layout_SFB =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(operator_args.problem_shape);

  if constexpr (!std::is_const_v<decltype(operator_args.scheduler.max_swizzle_size)>) {
    operator_args.scheduler.max_swizzle_size = 1;
  }

  operator_args.hw_info.cluster_shape = preferred_cluster;
  operator_args.hw_info.cluster_shape_fallback = fallback_cluster;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(operator_args);
  if (workspace_size > workspace_bytes) {
    fprintf(stderr, "[MXFP4 %s] workspace too small: need %zu, have %zu (M=%d N=%d K=%d)\n",
            sched_name, workspace_size, workspace_bytes, m, n, k);
    return;
  }
  void* ws = (workspace_size > 0) ? workspace : nullptr;

  auto can_impl = gemm.can_implement(operator_args);
  if (can_impl != cutlass::Status::kSuccess) {
    fprintf(stderr, "[MXFP4 %s] can_implement failed: %s (M=%d N=%d K=%d)\n",
            sched_name, cutlass::cutlassGetStatusString(can_impl), m, n, k);
    return;
  }

  auto run_status = gemm.run(operator_args, ws, stream, nullptr, /*launch_with_pdl=*/false);
  if (run_status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[MXFP4 %s] run failed: %s (M=%d N=%d K=%d ws=%zu)\n",
            sched_name, cutlass::cutlassGetStatusString(run_status), m, n, k, workspace_size);
  }
}

#if defined(ENABLE_FP4_SM100)
template <typename OutType>
static void dispatch_mxfp4_gemm_sm100(
    void* D, const void* A, const void* B,
    const void* input_sf, const void* weight_sf,
    int m, int n, int k,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream)
{
  if (m <= 128) {
    using GemmOp = CutlassMxfp4Gemm<OutType, 128, 256, 256, _1SM>;
    run_mxfp4_gemm<GemmOp>(D, A, B, input_sf, weight_sf, m, n, k,
                            dim3(1, 4, 1), dim3(1, 2, 1), workspace, workspace_bytes, stream);
  } else if (m <= 1024) {
    using GemmOp = CutlassMxfp4Gemm<OutType, 128, 256, 256, _2SM>;
    run_mxfp4_gemm<GemmOp>(D, A, B, input_sf, weight_sf, m, n, k,
                            dim3(2, 4, 1), dim3(2, 1, 1), workspace, workspace_bytes, stream);
  } else {
    using GemmOp = CutlassMxfp4Gemm<OutType, 128, 256, 256, _2SM>;
    run_mxfp4_gemm<GemmOp>(D, A, B, input_sf, weight_sf, m, n, k,
                            dim3(1, 4, 1), dim3(1, 2, 1), workspace, workspace_bytes, stream);
  }
}
#endif // ENABLE_FP4_SM100

#if defined(ENABLE_FP4_SM120)
template <typename OutType>
static void run_mxfp4_gemm_sm120(
    void* D, const void* A, const void* B,
    const void* input_sf, const void* weight_sf,
    int m, int n, int k,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream)
{
  using GemmOp = CutlassMxfp4GemmSm120<OutType, 128, 256, 256>;
  if (m < 128) {
    run_mxfp4_gemm<GemmOp, typename GemmOp::GemmStreamK>(
        D, A, B, input_sf, weight_sf, m, n, k,
        dim3(1, 1, 1), dim3(1, 1, 1), workspace, workspace_bytes, stream, "StreamK");
  } else {
    run_mxfp4_gemm<GemmOp>(
        D, A, B, input_sf, weight_sf, m, n, k,
        dim3(1, 1, 1), dim3(1, 1, 1), workspace, workspace_bytes, stream, "Persistent");
  }
}
#endif // ENABLE_FP4_SM120

// ============================================================================
// C API Entry Points
// ============================================================================

extern "C" {

void mxfp4_cutlass_gemm_f16(
    const void* input,
    const void* weight,
    const void* input_sf,
    const void* weight_sf,
    void* output,
    int M, int N, int K,
    void* workspace, int64_t workspace_bytes,
    int64_t stream)
{
  auto s = reinterpret_cast<cudaStream_t>(stream);
#if defined(ENABLE_FP4_SM120)
  run_mxfp4_gemm_sm120<cutlass::half_t>(
      output, input, weight, input_sf, weight_sf, M, N, K,
      workspace, static_cast<size_t>(workspace_bytes), s);
#elif defined(ENABLE_FP4_SM100)
  dispatch_mxfp4_gemm_sm100<cutlass::half_t>(
      output, input, weight, input_sf, weight_sf, M, N, K,
      workspace, static_cast<size_t>(workspace_bytes), s);
#endif
}

void mxfp4_cutlass_gemm_bf16(
    const void* input,
    const void* weight,
    const void* input_sf,
    const void* weight_sf,
    void* output,
    int M, int N, int K,
    void* workspace, int64_t workspace_bytes,
    int64_t stream)
{
  auto s = reinterpret_cast<cudaStream_t>(stream);
#if defined(ENABLE_FP4_SM120)
  run_mxfp4_gemm_sm120<cutlass::bfloat16_t>(
      output, input, weight, input_sf, weight_sf, M, N, K,
      workspace, static_cast<size_t>(workspace_bytes), s);
#elif defined(ENABLE_FP4_SM100)
  dispatch_mxfp4_gemm_sm100<cutlass::bfloat16_t>(
      output, input, weight, input_sf, weight_sf, M, N, K,
      workspace, static_cast<size_t>(workspace_bytes), s);
#endif
}

}  // extern "C"

#else  // !ENABLE_FP4

extern "C" {

void mxfp4_cutlass_gemm_f16(
    const void*, const void*, const void*, const void*,
    void*, int, int, int,
    void*, int64_t, int64_t)
{
}

void mxfp4_cutlass_gemm_bf16(
    const void*, const void*, const void*, const void*,
    void*, int, int, int,
    void*, int64_t, int64_t)
{
}

}  // extern "C"

#endif  // ENABLE_FP4
