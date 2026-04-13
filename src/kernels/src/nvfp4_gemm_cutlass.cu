/*
 * Hardware-accelerated NVFP4 GEMM using CUTLASS block-scaled tensor ops.
 * Targets Blackwell (SM100+) with native FP4 tensor core support.
 * Falls back gracefully: on SM < 100, the caller uses software dequant kernels instead.
 *
 * Based on FlashInfer/SGLang CUTLASS FP4 GEMM implementations.
 * Requires CUTLASS 3.x with SM100 block-scaled tensor op support.
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
#include "flashinfer/cutlass_utils.cuh"
#include "flashinfer/arch_condition.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

using namespace cute;

// ============================================================================
// SM100 NVFP4 Dense GEMM Kernel Configurations
// ============================================================================

#if defined(ENABLE_FP4_SM100)

struct _1SM {};
struct _2SM {};

template <typename T>
struct SMTypeAdapter {};

template <>
struct SMTypeAdapter<_1SM> {
  static int const Scale = 1;
  using AtomThrShape = cute::Shape<_1, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
};

template <>
struct SMTypeAdapter<_2SM> {
  static int const Scale = 2;
  using AtomThrShape = cute::Shape<_2, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
};

template <typename OutType, typename XSM>
struct Fp4GemmSm100Config {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using SFType = cutlass::float_ue4m3_t;
};

template <typename OutType>
struct KernelConfigSmallM : Fp4GemmSm100Config<OutType, _1SM> {
  using MmaTileShape = Shape<_128, _256, _256>;
  using ClusterShapeType = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;
  using EpilogueSchedule = typename SMTypeAdapter<_1SM>::EpilogueSchedule;
  using MainloopSchedule = typename SMTypeAdapter<_1SM>::MainloopSchedule;
  static constexpr int MmaScale = SMTypeAdapter<_1SM>::Scale;
  static dim3 preferred_cluster() { return dim3(1, 4, 1); }
  static dim3 fallback_cluster() { return dim3(1, 2, 1); }
};

template <typename OutType>
struct KernelConfigMediumM : Fp4GemmSm100Config<OutType, _2SM> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShapeType = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;
  using EpilogueSchedule = typename SMTypeAdapter<_2SM>::EpilogueSchedule;
  using MainloopSchedule = typename SMTypeAdapter<_2SM>::MainloopSchedule;
  static constexpr int MmaScale = SMTypeAdapter<_2SM>::Scale;
  static dim3 preferred_cluster() { return dim3(2, 4, 1); }
  static dim3 fallback_cluster() { return dim3(2, 1, 1); }
};

template <typename OutType>
struct KernelConfigLargeM : Fp4GemmSm100Config<OutType, _2SM> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShapeType = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;
  using EpilogueSchedule = typename SMTypeAdapter<_2SM>::EpilogueSchedule;
  using MainloopSchedule = typename SMTypeAdapter<_2SM>::MainloopSchedule;
  static constexpr int MmaScale = SMTypeAdapter<_2SM>::Scale;
  static dim3 preferred_cluster() { return dim3(1, 4, 1); }
  static dim3 fallback_cluster() { return dim3(1, 2, 1); }
};

#endif // ENABLE_FP4_SM100

// ============================================================================
// SM120 (Blackwell) NVFP4 Dense GEMM Configuration
// Uses KernelScheduleAuto, EpilogueScheduleAuto, 1x1x1 cluster, void/StreamK scheduler
// Based on FlashInfer's fp4_gemm_template_sm120.h / CUTLASS example 79
// ============================================================================

#if defined(ENABLE_FP4_SM120)

template <typename OutType, typename TileShape>
struct Fp4GemmSm120ConfigBase {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using SFType = cutlass::float_ue4m3_t;

  using MmaTileShape = TileShape;
  using ClusterShapeType = Shape<_1, _1, _1>;
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
};

template <typename OutType>
struct Fp4GemmSm120ConfigSmallM
    : Fp4GemmSm120ConfigBase<OutType, Shape<_128, _128, _256>> {};

template <typename OutType>
struct Fp4GemmSm120ConfigM256
    : Fp4GemmSm120ConfigBase<OutType, Shape<_128, _128, _128>> {};

template <typename OutType>
struct Fp4GemmSm120ConfigDefault
    : Fp4GemmSm120ConfigBase<OutType, Shape<_256, _128, _128>> {};

#endif // ENABLE_FP4_SM120

// ============================================================================
// CUTLASS GEMM Instantiation Template (SM100)
// ============================================================================

#if defined(ENABLE_FP4_SM100)

template <typename Config>
struct CutlassFp4Gemm {
  using ElementA = typename Config::ElementA;
  using LayoutA = typename Config::LayoutATag;
  using ElementB = typename Config::ElementB;
  using LayoutB = typename Config::LayoutBTag;
  using ElementD = typename Config::ElementD;
  using LayoutD = typename Config::LayoutDTag;
  using ElementC = typename Config::ElementC;
  using LayoutC = typename Config::LayoutCTag;
  using ElementAccumulator = typename Config::ElementAccumulator;
  using ElementCompute = typename Config::ElementCompute;
  using SFType = typename Config::SFType;
  using ArchTag = typename Config::ArchTag;
  using OperatorClass = typename Config::OperatorClass;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShapeType;
  using EpilogueTile = typename Config::EpilogueTile;
  using EpilogueSchedule = typename Config::EpilogueSchedule;
  using MainloopSchedule = typename Config::MainloopSchedule;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape, EpilogueTile,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, Config::AlignmentC,
      ElementD, LayoutD, Config::AlignmentD,
      EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, Config::AlignmentA,
      ElementB, LayoutB, Config::AlignmentB,
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
      if constexpr (flashinfer::arch::is_major_v<10> || flashinfer::arch::is_major_v<11>) {
        this->Base::operator()(params, smem_buf);
      } else {
        if (cute::thread0()) {
          printf("NVFP4 CUTLASS GEMM SM100: requires SM10x/SM11x\n");
          __trap();
        }
      }
    }
  };

  using GemmKernel = Sm10x11xOnly<
      cutlass::gemm::kernel::GemmUniversal<
          Shape<int, int, int, int>,
          CollectiveMainloop, CollectiveEpilogue,
          cutlass::gemm::PersistentScheduler>>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

#endif // ENABLE_FP4_SM100

// ============================================================================
// CUTLASS GEMM Instantiation Template (SM120 / Blackwell)
// ============================================================================

#if defined(ENABLE_FP4_SM120)

template <typename Config>
struct CutlassFp4GemmSm120 {
  using ElementA = typename Config::ElementA;
  using LayoutA = typename Config::LayoutATag;
  using ElementB = typename Config::ElementB;
  using LayoutB = typename Config::LayoutBTag;
  using ElementD = typename Config::ElementD;
  using LayoutD = typename Config::LayoutDTag;
  using ElementC = typename Config::ElementC;
  using LayoutC = typename Config::LayoutCTag;
  using ElementAccumulator = typename Config::ElementAccumulator;
  using ElementCompute = typename Config::ElementCompute;
  using ArchTag = typename Config::ArchTag;
  using OperatorClass = typename Config::OperatorClass;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShapeType;
  using EpilogueTile = typename Config::EpilogueTile;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape, EpilogueTile,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, Config::AlignmentC,
      ElementD, LayoutD, Config::AlignmentD,
      typename Config::EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, Config::AlignmentA,
      ElementB, LayoutB, Config::AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<2>,
      typename Config::MainloopSchedule
  >::CollectiveOp;

  using GemmKernelDP = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop, CollectiveEpilogue, void>;

  using GemmKernelStreamK = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelDP>;
  using GemmStreamK = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelStreamK>;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

#endif // ENABLE_FP4_SM120

// ============================================================================
// Kernel Launch
// ============================================================================

#if defined(ENABLE_FP4_SM100)

template <typename Config>
static void run_fp4_gemm(
    void* D, const void* A, const void* B,
    const void* input_sf, const void* weight_sf,
    const float* global_sf,
    int m, int n, int k,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream)
{
  using GemmOp = CutlassFp4Gemm<Config>;
  using Gemm = typename GemmOp::Gemm;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;
  using Sm1xxBlkScaledConfig = typename GemmOp::Sm1xxBlkScaledConfig;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  typename Gemm::Arguments operator_args;
  operator_args.mode = cutlass::gemm::GemmUniversalMode::kGemm;

  operator_args.epilogue.thread.alpha_ptr = static_cast<ElementCompute const*>(global_sf);
  operator_args.problem_shape = cute::make_shape(m, n, k, 1);

  operator_args.mainloop.ptr_A = static_cast<cutlass::float_e2m1_t const*>(A);
  operator_args.mainloop.ptr_B = static_cast<cutlass::float_e2m1_t const*>(B);
  operator_args.mainloop.ptr_SFA = static_cast<ElementSFA const*>(input_sf);
  operator_args.mainloop.ptr_SFB = static_cast<ElementSFB const*>(weight_sf);
  operator_args.epilogue.ptr_C = nullptr;
  operator_args.epilogue.ptr_D = static_cast<ElementD*>(D);

  operator_args.mainloop.dA =
      cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  operator_args.mainloop.dB =
      cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  operator_args.epilogue.dC =
      cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  operator_args.epilogue.dD = operator_args.epilogue.dC;

  operator_args.mainloop.layout_SFA =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(operator_args.problem_shape);
  operator_args.mainloop.layout_SFB =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(operator_args.problem_shape);

  if constexpr (!std::is_const_v<decltype(operator_args.scheduler.max_swizzle_size)>) {
    operator_args.scheduler.max_swizzle_size = 1;
  }

  operator_args.hw_info.cluster_shape = Config::preferred_cluster();
  operator_args.hw_info.cluster_shape_fallback = Config::fallback_cluster();

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(operator_args);
  if (workspace_size > workspace_bytes) {
    fprintf(stderr, "[NVFP4 SM100] workspace too small: need %zu, have %zu (M=%d N=%d K=%d)\n",
            workspace_size, workspace_bytes, m, n, k);
    return;
  }
  void* ws = (workspace_size > 0) ? workspace : nullptr;

  auto can_impl = gemm.can_implement(operator_args);
  if (can_impl != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 SM100] can_implement failed: %s (M=%d N=%d K=%d)\n",
            cutlass::cutlassGetStatusString(can_impl), m, n, k);
    return;
  }

  auto status = gemm.initialize(operator_args, ws);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 GEMM CUTLASS] initialize failed: %s\n",
            cutlass::cutlassGetStatusString(status));
  }
  auto run_status = gemm.run(operator_args, ws, stream);
  if (run_status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 SM100] run failed: %s (M=%d N=%d K=%d ws=%zu)\n",
            cutlass::cutlassGetStatusString(run_status), m, n, k, workspace_size);
  }
}

template <typename OutType>
static void dispatch_fp4_gemm_sm100(
    void* D, const void* A, const void* B,
    const void* input_sf, const void* weight_sf,
    const float* global_sf,
    int m, int n, int k,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream)
{
  if (m <= 128) {
    run_fp4_gemm<KernelConfigSmallM<OutType>>(
        D, A, B, input_sf, weight_sf, global_sf, m, n, k, workspace, workspace_bytes, stream);
  } else if (m <= 1024) {
    run_fp4_gemm<KernelConfigMediumM<OutType>>(
        D, A, B, input_sf, weight_sf, global_sf, m, n, k, workspace, workspace_bytes, stream);
  } else {
    run_fp4_gemm<KernelConfigLargeM<OutType>>(
        D, A, B, input_sf, weight_sf, global_sf, m, n, k, workspace, workspace_bytes, stream);
  }
}

#endif // ENABLE_FP4_SM100

#if defined(ENABLE_FP4_SM120)

template <typename Gemm, typename OutType>
static void run_fp4_gemm_sm120_impl(
    void* D, const void* A, const void* B,
    const void* input_sf, const void* weight_sf,
    const float* global_sf,
    int m, int n, int k,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream,
    const char* sched_name)
{
  using ElementD = OutType;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  typename Gemm::Arguments operator_args;
  operator_args.mode = cutlass::gemm::GemmUniversalMode::kGemm;

  operator_args.epilogue.thread.alpha_ptr = static_cast<ElementCompute const*>(global_sf);
  operator_args.problem_shape = cute::make_shape(m, n, k, 1);

  operator_args.mainloop.ptr_A = static_cast<cutlass::float_e2m1_t const*>(A);
  operator_args.mainloop.ptr_B = static_cast<cutlass::float_e2m1_t const*>(B);
  operator_args.mainloop.ptr_SFA = static_cast<ElementSFA const*>(input_sf);
  operator_args.mainloop.ptr_SFB = static_cast<ElementSFB const*>(weight_sf);
  operator_args.epilogue.ptr_C = nullptr;
  operator_args.epilogue.ptr_D = static_cast<ElementD*>(D);

  operator_args.mainloop.dA =
      cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  operator_args.mainloop.dB =
      cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  operator_args.epilogue.dC =
      cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  operator_args.epilogue.dD = operator_args.epilogue.dC;

  operator_args.mainloop.layout_SFA =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(operator_args.problem_shape);
  operator_args.mainloop.layout_SFB =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(operator_args.problem_shape);

  if constexpr (!std::is_const_v<decltype(operator_args.scheduler.max_swizzle_size)>) {
    operator_args.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(operator_args.scheduler.raster_order)>) {
    using Enum_t = decltype(operator_args.scheduler.raster_order);
    operator_args.scheduler.raster_order = Enum_t::Heuristic;
  }
  operator_args.hw_info.cluster_shape = dim3(1, 1, 1);
  operator_args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(operator_args);
  if (workspace_size > workspace_bytes) {
    fprintf(stderr, "[NVFP4 SM120 %s] workspace too small: need %zu, have %zu (M=%d N=%d K=%d)\n",
            sched_name, workspace_size, workspace_bytes, m, n, k);
    return;
  }
  void* ws = (workspace_size > 0) ? workspace : nullptr;

  auto can_impl = gemm.can_implement(operator_args);
  if (can_impl != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 SM120 %s] can_implement failed: %s (M=%d N=%d K=%d)\n",
            sched_name, cutlass::cutlassGetStatusString(can_impl), m, n, k);
    return;
  }

  auto status = gemm.initialize(operator_args, ws, stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 GEMM CUTLASS] initialize failed: %s\n",
            cutlass::cutlassGetStatusString(status));
  }

  auto run_status = gemm.run(operator_args, ws, stream, nullptr, /*enable_pdl=*/true);
  if (run_status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[NVFP4 SM120 %s] run failed: %s (M=%d N=%d K=%d ws=%zu)\n",
            sched_name, cutlass::cutlassGetStatusString(run_status), m, n, k, workspace_size);
  }
}

template <typename OutType>
static void run_fp4_gemm_sm120(
    void* D, const void* A, const void* B,
    const void* input_sf, const void* weight_sf,
    const float* global_sf,
    int m, int n, int k,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream)
{
  auto next_pow2 = [](int v) {
    unsigned x = static_cast<unsigned>(v <= 1 ? 1 : v - 1);
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return static_cast<int>(x + 1);
  };

  int mp2 = next_pow2(m < 16 ? 16 : m);
  if (mp2 <= 32) {
    using GemmOp = CutlassFp4GemmSm120<Fp4GemmSm120ConfigSmallM<OutType>>;
    using Gemm = typename GemmOp::Gemm;
    run_fp4_gemm_sm120_impl<Gemm, OutType>(
        D, A, B, input_sf, weight_sf, global_sf, m, n, k,
        workspace, workspace_bytes, stream, "SmallM");
  } else if (mp2 <= 256) {
    using GemmOp = CutlassFp4GemmSm120<Fp4GemmSm120ConfigM256<OutType>>;
    using Gemm = typename GemmOp::Gemm;
    run_fp4_gemm_sm120_impl<Gemm, OutType>(
        D, A, B, input_sf, weight_sf, global_sf, m, n, k,
        workspace, workspace_bytes, stream, "M256");
  } else {
    using GemmOp = CutlassFp4GemmSm120<Fp4GemmSm120ConfigDefault<OutType>>;
    using Gemm = typename GemmOp::Gemm;
    run_fp4_gemm_sm120_impl<Gemm, OutType>(
        D, A, B, input_sf, weight_sf, global_sf, m, n, k,
        workspace, workspace_bytes, stream, "Default");
  }
}

#endif // ENABLE_FP4_SM120

// ============================================================================
// C API Entry Points
// ============================================================================

extern "C" {

void nvfp4_cutlass_gemm_f16(
    const void* input,
    const void* weight,
    const void* input_sf,
    const void* weight_sf,
    const float* global_sf,
    void* output,
    int M, int N, int K,
    void* workspace, int64_t workspace_bytes,
    int64_t stream)
{
  auto s = reinterpret_cast<cudaStream_t>(stream);
#if defined(ENABLE_FP4_SM120)
  run_fp4_gemm_sm120<cutlass::half_t>(
      output, input, weight, input_sf, weight_sf, global_sf, M, N, K,
      workspace, static_cast<size_t>(workspace_bytes), s);
#elif defined(ENABLE_FP4_SM100)
  dispatch_fp4_gemm_sm100<cutlass::half_t>(
      output, input, weight, input_sf, weight_sf, global_sf, M, N, K,
      workspace, static_cast<size_t>(workspace_bytes), s);
#endif
}

void nvfp4_cutlass_gemm_bf16(
    const void* input,
    const void* weight,
    const void* input_sf,
    const void* weight_sf,
    const float* global_sf,
    void* output,
    int M, int N, int K,
    void* workspace, int64_t workspace_bytes,
    int64_t stream)
{
  auto s = reinterpret_cast<cudaStream_t>(stream);
#if defined(ENABLE_FP4_SM120)
  run_fp4_gemm_sm120<cutlass::bfloat16_t>(
      output, input, weight, input_sf, weight_sf, global_sf, M, N, K,
      workspace, static_cast<size_t>(workspace_bytes), s);
#elif defined(ENABLE_FP4_SM100)
  dispatch_fp4_gemm_sm100<cutlass::bfloat16_t>(
      output, input, weight, input_sf, weight_sf, global_sf, M, N, K,
      workspace, static_cast<size_t>(workspace_bytes), s);
#endif
}

}  // extern "C"

#else  // !ENABLE_FP4

extern "C" {

void nvfp4_cutlass_gemm_f16(
    const void*, const void*, const void*, const void*,
    const float*, void*, int, int, int,
    void*, int64_t, int64_t)
{
}

void nvfp4_cutlass_gemm_bf16(
    const void*, const void*, const void*, const void*,
    const float*, void*, int, int, int,
    void*, int64_t, int64_t)
{
}

}  // extern "C"

#endif  // ENABLE_FP4
