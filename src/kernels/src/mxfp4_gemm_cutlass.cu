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
 * Based on FlashInfer MXFP4/MXFP8 CUTLASS templates.
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
// SM100 MXFP4 Dense GEMM Kernel Configurations
// ============================================================================

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

template <typename OutType, typename XSM>
struct Mxfp4GemmSm100Config {
  using ElementA = cutlass::float_e2m1_t;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::float_e2m1_t;
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
  using SFType = cutlass::float_ue8m0_t;
};

// Small M (M <= 128): 1SM, 128x256x256 tile, 1x4 cluster
template <typename OutType>
struct MxKernelConfigSmallM : Mxfp4GemmSm100Config<OutType, _1SM> {
  using MmaTileShape = Shape<_128, _256, _256>;
  using ClusterShapeType = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;
  using EpilogueSchedule = typename MxSMTypeAdapter<_1SM>::EpilogueSchedule;
  using MainloopSchedule = typename MxSMTypeAdapter<_1SM>::MainloopSchedule;
  static constexpr int MmaScale = MxSMTypeAdapter<_1SM>::Scale;
  static dim3 preferred_cluster() { return dim3(1, 4, 1); }
  static dim3 fallback_cluster() { return dim3(1, 2, 1); }
};

// Medium M (128 < M <= 1024): 2SM, 256x256x256 tile, 2x4 cluster
template <typename OutType>
struct MxKernelConfigMediumM : Mxfp4GemmSm100Config<OutType, _2SM> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShapeType = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;
  using EpilogueSchedule = typename MxSMTypeAdapter<_2SM>::EpilogueSchedule;
  using MainloopSchedule = typename MxSMTypeAdapter<_2SM>::MainloopSchedule;
  static constexpr int MmaScale = MxSMTypeAdapter<_2SM>::Scale;
  static dim3 preferred_cluster() { return dim3(2, 4, 1); }
  static dim3 fallback_cluster() { return dim3(2, 1, 1); }
};

// Large M (M > 1024): 2SM, 256x256x256 tile, 1x4 cluster
template <typename OutType>
struct MxKernelConfigLargeM : Mxfp4GemmSm100Config<OutType, _2SM> {
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShapeType = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;
  using EpilogueSchedule = typename MxSMTypeAdapter<_2SM>::EpilogueSchedule;
  using MainloopSchedule = typename MxSMTypeAdapter<_2SM>::MainloopSchedule;
  static constexpr int MmaScale = MxSMTypeAdapter<_2SM>::Scale;
  static dim3 preferred_cluster() { return dim3(1, 4, 1); }
  static dim3 fallback_cluster() { return dim3(1, 2, 1); }
};

// ============================================================================
// CUTLASS GEMM Instantiation Template
// ============================================================================

template <typename Config>
struct CutlassMxfp4Gemm {
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
      EpilogueSchedule
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      cute::tuple<ElementA, SFType>, LayoutA, Config::AlignmentA,
      cute::tuple<ElementB, SFType>, LayoutB, Config::AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
  >::CollectiveOp;

  template <typename Base>
  struct Sm10xOnly : Base {
    using typename Base::Params;
    CUTLASS_DEVICE
    void operator()(Params const& params, char* smem_buf) {
      if constexpr (flashinfer::arch::is_major_v<10> || flashinfer::arch::is_major_v<11>) {
        this->Base::operator()(params, smem_buf);
      } else {
        if (cute::thread0()) {
          printf("MXFP4 CUTLASS GEMM: requires SM10x/SM11x\n");
          __trap();
        }
      }
    }
  };

  using GemmKernel = Sm10xOnly<
      cutlass::gemm::kernel::GemmUniversal<
          Shape<int, int, int, int>,
          CollectiveMainloop, CollectiveEpilogue,
          cutlass::gemm::PersistentScheduler>>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

// ============================================================================
// Kernel Launch
// ============================================================================

template <typename Config>
static void run_mxfp4_gemm(
    void* D, const void* A, const void* B,
    const void* input_sf, const void* weight_sf,
    int m, int n, int k,
    cudaStream_t stream)
{
  using GemmOp = CutlassMxfp4Gemm<Config>;
  using Gemm = typename GemmOp::Gemm;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue8m0_t;
  using ElementSFB = cutlass::float_ue8m0_t;
  using ElementCompute = float;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig = typename GemmOp::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {
          static_cast<ElementA const*>(A),
          stride_A,
          static_cast<ElementB const*>(B),
          stride_B,
          static_cast<ElementSFA const*>(input_sf),
          layout_SFA,
          static_cast<ElementSFB const*>(weight_sf),
          layout_SFB
      },
      {
          {},
          nullptr,
          stride_D,
          static_cast<ElementD*>(D),
          stride_D
      }
  };

  // MXFP4 has no global scale factor (unlike NVFP4)
  // The epilogue uses default alpha=1.0, beta=0.0

  arguments.hw_info.cluster_shape = Config::preferred_cluster();
  arguments.hw_info.cluster_shape_fallback = Config::fallback_cluster();

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaMallocAsync(&workspace, workspace_size, stream);
  }

  auto can_impl = gemm.can_implement(arguments);
  if (can_impl != cutlass::Status::kSuccess) {
    if (workspace) cudaFreeAsync(workspace, stream);
    fprintf(stderr, "[MXFP4 CUTLASS GEMM] can_implement failed: %s\n",
            cutlassGetStatusString(can_impl));
    return;
  }

  auto init_status = gemm.initialize(arguments, workspace, stream);
  if (init_status != cutlass::Status::kSuccess) {
    if (workspace) cudaFreeAsync(workspace, stream);
    fprintf(stderr, "[MXFP4 CUTLASS GEMM] initialize failed: %s\n",
            cutlassGetStatusString(init_status));
    return;
  }

  auto run_status = gemm.run(arguments, workspace, stream, nullptr, true);
  if (run_status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[MXFP4 CUTLASS GEMM] run failed: %s\n",
            cutlassGetStatusString(run_status));
  }

  if (workspace) cudaFreeAsync(workspace, stream);
}

// M-bucketed dispatch
template <typename OutType>
static void dispatch_mxfp4_gemm_sm100(
    void* D, const void* A, const void* B,
    const void* input_sf, const void* weight_sf,
    int m, int n, int k,
    cudaStream_t stream)
{
  if (m <= 128) {
    run_mxfp4_gemm<MxKernelConfigSmallM<OutType>>(
        D, A, B, input_sf, weight_sf, m, n, k, stream);
  } else if (m <= 1024) {
    run_mxfp4_gemm<MxKernelConfigMediumM<OutType>>(
        D, A, B, input_sf, weight_sf, m, n, k, stream);
  } else {
    run_mxfp4_gemm<MxKernelConfigLargeM<OutType>>(
        D, A, B, input_sf, weight_sf, m, n, k, stream);
  }
}

// ============================================================================
// C API Entry Points
// ============================================================================

extern "C" {

void mxfp4_cutlass_gemm_f16(
    const void* input,       // [M, K/2] packed FP4 activations (uint8)
    const void* weight,      // [N, K/2] packed FP4 weights (uint8)
    const void* input_sf,    // input block scales (E8M0, swizzled)
    const void* weight_sf,   // weight block scales (E8M0, swizzled)
    void* output,            // [M, N] output (FP16)
    int M, int N, int K,
    int64_t stream)
{
  dispatch_mxfp4_gemm_sm100<cutlass::half_t>(
      output, input, weight, input_sf, weight_sf,
      M, N, K, reinterpret_cast<cudaStream_t>(stream));
}

void mxfp4_cutlass_gemm_bf16(
    const void* input,
    const void* weight,
    const void* input_sf,
    const void* weight_sf,
    void* output,
    int M, int N, int K,
    int64_t stream)
{
  dispatch_mxfp4_gemm_sm100<cutlass::bfloat16_t>(
      output, input, weight, input_sf, weight_sf,
      M, N, K, reinterpret_cast<cudaStream_t>(stream));
}

}  // extern "C"

#else  // !ENABLE_FP4

extern "C" {

void mxfp4_cutlass_gemm_f16(
    const void*, const void*, const void*, const void*,
    void*, int, int, int, int64_t)
{
}

void mxfp4_cutlass_gemm_bf16(
    const void*, const void*, const void*, const void*,
    void*, int, int, int, int64_t)
{
}

}  // extern "C"

#endif  // ENABLE_FP4
