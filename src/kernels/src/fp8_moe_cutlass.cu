#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <type_traits>

#ifndef NO_HARDWARE_FP8
  #include "cuda_fp8.h"
#endif
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
#endif

namespace vllm_rs_moe {

// ---- Fused quantize-gather kernels (require hardware FP8, SM89+) -------------
#if !defined(NO_HARDWARE_FP8)

static constexpr float kFp8MoeMax = 448.0f;
static constexpr float kFp8MoeMin = -448.0f;
static constexpr float kAbsmaxEps = 1e-10f;

__device__ __forceinline__ int4 moe_ld_nc(const int4* ptr) {
    int4 ret;
    asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void moe_st_global(int4* ptr, const int4& v) {
    asm volatile(
        "st.global.v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr),
        "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

template <int WIDTH>
__device__ __forceinline__ float moe_subwarp_max(float val) {
    static_assert(WIDTH <= 16 && WIDTH >= 1);
    if constexpr (WIDTH >= 16) val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    if constexpr (WIDTH >= 8)  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    if constexpr (WIDTH >= 4)  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    if constexpr (WIDTH >= 2)  val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

// ---- Fused quantize-and-gather kernel ----------------------------------------
// Reads input via index map, quantizes to FP8, writes directly to gathered output.
// Fuses fp8_quantize + gather_u8 + gather_f32 into one launch.

template <typename T, bool IS_COLUMN_MAJOR>
__global__ void fused_quantize_gather_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ dst2src_map,
    __nv_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    int num_src_rows,
    int num_dst_rows,
    int K,
    int num_groups_per_row,
    int scale_stride,
    int map_divisor) {

    constexpr int GROUP_SIZE = 128;
    constexpr int THREADS_PER_SUBWARP = 8;
    constexpr int VEC_SIZE = 32 / sizeof(T);
    static_assert(THREADS_PER_SUBWARP * VEC_SIZE == GROUP_SIZE);

    const int subwarps_per_block = blockDim.x / THREADS_PER_SUBWARP;
    const int subwarp_id = threadIdx.x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx.x % THREADS_PER_SUBWARP;

    const int total_groups = num_dst_rows * num_groups_per_row;
    const int64_t group_id = static_cast<int64_t>(blockIdx.x) * subwarps_per_block + subwarp_id;
    if (group_id >= total_groups) return;

    const int dst_row = group_id / num_groups_per_row;
    const int group_within_row = group_id % num_groups_per_row;

    int src_row = dst2src_map[dst_row] / map_divisor;
    if (src_row >= num_src_rows) return;

    const int64_t src_offset = static_cast<int64_t>(src_row) * K +
                               group_within_row * GROUP_SIZE + lane_id * VEC_SIZE;

    int4 input_vec[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        input_vec[i] = moe_ld_nc(reinterpret_cast<const int4*>(input + src_offset) + i);
    T* data = reinterpret_cast<T*>(input_vec);

    float local_absmax = kAbsmaxEps;
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
        float val;
        if constexpr (std::is_same_v<T, __half>)
            val = __half2float(data[j]);
        else
            val = __bfloat162float(data[j]);
        local_absmax = fmaxf(local_absmax, fabsf(val));
    }

    local_absmax = moe_subwarp_max<THREADS_PER_SUBWARP>(local_absmax);

    float scale_inv = local_absmax / kFp8MoeMax;
    float scale = kFp8MoeMax / local_absmax;

    if (lane_id == 0) {
        if constexpr (IS_COLUMN_MAJOR)
            output_s[group_within_row * scale_stride + dst_row] = scale_inv;
        else
            output_s[static_cast<int64_t>(dst_row) * num_groups_per_row + group_within_row] = scale_inv;
    }

    const int64_t dst_offset = static_cast<int64_t>(dst_row) * K +
                               group_within_row * GROUP_SIZE + lane_id * VEC_SIZE;
    int4 output_buf;
    __nv_fp8x2_storage_t* out_fp8x2 = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);

    #pragma unroll
    for (int j = 0; j < VEC_SIZE; j += 2) {
        float2 vals;
        if constexpr (std::is_same_v<T, __half>) {
            vals.x = __half2float(data[j]);
            vals.y = __half2float(data[j + 1]);
        } else {
            vals.x = __bfloat162float(data[j]);
            vals.y = __bfloat162float(data[j + 1]);
        }
        vals.x = fminf(fmaxf(vals.x * scale, kFp8MoeMin), kFp8MoeMax);
        vals.y = fminf(fmaxf(vals.y * scale, kFp8MoeMin), kFp8MoeMax);
        out_fp8x2[j / 2] = __nv_cvt_float2_to_fp8x2(vals, __NV_SATFINITE, __NV_E4M3);
    }

    moe_st_global(reinterpret_cast<int4*>(output_q + dst_offset), output_buf);
}

// ---- Fused quantize-gather + expert offsets for small M (<=128) ---------------

template <typename T, bool IS_COLUMN_MAJOR>
__global__ void fused_quantize_gather_offsets_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ dst2src_map,
    const int32_t* __restrict__ expert_ids,
    __nv_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    int num_src_rows,
    int num_dst_rows,
    int K,
    int num_groups_per_row,
    int scale_stride,
    int map_divisor,
    int num_experts) {

    constexpr int GROUP_SIZE = 128;
    constexpr int THREADS_PER_SUBWARP = 8;
    constexpr int VEC_SIZE = 32 / sizeof(T);
    static_assert(THREADS_PER_SUBWARP * VEC_SIZE == GROUP_SIZE);

    extern __shared__ int32_t smem[];

    const int subwarps_per_block = blockDim.x / THREADS_PER_SUBWARP;
    const int subwarp_id = threadIdx.x / THREADS_PER_SUBWARP;
    const int lane_id = threadIdx.x % THREADS_PER_SUBWARP;

    if (blockIdx.x == 0) {
        for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
            smem[e] = 0;
        __syncthreads();

        for (int i = threadIdx.x; i < num_dst_rows; i += blockDim.x) {
            int32_t eid = expert_ids[i];
            if (eid >= 0 && eid < num_experts)
                atomicAdd(&smem[eid], 1);
        }
        __syncthreads();

        if (threadIdx.x < num_experts)
            expert_counts[threadIdx.x] = smem[threadIdx.x];

        if (threadIdx.x == 0) {
            int running = 0;
            for (int e = 0; e < num_experts; e++) {
                int c = smem[e];
                expert_offsets[e] = running;
                running += c;
            }
            expert_offsets[num_experts] = running;
        }
        __syncthreads();
    }

    const int total_groups = num_dst_rows * num_groups_per_row;
    const int64_t group_id = static_cast<int64_t>(blockIdx.x) * subwarps_per_block + subwarp_id;
    if (group_id >= total_groups) return;

    const int dst_row = group_id / num_groups_per_row;
    const int group_within_row = group_id % num_groups_per_row;

    int src_row = dst2src_map[dst_row] / map_divisor;
    if (src_row >= num_src_rows) return;

    const int64_t src_offset = static_cast<int64_t>(src_row) * K +
                               group_within_row * GROUP_SIZE + lane_id * VEC_SIZE;

    int4 input_vec[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        input_vec[i] = moe_ld_nc(reinterpret_cast<const int4*>(input + src_offset) + i);
    T* data = reinterpret_cast<T*>(input_vec);

    float local_absmax = kAbsmaxEps;
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
        float val;
        if constexpr (std::is_same_v<T, __half>)
            val = __half2float(data[j]);
        else
            val = __bfloat162float(data[j]);
        local_absmax = fmaxf(local_absmax, fabsf(val));
    }

    local_absmax = moe_subwarp_max<THREADS_PER_SUBWARP>(local_absmax);

    float scale_inv = local_absmax / kFp8MoeMax;
    float scale = kFp8MoeMax / local_absmax;

    if (lane_id == 0) {
        if constexpr (IS_COLUMN_MAJOR)
            output_s[group_within_row * scale_stride + dst_row] = scale_inv;
        else
            output_s[static_cast<int64_t>(dst_row) * num_groups_per_row + group_within_row] = scale_inv;
    }

    const int64_t dst_offset = static_cast<int64_t>(dst_row) * K +
                               group_within_row * GROUP_SIZE + lane_id * VEC_SIZE;
    int4 output_buf;
    __nv_fp8x2_storage_t* out_fp8x2 = reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf);

    #pragma unroll
    for (int j = 0; j < VEC_SIZE; j += 2) {
        float2 vals;
        if constexpr (std::is_same_v<T, __half>) {
            vals.x = __half2float(data[j]);
            vals.y = __half2float(data[j + 1]);
        } else {
            vals.x = __bfloat162float(data[j]);
            vals.y = __bfloat162float(data[j + 1]);
        }
        vals.x = fminf(fmaxf(vals.x * scale, kFp8MoeMin), kFp8MoeMax);
        vals.y = fminf(fmaxf(vals.y * scale, kFp8MoeMin), kFp8MoeMax);
        out_fp8x2[j / 2] = __nv_cvt_float2_to_fp8x2(vals, __NV_SATFINITE, __NV_E4M3);
    }

    moe_st_global(reinterpret_cast<int4*>(output_q + dst_offset), output_buf);
}

#endif  // !NO_HARDWARE_FP8

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

template <typename T, int BLOCK_SIZE = 256>
__global__ void gather_rows_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ dst2src_map,
    T* __restrict__ output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    int32_t map_divisor) {
  int64_t dst_row = blockIdx.x;
  if (dst_row >= num_dst_rows) return;

  int64_t src_row = dst2src_map[dst_row] / map_divisor;
  if (src_row >= num_src_rows) return;

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
    for (int64_t i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
      dst_vec[i] = src_vec[i];
    }
  } else {
    for (int64_t i = threadIdx.x; i < num_cols; i += BLOCK_SIZE) {
      output[dst_row * num_cols + i] = input[src_row * num_cols + i];
    }
  }
}

// Strided gather kernel for column-major scale tensors (SM100+ Blackwell)
// Source is column-major: elements in each row are strided by src_row_stride
// Destination is also column-major: elements in each row are strided by dst_row_stride
template <typename T>
__global__ void gather_rows_strided_kernel(
    const T* input,
    const int32_t* dst2src_map,
    T* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    int64_t src_row_stride,  // Stride between elements in the same row of source
    int64_t dst_row_stride,  // Stride between elements in the same row of dest
    int32_t map_divisor) {
  int64_t dst_row = blockIdx.x;
  if (dst_row >= num_dst_rows) {
    return;
  }
  int64_t src_row = dst2src_map[dst_row] / map_divisor;
  if (src_row >= num_src_rows) {
    return;
  }

  // For column-major: input[row, col] = input[row + col * src_row_stride]
  // Copy element by element since rows are not contiguous
  for (int64_t col = threadIdx.x; col < num_cols; col += blockDim.x) {
    output[dst_row + col * dst_row_stride] = input[src_row + col * src_row_stride];
  }
}

// Optimized scatter kernel with:
// - Adaptive block size (512 threads for large N, 256 for smaller)
// - Multiple rows per block for small M to improve occupancy
// - Vectorized read-modify-write with scale

template <typename T, int BLOCK_SIZE = 256>
__global__ void scatter_rows_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ src2dst_map,
    T* __restrict__ output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    const float* __restrict__ weights) {
  int64_t src_row = blockIdx.x;
  if (src_row >= num_src_rows) return;

  int64_t dst_row = src2dst_map[src_row];
  if (dst_row >= num_dst_rows) return;

  float w = (weights != nullptr) ? weights[dst_row] : 1.0f;

  constexpr int kBytesPerVec = 16;
  constexpr int kElemsPerVec = kBytesPerVec / sizeof(T);
  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(input + src_row * num_cols);
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(output + dst_row * num_cols);
  const bool use_vec = (src_addr % kBytesPerVec == 0) && (dst_addr % kBytesPerVec == 0) &&
      (num_cols % kElemsPerVec == 0);

  if (use_vec) {
    int64_t vec_cols = num_cols / kElemsPerVec;
    auto* src_base = reinterpret_cast<const uint4*>(input + src_row * num_cols);
    auto* dst_base = reinterpret_cast<uint4*>(output + dst_row * num_cols);
    for (int64_t i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
      uint4 v = src_base[i];
      T* vals = reinterpret_cast<T*>(&v);
      #pragma unroll
      for (int j = 0; j < kElemsPerVec; ++j) {
        float val = to_float<T>(vals[j]) * w;
        vals[j] = from_float<T>(val);
      }
      dst_base[i] = v;
    }
  } else {
    for (int64_t i = threadIdx.x; i < num_cols; i += BLOCK_SIZE) {
      float val = to_float<T>(input[src_row * num_cols + i]) * w;
      output[dst_row * num_cols + i] = from_float<T>(val);
    }
  }
}

// Scatter kernel variant that atomically adds to output (for multi-expert accumulation)
template <typename T, int BLOCK_SIZE = 256>
__global__ void scatter_rows_accumulate_kernel(
    const T* __restrict__ input,
    const int32_t* __restrict__ src2dst_map,
    T* __restrict__ output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    const float* __restrict__ weights) {
  int64_t src_row = blockIdx.x;
  if (src_row >= num_src_rows) return;

  int64_t dst_row = src2dst_map[src_row];
  if (dst_row >= num_dst_rows) return;

  float w = (weights != nullptr) ? weights[dst_row] : 1.0f;

  for (int64_t i = threadIdx.x; i < num_cols; i += BLOCK_SIZE) {
    float val = to_float<T>(input[src_row * num_cols + i]) * w;
    // Use non-atomic write since each dst_row is written exactly once
    // (sorted_token_ids guarantees unique mapping per output slot)
    output[dst_row * num_cols + i] = from_float<T>(val);
  }
}

#if defined(USE_CUTLASS)
template <typename ScaleConfig, typename LayoutSFA, typename LayoutSFB, typename StrideA, typename StrideB, typename StrideC,
          typename UnderlyingProblemShape, typename OutType>
__global__ void build_grouped_gemm_args(
    int num_experts,
    const int32_t* expert_offsets,
    int total_rows,
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
    OutType* out,
    bool column_major_a_scales) {
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
  a_scales_ptrs[expert_id] = a_scales +
      static_cast<int64_t>(column_major_a_scales ? start : start * k_blocks);
  b_scales_ptrs[expert_id] = b_scales + static_cast<int64_t>(expert_id) * n_blocks * k_blocks;

  stride_a[expert_id] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  stride_b[expert_id] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  stride_c[expert_id] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));

  int scale_rows = column_major_a_scales ? total_rows : m;
  layout_sfa[expert_id] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(scale_rows, n, k, 1));
  layout_sfb[expert_id] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
}

template <typename ScheduleConfig, typename LayoutD, typename ElementD, typename ElementAccumulator, typename ElementC>
struct GroupedEpilogueSelector {
  using ArchTag = typename ScheduleConfig::ArchTag;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutD*,
      AlignmentC,
      ElementD,
      LayoutD*,
      AlignmentC,
      typename ScheduleConfig::EpilogueSchedule>::CollectiveOp;
};

template <typename ScheduleConfig, typename LayoutD, typename ElementD, typename ElementAccumulator, typename ElementC>
struct GroupedEpilogueSelectorSm90 {
  using ArchTag = typename ScheduleConfig::ArchTag;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using CustomEVTIdentity = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Identity, ElementD, ElementAccumulator, RoundStyle>,
      cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      typename ScheduleConfig::MmaTileShape,
      typename ScheduleConfig::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutD*,
      AlignmentC,
      ElementD,
      LayoutD*,
      AlignmentC,
      typename ScheduleConfig::EpilogueSchedule,
      CustomEVTIdentity>::CollectiveOp;
};

// Cached device properties to avoid cudaGetDevice + cudaGetDeviceProperties per call
static cutlass::KernelHardwareInfo get_cached_hw_info() {
  static cutlass::KernelHardwareInfo cached{};
  static bool initialized = false;
  if (!initialized) {
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, device_id);
    cached.device_id = device_id;
    cached.sm_count = props.multiProcessorCount;
    initialized = true;
  }
  return cached;
}

// Compute aligned offset for metadata carving
static inline size_t align_up(size_t val, size_t alignment) {
  return (val + alignment - 1) & ~(alignment - 1);
}

template <typename OutType, typename ScheduleConfig, typename LayoutD>
cutlass::Status launch_grouped_gemm(
    const cutlass::float_e4m3_t* a,
    const cutlass::float_e4m3_t* b,
    const float* a_scales,
    const float* b_scales,
    const int32_t* expert_offsets,
    int num_experts,
    int total_rows,
    int n,
    int k,
    int n_blocks,
    int k_blocks,
    OutType* out,
    void* workspace_buf,
    int64_t workspace_bytes,
    cudaStream_t stream,
    bool column_major_a_scales) {
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

  using ArchTag = typename ScheduleConfig::ArchTag;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueSelector = std::conditional_t<
      std::is_same_v<ArchTag, cutlass::arch::Sm90>,
      GroupedEpilogueSelectorSm90<ScheduleConfig, LayoutC, ElementD, ElementAccumulator, ElementC>,
      GroupedEpilogueSelector<ScheduleConfig, LayoutC, ElementD, ElementAccumulator, ElementC>>;
  using CollectiveEpilogue = typename EpilogueSelector::CollectiveEpilogue;

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

  // Carve metadata arrays out of the pre-allocated workspace buffer.
  // All 11 metadata arrays are packed contiguously with 256-byte alignment.
  constexpr size_t ALIGN = 256;
  size_t offset = 0;
  auto carve = [&](size_t bytes) -> void* {
    offset = align_up(offset, ALIGN);
    void* ptr = static_cast<char*>(workspace_buf) + offset;
    offset += bytes;
    return ptr;
  };

  auto* a_ptrs = static_cast<const cutlass::float_e4m3_t**>(carve(sizeof(cutlass::float_e4m3_t const*) * num_experts));
  auto* b_ptrs = static_cast<const cutlass::float_e4m3_t**>(carve(sizeof(cutlass::float_e4m3_t const*) * num_experts));
  auto* out_ptrs = static_cast<OutType**>(carve(sizeof(OutType*) * num_experts));
  auto* a_scales_ptrs = static_cast<const float**>(carve(sizeof(float const*) * num_experts));
  auto* b_scales_ptrs = static_cast<const float**>(carve(sizeof(float const*) * num_experts));
  auto* stride_a = static_cast<StrideA*>(carve(sizeof(StrideA) * num_experts));
  auto* stride_b = static_cast<StrideB*>(carve(sizeof(StrideB) * num_experts));
  auto* stride_c = static_cast<StrideC*>(carve(sizeof(StrideC) * num_experts));
  auto* layout_sfa = static_cast<typename ScheduleConfig::LayoutSFA*>(carve(sizeof(typename ScheduleConfig::LayoutSFA) * num_experts));
  auto* layout_sfb = static_cast<typename ScheduleConfig::LayoutSFB*>(carve(sizeof(typename ScheduleConfig::LayoutSFB) * num_experts));
  auto* problem_sizes = static_cast<UnderlyingProblemShape*>(carve(sizeof(UnderlyingProblemShape) * num_experts));

  // Remaining workspace available for CUTLASS
  offset = align_up(offset, ALIGN);
  void* cutlass_workspace = static_cast<char*>(workspace_buf) + offset;
  int64_t cutlass_workspace_bytes = workspace_bytes - static_cast<int64_t>(offset);

  const int threads = 256;
  const int blocks = (num_experts + threads - 1) / threads;

  build_grouped_gemm_args<typename ScheduleConfig::ScaleConfig, typename ScheduleConfig::LayoutSFA, typename ScheduleConfig::LayoutSFB, StrideA, StrideB, StrideC, UnderlyingProblemShape, OutType>
      <<<blocks, threads, 0, stream>>>(
          num_experts,
          expert_offsets,
          total_rows,
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
          out,
          column_major_a_scales);

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

  cutlass::KernelHardwareInfo hw_info = get_cached_hw_info();

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  cutlass::Status status = gemm_op.can_implement(args);

  if (status == cutlass::Status::kSuccess) {
    size_t needed = gemm_op.get_workspace_size(args);
    if (needed > 0 && static_cast<int64_t>(needed) > cutlass_workspace_bytes) {
      return cutlass::Status::kErrorInternal;
    }
    status = gemm_op.initialize(args, needed > 0 ? cutlass_workspace : nullptr, stream);
  }

  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.run(stream);
  }

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

struct Sm100GroupConfig {
  using ArchTag = cutlass::arch::Sm100;
  using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      1,
      128,
      128,
      cute::UMMA::Major::MN,
      cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};


struct Sm120GroupConfig {
  using ArchTag = cutlass::arch::Sm120;
  using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using KernelSchedule = cutlass::gemm::KernelScheduleSm120Blockwise;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      1,
      128,
      128,
      cute::UMMA::Major::MN,
      cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};
#endif

}  // namespace vllm_rs_moe


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

// Strided version for column-major scale tensors (SM100+ Blackwell)
extern "C" void moe_fp8_shuffle_rows_f32_strided(
    const float* input,
    const int32_t* dst2src_map,
    float* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols,
    int64_t src_row_stride,
    int64_t dst_row_stride,
    int32_t map_divisor,
    cudaStream_t stream) {
  dim3 grid(static_cast<uint32_t>(num_dst_rows));
  dim3 block(256);
  vllm_rs_moe::gather_rows_strided_kernel<float><<<grid, block, 0, stream>>>(
      input, dst2src_map, output, num_src_rows, num_dst_rows, num_cols,
      src_row_stride, dst_row_stride, map_divisor);
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
  // Adaptive block size: use 512 threads for wide rows to improve bandwidth utilization
  int block_size = (num_cols >= 2048) ? 512 : 256;
  dim3 grid(static_cast<uint32_t>(num_src_rows));
  if (block_size == 512)
    vllm_rs_moe::scatter_rows_kernel<half, 512><<<grid, dim3(512), 0, stream>>>(
        input, src2dst_map, output, num_src_rows, num_dst_rows, num_cols, weights);
  else
    vllm_rs_moe::scatter_rows_kernel<half, 256><<<grid, dim3(256), 0, stream>>>(
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
  int block_size = (num_cols >= 2048) ? 512 : 256;
  dim3 grid(static_cast<uint32_t>(num_src_rows));
  if (block_size == 512)
    vllm_rs_moe::scatter_rows_kernel<__nv_bfloat16, 512><<<grid, dim3(512), 0, stream>>>(
        input, src2dst_map, output, num_src_rows, num_dst_rows, num_cols, weights);
  else
    vllm_rs_moe::scatter_rows_kernel<__nv_bfloat16, 256><<<grid, dim3(256), 0, stream>>>(
        input, src2dst_map, output, num_src_rows, num_dst_rows, num_cols, weights);
}

// ---- Fused quantize-gather entry points (require hardware FP8, SM89+) --------
#if !defined(NO_HARDWARE_FP8)

extern "C" void moe_fp8_fused_quantize_gather(
    const void* input,         // [num_src_rows, K] in f16 or bf16
    const int32_t* dst2src_map, // [num_dst_rows] sorted_token_ids
    void* output_q,             // [num_dst_rows, K] fp8 output
    float* output_s,            // scales in target layout
    int num_src_rows,
    int num_dst_rows,
    int K,
    int num_groups_per_row,
    int scale_stride,
    int map_divisor,
    bool is_input_f16,
    bool is_column_major_scales,
    cudaStream_t stream) {

  constexpr int THREADS_PER_SUBWARP = 8;
  const int total_groups = num_dst_rows * num_groups_per_row;

  int subwarps_per_block = 16;  // 128 threads/block
  if (total_groups < 16) {
    if (total_groups % 8 == 0) subwarps_per_block = 8;
    else if (total_groups % 4 == 0) subwarps_per_block = 4;
    else if (total_groups % 2 == 0) subwarps_per_block = 2;
    else subwarps_per_block = 1;
  }

  const int num_blocks = (total_groups + subwarps_per_block - 1) / subwarps_per_block;
  const int num_threads = subwarps_per_block * THREADS_PER_SUBWARP;

  if (is_input_f16) {
    if (is_column_major_scales) {
      vllm_rs_moe::fused_quantize_gather_kernel<__half, true><<<num_blocks, num_threads, 0, stream>>>(
          (const __half*)input, dst2src_map, (__nv_fp8_e4m3*)output_q, output_s,
          num_src_rows, num_dst_rows, K, num_groups_per_row, scale_stride, map_divisor);
    } else {
      vllm_rs_moe::fused_quantize_gather_kernel<__half, false><<<num_blocks, num_threads, 0, stream>>>(
          (const __half*)input, dst2src_map, (__nv_fp8_e4m3*)output_q, output_s,
          num_src_rows, num_dst_rows, K, num_groups_per_row, scale_stride, map_divisor);
    }
  } else {
    if (is_column_major_scales) {
      vllm_rs_moe::fused_quantize_gather_kernel<__nv_bfloat16, true><<<num_blocks, num_threads, 0, stream>>>(
          (const __nv_bfloat16*)input, dst2src_map, (__nv_fp8_e4m3*)output_q, output_s,
          num_src_rows, num_dst_rows, K, num_groups_per_row, scale_stride, map_divisor);
    } else {
      vllm_rs_moe::fused_quantize_gather_kernel<__nv_bfloat16, false><<<num_blocks, num_threads, 0, stream>>>(
          (const __nv_bfloat16*)input, dst2src_map, (__nv_fp8_e4m3*)output_q, output_s,
          num_src_rows, num_dst_rows, K, num_groups_per_row, scale_stride, map_divisor);
    }
  }
}

// Fused quantize-gather + expert offsets for small M (decode path)
extern "C" void moe_fp8_fused_quantize_gather_offsets(
    const void* input,
    const int32_t* dst2src_map,
    const int32_t* expert_ids,
    void* output_q,
    float* output_s,
    int32_t* expert_counts,
    int32_t* expert_offsets,
    int num_src_rows,
    int num_dst_rows,
    int K,
    int num_groups_per_row,
    int scale_stride,
    int map_divisor,
    int num_experts,
    bool is_input_f16,
    bool is_column_major_scales,
    cudaStream_t stream) {

  constexpr int THREADS_PER_SUBWARP = 8;
  const int total_groups = num_dst_rows * num_groups_per_row;

  int subwarps_per_block = 16;
  if (total_groups < 16) {
    if (total_groups % 8 == 0) subwarps_per_block = 8;
    else if (total_groups % 4 == 0) subwarps_per_block = 4;
    else if (total_groups % 2 == 0) subwarps_per_block = 2;
    else subwarps_per_block = 1;
  }

  const int num_blocks = (total_groups + subwarps_per_block - 1) / subwarps_per_block;
  const int num_threads = subwarps_per_block * THREADS_PER_SUBWARP;
  // Shared memory for expert offset computation in block 0
  int smem_bytes = num_experts * sizeof(int32_t);
  if (smem_bytes < num_threads * (int)sizeof(int32_t))
    smem_bytes = num_threads * sizeof(int32_t);

  if (is_input_f16) {
    if (is_column_major_scales) {
      vllm_rs_moe::fused_quantize_gather_offsets_kernel<__half, true><<<num_blocks, num_threads, smem_bytes, stream>>>(
          (const __half*)input, dst2src_map, expert_ids, (__nv_fp8_e4m3*)output_q, output_s,
          expert_counts, expert_offsets, num_src_rows, num_dst_rows, K,
          num_groups_per_row, scale_stride, map_divisor, num_experts);
    } else {
      vllm_rs_moe::fused_quantize_gather_offsets_kernel<__half, false><<<num_blocks, num_threads, smem_bytes, stream>>>(
          (const __half*)input, dst2src_map, expert_ids, (__nv_fp8_e4m3*)output_q, output_s,
          expert_counts, expert_offsets, num_src_rows, num_dst_rows, K,
          num_groups_per_row, scale_stride, map_divisor, num_experts);
    }
  } else {
    if (is_column_major_scales) {
      vllm_rs_moe::fused_quantize_gather_offsets_kernel<__nv_bfloat16, true><<<num_blocks, num_threads, smem_bytes, stream>>>(
          (const __nv_bfloat16*)input, dst2src_map, expert_ids, (__nv_fp8_e4m3*)output_q, output_s,
          expert_counts, expert_offsets, num_src_rows, num_dst_rows, K,
          num_groups_per_row, scale_stride, map_divisor, num_experts);
    } else {
      vllm_rs_moe::fused_quantize_gather_offsets_kernel<__nv_bfloat16, false><<<num_blocks, num_threads, smem_bytes, stream>>>(
          (const __nv_bfloat16*)input, dst2src_map, expert_ids, (__nv_fp8_e4m3*)output_q, output_s,
          expert_counts, expert_offsets, num_src_rows, num_dst_rows, K,
          num_groups_per_row, scale_stride, map_divisor, num_experts);
    }
  }
}

#endif  // !NO_HARDWARE_FP8

extern "C" void moe_fp8_grouped_gemm_f16(
    const uint8_t* a,
    const uint8_t* b,
    const float* a_scales,
    const float* b_scales,
    const int32_t* expert_offsets,
    int num_experts,
    int m,
    int n,
    int k,
    int block_size_n,
    int block_size_k,
    int sm_version,
    half* out,
    void* workspace,
    int64_t workspace_bytes,
    cudaStream_t stream) {
#if defined(USE_CUTLASS)
  const auto* a_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(a);
  const auto* b_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(b);
  auto* out_ptr = reinterpret_cast<cutlass::half_t*>(out);

  int n_blocks = (n + block_size_n - 1) / block_size_n;
  int k_blocks = (k + block_size_k - 1) / block_size_k;
  bool column_major_a_scales = sm_version >= 100;

  if (sm_version >= 120) {
    auto status = vllm_rs_moe::launch_grouped_gemm<cutlass::half_t, vllm_rs_moe::Sm120GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, m, n, k, n_blocks, k_blocks, out_ptr,
        workspace, workspace_bytes, stream, column_major_a_scales);
    if (status != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_f16 sm120 failed: %s\n", cutlass::cutlassGetStatusString(status));
    }
    return;
  }

  if (sm_version >= 100) {
    auto status = vllm_rs_moe::launch_grouped_gemm<cutlass::half_t, vllm_rs_moe::Sm100GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, m, n, k, n_blocks, k_blocks, out_ptr,
        workspace, workspace_bytes, stream, column_major_a_scales);
    if (status != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_f16 sm100 failed: %s\n", cutlass::cutlassGetStatusString(status));
    }
    return;
  }

  if (sm_version >= 90) {
    auto status1 = vllm_rs_moe::launch_grouped_gemm<cutlass::half_t, vllm_rs_moe::Sm90GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, m, n, k, n_blocks, k_blocks, out_ptr,
        workspace, workspace_bytes, stream, column_major_a_scales);
    if (status1 != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_f16 sm90 failed: %s\n", cutlass::cutlassGetStatusString(status1));
    }
    return;
  }
#endif
  printf("moe_fp8_grouped_gemm_f16 unsupported sm_version %d\n", sm_version);
}

extern "C" void moe_fp8_grouped_gemm_bf16(
    const uint8_t* a,
    const uint8_t* b,
    const float* a_scales,
    const float* b_scales,
    const int32_t* expert_offsets,
    int num_experts,
    int m,
    int n,
    int k,
    int block_size_n,
    int block_size_k,
    int sm_version,
    __nv_bfloat16* out,
    void* workspace,
    int64_t workspace_bytes,
    cudaStream_t stream) {
#if defined(USE_CUTLASS)
  const auto* a_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(a);
  const auto* b_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(b);
  auto* out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(out);

  int n_blocks = (n + block_size_n - 1) / block_size_n;
  int k_blocks = (k + block_size_k - 1) / block_size_k;
  bool column_major_a_scales = sm_version >= 100;

  if (sm_version >= 120) {
    auto status = vllm_rs_moe::launch_grouped_gemm<cutlass::bfloat16_t, vllm_rs_moe::Sm120GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, m, n, k, n_blocks, k_blocks, out_ptr,
        workspace, workspace_bytes, stream, column_major_a_scales);
    if (status != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_bf16 sm120 failed: %s\n", cutlass::cutlassGetStatusString(status));
    }
    return;
  }

  if (sm_version >= 100) {
    auto status = vllm_rs_moe::launch_grouped_gemm<cutlass::bfloat16_t, vllm_rs_moe::Sm100GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, m, n, k, n_blocks, k_blocks, out_ptr,
        workspace, workspace_bytes, stream, column_major_a_scales);
    if (status != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_bf16 sm100 failed: %s\n", cutlass::cutlassGetStatusString(status));
    }
    return;
  }

  if (sm_version >= 90) {
    auto status1 = vllm_rs_moe::launch_grouped_gemm<cutlass::bfloat16_t, vllm_rs_moe::Sm90GroupConfig, cutlass::layout::RowMajor>(
        a_ptr, b_ptr, a_scales, b_scales, expert_offsets, num_experts, m, n, k, n_blocks, k_blocks, out_ptr,
        workspace, workspace_bytes, stream, column_major_a_scales);
    if (status1 != cutlass::Status::kSuccess) {
      printf("moe_fp8_grouped_gemm_bf16 sm90 failed: %s\n", cutlass::cutlassGetStatusString(status1));
    }
    return;
  }
#endif
  printf("moe_fp8_grouped_gemm_bf16 unsupported sm_version %d\n", sm_version);
}

