#include <cuda.h>
#include <cuda_runtime.h>

#if defined(USE_CUTLASS)
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#endif

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#if defined(USE_CUTLASS)
using namespace cute;

namespace vllm_fp8_scale_pack {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
using MmaTileShape_MNK = Shape<_128, _128, _128>;
using ScaleConfig = decltype(cutlass::detail::sm100_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
#elif defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
using MmaTileShape_MNK = Shape<_128, _128, _128>;
using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
#endif

template <typename Layout>
__global__ void pack_sfb_layout(const float* scales_rm, float* scales_packed,
                                Layout layout_sfb, int n_blocks,
                                int k_blocks, bool input_k_major) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_blocks * k_blocks;
  if (idx >= total) {
    return;
  }
  int n_blk = idx / k_blocks;
  int k_blk = idx - n_blk * k_blocks;
  int src_idx = input_k_major ? (k_blk * n_blocks + n_blk) : idx;
  int packed_idx = layout_sfb(n_blk * 128, k_blk * 128, 0);
  scales_packed[packed_idx] = scales_rm[src_idx];
}

} // namespace vllm_fp8_scale_pack
#endif

extern "C" int fp8_sfb_packed_len(int N, int K) {
#if defined(USE_CUTLASS) && (defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED))
  auto layout_sfb = vllm_fp8_scale_pack::ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, N, K, 1));
  return int(size(filter_zeros(layout_sfb)));
#else
  (void)N;
  (void)K;
  return 0;
#endif
}

extern "C" void fp8_pack_sfb_scales(
    const float* scales_rm,
    float* scales_packed,
    int N,
    int K,
    int block_size_n,
    int block_size_k,
    int input_k_major,
    cudaStream_t stream) {
#if defined(USE_CUTLASS) && (defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED))
  int n_blocks = CEILDIV(N, block_size_n);
  int k_blocks = CEILDIV(K, block_size_k);
  auto layout_sfb = vllm_fp8_scale_pack::ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, N, K, 1));
  int total = n_blocks * k_blocks;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  vllm_fp8_scale_pack::pack_sfb_layout<<<blocks, threads, 0, stream>>>(
      scales_rm, scales_packed, layout_sfb, n_blocks, k_blocks, input_k_major != 0);
#else
  (void)scales_rm;
  (void)scales_packed;
  (void)N;
  (void)K;
  (void)block_size_n;
  (void)block_size_k;
  (void)input_k_major;
  (void)stream;
#endif
}
