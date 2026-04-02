#include <stdint.h>

template <typename scalar_t>
__global__ void scatter_rows_kernel(
    const scalar_t* __restrict__ src,      // [num_rows, src_row_stride]
    scalar_t* __restrict__ dst,            // [dst_rows, dst_row_stride]
    const int64_t* __restrict__ slots,     // [num_rows]
    const int num_rows,
    const int row_elems,
    const int64_t src_row_stride,
    const int64_t dst_row_stride) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (row >= num_rows || col >= row_elems) {
    return;
  }
  const int64_t dst_row = slots[row];
  dst[dst_row * dst_row_stride + col] = src[row * src_row_stride + col];
}

template <typename scalar_t>
void launch_scatter_rows(
    const void* src_ptr,
    void* dst_ptr,
    const int64_t* slots_ptr,
    int num_rows,
    int row_elems,
    int64_t src_row_stride,
    int64_t dst_row_stride,
    cudaStream_t stream) {
  if (num_rows <= 0 || row_elems <= 0) return;
  const dim3 block(256, 1, 1);
  const dim3 grid((row_elems + block.x - 1) / block.x, num_rows, 1);
  scatter_rows_kernel<scalar_t><<<grid, block, 0, stream>>>(
      reinterpret_cast<const scalar_t*>(src_ptr),
      reinterpret_cast<scalar_t*>(dst_ptr),
      slots_ptr,
      num_rows,
      row_elems,
      src_row_stride,
      dst_row_stride);
}

extern "C" void mamba_scatter_rows_f16(
    const void* src,
    void* dst,
    const int64_t* slots,
    int num_rows,
    int row_elems,
    int64_t src_row_stride,
    int64_t dst_row_stride,
    cudaStream_t stream) {
  launch_scatter_rows<int16_t>(
      src, dst, slots, num_rows, row_elems, src_row_stride, dst_row_stride, stream);
}

extern "C" void mamba_scatter_rows_bf16(
    const void* src,
    void* dst,
    const int64_t* slots,
    int num_rows,
    int row_elems,
    int64_t src_row_stride,
    int64_t dst_row_stride,
    cudaStream_t stream) {
  launch_scatter_rows<int16_t>(
      src, dst, slots, num_rows, row_elems, src_row_stride, dst_row_stride, stream);
}

extern "C" void mamba_scatter_rows_f32(
    const void* src,
    void* dst,
    const int64_t* slots,
    int num_rows,
    int row_elems,
    int64_t src_row_stride,
    int64_t dst_row_stride,
    cudaStream_t stream) {
  launch_scatter_rows<float>(
      src, dst, slots, num_rows, row_elems, src_row_stride, dst_row_stride, stream);
}
