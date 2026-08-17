/*
 * Native GGUF GEMM for Q/K/IQ weight blocks.
 *
 * The input is quantized to Q8_1 per activation row, while the weight stays
 * in its original GGUF block format.  This is intentionally a GEMV/GEMM
 * family for ordinary dense layers, separate from the MoE dispatch kernels.
 * In particular, IQ weights must not be expanded to a dense F32 matrix before
 * the matmul: CUDA graph capture would retain one such expansion per graph.
 */
#include "gguf/gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
constexpr int MATRIX_ROW_PADDING = 512;

constexpr int pad(int size, int padding) {
    return ((size + padding - 1) / padding) * padding;
}

constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void gguf_gemm_kernel(
    const void * __restrict__ weights, // [N, K] in GGUF block layout
    const void * __restrict__ inputs,  // [M, K] in Q8_1 block layout
    float * __restrict__ outputs,      // [M, N]
    int size_m,
    int size_n,
    int size_k,
    int k_padded
) {
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int n_warps = blockDim.y;
    const int row = blockIdx.x * n_warps + warp_id;
    const int m_idx = blockIdx.y;

    if (row >= size_n || m_idx >= size_m) {
        return;
    }

    const int weight_blocks_per_row = size_k / qk;
    const int input_blocks_per_row = k_padded / QK8_1;
    const size_t weight_row_offset = (size_t)row * weight_blocks_per_row;
    const size_t input_row_offset = (size_t)m_idx * input_blocks_per_row;
    const block_q_t * __restrict__ weight_row =
        reinterpret_cast<const block_q_t *>(weights) + weight_row_offset;
    const block_q8_1 * __restrict__ input_row =
        reinterpret_cast<const block_q8_1 *>(inputs) + input_row_offset;

    float acc = 0.0f;
    const int blocks_per_iter = vdr * WARP_SIZE / qi;
    for (int k_block = lane_id / (qi / vdr);
         k_block < weight_blocks_per_row;
         k_block += blocks_per_iter) {
        const int input_block = k_block * (qk / QK8_1);
        const int quant_index = vdr * (lane_id % (qi / vdr));
        acc += vec_dot_q_cuda(&weight_row[k_block], &input_row[input_block], quant_index);
    }

    const float value = warp_reduce_sum(acc);
    if (lane_id == 0) {
        outputs[(size_t)m_idx * size_n + row] = value;
    }
}

// Prefill is a matrix-matrix operation.  The original kernel above is
// intentionally a GEMV-shaped kernel: one block handles one token and four
// output rows.  That is reasonable for decode, but it makes a long prefill
// launch millions of small blocks and rereads the same compressed weight row
// for every token.  This tile keeps the weights compressed, but reuses one
// weight row for eight input tokens.
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__global__ void gguf_gemm_prefill_kernel(
    const void * __restrict__ weights,
    const void * __restrict__ inputs,
    float * __restrict__ outputs,
    int size_m,
    int size_n,
    int size_k,
    int k_padded
) {
    constexpr int n_warps = 8;
    constexpr int m_tile = 8;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row = blockIdx.x * n_warps + warp_id;
    const int m_base = blockIdx.y * m_tile;

    if (row >= size_n) {
        return;
    }

    const int weight_blocks_per_row = size_k / qk;
    const int input_blocks_per_row = k_padded / QK8_1;
    const block_q_t * __restrict__ weight_row =
        reinterpret_cast<const block_q_t *>(weights) +
        (size_t)row * weight_blocks_per_row;
    const block_q8_1 * __restrict__ input_base =
        reinterpret_cast<const block_q8_1 *>(inputs);

    // Each warp owns one row-sized shared-memory slice.  The largest GGUF
    // block used here is well below 256 bytes, and every slice is aligned by
    // the 256-byte stride below.  The bytes are copied once per K block and
    // then consumed by all m_tile accumulators in this warp.
    extern __shared__ unsigned char shared_weights[];
    constexpr int shared_stride = 256;
    block_q_t * __restrict__ shared_row = reinterpret_cast<block_q_t *>(
        shared_weights + warp_id * shared_stride);

    float acc[m_tile] = {0.0f};
    const int blocks_per_iter = vdr * WARP_SIZE / qi;
    const int first_k_block = lane_id / (qi / vdr);

    for (int k_block = first_k_block;
         k_block < weight_blocks_per_row;
         k_block += blocks_per_iter) {
        const unsigned char * __restrict__ src =
            reinterpret_cast<const unsigned char *>(&weight_row[k_block]);
        unsigned char * __restrict__ dst =
            reinterpret_cast<unsigned char *>(shared_row);
        for (int byte = lane_id; byte < (int)sizeof(block_q_t); byte += WARP_SIZE) {
            dst[byte] = src[byte];
        }
        __syncwarp();

        const int input_block = k_block * (qk / QK8_1);
        const int quant_index = vdr * (lane_id % (qi / vdr));
        #pragma unroll
        for (int mi = 0; mi < m_tile; ++mi) {
            const int m = m_base + mi;
            if (m < size_m) {
                const block_q8_1 * __restrict__ input_row =
                    input_base + (size_t)m * input_blocks_per_row;
                acc[mi] += vec_dot_q_cuda(
                    shared_row, &input_row[input_block], quant_index);
            }
        }
        __syncwarp();
    }

    #pragma unroll
    for (int mi = 0; mi < m_tile; ++mi) {
        const float value = warp_reduce_sum(acc[mi]);
        const int m = m_base + mi;
        if (lane_id == 0 && m < size_m) {
            outputs[(size_t)m * size_n + row] = value;
        }
    }
}

// IQ4_NL is the dominant format in Qwen3.8 IQ4_NL checkpoints.  Its generic
// MMVQ-shaped kernel assigns one output row to a warp and splits each 32-value
// block over only four lanes.  That mapping is useful for decode, but wastes
// most of the matrix-tile opportunity during prefill.  This kernel assigns
// one thread to a small output stripe, tiles 32 tokens x 64 output rows, and
// keeps both compressed weight blocks and Q8 input blocks in shared memory.
// It never materializes a dense weight tensor.
__global__ void gguf_gemm_iq4_nl_prefill_kernel(
    const void * __restrict__ weights,
    const void * __restrict__ inputs,
    float * __restrict__ outputs,
    int size_m,
    int size_n,
    int size_k,
    int k_padded
) {
    constexpr int m_tile = 32;
    constexpr int n_tile = 64;
    constexpr int n_per_thread = 2;
    constexpr int rows_per_warp = 4;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int m_base = blockIdx.y * m_tile + warp_id * rows_per_warp;
    const int n_base = blockIdx.x * n_tile + lane_id;
    const int weight_blocks_per_row = size_k / QK_IQ4_NL;
    const int input_blocks_per_row = k_padded / QK8_1;

    extern __shared__ unsigned char shared[];
    constexpr int weight_tile_bytes = n_tile * sizeof(block_iq4_nl);
    block_iq4_nl *shared_weights = reinterpret_cast<block_iq4_nl *>(shared);
    block_q8_1 *shared_inputs = reinterpret_cast<block_q8_1 *>(
        shared + weight_tile_bytes);

    const block_iq4_nl *all_weights = reinterpret_cast<const block_iq4_nl *>(weights);
    const block_q8_1 *all_inputs = reinterpret_cast<const block_q8_1 *>(inputs);
    float acc[rows_per_warp][n_per_thread] = {};

    for (int k_block = 0; k_block < weight_blocks_per_row; ++k_block) {
        for (int byte = tid; byte < weight_tile_bytes; byte += blockDim.x) {
            const int local_n = byte / (int)sizeof(block_iq4_nl);
            const int local_byte = byte % (int)sizeof(block_iq4_nl);
            if (blockIdx.x * n_tile + local_n < size_n) {
                const size_t offset = (size_t)(blockIdx.x * n_tile + local_n) *
                    weight_blocks_per_row + k_block;
                reinterpret_cast<unsigned char *>(shared_weights)[byte] =
                    reinterpret_cast<const unsigned char *>(&all_weights[offset])[local_byte];
            } else {
                reinterpret_cast<unsigned char *>(shared_weights)[byte] = 0;
            }
        }

        const int input_tile_bytes = m_tile * sizeof(block_q8_1);
        for (int byte = tid; byte < input_tile_bytes; byte += blockDim.x) {
            const int local_m = byte / (int)sizeof(block_q8_1);
            const int local_byte = byte % (int)sizeof(block_q8_1);
            if (blockIdx.y * m_tile + local_m < size_m) {
                const size_t offset = (size_t)(blockIdx.y * m_tile + local_m) *
                    input_blocks_per_row + k_block;
                reinterpret_cast<unsigned char *>(shared_inputs)[byte] =
                    reinterpret_cast<const unsigned char *>(&all_inputs[offset])[local_byte];
            } else {
                reinterpret_cast<unsigned char *>(shared_inputs)[byte] = 0;
            }
        }
        __syncthreads();

#pragma unroll
        for (int mi = 0; mi < rows_per_warp; ++mi) {
#pragma unroll
            for (int ni = 0; ni < n_per_thread; ++ni) {
                const int m = m_base + mi;
                const int n = n_base + 32 * ni;
                if (m < size_m && n < size_n) {
                    acc[mi][ni] += vec_dot_iq4_nl_full(
                        shared_weights + lane_id + 32 * ni,
                        shared_inputs + warp_id * rows_per_warp + mi);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < rows_per_warp; ++mi) {
#pragma unroll
        for (int ni = 0; ni < n_per_thread; ++ni) {
            const int m = m_base + mi;
            const int n = n_base + 32 * ni;
            if (m < size_m && n < size_n) {
                outputs[(size_t)m * size_n + n] = acc[mi][ni];
            }
        }
    }
}

#define LAUNCH_GGUF_GEMM(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    gguf_gemm_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda> \
        <<<grid_dim, block_dim, 0, stream>>>( \
            weights, input_q8_1, outputs, size_m, size_n, size_k, k_padded)

#define LAUNCH_GGUF_GEMM_PREFILL(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    gguf_gemm_prefill_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda> \
        <<<prefill_grid, prefill_block, 8 * 256, stream>>>( \
            weights, input_q8_1, outputs, size_m, size_n, size_k, k_padded)

#define LAUNCH_GGUF_IQ4_NL_PREFILL() \
    gguf_gemm_iq4_nl_prefill_kernel<<< \
        dim3(ceil_div(size_n, 64), ceil_div(size_m, 32), 1), \
        dim3(256, 1, 1), \
        64 * sizeof(block_iq4_nl) + 32 * sizeof(block_q8_1), stream>>>( \
            weights, input_q8_1, outputs, size_m, size_n, size_k, k_padded)

extern "C" void gguf_gemm(
    const float *inputs,
    const void *weights,
    float *outputs,
    int size_m,
    int size_n,
    int size_k,
    int quant_type,
    cudaStream_t stream
) {
    const int k_padded = pad(size_k, MATRIX_ROW_PADDING);
    const int input_blocks_per_row = k_padded / QK8_1;
    const size_t input_bytes =
        (size_t)size_m * input_blocks_per_row * sizeof(block_q8_1);

    void *input_q8_1 = nullptr;
    cudaMallocAsync(&input_q8_1, input_bytes, stream);
    const int quant_blocks = ceil_div(k_padded, CUDA_QUANTIZE_BLOCK_SIZE);
    dim3 quant_grid(quant_blocks, size_m, 1);
    dim3 quant_block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<quant_grid, quant_block, 0, stream>>>(
        inputs, input_q8_1, size_k, k_padded);

    constexpr int n_warps = 4;
    dim3 grid_dim(ceil_div(size_n, n_warps), size_m, 1);
    dim3 block_dim(WARP_SIZE, n_warps, 1);

    // For prefill, use a matrix-matrix tile.  Keep the original kernel for
    // small M because its lower shared-memory/setup cost is better for decode.
    const bool use_prefill_kernel = size_m >= 8;
    dim3 prefill_grid(ceil_div(size_n, 8), ceil_div(size_m, 8), 1);
    dim3 prefill_block(WARP_SIZE, 8, 1);

    switch (quant_type) {
        case 0:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1);
            else LAUNCH_GGUF_GEMM(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1);
            break;
        case 1:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1);
            else LAUNCH_GGUF_GEMM(QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1);
            break;
        case 2:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1);
            else LAUNCH_GGUF_GEMM(QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1);
            break;
        case 3:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1);
            else LAUNCH_GGUF_GEMM(QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1);
            break;
        case 4:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1);
            else LAUNCH_GGUF_GEMM(QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1);
            break;
        case 5:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1);
            else LAUNCH_GGUF_GEMM(QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1);
            break;
        case 6:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_IQ2_XXS, QI_IQ2_XXS, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1);
            else LAUNCH_GGUF_GEMM(QK_IQ2_XXS, QI_IQ2_XXS, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1);
            break;
        case 7:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_IQ2_XS, QI_IQ2_XS, block_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1);
            else LAUNCH_GGUF_GEMM(QK_IQ2_XS, QI_IQ2_XS, block_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1);
            break;
        case 8:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_IQ3_XXS, QI_IQ3_XXS, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1);
            else LAUNCH_GGUF_GEMM(QK_IQ3_XXS, QI_IQ3_XXS, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1);
            break;
        case 9:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_IQ4_XS, QI_IQ4_XS, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1);
            else LAUNCH_GGUF_GEMM(QK_IQ4_XS, QI_IQ4_XS, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1);
            break;
        case 10:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_IQ1_S, QI_IQ1_S, block_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1);
            else LAUNCH_GGUF_GEMM(QK_IQ1_S, QI_IQ1_S, block_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1);
            break;
        case 11:
            if (use_prefill_kernel) LAUNCH_GGUF_IQ4_NL_PREFILL();
            else LAUNCH_GGUF_GEMM(QK_IQ4_NL, QI_IQ4_NL, block_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1);
            break;
        case 12:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_IQ3_S, QI_IQ3_S, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1);
            else LAUNCH_GGUF_GEMM(QK_IQ3_S, QI_IQ3_S, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1);
            break;
        case 13:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_IQ2_S, QI_IQ2_S, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1);
            else LAUNCH_GGUF_GEMM(QK_IQ2_S, QI_IQ2_S, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1);
            break;
        case 14:
            if (use_prefill_kernel) LAUNCH_GGUF_GEMM_PREFILL(QK_IQ1_M, QI_IQ1_M, block_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1);
            else LAUNCH_GGUF_GEMM(QK_IQ1_M, QI_IQ1_M, block_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1);
            break;
        default:
            break;
    }
    cudaFreeAsync(input_q8_1, stream);
}
