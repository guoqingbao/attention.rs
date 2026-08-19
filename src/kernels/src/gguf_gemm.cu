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

static __device__ __forceinline__ void gguf_cp_async_16(void *smem, const void *glob) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned smem_ptr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr), "l"(glob));
#else
    *reinterpret_cast<int4 *>(smem) = __ldg(reinterpret_cast<const int4 *>(glob));
#endif
}

static __device__ __forceinline__ void gguf_cp_async_4(void *smem, const void *glob) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned smem_ptr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_ptr), "l"(glob));
#else
    *reinterpret_cast<int *>(smem) = __ldg(reinterpret_cast<const int *>(glob));
#endif
}

static __device__ __forceinline__ void gguf_cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

static __device__ __forceinline__ void gguf_cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

template <int n>
static __device__ __forceinline__ void gguf_cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
#endif
}

template <typename block_q_t>
static __device__ __forceinline__ void gguf_copy_block_async(
    unsigned char *dst, const block_q_t *src, int lane, int nlanes)
{
    const unsigned char *s = reinterpret_cast<const unsigned char *>(src);
    constexpr int nbytes = (int)sizeof(block_q_t);
    if (((uintptr_t)src & 15) == 0) {
        constexpr int nvec = nbytes / 16;
        for (int v = lane; v < nvec; v += nlanes) {
            gguf_cp_async_16(dst + 16 * v, s + 16 * v);
        }
        for (int b = 16 * nvec + lane; b < nbytes; b += nlanes) {
            dst[b] = s[b];
        }
    } else {
        constexpr int n4 = nbytes / 4;
        for (int v = lane; v < n4; v += nlanes) {
            gguf_cp_async_4(dst + 4 * v, s + 4 * v);
        }
        for (int b = 4 * n4 + lane; b < nbytes; b += nlanes) {
            dst[b] = s[b];
        }
    }
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

// Prefill tile: same vec_dot as decode, but each warp reuses its GGUF blocks
// across an M-tile.  SM80+ copies those blocks with cp.async + double buffering.
//
// Lanes in a warp own different k_block indices (MMVQ).  Each k_block gets its
// own 256-byte shared slot so the copy cannot mix Q4_K/Q6_K superblocks.
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
    constexpr int k_slots = vdr * WARP_SIZE / qi;
    constexpr int blk_align = 256;
    constexpr int nlanes_slot = qi / vdr;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row = blockIdx.x * n_warps + warp_id;
    const int m_base = blockIdx.y * m_tile;
    const int slot = lane_id / nlanes_slot;
    const int lane_in_slot = lane_id % nlanes_slot;

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

    extern __shared__ unsigned char shared_weights[];
    const int slot_off = (warp_id * k_slots + slot) * blk_align;
    const int stage_stride = n_warps * k_slots * blk_align;

    float acc[m_tile] = {0.0f};
    int stage = 0;

    if (slot < weight_blocks_per_row) {
        gguf_copy_block_async(
            shared_weights + slot_off, &weight_row[slot],
            lane_in_slot, nlanes_slot);
        gguf_cp_async_commit();
    }

    for (int k_base = 0; k_base < weight_blocks_per_row; k_base += k_slots) {
        const int k_block = k_base + slot;
        const int k_next = k_block + k_slots;
        unsigned char * cur = shared_weights + stage * stage_stride + slot_off;
        if (k_base + k_slots < weight_blocks_per_row) {
            if (k_next < weight_blocks_per_row) {
                gguf_copy_block_async(
                    shared_weights + (1 - stage) * stage_stride + slot_off,
                    &weight_row[k_next], lane_in_slot, nlanes_slot);
            }
            gguf_cp_async_commit();
            gguf_cp_async_wait_group<1>();
        } else {
            gguf_cp_async_wait();
        }
        __syncwarp();

        if (k_block < weight_blocks_per_row) {
            const block_q_t * __restrict__ wblk =
                reinterpret_cast<const block_q_t *>(cur);
            const int input_block = k_block * (qk / QK8_1);
            const int quant_index = vdr * lane_in_slot;
            #pragma unroll
            for (int mi = 0; mi < m_tile; ++mi) {
                const int m = m_base + mi;
                if (m < size_m) {
                    const block_q8_1 * __restrict__ input_row =
                        input_base + (size_t)m * input_blocks_per_row;
                    acc[mi] += vec_dot_q_cuda(wblk, &input_row[input_block], quant_index);
                }
            }
        }
        stage = 1 - stage;
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


// ============================================================================
// IQ4_NL optimized prefill kernel (SM80+ Ampere through SM121 Blackwell).
//
// Pre-Ampere (SM70/SM75): host dispatch falls back to gguf_gemm_prefill_kernel
// (m_tile=8, ~2 KB shmem) — the large-tile kernel needs ~147 KB double-buffered
// shmem which exceeds the 96 KB per-block limit on Volta/Turing.
//
// Strategy (llama.cpp MMQ-style):
//  1. Pre-decode IQ4_NL nibbles → int8 in shared memory; inner loop uses dp4a.
//  2. Tiles: N_TILE=128 × M_TILE=128; K_ITER=256 per pipeline stage.
//  3. Double-buffered load/compute overlap.
//  4. Vectorized __byte_perm decode; __ldg for global loads.
// ============================================================================

#define IQ4_NL_K_ITER   256
#define IQ4_NL_N_TILE   128
#define IQ4_NL_M_TILE   128
#define IQ4_NL_NWARPS   8
#define IQ4_NL_K_BLOCKS (IQ4_NL_K_ITER / QK_IQ4_NL)

// Per-buffer: 73728 bytes; × 2 = 147456 bytes (fits SM80 164 KB opt-in, SM90 228 KB).
#define IQ4_NL_BUF_W_INT8   (IQ4_NL_N_TILE * IQ4_NL_K_ITER)
#define IQ4_NL_BUF_W_SCALE  (IQ4_NL_N_TILE * IQ4_NL_K_BLOCKS * 4)
#define IQ4_NL_BUF_X_INT8   (IQ4_NL_M_TILE * IQ4_NL_K_ITER)
#define IQ4_NL_BUF_X_SCALE  (IQ4_NL_M_TILE * IQ4_NL_K_BLOCKS * 4)
#define IQ4_NL_BUF_BYTES    (IQ4_NL_BUF_W_INT8 + IQ4_NL_BUF_W_SCALE + IQ4_NL_BUF_X_INT8 + IQ4_NL_BUF_X_SCALE)
#define IQ4_NL_SHMEM_BYTES  (2 * IQ4_NL_BUF_BYTES)

static __device__ __forceinline__ void iq4nl_load_weights(
    const block_iq4_nl * __restrict__ all_w,
    int8_t * __restrict__ smem_w, float * __restrict__ smem_d,
    const int n_base, const int k_block_start,
    const int wbpr, const int size_n, const int tid
) {
    #pragma unroll 4
    for (int blk = tid; blk < IQ4_NL_N_TILE * IQ4_NL_K_BLOCKS; blk += 256) {
        const int local_n  = blk / IQ4_NL_K_BLOCKS;
        const int local_kb = blk % IQ4_NL_K_BLOCKS;
        const int global_n  = n_base + local_n;
        const int global_kb = k_block_start + local_kb;
        const int w_off = local_n * IQ4_NL_K_ITER + local_kb * QK_IQ4_NL;

        float d_val = 0.0f;
        if (global_n < size_n && global_kb < wbpr) {
            const block_iq4_nl* bq = all_w + (size_t)global_n * wbpr + global_kb;
            d_val = __half2float(__ldg(&bq->d));
            int32_t* dst_lo = reinterpret_cast<int32_t *>(&smem_w[w_off]);
            int32_t* dst_hi = reinterpret_cast<int32_t *>(&smem_w[w_off + QK_IQ4_NL / 2]);
            #pragma unroll
            for (int i = 0; i < QK_IQ4_NL / 8; ++i) {
                const int2 decoded = iq4_nl_table_lookup(get_int_from_uint8(bq->qs, i));
                dst_lo[i] = decoded.x;
                dst_hi[i] = decoded.y;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < QK_IQ4_NL; ++i) smem_w[w_off + i] = 0;
        }
        smem_d[local_n * IQ4_NL_K_BLOCKS + local_kb] = d_val;
    }
}

static __device__ __forceinline__ void iq4nl_load_acts_async(
    const block_q8_1 * __restrict__ all_x,
    int8_t * __restrict__ smem_x, float * __restrict__ smem_s,
    const int m_base, const int k_block_start,
    const int ibpr, const int size_m, const int tid
) {
    #pragma unroll 4
    for (int blk = tid; blk < IQ4_NL_M_TILE * IQ4_NL_K_BLOCKS; blk += 256) {
        const int local_m  = blk / IQ4_NL_K_BLOCKS;
        const int local_kb = blk % IQ4_NL_K_BLOCKS;
        const int global_m  = m_base + local_m;
        const int global_kb = k_block_start + local_kb;
        const int x_off = local_m * IQ4_NL_K_ITER + local_kb * QK8_1;

        float ds_val = 0.0f;
        if (global_m < size_m && global_kb < ibpr) {
            const block_q8_1* bx = all_x + (size_t)global_m * ibpr + global_kb;
            ds_val = __low2float(__ldg(&bx->ds));
            const int* qs_i = reinterpret_cast<const int*>(bx->qs);
            int* dst_i = reinterpret_cast<int*>(&smem_x[x_off]);
            #pragma unroll
            for (int j = 0; j < QK8_1 / 4; ++j) {
                gguf_cp_async_4(&dst_i[j], &qs_i[j]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < QK8_1; ++i) smem_x[x_off + i] = 0;
        }
        smem_s[local_m * IQ4_NL_K_BLOCKS + local_kb] = ds_val;
    }
}

__launch_bounds__(256, 1)
__global__ void gguf_gemm_iq4_nl_prefill(
    const void * __restrict__ weights,
    const void * __restrict__ inputs,
    float       * __restrict__ outputs,
    int size_m, int size_n, int size_k, int k_padded
) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
    constexpr int N_TILE     = IQ4_NL_N_TILE;
    constexpr int M_TILE     = IQ4_NL_M_TILE;
    constexpr int K_ITER     = IQ4_NL_K_ITER;
    constexpr int K_BLOCKS   = IQ4_NL_K_BLOCKS;
    constexpr int N_PER_WARP = N_TILE / IQ4_NL_NWARPS;
    constexpr int M_PER_THR  = M_TILE / WARP_SIZE;

    extern __shared__ unsigned char smem_raw[];
    int8_t * smem_wA = reinterpret_cast<int8_t *>(smem_raw);
    float  * smem_dA = reinterpret_cast<float  *>(smem_raw + IQ4_NL_BUF_W_INT8);
    int8_t * smem_xA = reinterpret_cast<int8_t *>(smem_raw + IQ4_NL_BUF_W_INT8 + IQ4_NL_BUF_W_SCALE);
    float  * smem_sA = reinterpret_cast<float  *>(smem_raw + IQ4_NL_BUF_W_INT8 + IQ4_NL_BUF_W_SCALE + IQ4_NL_BUF_X_INT8);
    int8_t * smem_wB = reinterpret_cast<int8_t *>(smem_raw + IQ4_NL_BUF_BYTES);
    float  * smem_dB = reinterpret_cast<float  *>(smem_raw + IQ4_NL_BUF_BYTES + IQ4_NL_BUF_W_INT8);
    int8_t * smem_xB = reinterpret_cast<int8_t *>(smem_raw + IQ4_NL_BUF_BYTES + IQ4_NL_BUF_W_INT8 + IQ4_NL_BUF_W_SCALE);
    float  * smem_sB = reinterpret_cast<float  *>(smem_raw + IQ4_NL_BUF_BYTES + IQ4_NL_BUF_W_INT8 + IQ4_NL_BUF_W_SCALE + IQ4_NL_BUF_X_INT8);

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int n_base  = blockIdx.x * N_TILE;
    const int m_base  = blockIdx.y * M_TILE;
    const int wbpr    = size_k / QK_IQ4_NL;
    const int ibpr    = k_padded / QK8_1;

    const block_iq4_nl * all_w = reinterpret_cast<const block_iq4_nl *>(weights);
    const block_q8_1   * all_x = reinterpret_cast<const block_q8_1   *>(inputs);

    float acc[N_PER_WARP][M_PER_THR] = {};
    const int n_warp_base = warp_id * N_PER_WARP;

    if (wbpr > 0) {
        iq4nl_load_acts_async(all_x, smem_xA, smem_sA,
                              m_base, 0, ibpr, size_m, tid);
        gguf_cp_async_commit();
        iq4nl_load_weights(all_w, smem_wA, smem_dA,
                           n_base, 0, wbpr, size_n, tid);
        gguf_cp_async_wait();
    }
    __syncthreads();

    for (int kb_start = 0; kb_start < wbpr; kb_start += K_BLOCKS) {
        const bool use_a = ((kb_start / K_BLOCKS) % 2 == 0);
        int8_t * w_cur = use_a ? smem_wA : smem_wB;
        float  * d_cur = use_a ? smem_dA : smem_dB;
        int8_t * x_cur = use_a ? smem_xA : smem_xB;
        float  * s_cur = use_a ? smem_sA : smem_sB;
        int8_t * w_nxt = use_a ? smem_wB : smem_wA;
        float  * d_nxt = use_a ? smem_dB : smem_dA;
        int8_t * x_nxt = use_a ? smem_xB : smem_xA;
        float  * s_nxt = use_a ? smem_sB : smem_sA;

        const int next_start = kb_start + K_BLOCKS;
        if (next_start < wbpr) {
            iq4nl_load_acts_async(all_x, x_nxt, s_nxt,
                                  m_base, next_start, ibpr, size_m, tid);
            gguf_cp_async_commit();
            iq4nl_load_weights(all_w, w_nxt, d_nxt,
                               n_base, next_start, wbpr, size_n, tid);
        }

        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; ++kb) {
            #pragma unroll
            for (int ni = 0; ni < N_PER_WARP; ++ni) {
                const int8_t * w_row = w_cur + (n_warp_base + ni) * K_ITER + kb * QK_IQ4_NL;
                const float d_w = d_cur[(n_warp_base + ni) * K_BLOCKS + kb];
                #pragma unroll
                for (int mi = 0; mi < M_PER_THR; ++mi) {
                    const int m_local = lane_id * M_PER_THR + mi;
                    const int8_t * x_row = x_cur + m_local * K_ITER + kb * QK8_1;
                    const float d_x = s_cur[m_local * K_BLOCKS + kb];
                    int dot = 0;
                    #pragma unroll
                    for (int i = 0; i < QI_IQ4_NL; ++i) {
                        dot = ggml_cuda_dp4a(
                            *reinterpret_cast<const int *>(w_row + 4 * i),
                            *reinterpret_cast<const int *>(x_row + 4 * i),
                            dot);
                        dot = ggml_cuda_dp4a(
                            *reinterpret_cast<const int *>(w_row + 16 + 4 * i),
                            *reinterpret_cast<const int *>(x_row + 16 + 4 * i),
                            dot);
                    }
                    acc[ni][mi] += d_w * d_x * (float)dot;
                }
            }
        }

        if (next_start < wbpr) {
            gguf_cp_async_wait();
        }
        __syncthreads();
    }

    #pragma unroll
    for (int ni = 0; ni < N_PER_WARP; ++ni) {
        const int gn = n_base + n_warp_base + ni;
        if (gn >= size_n) continue;
        #pragma unroll
        for (int mi = 0; mi < M_PER_THR; ++mi) {
            const int gm = m_base + lane_id * M_PER_THR + mi;
            if (gm < size_m) {
                outputs[(size_t)gm * size_n + gn] = acc[ni][mi];
            }
        }
    }
#endif // SM80+ device path
}

// ============================================================================
// IQ4_NL decode kernel (SM70+ through SM121).
//
// Decode uses M < 8 (typically M=1).  Zero shared memory for full occupancy;
// the input row is reused across warps via L2 cache instead of explicit staging.
// Each lane owns one K block (32-way parallelism vs 8 in the generic MMVQ path).
// ============================================================================

static __device__ __forceinline__ float iq4nl_dot_block_ldg(
    const block_iq4_nl * __restrict__ bq,
    const block_q8_1   * __restrict__ bx
) {
    const float scale =
        __half2float(__ldg(&bq->d)) * __low2float(__ldg(&bx->ds));
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < QI_IQ4_NL; ++i) {
        const int2 decoded = iq4_nl_table_lookup(get_int_from_uint8(bq->qs, i));
        const int x_lo = __ldg(reinterpret_cast<const int *>(bx->qs + 4 * i));
        const int x_hi = __ldg(reinterpret_cast<const int *>(bx->qs + 16 + 4 * i));
        sum = ggml_cuda_dp4a(decoded.x, x_lo, sum);
        sum = ggml_cuda_dp4a(decoded.y, x_hi, sum);
    }
    return scale * (float)sum;
}

__global__ void gguf_gemm_iq4_nl_decode(
    const void * __restrict__ weights,
    const void * __restrict__ inputs,
    float       * __restrict__ outputs,
    int size_m, int size_n, int size_k, int k_padded
) {
    constexpr int NWARPS = 4;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int row = blockIdx.x * NWARPS + warp_id;
    const int m_idx = blockIdx.y;

    if (row >= size_n || m_idx >= size_m) {
        return;
    }

    const int wbpr = size_k / QK_IQ4_NL;
    const int ibpr = k_padded / QK8_1;
    const block_iq4_nl * w_row =
        reinterpret_cast<const block_iq4_nl *>(weights) +
        (size_t)row * wbpr;
    const block_q8_1 * x_row =
        reinterpret_cast<const block_q8_1 *>(inputs) +
        (size_t)m_idx * ibpr;

    float acc = 0.0f;
    for (int kb = lane_id; kb < wbpr; kb += WARP_SIZE) {
        acc += iq4nl_dot_block_ldg(w_row + kb, x_row + kb);
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) {
        outputs[(size_t)m_idx * size_n + row] = acc;
    }
}

#define LAUNCH_GGUF_GEMM(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    gguf_gemm_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda> \
        <<<grid_dim, block_dim, 0, stream>>>( \
            weights, input_q8_1, outputs, size_m, size_n, size_k, k_padded)

#define LAUNCH_GGUF_GEMM_PREFILL(qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
    gguf_gemm_prefill_kernel<qk, qi, block_q_t, vdr, vec_dot_q_cuda> \
        <<<prefill_grid, prefill_block, \
            (2 * 8 * ((vdr) * WARP_SIZE / (qi)) * 256), stream>>>( \
            weights, input_q8_1, outputs, size_m, size_n, size_k, k_padded)

#define LAUNCH_GGUF_IQ4_NL_PREFILL() \
    do { \
        cudaFuncSetAttribute(gguf_gemm_iq4_nl_prefill, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, IQ4_NL_SHMEM_BYTES); \
        gguf_gemm_iq4_nl_prefill<<< \
            dim3(ceil_div(size_n, IQ4_NL_N_TILE), ceil_div(size_m, IQ4_NL_M_TILE), 1), \
            dim3(256, 1, 1), \
            IQ4_NL_SHMEM_BYTES, stream>>>( \
                weights, input_q8_1, outputs, size_m, size_n, size_k, k_padded); \
    } while(0)

#define LAUNCH_GGUF_IQ4_NL_DECODE() \
    gguf_gemm_iq4_nl_decode<<< \
        dim3(ceil_div(size_n, 4), size_m, 1), \
        dim3(WARP_SIZE, 4, 1), \
        0, stream>>>( \
            weights, input_q8_1, outputs, size_m, size_n, size_k, k_padded)

static int gguf_gemm_device_cc() {
    static int cc = -1;
    if (cc < 0) {
        int device = 0;
        cudaDeviceProp prop{};
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        cc = prop.major * 10 + prop.minor;
    }
    return cc;
}

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
            if (use_prefill_kernel) {
                // 147 KB double-buffered shmem exceeds the 96 KB per-block limit on SM70/SM75.
                if (gguf_gemm_device_cc() >= 80) {
                    LAUNCH_GGUF_IQ4_NL_PREFILL();
                } else {
                    LAUNCH_GGUF_GEMM_PREFILL(
                        QK_IQ4_NL, QI_IQ4_NL, block_iq4_nl,
                        VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1);
                }
            } else {
                LAUNCH_GGUF_IQ4_NL_DECODE();
            }
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
