/*
 * Small-M GGUF matmul: each warp computes several output rows, unpacks every
 * 32-value weight subblock once, and reuses it across all tokens in the batch.
 *
 * Candle's MMVQ kernels (`mul_mat_vec_q*_cuda8`) call vec_dot once per
 * (row, token), repeating the weight unpack, scale decode and weight loads for
 * every token. ncu on Q4_K at batch 8 shows that path is instruction/L1 bound
 * (SM 60%, L1 57%, DRAM 16%). Here lane l owns the 32-value subblock
 * it*32 + l (exactly one activation block), unpacks it once per row, and runs
 * dp4a against each token. Activation blocks are loaded once per token and
 * shared across the ROWS rows handled by the warp.
 *
 * A cp.async staged variant was measured slower: the weight payload is already
 * reused from registers across NCOLS tokens, so the shared-memory round-trip
 * only added LDS instructions. The direct register path is kept.
 */
#include "gguf/gguf.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int WARP = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int Q8_BLOCK = QK8_1; // 32 values

// Activation block with a 16-byte aligned payload so the inner loop can use
// int4 loads. The 48-byte stride keeps every qs at offset 0 mod 16.
struct alignas(16) q8_block_p {
    int8_t qs[Q8_BLOCK]; // offset 0
    float d;             // offset 32
    float sum;           // offset 36
    int32_t pad0;
    int32_t pad1;
};
static_assert(sizeof(q8_block_p) == 48, "unexpected padded q8 block size");

enum GgufFmt : int {
    FMT_Q8_0 = 0,
    FMT_Q4_K = 1,
    FMT_Q5_K = 4,
    FMT_Q6_K = 5,
};

static_assert(sizeof(block_q4_K) == 144, "unexpected Q4_K block size");
static_assert(sizeof(block_q5_K) == 176, "unexpected Q5_K block size");
static_assert(sizeof(block_q6_K) == 210, "unexpected Q6_K block size");
static_assert(sizeof(block_q8_0) == 34, "unexpected Q8_0 block size");

// Load 8 packed uint32 words (32 bytes). ALIGN is the guaranteed alignment of
// src in bytes.
template <int ALIGN>
__device__ __forceinline__ void load_words8(const uint8_t *src, int *w) {
    if constexpr (ALIGN >= 16) {
        const int4 a = *reinterpret_cast<const int4 *>(src);
        const int4 b = *reinterpret_cast<const int4 *>(src + 16);
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
        w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    } else if constexpr (ALIGN >= 8) {
        const int2 *s = reinterpret_cast<const int2 *>(src);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int2 v = s[i];
            w[2 * i] = v.x;
            w[2 * i + 1] = v.y;
        }
    } else if constexpr (ALIGN >= 4) {
        const int *s = reinterpret_cast<const int *>(src);
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = s[i];
    } else {
        const uint16_t *s = reinterpret_cast<const uint16_t *>(src);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[i] = static_cast<int>(s[2 * i]) | (static_cast<int>(s[2 * i + 1]) << 16);
        }
    }
}

// Unpack one 32-value subblock of a K-quant weight block into int8 lanes.
// `sub` is the 0..7 subblock index inside the superblock.
template <int FMT>
__device__ __forceinline__ void unpack_subblock(
    const uint8_t *block, int sub, uint32_t *w, float &scale, float &minv,
    float &half_scale0, float &half_scale1) {
    static_assert(FMT == FMT_Q4_K || FMT == FMT_Q5_K || FMT == FMT_Q6_K,
                  "K-quant only");
    if constexpr (FMT == FMT_Q6_K) {
        const block_q6_K *b = reinterpret_cast<const block_q6_K *>(block);
        const int half128 = sub / 4;
        const int s8 = sub & 7;
        const uint8_t *ql = b->ql + 64 * half128 + 32 * (s8 & 1);
        const uint8_t *qh = b->qh + 32 * half128;
        const int shift = 2 * (s8 & 3);
        const bool low = (s8 & 3) < 2;

        uint32_t qw[8];
        load_words8<2>(ql, reinterpret_cast<int *>(qw));
        uint32_t hw[8];
        load_words8<2>(qh, reinterpret_cast<int *>(hw));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t nib = low ? (qw[i] & 0x0F0F0F0Fu) : ((qw[i] >> 4) & 0x0F0F0F0Fu);
            const uint32_t hi = (hw[i] >> shift) & 0x03030303u;
            w[i] = static_cast<uint32_t>(nib | (hi << 4)) - 0x20202020u;
        }
        scale = __half2float(b->d);
        minv = 0.0f;
        half_scale0 = static_cast<float>(b->scales[2 * s8]);
        half_scale1 = static_cast<float>(b->scales[2 * s8 + 1]);
        return;
    }

    const int j4 = sub / 2;
    const bool high = (sub & 1) != 0;
    const half2 dm = *reinterpret_cast<const half2 *>(block);
    const float2 dmf = __half22float2(dm);
    uint8_t sc, m;
    get_scale_min_k4(sub, reinterpret_cast<const uint8_t *>(block) + 4, sc, m);

    constexpr int qs_offset = (FMT == FMT_Q5_K) ? 48 : 16;
    uint32_t qw[8];
    load_words8<16>(reinterpret_cast<const uint8_t *>(block) + qs_offset + 32 * j4,
                    reinterpret_cast<int *>(qw));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if constexpr (FMT == FMT_Q4_K) {
            w[i] = high ? ((qw[i] >> 4) & 0x0F0F0F0Fu) : (qw[i] & 0x0F0F0F0Fu);
        } else {
            const block_q5_K *b = reinterpret_cast<const block_q5_K *>(block);
            const uint32_t hbit =
                (*(reinterpret_cast<const uint32_t *>(b->qh) + i) >> sub) & 0x01010101u;
            const uint32_t nib = high ? ((qw[i] >> 4) & 0x0F0F0F0Fu) : (qw[i] & 0x0F0F0F0Fu);
            w[i] = nib | (hbit << 4);
        }
    }
    scale = dmf.x * static_cast<float>(sc);
    minv = dmf.y * static_cast<float>(m);
    half_scale0 = 0.0f;
    half_scale1 = 0.0f;
}

template <int FMT, int NCOLS, int ROWS>
__global__ void gguf_gemm_multi_kernel(
    const uint8_t *__restrict__ weights,
    const q8_block_p *__restrict__ inputs,
    float *__restrict__ outputs,
    int size_m, int size_n, int size_k, int k_padded) {
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * ROWS;
    if (row0 >= size_n) return;

    const int subblocks = size_k / Q8_BLOCK;
    const int iters = (subblocks + WARP - 1) / WARP;
    const int y_stride = k_padded / Q8_BLOCK;

    constexpr size_t kquant_bytes = FMT == FMT_Q4_K   ? sizeof(block_q4_K)
                                    : FMT == FMT_Q5_K ? sizeof(block_q5_K)
                                                      : sizeof(block_q6_K);
    const size_t w_row_bytes =
        FMT == FMT_Q8_0
            ? static_cast<size_t>(size_k / QK8_0) * sizeof(block_q8_0)
            : static_cast<size_t>(size_k / QK_K) * kquant_bytes;

    float acc[ROWS][NCOLS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) acc[r][j] = 0.0f;

    for (int it = 0; it < iters; ++it) {
        const int sub = it * WARP + lane;
        if (sub >= subblocks) break;

        uint32_t w[ROWS][8];
        float scale[ROWS], minv[ROWS], hs0[ROWS], hs1[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = row0 + r;
            if (row >= size_n) break;
            const uint8_t *wrow = weights + static_cast<size_t>(row) * w_row_bytes;
            if constexpr (FMT == FMT_Q8_0) {
                const block_q8_0 *b = reinterpret_cast<const block_q8_0 *>(wrow) + sub;
                load_words8<2>(reinterpret_cast<const uint8_t *>(b->qs),
                               reinterpret_cast<int *>(w[r]));
                scale[r] = __half2float(b->d);
                minv[r] = 0.0f;
            } else {
                const int super = sub / 8;
                const int s8 = sub & 7;
                constexpr size_t bsz = (FMT == FMT_Q6_K) ? sizeof(block_q6_K)
                                       : (FMT == FMT_Q5_K) ? sizeof(block_q5_K)
                                                           : sizeof(block_q4_K);
                const uint8_t *block = wrow + static_cast<size_t>(super) * bsz;
                unpack_subblock<FMT>(block, s8, w[r], scale[r], minv[r], hs0[r], hs1[r]);
            }
        }

#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            if (j >= size_m) break;
            const q8_block_p *yb = inputs + static_cast<size_t>(j) * y_stride + sub;
            int u[8];
            load_words8<16>(reinterpret_cast<const uint8_t *>(yb->qs),
                            reinterpret_cast<int *>(u));
            const float dsx = yb->d;
            const float dsy = yb->sum;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (row0 + r >= size_n) break;
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) sumi = ggml_cuda_dp4a(w[r][i], u[i], sumi);
                if constexpr (FMT == FMT_Q6_K) {
                    int sumi0 = 0, sumi1 = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) sumi0 = ggml_cuda_dp4a(w[r][i], u[i], sumi0);
#pragma unroll
                    for (int i = 4; i < 8; ++i) sumi1 = ggml_cuda_dp4a(w[r][i], u[i], sumi1);
                    acc[r][j] += dsx * scale[r] * (hs0[r] * static_cast<float>(sumi0) +
                                                   hs1[r] * static_cast<float>(sumi1));
                } else if constexpr (FMT == FMT_Q8_0) {
                    acc[r][j] += dsx * scale[r] * static_cast<float>(sumi);
                } else {
                    // w·y = d_y·(d_all·sc)·Σ(q·q8) − (d_min·m)·Σy. The min term
                    // is already in activation units, so it must not be scaled
                    // by d_y again.
                    acc[r][j] += dsx * scale[r] * static_cast<float>(sumi) - minv[r] * dsy;
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int row = row0 + r;
        if (row >= size_n) break;
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            if (j >= size_m) break;
            const float v = warp_reduce_sum(acc[r][j]);
            if (lane == 0) outputs[static_cast<size_t>(j) * size_n + row] = v;
        }
    }
}

__global__ void gguf_quantize_q8_1_multi(
    const float *__restrict__ x, void *__restrict__ vy, int kx, int kx_padded) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) return;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_padded = iy * kx_padded + ix;
    q8_block_p *y = reinterpret_cast<q8_block_p *>(vy);
    const int ib = i_padded / QK8_1;
    const int iqs = i_padded % QK8_1;

    const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;
    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);
    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(xi / d));
    y[ib].qs[iqs] = q;
    if (iqs > 0) return;
    y[ib].d = d;
    y[ib].sum = sum;
}

} // namespace

extern "C" void gguf_gemm_multi(
    const float *inputs, const void *weights, float *outputs, int size_m, int size_n,
    int size_k, int quant_type, cudaStream_t stream) {
    constexpr int k_row_padding = 512;
    if (size_m <= 0 || size_n <= 0 || size_k <= 0 || (size_k % Q8_BLOCK) != 0) return;

    const int k_padded = ((size_k + k_row_padding - 1) / k_row_padding) * k_row_padding;
    const int input_blocks = (k_padded / QK8_1) * size_m;
    void *input_q8 = nullptr;
    if (cudaMallocAsync(&input_q8, static_cast<size_t>(input_blocks) * sizeof(q8_block_p),
                        stream) != cudaSuccess) {
        return;
    }
    constexpr int quant_threads = 256;
    dim3 qgrid((k_padded + quant_threads - 1) / quant_threads, size_m, 1);
    gguf_quantize_q8_1_multi<<<qgrid, dim3(quant_threads, 1, 1), 0, stream>>>(
        inputs, input_q8, size_k, k_padded);

    const dim3 block(WARP, WARPS_PER_BLOCK, 1);
    const q8_block_p *y = reinterpret_cast<const q8_block_p *>(input_q8);
    const uint8_t *w = reinterpret_cast<const uint8_t *>(weights);

#define GGUF_MULTI_LAUNCH(FMT, NCOLS, ROWS)                                      \
    do {                                                                         \
        dim3 grid_((size_n + WARPS_PER_BLOCK * ROWS - 1) /                        \
                       (WARPS_PER_BLOCK * ROWS), 1, 1);                           \
        gguf_gemm_multi_kernel<FMT, NCOLS, ROWS><<<grid_, block, 0, stream>>>(    \
            w, y, outputs, size_m, size_n, size_k, k_padded);                     \
    } while (0)

#define GGUF_MULTI_DISPATCH(FMT)                                                 \
    do {                                                                         \
        switch (size_m) {                                                        \
            case 1: GGUF_MULTI_LAUNCH(FMT, 1, 1); break;                          \
            case 2: GGUF_MULTI_LAUNCH(FMT, 2, 2); break;                          \
            case 3:                                                              \
            case 4: GGUF_MULTI_LAUNCH(FMT, 4, 2); break;                          \
            case 5:                                                              \
            case 6:                                                              \
            case 7:                                                              \
            case 8: GGUF_MULTI_LAUNCH(FMT, 8, 4); break;                          \
            default:                                                             \
                if (size_m <= 16) GGUF_MULTI_LAUNCH(FMT, 16, 2);                  \
                else GGUF_MULTI_LAUNCH(FMT, 32, 2);                               \
                break;                                                           \
        }                                                                        \
    } while (0)

    switch (quant_type) {
        case 0: GGUF_MULTI_DISPATCH(FMT_Q8_0); break;
        case 1: GGUF_MULTI_DISPATCH(FMT_Q4_K); break;
        case 4: GGUF_MULTI_DISPATCH(FMT_Q5_K); break;
        case 5: GGUF_MULTI_DISPATCH(FMT_Q6_K); break;
        default: break;
    }
#undef GGUF_MULTI_DISPATCH
#undef GGUF_MULTI_LAUNCH
    cudaFreeAsync(input_q8, stream);
}
