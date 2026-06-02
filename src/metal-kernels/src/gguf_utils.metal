// GGUF block type definitions and dequantization helpers for Metal.
// Ported from CUDA gguf/gguf.cuh (adapted from llama.cpp).
//
// Supports: Q8_0, Q2_K, Q4_K, Q5_K, Q6_K
//
// Copyright (c) 2025, Guoqing Bao. All rights reserved.
// Licensed under the Apache License, Version 2.0.

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Constants
// ============================================================

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32
#define QK8_1 32
#define GGUF_WARP_SIZE 32

// ============================================================
// GGUF Block Type Definitions
//
// Layout must match the CUDA/llama.cpp byte layout exactly.
// Metal uses packed structs to avoid padding.
// ============================================================

// Q8_0: 34 bytes per block, dequantizes to 32 floats
// Layout: half d (2B) + int8_t qs[32] (32B)
struct block_q8_0 {
    half    d;          // delta (scale)
    int8_t  qs[QK8_0];  // quantized values
};
static_assert(sizeof(block_q8_0) == 34, "wrong q8_0 block size");

// Q8_1: 36 bytes — used as runtime input quantization format
// Layout: half ds[2] (4B) + int8_t qs[32] (32B)
struct block_q8_1 {
    half    ds[2];      // ds[0] = delta, ds[1] = sum
    int8_t  qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "wrong q8_1 block size");

// Q2_K: QK_K/16 + QK_K/4 + 4 = 84 bytes, dequantizes to 256 floats
struct block_q2_K {
    uint8_t scales[QK_K / 16]; // 16 bytes: scales and mins, quantized with 4 bits
    uint8_t qs[QK_K / 4];     // 64 bytes: quants
    half    dm[2];             // 4 bytes: super-block scale for quantized scales/mins
};
static_assert(sizeof(block_q2_K) == 84, "wrong q2_K block size");

// Q3_K: QK_K/8 + QK_K/4 + K_SCALE_SIZE + 2 = 110 bytes
struct block_q3_K {
    uint8_t hmask[QK_K / 8];       // 32 bytes: high bit mask
    uint8_t qs[QK_K / 4];          // 64 bytes: quants (low 2 bits)
    uint8_t scales[K_SCALE_SIZE];  // 12 bytes: scales, quantized with 6 bits
    half    d;                      // 2 bytes: super-block scale
};
static_assert(sizeof(block_q3_K) == 110, "wrong q3_K block size");

// Q4_K: 4 + 12 + 128 = 144 bytes, dequantizes to 256 floats
struct block_q4_K {
    half    dm[2];                  // 4 bytes: super-block scale for quantized scales/mins
    uint8_t scales[3 * QK_K / 64]; // 12 bytes: scales, quantized with 6 bits
    uint8_t qs[QK_K / 2];          // 128 bytes: 4-bit quants
};
static_assert(sizeof(block_q4_K) == 144, "wrong q4_K block size");

// Q5_K: 4 + 12 + 32 + 128 = 176 bytes, dequantizes to 256 floats
struct block_q5_K {
    half    dm[2];                  // 4 bytes: super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE];  // 12 bytes: scales and mins, quantized with 6 bits
    uint8_t qh[QK_K / 8];          // 32 bytes: quants, high bit
    uint8_t qs[QK_K / 2];          // 128 bytes: quants, low 4 bits
};
static_assert(sizeof(block_q5_K) == 176, "wrong q5_K block size");

// Q6_K: 128 + 64 + 16 + 2 = 210 bytes, dequantizes to 256 floats
struct block_q6_K {
    uint8_t ql[QK_K / 2];    // 128 bytes: quants, lower 4 bits
    uint8_t qh[QK_K / 4];    // 64 bytes: quants, upper 2 bits
    int8_t  scales[QK_K / 16]; // 16 bytes: scales
    half    d;                 // 2 bytes: delta
};
static_assert(sizeof(block_q6_K) == 210, "wrong q6_K block size");


// ============================================================
// Metal DP4A equivalent — 4x int8 dot product
// ============================================================

inline int metal_dp4a(int a, int b, int c) {
    const thread int8_t *a8 = reinterpret_cast<const thread int8_t *>(&a);
    const thread int8_t *b8 = reinterpret_cast<const thread int8_t *>(&b);
    return c + int(a8[0]) * int(b8[0]) + int(a8[1]) * int(b8[1])
             + int(a8[2]) * int(b8[2]) + int(a8[3]) * int(b8[3]);
}

// ============================================================
// Packed int loaders (byte-level access helpers)
// ============================================================

inline int get_int_from_uint8(const device uint8_t *x8, int i32) {
    const device uint16_t *x16 = reinterpret_cast<const device uint16_t *>(x8 + 4 * i32);
    int x32 = 0;
    x32 |= int(x16[0]) << 0;
    x32 |= int(x16[1]) << 16;
    return x32;
}

inline int get_int_from_int8(const device int8_t *x8, int i32) {
    const device uint16_t *x16 = reinterpret_cast<const device uint16_t *>(x8 + 4 * i32);
    int x32 = 0;
    x32 |= int(x16[0]) << 0;
    x32 |= int(x16[1]) << 16;
    return x32;
}

inline int get_int_from_int8_aligned(const device int8_t *x8, int i32) {
    return *reinterpret_cast<const device int *>(x8 + 4 * i32);
}

inline int get_int_from_uint8_aligned(const device uint8_t *x8, int i32) {
    return *reinterpret_cast<const device int *>(x8 + 4 * i32);
}

// Threadgroup variants for shared memory access
inline int get_int_from_int8_tg(const threadgroup int8_t *x8, int i32) {
    const threadgroup uint16_t *x16 = reinterpret_cast<const threadgroup uint16_t *>(x8 + 4 * i32);
    int x32 = 0;
    x32 |= int(x16[0]) << 0;
    x32 |= int(x16[1]) << 16;
    return x32;
}

inline int get_int_from_int8_aligned_tg(const threadgroup int8_t *x8, int i32) {
    return *reinterpret_cast<const threadgroup int *>(x8 + 4 * i32);
}

// ============================================================
// Scale extraction helper for Q4_K/Q5_K
// ============================================================

inline void get_scale_min_k4(int j, const device uint8_t *q, thread uint8_t &d, thread uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// ============================================================
// Per-element dequantization functions
//
// Each function dequantizes from a single GGUF block at a given
// element index within that block.
// ============================================================

// Q8_0: block of 32 elements
inline float dequant_q8_0(const device block_q8_0 *blk, int idx) {
    return float(blk->d) * float(blk->qs[idx]);
}

// Q2_K: block of 256 elements
// qs[64] with interleaved packing: qs[32*n + l] contains 4 2-bit values
// for elements at positions 128*n + l + {0, 32, 64, 96}
inline float dequant_q2_K(const device block_q2_K *blk, int idx) {
    float dall = float(blk->dm[0]);
    float dmin = float(blk->dm[1]);

    int n = idx / 128;
    int wh = idx % 128;
    int byte_idx = 32 * n + (wh % 32);
    int shift = (wh / 32) * 2;

    uint8_t q = (blk->qs[byte_idx] >> shift) & 3;
    int group = idx / 16;
    uint8_t sc = blk->scales[group];

    return dall * float(sc & 0xF) * float(q) - dmin * float(sc >> 4);
}

// Q3_K: block of 256 elements
// qs[64]: 2-bit quants interleaved, hmask[32]: high bit, scales[12]: 6-bit packed
// Packing: qs[32*n + l] at shift 2*j holds quant for element 128*n + 32*j + l
inline float dequant_q3_K(const device block_q3_K *blk, int idx) {
    float d_all = float(blk->d);

    int is = idx / 16;

    int sc_val;
    if (is < 4) {
        sc_val = int((blk->scales[is] & 0xF) | (((blk->scales[is + 8] >> 0) & 3) << 4)) - 32;
    } else if (is < 8) {
        sc_val = int((blk->scales[is] & 0xF) | (((blk->scales[is + 4] >> 2) & 3) << 4)) - 32;
    } else if (is < 12) {
        sc_val = int((blk->scales[is - 8] >> 4) | (((blk->scales[is] >> 4) & 3) << 4)) - 32;
    } else {
        sc_val = int((blk->scales[is - 8] >> 4) | (((blk->scales[is - 4] >> 6) & 3) << 4)) - 32;
    }

    int n = idx / 128;
    int j = (idx % 128) / 32;
    int l = idx % 32;
    int ql_val = (int(blk->qs[32 * n + l]) >> (2 * j)) & 3;

    int high_bit = (int(blk->hmask[idx % 32]) >> (idx / 32)) & 1;
    int qval = ql_val - (high_bit ? 0 : 4);

    return d_all * float(sc_val) * float(qval);
}

// Q4_K: block of 256 elements
// qs[128]: 4-bit quants. qs[32*il+j] has low nibble for element 64*il+j
// and high nibble for element 64*il+j+32.
inline float dequant_q4_K(const device block_q4_K *blk, int idx) {
    float dall = float(blk->dm[0]);
    float dmin = float(blk->dm[1]);

    int il = idx / 64;
    int pos = idx % 64;
    int scale_idx = 2 * il + (pos >= 32 ? 1 : 0);

    uint8_t sc_val, m_val;
    if (scale_idx < 4) {
        sc_val = blk->scales[scale_idx] & 63;
        m_val = blk->scales[scale_idx + 4] & 63;
    } else {
        sc_val = (blk->scales[scale_idx + 4] & 0xF) | ((blk->scales[scale_idx - 4] >> 6) << 4);
        m_val = (blk->scales[scale_idx + 4] >> 4) | ((blk->scales[scale_idx] >> 6) << 4);
    }

    int nibble;
    if (pos < 32) {
        nibble = int(blk->qs[32 * il + pos]) & 0xF;
    } else {
        nibble = int(blk->qs[32 * il + (pos - 32)]) >> 4;
    }

    return dall * float(sc_val) * float(nibble) - dmin * float(m_val);
}

// Q5_K: block of 256 elements
// Same 64-element group packing as Q4_K for qs[128], plus qh[32] high bits.
inline float dequant_q5_K(const device block_q5_K *blk, int idx) {
    float dall = float(blk->dm[0]);
    float dmin = float(blk->dm[1]);

    int il = idx / 64;
    int pos = idx % 64;
    int scale_idx = 2 * il + (pos >= 32 ? 1 : 0);

    uint8_t sc_val, m_val;
    if (scale_idx < 4) {
        sc_val = blk->scales[scale_idx] & 63;
        m_val = blk->scales[scale_idx + 4] & 63;
    } else {
        sc_val = (blk->scales[scale_idx + 4] & 0xF) | ((blk->scales[scale_idx - 4] >> 6) << 4);
        m_val = (blk->scales[scale_idx + 4] >> 4) | ((blk->scales[scale_idx] >> 6) << 4);
    }

    int low_nibble;
    if (pos < 32) {
        low_nibble = int(blk->qs[32 * il + pos]) & 0xF;
    } else {
        low_nibble = int(blk->qs[32 * il + (pos - 32)]) >> 4;
    }
    int hm_bit = 2 * il + (pos >= 32 ? 1 : 0);
    int high_bit = (int(blk->qh[idx % 32]) >> hm_bit) & 1;

    int val = low_nibble + (high_bit << 4);
    return dall * float(sc_val) * float(val) - dmin * float(m_val);
}

// Q6_K: block of 256 elements
// ql[128]: 4-bit low quants, qh[64]: 2-bit high quants, scales[16], d
inline float dequant_q6_K(const device block_q6_K *blk, int idx) {
    float d = float(blk->d);

    int group = idx / 16;
    int8_t sc = blk->scales[group];

    int half_idx = idx / 128;
    int within_half = idx % 128;
    int qh_idx = half_idx * 32 + (within_half % 32);

    int ql_val, qh_val;
    if (within_half < 64) {
        int ql_idx = half_idx * 64 + within_half;
        ql_val = int(blk->ql[ql_idx]) & 0xF;
        qh_val = (int(blk->qh[qh_idx]) >> ((within_half / 32) * 2)) & 3;
    } else {
        int ql_idx = half_idx * 64 + (within_half - 64);
        ql_val = int(blk->ql[ql_idx]) >> 4;
        int qh_shift = ((within_half - 64) / 32) * 2 + 4;
        qh_val = (int(blk->qh[qh_idx]) >> qh_shift) & 3;
    }

    int val = ql_val | (qh_val << 4);
    return d * float(sc) * float(int8_t(val) - 32);
}

// ============================================================
// Block dequantization — dequantize entire block to float array
//
// tid = thread index within the dequantizing group
// threads_per_block = number of threads cooperating
// ============================================================

inline void dequant_block_q8_0_to_float(
    const device block_q8_0 *blk,
    threadgroup float *out,
    uint tid,
    uint threads_per_block
) {
    float d_val = float(blk->d);
    for (uint i = tid; i < QK8_0; i += threads_per_block) {
        out[i] = d_val * float(blk->qs[i]);
    }
}

inline void dequant_block_q2_K_to_float(
    const device block_q2_K *blk,
    threadgroup float *out,
    uint tid,
    uint threads_per_block
) {
    float dall = float(blk->dm[0]);
    float dmin = float(blk->dm[1]);

    for (uint i = tid; i < QK_K; i += threads_per_block) {
        int n = i / 128;
        int wh = i % 128;
        int byte_idx = 32 * n + (wh % 32);
        int shift = (wh / 32) * 2;
        uint8_t q = (blk->qs[byte_idx] >> shift) & 3;
        int group = i / 16;
        uint8_t sc = blk->scales[group];
        out[i] = dall * float(sc & 0xF) * float(q) - dmin * float(sc >> 4);
    }
}

inline void dequant_block_q3_K_to_float(
    const device block_q3_K *blk,
    threadgroup float *out,
    uint tid,
    uint threads_per_block
) {
    float d_all = float(blk->d);

    for (uint i = tid; i < QK_K; i += threads_per_block) {
        int is = i / 16;

        int sc_val;
        if (is < 4) {
            sc_val = int((blk->scales[is] & 0xF) | (((blk->scales[is + 8] >> 0) & 3) << 4)) - 32;
        } else if (is < 8) {
            sc_val = int((blk->scales[is] & 0xF) | (((blk->scales[is + 4] >> 2) & 3) << 4)) - 32;
        } else if (is < 12) {
            sc_val = int((blk->scales[is - 8] >> 4) | (((blk->scales[is] >> 4) & 3) << 4)) - 32;
        } else {
            sc_val = int((blk->scales[is - 8] >> 4) | (((blk->scales[is - 4] >> 6) & 3) << 4)) - 32;
        }

        int n = i / 128;
        int j = (i % 128) / 32;
        int l = i % 32;
        int ql_val = (int(blk->qs[32 * n + l]) >> (2 * j)) & 3;

        int high_bit = (int(blk->hmask[i % 32]) >> (i / 32)) & 1;
        int qval = ql_val - (high_bit ? 0 : 4);

        out[i] = d_all * float(sc_val) * float(qval);
    }
}

inline void dequant_block_q4_K_to_float(
    const device block_q4_K *blk,
    threadgroup float *out,
    uint tid,
    uint threads_per_block
) {
    float dall = float(blk->dm[0]);
    float dmin = float(blk->dm[1]);

    for (uint i = tid; i < QK_K; i += threads_per_block) {
        int il = i / 64;
        int pos = i % 64;
        int scale_idx = 2 * il + (pos >= 32 ? 1 : 0);

        uint8_t sc_val, m_val;
        if (scale_idx < 4) {
            sc_val = blk->scales[scale_idx] & 63;
            m_val = blk->scales[scale_idx + 4] & 63;
        } else {
            sc_val = (blk->scales[scale_idx + 4] & 0xF) | ((blk->scales[scale_idx - 4] >> 6) << 4);
            m_val = (blk->scales[scale_idx + 4] >> 4) | ((blk->scales[scale_idx] >> 6) << 4);
        }

        int nibble;
        if (pos < 32) {
            nibble = int(blk->qs[32 * il + pos]) & 0xF;
        } else {
            nibble = int(blk->qs[32 * il + (pos - 32)]) >> 4;
        }

        out[i] = dall * float(sc_val) * float(nibble) - dmin * float(m_val);
    }
}

inline void dequant_block_q5_K_to_float(
    const device block_q5_K *blk,
    threadgroup float *out,
    uint tid,
    uint threads_per_block
) {
    float dall = float(blk->dm[0]);
    float dmin = float(blk->dm[1]);

    for (uint i = tid; i < QK_K; i += threads_per_block) {
        int il = i / 64;
        int pos = i % 64;
        int scale_idx = 2 * il + (pos >= 32 ? 1 : 0);

        uint8_t sc_val, m_val;
        if (scale_idx < 4) {
            sc_val = blk->scales[scale_idx] & 63;
            m_val = blk->scales[scale_idx + 4] & 63;
        } else {
            sc_val = (blk->scales[scale_idx + 4] & 0xF) | ((blk->scales[scale_idx - 4] >> 6) << 4);
            m_val = (blk->scales[scale_idx + 4] >> 4) | ((blk->scales[scale_idx] >> 6) << 4);
        }

        int low_nibble;
        if (pos < 32) {
            low_nibble = int(blk->qs[32 * il + pos]) & 0xF;
        } else {
            low_nibble = int(blk->qs[32 * il + (pos - 32)]) >> 4;
        }
        int hm_bit = 2 * il + (pos >= 32 ? 1 : 0);
        int high_bit = (int(blk->qh[i % 32]) >> hm_bit) & 1;

        int val = low_nibble + (high_bit << 4);
        out[i] = dall * float(sc_val) * float(val) - dmin * float(m_val);
    }
}

inline void dequant_block_q6_K_to_float(
    const device block_q6_K *blk,
    threadgroup float *out,
    uint tid,
    uint threads_per_block
) {
    float d = float(blk->d);

    for (uint i = tid; i < QK_K; i += threads_per_block) {
        int group = i / 16;
        int8_t sc = blk->scales[group];

        int half_idx = i / 128;
        int within_half = i % 128;
        int ql_idx, qh_idx;

        if (within_half < 64) {
            ql_idx = half_idx * 64 + within_half;
            qh_idx = half_idx * 32 + (within_half % 32);
            int ql_val = int(blk->ql[ql_idx]) & 0xF;
            int qh_val = (int(blk->qh[qh_idx]) >> ((within_half / 32) * 2)) & 3;
            int val = ql_val | (qh_val << 4);
            out[i] = d * float(sc) * float(int8_t(val) - 32);
        } else {
            ql_idx = half_idx * 64 + (within_half - 64);
            qh_idx = half_idx * 32 + (within_half % 32);
            int ql_val = int(blk->ql[ql_idx]) >> 4;
            int qh_shift = ((within_half - 64) / 32) * 2 + 4;
            int qh_val = (int(blk->qh[qh_idx]) >> qh_shift) & 3;
            int val = ql_val | (qh_val << 4);
            out[i] = d * float(sc) * float(int8_t(val) - 32);
        }
    }
}

// ============================================================
// Unified dequant dispatcher — selects format by gguf_type ID
//
// gguf_type: 0=Q8_0, 1=Q4_K, 2=Q2_K, 3=Q3_K, 4=Q5_K, 5=Q6_K
// Returns: block size in bytes, qk (elements per block)
// ============================================================

constant constexpr int GGUF_BLOCK_SIZES[] = {
    34,  // Q8_0
    144, // Q4_K
    84,  // Q2_K
    110, // Q3_K
    176, // Q5_K
    210, // Q6_K
};

constant constexpr int GGUF_QK[] = {
    32,  // Q8_0
    256, // Q4_K
    256, // Q2_K
    256, // Q3_K
    256, // Q5_K
    256, // Q6_K
};
