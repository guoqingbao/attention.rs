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

// IQ2_XXS: 66 bytes per block, dequantizes to 256 floats
// Layout: half d (2B) + uint16_t qs[32] (64B)
struct block_iq2_xxs {
    half     d;
    uint16_t qs[QK_K / 8];
};
static_assert(sizeof(block_iq2_xxs) == 66, "wrong iq2_xxs block size");

// IQ2_XS: 74 bytes per block, dequantizes to 256 floats
// Layout: half d (2B) + uint16_t qs[32] (64B) + uint8_t scales[8] (8B)
struct block_iq2_xs {
    half     d;
    uint16_t qs[QK_K / 8];
    uint8_t  scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_xs) == 74, "wrong iq2_xs block size");

// IQ3_XXS: 98 bytes per block, dequantizes to 256 floats
// Layout: half d (2B) + uint8_t qs[96] (96B)
struct block_iq3_xxs {
    half    d;
    uint8_t qs[3 * QK_K / 8];
};
static_assert(sizeof(block_iq3_xxs) == 98, "wrong iq3_xxs block size");

// IQ4_XS: 136 bytes per block, dequantizes to 256 floats
// Layout: half d (2B) + uint16_t scales_h (2B) + uint8_t scales_l[4] (4B) + uint8_t qs[128] (128B)
struct block_iq4_xs {
    half    d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K / 64];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_iq4_xs) == 136, "wrong iq4_xs block size");


// ============================================================
// IQ Lookup Tables (ported from llama.cpp / CUDA gguf.cuh)
// ============================================================

constant const uint8_t ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

constant const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

constant const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

constant const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

constant const uint64_t iq2xs_grid[512] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL,
    0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL,
    0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL,
    0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL,
    0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL,
    0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL,
    0x080808082b191908ULL, 0x080808082b192b19ULL, 0x080808082b2b0808ULL, 0x0808081908080819ULL,
    0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL,
    0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL,
    0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL,
    0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL,
    0x08080819192b0808ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL,
    0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL,
    0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL,
    0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL,
    0x0808082b2b080808ULL, 0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL,
    0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL,
    0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL,
    0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL,
    0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL,
    0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL,
    0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908190819ULL,
    0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL,
    0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL,
    0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL,
    0x0808192b1908082bULL, 0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL,
    0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808082b2bULL, 0x08082b0808190819ULL,
    0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL,
    0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL,
    0x08082b1908190808ULL, 0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL,
    0x08082b2b08080808ULL, 0x08082b2b082b0808ULL, 0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL,
    0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL,
    0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL,
    0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL,
    0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL,
    0x0819080819191908ULL, 0x08190808192b0808ULL, 0x08190808192b2b2bULL, 0x081908082b080819ULL,
    0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL, 0x081908190808082bULL,
    0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL,
    0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL,
    0x081908192b080808ULL, 0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL,
    0x0819082b08081908ULL, 0x0819082b0808192bULL, 0x0819082b08190808ULL, 0x0819082b19080808ULL,
    0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL,
    0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL,
    0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL,
    0x08191908192b1908ULL, 0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL,
    0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192b08080808ULL, 0x0819192b08191908ULL,
    0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL,
    0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL,
    0x08192b2b2b2b2b19ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL,
    0x082b080808082b08ULL, 0x082b080808082b2bULL, 0x082b080808190819ULL, 0x082b080808191908ULL,
    0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL, 0x082b080819190808ULL,
    0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL,
    0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL,
    0x082b082b08080808ULL, 0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL,
    0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b1908082b2b19ULL,
    0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b19191919082bULL,
    0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL,
    0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL,
    0x082b2b0819191919ULL, 0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL,
    0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL, 0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL,
    0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL,
    0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL,
    0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL,
    0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL,
    0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192b0808ULL,
    0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL,
    0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL,
    0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL,
    0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL,
    0x190808192b2b082bULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL,
    0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL, 0x1908190808080808ULL,
    0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL,
    0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL,
    0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL,
    0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082b1908ULL,
    0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL, 0x1908192b08082b2bULL,
    0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL,
    0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL,
    0x19082b08192b082bULL, 0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL,
    0x19082b1919190808ULL, 0x19082b19192b2b19ULL, 0x19082b2b08081908ULL, 0x1919080808080808ULL,
    0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL,
    0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL,
    0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL,
    0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL,
    0x191908191908082bULL, 0x1919082b08080808ULL, 0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL,
    0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082b0819ULL,
    0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL,
    0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL,
    0x1919192b082b0819ULL, 0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL,
    0x19192b0808191908ULL, 0x19192b0819080819ULL, 0x19192b0819190808ULL, 0x19192b082b192b19ULL,
    0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL, 0x19192b2b2b081919ULL,
    0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL,
    0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL,
    0x192b081908080808ULL, 0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL,
    0x192b190808080808ULL, 0x192b19080819192bULL, 0x192b191908190808ULL, 0x192b191919080808ULL,
    0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL, 0x192b2b08192b2b2bULL,
    0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL,
    0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL,
    0x2b08080808190819ULL, 0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL,
    0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808082b080808ULL,
    0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL, 0x2b08081908080819ULL,
    0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL,
    0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL,
    0x2b08082b2b080808ULL, 0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL,
    0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL,
    0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b0819082b082b19ULL,
    0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL,
    0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL,
    0x2b082b0819192b2bULL, 0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL,
    0x2b082b190808192bULL, 0x2b082b2b082b082bULL, 0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL,
    0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL,
    0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL,
    0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL,
    0x2b19082b2b082b19ULL, 0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL,
    0x2b19190819190808ULL, 0x2b19190819192b08ULL, 0x2b191919082b2b19ULL, 0x2b1919192b190808ULL,
    0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL, 0x2b192b082b2b192bULL,
    0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL,
    0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL,
    0x2b2b0808082b2b2bULL, 0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL,
    0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL,
    0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL, 0x2b2b190819080808ULL,
    0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL,
    0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL,
    0x2b2b2b082b2b2b08ULL, 0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL,
    0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL, 0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL,
};

constant const uint32_t iq3xxs_grid[256] = {
    0x04040404U, 0x04040414U, 0x04040424U, 0x04040c0cU,
    0x04040c1cU, 0x04040c3eU, 0x04041404U, 0x04041414U,
    0x04041c0cU, 0x04042414U, 0x04043e1cU, 0x04043e2cU,
    0x040c040cU, 0x040c041cU, 0x040c0c04U, 0x040c0c14U,
    0x040c140cU, 0x040c142cU, 0x040c1c04U, 0x040c1c14U,
    0x040c240cU, 0x040c2c24U, 0x040c3e04U, 0x04140404U,
    0x04140414U, 0x04140424U, 0x04140c0cU, 0x04141404U,
    0x04141414U, 0x04141c0cU, 0x04141c1cU, 0x04141c3eU,
    0x04142c0cU, 0x04142c3eU, 0x04143e2cU, 0x041c040cU,
    0x041c043eU, 0x041c0c04U, 0x041c0c14U, 0x041c142cU,
    0x041c3e04U, 0x04240c1cU, 0x04241c3eU, 0x04242424U,
    0x04242c3eU, 0x04243e1cU, 0x04243e2cU, 0x042c040cU,
    0x042c043eU, 0x042c1c14U, 0x042c2c14U, 0x04341c2cU,
    0x04343424U, 0x043e0c04U, 0x043e0c24U, 0x043e0c34U,
    0x043e241cU, 0x043e340cU, 0x0c04040cU, 0x0c04041cU,
    0x0c040c04U, 0x0c040c14U, 0x0c04140cU, 0x0c04141cU,
    0x0c041c04U, 0x0c041c14U, 0x0c041c24U, 0x0c04243eU,
    0x0c042c04U, 0x0c0c0404U, 0x0c0c0414U, 0x0c0c0c0cU,
    0x0c0c1404U, 0x0c0c1414U, 0x0c14040cU, 0x0c14041cU,
    0x0c140c04U, 0x0c140c14U, 0x0c14140cU, 0x0c141c04U,
    0x0c143e14U, 0x0c1c0404U, 0x0c1c0414U, 0x0c1c1404U,
    0x0c1c1c0cU, 0x0c1c2434U, 0x0c1c3434U, 0x0c24040cU,
    0x0c24042cU, 0x0c242c04U, 0x0c2c1404U, 0x0c2c1424U,
    0x0c2c2434U, 0x0c2c3e0cU, 0x0c34042cU, 0x0c3e1414U,
    0x0c3e2404U, 0x14040404U, 0x14040414U, 0x14040c0cU,
    0x14040c1cU, 0x14041404U, 0x14041414U, 0x14041434U,
    0x14041c0cU, 0x14042414U, 0x140c040cU, 0x140c041cU,
    0x140c042cU, 0x140c0c04U, 0x140c0c14U, 0x140c140cU,
    0x140c1c04U, 0x140c341cU, 0x140c343eU, 0x140c3e04U,
    0x14140404U, 0x14140414U, 0x14140c0cU, 0x14140c3eU,
    0x14141404U, 0x14141414U, 0x14141c3eU, 0x14142404U,
    0x14142c2cU, 0x141c040cU, 0x141c0c04U, 0x141c0c24U,
    0x141c3e04U, 0x141c3e24U, 0x14241c2cU, 0x14242c1cU,
    0x142c041cU, 0x142c143eU, 0x142c240cU, 0x142c3e24U,
    0x143e040cU, 0x143e041cU, 0x143e0c34U, 0x143e242cU,
    0x1c04040cU, 0x1c040c04U, 0x1c040c14U, 0x1c04140cU,
    0x1c04141cU, 0x1c042c04U, 0x1c04342cU, 0x1c043e14U,
    0x1c0c0404U, 0x1c0c0414U, 0x1c0c1404U, 0x1c0c1c0cU,
    0x1c0c2424U, 0x1c0c2434U, 0x1c14040cU, 0x1c14041cU,
    0x1c140c04U, 0x1c14142cU, 0x1c142c14U, 0x1c143e14U,
    0x1c1c0c0cU, 0x1c1c1c1cU, 0x1c241c04U, 0x1c24243eU,
    0x1c243e14U, 0x1c2c0404U, 0x1c2c0434U, 0x1c2c1414U,
    0x1c2c2c2cU, 0x1c340c24U, 0x1c341c34U, 0x1c34341cU,
    0x1c3e1c1cU, 0x1c3e3404U, 0x24040424U, 0x24040c3eU,
    0x24041c2cU, 0x24041c3eU, 0x24042c1cU, 0x24042c3eU,
    0x240c3e24U, 0x24141404U, 0x24141c3eU, 0x24142404U,
    0x24143404U, 0x24143434U, 0x241c043eU, 0x241c242cU,
    0x24240424U, 0x24242c0cU, 0x24243424U, 0x242c142cU,
    0x242c241cU, 0x242c3e04U, 0x243e042cU, 0x243e0c04U,
    0x243e0c14U, 0x243e1c04U, 0x2c040c14U, 0x2c04240cU,
    0x2c043e04U, 0x2c0c0404U, 0x2c0c0434U, 0x2c0c1434U,
    0x2c0c2c2cU, 0x2c140c24U, 0x2c141c14U, 0x2c143e14U,
    0x2c1c0414U, 0x2c1c2c1cU, 0x2c240c04U, 0x2c24141cU,
    0x2c24143eU, 0x2c243e14U, 0x2c2c0414U, 0x2c2c1c0cU,
    0x2c342c04U, 0x2c3e1424U, 0x2c3e2414U, 0x34041424U,
    0x34042424U, 0x34042434U, 0x34043424U, 0x340c140cU,
    0x340c340cU, 0x34140c3eU, 0x34143424U, 0x341c1c04U,
    0x341c1c34U, 0x34242424U, 0x342c042cU, 0x342c2c14U,
    0x34341c1cU, 0x343e041cU, 0x343e140cU, 0x3e04041cU,
    0x3e04042cU, 0x3e04043eU, 0x3e040c04U, 0x3e041c14U,
    0x3e042c14U, 0x3e0c1434U, 0x3e0c2404U, 0x3e140c14U,
    0x3e14242cU, 0x3e142c14U, 0x3e1c0404U, 0x3e1c0c2cU,
    0x3e1c1c1cU, 0x3e1c3404U, 0x3e24140cU, 0x3e24240cU,
    0x3e2c0404U, 0x3e2c0414U, 0x3e2c1424U, 0x3e341c04U,
};


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
    34,  // 0: Q8_0
    144, // 1: Q4_K
    84,  // 2: Q2_K
    110, // 3: Q3_K
    176, // 4: Q5_K
    210, // 5: Q6_K
    66,  // 6: IQ2_XXS
    74,  // 7: IQ2_XS
    98,  // 8: IQ3_XXS
    136, // 9: IQ4_XS
};

constant constexpr int GGUF_QK[] = {
    32,  // 0: Q8_0
    256, // 1: Q4_K
    256, // 2: Q2_K
    256, // 3: Q3_K
    256, // 4: Q5_K
    256, // 5: Q6_K
    256, // 6: IQ2_XXS
    256, // 7: IQ2_XS
    256, // 8: IQ3_XXS
    256, // 9: IQ4_XS
};
