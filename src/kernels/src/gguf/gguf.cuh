// Kernels adapted from llama.cpp ggml-cuda.cu
// https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda.cu
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

#define GGML_UNUSED(x) (void)(x)
#define GGML_CUDA_ASSUME(x)

#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

#undef GGML_CUDA_F16
#define GGML_CUDA_DMMV_X 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define K_QUANTS_PER_ITERATION 2

typedef uint16_t ggml_fp16_t;
typedef float dfloat; // dequantize float
typedef float2 dfloat2;
typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, dfloat2 & v);

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

static __device__ __forceinline__ int get_int_from_int8(const int8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __device__ __forceinline__ int get_int_from_uint8_aligned(const uint8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}


#define WARP_SIZE 32
#define CUDART_HMAX     11070 // CUDA 11.7, min. ver. for which __hmax and __hmax2 are known to work (may be higher than needed)

#define CC_PASCAL     600
#define MIN_CC_DP4A   610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define CC_VOLTA      700
#define CC_OFFSET_AMD 1000000
#define CC_RDNA1      (CC_OFFSET_AMD + 1010)
#define CC_RDNA2      (CC_OFFSET_AMD + 1030)
#define CC_RDNA3      (CC_OFFSET_AMD + 1100)

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= MIN_CC_DP4A
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}


#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8
#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_0_AMPERE 4
#define  MMQ_Y_Q4_0_AMPERE 32
#define NWARPS_Q4_0_AMPERE 4
#else
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#endif
#define  MMQ_X_Q4_0_PASCAL 64
#define  MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

#define  MMQ_X_Q4_1_RDNA2  64
#define  MMQ_Y_Q4_1_RDNA2  128
#define NWARPS_Q4_1_RDNA2  8
#define  MMQ_X_Q4_1_RDNA1  64
#define  MMQ_Y_Q4_1_RDNA1  64
#define NWARPS_Q4_1_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_1_AMPERE 4
#define  MMQ_Y_Q4_1_AMPERE 32
#define NWARPS_Q4_1_AMPERE 4
#else
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 4
#endif
#define  MMQ_X_Q4_1_PASCAL 64
#define  MMQ_Y_Q4_1_PASCAL 64
#define NWARPS_Q4_1_PASCAL 8

#define  MMQ_X_Q5_0_RDNA2  64
#define  MMQ_Y_Q5_0_RDNA2  128
#define NWARPS_Q5_0_RDNA2  8
#define  MMQ_X_Q5_0_RDNA1  64
#define  MMQ_Y_Q5_0_RDNA1  64
#define NWARPS_Q5_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_0_AMPERE 4
#define  MMQ_Y_Q5_0_AMPERE 32
#define NWARPS_Q5_0_AMPERE 4
#else
#define  MMQ_X_Q5_0_AMPERE 128
#define  MMQ_Y_Q5_0_AMPERE 64
#define NWARPS_Q5_0_AMPERE 4
#endif
#define  MMQ_X_Q5_0_PASCAL 64
#define  MMQ_Y_Q5_0_PASCAL 64
#define NWARPS_Q5_0_PASCAL 8

#define  MMQ_X_Q5_1_RDNA2  64
#define  MMQ_Y_Q5_1_RDNA2  128
#define NWARPS_Q5_1_RDNA2  8
#define  MMQ_X_Q5_1_RDNA1  64
#define  MMQ_Y_Q5_1_RDNA1  64
#define NWARPS_Q5_1_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_1_AMPERE 4
#define  MMQ_Y_Q5_1_AMPERE 32
#define NWARPS_Q5_1_AMPERE 4
#else
#define  MMQ_X_Q5_1_AMPERE 128
#define  MMQ_Y_Q5_1_AMPERE 64
#define NWARPS_Q5_1_AMPERE 4
#endif
#define  MMQ_X_Q5_1_PASCAL 64
#define  MMQ_Y_Q5_1_PASCAL 64
#define NWARPS_Q5_1_PASCAL 8

#define  MMQ_X_Q8_0_RDNA2  64
#define  MMQ_Y_Q8_0_RDNA2  128
#define NWARPS_Q8_0_RDNA2  8
#define  MMQ_X_Q8_0_RDNA1  64
#define  MMQ_Y_Q8_0_RDNA1  64
#define NWARPS_Q8_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q8_0_AMPERE 4
#define  MMQ_Y_Q8_0_AMPERE 32
#define NWARPS_Q8_0_AMPERE 4
#else
#define  MMQ_X_Q8_0_AMPERE 128
#define  MMQ_Y_Q8_0_AMPERE 64
#define NWARPS_Q8_0_AMPERE 4
#endif
#define  MMQ_X_Q8_0_PASCAL 64
#define  MMQ_Y_Q8_0_PASCAL 64
#define NWARPS_Q8_0_PASCAL 8

#define  MMQ_X_Q2_K_RDNA2  64
#define  MMQ_Y_Q2_K_RDNA2  128
#define NWARPS_Q2_K_RDNA2  8
#define  MMQ_X_Q2_K_RDNA1  128
#define  MMQ_Y_Q2_K_RDNA1  32
#define NWARPS_Q2_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q2_K_AMPERE 4
#define  MMQ_Y_Q2_K_AMPERE 32
#define NWARPS_Q2_K_AMPERE 4
#else
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 4
#endif
#define  MMQ_X_Q2_K_PASCAL 64
#define  MMQ_Y_Q2_K_PASCAL 64
#define NWARPS_Q2_K_PASCAL 8

#define  MMQ_X_Q3_K_RDNA2  128
#define  MMQ_Y_Q3_K_RDNA2  64
#define NWARPS_Q3_K_RDNA2  8
#define  MMQ_X_Q3_K_RDNA1  32
#define  MMQ_Y_Q3_K_RDNA1  128
#define NWARPS_Q3_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q3_K_AMPERE 4
#define  MMQ_Y_Q3_K_AMPERE 32
#define NWARPS_Q3_K_AMPERE 4
#else
#define  MMQ_X_Q3_K_AMPERE 128
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 4
#endif
#define  MMQ_X_Q3_K_PASCAL 64
#define  MMQ_Y_Q3_K_PASCAL 64
#define NWARPS_Q3_K_PASCAL 8

#define  MMQ_X_Q4_K_RDNA2  64
#define  MMQ_Y_Q4_K_RDNA2  128
#define NWARPS_Q4_K_RDNA2  8
#define  MMQ_X_Q4_K_RDNA1  32
#define  MMQ_Y_Q4_K_RDNA1  64
#define NWARPS_Q4_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_K_AMPERE 4
#define  MMQ_Y_Q4_K_AMPERE 32
#define NWARPS_Q4_K_AMPERE 4
#else
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#endif
#define  MMQ_X_Q4_K_PASCAL 64
#define  MMQ_Y_Q4_K_PASCAL 64
#define NWARPS_Q4_K_PASCAL 8

#define  MMQ_X_Q5_K_RDNA2  64
#define  MMQ_Y_Q5_K_RDNA2  128
#define NWARPS_Q5_K_RDNA2  8
#define  MMQ_X_Q5_K_RDNA1  32
#define  MMQ_Y_Q5_K_RDNA1  64
#define NWARPS_Q5_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_K_AMPERE 4
#define  MMQ_Y_Q5_K_AMPERE 32
#define NWARPS_Q5_K_AMPERE 4
#else
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 4
#endif
#define  MMQ_X_Q5_K_PASCAL 64
#define  MMQ_Y_Q5_K_PASCAL 64
#define NWARPS_Q5_K_PASCAL 8

#define  MMQ_X_Q6_K_RDNA2  64
#define  MMQ_Y_Q6_K_RDNA2  128
#define NWARPS_Q6_K_RDNA2  8
#define  MMQ_X_Q6_K_RDNA1  32
#define  MMQ_Y_Q6_K_RDNA1  64
#define NWARPS_Q6_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q6_K_AMPERE 4
#define  MMQ_Y_Q6_K_AMPERE 32
#define NWARPS_Q6_K_AMPERE 4
#else
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 64
#define NWARPS_Q6_K_AMPERE 4
#endif
#define  MMQ_X_Q6_K_PASCAL 64
#define  MMQ_Y_Q6_K_PASCAL 64
#define NWARPS_Q6_K_PASCAL 8


// QK = number of values after dequantization
// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
    half    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
    half2   dm;             // dm.x = delta, dm.y = min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(ggml_fp16_t) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
    half d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
    half2 dm;               // dm.x = delta, dm.y = min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
    half    d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct {
    half2   ds;             // ds.x = delta, ds.y = sum
    int8_t  qs[QK8_0];      // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_fp16_t) + QK8_0, "wrong q8_1 block size/padding");

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);

#define QR2_K 4
#define QI2_K (QK_K / (4*QR2_K))
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half2 dm;                // super-block scale for quantized scales/mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

#define QR3_K 4
#define QI3_K (QK_K / (4*QR3_K))
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
#ifdef GGML_QKK_64
    uint8_t scales[2]; // scales, quantized with 8 bits
#else
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
#endif
    half d;             // super-block scale
} block_q3_K;
//static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + K_SCALE_SIZE, "wrong q3_K block size/padding");

#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))
#ifdef GGML_QKK_64
typedef struct {
    half    dm[2];             // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == sizeof(half2) + QK_K/2 + 2, "wrong q4_K block size/padding");
#else
typedef struct {
    half2 dm;                  // super-block scale for quantized scales/mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + 3*QK_K/64 + QK_K/2, "wrong q4_K block size/padding");
#endif

#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))
#ifdef GGML_QKK_64
typedef struct {
    half d;                  // super-block scale
    int8_t scales[QK_K/16];  // block scales
    uint8_t qh[QK_K/8];      // quants, high bit
    uint8_t qs[QK_K/2];      // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == sizeof(ggml_fp16_t) + QK_K/2 + QK_K/8 + QK_K/16, "wrong q5_K block size/padding");
#else
typedef struct {
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");
#endif

#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
typedef struct {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    half    d;         // delta
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + 13*QK_K/16, "wrong q6_K block size/padding");

// In llama.cpp this is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");


// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const float & d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * sumi;
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
#else
    const float2 dm8f = __half22float2(dm8);
    const float2 ds8f = __half22float2(ds8);
    const float d8d8 = dm8f.x * ds8f.x;
    const float m8s8 = dm8f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float & d8) {

    int sumi_d = 0;
    int sumi_m = 0;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        int sumi_d_sc = 0;

        const int sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = ggml_cuda_dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m    = ggml_cuda_dp4a(m,    u[i], sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    const float2 dm2f = __half22float2(dm2);

    return d8 * (dm2f.x*sumi_d - dm2f.y*sumi_m);
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {

    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = ggml_cuda_dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = ggml_cuda_dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = ggml_cuda_dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = ggml_cuda_dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = ggml_cuda_dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc[i0/2+0]*sumi_d.x + sc[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}


static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

#ifndef GGML_QKK_64
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);

#else

    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    const uint16_t * a = (const uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    const float dall = bq4_K->dm[0];
    const float dmin = bq4_K->dm[1];

    const float d8_1 = __low2float(bq8_1[0].ds);
    const float d8_2 = __low2float(bq8_1[1].ds);

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * q4 = (const int *)bq4_K->qs + (iqs/2);
    const int v1 = q4[0];
    const int v2 = q4[4];

    const int dot1 = ggml_cuda_dp4a(ui2, v2 & 0x0f0f0f0f, ggml_cuda_dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    const int dot2 = ggml_cuda_dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, ggml_cuda_dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    const int dot3 = ggml_cuda_dp4a(0x01010101, ui2, ggml_cuda_dp4a(0x01010101, ui1, 0));
    const int dot4 = ggml_cuda_dp4a(0x01010101, ui4, ggml_cuda_dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    return dall * sumf_d - dmin * sumf_m;
#endif
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

#ifndef GGML_QKK_64
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#else

    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    const int8_t * s = bq5_K->scales;

    const float d = bq5_K->d;

    const float d8_1 = __low2half(bq8_1[0].ds);
    const float d8_2 = __low2half(bq8_1[1].ds);

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * ql = (const int *)bq5_K->qs + (iqs/2);
    const int vl1 = ql[0];
    const int vl2 = ql[4];

    const int step = 4 * (iqs/2); // 0, 4, 8, 12
    const int im = step/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    const int in = step%8; // 0, 4, 0, 4
    const int vh = (*((const int *)(bq5_K->qh + in))) >> im;

    const int v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    const int v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    const int v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    const int v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    const float sumf_d = d8_1 * (ggml_cuda_dp4a(ui1, v1, 0) * s[0] + ggml_cuda_dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (ggml_cuda_dp4a(ui3, v3, 0) * s[2] + ggml_cuda_dp4a(ui4, v4, 0) * s[3]);

    return d * sumf_d;
#endif
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

static __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded) {
    const int ix = blockDim.x*blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) {
        return;
    }
    const int iy = blockDim.y*blockIdx.y + threadIdx.y;
    const int i_padded = iy*kx_padded + ix;
    block_q8_1 * y = (block_q8_1 *) vy;

    const int ib = i_padded / QK8_1; // block index
    const int iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy*kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;
    if (iqs > 0) {
        return;
    }
    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

template<typename dst_t>
static __device__ __forceinline__ dst_t convert_from_half(half val) {
    return val;
}

template<>
__device__ __forceinline__ nv_bfloat16 convert_from_half<nv_bfloat16>(half val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __float2bfloat16(__half2float(val));
#else
    return __half2float(val);
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
}

template<>
__device__ __forceinline__ float convert_from_half<float>(half val) {
    return __half2float(val);
}

template<typename dst_t>
inline __device__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = 0; //we only need dequant one block in each call
    const block_q2_K * x = (const block_q2_K *) vx;

    const auto tid = threadIdx.x;
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    half dall = __low2half(x[i].dm);
    half dmin = __high2half(x[i].dm);
    y[l+ 0] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+0] & 0xF) * ((q >> 0) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+0] >> 4))));
    y[l+32] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+2] & 0xF) * ((q >> 2) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+2] >> 4))));
    y[l+64] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+4] & 0xF) * ((q >> 4) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+4] >> 4))));
    y[l+96] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+6] & 0xF) * ((q >> 6) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+6] >> 4))));
}

template<typename dst_t>
inline __device__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i = 0;
    const block_q3_K * x = (const block_q3_K *) vx;

    const auto r = threadIdx.x/4;
    const int tid = r/2;
    const int is0 = r%2;
    const int l0 = 16*is0 + 4*(threadIdx.x%4);
    const int n = tid / 4;
    const int j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    half d_all = x[i].d;
    half dl = __hmul(d_all,  __int2half_rn(us - 32));

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = convert_from_half<dst_t>(__hmul(dl,  __int2half_rn((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4))));
    }
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
inline __device__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const auto i = 0;

    // assume 32 threads
    const auto tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const half dall = __low2half(x[i].dm);
    const half dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const half d1 = __hmul(dall, __int2half_rn(sc));
    const half m1 = __hmul(dmin,  __int2half_rn(m));
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const half d2 = __hmul(dall, __int2half_rn(sc));
    const half m2 = __hmul(dmin, __int2half_rn(m));
    for (int l = 0; l < n; ++l) {
        y[l + 0] = convert_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn(q[l] & 0xF)), m1));
        y[l +32] = convert_from_half<dst_t>(__hsub(__hmul(d2,  __int2half_rn(q[l] >> 4)), m2));
    }
}

template<typename dst_t>
inline __device__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const auto i = 0;

    // assume 64 threads - this is very slightly better than the one below
    const auto tid = threadIdx.x;
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const half dall = __low2half(x[i].dm);
    const half dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const half d1 = __hmul(dall, __int2half_rn(sc)); const half m1 = __hmul(dmin, __int2half_rn(m));
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const half d2 = __hmul(dall, __int2half_rn(sc)); const half m2 = __hmul(dmin, __int2half_rn(m));

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = convert_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0))), m1));
    y[ 1] = convert_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0))), m1));
    hm <<= 1;
    y[32] = convert_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn((ql[0] >>  4) + (qh[0] & hm ? 16 : 0))), m2));
    y[33] = convert_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn((ql[1] >>  4) + (qh[1] & hm ? 16 : 0))), m2));
}

template<typename dst_t>
inline __device__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const auto i = 0;

    // assume 64 threads - this is very slightly better than the one below
    const auto tid = threadIdx.x;
    const int ip  = tid/32;   // ip is 0 or 1
    const int il  = tid - 32*ip; // 0...32
    const int is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const half d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32))));
    y[32] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32))));
    y[64] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32))));
    y[96] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32))));
}

// ===================== IQ Quantization Types =====================

// IQ2_XXS
#define QK_IQ2_XXS 256
#define QR_IQ2_XXS 8
#define QI_IQ2_XXS (QK_IQ2_XXS / (4 * QR_IQ2_XXS))

typedef struct {
    half d;
    uint16_t qs[QK_IQ2_XXS / 8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(ggml_fp16_t) + QK_IQ2_XXS/8*2, "wrong iq2_xxs block size");

// IQ2_XS
#define QK_IQ2_XS 256
#define QR_IQ2_XS 8
#define QI_IQ2_XS (QK_IQ2_XS / (4 * QR_IQ2_XS))

typedef struct {
    half d;
    uint16_t qs[QK_IQ2_XS / 8];
    uint8_t  scales[QK_IQ2_XS / 32];
} block_iq2_xs;
static_assert(sizeof(block_iq2_xs) == sizeof(ggml_fp16_t) + QK_IQ2_XS/8*2 + QK_IQ2_XS/32, "wrong iq2_xs block size");

// IQ3_XXS
#define QK_IQ3_XXS 256
#define QR_IQ3_XXS 8
#define QI_IQ3_XXS (QK_IQ3_XXS / (4 * QR_IQ3_XXS))

typedef struct {
    half d;
    uint8_t qs[3 * QK_IQ3_XXS / 8];
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == sizeof(ggml_fp16_t) + 3*(QK_IQ3_XXS/8), "wrong iq3_xxs block size");

// IQ4_XS
#define QK_IQ4_XS 256
#define QR_IQ4_XS 8
#define QI_IQ4_XS (QK_IQ4_XS / (4 * QR_IQ4_XS))

typedef struct {
    half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_IQ4_XS / 64];
    uint8_t  qs[QK_IQ4_XS / 2];
} block_iq4_xs;
static_assert(sizeof(block_iq4_xs) == sizeof(ggml_fp16_t) + 2 + QK_IQ4_XS/64 + QK_IQ4_XS/2, "wrong iq4_xs block size");

// IQ1_S
#define QK_IQ1_S 256
#define QI_IQ1_S (QK_IQ1_S / 32)
typedef struct {
    half d;
    uint8_t qs[QK_IQ1_S / 8];
    uint16_t qh[QK_IQ1_S / 32];
} block_iq1_s;
static_assert(sizeof(block_iq1_s) == 50, "wrong iq1_s block size");

// IQ4_NL
#define QK_IQ4_NL 32
#define QI_IQ4_NL 4
typedef struct {
    half d;
    uint8_t qs[QK_IQ4_NL / 2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == 18, "wrong iq4_nl block size");

// IQ3_S
#define QK_IQ3_S 256
#define QI_IQ3_S (QK_IQ3_S / 32)
typedef struct {
    half d;
    uint8_t qs[QK_IQ3_S / 4];
    uint8_t qh[QK_IQ3_S / 32];
    uint8_t signs[QK_IQ3_S / 8];
    uint8_t scales[QK_IQ3_S / 64];
} block_iq3_s;
static_assert(sizeof(block_iq3_s) == 110, "wrong iq3_s block size");

// IQ2_S
#define QK_IQ2_S 256
#define QI_IQ2_S (QK_IQ2_S / 32)
typedef struct {
    half d;
    uint8_t qs[QK_IQ2_S / 4];
    uint8_t qh[QK_IQ2_S / 32];
    uint8_t scales[QK_IQ2_S / 32];
} block_iq2_s;
static_assert(sizeof(block_iq2_s) == 82, "wrong iq2_s block size");

// IQ1_M
#define QK_IQ1_M 256
#define QI_IQ1_M (QK_IQ1_M / 32)
typedef struct {
    uint8_t qs[QK_IQ1_M / 8];
    uint8_t qh[QK_IQ1_M / 16];
    uint8_t scales[QK_IQ1_M / 32];
} block_iq1_m;
static_assert(sizeof(block_iq1_m) == 56, "wrong iq1_m block size");

static __device__ const uint8_t ksigns_iq2xs_cu[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static __device__ const uint8_t kmask_iq2xs_cu[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static __device__ const int8_t kvalues_iq4nl_cu[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static __device__ const uint64_t iq2xxs_grid_cu[256] = {
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

static __device__ const uint64_t iq2xs_grid_cu[512] = {
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

static __device__ const uint32_t iq3xxs_grid_cu[256] = {
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

#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

static __device__ const uint64_t iq2s_grid_cu[1024] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL,
    0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL, 0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL,
    0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x08080808192b192bULL,
    0x08080808192b2b19ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL, 0x080808082b191908ULL, 0x080808082b2b0808ULL,
    0x080808082b2b1919ULL, 0x080808082b2b2b2bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL, 0x080808190819082bULL,
    0x0808081908191919ULL, 0x0808081908192b08ULL, 0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL, 0x0808081919081919ULL, 0x0808081919082b08ULL,
    0x0808081919190819ULL, 0x0808081919191908ULL, 0x080808191919192bULL, 0x0808081919192b19ULL, 0x08080819192b0808ULL, 0x08080819192b1919ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL,
    0x080808192b081908ULL, 0x080808192b190808ULL, 0x080808192b19082bULL, 0x080808192b191919ULL, 0x080808192b2b0819ULL, 0x080808192b2b1908ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL,
    0x0808082b08081919ULL, 0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL, 0x0808082b082b2b2bULL, 0x0808082b19080819ULL, 0x0808082b19081908ULL,
    0x0808082b1908192bULL, 0x0808082b19082b19ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL, 0x0808082b2b080808ULL, 0x0808082b2b081919ULL, 0x0808082b2b082b2bULL, 0x0808082b2b191908ULL,
    0x0808082b2b2b082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL, 0x0808190808191919ULL,
    0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x08081908082b192bULL, 0x08081908082b2b19ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL,
    0x0808190819082b08ULL, 0x0808190819082b2bULL, 0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x0808190819192b19ULL, 0x08081908192b0808ULL, 0x08081908192b082bULL,
    0x08081908192b1919ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b08192bULL, 0x080819082b082b19ULL, 0x080819082b190808ULL, 0x080819082b191919ULL, 0x080819082b192b08ULL,
    0x080819082b2b0819ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908082b2bULL, 0x0808191908190819ULL,
    0x0808191908191908ULL, 0x080819190819192bULL, 0x0808191908192b19ULL, 0x08081919082b0808ULL, 0x08081919082b1919ULL, 0x08081919082b2b08ULL, 0x0808191919080819ULL, 0x0808191919081908ULL,
    0x080819191908192bULL, 0x0808191919082b19ULL, 0x0808191919190808ULL, 0x080819191919082bULL, 0x0808191919191919ULL, 0x0808191919192b08ULL, 0x08081919192b0819ULL, 0x08081919192b1908ULL,
    0x080819192b080808ULL, 0x080819192b08082bULL, 0x080819192b081919ULL, 0x080819192b082b08ULL, 0x080819192b190819ULL, 0x080819192b191908ULL, 0x080819192b2b0808ULL, 0x0808192b08080819ULL,
    0x0808192b08081908ULL, 0x0808192b0808192bULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b08191919ULL, 0x0808192b19080808ULL, 0x0808192b19081919ULL, 0x0808192b19082b08ULL,
    0x0808192b19190819ULL, 0x0808192b19191908ULL, 0x0808192b192b0808ULL, 0x0808192b2b080819ULL, 0x0808192b2b081908ULL, 0x0808192b2b190808ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL,
    0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808190819ULL, 0x08082b0808191908ULL, 0x08082b080819192bULL, 0x08082b0808192b19ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL,
    0x08082b08082b2b2bULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b081908192bULL, 0x08082b0819082b19ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b0819191919ULL,
    0x08082b0819192b08ULL, 0x08082b08192b0819ULL, 0x08082b08192b1908ULL, 0x08082b082b080808ULL, 0x08082b082b081919ULL, 0x08082b082b191908ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL,
    0x08082b1908081908ULL, 0x08082b1908190808ULL, 0x08082b190819082bULL, 0x08082b1908191919ULL, 0x08082b1908192b08ULL, 0x08082b19082b0819ULL, 0x08082b1919080808ULL, 0x08082b1919081919ULL,
    0x08082b1919082b08ULL, 0x08082b1919190819ULL, 0x08082b1919191908ULL, 0x08082b19192b0808ULL, 0x08082b192b080819ULL, 0x08082b192b190808ULL, 0x08082b2b08080808ULL, 0x08082b2b08190819ULL,
    0x08082b2b08191908ULL, 0x08082b2b082b082bULL, 0x08082b2b082b2b08ULL, 0x08082b2b082b2b2bULL, 0x08082b2b19190808ULL, 0x08082b2b2b192b19ULL, 0x0819080808080819ULL, 0x0819080808081908ULL,
    0x081908080808192bULL, 0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL, 0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL,
    0x08190808082b192bULL, 0x0819080819080808ULL, 0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL, 0x0819080819191908ULL, 0x081908081919192bULL,
    0x0819080819192b19ULL, 0x08190808192b0808ULL, 0x08190808192b082bULL, 0x08190808192b1919ULL, 0x08190808192b2b08ULL, 0x081908082b080819ULL, 0x081908082b081908ULL, 0x081908082b08192bULL,
    0x081908082b190808ULL, 0x081908082b191919ULL, 0x081908082b192b08ULL, 0x081908082b2b0819ULL, 0x081908082b2b1908ULL, 0x0819081908080808ULL, 0x081908190808082bULL, 0x0819081908081919ULL,
    0x0819081908082b08ULL, 0x0819081908082b2bULL, 0x0819081908190819ULL, 0x0819081908191908ULL, 0x081908190819192bULL, 0x0819081908192b19ULL, 0x08190819082b0808ULL, 0x08190819082b082bULL,
    0x08190819082b1919ULL, 0x08190819082b2b08ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x081908191908192bULL, 0x0819081919082b19ULL, 0x0819081919190808ULL, 0x081908191919082bULL,
    0x0819081919191919ULL, 0x0819081919192b08ULL, 0x08190819192b0819ULL, 0x08190819192b1908ULL, 0x081908192b080808ULL, 0x081908192b08082bULL, 0x081908192b081919ULL, 0x081908192b082b08ULL,
    0x081908192b190819ULL, 0x081908192b191908ULL, 0x0819082b08080819ULL, 0x0819082b08081908ULL, 0x0819082b08082b19ULL, 0x0819082b08190808ULL, 0x0819082b08191919ULL, 0x0819082b082b0819ULL,
    0x0819082b082b1908ULL, 0x0819082b19080808ULL, 0x0819082b19081919ULL, 0x0819082b19190819ULL, 0x0819082b19191908ULL, 0x0819082b2b080819ULL, 0x0819082b2b081908ULL, 0x0819082b2b190808ULL,
    0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL, 0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x081919080819192bULL, 0x0819190808192b19ULL,
    0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x08191908082b2b08ULL, 0x0819190819080819ULL, 0x0819190819081908ULL, 0x081919081908192bULL, 0x0819190819082b19ULL, 0x0819190819190808ULL,
    0x081919081919082bULL, 0x0819190819191919ULL, 0x0819190819192b08ULL, 0x08191908192b0819ULL, 0x08191908192b1908ULL, 0x081919082b080808ULL, 0x081919082b08082bULL, 0x081919082b081919ULL,
    0x081919082b082b08ULL, 0x081919082b190819ULL, 0x081919082b191908ULL, 0x081919082b2b0808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL, 0x081919190808192bULL, 0x0819191908082b19ULL,
    0x0819191908190808ULL, 0x081919190819082bULL, 0x0819191908191919ULL, 0x0819191908192b08ULL, 0x08191919082b0819ULL, 0x08191919082b1908ULL, 0x0819191919080808ULL, 0x081919191908082bULL,
    0x0819191919081919ULL, 0x0819191919082b08ULL, 0x0819191919190819ULL, 0x0819191919191908ULL, 0x08191919192b0808ULL, 0x081919192b080819ULL, 0x081919192b081908ULL, 0x081919192b190808ULL,
    0x0819192b08080808ULL, 0x0819192b08081919ULL, 0x0819192b08082b08ULL, 0x0819192b08190819ULL, 0x0819192b08191908ULL, 0x0819192b082b0808ULL, 0x0819192b19080819ULL, 0x0819192b19081908ULL,
    0x0819192b19190808ULL, 0x0819192b2b080808ULL, 0x0819192b2b2b2b2bULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b080808192bULL, 0x08192b0808082b19ULL, 0x08192b0808190808ULL,
    0x08192b0808191919ULL, 0x08192b0808192b08ULL, 0x08192b08082b0819ULL, 0x08192b0819080808ULL, 0x08192b081908082bULL, 0x08192b0819081919ULL, 0x08192b0819082b08ULL, 0x08192b0819190819ULL,
    0x08192b0819191908ULL, 0x08192b08192b0808ULL, 0x08192b082b080819ULL, 0x08192b082b081908ULL, 0x08192b1908080808ULL, 0x08192b190808082bULL, 0x08192b1908081919ULL, 0x08192b1908082b08ULL,
    0x08192b1908190819ULL, 0x08192b1908191908ULL, 0x08192b19082b0808ULL, 0x08192b1919080819ULL, 0x08192b1919081908ULL, 0x08192b1919190808ULL, 0x08192b19192b2b19ULL, 0x08192b192b2b082bULL,
    0x08192b2b08081908ULL, 0x08192b2b08190808ULL, 0x08192b2b19080808ULL, 0x08192b2b1919192bULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL, 0x082b080808082b08ULL,
    0x082b080808190819ULL, 0x082b080808191908ULL, 0x082b08080819192bULL, 0x082b080808192b19ULL, 0x082b0808082b0808ULL, 0x082b0808082b1919ULL, 0x082b0808082b2b2bULL, 0x082b080819080819ULL,
    0x082b080819081908ULL, 0x082b080819190808ULL, 0x082b08081919082bULL, 0x082b080819191919ULL, 0x082b0808192b1908ULL, 0x082b08082b080808ULL, 0x082b08082b082b2bULL, 0x082b08082b191908ULL,
    0x082b08082b2b2b2bULL, 0x082b081908080819ULL, 0x082b081908081908ULL, 0x082b081908190808ULL, 0x082b08190819082bULL, 0x082b081908191919ULL, 0x082b0819082b0819ULL, 0x082b081919080808ULL,
    0x082b08191908082bULL, 0x082b081919081919ULL, 0x082b081919190819ULL, 0x082b081919191908ULL, 0x082b0819192b0808ULL, 0x082b08192b080819ULL, 0x082b08192b081908ULL, 0x082b08192b190808ULL,
    0x082b082b08080808ULL, 0x082b082b08082b2bULL, 0x082b082b082b082bULL, 0x082b082b082b2b08ULL, 0x082b082b082b2b2bULL, 0x082b082b19081908ULL, 0x082b082b19190808ULL, 0x082b082b2b082b08ULL,
    0x082b082b2b082b2bULL, 0x082b082b2b2b2b08ULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b19080808192bULL, 0x082b190808082b19ULL, 0x082b190808190808ULL, 0x082b190808191919ULL,
    0x082b190808192b08ULL, 0x082b1908082b0819ULL, 0x082b1908082b1908ULL, 0x082b190819080808ULL, 0x082b19081908082bULL, 0x082b190819081919ULL, 0x082b190819082b08ULL, 0x082b190819190819ULL,
    0x082b190819191908ULL, 0x082b1908192b0808ULL, 0x082b19082b080819ULL, 0x082b19082b081908ULL, 0x082b19082b190808ULL, 0x082b191908080808ULL, 0x082b191908081919ULL, 0x082b191908082b08ULL,
    0x082b191908190819ULL, 0x082b191908191908ULL, 0x082b1919082b0808ULL, 0x082b191919080819ULL, 0x082b191919081908ULL, 0x082b191919190808ULL, 0x082b1919192b192bULL, 0x082b19192b080808ULL,
    0x082b192b08080819ULL, 0x082b192b08081908ULL, 0x082b192b08190808ULL, 0x082b192b19080808ULL, 0x082b192b19192b19ULL, 0x082b2b0808080808ULL, 0x082b2b0808081919ULL, 0x082b2b0808190819ULL,
    0x082b2b0808191908ULL, 0x082b2b0819080819ULL, 0x082b2b0819081908ULL, 0x082b2b0819190808ULL, 0x082b2b082b082b2bULL, 0x082b2b082b2b2b2bULL, 0x082b2b1908080819ULL, 0x082b2b1908081908ULL,
    0x082b2b1908190808ULL, 0x082b2b192b191919ULL, 0x082b2b2b08082b2bULL, 0x082b2b2b082b082bULL, 0x082b2b2b192b1908ULL, 0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL,
    0x1908080808081908ULL, 0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL, 0x1908080808191919ULL, 0x1908080808192b08ULL, 0x1908080808192b2bULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x19080808082b192bULL, 0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL, 0x1908080819082b2bULL,
    0x1908080819190819ULL, 0x1908080819191908ULL, 0x190808081919192bULL, 0x1908080819192b19ULL, 0x19080808192b0808ULL, 0x19080808192b082bULL, 0x19080808192b1919ULL, 0x190808082b080819ULL,
    0x190808082b081908ULL, 0x190808082b190808ULL, 0x190808082b191919ULL, 0x190808082b192b08ULL, 0x190808082b2b0819ULL, 0x190808082b2b1908ULL, 0x1908081908080808ULL, 0x190808190808082bULL,
    0x1908081908081919ULL, 0x1908081908082b08ULL, 0x1908081908190819ULL, 0x1908081908191908ULL, 0x190808190819192bULL, 0x1908081908192b19ULL, 0x19080819082b0808ULL, 0x19080819082b082bULL,
    0x19080819082b1919ULL, 0x1908081919080819ULL, 0x1908081919081908ULL, 0x190808191908192bULL, 0x1908081919082b19ULL, 0x1908081919190808ULL, 0x190808191919082bULL, 0x1908081919191919ULL,
    0x1908081919192b08ULL, 0x19080819192b0819ULL, 0x19080819192b1908ULL, 0x190808192b080808ULL, 0x190808192b08082bULL, 0x190808192b081919ULL, 0x190808192b082b08ULL, 0x190808192b190819ULL,
    0x190808192b191908ULL, 0x190808192b2b0808ULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL, 0x1908082b0819082bULL, 0x1908082b08191919ULL, 0x1908082b08192b08ULL,
    0x1908082b082b1908ULL, 0x1908082b19080808ULL, 0x1908082b19081919ULL, 0x1908082b19082b08ULL, 0x1908082b19190819ULL, 0x1908082b19191908ULL, 0x1908082b192b0808ULL, 0x1908082b2b080819ULL,
    0x1908082b2b081908ULL, 0x1908190808080808ULL, 0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808082b2bULL, 0x1908190808190819ULL, 0x1908190808191908ULL,
    0x190819080819192bULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x19081908082b082bULL, 0x19081908082b1919ULL, 0x19081908082b2b08ULL, 0x1908190819080819ULL, 0x1908190819081908ULL,
    0x190819081908192bULL, 0x1908190819082b19ULL, 0x1908190819190808ULL, 0x190819081919082bULL, 0x1908190819191919ULL, 0x1908190819192b08ULL, 0x19081908192b0819ULL, 0x19081908192b1908ULL,
    0x190819082b080808ULL, 0x190819082b08082bULL, 0x190819082b081919ULL, 0x190819082b082b08ULL, 0x190819082b190819ULL, 0x190819082b191908ULL, 0x190819082b2b0808ULL, 0x1908191908080819ULL,
    0x1908191908081908ULL, 0x190819190808192bULL, 0x1908191908082b19ULL, 0x1908191908190808ULL, 0x190819190819082bULL, 0x1908191908191919ULL, 0x1908191908192b08ULL, 0x19081919082b0819ULL,
    0x19081919082b1908ULL, 0x1908191919080808ULL, 0x190819191908082bULL, 0x1908191919081919ULL, 0x1908191919082b08ULL, 0x1908191919190819ULL, 0x1908191919191908ULL, 0x19081919192b0808ULL,
    0x19081919192b2b2bULL, 0x190819192b080819ULL, 0x190819192b081908ULL, 0x190819192b190808ULL, 0x1908192b08080808ULL, 0x1908192b0808082bULL, 0x1908192b08081919ULL, 0x1908192b08082b08ULL,
    0x1908192b08190819ULL, 0x1908192b08191908ULL, 0x1908192b082b0808ULL, 0x1908192b19080819ULL, 0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x1908192b2b080808ULL, 0x1908192b2b2b1919ULL,
    0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808082b19ULL, 0x19082b0808190808ULL, 0x19082b080819082bULL, 0x19082b0808191919ULL, 0x19082b0808192b08ULL, 0x19082b08082b0819ULL,
    0x19082b08082b1908ULL, 0x19082b0819080808ULL, 0x19082b081908082bULL, 0x19082b0819081919ULL, 0x19082b0819082b08ULL, 0x19082b0819190819ULL, 0x19082b0819191908ULL, 0x19082b08192b0808ULL,
    0x19082b082b081908ULL, 0x19082b082b190808ULL, 0x19082b1908080808ULL, 0x19082b190808082bULL, 0x19082b1908081919ULL, 0x19082b1908082b08ULL, 0x19082b1908190819ULL, 0x19082b1908191908ULL,
    0x19082b19082b0808ULL, 0x19082b1919080819ULL, 0x19082b1919081908ULL, 0x19082b1919190808ULL, 0x19082b192b080808ULL, 0x19082b192b19192bULL, 0x19082b2b08080819ULL, 0x19082b2b08081908ULL,
    0x19082b2b08190808ULL, 0x19082b2b19080808ULL, 0x1919080808080808ULL, 0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808191908ULL,
    0x191908080819192bULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x19190808082b082bULL, 0x19190808082b1919ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL, 0x1919080819081908ULL,
    0x191908081908192bULL, 0x1919080819082b19ULL, 0x1919080819190808ULL, 0x191908081919082bULL, 0x1919080819191919ULL, 0x1919080819192b08ULL, 0x19190808192b0819ULL, 0x19190808192b1908ULL,
    0x191908082b080808ULL, 0x191908082b08082bULL, 0x191908082b081919ULL, 0x191908082b082b08ULL, 0x191908082b190819ULL, 0x191908082b191908ULL, 0x1919081908080819ULL, 0x1919081908081908ULL,
    0x191908190808192bULL, 0x1919081908082b19ULL, 0x1919081908190808ULL, 0x191908190819082bULL, 0x1919081908191919ULL, 0x1919081908192b08ULL, 0x19190819082b0819ULL, 0x19190819082b1908ULL,
    0x1919081919080808ULL, 0x191908191908082bULL, 0x1919081919081919ULL, 0x1919081919082b08ULL, 0x1919081919190819ULL, 0x1919081919191908ULL, 0x19190819192b0808ULL, 0x191908192b080819ULL,
    0x191908192b081908ULL, 0x191908192b190808ULL, 0x1919082b08080808ULL, 0x1919082b08081919ULL, 0x1919082b08082b08ULL, 0x1919082b08190819ULL, 0x1919082b08191908ULL, 0x1919082b082b0808ULL,
    0x1919082b19080819ULL, 0x1919082b19081908ULL, 0x1919082b19190808ULL, 0x1919082b192b2b19ULL, 0x1919082b2b080808ULL, 0x1919190808080819ULL, 0x1919190808081908ULL, 0x191919080808192bULL,
    0x1919190808082b19ULL, 0x1919190808190808ULL, 0x191919080819082bULL, 0x1919190808191919ULL, 0x1919190808192b08ULL, 0x19191908082b0819ULL, 0x19191908082b1908ULL, 0x1919190819080808ULL,
    0x191919081908082bULL, 0x1919190819081919ULL, 0x1919190819082b08ULL, 0x1919190819190819ULL, 0x1919190819191908ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b081908ULL,
    0x191919082b190808ULL, 0x1919191908080808ULL, 0x191919190808082bULL, 0x1919191908081919ULL, 0x1919191908082b08ULL, 0x1919191908190819ULL, 0x1919191908191908ULL, 0x19191919082b0808ULL,
    0x1919191919080819ULL, 0x1919191919081908ULL, 0x1919191919190808ULL, 0x191919192b080808ULL, 0x1919192b08080819ULL, 0x1919192b08081908ULL, 0x1919192b08190808ULL, 0x1919192b082b192bULL,
    0x1919192b19080808ULL, 0x19192b0808080808ULL, 0x19192b080808082bULL, 0x19192b0808081919ULL, 0x19192b0808082b08ULL, 0x19192b0808190819ULL, 0x19192b0808191908ULL, 0x19192b08082b0808ULL,
    0x19192b0819080819ULL, 0x19192b0819081908ULL, 0x19192b0819190808ULL, 0x19192b0819192b2bULL, 0x19192b082b080808ULL, 0x19192b1908080819ULL, 0x19192b1908081908ULL, 0x19192b1908190808ULL,
    0x19192b1919080808ULL, 0x19192b2b08080808ULL, 0x19192b2b08192b19ULL, 0x19192b2b2b081919ULL, 0x19192b2b2b2b2b08ULL, 0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b08080808192bULL,
    0x192b080808190808ULL, 0x192b08080819082bULL, 0x192b080808191919ULL, 0x192b080808192b08ULL, 0x192b0808082b0819ULL, 0x192b0808082b1908ULL, 0x192b080819080808ULL, 0x192b080819081919ULL,
    0x192b080819082b08ULL, 0x192b080819190819ULL, 0x192b080819191908ULL, 0x192b0808192b0808ULL, 0x192b08082b081908ULL, 0x192b08082b190808ULL, 0x192b081908080808ULL, 0x192b08190808082bULL,
    0x192b081908081919ULL, 0x192b081908082b08ULL, 0x192b081908190819ULL, 0x192b081908191908ULL, 0x192b0819082b0808ULL, 0x192b081919080819ULL, 0x192b081919081908ULL, 0x192b081919190808ULL,
    0x192b08192b080808ULL, 0x192b08192b192b19ULL, 0x192b082b08081908ULL, 0x192b082b08190808ULL, 0x192b082b19080808ULL, 0x192b082b1919192bULL, 0x192b082b2b2b0819ULL, 0x192b190808080808ULL,
    0x192b190808081919ULL, 0x192b190808082b08ULL, 0x192b190808190819ULL, 0x192b190808191908ULL, 0x192b1908082b0808ULL, 0x192b190819080819ULL, 0x192b190819081908ULL, 0x192b190819190808ULL,
    0x192b19082b080808ULL, 0x192b191908080819ULL, 0x192b191908081908ULL, 0x192b191908190808ULL, 0x192b191919080808ULL, 0x192b191919082b2bULL, 0x192b1919192b2b08ULL, 0x192b19192b19082bULL,
    0x192b192b08080808ULL, 0x192b192b2b191908ULL, 0x192b2b0808080819ULL, 0x192b2b0808081908ULL, 0x192b2b0808190808ULL, 0x192b2b08192b1919ULL, 0x192b2b082b192b08ULL, 0x192b2b1908080808ULL,
    0x192b2b19082b2b2bULL, 0x192b2b2b1908082bULL, 0x192b2b2b2b2b0819ULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL, 0x2b08080808190819ULL,
    0x2b08080808191908ULL, 0x2b08080808192b19ULL, 0x2b080808082b0808ULL, 0x2b080808082b1919ULL, 0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808081919082bULL,
    0x2b08080819191919ULL, 0x2b08080819192b08ULL, 0x2b080808192b0819ULL, 0x2b0808082b080808ULL, 0x2b0808082b081919ULL, 0x2b0808082b190819ULL, 0x2b0808082b191908ULL, 0x2b08081908080819ULL,
    0x2b08081908081908ULL, 0x2b08081908082b19ULL, 0x2b08081908190808ULL, 0x2b0808190819082bULL, 0x2b08081908191919ULL, 0x2b08081908192b08ULL, 0x2b080819082b0819ULL, 0x2b080819082b1908ULL,
    0x2b08081919080808ULL, 0x2b0808191908082bULL, 0x2b08081919081919ULL, 0x2b08081919082b08ULL, 0x2b08081919190819ULL, 0x2b08081919191908ULL, 0x2b0808192b080819ULL, 0x2b0808192b081908ULL,
    0x2b0808192b190808ULL, 0x2b0808192b2b2b19ULL, 0x2b08082b08080808ULL, 0x2b08082b08081919ULL, 0x2b08082b08082b2bULL, 0x2b08082b08190819ULL, 0x2b08082b08191908ULL, 0x2b08082b19080819ULL,
    0x2b08082b19081908ULL, 0x2b08082b19190808ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b0819080808192bULL, 0x2b08190808082b19ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL,
    0x2b08190808191919ULL, 0x2b08190808192b08ULL, 0x2b081908082b0819ULL, 0x2b08190819080808ULL, 0x2b0819081908082bULL, 0x2b08190819081919ULL, 0x2b08190819082b08ULL, 0x2b08190819190819ULL,
    0x2b08190819191908ULL, 0x2b081908192b0808ULL, 0x2b0819082b080819ULL, 0x2b0819082b081908ULL, 0x2b0819082b190808ULL, 0x2b08191908080808ULL, 0x2b0819190808082bULL, 0x2b08191908081919ULL,
    0x2b08191908082b08ULL, 0x2b08191908190819ULL, 0x2b08191908191908ULL, 0x2b081919082b0808ULL, 0x2b08191919080819ULL, 0x2b08191919081908ULL, 0x2b08191919190808ULL, 0x2b0819192b080808ULL,
    0x2b0819192b082b2bULL, 0x2b08192b08080819ULL, 0x2b08192b08081908ULL, 0x2b08192b08190808ULL, 0x2b08192b082b2b19ULL, 0x2b08192b19080808ULL, 0x2b082b0808080808ULL, 0x2b082b0808081919ULL,
    0x2b082b0808190819ULL, 0x2b082b0808191908ULL, 0x2b082b0819080819ULL, 0x2b082b0819081908ULL, 0x2b082b0819190808ULL, 0x2b082b082b2b082bULL, 0x2b082b1908080819ULL, 0x2b082b1908081908ULL,
    0x2b082b1919080808ULL, 0x2b082b19192b1919ULL, 0x2b082b2b082b082bULL, 0x2b082b2b19192b08ULL, 0x2b082b2b19192b2bULL, 0x2b082b2b2b08082bULL, 0x2b082b2b2b2b082bULL, 0x2b19080808080819ULL,
    0x2b19080808081908ULL, 0x2b19080808082b19ULL, 0x2b19080808190808ULL, 0x2b1908080819082bULL, 0x2b19080808191919ULL, 0x2b19080808192b08ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL,
    0x2b1908081908082bULL, 0x2b19080819081919ULL, 0x2b19080819082b08ULL, 0x2b19080819190819ULL, 0x2b19080819191908ULL, 0x2b190808192b0808ULL, 0x2b1908082b080819ULL, 0x2b1908082b081908ULL,
    0x2b1908082b190808ULL, 0x2b19081908080808ULL, 0x2b19081908081919ULL, 0x2b19081908190819ULL, 0x2b19081908191908ULL, 0x2b19081919080819ULL, 0x2b19081919081908ULL, 0x2b19081919190808ULL,
    0x2b19081919192b2bULL, 0x2b19082b08080819ULL, 0x2b19082b08081908ULL, 0x2b19082b08190808ULL, 0x2b19082b19080808ULL, 0x2b19082b2b2b192bULL, 0x2b19190808080808ULL, 0x2b1919080808082bULL,
    0x2b19190808081919ULL, 0x2b19190808082b08ULL, 0x2b19190808190819ULL, 0x2b19190808191908ULL, 0x2b191908082b0808ULL, 0x2b19190819080819ULL, 0x2b19190819081908ULL, 0x2b19190819190808ULL,
    0x2b1919082b080808ULL, 0x2b1919082b19192bULL, 0x2b19191908080819ULL, 0x2b19191908081908ULL, 0x2b19191908190808ULL, 0x2b19191919080808ULL, 0x2b1919192b192b08ULL, 0x2b1919192b2b0819ULL,
    0x2b19192b08080808ULL, 0x2b19192b1908192bULL, 0x2b19192b192b1908ULL, 0x2b192b0808080819ULL, 0x2b192b0808081908ULL, 0x2b192b0808190808ULL, 0x2b192b08082b192bULL, 0x2b192b0819080808ULL,
    0x2b192b082b2b2b19ULL, 0x2b192b1908080808ULL, 0x2b192b1919082b19ULL, 0x2b192b191919082bULL, 0x2b192b2b2b190808ULL, 0x2b2b080808080808ULL, 0x2b2b080808081919ULL, 0x2b2b080808082b2bULL,
    0x2b2b080808191908ULL, 0x2b2b0808082b082bULL, 0x2b2b0808082b2b2bULL, 0x2b2b080819080819ULL, 0x2b2b080819081908ULL, 0x2b2b080819190808ULL, 0x2b2b08082b2b082bULL, 0x2b2b08082b2b2b2bULL,
    0x2b2b081919080808ULL, 0x2b2b0819192b1919ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b2bULL, 0x2b2b082b082b082bULL, 0x2b2b082b082b2b08ULL, 0x2b2b082b082b2b2bULL, 0x2b2b082b2b08082bULL,
    0x2b2b082b2b082b08ULL, 0x2b2b082b2b082b2bULL, 0x2b2b082b2b2b2b08ULL, 0x2b2b190808080819ULL, 0x2b2b190808081908ULL, 0x2b2b190808190808ULL, 0x2b2b190819080808ULL, 0x2b2b19082b082b19ULL,
    0x2b2b19082b2b1908ULL, 0x2b2b191908080808ULL, 0x2b2b191908192b19ULL, 0x2b2b192b19190819ULL, 0x2b2b2b0808082b2bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b082bULL, 0x2b2b2b1919191908ULL,
    0x2b2b2b192b08192bULL, 0x2b2b2b2b08082b08ULL, 0x2b2b2b2b08082b2bULL, 0x2b2b2b2b082b0808ULL, 0x2b2b2b2b082b082bULL, 0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b2b082b08ULL, 0x2b2b2b2b2b2b2b2bULL,
};

static __device__ const uint32_t iq3xs_grid_cu[512] = {
    0x04040404U, 0x0404040cU, 0x04040414U, 0x0404042cU, 0x0404043eU, 0x04040c04U, 0x04040c0cU, 0x04040c14U,
    0x04040c24U, 0x04040c34U, 0x04041404U, 0x0404140cU, 0x0404142cU, 0x04041c1cU, 0x04042404U, 0x04042414U,
    0x0404242cU, 0x0404243eU, 0x04042c0cU, 0x04042c1cU, 0x04043404U, 0x04043414U, 0x04043e0cU, 0x04043e24U,
    0x04043e3eU, 0x040c0404U, 0x040c040cU, 0x040c0414U, 0x040c0424U, 0x040c0c04U, 0x040c0c0cU, 0x040c0c2cU,
    0x040c1404U, 0x040c141cU, 0x040c143eU, 0x040c1c0cU, 0x040c1c2cU, 0x040c2424U, 0x040c340cU, 0x040c342cU,
    0x040c3e14U, 0x04140404U, 0x0414040cU, 0x0414042cU, 0x0414043eU, 0x04140c04U, 0x04140c1cU, 0x04140c34U,
    0x0414140cU, 0x0414142cU, 0x04141c04U, 0x04141c24U, 0x04142414U, 0x0414242cU, 0x0414243eU, 0x04142c0cU,
    0x04142c1cU, 0x04143e04U, 0x04143e1cU, 0x041c041cU, 0x041c0c0cU, 0x041c0c2cU, 0x041c1404U, 0x041c1414U,
    0x041c1c0cU, 0x041c1c1cU, 0x041c1c34U, 0x041c2424U, 0x041c2c04U, 0x041c2c14U, 0x041c343eU, 0x041c3e0cU,
    0x041c3e2cU, 0x04240404U, 0x04240c1cU, 0x04240c3eU, 0x0424140cU, 0x04241424U, 0x04241c14U, 0x04242404U,
    0x0424241cU, 0x04242c0cU, 0x04243e04U, 0x042c0414U, 0x042c0424U, 0x042c1404U, 0x042c1414U, 0x042c1434U,
    0x042c1c1cU, 0x042c240cU, 0x042c242cU, 0x042c243eU, 0x042c3434U, 0x042c3e1cU, 0x04340434U, 0x04340c0cU,
    0x04340c1cU, 0x04341c0cU, 0x04342c14U, 0x04343e0cU, 0x043e0404U, 0x043e0414U, 0x043e0424U, 0x043e1404U,
    0x043e1414U, 0x043e1434U, 0x043e1c1cU, 0x043e2c04U, 0x043e2c24U, 0x0c040404U, 0x0c04040cU, 0x0c040414U,
    0x0c040424U, 0x0c040c04U, 0x0c040c0cU, 0x0c040c1cU, 0x0c040c2cU, 0x0c040c3eU, 0x0c041404U, 0x0c041414U,
    0x0c041c0cU, 0x0c041c24U, 0x0c041c34U, 0x0c042c24U, 0x0c042c34U, 0x0c04340cU, 0x0c043e14U, 0x0c0c0404U,
    0x0c0c040cU, 0x0c0c041cU, 0x0c0c0434U, 0x0c0c0c04U, 0x0c0c0c24U, 0x0c0c140cU, 0x0c0c1c04U, 0x0c0c1c1cU,
    0x0c0c240cU, 0x0c0c2c04U, 0x0c0c2c14U, 0x0c0c3e04U, 0x0c0c3e34U, 0x0c140404U, 0x0c140c14U, 0x0c140c2cU,
    0x0c140c3eU, 0x0c141404U, 0x0c141424U, 0x0c141c14U, 0x0c142404U, 0x0c14241cU, 0x0c142c2cU, 0x0c143404U,
    0x0c143e14U, 0x0c1c040cU, 0x0c1c0424U, 0x0c1c043eU, 0x0c1c0c04U, 0x0c1c0c1cU, 0x0c1c140cU, 0x0c1c143eU,
    0x0c1c1c04U, 0x0c1c1c24U, 0x0c1c240cU, 0x0c1c3414U, 0x0c1c3e04U, 0x0c24041cU, 0x0c24042cU, 0x0c240c14U,
    0x0c240c24U, 0x0c241c0cU, 0x0c241c1cU, 0x0c242414U, 0x0c242434U, 0x0c242c04U, 0x0c242c24U, 0x0c2c040cU,
    0x0c2c0c04U, 0x0c2c0c1cU, 0x0c2c140cU, 0x0c2c1c04U, 0x0c2c1c14U, 0x0c2c2c0cU, 0x0c341404U, 0x0c341424U,
    0x0c34143eU, 0x0c342424U, 0x0c342434U, 0x0c3e040cU, 0x0c3e041cU, 0x0c3e0c04U, 0x0c3e0c14U, 0x0c3e140cU,
    0x0c3e1c2cU, 0x0c3e240cU, 0x0c3e3414U, 0x0c3e3e04U, 0x14040404U, 0x1404040cU, 0x1404041cU, 0x1404042cU,
    0x1404043eU, 0x14040c04U, 0x14040c14U, 0x14040c24U, 0x14040c34U, 0x1404140cU, 0x1404141cU, 0x1404143eU,
    0x14041c04U, 0x14041c14U, 0x1404240cU, 0x1404241cU, 0x1404242cU, 0x14042c04U, 0x14042c14U, 0x1404343eU,
    0x14043e04U, 0x14043e1cU, 0x14043e2cU, 0x140c0404U, 0x140c0414U, 0x140c0c04U, 0x140c0c1cU, 0x140c0c3eU,
    0x140c1414U, 0x140c142cU, 0x140c1c0cU, 0x140c1c24U, 0x140c2414U, 0x140c2c0cU, 0x1414040cU, 0x14140424U,
    0x1414043eU, 0x1414140cU, 0x1414141cU, 0x14141c04U, 0x14141c3eU, 0x1414240cU, 0x14142c1cU, 0x14142c3eU,
    0x14143e0cU, 0x14143e24U, 0x141c0404U, 0x141c0414U, 0x141c042cU, 0x141c0c0cU, 0x141c1414U, 0x141c1424U,
    0x141c1c0cU, 0x141c1c1cU, 0x141c2414U, 0x141c2c04U, 0x141c3434U, 0x1424040cU, 0x1424043eU, 0x14241404U,
    0x1424141cU, 0x14241c14U, 0x14241c2cU, 0x1424240cU, 0x14243e14U, 0x14243e2cU, 0x142c0424U, 0x142c0c0cU,
    0x142c1414U, 0x142c1c3eU, 0x142c2404U, 0x142c2c1cU, 0x142c3e04U, 0x14340404U, 0x14340414U, 0x1434043eU,
    0x1434140cU, 0x14342c2cU, 0x1434340cU, 0x143e042cU, 0x143e0c0cU, 0x143e1434U, 0x143e1c04U, 0x143e241cU,
    0x143e2c04U, 0x1c040414U, 0x1c040c0cU, 0x1c040c1cU, 0x1c040c2cU, 0x1c040c3eU, 0x1c041414U, 0x1c041c0cU,
    0x1c041c1cU, 0x1c041c2cU, 0x1c042414U, 0x1c042424U, 0x1c04243eU, 0x1c042c0cU, 0x1c04341cU, 0x1c043e0cU,
    0x1c0c040cU, 0x1c0c041cU, 0x1c0c042cU, 0x1c0c0c24U, 0x1c0c140cU, 0x1c0c141cU, 0x1c0c2404U, 0x1c0c3404U,
    0x1c0c3e14U, 0x1c0c3e34U, 0x1c140404U, 0x1c140c14U, 0x1c141404U, 0x1c141c14U, 0x1c141c24U, 0x1c142c04U,
    0x1c1c040cU, 0x1c1c0c04U, 0x1c1c0c24U, 0x1c1c140cU, 0x1c1c141cU, 0x1c1c143eU, 0x1c1c1c04U, 0x1c1c240cU,
    0x1c1c241cU, 0x1c1c243eU, 0x1c1c2c2cU, 0x1c1c3e1cU, 0x1c24041cU, 0x1c240c0cU, 0x1c240c34U, 0x1c241414U,
    0x1c241c0cU, 0x1c242c14U, 0x1c243404U, 0x1c243424U, 0x1c2c040cU, 0x1c2c0c04U, 0x1c2c0c14U, 0x1c2c142cU,
    0x1c2c1c14U, 0x1c2c2424U, 0x1c2c2c34U, 0x1c2c3e1cU, 0x1c340c34U, 0x1c34240cU, 0x1c3e040cU, 0x1c3e041cU,
    0x1c3e1404U, 0x1c3e1414U, 0x1c3e1c2cU, 0x24040404U, 0x24040424U, 0x24040c14U, 0x24041404U, 0x24041424U,
    0x2404143eU, 0x24041c14U, 0x2404240cU, 0x24042c04U, 0x24043e04U, 0x240c0414U, 0x240c043eU, 0x240c0c0cU,
    0x240c0c1cU, 0x240c1414U, 0x240c1c04U, 0x240c1c2cU, 0x240c241cU, 0x240c2c0cU, 0x240c2c2cU, 0x2414040cU,
    0x2414041cU, 0x24140c04U, 0x24140c2cU, 0x2414140cU, 0x24141c1cU, 0x24142404U, 0x24142c3eU, 0x24143414U,
    0x24143e04U, 0x241c0424U, 0x241c0c0cU, 0x241c0c1cU, 0x241c1404U, 0x241c1414U, 0x241c1c0cU, 0x241c1c2cU,
    0x24240404U, 0x24240414U, 0x24241424U, 0x24241c3eU, 0x24242404U, 0x24243e0cU, 0x242c042cU, 0x242c043eU,
    0x242c140cU, 0x242c3414U, 0x24340c1cU, 0x24341c24U, 0x24343404U, 0x243e0c04U, 0x243e0c2cU, 0x243e1c04U,
    0x243e241cU, 0x243e2c0cU, 0x2c040414U, 0x2c040c04U, 0x2c040c24U, 0x2c041414U, 0x2c042404U, 0x2c042424U,
    0x2c04243eU, 0x2c042c14U, 0x2c043434U, 0x2c043e24U, 0x2c0c040cU, 0x2c0c041cU, 0x2c0c042cU, 0x2c0c0c14U,
    0x2c0c140cU, 0x2c0c1c14U, 0x2c0c3e14U, 0x2c140404U, 0x2c140c0cU, 0x2c14141cU, 0x2c141c04U, 0x2c141c34U,
    0x2c142c1cU, 0x2c1c0414U, 0x2c1c043eU, 0x2c1c0c04U, 0x2c1c143eU, 0x2c1c2424U, 0x2c1c2c0cU, 0x2c1c342cU,
    0x2c1c3e1cU, 0x2c24040cU, 0x2c240424U, 0x2c241404U, 0x2c241c14U, 0x2c242434U, 0x2c2c0c14U, 0x2c2c1434U,
    0x2c2c2c0cU, 0x2c2c2c1cU, 0x2c342414U, 0x2c3e0414U, 0x2c3e0424U, 0x2c3e1414U, 0x34040c0cU, 0x34040c1cU,
    0x34040c2cU, 0x34041c0cU, 0x34041c1cU, 0x34043404U, 0x340c0404U, 0x340c1404U, 0x340c143eU, 0x340c3424U,
    0x34140c14U, 0x34141c24U, 0x34142414U, 0x34142c2cU, 0x34143414U, 0x34143e04U, 0x341c0404U, 0x341c0c24U,
    0x341c140cU, 0x341c2404U, 0x3424142cU, 0x3424241cU, 0x34243414U, 0x342c0404U, 0x342c041cU, 0x342c1c24U,
    0x342c3404U, 0x3434042cU, 0x34342404U, 0x343e0c0cU, 0x343e0c1cU, 0x3e040404U, 0x3e040424U, 0x3e04043eU,
    0x3e041404U, 0x3e041414U, 0x3e041c34U, 0x3e042404U, 0x3e042c24U, 0x3e043414U, 0x3e0c0414U, 0x3e0c0c0cU,
    0x3e0c1424U, 0x3e0c241cU, 0x3e0c242cU, 0x3e14040cU, 0x3e140424U, 0x3e140c04U, 0x3e140c34U, 0x3e14140cU,
    0x3e141c04U, 0x3e142c0cU, 0x3e1c0414U, 0x3e1c1c14U, 0x3e1c1c2cU, 0x3e1c2c1cU, 0x3e24040cU, 0x3e24042cU,
    0x3e240c1cU, 0x3e241404U, 0x3e242c04U, 0x3e2c1414U, 0x3e2c2414U, 0x3e340414U, 0x3e341c0cU, 0x3e3e0404U,
};

static __device__ const uint64_t iq1s_grid_cu[2048] = {
    0x00000000ULL, 0x00000002ULL, 0x00000101ULL, 0x00000200ULL, 0x00000202ULL, 0x00010001ULL, 0x00010101ULL, 0x00020000ULL,
    0x00020002ULL, 0x00020200ULL, 0x00020202ULL, 0x01000101ULL, 0x01010001ULL, 0x01010100ULL, 0x01010102ULL, 0x01020101ULL,
    0x02000000ULL, 0x02000002ULL, 0x02000200ULL, 0x02000202ULL, 0x02010101ULL, 0x02020000ULL, 0x02020002ULL, 0x02020200ULL,
    0x02020202ULL, 0x00000110ULL, 0x00000111ULL, 0x00010011ULL, 0x00010110ULL, 0x00010112ULL, 0x00010211ULL, 0x00010212ULL,
    0x00020111ULL, 0x01000011ULL, 0x01000112ULL, 0x01000211ULL, 0x01010012ULL, 0x01010111ULL, 0x01010212ULL, 0x01020011ULL,
    0x01020110ULL, 0x01020112ULL, 0x01020210ULL, 0x02000111ULL, 0x02010011ULL, 0x02010110ULL, 0x02010112ULL, 0x02020111ULL,
    0x00000020ULL, 0x00000022ULL, 0x00000220ULL, 0x00000222ULL, 0x00010121ULL, 0x00020020ULL, 0x00020022ULL, 0x00020220ULL,
    0x00020222ULL, 0x01000121ULL, 0x01010021ULL, 0x01010221ULL, 0x01020120ULL, 0x01020221ULL, 0x02000020ULL, 0x02000022ULL,
    0x02000220ULL, 0x02000222ULL, 0x02010021ULL, 0x02010121ULL, 0x02010221ULL, 0x02020020ULL, 0x02020022ULL, 0x02020220ULL,
    0x02020222ULL, 0x00011001ULL, 0x00011100ULL, 0x00011102ULL, 0x00021101ULL, 0x01001001ULL, 0x01001201ULL, 0x01011101ULL,
    0x01011202ULL, 0x01021100ULL, 0x01021101ULL, 0x02011001ULL, 0x02011201ULL, 0x02021101ULL, 0x00001011ULL, 0x00001110ULL,
    0x00001111ULL, 0x00001112ULL, 0x00011111ULL, 0x00011210ULL, 0x00011212ULL, 0x00021211ULL, 0x01001010ULL, 0x01001111ULL,
    0x01001212ULL, 0x01011010ULL, 0x01011011ULL, 0x01011110ULL, 0x01011111ULL, 0x01011112ULL, 0x01011211ULL, 0x01021010ULL,
    0x01021012ULL, 0x01021111ULL, 0x01021210ULL, 0x01021212ULL, 0x02001011ULL, 0x02011011ULL, 0x02011111ULL, 0x02011210ULL,
    0x02011212ULL, 0x02021011ULL, 0x02021110ULL, 0x02021111ULL, 0x02021112ULL, 0x02021211ULL, 0x00011120ULL, 0x00011221ULL,
    0x01001021ULL, 0x01001120ULL, 0x01011020ULL, 0x01011022ULL, 0x01011121ULL, 0x01011220ULL, 0x01021020ULL, 0x01021021ULL,
    0x01021122ULL, 0x01021221ULL, 0x02001121ULL, 0x02011021ULL, 0x02011120ULL, 0x02011221ULL, 0x00002000ULL, 0x00002002ULL,
    0x00002200ULL, 0x00002202ULL, 0x00012101ULL, 0x00022000ULL, 0x00022002ULL, 0x00022200ULL, 0x00022202ULL, 0x01002101ULL,
    0x01012001ULL, 0x01012102ULL, 0x01022101ULL, 0x02002000ULL, 0x02002002ULL, 0x02002200ULL, 0x02002202ULL, 0x02012101ULL,
    0x02022000ULL, 0x02022002ULL, 0x02022200ULL, 0x02022202ULL, 0x00002111ULL, 0x00012011ULL, 0x00012110ULL, 0x00012211ULL,
    0x00022110ULL, 0x00022111ULL, 0x01002011ULL, 0x01012010ULL, 0x01012011ULL, 0x01012111ULL, 0x01022011ULL, 0x01022110ULL,
    0x01022211ULL, 0x02012011ULL, 0x02012110ULL, 0x02012112ULL, 0x02012211ULL, 0x02022111ULL, 0x00002020ULL, 0x00002022ULL,
    0x00002220ULL, 0x00002222ULL, 0x00012121ULL, 0x00022020ULL, 0x00022022ULL, 0x00022220ULL, 0x00022222ULL, 0x01002121ULL,
    0x01012021ULL, 0x01012221ULL, 0x01022021ULL, 0x01022121ULL, 0x02002020ULL, 0x02002022ULL, 0x02002121ULL, 0x02002220ULL,
    0x02002222ULL, 0x02012121ULL, 0x02022020ULL, 0x02022022ULL, 0x02022220ULL, 0x02022222ULL, 0x00110000ULL, 0x00110001ULL,
    0x00110100ULL, 0x00110201ULL, 0x00120100ULL, 0x00120101ULL, 0x01100001ULL, 0x01100100ULL, 0x01110000ULL, 0x01110101ULL,
    0x01110200ULL, 0x01120001ULL, 0x01120100ULL, 0x01120101ULL, 0x01120201ULL, 0x02110001ULL, 0x02110100ULL, 0x02110102ULL,
    0x02120001ULL, 0x02120101ULL, 0x00100011ULL, 0x00100110ULL, 0x00100112ULL, 0x00100211ULL, 0x00110010ULL, 0x00110012ULL,
    0x00110111ULL, 0x00110210ULL, 0x00120011ULL, 0x00120110ULL, 0x00120211ULL, 0x01100111ULL, 0x01100212ULL, 0x01110010ULL,
    0x01110011ULL, 0x01110012ULL, 0x01110110ULL, 0x01110111ULL, 0x01110112ULL, 0x01110211ULL, 0x01120010ULL, 0x01120111ULL,
    0x02100110ULL, 0x02110012ULL, 0x02110111ULL, 0x02120011ULL, 0x02120110ULL, 0x00110021ULL, 0x00110120ULL, 0x00110122ULL,
    0x00120121ULL, 0x01100020ULL, 0x01100122ULL, 0x01100221ULL, 0x01110022ULL, 0x01110121ULL, 0x01110220ULL, 0x01110222ULL,
    0x01120120ULL, 0x01120122ULL, 0x02100121ULL, 0x02110021ULL, 0x02110120ULL, 0x02110122ULL, 0x02120121ULL, 0x00101001ULL,
    0x00101102ULL, 0x00101201ULL, 0x00111100ULL, 0x00111101ULL, 0x00111200ULL, 0x00111201ULL, 0x00121001ULL, 0x00121102ULL,
    0x01101001ULL, 0x01101101ULL, 0x01101102ULL, 0x01101200ULL, 0x01101202ULL, 0x01111001ULL, 0x01111100ULL, 0x01111101ULL,
    0x01111102ULL, 0x01111201ULL, 0x01121002ULL, 0x01121101ULL, 0x01121200ULL, 0x02101100ULL, 0x02101201ULL, 0x02111000ULL,
    0x02111100ULL, 0x02111101ULL, 0x02111200ULL, 0x02111201ULL, 0x02111202ULL, 0x02121001ULL, 0x02121100ULL, 0x02121101ULL,
    0x02121201ULL, 0x00101012ULL, 0x00101111ULL, 0x00101212ULL, 0x00111011ULL, 0x00111110ULL, 0x00111111ULL, 0x00111112ULL,
    0x00111211ULL, 0x00121010ULL, 0x00121012ULL, 0x00121111ULL, 0x00121210ULL, 0x00121212ULL, 0x01101011ULL, 0x01101110ULL,
    0x01101111ULL, 0x01101112ULL, 0x01111011ULL, 0x01111012ULL, 0x01111110ULL, 0x01111111ULL, 0x01111112ULL, 0x01111211ULL,
    0x01111212ULL, 0x01121011ULL, 0x01121110ULL, 0x01121111ULL, 0x01121112ULL, 0x01121211ULL, 0x02101010ULL, 0x02101012ULL,
    0x02101110ULL, 0x02101111ULL, 0x02101210ULL, 0x02101212ULL, 0x02111010ULL, 0x02111011ULL, 0x02111110ULL, 0x02111111ULL,
    0x02111112ULL, 0x02111211ULL, 0x02111212ULL, 0x02121010ULL, 0x02121012ULL, 0x02121111ULL, 0x00101021ULL, 0x00101120ULL,
    0x00101121ULL, 0x00101122ULL, 0x00111121ULL, 0x00111122ULL, 0x00111220ULL, 0x00111222ULL, 0x00121021ULL, 0x00121122ULL,
    0x01101020ULL, 0x01101022ULL, 0x01101120ULL, 0x01101121ULL, 0x01101220ULL, 0x01101222ULL, 0x01111021ULL, 0x01111121ULL,
    0x01111122ULL, 0x01111220ULL, 0x01111221ULL, 0x01121021ULL, 0x01121120ULL, 0x01121121ULL, 0x01121220ULL, 0x01121221ULL,
    0x01121222ULL, 0x02101122ULL, 0x02101222ULL, 0x02111022ULL, 0x02111121ULL, 0x02121120ULL, 0x02121221ULL, 0x00112001ULL,
    0x00112102ULL, 0x00122101ULL, 0x01102001ULL, 0x01102100ULL, 0x01102102ULL, 0x01102201ULL, 0x01112000ULL, 0x01112101ULL,
    0x01112200ULL, 0x01112202ULL, 0x01122000ULL, 0x01122001ULL, 0x01122100ULL, 0x01122102ULL, 0x01122201ULL, 0x02102101ULL,
    0x02112001ULL, 0x02112100ULL, 0x02122101ULL, 0x00112010ULL, 0x00112012ULL, 0x00112111ULL, 0x00112212ULL, 0x00122011ULL,
    0x00122111ULL, 0x01102012ULL, 0x01102110ULL, 0x01102111ULL, 0x01102210ULL, 0x01112011ULL, 0x01112110ULL, 0x01112111ULL,
    0x01112112ULL, 0x01112211ULL, 0x01112212ULL, 0x01122010ULL, 0x01122111ULL, 0x01122212ULL, 0x02102211ULL, 0x02112011ULL,
    0x02112012ULL, 0x02112111ULL, 0x02112210ULL, 0x02122011ULL, 0x02122112ULL, 0x02122211ULL, 0x00102221ULL, 0x00112122ULL,
    0x00122120ULL, 0x00122122ULL, 0x01102120ULL, 0x01102122ULL, 0x01102221ULL, 0x01112020ULL, 0x01112022ULL, 0x01112121ULL,
    0x01112220ULL, 0x01122021ULL, 0x01122122ULL, 0x01122221ULL, 0x02102121ULL, 0x02112021ULL, 0x02112122ULL, 0x02112222ULL,
    0x00200000ULL, 0x00200002ULL, 0x00200200ULL, 0x00200202ULL, 0x00210101ULL, 0x00220000ULL, 0x00220002ULL, 0x00220101ULL,
    0x00220200ULL, 0x00220202ULL, 0x01200101ULL, 0x01210001ULL, 0x01210201ULL, 0x01220001ULL, 0x01220101ULL, 0x02200000ULL,
    0x02200002ULL, 0x02200200ULL, 0x02200202ULL, 0x02210101ULL, 0x02220000ULL, 0x02220002ULL, 0x02220101ULL, 0x02220200ULL,
    0x02220202ULL, 0x00200111ULL, 0x00210011ULL, 0x00210110ULL, 0x00210211ULL, 0x00220111ULL, 0x01200012ULL, 0x01200110ULL,
    0x01200211ULL, 0x01210111ULL, 0x01210210ULL, 0x01210212ULL, 0x01220011ULL, 0x01220110ULL, 0x01220111ULL, 0x01220112ULL,
    0x02200111ULL, 0x02210010ULL, 0x02210112ULL, 0x02210211ULL, 0x02220111ULL, 0x00200021ULL, 0x00200220ULL, 0x00200222ULL,
    0x00210021ULL, 0x00210121ULL, 0x00220020ULL, 0x00220022ULL, 0x00220220ULL, 0x00220222ULL, 0x01200121ULL, 0x01210021ULL,
    0x01210122ULL, 0x01210221ULL, 0x01220121ULL, 0x02200021ULL, 0x02200220ULL, 0x02200222ULL, 0x02210021ULL, 0x02210121ULL,
    0x02220020ULL, 0x02220022ULL, 0x02220220ULL, 0x02220222ULL, 0x00201101ULL, 0x00211100ULL, 0x00211102ULL, 0x00211201ULL,
    0x00221101ULL, 0x01201100ULL, 0x01201101ULL, 0x01201102ULL, 0x01201201ULL, 0x01211002ULL, 0x01211101ULL, 0x01211200ULL,
    0x01211202ULL, 0x01221102ULL, 0x02201101ULL, 0x02211001ULL, 0x02211100ULL, 0x02211201ULL, 0x02221001ULL, 0x02221101ULL,
    0x00201211ULL, 0x00211111ULL, 0x00221011ULL, 0x00221211ULL, 0x01201010ULL, 0x01201111ULL, 0x01201210ULL, 0x01211011ULL,
    0x01211110ULL, 0x01211111ULL, 0x01211211ULL, 0x01221012ULL, 0x01221111ULL, 0x01221210ULL, 0x02201211ULL, 0x02211010ULL,
    0x02211110ULL, 0x02211111ULL, 0x02211210ULL, 0x02211212ULL, 0x02221011ULL, 0x02221110ULL, 0x02221112ULL, 0x02221211ULL,
    0x00201121ULL, 0x00211020ULL, 0x00211022ULL, 0x00211221ULL, 0x00221121ULL, 0x01201021ULL, 0x01201221ULL, 0x01211121ULL,
    0x01221020ULL, 0x01221021ULL, 0x01221221ULL, 0x02201120ULL, 0x02201122ULL, 0x02211020ULL, 0x02211222ULL, 0x00202000ULL,
    0x00202002ULL, 0x00202200ULL, 0x00202202ULL, 0x00212101ULL, 0x00222000ULL, 0x00222002ULL, 0x00222200ULL, 0x00222202ULL,
    0x01202101ULL, 0x01212001ULL, 0x01212100ULL, 0x01222101ULL, 0x02202000ULL, 0x02202002ULL, 0x02202200ULL, 0x02202202ULL,
    0x02222000ULL, 0x02222002ULL, 0x02222200ULL, 0x02222202ULL, 0x00202211ULL, 0x00212011ULL, 0x00212110ULL, 0x00212211ULL,
    0x00222111ULL, 0x01202112ULL, 0x01202211ULL, 0x01212012ULL, 0x01212111ULL, 0x01222011ULL, 0x01222110ULL, 0x01222112ULL,
    0x01222211ULL, 0x02202111ULL, 0x02212010ULL, 0x02212112ULL, 0x02212211ULL, 0x02222110ULL, 0x02222111ULL, 0x00202020ULL,
    0x00202022ULL, 0x00202220ULL, 0x00202222ULL, 0x00222020ULL, 0x00222022ULL, 0x00222220ULL, 0x00222222ULL, 0x01202121ULL,
    0x01212021ULL, 0x01212122ULL, 0x01212221ULL, 0x01222121ULL, 0x02202020ULL, 0x02202022ULL, 0x02202220ULL, 0x02202222ULL,
    0x02212121ULL, 0x02222020ULL, 0x02222022ULL, 0x02222220ULL, 0x02222222ULL, 0x10000101ULL, 0x10010001ULL, 0x10010102ULL,
    0x10020101ULL, 0x11000201ULL, 0x11010002ULL, 0x11010101ULL, 0x11010200ULL, 0x11010202ULL, 0x11020001ULL, 0x11020100ULL,
    0x11020102ULL, 0x12010100ULL, 0x12010201ULL, 0x12020001ULL, 0x12020102ULL, 0x10000010ULL, 0x10000011ULL, 0x10000110ULL,
    0x10000112ULL, 0x10000211ULL, 0x10010012ULL, 0x10010111ULL, 0x10010112ULL, 0x10010210ULL, 0x10010212ULL, 0x10020011ULL,
    0x10020112ULL, 0x10020211ULL, 0x11000111ULL, 0x11000210ULL, 0x11000212ULL, 0x11010011ULL, 0x11010110ULL, 0x11010111ULL,
    0x11010112ULL, 0x11010211ULL, 0x11010212ULL, 0x11020111ULL, 0x11020210ULL, 0x11020212ULL, 0x12000011ULL, 0x12000110ULL,
    0x12000112ULL, 0x12010010ULL, 0x12010012ULL, 0x12010111ULL, 0x12020010ULL, 0x12020011ULL, 0x12020012ULL, 0x10000121ULL,
    0x10010021ULL, 0x10010120ULL, 0x10010122ULL, 0x10020121ULL, 0x11000021ULL, 0x11010022ULL, 0x11010121ULL, 0x11010222ULL,
    0x11020120ULL, 0x11020221ULL, 0x12000221ULL, 0x12010120ULL, 0x12020121ULL, 0x10001001ULL, 0x10011101ULL, 0x10011201ULL,
    0x10021201ULL, 0x11001101ULL, 0x11001200ULL, 0x11001202ULL, 0x11011001ULL, 0x11011100ULL, 0x11011101ULL, 0x11011102ULL,
    0x11021001ULL, 0x11021002ULL, 0x11021101ULL, 0x11021200ULL, 0x11021202ULL, 0x12001001ULL, 0x12001102ULL, 0x12001201ULL,
    0x12011000ULL, 0x12011002ULL, 0x12011101ULL, 0x12021000ULL, 0x12021001ULL, 0x12021201ULL, 0x10001011ULL, 0x10001012ULL,
    0x10001111ULL, 0x10001212ULL, 0x10011011ULL, 0x10011110ULL, 0x10011111ULL, 0x10011112ULL, 0x10011211ULL, 0x10021010ULL,
    0x10021111ULL, 0x10021212ULL, 0x11001011ULL, 0x11001110ULL, 0x11001111ULL, 0x11001112ULL, 0x11001211ULL, 0x11011010ULL,
    0x11011011ULL, 0x11011110ULL, 0x11011111ULL, 0x11011112ULL, 0x11011210ULL, 0x11011211ULL, 0x11021011ULL, 0x11021110ULL,
    0x11021111ULL, 0x11021112ULL, 0x11021211ULL, 0x12001012ULL, 0x12001110ULL, 0x12001111ULL, 0x12001210ULL, 0x12011011ULL,
    0x12011110ULL, 0x12011111ULL, 0x12011112ULL, 0x12011211ULL, 0x12011212ULL, 0x12021111ULL, 0x12021210ULL, 0x12021212ULL,
    0x10001021ULL, 0x10001121ULL, 0x10001221ULL, 0x10011120ULL, 0x10011121ULL, 0x10011220ULL, 0x10011222ULL, 0x10021021ULL,
    0x10021120ULL, 0x10021221ULL, 0x11001020ULL, 0x11001022ULL, 0x11001121ULL, 0x11001220ULL, 0x11011020ULL, 0x11011021ULL,
    0x11011022ULL, 0x11011121ULL, 0x11011122ULL, 0x11011221ULL, 0x11021022ULL, 0x11021121ULL, 0x11021220ULL, 0x12001021ULL,
    0x12001121ULL, 0x12001222ULL, 0x12011120ULL, 0x12011121ULL, 0x12021021ULL, 0x12021120ULL, 0x12021122ULL, 0x10002101ULL,
    0x10012001ULL, 0x10012101ULL, 0x10012202ULL, 0x10022101ULL, 0x11002002ULL, 0x11002201ULL, 0x11012000ULL, 0x11012101ULL,
    0x11012200ULL, 0x11022001ULL, 0x11022100ULL, 0x11022102ULL, 0x11022201ULL, 0x12002101ULL, 0x12012001ULL, 0x12012100ULL,
    0x12012102ULL, 0x12012201ULL, 0x12022101ULL, 0x10002011ULL, 0x10002111ULL, 0x10002112ULL, 0x10002212ULL, 0x10012010ULL,
    0x10012110ULL, 0x10012111ULL, 0x10012210ULL, 0x10022011ULL, 0x10022110ULL, 0x10022112ULL, 0x11002010ULL, 0x11002111ULL,
    0x11002212ULL, 0x11012011ULL, 0x11012012ULL, 0x11012110ULL, 0x11012111ULL, 0x11012112ULL, 0x11012211ULL, 0x11022010ULL,
    0x11022012ULL, 0x11022111ULL, 0x11022112ULL, 0x11022212ULL, 0x12002112ULL, 0x12002211ULL, 0x12012012ULL, 0x12012111ULL,
    0x12012112ULL, 0x12012210ULL, 0x12022011ULL, 0x12022110ULL, 0x12022112ULL, 0x12022211ULL, 0x10012122ULL, 0x11002120ULL,
    0x11002122ULL, 0x11002221ULL, 0x11012121ULL, 0x11012220ULL, 0x11012222ULL, 0x11022120ULL, 0x11022221ULL, 0x12012120ULL,
    0x12022121ULL, 0x10100001ULL, 0x10100100ULL, 0x10100101ULL, 0x10100102ULL, 0x10100201ULL, 0x10110002ULL, 0x10110101ULL,
    0x10110202ULL, 0x10120001ULL, 0x10120100ULL, 0x10120201ULL, 0x11100000ULL, 0x11100101ULL, 0x11100200ULL, 0x11110001ULL,
    0x11110100ULL, 0x11110101ULL, 0x11110102ULL, 0x11110201ULL, 0x11120101ULL, 0x11120200ULL, 0x12100102ULL, 0x12100201ULL,
    0x12110101ULL, 0x12110200ULL, 0x12120000ULL, 0x12120001ULL, 0x12120102ULL, 0x12120201ULL, 0x10100111ULL, 0x10100210ULL,
    0x10100211ULL, 0x10100212ULL, 0x10110011ULL, 0x10110110ULL, 0x10110111ULL, 0x10110112ULL, 0x10110210ULL, 0x10110211ULL,
    0x10120010ULL, 0x10120111ULL, 0x10120112ULL, 0x10120210ULL, 0x10120212ULL, 0x11100011ULL, 0x11100110ULL, 0x11100111ULL,
    0x11100112ULL, 0x11100211ULL, 0x11110010ULL, 0x11110011ULL, 0x11110012ULL, 0x11110110ULL, 0x11110111ULL, 0x11110112ULL,
    0x11110210ULL, 0x11110211ULL, 0x11110212ULL, 0x11120011ULL, 0x11120110ULL, 0x11120111ULL, 0x11120112ULL, 0x11120211ULL,
    0x12100012ULL, 0x12100111ULL, 0x12110011ULL, 0x12110110ULL, 0x12110111ULL, 0x12110112ULL, 0x12110211ULL, 0x12120010ULL,
    0x12120111ULL, 0x12120212ULL, 0x10100021ULL, 0x10100122ULL, 0x10110022ULL, 0x10110121ULL, 0x10110222ULL, 0x10120021ULL,
    0x10120120ULL, 0x11100022ULL, 0x11100121ULL, 0x11100222ULL, 0x11110021ULL, 0x11110120ULL, 0x11110121ULL, 0x11110122ULL,
    0x11110221ULL, 0x11120022ULL, 0x11120121ULL, 0x12100121ULL, 0x12110020ULL, 0x12110022ULL, 0x12110121ULL, 0x12110221ULL,
    0x12110222ULL, 0x12120120ULL, 0x10101100ULL, 0x10101101ULL, 0x10111001ULL, 0x10111100ULL, 0x10111101ULL, 0x10111102ULL,
    0x10111200ULL, 0x10111201ULL, 0x10121001ULL, 0x10121101ULL, 0x10121200ULL, 0x10121202ULL, 0x11101001ULL, 0x11101100ULL,
    0x11101101ULL, 0x11101102ULL, 0x11101201ULL, 0x11101202ULL, 0x11111000ULL, 0x11111001ULL, 0x11111100ULL, 0x11111101ULL,
    0x11111102ULL, 0x11111200ULL, 0x11111201ULL, 0x11111202ULL, 0x11121001ULL, 0x11121002ULL, 0x11121100ULL, 0x11121101ULL,
    0x11121102ULL, 0x11121201ULL, 0x12101000ULL, 0x12101200ULL, 0x12101202ULL, 0x12111001ULL, 0x12111100ULL, 0x12111101ULL,
    0x12111102ULL, 0x12111201ULL, 0x12121001ULL, 0x12121100ULL, 0x12121101ULL, 0x12121202ULL, 0x10101011ULL, 0x10101012ULL,
    0x10101110ULL, 0x10101111ULL, 0x10101112ULL, 0x10101211ULL, 0x10111010ULL, 0x10111011ULL, 0x10111012ULL, 0x10111110ULL,
    0x10111111ULL, 0x10111112ULL, 0x10111211ULL, 0x10111212ULL, 0x10121011ULL, 0x10121110ULL, 0x10121111ULL, 0x10121112ULL,
    0x10121211ULL, 0x11101010ULL, 0x11101011ULL, 0x11101012ULL, 0x11101110ULL, 0x11101111ULL, 0x11101112ULL, 0x11101210ULL,
    0x11101211ULL, 0x11111010ULL, 0x11111011ULL, 0x11111012ULL, 0x11111110ULL, 0x11111111ULL, 0x11111112ULL, 0x11111210ULL,
    0x11111211ULL, 0x11111212ULL, 0x11121010ULL, 0x11121011ULL, 0x11121110ULL, 0x11121111ULL, 0x11121112ULL, 0x11121210ULL,
    0x11121211ULL, 0x11121212ULL, 0x12101011ULL, 0x12101110ULL, 0x12101111ULL, 0x12101211ULL, 0x12101212ULL, 0x12111010ULL,
    0x12111011ULL, 0x12111110ULL, 0x12111111ULL, 0x12111112ULL, 0x12111210ULL, 0x12111211ULL, 0x12121011ULL, 0x12121110ULL,
    0x12121111ULL, 0x12121112ULL, 0x12121211ULL, 0x10101020ULL, 0x10101021ULL, 0x10101022ULL, 0x10101120ULL, 0x10101122ULL,
    0x10101220ULL, 0x10101221ULL, 0x10111021ULL, 0x10111120ULL, 0x10111121ULL, 0x10111220ULL, 0x10111221ULL, 0x10121020ULL,
    0x10121021ULL, 0x10121022ULL, 0x10121120ULL, 0x10121121ULL, 0x10121122ULL, 0x10121220ULL, 0x10121221ULL, 0x11101021ULL,
    0x11101121ULL, 0x11101122ULL, 0x11101220ULL, 0x11101221ULL, 0x11101222ULL, 0x11111020ULL, 0x11111021ULL, 0x11111022ULL,
    0x11111120ULL, 0x11111121ULL, 0x11111122ULL, 0x11111220ULL, 0x11111221ULL, 0x11111222ULL, 0x11121021ULL, 0x11121120ULL,
    0x11121121ULL, 0x11121221ULL, 0x12101022ULL, 0x12101121ULL, 0x12101122ULL, 0x12101220ULL, 0x12101221ULL, 0x12101222ULL,
    0x12111021ULL, 0x12111121ULL, 0x12111222ULL, 0x12121022ULL, 0x12121121ULL, 0x12121122ULL, 0x12121220ULL, 0x12121221ULL,
    0x10102100ULL, 0x10102101ULL, 0x10102102ULL, 0x10102201ULL, 0x10112000ULL, 0x10112101ULL, 0x10112200ULL, 0x10122001ULL,
    0x10122202ULL, 0x11102101ULL, 0x11102200ULL, 0x11102202ULL, 0x11112001ULL, 0x11112100ULL, 0x11112101ULL, 0x11112102ULL,
    0x11112200ULL, 0x11112201ULL, 0x11122000ULL, 0x11122002ULL, 0x11122100ULL, 0x11122101ULL, 0x12102002ULL, 0x12102201ULL,
    0x12112000ULL, 0x12112002ULL, 0x12112101ULL, 0x12112200ULL, 0x12122001ULL, 0x12122201ULL, 0x10102011ULL, 0x10102012ULL,
    0x10102111ULL, 0x10102212ULL, 0x10112011ULL, 0x10112110ULL, 0x10112111ULL, 0x10112112ULL, 0x10112211ULL, 0x10122111ULL,
    0x11102011ULL, 0x11102110ULL, 0x11102111ULL, 0x11102112ULL, 0x11102211ULL, 0x11112010ULL, 0x11112011ULL, 0x11112012ULL,
    0x11112110ULL, 0x11112111ULL, 0x11112112ULL, 0x11112210ULL, 0x11112211ULL, 0x11112212ULL, 0x11122011ULL, 0x11122110ULL,
    0x11122111ULL, 0x11122112ULL, 0x11122211ULL, 0x12102011ULL, 0x12102111ULL, 0x12102211ULL, 0x12112011ULL, 0x12112110ULL,
    0x12112111ULL, 0x12112112ULL, 0x12112210ULL, 0x12112211ULL, 0x12122111ULL, 0x10102120ULL, 0x10102220ULL, 0x10112121ULL,
    0x10112222ULL, 0x10122020ULL, 0x10122121ULL, 0x10122122ULL, 0x10122221ULL, 0x11102121ULL, 0x11102220ULL, 0x11102221ULL,
    0x11112021ULL, 0x11112121ULL, 0x11112122ULL, 0x11112220ULL, 0x11112221ULL, 0x11122022ULL, 0x11122121ULL, 0x11122220ULL,
    0x11122222ULL, 0x12102021ULL, 0x12102222ULL, 0x12112022ULL, 0x12112121ULL, 0x12112122ULL, 0x12112220ULL, 0x12112222ULL,
    0x12122021ULL, 0x10200101ULL, 0x10210100ULL, 0x10210102ULL, 0x10210201ULL, 0x10220101ULL, 0x11200100ULL, 0x11210000ULL,
    0x11210101ULL, 0x11210102ULL, 0x11210200ULL, 0x11210202ULL, 0x11220001ULL, 0x11220100ULL, 0x11220102ULL, 0x11220201ULL,
    0x12200001ULL, 0x12210102ULL, 0x12220101ULL, 0x10200011ULL, 0x10200110ULL, 0x10200112ULL, 0x10200211ULL, 0x10210012ULL,
    0x10210111ULL, 0x10220011ULL, 0x10220012ULL, 0x10220112ULL, 0x10220211ULL, 0x11200111ULL, 0x11200211ULL, 0x11210011ULL,
    0x11210111ULL, 0x11210112ULL, 0x11210211ULL, 0x11220111ULL, 0x11220112ULL, 0x11220212ULL, 0x12200110ULL, 0x12200212ULL,
    0x12210012ULL, 0x12210111ULL, 0x12220011ULL, 0x12220112ULL, 0x12220211ULL, 0x10210021ULL, 0x10210122ULL, 0x10210221ULL,
    0x11200020ULL, 0x11200021ULL, 0x11200122ULL, 0x11210121ULL, 0x11210122ULL, 0x11210220ULL, 0x11220020ULL, 0x12200121ULL,
    0x12210021ULL, 0x12210122ULL, 0x12220121ULL, 0x10211001ULL, 0x10211002ULL, 0x10211101ULL, 0x10211102ULL, 0x10211202ULL,
    0x10221001ULL, 0x10221102ULL, 0x10221201ULL, 0x11201000ULL, 0x11201002ULL, 0x11201101ULL, 0x11201200ULL, 0x11201202ULL,
    0x11211001ULL, 0x11211100ULL, 0x11211101ULL, 0x11211102ULL, 0x11211201ULL, 0x11211202ULL, 0x11221000ULL, 0x11221002ULL,
    0x11221101ULL, 0x12201100ULL, 0x12201101ULL, 0x12201201ULL, 0x12211000ULL, 0x12211002ULL, 0x12211100ULL, 0x12211101ULL,
    0x12211102ULL, 0x12211200ULL, 0x12211202ULL, 0x12221001ULL, 0x12221100ULL, 0x12221201ULL, 0x10201111ULL, 0x10201210ULL,
    0x10201212ULL, 0x10211011ULL, 0x10211111ULL, 0x10211112ULL, 0x10211211ULL, 0x11201110ULL, 0x11201111ULL, 0x11201112ULL,
    0x11201211ULL, 0x11211010ULL, 0x11211011ULL, 0x11211110ULL, 0x11211111ULL, 0x11211112ULL, 0x11211211ULL, 0x11221011ULL,
    0x11221110ULL, 0x11221111ULL, 0x11221112ULL, 0x11221211ULL, 0x12201112ULL, 0x12201211ULL, 0x12201212ULL, 0x12211011ULL,
    0x12211111ULL, 0x12211112ULL, 0x12211211ULL, 0x12211212ULL, 0x12221012ULL, 0x12221111ULL, 0x12221112ULL, 0x12221210ULL,
    0x10201022ULL, 0x10201221ULL, 0x10211121ULL, 0x10221020ULL, 0x10221122ULL, 0x10221220ULL, 0x10221221ULL, 0x11201020ULL,
    0x11201121ULL, 0x11201220ULL, 0x11201222ULL, 0x11211021ULL, 0x11211120ULL, 0x11211121ULL, 0x11211122ULL, 0x11211220ULL,
    0x11211222ULL, 0x11221020ULL, 0x11221121ULL, 0x11221220ULL, 0x12201020ULL, 0x12201022ULL, 0x12201121ULL, 0x12201222ULL,
    0x12211120ULL, 0x12211122ULL, 0x12211220ULL, 0x12211221ULL, 0x12221020ULL, 0x12221120ULL, 0x12221122ULL, 0x12221222ULL,
    0x10212102ULL, 0x10212201ULL, 0x10222101ULL, 0x11202001ULL, 0x11212002ULL, 0x11212101ULL, 0x11212202ULL, 0x11222001ULL,
    0x11222201ULL, 0x12202101ULL, 0x12212001ULL, 0x12212200ULL, 0x12222102ULL, 0x10202011ULL, 0x10202110ULL, 0x10212010ULL,
    0x10212111ULL, 0x10222011ULL, 0x10222110ULL, 0x10222112ULL, 0x10222211ULL, 0x11202010ULL, 0x11202011ULL, 0x11202111ULL,
    0x11202112ULL, 0x11202210ULL, 0x11212011ULL, 0x11212110ULL, 0x11212111ULL, 0x11212112ULL, 0x11212211ULL, 0x11222010ULL,
    0x11222111ULL, 0x11222212ULL, 0x12202012ULL, 0x12202110ULL, 0x12202212ULL, 0x12212111ULL, 0x12222011ULL, 0x12222110ULL,
    0x12222111ULL, 0x12222211ULL, 0x10212021ULL, 0x10212122ULL, 0x10212220ULL, 0x11202021ULL, 0x11202120ULL, 0x11202221ULL,
    0x11212020ULL, 0x11212121ULL, 0x11212220ULL, 0x11212222ULL, 0x11222120ULL, 0x11222121ULL, 0x11222221ULL, 0x12202122ULL,
    0x12212120ULL, 0x12212220ULL, 0x12212222ULL, 0x12222122ULL, 0x20000000ULL, 0x20000002ULL, 0x20000200ULL, 0x20000202ULL,
    0x20020000ULL, 0x20020002ULL, 0x20020200ULL, 0x20020202ULL, 0x21000101ULL, 0x21010000ULL, 0x21010001ULL, 0x21010100ULL,
    0x21010102ULL, 0x21010201ULL, 0x21020101ULL, 0x22000000ULL, 0x22000002ULL, 0x22000200ULL, 0x22000202ULL, 0x22010101ULL,
    0x22020000ULL, 0x22020002ULL, 0x22020200ULL, 0x22020202ULL, 0x20000111ULL, 0x20010011ULL, 0x20010110ULL, 0x20010112ULL,
    0x20010211ULL, 0x20020111ULL, 0x21000011ULL, 0x21000110ULL, 0x21000211ULL, 0x21010010ULL, 0x21010012ULL, 0x21010111ULL,
    0x21010112ULL, 0x21010210ULL, 0x21010211ULL, 0x21020110ULL, 0x21020112ULL, 0x21020211ULL, 0x22000111ULL, 0x22000211ULL,
    0x22010110ULL, 0x22010112ULL, 0x22010211ULL, 0x22020111ULL, 0x20000020ULL, 0x20000022ULL, 0x20000220ULL, 0x20000222ULL,
    0x20010121ULL, 0x20020020ULL, 0x20020022ULL, 0x20020220ULL, 0x20020222ULL, 0x21010021ULL, 0x21010120ULL, 0x21010221ULL,
    0x21020121ULL, 0x22000020ULL, 0x22000022ULL, 0x22000220ULL, 0x22000222ULL, 0x22010121ULL, 0x22020020ULL, 0x22020022ULL,
    0x22020220ULL, 0x22020222ULL, 0x20011100ULL, 0x20011201ULL, 0x21001001ULL, 0x21001100ULL, 0x21011001ULL, 0x21011101ULL,
    0x21011202ULL, 0x21021001ULL, 0x21021100ULL, 0x21021201ULL, 0x22011100ULL, 0x22011201ULL, 0x20001011ULL, 0x20001211ULL,
    0x20011012ULL, 0x20011111ULL, 0x20011212ULL, 0x20021112ULL, 0x20021211ULL, 0x21001010ULL, 0x21001011ULL, 0x21001111ULL,
    0x21001210ULL, 0x21011011ULL, 0x21011110ULL, 0x21011111ULL, 0x21011112ULL, 0x21011211ULL, 0x21011212ULL, 0x21021111ULL,
    0x21021112ULL, 0x21021210ULL, 0x21021212ULL, 0x22001011ULL, 0x22001110ULL, 0x22001112ULL, 0x22001211ULL, 0x22011010ULL,
    0x22011012ULL, 0x22011111ULL, 0x22011210ULL, 0x22021112ULL, 0x20011021ULL, 0x20011122ULL, 0x20011221ULL, 0x20021121ULL,
    0x21001021ULL, 0x21001120ULL, 0x21001221ULL, 0x21001222ULL, 0x21011020ULL, 0x21011121ULL, 0x21011221ULL, 0x21011222ULL,
    0x21021021ULL, 0x21021122ULL, 0x21021222ULL, 0x22001121ULL, 0x22011021ULL, 0x22011222ULL, 0x22021120ULL, 0x20002000ULL,
    0x20002002ULL, 0x20002200ULL, 0x20002202ULL, 0x20012101ULL, 0x20022000ULL, 0x20022002ULL, 0x20022200ULL, 0x20022202ULL,
    0x21002001ULL, 0x21002101ULL, 0x21012001ULL, 0x21012100ULL, 0x21012201ULL, 0x21022101ULL, 0x21022201ULL, 0x22002000ULL,
    0x22002002ULL, 0x22002200ULL, 0x22002202ULL, 0x22012101ULL, 0x22022000ULL, 0x22022002ULL, 0x22022200ULL, 0x22022202ULL,
    0x20002111ULL, 0x20002112ULL, 0x20012011ULL, 0x20012110ULL, 0x20012112ULL, 0x20022111ULL, 0x21002011ULL, 0x21002110ULL,
    0x21002112ULL, 0x21002211ULL, 0x21012010ULL, 0x21012012ULL, 0x21012111ULL, 0x21012212ULL, 0x21022011ULL, 0x21022110ULL,
    0x22002111ULL, 0x22012112ULL, 0x22012211ULL, 0x22022111ULL, 0x20002020ULL, 0x20002022ULL, 0x20002220ULL, 0x20002222ULL,
    0x20012121ULL, 0x20022020ULL, 0x20022022ULL, 0x20022220ULL, 0x20022222ULL, 0x21002121ULL, 0x21012021ULL, 0x21012120ULL,
    0x21012122ULL, 0x22002020ULL, 0x22002022ULL, 0x22002220ULL, 0x22002222ULL, 0x22012121ULL, 0x22022020ULL, 0x22022022ULL,
    0x22022220ULL, 0x22022222ULL, 0x20100101ULL, 0x20110001ULL, 0x20110102ULL, 0x20110200ULL, 0x20110201ULL, 0x20120101ULL,
    0x21100001ULL, 0x21100102ULL, 0x21100201ULL, 0x21110101ULL, 0x21110200ULL, 0x21110202ULL, 0x21120201ULL, 0x21120202ULL,
    0x22100101ULL, 0x22110001ULL, 0x22110100ULL, 0x22110102ULL, 0x22110201ULL, 0x22120101ULL, 0x20100011ULL, 0x20100110ULL,
    0x20100112ULL, 0x20100211ULL, 0x20110010ULL, 0x20110111ULL, 0x20110210ULL, 0x20110212ULL, 0x20120011ULL, 0x20120110ULL,
    0x20120112ULL, 0x20120211ULL, 0x21100010ULL, 0x21100111ULL, 0x21110010ULL, 0x21110011ULL, 0x21110110ULL, 0x21110111ULL,
    0x21110112ULL, 0x21110211ULL, 0x21120012ULL, 0x21120111ULL, 0x22100110ULL, 0x22100112ULL, 0x22110012ULL, 0x22110111ULL,
    0x22110210ULL, 0x22120011ULL, 0x22120110ULL, 0x22120112ULL, 0x22120211ULL, 0x20100121ULL, 0x20110021ULL, 0x20110120ULL,
    0x20110221ULL, 0x20120121ULL, 0x21100120ULL, 0x21100122ULL, 0x21100221ULL, 0x21110020ULL, 0x21110022ULL, 0x21110121ULL,
    0x21110220ULL, 0x21120122ULL, 0x21120221ULL, 0x22100121ULL, 0x22110120ULL, 0x22110122ULL, 0x22120221ULL, 0x20101001ULL,
    0x20101100ULL, 0x20101102ULL, 0x20111000ULL, 0x20111101ULL, 0x20111200ULL, 0x20121102ULL, 0x21101000ULL, 0x21101202ULL,
    0x21111001ULL, 0x21111100ULL, 0x21111101ULL, 0x21111102ULL, 0x21111200ULL, 0x21111201ULL, 0x21121000ULL, 0x21121001ULL,
    0x21121002ULL, 0x21121101ULL, 0x22101100ULL, 0x22101102ULL, 0x22111002ULL, 0x22111100ULL, 0x22111101ULL, 0x22111200ULL,
    0x22121001ULL, 0x22121201ULL, 0x20101010ULL, 0x20101111ULL, 0x20101210ULL, 0x20101212ULL, 0x20111010ULL, 0x20111011ULL,
    0x20111110ULL, 0x20111111ULL, 0x20111112ULL, 0x20111211ULL, 0x20121011ULL, 0x20121111ULL, 0x20121211ULL, 0x20121212ULL,
    0x21101011ULL, 0x21101110ULL, 0x21101111ULL, 0x21101112ULL, 0x21101211ULL, 0x21111010ULL, 0x21111011ULL, 0x21111012ULL,
    0x21111110ULL, 0x21111111ULL, 0x21111112ULL, 0x21111210ULL, 0x21111211ULL, 0x21111212ULL, 0x21121011ULL, 0x21121110ULL,
    0x21121111ULL, 0x21121112ULL, 0x21121211ULL, 0x22101011ULL, 0x22101111ULL, 0x22101210ULL, 0x22111011ULL, 0x22111012ULL,
    0x22111110ULL, 0x22111111ULL, 0x22111112ULL, 0x22111211ULL, 0x22111212ULL, 0x22121010ULL, 0x22121012ULL, 0x22121111ULL,
    0x22121210ULL, 0x22121212ULL, 0x20101021ULL, 0x20101120ULL, 0x20111020ULL, 0x20111121ULL, 0x20111221ULL, 0x20121020ULL,
    0x20121122ULL, 0x20121221ULL, 0x21101121ULL, 0x21101220ULL, 0x21101221ULL, 0x21111021ULL, 0x21111022ULL, 0x21111121ULL,
    0x21111122ULL, 0x21111221ULL, 0x21121121ULL, 0x21121220ULL, 0x22101022ULL, 0x22101120ULL, 0x22101221ULL, 0x22101222ULL,
    0x22111022ULL, 0x22111120ULL, 0x22111121ULL, 0x22121120ULL, 0x22121122ULL, 0x22121221ULL, 0x20102101ULL, 0x20112102ULL,
    0x20112201ULL, 0x20122101ULL, 0x21102001ULL, 0x21102102ULL, 0x21112000ULL, 0x21112002ULL, 0x21112101ULL, 0x21112102ULL,
    0x21112202ULL, 0x21122100ULL, 0x21122101ULL, 0x22102101ULL, 0x22112001ULL, 0x22112102ULL, 0x22112201ULL, 0x22122101ULL,
    0x20102110ULL, 0x20102112ULL, 0x20102211ULL, 0x20112010ULL, 0x20112012ULL, 0x20112111ULL, 0x20112210ULL, 0x20112212ULL,
    0x20122010ULL, 0x20122011ULL, 0x20122110ULL, 0x20122112ULL, 0x21102010ULL, 0x21102012ULL, 0x21102111ULL, 0x21102210ULL,
    0x21102212ULL, 0x21112011ULL, 0x21112110ULL, 0x21112111ULL, 0x21112112ULL, 0x21112211ULL, 0x21122012ULL, 0x21122111ULL,
    0x21122112ULL, 0x21122212ULL, 0x22102011ULL, 0x22102110ULL, 0x22112010ULL, 0x22112012ULL, 0x22112111ULL, 0x22112212ULL,
    0x22122011ULL, 0x22122112ULL, 0x20102121ULL, 0x20112121ULL, 0x20122121ULL, 0x21102120ULL, 0x21102122ULL, 0x21102221ULL,
    0x21112020ULL, 0x21112121ULL, 0x21112220ULL, 0x21122021ULL, 0x22102121ULL, 0x22112021ULL, 0x22112120ULL, 0x22112121ULL,
    0x22112122ULL, 0x20200000ULL, 0x20200002ULL, 0x20200200ULL, 0x20200202ULL, 0x20210101ULL, 0x20220000ULL, 0x20220002ULL,
    0x20220200ULL, 0x20220202ULL, 0x21200101ULL, 0x21210001ULL, 0x21210100ULL, 0x21210102ULL, 0x21210201ULL, 0x22200000ULL,
    0x22200002ULL, 0x22200200ULL, 0x22200202ULL, 0x22210101ULL, 0x22220000ULL, 0x22220002ULL, 0x22220200ULL, 0x22220202ULL,
    0x20200111ULL, 0x20200211ULL, 0x20210011ULL, 0x20210110ULL, 0x20210112ULL, 0x20210211ULL, 0x20210212ULL, 0x21200112ULL,
    0x21200211ULL, 0x21210011ULL, 0x21210111ULL, 0x21210210ULL, 0x21210212ULL, 0x21220011ULL, 0x21220110ULL, 0x22200111ULL,
    0x22210010ULL, 0x22210012ULL, 0x22210112ULL, 0x22210211ULL, 0x20200022ULL, 0x20200220ULL, 0x20200222ULL, 0x20210020ULL,
    0x20210221ULL, 0x20220022ULL, 0x20220220ULL, 0x20220222ULL, 0x21200121ULL, 0x21210021ULL, 0x21210122ULL, 0x21210221ULL,
    0x21220121ULL, 0x22200020ULL, 0x22200022ULL, 0x22200220ULL, 0x22200222ULL, 0x22210121ULL, 0x22220020ULL, 0x22220022ULL,
    0x22220220ULL, 0x22220222ULL, 0x20211201ULL, 0x20221101ULL, 0x21201001ULL, 0x21201100ULL, 0x21211000ULL, 0x21211100ULL,
    0x21211101ULL, 0x21211200ULL, 0x21211202ULL, 0x21221001ULL, 0x21221101ULL, 0x21221102ULL, 0x21221200ULL, 0x21221201ULL,
    0x22201101ULL, 0x20201112ULL, 0x20201211ULL, 0x20211010ULL, 0x20211012ULL, 0x20211111ULL, 0x20211210ULL, 0x20221112ULL,
    0x20221211ULL, 0x21201012ULL, 0x21201111ULL, 0x21211011ULL, 0x21211110ULL, 0x21211111ULL, 0x21211112ULL, 0x21211211ULL,
    0x21221111ULL, 0x21221212ULL, 0x22201011ULL, 0x22201110ULL, 0x22201111ULL, 0x22201112ULL, 0x22201211ULL, 0x22211012ULL,
    0x22211111ULL, 0x22211210ULL, 0x20201121ULL, 0x20211021ULL, 0x20211122ULL, 0x20211222ULL, 0x20221021ULL, 0x20221121ULL,
    0x21201120ULL, 0x21201122ULL, 0x21201222ULL, 0x21211022ULL, 0x21211121ULL, 0x21211122ULL, 0x21211220ULL, 0x21221020ULL,
    0x21221022ULL, 0x22201122ULL, 0x22211020ULL, 0x22211121ULL, 0x22211122ULL, 0x22211221ULL, 0x22221021ULL, 0x22221120ULL,
    0x22221122ULL, 0x20202000ULL, 0x20202002ULL, 0x20202200ULL, 0x20202202ULL, 0x20222000ULL, 0x20222002ULL, 0x20222200ULL,
    0x20222202ULL, 0x21212001ULL, 0x21212100ULL, 0x21212102ULL, 0x21212201ULL, 0x22202000ULL, 0x22202002ULL, 0x22202200ULL,
    0x22202202ULL, 0x22212101ULL, 0x22222000ULL, 0x22222002ULL, 0x22222200ULL, 0x22222202ULL, 0x20202111ULL, 0x20212110ULL,
    0x20212211ULL, 0x20222011ULL, 0x20222111ULL, 0x21202011ULL, 0x21212010ULL, 0x21212111ULL, 0x21212212ULL, 0x21222011ULL,
    0x21222112ULL, 0x21222211ULL, 0x22212010ULL, 0x22212112ULL, 0x20202020ULL, 0x20202022ULL, 0x20202220ULL, 0x20202222ULL,
    0x20222020ULL, 0x20222022ULL, 0x20222220ULL, 0x20222222ULL, 0x21212021ULL, 0x21212120ULL, 0x21212122ULL, 0x22202020ULL,
    0x22202022ULL, 0x22202220ULL, 0x22202222ULL, 0x22212121ULL, 0x22222020ULL, 0x22222022ULL, 0x22222220ULL, 0x22222222ULL,
};

#define VDR_IQ2_XXS_Q8_1_MMVQ 1
#define VDR_IQ2_XS_Q8_1_MMVQ 1
#define VDR_IQ3_XXS_Q8_1_MMVQ 1
#define VDR_IQ4_XS_Q8_1_MMVQ 1

template<typename dst_t>
inline __device__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_iq2_xxs * x = (const block_iq2_xxs *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;

    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + 32 * ib + 8 * il;
    const uint16_t * q2 = x->qs + 4 * ib;
    const uint8_t * aux8 = (const uint8_t *)q2;
    const uint8_t * grid = (const uint8_t *)(iq2xxs_grid_cu + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = __half2float(x->d) * (0.5f + (float)(aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs_cu[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) {
        y[j] = (dst_t)(d * grid[j] * (signs & kmask_iq2xs_cu[j] ? -1.f : 1.f));
    }
}

template<typename dst_t>
inline __device__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_iq2_xs * x = (const block_iq2_xs *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;

    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + 32 * ib + 8 * il;
    const uint16_t * q2 = x->qs + 4 * ib;
    const uint8_t * grid = (const uint8_t *)(iq2xs_grid_cu + (q2[il] & 511));
    const float d = __half2float(x->d) * (0.5f + (float)((x->scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs_cu[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) {
        y[j] = convert_from_half<dst_t>(__float2half(d * grid[j] * (signs & kmask_iq2xs_cu[j] ? -1.f : 1.f)));
    }
}

template<typename dst_t>
inline __device__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_iq3_xxs * x = (const block_iq3_xxs *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;

    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + 32 * ib + 8 * il;
    const uint8_t * q3 = x->qs + 8 * ib;
    const uint16_t * gas = (const uint16_t *)(x->qs + QK_IQ3_XXS/4) + 2*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid_cu + q3[2*il+0]);
    const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid_cu + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = __half2float(x->d) * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs_cu[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = convert_from_half<dst_t>(__float2half(d * grid1[j] * (signs & kmask_iq2xs_cu[j+0] ? -1.f : 1.f)));
        y[j+4] = convert_from_half<dst_t>(__float2half(d * grid2[j] * (signs & kmask_iq2xs_cu[j+4] ? -1.f : 1.f)));
    }
}

template<typename dst_t>
inline __device__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_iq4_xs * x = (const block_iq4_xs *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;

    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + 32 * ib + 4 * il;
    const uint8_t * q4 = x->qs + 16 * ib + 4 * il;
    const float d = __half2float(x->d) * (float)((((x->scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x->scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = convert_from_half<dst_t>(__float2half(d * kvalues_iq4nl_cu[q4[j] & 0xf]));
        y[j+16] = convert_from_half<dst_t>(__float2half(d * kvalues_iq4nl_cu[q4[j] >> 4]));
    }
}

// Native dot products for the newer IQ layouts. These intentionally decode
// the selected 32-value sub-block directly against Q8_1 input, preserving the
// GGUF bytes and avoiding a load-time requantization pass.
#define VDR_IQ1_S_Q8_1_MMVQ 1
#define VDR_IQ2_S_Q8_1_MMVQ 1
#define VDR_IQ3_S_Q8_1_MMVQ 1
#define VDR_IQ1_M_Q8_1_MMVQ 1
#define VDR_IQ4_NL_Q8_1_MMVQ 1

// CUDA has no direct 4-bit table lookup.  Four __byte_perm calls turn the
// eight nibbles in four IQ4_NL bytes into two packed signed-byte words.  This
// is the same lookup strategy used by llama.cpp's native IQ MMQ kernels and
// is substantially cheaper than a loop containing eight table reads.
static __device__ __forceinline__ int2 iq4_nl_table_lookup(const int q4) {
    const uint32_t* table = reinterpret_cast<const uint32_t *>(kvalues_iq4nl_cu);
    const uint32_t selection = 0x32103210 | ((q4 & 0x88888888) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table[0], table[1], q4 >> shift);
        const uint32_t high = __byte_perm(table[2], table[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, selection >> shift);
    }
    return make_int2(
        __byte_perm(tmp[0], tmp[1], 0x6420),
        __byte_perm(tmp[0], tmp[1], 0x7531));
}

static __device__ __forceinline__ float vec_dot_iq4_nl_full(
    const block_iq4_nl* __restrict__ bq,
    const block_q8_1* __restrict__ bq8_1) {
    int sum = 0;
#pragma unroll
    for (int iqs = 0; iqs < QI_IQ4_NL; ++iqs) {
        const int2 packed = iq4_nl_table_lookup(get_int_from_uint8(bq->qs, iqs));
        const int8_t* q8 = bq8_1->qs;
        sum += ggml_cuda_dp4a(packed.x, *reinterpret_cast<const int *>(q8 + 4 * iqs), 0);
        sum += ggml_cuda_dp4a(packed.y, *reinterpret_cast<const int *>(q8 + 16 + 4 * iqs), 0);
    }
    return (float)sum * __half2float(bq->d) * __low2float(bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& iqs) {
    const block_iq2_s* bq = (const block_iq2_s*)vbq;
    const int8_t* q8 = bq8_1[iqs].qs;
    float sum = 0.0f;
    for (int l = 0; l < 4; ++l) {
        const uint64_t grid_value = iq2s_grid_cu[
            bq->qs[4 * iqs + l] | ((bq->qh[iqs] << (8 - 2 * l)) & 0x300)];
        const uint8_t* grid = (const uint8_t*)&grid_value;
        const uint8_t signs = bq->qs[QK_IQ2_S / 8 + 4 * iqs + l];
        const float d = __half2float(bq->d) *
            (0.5f + (float)((bq->scales[iqs] >> (4 * (l / 2))) & 0xf)) * 0.25f;
        for (int j = 0; j < 8; ++j) {
            const float sign = signs & kmask_iq2xs_cu[j] ? -1.0f : 1.0f;
            sum += d * (float)grid[j] * sign * (float)q8[8 * l + j];
        }
    }
    return sum * __low2float(bq8_1[iqs].ds);
}

static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& iqs) {
    const block_iq3_s* bq = (const block_iq3_s*)vbq;
    const int8_t* q8 = bq8_1[iqs].qs;
    const float d = __half2float(bq->d) *
        (0.5f + (float)((bq->scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf)) * 0.5f;
    float sum = 0.0f;
    const uint8_t* qs = bq->qs + 8 * iqs;
    for (int l = 0; l < 4; ++l) {
        const uint32_t grid1_value = iq3xs_grid_cu[
            qs[2 * l] | ((bq->qh[iqs] << (8 - 2 * l)) & 256)];
        const uint32_t grid2_value = iq3xs_grid_cu[
            qs[2 * l + 1] | ((bq->qh[iqs] << (7 - 2 * l)) & 256)];
        const uint8_t* grid1 = (const uint8_t*)&grid1_value;
        const uint8_t* grid2 = (const uint8_t*)&grid2_value;
        const uint8_t signs = bq->signs[4 * iqs + l];
        for (int j = 0; j < 4; ++j) {
            const float s0 = signs & kmask_iq2xs_cu[j] ? -1.0f : 1.0f;
            const float s1 = signs & kmask_iq2xs_cu[j + 4] ? -1.0f : 1.0f;
            sum += d * (float)grid1[j] * s0 * (float)q8[8 * l + j];
            sum += d * (float)grid2[j] * s1 * (float)q8[8 * l + 4 + j];
        }
    }
    return sum * __low2float(bq8_1[iqs].ds);
}

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& iqs) {
    const block_iq1_s* bq = (const block_iq1_s*)vbq;
    const int8_t* q8 = bq8_1[iqs].qs;
    const uint16_t qh = bq->qh[iqs];
    const float d = __half2float(bq->d) * (2.0f * ((qh >> 12) & 7) + 1.0f);
    const float delta = qh & 0x8000 ? -1.125f : -0.875f;
    float sum = 0.0f;
    for (int l = 0; l < 4; ++l) {
        const uint32_t grid = (uint32_t)iq1s_grid_cu[
            bq->qs[4 * iqs + l] | (((qh >> (3 * l)) & 7) << 8)];
        const uint8_t* g = (const uint8_t*)&grid;
        for (int j = 0; j < 4; ++j) {
            sum += ((float)(g[j] & 0xf) + delta) * (float)q8[8 * l + j];
            sum += ((float)(g[j] >> 4) + delta) * (float)q8[8 * l + 4 + j];
        }
    }
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d * (ds.x * sum + ds.y * delta);
}

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& iqs) {
    const block_iq1_m* bq = (const block_iq1_m*)vbq;
    const int8_t* q8 = bq8_1[iqs].qs;
    const uint16_t* sc = (const uint16_t*)bq->scales;
    const uint16_t scale_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
        ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float base_d = __half2float(*((const half*)&scale_bits));
    float sum = 0.0f;
    for (int l = 0; l < 4; ++l) {
        const int ib16 = 2 * iqs + l / 2;
        const float scale =
            (2.0f * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 7) + 1.0f);
        const uint8_t qh = bq->qh[2 * iqs + l / 2];
        const float delta = qh & (0x08 << (4 * (l % 2))) ? -1.125f : -0.875f;
        const uint32_t grid = (uint32_t)iq1s_grid_cu[
            bq->qs[4 * iqs + l] | (((qh >> (4 * (l % 2))) & 7) << 8)];
        const uint8_t* g = (const uint8_t*)&grid;
        for (int j = 0; j < 4; ++j) {
            sum += scale * ((float)(g[j] & 0xf) + delta) * (float)q8[8 * l + j];
            sum += scale * ((float)(g[j] >> 4) + delta) * (float)q8[8 * l + 4 + j];
        }
    }
    return sum * base_d * __low2float(bq8_1[iqs].ds);
}

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& iqs) {
    const block_iq4_nl* bq = (const block_iq4_nl*)vbq;
    const int8_t* q8 = bq8_1->qs;
    const int2 packed = iq4_nl_table_lookup(get_int_from_uint8(bq->qs, iqs));
    const int sum = ggml_cuda_dp4a(
        packed.x, *reinterpret_cast<const int *>(q8 + 4 * iqs), 0) +
        ggml_cuda_dp4a(
            packed.y, *reinterpret_cast<const int *>(q8 + 16 + 4 * iqs), 0);
    return (float)sum * __half2float(bq->d) * __low2float(bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_iq2_xxs * bq = (const block_iq2_xxs *) vbq;
    const float d = __half2float(bq->d);

    const uint16_t * q2 = bq->qs + 4 * iqs;
    const uint32_t aux32_0 = q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux32_1 = q2[2] | ((uint32_t)q2[3] << 16);

    const float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;

    float sum = 0.0f;
    const float d8 = __low2float(bq8_1[iqs].ds);
    const int8_t * q8 = bq8_1[iqs].qs;

    for (int l = 0; l < 4; l++) {
        const uint8_t grid_idx = (aux32_0 >> (8*l)) & 0xFF;
        const uint64_t grid = iq2xxs_grid_cu[grid_idx];
        const uint8_t * gv = (const uint8_t *)&grid;
        const uint8_t signs = ksigns_iq2xs_cu[(aux32_1 >> (7*l)) & 0x7F];

        for (int j = 0; j < 8; j++) {
            float val = db * (float)gv[j];
            if (signs & kmask_iq2xs_cu[j]) val = -val;
            sum += val * (float)q8[8*l + j] * d8;
        }
    }

    return sum;
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_iq2_xs * bq = (const block_iq2_xs *) vbq;
    const float d = __half2float(bq->d);

    const int ls0 = (bq->scales[iqs] >> 0) & 0x0F;
    const int ls1 = (bq->scales[iqs] >> 4) & 0x0F;
    const float db0 = d * (0.5f + (float)ls0) * 0.25f;
    const float db1 = d * (0.5f + (float)ls1) * 0.25f;

    float sum = 0.0f;
    const float d8 = __low2float(bq8_1[iqs].ds);
    const int8_t * q8 = bq8_1[iqs].qs;

    for (int l = 0; l < 4; l++) {
        const float db = (l < 2) ? db0 : db1;
        const uint16_t q_val = bq->qs[4*iqs + l];
        const uint16_t grid_idx = q_val & 0x01FF;
        const uint8_t signs = ksigns_iq2xs_cu[(q_val >> 9) & 0x7F];
        const uint64_t grid = iq2xs_grid_cu[grid_idx];
        const uint8_t * gv = (const uint8_t *)&grid;

        for (int j = 0; j < 8; j++) {
            float val = db * (float)gv[j];
            if (signs & kmask_iq2xs_cu[j]) val = -val;
            sum += val * (float)q8[8*l + j] * d8;
        }
    }

    return sum;
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_iq3_xxs * bq = (const block_iq3_xxs *) vbq;
    const float d = __half2float(bq->d);

    const uint8_t * q3 = bq->qs + 8 * iqs;
    const uint8_t * gas = bq->qs + QK_IQ3_XXS / 4 + 4 * iqs;
    const uint32_t aux32 = gas[0] | (gas[1] << 8) | (gas[2] << 16) | (gas[3] << 24);
    const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

    float sum = 0.0f;
    const float d8 = __low2float(bq8_1[iqs].ds);
    const int8_t * q8 = bq8_1[iqs].qs;

    for (int il = 0; il < 4; il++) {
        const uint32_t grid1 = iq3xxs_grid_cu[q3[2*il]];
        const uint32_t grid2 = iq3xxs_grid_cu[q3[2*il + 1]];
        const uint8_t * g1 = (const uint8_t *)&grid1;
        const uint8_t * g2 = (const uint8_t *)&grid2;
        const uint8_t signs = ksigns_iq2xs_cu[(aux32 >> (7*il)) & 0x7F];

        for (int j = 0; j < 4; j++) {
            float v1 = db * (float)g1[j];
            if (signs & kmask_iq2xs_cu[j]) v1 = -v1;
            sum += v1 * (float)q8[8*il + j] * d8;
        }
        for (int j = 0; j < 4; j++) {
            float v2 = db * (float)g2[j];
            if (signs & kmask_iq2xs_cu[4+j]) v2 = -v2;
            sum += v2 * (float)q8[8*il + 4 + j] * d8;
        }
    }

    return sum;
}

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_iq4_xs * bq = (const block_iq4_xs *) vbq;
    const float d = __half2float(bq->d);

    const int ls = ((bq->scales_l[iqs/2] >> (4*(iqs%2))) & 0xF) | (((bq->scales_h >> (2*iqs)) & 3) << 4);
    const float db = d * (float)(ls - 32);

    float sum = 0.0f;
    const float d8 = __low2float(bq8_1[iqs].ds);
    const int8_t * q8 = bq8_1[iqs].qs;
    const uint8_t * qs = bq->qs + 16 * iqs;

    for (int j = 0; j < 32; j++) {
        uint8_t nibble = (j < 16) ? (qs[j] & 0xF) : (qs[j - 16] >> 4);
        float val = db * (float)kvalues_iq4nl_cu[nibble];
        sum += val * (float)q8[j] * d8;
    }

    return sum;
}
