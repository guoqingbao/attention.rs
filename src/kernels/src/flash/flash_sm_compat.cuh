/**
 * @brief SM compatibility + dual-dtype traits for native flash / TurboQuant.
 *
 * Arch (independent of dtype):
 *   SM70/75 (NO_BF16_KERNEL): LDG+STS copy, F16 MMA only (m8n8k4 / m16n8k8)
 *   SM80+: cp.async pipeline; F16 and BF16 m16n8k16 MMA
 *
 * Dtype: FlashHalfTraits<__half> always; FlashHalfTraits<__nv_bfloat16>
 * only when !NO_BF16_KERNEL (SM80+).
 */

#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifndef NO_BF16_KERNEL
#include <cuda_bf16.h>
#endif

// ============================================================================
// Arch: async copy
// ============================================================================
#ifdef NO_BF16_KERNEL
#define FLASH_CP_ASYNC(sa, gm_ptr) \
    do { \
        uint4 _tmp = *reinterpret_cast<const uint4*>(gm_ptr); \
        *reinterpret_cast<uint4*>(__cvta_shared_to_generic( \
            static_cast<size_t>(sa))) = _tmp; \
    } while(0)
#define FLASH_ASYNC_COMMIT()
#define FLASH_ASYNC_WAIT()      __syncthreads()
#else
#define FLASH_CP_ASYNC(sa, gm_ptr) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" \
                 :: "r"(sa), "l"(gm_ptr))
#define FLASH_ASYNC_COMMIT() asm volatile("cp.async.commit_group;")
#define FLASH_ASYNC_WAIT()   asm volatile("cp.async.wait_group 0;")
#endif

// ============================================================================
// Arch: F16 MMA (SM70 / SM75 / SM80+)
// ============================================================================
// NOTE: Do not gate declarations on __CUDA_ARCH__. That macro exists only during
// the device compilation pass; host pass would omit the helper while
// flash_mma_k16_f16 still references it. NO_BF16_KERNEL is the build-time
// (SM70/75) switch; __CUDA_ARCH__ selects the ISA inside the body.
#ifdef NO_BF16_KERNEL
__device__ __forceinline__ void flash_mma_k16_sm70(
    float &d0, float &d1, float &d2, float &d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
    const unsigned int lane  = threadIdx.x & 31;
    const unsigned int gid   = lane >> 2;
    const unsigned int tig   = lane & 3;
    const unsigned int is_hi = lane >> 4;
    const unsigned int qp    = (lane >> 2) & 3;
    const unsigned int lt    = lane & 3;
    const unsigned int my_row = is_hi ? (lt + 4) : lt;
    float d_lo[8] = {0,0,0,0,0,0,0,0};
    float d_hi[8] = {0,0,0,0,0,0,0,0};

    #pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        const unsigned int t01 = (ks & 1) * 2;
        const unsigned int t23 = (ks & 1) * 2 + 1;
        const unsigned int src01 = my_row * 4 + t01;
        const unsigned int src23 = my_row * 4 + t23;
        const unsigned int a_lo_reg = (ks < 2) ? a0 : a2;
        const unsigned int a_hi_reg = (ks < 2) ? a1 : a3;
        unsigned int mma_a0_lo = __shfl_sync(0xFFFFFFFF, a_lo_reg, src01);
        unsigned int mma_a1_lo = __shfl_sync(0xFFFFFFFF, a_lo_reg, src23);
        unsigned int mma_a0_hi = __shfl_sync(0xFFFFFFFF, a_hi_reg, src01);
        unsigned int mma_a1_hi = __shfl_sync(0xFFFFFFFF, a_hi_reg, src23);
        const unsigned int b_col = is_hi ? (lt + 4) : lt;
        const unsigned int b_reg = (ks < 2) ? b0 : b1;
        const unsigned int bt01 = b_col * 4 + t01;
        const unsigned int bt23 = b_col * 4 + t23;
        unsigned int mma_b0 = __shfl_sync(0xFFFFFFFF, b_reg, bt01);
        unsigned int mma_b1 = __shfl_sync(0xFFFFFFFF, b_reg, bt23);
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},"
            "{%12,%13,%14,%15,%16,%17,%18,%19};"
            : "=f"(d_lo[0]), "=f"(d_lo[1]), "=f"(d_lo[2]), "=f"(d_lo[3]),
              "=f"(d_lo[4]), "=f"(d_lo[5]), "=f"(d_lo[6]), "=f"(d_lo[7])
            : "r"(mma_a0_lo), "r"(mma_a1_lo), "r"(mma_b0), "r"(mma_b1),
              "f"(d_lo[0]), "f"(d_lo[1]), "f"(d_lo[2]), "f"(d_lo[3]),
              "f"(d_lo[4]), "f"(d_lo[5]), "f"(d_lo[6]), "f"(d_lo[7]));
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},"
            "{%12,%13,%14,%15,%16,%17,%18,%19};"
            : "=f"(d_hi[0]), "=f"(d_hi[1]), "=f"(d_hi[2]), "=f"(d_hi[3]),
              "=f"(d_hi[4]), "=f"(d_hi[5]), "=f"(d_hi[6]), "=f"(d_hi[7])
            : "r"(mma_a0_hi), "r"(mma_a1_hi), "r"(mma_b0), "r"(mma_b1),
              "f"(d_hi[0]), "f"(d_hi[1]), "f"(d_hi[2]), "f"(d_hi[3]),
              "f"(d_hi[4]), "f"(d_hi[5]), "f"(d_hi[6]), "f"(d_hi[7]));
    }

    const unsigned int r0 = gid;
    const unsigned int c0_t = tig * 2;
    const unsigned int r0_adj = r0 & 3;
    const unsigned int r0_hi = r0 >> 2;
    const unsigned int lb0_0 = r0_adj & 1;
    const unsigned int ib_2  = r0_adj & 2;
    const unsigned int ib4_0 = (c0_t >= 4) ? 4u : 0u;
    const unsigned int c0_a  = (c0_t >= 4) ? (c0_t - 4) : c0_t;
    const unsigned int lb2_0 = c0_a & ~1u;
    const unsigned int ib1_0 = c0_a & 1;
    const unsigned int si_0  = ib4_0 | ib_2 | ib1_0;
    const unsigned int sl_0  = qp * 4 + (r0_hi ? 16u : 0u) + lb2_0 + lb0_0;
    const unsigned int c1_t = tig * 2 + 1;
    const unsigned int ib4_1 = (c1_t >= 4) ? 4u : 0u;
    const unsigned int c1_a  = (c1_t >= 4) ? (c1_t - 4) : c1_t;
    const unsigned int lb2_1 = c1_a & ~1u;
    const unsigned int ib1_1 = c1_a & 1;
    const unsigned int si_1  = ib4_1 | ib_2 | ib1_1;
    const unsigned int sl_1  = qp * 4 + (r0_hi ? 16u : 0u) + lb2_1 + lb0_0;

    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    #pragma unroll
    for (unsigned int j = 0; j < 8; j++) {
        float bl = __shfl_sync(0xFFFFFFFF, d_lo[j], sl_0);
        if (j == si_0) v0 = bl;
        float bl1 = __shfl_sync(0xFFFFFFFF, d_lo[j], sl_1);
        if (j == si_1) v1 = bl1;
        float bh = __shfl_sync(0xFFFFFFFF, d_hi[j], sl_0);
        if (j == si_0) v2 = bh;
        float bh1 = __shfl_sync(0xFFFFFFFF, d_hi[j], sl_1);
        if (j == si_1) v3 = bh1;
    }
    d0 = v0 + c0; d1 = v1 + c1; d2 = v2 + c2; d3 = v3 + c3;
#else
    // Host pass (no __CUDA_ARCH__) or SM75+ device: not taken at runtime on SM75
    // (flash_mma_k16_f16 uses m16n8k8). Stub keeps the symbol visible to host parse.
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}
#endif // NO_BF16_KERNEL — SM70 MMA helper

__device__ __forceinline__ void flash_mma_k16_f16(
    float &d0, float &d1, float &d2, float &d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3
) {
#ifdef NO_BF16_KERNEL
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    float _t0=(c0), _t1=(c1), _t2=(c2), _t3=(c3);
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};"
        : "=f"(_t0), "=f"(_t1), "=f"(_t2), "=f"(_t3)
        : "r"(a0), "r"(a1), "r"(b0),
          "f"(_t0), "f"(_t1), "f"(_t2), "f"(_t3));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};"
        : "=f"(_t0), "=f"(_t1), "=f"(_t2), "=f"(_t3)
        : "r"(a2), "r"(a3), "r"(b1),
          "f"(_t0), "f"(_t1), "f"(_t2), "f"(_t3));
    d0 = _t0; d1 = _t1; d2 = _t2; d3 = _t3;
#else
    flash_mma_k16_sm70(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
#endif
#else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
}

#ifndef NO_BF16_KERNEL
__device__ __forceinline__ void flash_mma_k16_bf16(
    float &d0, float &d1, float &d2, float &d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}
#endif

// ============================================================================
// Dtype traits
// ============================================================================
template <typename T>
struct FlashHalfTraits;

template <>
struct FlashHalfTraits<__half> {
    using type = __half;
    __device__ __forceinline__ static float to_float(__half x) { return __half2float(x); }
    __device__ __forceinline__ static __half from_float(float x) { return __float2half(x); }
    __device__ __forceinline__ static unsigned short as_ushort(__half x) {
        return __half_as_ushort(x);
    }
    __device__ __forceinline__ static void mma_k16(
        float &d0, float &d1, float &d2, float &d3,
        unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
        unsigned int b0, unsigned int b1,
        float c0, float c1, float c2, float c3
    ) {
        flash_mma_k16_f16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
    }
};

#ifndef NO_BF16_KERNEL
template <>
struct FlashHalfTraits<__nv_bfloat16> {
    using type = __nv_bfloat16;
    __device__ __forceinline__ static float to_float(__nv_bfloat16 x) {
        return __bfloat162float(x);
    }
    __device__ __forceinline__ static __nv_bfloat16 from_float(float x) {
        return __float2bfloat16(x);
    }
    __device__ __forceinline__ static unsigned short as_ushort(__nv_bfloat16 x) {
        return __bfloat16_as_ushort(x);
    }
    __device__ __forceinline__ static void mma_k16(
        float &d0, float &d1, float &d2, float &d3,
        unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
        unsigned int b0, unsigned int b1,
        float c0, float c1, float c2, float c3
    ) {
        flash_mma_k16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
    }
};
#endif

// Convenience macros used inside templated kernels (HalfT in scope).
#define FLASH_TO_FLOAT(x)       (FlashHalfTraits<HalfT>::to_float(x))
#define FLASH_FROM_FLOAT(x)     (FlashHalfTraits<HalfT>::from_float(x))
#define FLASH_AS_USHORT(x)      (FlashHalfTraits<HalfT>::as_ushort(x))
#define FLASH_MMA_K16(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3) \
    FlashHalfTraits<HalfT>::mma_k16(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3)
