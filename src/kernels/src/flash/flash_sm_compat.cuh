/**
 * @brief SM compatibility layer for native flash attention kernels.
 *
 * Provides macros to abstract SM-specific features so kernels can compile
 * and run on SM75 (Turing) with FP16 fallback, in addition to SM80+ (Ampere+)
 * with BF16.
 *
 * SM80+ path (NO_BF16_KERNEL not defined):
 *   - __nv_bfloat16 (BF16) as the half-precision type
 *   - mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
 *   - cp.async.cg.shared.global for async global→shared copy
 *   - cp.async.commit_group / cp.async.wait_group for pipeline control
 *
 * SM75 path (NO_BF16_KERNEL defined, __CUDA_ARCH__ >= 750):
 *   - __half (FP16) as the half-precision type
 *   - mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 (2× iterations)
 *   - Direct LDG+STS for global→shared copy (no cp.async)
 *   - __syncthreads() replaces async pipeline barriers
 *
 * SM70 path (NO_BF16_KERNEL defined, __CUDA_ARCH__ < 750):
 *   - __half (FP16) as the half-precision type
 *   - Scalar FMA with warp-shuffle A+B redistribution to emulate m16n8k16
 *     (iterates over all 4 group members for full 16-element K-reduction)
 *   - Direct LDG+STS for global→shared copy (same as SM75)
 *   - __syncthreads() replaces async pipeline barriers
 */

#pragma once
#include <cstdint> 
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef NO_BF16_KERNEL
// ============================================================================
// SM70/SM75: FP16 path
// ============================================================================

using flash_half_t = __half;

#define FLASH_FLOAT2HALF(x)     __float2half(x)
#define FLASH_HALF_AS_USHORT(x) __half_as_ushort(x)
#define FLASH_HALF2FLOAT(x)     __half2float(x)

// No cp.async; use direct 16-byte global→shared copy via LDG128 + STS128
#define FLASH_CP_ASYNC(sa, gm_ptr) \
    do { \
        uint4 _tmp = *reinterpret_cast<const uint4*>(gm_ptr); \
        *reinterpret_cast<uint4*>(__cvta_shared_to_generic( \
            static_cast<size_t>(sa))) = _tmp; \
    } while(0)

#define FLASH_ASYNC_COMMIT()
#define FLASH_ASYNC_WAIT()      __syncthreads()

// MMA dispatch: m16n8k8.f32.f16.f16.f32 on SM75+, scalar fallback on SM70.
// The m16n8k8 uses 2 A-regs (vs 4 for k16) and 1 B-reg (vs 2 for k16).
// To match the k16 iteration, each MMA_K16 call performs TWO m16n8k8 ops.
//
// Register layout for m16n8k8.f32.f16.f16.f32:
//   A: 2 u32 regs = 4 f16 values (rows determined by thread, k=8 cols)
//   B: 1 u32 reg  = 2 f16 values (cols determined by thread, k=8 rows)
//   C/D: 4 f32 regs (same mapping as m16n8k16)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)

#define FLASH_MMA_K16(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3) \
    do { \
        float _t0=(c0), _t1=(c1), _t2=(c2), _t3=(c3); \
        asm volatile( \
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 " \
            "{%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};" \
            : "=f"(_t0), "=f"(_t1), "=f"(_t2), "=f"(_t3) \
            : "r"(a0), "r"(a1), "r"(b0), \
              "f"(_t0), "f"(_t1), "f"(_t2), "f"(_t3)); \
        asm volatile( \
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 " \
            "{%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};" \
            : "=f"(_t0), "=f"(_t1), "=f"(_t2), "=f"(_t3) \
            : "r"(a2), "r"(a3), "r"(b1), \
              "f"(_t0), "f"(_t1), "f"(_t2), "f"(_t3)); \
        (d0) = _t0; (d1) = _t1; (d2) = _t2; (d3) = _t3; \
    } while(0)

#else
// SM70 (Volta): scalar FMA emulation of m16n8k16 MMA.
//
// Thread layout: group_id = lane_id/4, tid_in_group = lane_id & 3
// A regs: a0/a2 = Q[group_id, k_subset], a1/a3 = Q[group_id+8, k_subset]
//   Each thread holds 4 of 16 K-elements (determined by tid_in_group).
// B regs: b0/b1 = K[group_id, k_subset] (same K-positions as A).
// D regs: d0 = S[group_id, tid_in_group*2],
//         d1 = S[group_id, tid_in_group*2+1],
//         d2 = S[group_id+8, tid_in_group*2],
//         d3 = S[group_id+8, tid_in_group*2+1]
//
// The hardware MMA internally reduces across all 4 threads in a group to
// accumulate all 16 K-elements per output. We replicate this by iterating
// over all 4 group members (s=0..3), shuffling A from each to pair with
// the correctly-sourced B values for each output column.
//
// For output col c0 = tid_in_group*2: B comes from thread with
//   group_id = c0, i.e. lane = c0*4 + s (for each member s).
// For output col c1 = tid_in_group*2+1: lane = c1*4 + s.
#define FLASH_MMA_K16(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3) \
    do { \
        const unsigned int _lane = threadIdx.x & 31; \
        const unsigned int _gid  = _lane >> 2; \
        const unsigned int _tig  = _lane & 3; \
        float _r0=(c0), _r1=(c1), _r2=(c2), _r3=(c3); \
        for (int _s = 0; _s < 4; _s++) { \
            unsigned int _src_a = _gid * 4 + _s; \
            unsigned int _sa0 = __shfl_sync(0xFFFFFFFF, (a0), _src_a); \
            unsigned int _sa1 = __shfl_sync(0xFFFFFFFF, (a1), _src_a); \
            unsigned int _sa2 = __shfl_sync(0xFFFFFFFF, (a2), _src_a); \
            unsigned int _sa3 = __shfl_sync(0xFFFFFFFF, (a3), _src_a); \
            unsigned int _src_b0 = _tig * 2 * 4 + _s; \
            unsigned int _src_b1 = (_tig * 2 + 1) * 4 + _s; \
            unsigned int _sb0_e = __shfl_sync(0xFFFFFFFF, (b0), _src_b0); \
            unsigned int _sb1_e = __shfl_sync(0xFFFFFFFF, (b1), _src_b0); \
            unsigned int _sb0_o = __shfl_sync(0xFFFFFFFF, (b0), _src_b1); \
            unsigned int _sb1_o = __shfl_sync(0xFFFFFFFF, (b1), _src_b1); \
            const __half* _ap0 = reinterpret_cast<const __half*>(&_sa0); \
            const __half* _ap1 = reinterpret_cast<const __half*>(&_sa1); \
            const __half* _ap2 = reinterpret_cast<const __half*>(&_sa2); \
            const __half* _ap3 = reinterpret_cast<const __half*>(&_sa3); \
            const __half* _be0 = reinterpret_cast<const __half*>(&_sb0_e); \
            const __half* _bf0 = reinterpret_cast<const __half*>(&_sb1_e); \
            const __half* _be1 = reinterpret_cast<const __half*>(&_sb0_o); \
            const __half* _bf1 = reinterpret_cast<const __half*>(&_sb1_o); \
            for (int _k = 0; _k < 2; _k++) { \
                float _av0 = __half2float(_ap0[_k]); \
                float _av1 = __half2float(_ap1[_k]); \
                float _bv_e = __half2float(_be0[_k]); \
                float _bv_o = __half2float(_be1[_k]); \
                _r0 = __fmaf_rn(_av0, _bv_e, _r0); \
                _r1 = __fmaf_rn(_av0, _bv_o, _r1); \
                _r2 = __fmaf_rn(_av1, _bv_e, _r2); \
                _r3 = __fmaf_rn(_av1, _bv_o, _r3); \
            } \
            for (int _k = 0; _k < 2; _k++) { \
                float _av0 = __half2float(_ap2[_k]); \
                float _av1 = __half2float(_ap3[_k]); \
                float _bv_e = __half2float(_bf0[_k]); \
                float _bv_o = __half2float(_bf1[_k]); \
                _r0 = __fmaf_rn(_av0, _bv_e, _r0); \
                _r1 = __fmaf_rn(_av0, _bv_o, _r1); \
                _r2 = __fmaf_rn(_av1, _bv_e, _r2); \
                _r3 = __fmaf_rn(_av1, _bv_o, _r3); \
            } \
        } \
        (d0) = _r0; (d1) = _r1; (d2) = _r2; (d3) = _r3; \
    } while(0)

#endif // __CUDA_ARCH__ >= 750

#else
// ============================================================================
// SM80+: BF16 path (Ampere, Ada, Hopper, Blackwell)
// ============================================================================
#include <cuda_bf16.h>

using flash_half_t = __nv_bfloat16;

#define FLASH_FLOAT2HALF(x)     __float2bfloat16(x)
#define FLASH_HALF_AS_USHORT(x) __bfloat16_as_ushort(x)
#define FLASH_HALF2FLOAT(x)     __bfloat162float(x)

#define FLASH_CP_ASYNC(sa, gm_ptr) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" \
                 :: "r"(sa), "l"(gm_ptr))

#define FLASH_ASYNC_COMMIT() asm volatile("cp.async.commit_group;")
#define FLASH_ASYNC_WAIT()   asm volatile("cp.async.wait_group 0;")

#define FLASH_MMA_K16(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3) \
    asm volatile( \
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
          "r"(b0), "r"(b1), \
          "f"(c0), "f"(c1), "f"(c2), "f"(c3))

#endif // NO_BF16_KERNEL
