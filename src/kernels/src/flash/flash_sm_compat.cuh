/**
 * @brief SM compatibility layer for native flash attention kernels.
 *
 * Provides macros to abstract SM-specific features so kernels can compile
 * and run across SM70 (Volta), SM75 (Turing), and SM80+ (Ampere/Hopper/Blackwell)
 * using the best available Tensor Core MMA instructions for each architecture.
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
 *   - mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 Tensor Core MMA
 *     with warp-shuffle redistribution to emulate m16n8k16 layout
 *     (8 m8n8k4 ops: 2 M-halves × 4 K-slices, then output remap)
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

// MMA dispatch: m16n8k8 on SM75+, m8n8k4 TC MMA on SM70.
// SM75: m16n8k8 uses 2 A-regs (vs 4 for k16) and 1 B-reg (vs 2 for k16).
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
// SM70 (Volta): m8n8k4 Tensor Core implementation of m16n8k16 MMA.
//
// Uses Volta's native mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32
// to replace scalar FMA emulation. Input A/B registers arrive in m16n8k16
// fragment layout; we redistribute via warp shuffles to m8n8k4 quadpair
// layout, issue 8 m8n8k4 ops (2 M-halves × 4 K-slices), then remap
// outputs back to m16n8k16 D layout.
//
// m16n8k16 caller layout (groupID = lane/4, tid = lane & 3):
//   A: a0..a3 (4 u32 = 8 f16), rows {gid, gid+8}, K cols via tid
//   B: b0..b1 (2 u32 = 4 f16), K rows via tid, col = gid
//   D: d0..d3 (4 f32), rows {gid, gid+8}, cols {tid*2, tid*2+1}
//
// m8n8k4 hardware layout (quadpair: 4 low + 4 high lanes):
//   A: 2 u32 (row-major), row = lane%4 [+4 if hi], K = i (0..3)
//   B: 2 u32 (col-major), row = i (0..3), col = lane%4 [+4 if hi]
//   D: 8 f32, row = (lane&1)+(i&2) [+4 hi], col = (i&4)+(lane&2)+(i&1)

__device__ __forceinline__ void flash_mma_k16_sm70(
    float &d0, float &d1, float &d2, float &d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3
) {
    const unsigned int lane  = threadIdx.x & 31;
    const unsigned int gid   = lane >> 2;
    const unsigned int tig   = lane & 3;
    const unsigned int is_hi = lane >> 4;
    const unsigned int qp    = (lane >> 2) & 3;
    const unsigned int lt    = lane & 3;

    // Row this thread computes in m8n8k4 (0..7 within each M-half)
    const unsigned int my_row = is_hi ? (lt + 4) : lt;

    // m8n8k4 accumulators: zero-initialized, MMA accumulates into these
    float d_lo[8] = {0,0,0,0,0,0,0,0};
    float d_hi[8] = {0,0,0,0,0,0,0,0};

    #pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        // Source tid indices for K-slice ks
        // K cols [ks*4..ks*4+3]: tid_01 holds cols ks*4,ks*4+1; tid_23 holds ks*4+2,ks*4+3
        const unsigned int t01 = (ks & 1) * 2;       // 0 or 2
        const unsigned int t23 = (ks & 1) * 2 + 1;   // 1 or 3
        const unsigned int src01 = my_row * 4 + t01;
        const unsigned int src23 = my_row * 4 + t23;

        // Get A fragments: low-M uses a0/a2, high-M uses a1/a3
        const unsigned int a_lo_reg = (ks < 2) ? a0 : a2;
        const unsigned int a_hi_reg = (ks < 2) ? a1 : a3;

        unsigned int mma_a0_lo = __shfl_sync(0xFFFFFFFF, a_lo_reg, src01);
        unsigned int mma_a1_lo = __shfl_sync(0xFFFFFFFF, a_lo_reg, src23);
        unsigned int mma_a0_hi = __shfl_sync(0xFFFFFFFF, a_hi_reg, src01);
        unsigned int mma_a1_hi = __shfl_sync(0xFFFFFFFF, a_hi_reg, src23);

        // Get B fragment: col = lt (low half) or lt+4 (high half)
        const unsigned int b_col = is_hi ? (lt + 4) : lt;
        const unsigned int b_reg = (ks < 2) ? b0 : b1;
        const unsigned int bt01 = b_col * 4 + t01;
        const unsigned int bt23 = b_col * 4 + t23;
        unsigned int mma_b0 = __shfl_sync(0xFFFFFFFF, b_reg, bt01);
        unsigned int mma_b1 = __shfl_sync(0xFFFFFFFF, b_reg, bt23);

        // Issue m8n8k4 for low-M half (rows 0-7)
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7},"
            "{%8,%9},{%10,%11},"
            "{%12,%13,%14,%15,%16,%17,%18,%19};"
            : "=f"(d_lo[0]), "=f"(d_lo[1]), "=f"(d_lo[2]), "=f"(d_lo[3]),
              "=f"(d_lo[4]), "=f"(d_lo[5]), "=f"(d_lo[6]), "=f"(d_lo[7])
            : "r"(mma_a0_lo), "r"(mma_a1_lo), "r"(mma_b0), "r"(mma_b1),
              "f"(d_lo[0]), "f"(d_lo[1]), "f"(d_lo[2]), "f"(d_lo[3]),
              "f"(d_lo[4]), "f"(d_lo[5]), "f"(d_lo[6]), "f"(d_lo[7]));

        // Issue m8n8k4 for high-M half (rows 8-15)
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7},"
            "{%8,%9},{%10,%11},"
            "{%12,%13,%14,%15,%16,%17,%18,%19};"
            : "=f"(d_hi[0]), "=f"(d_hi[1]), "=f"(d_hi[2]), "=f"(d_hi[3]),
              "=f"(d_hi[4]), "=f"(d_hi[5]), "=f"(d_hi[6]), "=f"(d_hi[7])
            : "r"(mma_a0_hi), "r"(mma_a1_hi), "r"(mma_b0), "r"(mma_b1),
              "f"(d_hi[0]), "f"(d_hi[1]), "f"(d_hi[2]), "f"(d_hi[3]),
              "f"(d_hi[4]), "f"(d_hi[5]), "f"(d_hi[6]), "f"(d_hi[7]));
    }

    // Remap m8n8k4 D → m16n8k16 D layout via warp shuffles.
    //
    // m8n8k4 D[i] (f32) at thread with lt, is_hi, within quadpair qp:
    //   row = (lt&1) + (i&2) + (is_hi ? 4 : 0)
    //   col = (i&4) + (lt&2) + (i&1)
    //
    // m16n8k16 D layout:
    //   d0: (row=gid, col=tig*2)      from low-M  result
    //   d1: (row=gid, col=tig*2+1)    from low-M  result
    //   d2: (row=gid+8, col=tig*2)    from high-M result (= d2 in m16n8k16 maps to row gid = gid in high-M 8×8)
    //   d3: (row=gid+8, col=tig*2+1)  from high-M result
    //
    // To find which m8n8k4 thread holds (row=r, col=c):
    //   r_adj = r & 3;  is_hi_src = r >= 4
    //   src lane_bit0 = r_adj & 1;  src i_bit2 = r_adj & 2
    //   if c < 4: i_bit4=0, lane_bit2 = c & ~1, i_bit1 = c & 1
    //   else:     i_bit4=4, lane_bit2 = (c-4) & ~1, i_bit1 = (c-4) & 1
    //   src_i = i_bit4 | i_bit2 | i_bit1
    //   src_lane = qp*4 + (is_hi_src ? 16 : 0) + lane_bit2 + lane_bit0

    // Compute source (lane, reg_index) for d0 target: (gid, tig*2)
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

    // For d1 target: (gid, tig*2+1)
    const unsigned int c1_t = tig * 2 + 1;
    const unsigned int ib4_1 = (c1_t >= 4) ? 4u : 0u;
    const unsigned int c1_a  = (c1_t >= 4) ? (c1_t - 4) : c1_t;
    const unsigned int lb2_1 = c1_a & ~1u;
    const unsigned int ib1_1 = c1_a & 1;
    const unsigned int si_1  = ib4_1 | ib_2 | ib1_1;
    const unsigned int sl_1  = qp * 4 + (r0_hi ? 16u : 0u) + lb2_1 + lb0_0;

    // Extract values using 8-round shuffle: each round broadcasts d[j],
    // receiver captures when j matches its target register index.
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

    d0 = v0 + c0;
    d1 = v1 + c1;
    d2 = v2 + c2;
    d3 = v3 + c3;
}

#define FLASH_MMA_K16(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3) \
    flash_mma_k16_sm70(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3)

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
