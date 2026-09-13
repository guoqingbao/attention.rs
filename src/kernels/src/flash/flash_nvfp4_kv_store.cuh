/**
 * NVFP4 KV cache store: FP4 E2M1 + E4M3 group-16 scales.
 *
 * ProVEN SM120 format (hikarioyama/vllm-nvfp4-kv-sm120, FlashMLA V41_FP4):
 *   - 16-element groups, one E4M3 scale per group
 *   - FP4 E2M1 codes: 2 per byte (even in low nibble, odd in high)
 *   - Scale = clamp(amax / 6.0, 2^-9, 448.0) E4M3
 *   - 72 bytes per head per token (HD=128): 64 data + 8 scales
 *   - 1.78 vs cache
 *
 * K: optionally WHT-rotated before quantization (UltraQuant paper)
 * V: unrotated
 *
 * Cache layout (separate buffers, matching existing TQ4/FP8 pattern):
 *   K_fp4: [num_blocks, block_size, num_kv_heads, head_dim/2] U8
 *   K_sf:  [num_blocks, block_size, num_kv_heads, head_dim/16] U8 (E4M3)
 *   V_fp4: same as K_fp4
 *   V_sf:  same as K_sf
 */

#include "flash_sm_compat.cuh"
// wht_transform, get_sign_flip from flash_turboquant.cuh (included before this)

#ifndef FLASH_HDIM
#define FLASH_HDIM 128
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef HDIM
#define HDIM FLASH_HDIM
#endif

#define NVFP4_GROUP 16
#define NVFP4_GROUPS (HDIM / NVFP4_GROUP)
#define NVFP4_VEC (HDIM / WARP_SIZE)
#define NVFP4_LANES_PER_GROUP (NVFP4_GROUP / NVFP4_VEC)

#ifndef NVFP4_HELPERS_DEFINED
#define NVFP4_HELPERS_DEFINED

// FP4 E2M1 quantize: round to nearest of {0, 0.5, 1, 1.5, 2, 3, 4, 6}
// Matches PTX cvt.rn.satfinite.e2m1x2.f32 semantics.
__device__ __forceinline__ uint8_t nvfp4_float_to_e2m1(float val) {
    float abs_val = fabsf(val);
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;
    uint8_t code;
    if (abs_val < 0.25f)       code = 0x0;  // 0.0
    else if (abs_val < 0.75f) code = 0x1;  // 0.5
    else if (abs_val < 1.25f) code = 0x2;  // 1.0
    else if (abs_val < 1.75f) code = 0x3;  // 1.5
    else if (abs_val < 2.5f)  code = 0x4;  // 2.0
    else if (abs_val < 3.5f)  code = 0x5;  // 3.0
    else if (abs_val < 5.0f)  code = 0x6;  // 4.0
    else                      code = 0x7;  // 6.0 (sfinite)
    return sign | code;
}

// FP4 E2M1 dequant: code -> float
__device__ __forceinline__ float nvfp4_e2m1_to_float(uint8_t code) {
    static const float vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = vals[code & 0x7];
    return (code & 0x8) ? -v : v;
}

// FP8 E4M3 quantize: round to nearest E4M3 value
// E4M3: 1 sign + 4 exp + 3 mantissa, bias 7
// Range: [-448, 448], subnormals down to 2^-9
__device__ __forceinline__ uint8_t float_to_e4m3(float val) {
    if (val == 0.0f) return 0;
    uint8_t sign = (val < 0.0f) ? 0x80 : 0x00;
    float abs_val = fabsf(val);
    if (abs_val > 448.0f) return sign | 0x7E;  // saturate to max finite
    // Use __nv_cvt_float_to_fp8 if available, else manual
    int exp;
    float mant = frexpf(abs_val, &exp);  // abs_val = mant * 2^exp, mant in [0.5, 1)
    // E4M3: value = (1 + m/8) * 2^(e-7), where m is 3-bit mantissa
    // We need: abs_val = (1 + m/8) * 2^(e-7)
    // So: e = exp + 7 (approximately), m = round((abs_val / 2^(e-7) - 1) * 8)
    // Simpler: just use the CUDA intrinsic
    __nv_fp8_storage_t result = __nv_cvt_float_to_fp8(abs_val, __NV_SATFINITE, __NV_E4M3);
    return sign | (uint8_t)result;
}

// FP8 E4M3 dequant (direct bit manipulation, no CUDA intrinsic needed)
__device__ __forceinline__ float e4m3_to_float_direct(uint8_t code) {
    uint8_t sign = (code >> 7) & 1;
    uint8_t exp  = (code >> 3) & 0xF;
    uint8_t mant = code & 0x7;
    float val;
    if (exp == 0) {
        val = (float)mant * (1.0f / 8.0f) * 0.001953125f;  // subnormal: mant/8 * 2^-6
    } else {
        val = (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, (int)exp - 7);
    }
    return sign ? -val : val;
}

#endif // NVFP4_HELPERS_DEFINED

template<typename HalfT>
__global__ void flash_nvfp4_kv_store(
    const HalfT* __restrict__ K,
    const HalfT* __restrict__ V,
    unsigned char* K_fp4,
    unsigned char* K_sf,
    unsigned char* V_fp4,
    unsigned char* V_sf,
    const long long* __restrict__ slot_mapping,
    const unsigned int num_tokens,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int block_size,
    const float c_k,
    const float c_v,
    const bool rotate
) {
    const unsigned int token_idx = blockIdx.x;
    const unsigned int head_idx = blockIdx.y;
    const unsigned int lane_id = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;
    long long slot = slot_mapping[token_idx];
    if (slot < 0) return;

    unsigned int block_idx = (unsigned int)(slot / block_size);
    unsigned int block_off = (unsigned int)(slot % block_size);

    unsigned int base = token_idx * num_kv_heads * head_dim + head_idx * head_dim;

    // Paged offsets
    unsigned long long fp4_k_off = (unsigned long long)block_idx * block_size * num_kv_heads * (head_dim / 2)
        + (unsigned long long)block_off * num_kv_heads * (head_dim / 2)
        + (unsigned long long)head_idx * (head_dim / 2);
    unsigned long long sf_k_off = (unsigned long long)block_idx * block_size * num_kv_heads * NVFP4_GROUPS
        + (unsigned long long)block_off * num_kv_heads * NVFP4_GROUPS
        + (unsigned long long)head_idx * NVFP4_GROUPS;

    // K: load, optionally rotate
    float k_reg[NVFP4_VEC];
    #pragma unroll
    for (int i = 0; i < NVFP4_VEC; i++) {
        unsigned int ch = lane_id * NVFP4_VEC + i;
        k_reg[i] = FLASH_TO_FLOAT(K[base + ch]);
        if (rotate) {
            k_reg[i] *= get_sign_flip(head_idx, ch);
        }
    }
    if (rotate) {
        wht_transform(k_reg, lane_id);
    }

    // Per-group-16 quantization for K
    #pragma unroll
    for (int g = 0; g < NVFP4_GROUPS; g++) {
        unsigned int lane_start = g * NVFP4_LANES_PER_GROUP;
        if (lane_id < lane_start || lane_id >= lane_start + NVFP4_LANES_PER_GROUP) continue;

        int local = lane_id - lane_start;

        // Per-group absmax (each lane has NVFP4_VEC elements, group has NVFP4_LANES_PER_GROUP lanes)
        float local_max = 0.f;
        #pragma unroll
        for (int i = 0; i < NVFP4_VEC; i++)
            local_max = fmaxf(local_max, fabsf(k_reg[i]));

        // Reduce across the group (NVFP4_LANES_PER_GROUP lanes)
        #pragma unroll
        for (int off = NVFP4_LANES_PER_GROUP/2; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

        // E4M3 scale = c * amax / 6.0 (6 = max FP4 E2M1 magnitude)
        float scale_val = c_k * local_max / 6.0f;
        uint8_t sf_byte = float_to_e4m3(scale_val);

        if (local == 0) {
            K_sf[sf_k_off + g] = sf_byte;
        }
        // All lanes in the group hold the same sf_byte (reduced via shfl above);
        // use it directly instead of reading back from global (avoids a
        // write/read race that zeroed the SF for non-local==0 lanes).
        float scale = e4m3_to_float_direct(sf_byte);
        float inv_scale = (scale > 0.f) ? (1.0f / scale) : 0.f;

        // Quantize to FP4 E2M1. Lane lane_id holds elements [4*lane_id, 4*lane_id+3]
        // in k_reg[0..3]; they map to bytes [2*lane_id, 2*lane_id+1] (2 codes/byte).
        #pragma unroll
        for (int i = 0; i < NVFP4_VEC; i += 2) {
            uint8_t lo = nvfp4_float_to_e2m1(k_reg[i] * inv_scale);
            uint8_t hi = nvfp4_float_to_e2m1(k_reg[i + 1] * inv_scale);
            K_fp4[fp4_k_off + 2 * lane_id + i / 2] = (hi << 4) | (lo & 0xF);
        }
    }

    // V: load rotation
    float v_reg[NVFP4_VEC];
    #pragma unroll
    for (int i = 0; i < NVFP4_VEC; i++) {
        unsigned int ch = lane_id * NVFP4_VEC + i;
        v_reg[i] = FLASH_TO_FLOAT(V[base + ch]);
    }

    unsigned long long fp4_v_off = fp4_k_off;
    unsigned long long sf_v_off = sf_k_off;

    #pragma unroll
    for (int g = 0; g < NVFP4_GROUPS; g++) {
        unsigned int lane_start = g * NVFP4_LANES_PER_GROUP;
        if (lane_id < lane_start || lane_id >= lane_start + NVFP4_LANES_PER_GROUP) continue;

        int local = lane_id - lane_start;

        float local_max = 0.f;
        #pragma unroll
        for (int i = 0; i < NVFP4_VEC; i++)
            local_max = fmaxf(local_max, fabsf(v_reg[i]));

        #pragma unroll
        for (int off = NVFP4_LANES_PER_GROUP/2; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));

        float scale_val = c_v * local_max / 6.0f;
        uint8_t sf_byte = float_to_e4m3(scale_val);

        if (local == 0) {
            V_sf[sf_v_off + g] = sf_byte;
        }
        // Use the locally-reduced sf_byte directly (avoids the global read-back race).
        float scale = e4m3_to_float_direct(sf_byte);
        float inv_scale = (scale > 0.f) ? (1.0f / scale) : 0.f;

        #pragma unroll
        for (int i = 0; i < NVFP4_VEC; i += 2) {
            uint8_t lo = nvfp4_float_to_e2m1(v_reg[i] * inv_scale);
            uint8_t hi = nvfp4_float_to_e2m1(v_reg[i + 1] * inv_scale);
            V_fp4[fp4_v_off + 2 * lane_id + i / 2] = (hi << 4) | (lo & 0xF);
        }
    }
}