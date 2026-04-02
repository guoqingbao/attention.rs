#include "metal_dtype.metal"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// -----------------------------------------------------------------------------
// Constants & Helper Types
// -----------------------------------------------------------------------------
#define SIMD_SIZE 32
#define WARP_SIZE 32

// Helper to convert FP8 (represented as uint8_t) to float, then cast to half
inline half fp8_to_half(uint8_t val, float scale) {
    return static_cast<half>(softmax_fp8_to_float(val) * scale);
}

// Helper to get scale with stride handling
inline float get_scale(const device float* scale,
                       int n, int k, int scale_stride,
                       int block_size_y, int block_size_x) {
  int sr = n / block_size_y;
  int sc = k / block_size_x;
  return scale[sr * scale_stride + sc];
}

// -----------------------------------------------------------------------------
// AMX-based Blocked Kernel (For M > 1)
// -----------------------------------------------------------------------------
template <typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
[[kernel]] void fp8_matmul_kernel(
    device const T  *input        [[ buffer(0) ]],
    device const uint8_t *weight       [[ buffer(1) ]],
    device const float *weight_scale [[ buffer(2) ]],
    device       T  *output       [[ buffer(3) ]],
    constant     int   &M            [[ buffer(4) ]],
    constant     int   &N            [[ buffer(5) ]],
    constant     int   &K            [[ buffer(6) ]],
    constant     int   &row_stride   [[ buffer(7) ]],
    constant     int   &block_size_y [[ buffer(8) ]],
    constant     int   &block_size_x [[ buffer(9) ]],
    uint2 gid [[ threadgroup_position_in_grid ]],
    uint  simd_lane_id [[ thread_index_in_simdgroup ]]
) {
    // -------------------------------------------------------------------------
    // 1. Threadgroup Memory (Shared Memory)
    // -------------------------------------------------------------------------
    threadgroup half s_a[BLOCK_M][BLOCK_K];
    threadgroup half s_b[BLOCK_K][BLOCK_N];

    // Accumulators
    simdgroup_matrix<float, 8, 8> acc[BLOCK_M/8][BLOCK_N/8];
    
    // Initialize accumulators to 0
    #pragma unroll
    for (int i = 0; i < BLOCK_M/8; ++i) {
        #pragma unroll
        for (int j = 0; j < BLOCK_N/8; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // Global Offsets
    int global_row_base = gid.y * BLOCK_M;
    int global_col_base = gid.x * BLOCK_N;

    // Linear thread index within the group (0-31)
    int tid = simd_lane_id; 

    // -------------------------------------------------------------------------
    // 2. Main Loop over K
    // -------------------------------------------------------------------------
    for (int k = 0; k < K; k += BLOCK_K) {
        
        // --- Load A (T) -> Dequant/Cast -> Threadgroup (half) ---
        for (int i = 0; i < BLOCK_M; ++i) {
            int r = i; 
            int c = tid; 
            
            int gr = global_row_base + r;
            int gc = k + c;
            
            half val = 0.0h;
            if (gr < M && gc < K) {
                // Load T and convert to half
                if constexpr (is_same_v<T, bfloat16_t>) {
                     val = static_cast<half>(static_cast<float>(input[gr * K + gc]));
                } else {
                     val = static_cast<half>(input[gr * K + gc]);
                }
            }
            s_a[r][c] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // --- Load B (TRANSPOSED in Shared Mem) ---
        // Optimization: Hoist scale loading.
        float tile_scale = 1.0f;
        if (global_col_base < N && k < K) {
             tile_scale = get_scale(weight_scale, global_col_base, k, row_stride, block_size_y, block_size_x);
        }
        
        // Vectorized Load Strategy (Row-wise from Global, Col-wise to Shared)
        int n_local = tid; // Row of the tile we are loading (0..31)
        int gn = global_col_base + n_local;
        
        if (gn < N) {
             // We load 32 bytes from `weight[gn][k..k+31]`.
             const device uint* w_row = (const device uint*)(weight + gn * K + k);
             
             #pragma unroll
             for (int j = 0; j < 8; ++j) {
                  uint val_u = w_row[j];
                  half4 h4 = scaled_vec_conversion<half4, uint32_t>(val_u, tile_scale);
                  
                  int k_idx_base = j * 4;
                  s_b[k_idx_base + 0][n_local] = h4.x;
                  s_b[k_idx_base + 1][n_local] = h4.y;
                  s_b[k_idx_base + 2][n_local] = h4.z;
                  s_b[k_idx_base + 3][n_local] = h4.w;
             }
        } else {
             #pragma unroll
             for (int j = 0; j < 32; ++j) {
                  s_b[j][n_local] = 0.0h;
             }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Compute (SIMD Matrix MMA) ---
        for (int k_step = 0; k_step < BLOCK_K; k_step += 8) {
            simdgroup_matrix<half, 8, 8> fragA;
            simdgroup_matrix<half, 8, 8> fragB;
            
            #pragma unroll
            for (int row_tile = 0; row_tile < BLOCK_M/8; ++row_tile) {
                #pragma unroll
                for (int col_tile = 0; col_tile < BLOCK_N/8; ++col_tile) {
                    simdgroup_load(fragA, &s_a[row_tile * 8 + 0][k_step], BLOCK_K, ulong2(0,0), false);
                    simdgroup_load(fragB, &s_b[k_step][col_tile * 8 + 0], BLOCK_N, ulong2(0,0), false);
                    simdgroup_multiply_accumulate(acc[row_tile][col_tile], fragA, fragB, acc[row_tile][col_tile]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------------------------------------------------------------------------
    // 3. Store Results
    // -------------------------------------------------------------------------
    threadgroup float s_out[BLOCK_M][BLOCK_N];
    
    #pragma unroll
    for (int row_tile = 0; row_tile < BLOCK_M/8; ++row_tile) {
        #pragma unroll
        for (int col_tile = 0; col_tile < BLOCK_N/8; ++col_tile) {
            simdgroup_store(acc[row_tile][col_tile], &s_out[row_tile * 8][col_tile * 8], BLOCK_N, ulong2(0,0), false);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    int local_r = tid; 
    int global_r = global_row_base + local_r;
         
    if (local_r < BLOCK_M && global_r < M) { 
         for (int local_c = 0; local_c < BLOCK_N; ++local_c) {
              int global_c = global_col_base + local_c;
              if (global_c < N) {
                  float val = s_out[local_r][local_c];
                  if constexpr (is_same_v<T, bfloat16_t>) {
                       output[global_r * N + global_c] = static_cast<bfloat16_t>(val);
                  } else {
                       output[global_r * N + global_c] = static_cast<T>(val);
                  }
              }
         }
    }
}


// -----------------------------------------------------------------------------
// Warp-Parallel GEMV Kernel (For M = 1 / Small M)
// -----------------------------------------------------------------------------
// Specialized for Matrix [M, K] x Weight [N, K] -> Output [M, N].
// Grid: (N * 32, M, 1). Each warp handles one output element (n).
template <typename T>
[[kernel]] void fp8_gemv_kernel(
    device const T  *input        [[ buffer(0) ]],
    device const uint8_t *weight       [[ buffer(1) ]],
    device const float *weight_scale [[ buffer(2) ]],
    device       T  *output       [[ buffer(3) ]],
    constant     int   &M            [[ buffer(4) ]],
    constant     int   &N            [[ buffer(5) ]],
    constant     int   &K            [[ buffer(6) ]],
    constant     int   &row_stride   [[ buffer(7) ]],
    constant     int   &block_size_y [[ buffer(8) ]], // N block size for scale
    constant     int   &block_size_x [[ buffer(9) ]],  // K block size for scale
    uint3 gid [[ thread_position_in_grid ]],
    uint  simd_lane_id [[ thread_index_in_simdgroup ]]
) {
    // Determine which output element (row m, col n) this warp is computing.
    // gridDim.x = N * 32.
    int n_out = gid.x / 32;
    int m_out = gid.y;
    
    if (n_out >= N || m_out >= M) return;

    // Pointers
    // Input Row m: input + m_out * K
    // Weight Row n: weight + n_out * K
    device const T* x_ptr = input + m_out * K;
    device const uint8_t* w_ptr = weight + n_out * K;

    // Accumulators
    float sum_f = 0.0f;

    // Warp Loop over K
    // Lane ID (0..31) acts as k_offset.
    // Unroll manually / Vector loads?
    // User reference used 'uint2' (8 bytes) + 'half2' (vector math).
    
    // We iterate K in chunks of 32 * UNROLL.
    // Let's use 32 * 4 elements per iteration (4 elements per thread).
    
    int lane = simd_lane_id;

    // Aggressive Unrolling: 16 elements per thread.
    // Warp Stride: 32 * 16 = 512.
    // Vector Weights: uint4 (16 bytes = 16 FP8 values).
    
    // Fast Path Loop Limit
    // We can run the fast loop as long as the entire warp is within bounds.
    // The last thread (lane 31) accesses `k + 15`.
    // Warp base `k_warp`. Thread k = `k_warp + lane * 16`.
    // Last element accessed by wrap: `k_warp + 31 * 16 + 15` = `k_warp + 511`.
    // So we can run while `k_warp + 511 < K`.
    
    int k_warp = 0;
    int K_fast_limit = K - 512; 
    
    // 1. FAST PATH (No Bounds Checks)
    while (k_warp <= K_fast_limit) {
         int k = k_warp + lane * 16;
         
         // Load Scale (Assume shared for block of 16)
         float s = get_scale(weight_scale, n_out, k, row_stride, block_size_y, block_size_x);

         // Load Weights (uint4 = 16 bytes)
         uint4 w_u4 = *(device const uint4*)(w_ptr + k);
         
         half4 wh_0 = scaled_vec_conversion<half4, uint32_t>(w_u4.x, s);
         half4 wh_1 = scaled_vec_conversion<half4, uint32_t>(w_u4.y, s);
         half4 wh_2 = scaled_vec_conversion<half4, uint32_t>(w_u4.z, s);
         half4 wh_3 = scaled_vec_conversion<half4, uint32_t>(w_u4.w, s);
         
         // Load Inputs
         if constexpr (is_same_v<T, bfloat16_t>) {
              // Load 16 bfloats (32 bytes).
              // Manually load or cast. 
              // Load 2 x (type compatible with 16 bytes)? 
              // bfloat16 is 2 bytes. 16 = 32 bytes.
              // Load 2 x ulong2 (16 bytes each)? Or 8 x float?
              // Let's just load individual blocks or use pointer cast if aligned.
              // Assuming aligned to 2 bytes always.
              
              // Load 0..3
              T x0 = x_ptr[k+0]; T x1 = x_ptr[k+1]; T x2 = x_ptr[k+2]; T x3 = x_ptr[k+3];
              float4 xf_0 = float4(float(x0), float(x1), float(x2), float(x3));
              sum_f += dot(float4(wh_0), xf_0);
              
              // Load 4..7
              T x4 = x_ptr[k+4]; T x5 = x_ptr[k+5]; T x6 = x_ptr[k+6]; T x7 = x_ptr[k+7];
              float4 xf_1 = float4(float(x4), float(x5), float(x6), float(x7));
              sum_f += dot(float4(wh_1), xf_1);

              // Load 8..11
              T x8 = x_ptr[k+8]; T x9 = x_ptr[k+9]; T x10 = x_ptr[k+10]; T x11 = x_ptr[k+11];
              float4 xf_2 = float4(float(x8), float(x9), float(x10), float(x11));
              sum_f += dot(float4(wh_2), xf_2);

              // Load 12..15
              T x12 = x_ptr[k+12]; T x13 = x_ptr[k+13]; T x14 = x_ptr[k+14]; T x15 = x_ptr[k+15];
              float4 xf_3 = float4(float(x12), float(x13), float(x14), float(x15));
              sum_f += dot(float4(wh_3), xf_3);
              
         } else {
              // Half (16 halves = 32 bytes)
              // Load 4 x half4
              half4 xh_0 = *(device const half4*)(x_ptr + k);
              half4 xh_1 = *(device const half4*)(x_ptr + k + 4);
              half4 xh_2 = *(device const half4*)(x_ptr + k + 8);
              half4 xh_3 = *(device const half4*)(x_ptr + k + 12);
              
              sum_f += dot(float4(wh_0), float4(xh_0));
              sum_f += dot(float4(wh_1), float4(xh_1));
              sum_f += dot(float4(wh_2), float4(xh_2));
              sum_f += dot(float4(wh_3), float4(xh_3));
         }
         
         k_warp += 512;
    }
    
    // 2. TAIL / CLEANUP LOOP
    // Iterate from k_warp to K with checks.
    // Stride is still 512 (conceptually) but we just fill the rest.
    // Actually, we can just switch to a scalar-ish loop per thread.
    // Each thread processes `lane * 16`, `lane*16 + 512`...
    
    for (int k = k_warp + lane * 16; k < K; k += 512) {
         // Scalar or small-vector cleanup for this chunk
         // We have a chunk of 16 elements to process (potentially partial)
         
         if (k + 15 < K) {
             // Safe to do vector load even in cleanup? 
             // Yes if strictly < K.
             float s = get_scale(weight_scale, n_out, k, row_stride, block_size_y, block_size_x);
             uint4 w_u4 = *(device const uint4*)(w_ptr + k);
             
             half4 wh_0 = scaled_vec_conversion<half4, uint32_t>(w_u4.x, s);
             half4 wh_1 = scaled_vec_conversion<half4, uint32_t>(w_u4.y, s);
             half4 wh_2 = scaled_vec_conversion<half4, uint32_t>(w_u4.z, s);
             half4 wh_3 = scaled_vec_conversion<half4, uint32_t>(w_u4.w, s);

             if constexpr (is_same_v<T, bfloat16_t>) {
                  // (Repeat inner logic or consolidate)
                   T x0 = x_ptr[k+0]; T x1 = x_ptr[k+1]; T x2 = x_ptr[k+2]; T x3 = x_ptr[k+3];
                   sum_f += dot(float4(wh_0), float4((float)x0, (float)x1, (float)x2, (float)x3));
                   T x4 = x_ptr[k+4]; T x5 = x_ptr[k+5]; T x6 = x_ptr[k+6]; T x7 = x_ptr[k+7];
                   sum_f += dot(float4(wh_1), float4((float)x4, (float)x5, (float)x6, (float)x7));
                   T x8 = x_ptr[k+8]; T x9 = x_ptr[k+9]; T x10 = x_ptr[k+10]; T x11 = x_ptr[k+11];
                   sum_f += dot(float4(wh_2), float4((float)x8, (float)x9, (float)x10, (float)x11));
                   T x12 = x_ptr[k+12]; T x13 = x_ptr[k+13]; T x14 = x_ptr[k+14]; T x15 = x_ptr[k+15];
                   sum_f += dot(float4(wh_3), float4((float)x12, (float)x13, (float)x14, (float)x15));
             } else {
                  half4 xh_0 = *(device const half4*)(x_ptr + k);
                  half4 xh_1 = *(device const half4*)(x_ptr + k + 4);
                  half4 xh_2 = *(device const half4*)(x_ptr + k + 8);
                  half4 xh_3 = *(device const half4*)(x_ptr + k + 12);
                  sum_f += dot(float4(wh_0), float4(xh_0));
                  sum_f += dot(float4(wh_1), float4(xh_1));
                  sum_f += dot(float4(wh_2), float4(xh_2));
                  sum_f += dot(float4(wh_3), float4(xh_3));
             }
         } else {
             // Very safe scalar fallback for extreme tail
             float s = get_scale(weight_scale, n_out, k, row_stride, block_size_y, block_size_x);
             for (int i = 0; i < 16; ++i) {
                  int curr_k = k + i;
                  if (curr_k < K) {
                       uint8_t w = w_ptr[curr_k];
                       T x = x_ptr[curr_k];
                       float wf = softmax_fp8_to_float(w) * s;
                       float xf; 
                       if constexpr (is_same_v<T, bfloat16_t>) xf = (float)x; 
                       else xf = (float)(half)x;
                       sum_f += wf * xf;
                  }
             }
         }
    }
    
    // Warp Reduction
    sum_f = simd_sum(sum_f);
    
    // Write Output
    if (lane == 0) {
        if constexpr (is_same_v<T, bfloat16_t>) {
             output[m_out * N + n_out] = static_cast<bfloat16_t>(sum_f);
        } else {
             output[m_out * N + n_out] = static_cast<T>(sum_f);
        }
    }
}


// -----------------------------------------------------------------------------
// Instantiations
// -----------------------------------------------------------------------------

// Standard AMX Kernels
template [[host_name("fp8_matmul_half_32_32_32")]] [[kernel]] void fp8_matmul_kernel<half, 32, 32, 32>(
    device const half* input [[buffer(0)]],
    device const uint8_t* weight [[buffer(1)]],
    device const float* weight_scale [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    constant int& row_stride [[buffer(7)]],
    constant int& block_size_y [[buffer(8)]],
    constant int& block_size_x [[buffer(9)]],
    uint2 gid [[ threadgroup_position_in_grid ]],
    uint simd_lane_id [[ thread_index_in_simdgroup ]]
);

#if defined(__HAVE_BFLOAT__)
template [[host_name("fp8_matmul_bfloat16_32_32_32")]] [[kernel]] void fp8_matmul_kernel<bfloat16_t, 32, 32, 32>(
    device const bfloat16_t* input [[buffer(0)]],
    device const uint8_t* weight [[buffer(1)]],
    device const float* weight_scale [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    constant int& row_stride [[buffer(7)]],
    constant int& block_size_y [[buffer(8)]],
    constant int& block_size_x [[buffer(9)]],
    uint2 gid [[ threadgroup_position_in_grid ]],
    uint simd_lane_id [[ thread_index_in_simdgroup ]]
);
#endif

// Small-AMX Kernels 
template [[host_name("fp8_matmul_half_16_32_32")]] [[kernel]] void fp8_matmul_kernel<half, 16, 32, 32>(
    device const half* input [[buffer(0)]],
    device const uint8_t* weight [[buffer(1)]],
    device const float* weight_scale [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    constant int& row_stride [[buffer(7)]],
    constant int& block_size_y [[buffer(8)]],
    constant int& block_size_x [[buffer(9)]],
    uint2 gid [[ threadgroup_position_in_grid ]],
    uint simd_lane_id [[ thread_index_in_simdgroup ]]
);

#if defined(__HAVE_BFLOAT__)
template [[host_name("fp8_matmul_bfloat16_16_32_32")]] [[kernel]] void fp8_matmul_kernel<bfloat16_t, 16, 32, 32>(
    device const bfloat16_t* input [[buffer(0)]],
    device const uint8_t* weight [[buffer(1)]],
    device const float* weight_scale [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    constant int& row_stride [[buffer(7)]],
    constant int& block_size_y [[buffer(8)]],
    constant int& block_size_x [[buffer(9)]],
    uint2 gid [[ threadgroup_position_in_grid ]],
    uint simd_lane_id [[ thread_index_in_simdgroup ]]
);
#endif

// GEMV Kernels (M=1)
template [[host_name("fp8_gemv_half")]] [[kernel]] void fp8_gemv_kernel<half>(
    device const half* input [[buffer(0)]],
    device const uint8_t* weight [[buffer(1)]],
    device const float* weight_scale [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    constant int& row_stride [[buffer(7)]],
    constant int& block_size_y [[buffer(8)]],
    constant int& block_size_x [[buffer(9)]],
    uint3 gid [[ thread_position_in_grid ]],
    uint simd_lane_id [[ thread_index_in_simdgroup ]]
);

#if defined(__HAVE_BFLOAT__)
template [[host_name("fp8_gemv_bfloat16")]] [[kernel]] void fp8_gemv_kernel<bfloat16_t>(
    device const bfloat16_t* input [[buffer(0)]],
    device const uint8_t* weight [[buffer(1)]],
    device const float* weight_scale [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    constant int& row_stride [[buffer(7)]],
    constant int& block_size_y [[buffer(8)]],
    constant int& block_size_x [[buffer(9)]],
    uint3 gid [[ thread_position_in_grid ]],
    uint simd_lane_id [[ thread_index_in_simdgroup ]]
);
#endif
