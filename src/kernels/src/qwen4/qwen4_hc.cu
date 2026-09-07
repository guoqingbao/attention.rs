// Qwen4 (Qwen3.8-Flash-Next) Gated Residual / Hyper-Connection kernels.
//
// Reference: HuggingFace transformers `Qwen4ExpTextGatedResidual`
// (models/qwen4_exp/modeling_qwen4_exp.py):
//
//   read (mix):
//     normed  = grouped_rmsnorm(hyper_input, group=hidden) * (1 + w_norm)
//     mix_dn  = silu(W_down @ normed / hc)                      [lowrank]
//     mix_up  = sigmoid(W_up @ mix_dn)                          [hc*hidden]
//     mixed   = mean_h(mix_up[h] * normed[h])                   [hidden]
//   combine (inject gates, computed together with read):
//     inject  = 2 * sigmoid(W_inject @ normed / hc)             [hc]
//   write:
//     hyper_out[h] = hyper_in[h] + inject[h] * block_out        elementwise
//
// One thread block per token; 256 threads per block.
//
// Dtype-generic: BF16 (SM80+), F16 (SM70/75 fallback), F32. All math is
// done in FP32; only loads/stores convert, so no arch-specific intrinsics
// are needed (warp shuffles and atomicAdd on float work on SM70+).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int QWEN4_HC_MAX_HC = 16;
constexpr int QWEN4_HC_MAX_LOWRANK = 1024;
constexpr int QWEN4_HC_THREADS = 256;

__device__ __forceinline__ float hc_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float hc_to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float hc_to_f32(float v) { return v; }

__device__ __forceinline__ void hc_from_f32(float v, __nv_bfloat16* out) {
  *out = __float2bfloat16(v);
}
__device__ __forceinline__ void hc_from_f32(float v, __half* out) {
  *out = __float2half(v);
}
__device__ __forceinline__ void hc_from_f32(float v, float* out) { *out = v; }

template <typename T>
__global__ void qwen4_hc_read_kernel(
    const T* __restrict__ hyper_input,   // [seq, hc*hidden]
    const T* __restrict__ norm_weight,   // [hc*hidden] (stored zero-centered; +1 applied)
    const T* __restrict__ mix_down_weight, // [lowrank, hc*hidden]
    const T* __restrict__ mix_up_weight,   // [hc*hidden, lowrank]
    const T* __restrict__ inject_weight,   // [hc, hc*hidden] or nullptr
    T* __restrict__ mixed_out,             // [seq, hidden]
    T* __restrict__ inject_out,            // [seq, hc] or nullptr
    T* __restrict__ normed_scratch,        // [seq, hc*hidden]
    int seq_len,
    int hc,
    int hidden,
    int lowrank,
    float eps) {
    const int token = blockIdx.x;
    if (token >= seq_len) return;
    const int hc_hidden = hc * hidden;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    constexpr int NWARPS = QWEN4_HC_THREADS / 32;

    const T* x = hyper_input + (size_t)token * hc_hidden;
    T* normed = normed_scratch + (size_t)token * hc_hidden;

    __shared__ float ssq[QWEN4_HC_MAX_HC];
    __shared__ float mix_down_sh[QWEN4_HC_MAX_LOWRANK];

    if (tid < hc) ssq[tid] = 0.0f;
    __syncthreads();

    // 1) grouped sum of squares (group = one residual branch of `hidden`).
    // Warp lanes always cover 32 consecutive elements; when hidden is a
    // multiple of 32 a warp iteration stays inside one group, so reduce in
    // the warp and do a single atomicAdd instead of one per element.
    if ((hidden & 31) == 0) {
        for (int base = warp * 32; base < hc_hidden; base += QWEN4_HC_THREADS) {
            const int i = base + lane;
            const float xv = hc_to_f32(x[i]);
            float sq = xv * xv;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffffu, sq, off);
            if (lane == 0) atomicAdd(&ssq[i / hidden], sq);
        }
    } else {
        for (int i = tid; i < hc_hidden; i += QWEN4_HC_THREADS) {
            const float xv = hc_to_f32(x[i]);
            atomicAdd(&ssq[i / hidden], xv * xv);
        }
    }
    __syncthreads();

    // 2) normed = x * (1 + w) * rsqrt(mean_g(x^2) + eps)
    for (int i = tid; i < hc_hidden; i += QWEN4_HC_THREADS) {
        const float inv_rms = rsqrtf(ssq[i / hidden] / (float)hidden + eps);
        hc_from_f32(hc_to_f32(x[i]) * (1.0f + hc_to_f32(norm_weight[i])) * inv_rms,
                    &normed[i]);
    }
    __syncthreads();

    // 3) mix_down[r] = silu((W_down[r] . normed) / hc) — one warp per row
    for (int r = warp; r < lowrank; r += NWARPS) {
        const T* wrow = mix_down_weight + (size_t)r * hc_hidden;
        float acc = 0.0f;
        for (int i = lane; i < hc_hidden; i += 32) {
            acc += hc_to_f32(wrow[i]) * hc_to_f32(normed[i]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) {
            const float v = acc / (float)hc;
            mix_down_sh[r] = v / (1.0f + expf(-v)); // silu
        }
    }
    __syncthreads();

    // 4) mixed[d] = mean_h( sigmoid(W_up[h*hidden+d] . mix_down) * normed[h*hidden+d] )
    // One warp per output dim: lanes stride the lowrank reduction so global
    // loads are coalesced (32 consecutive elements per transaction) and the
    // reduction is a warp shuffle instead of a serial per-thread loop.
    T* mixed = mixed_out + (size_t)token * hidden;
    for (int d = warp; d < hidden; d += NWARPS) {
        float acc = 0.0f;
        for (int h = 0; h < hc; ++h) {
            const int idx = h * hidden + d;
            const T* wrow = mix_up_weight + (size_t)idx * lowrank;
            float macc = 0.0f;
            for (int r = lane; r < lowrank; r += 32) {
                macc += hc_to_f32(wrow[r]) * mix_down_sh[r];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) macc += __shfl_down_sync(0xffffffffu, macc, off);
            // All lanes hold the reduced value after the broadcast-free variant below.
            macc = __shfl_sync(0xffffffffu, macc, 0);
            const float gate = 1.0f / (1.0f + expf(-macc));
            acc += gate * hc_to_f32(normed[idx]);
        }
        if (lane == 0) {
            hc_from_f32(acc / (float)hc, &mixed[d]);
        }
    }

    // 5) inject[h] = 2 * sigmoid((W_inject[h] . normed) / hc) — one warp per branch
    if (inject_weight != nullptr && inject_out != nullptr) {
        T* inject = inject_out + (size_t)token * hc;
        for (int h = warp; h < hc; h += NWARPS) {
            const T* wrow = inject_weight + (size_t)h * hc_hidden;
            float acc = 0.0f;
            for (int i = lane; i < hc_hidden; i += 32) {
                acc += hc_to_f32(wrow[i]) * hc_to_f32(normed[i]);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) {
                hc_from_f32(2.0f / (1.0f + expf(-acc / (float)hc)), &inject[h]);
            }
        }
    }
}

template <typename T>
__global__ void qwen4_hc_write_kernel(
    const T* __restrict__ hyper_input, // [seq, hc*hidden]
    const T* __restrict__ block_output, // [seq, hidden]
    const T* __restrict__ inject,       // [seq, hc]
    T* __restrict__ out,                // [seq, hc*hidden]
    int seq_len,
    int hc,
    int hidden) {
    const int token = blockIdx.x;
    if (token >= seq_len) return;
    const int hc_hidden = hc * hidden;
    const T* hyper = hyper_input + (size_t)token * hc_hidden;
    const T* block = block_output + (size_t)token * hidden;
    const T* gate = inject + (size_t)token * hc;
    T* o = out + (size_t)token * hc_hidden;

    for (int i = threadIdx.x; i < hc_hidden; i += QWEN4_HC_THREADS) {
        const int h = i / hidden;
        const float g = hc_to_f32(gate[h]);
        hc_from_f32(hc_to_f32(hyper[i]) + g * hc_to_f32(block[i - h * hidden]), &o[i]);
    }
}

} // namespace

// dtype: 0 = BF16, 1 = F16, 2 = F32
extern "C" int qwen4_hc_read(
    const void* hyper_input,
    const void* norm_weight,
    const void* mix_down_weight,
    const void* mix_up_weight,
    const void* inject_weight,
    void* mixed_out,
    void* inject_out,
    void* normed_scratch,
    int seq_len,
    int hc,
    int hidden,
    int lowrank,
    float eps,
    int dtype,
    cudaStream_t stream) {
    if (seq_len <= 0) return 0;
    if (hc <= 0 || hc > QWEN4_HC_MAX_HC || hidden <= 0 || lowrank <= 0 ||
        lowrank > QWEN4_HC_MAX_LOWRANK) {
        return -1;
    }
    if (dtype == 1) {
        qwen4_hc_read_kernel<__half><<<seq_len, QWEN4_HC_THREADS, 0, stream>>>(
            reinterpret_cast<const __half*>(hyper_input),
            reinterpret_cast<const __half*>(norm_weight),
            reinterpret_cast<const __half*>(mix_down_weight),
            reinterpret_cast<const __half*>(mix_up_weight),
            reinterpret_cast<const __half*>(inject_weight),
            reinterpret_cast<__half*>(mixed_out),
            reinterpret_cast<__half*>(inject_out),
            reinterpret_cast<__half*>(normed_scratch),
            seq_len, hc, hidden, lowrank, eps);
    } else if (dtype == 2) {
        qwen4_hc_read_kernel<float><<<seq_len, QWEN4_HC_THREADS, 0, stream>>>(
            reinterpret_cast<const float*>(hyper_input),
            reinterpret_cast<const float*>(norm_weight),
            reinterpret_cast<const float*>(mix_down_weight),
            reinterpret_cast<const float*>(mix_up_weight),
            reinterpret_cast<const float*>(inject_weight),
            reinterpret_cast<float*>(mixed_out),
            reinterpret_cast<float*>(inject_out),
            reinterpret_cast<float*>(normed_scratch),
            seq_len, hc, hidden, lowrank, eps);
    } else {
        qwen4_hc_read_kernel<__nv_bfloat16><<<seq_len, QWEN4_HC_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hyper_input),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight),
            reinterpret_cast<const __nv_bfloat16*>(mix_down_weight),
            reinterpret_cast<const __nv_bfloat16*>(mix_up_weight),
            reinterpret_cast<const __nv_bfloat16*>(inject_weight),
            reinterpret_cast<__nv_bfloat16*>(mixed_out),
            reinterpret_cast<__nv_bfloat16*>(inject_out),
            reinterpret_cast<__nv_bfloat16*>(normed_scratch),
            seq_len, hc, hidden, lowrank, eps);
    }
    return (int)cudaGetLastError();
}

// dtype: 0 = BF16, 1 = F16, 2 = F32
extern "C" int qwen4_hc_write(
    const void* hyper_input,
    const void* block_output,
    const void* inject,
    void* out,
    int seq_len,
    int hc,
    int hidden,
    int dtype,
    cudaStream_t stream) {
    if (seq_len <= 0) return 0;
    if (hc <= 0 || hc > QWEN4_HC_MAX_HC || hidden <= 0) return -1;
    if (dtype == 1) {
        qwen4_hc_write_kernel<__half><<<seq_len, QWEN4_HC_THREADS, 0, stream>>>(
            reinterpret_cast<const __half*>(hyper_input),
            reinterpret_cast<const __half*>(block_output),
            reinterpret_cast<const __half*>(inject),
            reinterpret_cast<__half*>(out),
            seq_len, hc, hidden);
    } else if (dtype == 2) {
        qwen4_hc_write_kernel<float><<<seq_len, QWEN4_HC_THREADS, 0, stream>>>(
            reinterpret_cast<const float*>(hyper_input),
            reinterpret_cast<const float*>(block_output),
            reinterpret_cast<const float*>(inject),
            reinterpret_cast<float*>(out),
            seq_len, hc, hidden);
    } else {
        qwen4_hc_write_kernel<__nv_bfloat16><<<seq_len, QWEN4_HC_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hyper_input),
            reinterpret_cast<const __nv_bfloat16*>(block_output),
            reinterpret_cast<const __nv_bfloat16*>(inject),
            reinterpret_cast<__nv_bfloat16*>(out),
            seq_len, hc, hidden);
    }
    return (int)cudaGetLastError();
}
