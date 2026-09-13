# NVFP4 KV Cache for SM120

## Status

All NVFP4 KV cache kernels are implemented, build clean on SM120
(RTX 5090 Laptop, CUDA 13.3, sm_120a), and pass 23/23 unit tests.
Full numerical accuracy verification against a BF16 reference
requires model weights and is deferred to the inference engine
integration phase.

## Interfaces

All functions are in `src/flash.rs`, gated behind
`#[cfg(feature = "cuda")]`. FFI declarations are in
`src/kernels/src/ffi.rs`. Kernel implementations are in
`src/kernels/src/flash/`.

### Store (analogous to `flash_reshape_and_cache`)

```rust
pub fn flash_nvfp4_kv_store(
    key: &Tensor,        // [num_tokens, num_kv_heads, head_dim] BF16/FP16
    value: &Tensor,      // [num_tokens, num_kv_heads, head_dim] BF16/FP16
    k_fp4: &Tensor,      // [num_blocks, block_size, num_kv_heads, head_dim/2] U8
    k_sf: &Tensor,       // [num_blocks, block_size, num_kv_heads, head_dim/16] U8
    v_fp4: &Tensor,      // same shape as k_fp4
    v_sf: &Tensor,       // same shape as k_sf
    slot_mapping: &Tensor, // [num_tokens] I64
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    c_k: f32,            // K scale constant (1.0 = standard NVFP4)
    c_v: f32,            // V scale constant (1.0 = standard NVFP4)
    rotate: bool,        // WHT rotation on K (UltraQuant paper)
) -> Result<()>
```

Comparison to existing paths:
- `flash_reshape_and_cache` (BF16/FP16): writes raw half to paged cache
- `flash_reshape_and_cache` with `is_fp8=true`: writes FP8 E4M3 with per-head float scales
- `flash_nvfp4_kv_store`: writes FP4 E2M1 + E4M3 group-16 scales, optional WHT rotation

The inference engine allocates `k_fp4`, `k_sf`, `v_fp4`, `v_sf` as
contiguous U8 tensors with the shapes above. Total bytes per head
per token: `head_dim/2 + head_dim/16` (e.g. 64 + 8 = 72 for hd=128).

### Decode (analogous to `flash_decode`)

```rust
pub fn flash_nvfp4_kv_decode(
    query: &Tensor,      // [num_seqs, num_q_heads, head_dim] BF16/FP16
    k_fp4: &Tensor,      // [num_blocks, block_size, num_kv_heads, head_dim/2] U8
    k_sf: &Tensor,       // [num_blocks, block_size, num_kv_heads, head_dim/16] U8
    v_fp4: &Tensor,      // same as k_fp4
    v_sf: &Tensor,       // same as k_sf
    block_tables: &Tensor, // [num_seqs, max_blocks_per_seq] U32
    context_lens: &Tensor,  // [num_seqs] U32
    output: &Tensor,     // [num_seqs, num_q_heads, head_dim] BF16/FP16
    max_context_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,          // 1/sqrt(head_dim)
    softcap: f32,        // 0.0 = disabled
    sliding_window: Option<usize>,
    rotate: bool,        // must match store's rotate parameter
) -> Result<Tensor>
```

Comparison to existing paths:
- `flash_decode` (BF16/FP16): reads raw paged cache, standard WMMA attention
- `flash_decode` with `is_fp8=true`: reads FP8 paged cache, dequant in registers
- `flash_nvfp4_kv_decode`: reads NVFP4 paged cache, software LUT dequant in registers

The decode kernel uses the same paged iteration pattern as
`flash_tq4_decode` (TQ4_NUM_WARPS=8 warps per block, online softmax,
inter-warp reduction via shared memory). The only difference is the
dequantization: FP4 E2M1 LUT (8 entries) * E4M3 scale (per 16-element
group) instead of uniform 4-bit * per-head float absmax.

### Prefill (analogous to `flash_prefill`)

```rust
pub fn flash_nvfp4_kv_prefill(
    query: &Tensor,      // [total_q_tokens, num_q_heads, head_dim] BF16/FP16
    k_fp4: &Tensor,      // [num_blocks, block_size, num_kv_heads, head_dim/2] U8
    k_sf: &Tensor,       // [num_blocks, block_size, num_kv_heads, head_dim/16] U8
    v_fp4: &Tensor,      // same as k_fp4
    v_sf: &Tensor,       // same as k_sf
    block_tables: &Tensor, // [num_seqs, max_blocks_per_seq] U32
    context_lens: &Tensor,  // [num_seqs] U32
    output: &Tensor,     // [total_q_tokens, num_q_heads, head_dim] BF16/FP16
    max_context_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    softcap: f32,
    sliding_window: Option<usize>,
    rotate: bool,
) -> Result<Tensor>
```

The prefill kernel uses 4 warps per block (128 threads) and
processes one query tile per block. It reads the same paged NVFP4
cache as the decode kernel.

## Buffer Layout (head_dim=128 example)

Per token per KV:
- K_fp4: 64 bytes (128 FP4 codes, 2 per byte)
- K_sf:  8 bytes (8 E4M3 scales, one per 16-element group)
- V_fp4: 64 bytes
- V_sf:  8 bytes
- Total: 148 bytes per token per head (K+V)
- vs BF16: 512 bytes (2 * 128 * 2)
- vs FP8: 256 bytes (2 * 128 * 1)
- Compression: 3.45x vs BF16, 1.73x vs FP8

Paged tensor shapes (num_blocks=B, block_size=S, num_kv_heads=H):
- k_fp4: [B, S, H, 64] U8
- k_sf:  [B, S, H, 8] U8
- v_fp4: [B, S, H, 64] U8
- v_sf:  [B, S, H, 8] U8

## Rotation and Constants

The `rotate` parameter enables Walsh-Hadamard rotation on K (store)
and Q (decode/prefill), per the UltraQuant paper. The rotation is
orthogonal (H^T H = I), so attention scores are invariant:
(H Q) . (H K)^T = Q . K^T.

The `c_k` and `c_v` parameters scale the E4M3 block:
- c=1.0: standard NVFP4 (scale = amax/6.0, matches FlashMLA V41_FP4)
- c<1.0: shrinks the quantization grid (finer resolution for the
  bulk of the distribution, clips the tails)
- The paper's c=0.156 is tuned for FP4 E2M1 + UE8M0 (AMD CDNA4).
  For NVFP4 (E4M3, group 16) on NVIDIA, the optimal c must be
  calibrated per-model from real activation distributions.

## What the Inference Engine Must Do

1. Allocate k_fp4, k_sf, v_fp4, v_sf tensors with the shapes above
2. Call `flash_nvfp4_kv_store` when writing new KV tokens to the cache
3. Call `flash_nvfp4_kv_decode` for single-token attention (generation)
4. Call `flash_nvfp4_kv_prefill` for multi-token attention (prompt processing)
5. Pass `rotate=true` and calibrated `c_k`, `c_v` for the UltraQuant path,
   or `rotate=false, c_k=1.0, c_v=1.0` for standard NVFP4
6. The `block_tables` and `context_lens` tensors use the same format
   as the existing FP8/BF16 paged attention paths

## Verified

- 23/23 unit tests pass on SM120 (RTX 5090 Laptop, CUDA 13.3)
- Store kernel: valid E4M3 scales, valid FP4 codes, correct paged offsets
- Decode kernel: launches without GPU crash, produces finite non-zero output
- Prefill kernel: launches without GPU crash
- TQ4 c_k/c_v: bit-identical to existing TQ4 when c=1.0
- FP8 rotation: store and decode run cleanly

## Deferred

- Numerical accuracy of NVFP4 decode vs BF16 reference attention
  (requires model weights for a meaningful comparison)
- Throughput benchmark (decode tok/s at various4K/32K/128K context)
- Long-context stability (128K+ tokens, numeric accumulation error)
- Hardware block-scaled MMA path (CUTLASS SM120, 99 KB SMEM constraint
  forces 32x32 tiles; future optimization)

## File Map

| File | Role |
|---|---|
| `src/kernels/src/flash/flash_nvfp4_kv_store.cuh` | Store kernel + FP4/E4M3 helpers |
| `src/kernels/src/flash/flash_nvfp4_kv_decode.cuh` | Paged decode kernel |
| `src/kernels/src/flash/flash_nvfp4_kv_prefill.cuh` | Paged prefill kernel |
| `src/kernels/src/flash/flash_instantiate.cu` | HD macros, includes, FFI launchers |
| `src/kernels/src/ffi.rs` | Rust FFI declarations |
| `src/flash.rs` | Public Rust API (store, decode, prefill) |
| `src/nvfp4_kv_tests.rs` | 7 unit tests |
| `src/kernels/src/flash/flash_fp8_rot_store.cuh` | FP8 + WHT rotation store |
| `src/kernels/src/flash/flash_fp8_rot_decode.cuh` | FP8 + WHT rotation decode |
| `src/fp8_rot_tests.rs` | 3 unit tests for FP8 rotation |
| `src/kernels/src/flash/flash_turboquant_lowbit.cuh` | TQ4 store (c_k/c_v added) |
| `src/tq4_c_tests.rs` | 4 unit tests for TQ4 c_k/c_v |

## CUDA Version Gating

The FlashInfer dependency (commit 2bfb9334+) references CCCL types
(`cuda::fast_mod_div`, `cuda::maximum`) that have different
availability depending on the CUDA toolkit version:

- CUDA 12.x (CCCL 2.x): `cuda::fast_mod_div` is an internal type,
  not on the default include path when CUTLASS's bundled CUB is
  active. Our polyfill in `flashinfer_cccl_compat.h` provides it.
  Guarded by `#if __CUDACC_VER_MAJOR__ < 13`.

- CUDA 13+ (CCCL 3.0): `cuda::fast_mod_div` is a public type in
  `<cuda/std/...>`, always on the include path. The polyfill is
  skipped to avoid an ambiguous overload.

Minimum supported CUDA: 12.6 (per Dockerfile base image
nvidia/cuda:12.9.1). The NVFP4 KV cache kernels themselves have
no CCCL dependency and build cleanly on any CUDA 12.0+.

The `cuda::maximum` polyfill (for the TRT-LLM quantization path)
is guarded by `#ifndef _CUDA_FUNCTIONAL_MAXIMUM_H` which is the
include guard for `<cuda/functional>`. On CUDA 13+ this header is
always available, so the polyfill is skipped. On CUDA 12.x with
CUTLASS's bundled CUB, it may be missing, so the polyfill is
active.

## Citations

Chakrabarti et al., "UltraQuant: 4-bit KV Caching for
Context-Heavy Agents," arXiv:2606.20474v2 (2026). Source of
the WHT rotation + asymmetric-tensor scale constant design.
The paper targets AMD CDNA4 (MFMA, UE8M0 group-32); this
branch adapts the rotation and constant to the NVIDIA NVFP4
format (E4M3 group-16) proven on SM120.

Zandieh et al., "TurboQuant: Online vector quantization
with near-optimal distortion rate," ICLR 2026. Source of
the Walsh-Hadamard rotation + codebook quantization approach
used in the existing TQ4 path (flash_tq4_store, flash_tq4_decode).

DeepSeek-AI, FlashMLA (github.com/deepseek-ai/FlashMLA).
V41_FP4 KV cache layout: FP4 E2M1 data + E4M3 group-16
scales, 288 bytes/token for D=512. This branch uses separate
k_fp4/k_sf tensors instead of the interleaved single-buffer
layout, matching the existing TQ4/FP8 paged pattern in this repo.

hikarioyama/vllm-nvfp4-kv-sm120 (github.com). Proven SM120
NVFP4 KV decode via FlashInfer FA2 with explicit SF strides
and in-register dequant. 1.78x FP8 KV pool at 91-100% decode
speed on RTX PRO 6000. This branch implements the same
software-dequant pattern in native CUDA kernels.

NVIDIA CUTLASS 4.5.2, SM120 block-scaled GEMM
(OpClassBlockScaledTensorOp, float_e2m1_t + float_ue8m0_t).
Referenced for the hardware path (future optimization);
this branch uses the software LUT path for decode.

## CUDA Version Gating

The FlashInfer dependency (commit 2bfb9334+) references CCCL types (`cuda::fast_mod_div`, `cuda::maximum`) that
are only on the default include path in CUDA 13.0+.
In CUDA 12.x, libcu++ ships as a separate package and is
not found when CUTLASS's bundled CUB is used.

`src/kernels/src/flashinfer_cccl_compat.h` provides
polyfills for both types, guarded by:

    #if !defined(CUDA_VERSION_13) && !defined(_CUDA_STD_DETAIL_FAST_MATH_H)

- CUDA 12.x: polyfill is active (CCCL types unavailable)
- CUDA 13+: polyfill is skipped (native CCCL types found)

`build.rs` detects the CUDA major version via `nvcc --version`
and sets `-DCUDA_VERSION_13` when >= 13. The minimum
supported CUDA version for this project is 12.6 (per
Dockerfile base image nvidia/cuda:12.9.1).

The NVFP4 KV cache kernels themselves (store, decode,
prefill) have no CCCL dependency and build cleanly on
CUDA 12.6+. Only the FlashInfer adapter files
(flashinfer_adapter_decode.cu, flashinfer_prefill_fp8_fa2.cu,
flashinfer_adapter_prefill.cu) require the polyfill.