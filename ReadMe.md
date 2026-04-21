# attention-rs

High-performance GPU kernels for LLM inference, built on [Candle](https://github.com/huggingface/candle).

`attention-rs` provides production-ready CUDA and Metal implementations of the core
operations needed to serve large language models: paged attention, FlashInfer, mixture-of-experts,
FP8/FP4 quantized linear layers, rotary embeddings, Mamba/GDN state-space models, and more.

Used in production by [vllm.rs](https://github.com/guoqingbao/vllm.rs) and
[candle-vllm](https://github.com/EricLBuehler/candle-vllm).

[![Crates.io](https://img.shields.io/crates/v/attention-rs.svg)](https://crates.io/crates/attention-rs)
[![docs.rs](https://docs.rs/attention-rs/badge.svg)](https://docs.rs/attention-rs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Features at a glance

| Component | CUDA | Metal | Description |
|-----------|:----:|:-----:|-------------|
| [Paged Attention](#paged-attention) | Yes | Yes | Memory-efficient KV-cache with block allocation |
| [FlashInfer](#flashinfer) | Yes | -- | Plan-then-run attention (prefill, decode, MLA) |
| [Flash Attention v2/V3](#flash-attention) | Yes | -- | Variable-length and KV-cache-aware flash attention |
| [Mixture of Experts](#mixture-of-experts) | Yes | Yes | Fused MoE GEMM (F16/BF16, FP8, NVFP4, MXFP4, GGUF) |
| [FP8 Linear](#fp8-linear) | Yes | Yes | Block-scaled FP8 matmul (FlashInfer / CUTLASS / fallback for SM70/SM75) |
| [MXFP4 Linear](#mxfp4-linear) | Yes | Yes | MX FP4 with E8M0 scales (CUTLASS Blackwell + WMMA) |
| [NVFP4 Linear](#nvfp4-linear) | Yes | -- | NVIDIA FP4 E2M1 (FlashInfer / CUTLASS / software for SM70/SM75/SM80/SM89) |
| [Gated Delta Net](#gated-delta-net) | Yes | Yes | Linear attention kernels for Qwen 3.5 |
| [MLA](#multi-head-latent-attention) | Yes | -- | DeepSeek-style compressed KV heads |
| [Fused RoPE](#fused-rope) | Yes | Yes | Fused rotary position embeddings |
| [GPU Sampling](#gpu-sampling) | Yes | -- | Fused Top-K / Top-P / temperature sampling |
| [Mamba Cache](#mamba-cache) | Yes | Yes | Slot-based state management for SSM layers |

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
attention-rs = { version = "0.5", features = ["cuda"] }
```

Or for Metal (macOS / Apple Silicon):

```toml
[dependencies]
attention-rs = { version = "0.5", features = ["metal"] }
```

### Feature flags

| Feature | Description |
|---------|-------------|
| **`candle-upstream`** *(default)* | Use `candle-core` / `candle-nn` 0.10 from crates.io |
| **`candle-custom`** | Use the [guoqingbao/candle](https://github.com/guoqingbao/candle) fork |
| **`cuda`** | Enable CUDA kernels (requires NVIDIA GPU + CUDA toolkit) |
| **`metal`** | Enable Metal kernels (requires macOS + Apple Silicon) |
| **`flashattn`** | Flash Attention v2 integration via `flashattn-rs`, v3 for Hopper |
| **`flashinfer`** | FlashInfer paged attention (implies `cuda`) |
| **`cutlass`** | CUTLASS-optimized FP8/FP4 kernels for SM90+ / SM100+ |
| **`trtllm`** | TensorRT-LLM cubin loader (implies `flashinfer`, Placeholder, WIP) |
| **`graph`** | CUDA graph capture support (custom fork only) |
| **`no-fp8-kvcache`** | Disable FP8 KV-cache kernels (smaller build) |
| **`no-marlin`** | Disable Marlin kernels (smaller build) |

> **Note:** Enable exactly one of `candle-upstream` or `candle-custom`, and at
> most one of `cuda` or `metal`. Enabling conflicting features is a compile error.

---

## Candle compatibility

`attention-rs` is designed to work with **any Candle fork** that preserves the
core `candle-core` API surface (`Tensor`, `Storage`, `CustomOp1`, `InplaceOp1`,
`DType`, `Device`, etc.). A thin [compatibility layer](src/compat.rs) handles
the small API differences between upstream Candle and the custom fork.

| Candle variant | How to use |
|----------------|------------|
| **Upstream** (crates.io 0.10+) | Default -- just add `attention-rs` |
| **guoqingbao/candle** fork | `default-features = false, features = ["candle-custom", "cuda"]` |
| **Other forks** | Should work if `candle-core` API is compatible with 0.10 |

When another crate in your workspace already provides `candle-core` via a path
dependency, use `[patch.crates-io]` to redirect resolution:

```toml
[patch.crates-io]
candle-core = { path = "./your-candle/candle-core" }
```

---

## Components

### Paged Attention

Memory-efficient attention with paged KV-cache allocation, eliminating
fragmentation for long-running serving workloads.

```rust
use attention_rs::{PagedAttention, InputMetadata};

let attn = PagedAttention::new(
    32,       // num_attention_heads
    128,      // head_dim
    0.08838,  // scale = 1/sqrt(head_dim)
    Some(8),  // num_key_value_heads (GQA)
    None,     // sliding_window
    device,
    None,     // alibi_slopes
    false,    // fp8_kvcache
)?;

// forward() auto-selects the best backend:
// FlashInfer > FlashAttn > PagedAttn > naive SDP
let output = attn.forward(
    &query, &key, &value,
    None,               // attention_mask (for SDP fallback)
    Some(key_cache),    // paged KV cache
    Some(value_cache),
    &input_metadata,
    None,               // softcapping
)?;
```

**Supports:** GQA/MQA, ALiBi slopes, sliding window, softcapping, FP8 KV caches
with dynamic scale tracking, chunked prefill with block tables.

### FlashInfer

Plan-then-run FlashInfer attention for maximum throughput. Separates the
expensive index computation (plan) from the GPU kernel (run), enabling
CUDA graph capture.

```rust
use attention_rs::flashinfer;

// Step 1: Plan on host (can be cached across iterations with same shape)
let plan = flashinfer::decode_plan(
    &device, kv_dtype, out_dtype,
    &indptr_host, &last_len_host, &kv_len_arr_host,
    batch_size, num_qo_heads, num_kv_heads, head_dim,
    page_size, enable_cuda_graph,
)?;

// Step 2: Run on GPU
let output = flashinfer::decode_with_plan(
    &query, &key_cache, &value_cache,
    k_scale.as_ref(), v_scale.as_ref(),
    &indices, &indptr, &last_len,
    block_size, num_qo_heads, num_kv_heads, head_dim,
    sm_scale, &plan, enable_cuda_graph,
    Some(window_left), Some(logits_soft_cap),
)?;
```

Also provides: `prefill_plan` / `prefill_with_plan`, `prefill_ragged`,
`append_kv_cache`, and MLA variants via [`mla`].

### Flash Attention

Flash Attention v2 (v3 for Hopper) integration via `flashattn-rs` for variable-length
sequences and KV-cache-aware prefill/decode.

```rust
// Available through PagedAttention::forward() when flashattn feature is enabled,
// or directly:
let output = attn.flash_forward(
    &query, &key, &value,
    Some(key_cache), Some(value_cache),
    &input_metadata,
    Some(softcapping),
)?;
```

### Mixture of Experts

Fused MoE GEMM supporting multiple quantization formats:

```rust
use attention_rs::moe;

// Standard F16/BF16 MoE
let output = moe::moe_gemm(
    &input, &weights, &topk_weights,
    &sorted_token_ids, &experts_ids,
    topk, is_prefill,
)?;

// FP8 quantized experts
let output = moe::moe_gemm_fp8(
    &input, &weights, &topk_weights,
    &sorted_token_ids, &experts_ids,
    topk, is_prefill,
    &weight_scales, block_size,
)?;

// MXFP4 quantized experts (CUDA + Metal)
let output = moe::moe_gemm_mxfp4(
    &input, &weights, &scales,
    &topk_weights, &sorted_token_ids, &experts_ids,
    topk, is_prefill,
)?;

// NVFP4 quantized experts (Blackwell SM100+)
let output = moe::moe_gemm_nvfp4(
    &input, &weights, &scales,
    global_scale, &topk_weights,
    &sorted_token_ids, &experts_ids,
    topk, is_prefill,
)?;

// GGUF/ISQ quantized experts
let output = moe::moe_gemm_gguf(
    &input, &weights_qtensor, &topk_weights,
    &sorted_token_ids, &experts_ids,
    topk, is_prefill, dtype,
)?;
```

Use `topk::topk_softmax` for fused Top-K routing:

```rust
let (topk_weights, topk_indices) = attention_rs::topk::topk_softmax(&router_logits, topk)?;
```

### FP8 Linear

Block-scaled FP8 (E4M3) matrix multiplication with automatic dispatch:

```rust
use attention_rs::fp8_linear::fp8_matmul;

// Auto-selects: FlashInfer (sm90 decode) > CUTLASS (sm90+) > fallback
let output = fp8_matmul(
    &input,           // [M, K] F16/BF16
    &weight,          // [N, K] U8 (FP8 E4M3)
    &weight_scale,    // [N/block, K/block] F32
    cutlass_scale,    // optional pre-transposed scales
    &[128, 128],      // block_size
    is_prefill,
)?;
```

### MXFP4 Linear

MX FP4 with E8M0 block scales. On Blackwell (SM100+) with `cutlass` feature,
uses hardware FP4 tensor cores; otherwise uses WMMA software dequant.

```rust
use attention_rs::mxfp4_linear::{mxfp4_matmul, MXFP4_BLOCK_SIZE};

let output = mxfp4_matmul(
    &input,   // [M, K] F16/BF16
    &weight,  // [N, K/2] U8 (packed FP4 nibbles)
    &scale,   // [N, K/32] U8 (E8M0 scales)
    bias,     // Optional [N]
    is_prefill,
)?;
```

### NVFP4 Linear

NVIDIA FP4 (E2M1) with per-block FP8 scales and global scale factors.
Requires Blackwell SM100+ for the hardware CUTLASS/FlashInfer paths.

```rust
use attention_rs::nvfp4_linear::{nvfp4_matmul, swizzle_nvfp4_weight_scales};

// Pre-swizzle scales once at model load time (CUTLASS layout)
let swizzled = swizzle_nvfp4_weight_scales(&scale, n, k)?;

let output = nvfp4_matmul(
    &input,               // [M, K] F16/BF16
    &weight,              // [N, K/2] U8 (packed FP4)
    &scale,               // [N, K/16] U8 (FP8 E4M3 block scales)
    weight_global_scale,  // f32
    input_scale,          // f32
    bias,                 // Optional [N]
    is_prefill,
    Some(&swizzled),      // pre-swizzled scales
)?;
```

### Gated Delta Net

Custom kernels for Qwen 3.5's linear attention layers (Gated Delta Net
architecture). Provides causal conv1d, delta-rule recurrence, gated RMSNorm,
and L2 normalization.

```rust
use attention_rs::gdn;

// Causal Conv1d forward
let output = gdn::causal_conv1d_fwd(
    &input, &weight, bias.as_ref(), &conv_state,
    /*has_initial_state=*/ false, cu_seqlens.as_ref(),
)?;

// Delta rule recurrence
let output = gdn::gated_delta_rule_recurrence(
    &q, &k, &v, &beta, &output_state,
    head_first, cu_seqlens.as_ref(),
)?;

// Gated RMSNorm + SiLU + element-wise multiply
let output = gdn::gated_rmsnorm_silu_mul(
    &input, &residual, &weight, /*eps=*/ 1e-6,
)?;
```

### Mamba Cache

Slot-based state management for Mamba / GDN convolutional and recurrent
states. Handles dynamic batch sizes with sequence-to-slot mapping.

```rust
use attention_rs::mamba_cache::MambaCache;

let mut cache = MambaCache::new(
    num_gdn_layers,
    max_batch_size,
    conv_state_shape,    // [d_conv, kernel_size - 1]
    recurrent_state_shape, // [num_heads, v_head_dim, k_head_dim]
    dtype,
    &device,
)?;

// Allocate slots for active sequences
let slots = cache.ensure_slots_for_sequences(&sequence_ids)?;

// Access state tensors for a specific layer
let conv_state = cache.conv_state(gdn_layer_idx);
let recurrent_state = cache.recurrent_state(gdn_layer_idx);
```

### Multi-Head Latent Attention

Paged attention and FlashInfer-accelerated MLA for DeepSeek-style models
with compressed KV representations.

```rust
use attention_rs::mla;

// Write compressed KV into paged cache
mla::concat_and_cache_mla(
    &ckv, &k_pe, &ckv_cache, &kpe_cache, &slot_mapping,
)?;

// Decode with paged MLA
let output = mla::mla_paged_decode(
    &query, &ckv_cache, &kpe_cache,
    &block_tables, &context_lens,
    max_context_len, num_heads, head_dim,
    kv_lora_rank, qk_rope_head_dim, softmax_scale,
)?;
```

### Fused RoPE

Fused rotary position embeddings that eliminate the separate index-select
kernel, reducing memory traffic and kernel launch overhead.

```rust
use attention_rs::fused_rope::FusedRope;

// In-place RoPE application
FusedRope::apply_inplace(
    &mut query, &mut key, &cos_cache, &sin_cache, &positions,
)?;

// With partial rotation (only first `rot_dim` dimensions)
FusedRope::apply_inplace_partial(
    &mut query, &mut key, &cos_cache, &sin_cache, &positions,
    rot_dim,
)?;
```

### GPU Sampling

Fused Top-K / Top-P / temperature sampling in a single CUDA kernel launch:

```rust
use attention_rs::sampler::Sampler;

let sampler = Sampler::new();
let token_ids: Vec<u32> = sampler.sample_cuda(
    &logits,       // [batch, vocab_size]
    top_k,         // e.g. 50
    top_p,         // e.g. 0.95
    temperature,   // e.g. 0.7
    seed,          // random seed
)?;
```

---

## Utility modules

| Module | Description |
|--------|-------------|
| `cache` | `swap_blocks` / `clear_blocks` for KV-cache block management |
| `mask` | GPU causal mask generation with optional sliding window |
| `sort` | `ArgSortOp` trait -- GPU-accelerated argsort / sort |
| `ops` | `NonZeroOp`, `SplitOp`, `BincountOp` tensor extension traits |
| `silu_and_mul` | Fused SiLU-and-multiply for gated FFN layers |
| `swiglu` | GPT-OSS SwiGLU with alpha scaling and value clamping |
| `scale_update` | Dynamic FP8 KV-cache scale tracking |
| `compat` | Cross-fork compatibility helpers |
| `workspace` | GPU workspace allocation for FlashInfer / CUTLASS |

---

## Platform support

| GPU family | Compute capability | Supported features |
|------------|-------------------|-------------------|
| NVIDIA V100 | SM70 | Paged attention, MoE, RoPE, software FP8 |
| NVIDIA A100 | SM80 | All CUDA features, with flashinfer/flashattn |
| NVIDIA H100 | SM90 | + CUTLASS FP8, FlashInfer FP8 blockscale, software MXFP4/NVFP4 |
| NVIDIA B100/B200 | SM100 | + Hardware MXFP4/NVFP4, CUTLASS Blackwell |
| Apple M1/M2/M3/M4 | Metal 3 | Paged attention, MoE, MXFP4 (WIP), software FP8, RoPE, GDN |

---

## Integration with upstream Candle

`attention-rs` can be integrated into the Candle workspace as an optional
dependency. For example, to use the MoE kernels from `candle-nn`:

```toml
# candle/Cargo.toml (workspace root)
[workspace.dependencies]
attention-rs = { version = "0.5", default-features = false }

# candle/candle-nn/Cargo.toml
[dependencies]
attention-rs = { workspace = true, optional = true }

[features]
attention-moe = ["dep:attention-rs", "attention-rs/candle-upstream-core-only"]
```

The `candle-upstream-core-only` feature avoids circular dependencies by
depending only on `candle-core` (not `candle-nn`).

---

> 💡 **Used in [vllm.rs](https://github.com/guoqingbao/vllm.rs) and [candle-vllm](https://github.com/EricLBuehler/candle-vllm)**

## License

MIT