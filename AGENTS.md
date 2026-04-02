# attention.rs: Comprehensive Project Structure & Component Guide

`attention.rs` is a high-performance library of optimized kernels for LLM inference, built on the [Candle](https://github.com/huggingface/candle) framework. It is designed to power the [vllm.rs](https://github.com/guoqingbao/vllm.rs) engine and [candle-vllm](https://github.com/EricLBuehler/candle-vllm).

## Directory Structure

- **`docs/`**: Feature-specific documentation (RoPE, MoE, PagedAttention, etc.).
- **`src/`**: Rust source code.
    - **`kernels/`**: CUDA C++ kernels and Rust FFI.
    - **`metal-kernels/`**: Metal Shading Language kernels and Metal glue.
- **`Cargo.toml`**: Project configuration, including features like `cuda`, `metal`, `cutlass`, and `flashinfer`.

## Core Components & Modules

### 1. Attention Engines
- **`src/paged_attention.rs`**: The primary API for memory-efficient attention. Handles `PagedAttention` struct which dispatches to CUDA or Metal kernels.
- **`src/flashinfer.rs`**: Integration with the FlashInfer library for state-of-the-art prefill and decoding performance on NVIDIA GPUs.
- **`src/mask.rs`**: Efficient generation of causal and custom attention masks.

### 2. Positional Embeddings
- **`src/fused_rope.rs`**: Implements Fused Rotary Position Embeddings. Fuses the `index_select` of `cos`/`sin` tables with the rotation computation into a single kernel.

### 3. Mixture of Experts (MoE)
- **`src/moe.rs`**: High-level MoE logic. Provides `moe_gemm` (F16/BF16) and `moe_gemm_fp8` for quantized experts.
- **`src/topk.rs`**: Optimized `topk_softmax` kernel for expert routing.
- **`src/sort.rs`**: `ArgSortOp` trait extending `Tensor` with GPU-accelerated sorting, critical for grouping tokens by expert.

### 4. Quantization & FP8
- **`src/fp8_linear.rs`**: Interface for FP8 matrix multiplications.
- **`src/ops.rs`**: General-purpose optimized ops like `SplitOp`, `NonZeroOp`, and quantization kernels.
- **`src/scale_update.rs`**: Utilities for updating quantization scales.

### 5. Linear Attention & Mamba/GDN
- **`src/gdn.rs`**: Implementation of Gated Delta Net (GDN) and Mamba operations (Causal Conv1d, Gated Delta Rule).
- **`src/mamba_cache.rs`**: `MambaCache` manager for per-sequence convolution and recurrence states.
- **Reference**: See `vllm.rs/src/models/layers/deltanet.rs` for a complete implementation of a hybrid Qwen3.5 layer.

### 6. Sampling
- **`src/sampler.rs`**: Fully fused GPU sampling. Handles Softmax + Top-K + Top-P + Random Selection in a single kernel launch.

### 7. Utilities
- **`src/cache.rs`**: Paged KV cache management (swapping, copying).
- **`src/cuda_utils.rs`**: CUDA-specific utilities like Compute Capability (`sm_version`) detection.
- **`src/lib.rs`**: Public API surface and core metadata structures like `InputMetadata`.

## Key Metadata: `InputMetadata`

`InputMetadata` is the "brain" of most operations in `attention-rs`. It tells the kernels:
- `is_prefill`: Whether we are in the prompt processing phase.
- `slot_mapping`: Where each token belongs in the physical KV cache.
- `block_tables`: Which physical blocks are assigned to each sequence (for decoding).
- `seqlens`: Individual sequence lengths in a packed batch.

## Developer & Sub-Agent Guidance

1.  **Backend Dispatch**: Always check `tensor.device()` and provide both CUDA and CPU (or Metal) paths.
2.  **Contiguity**: Most optimized kernels require contiguous tensors. Ensure `tensor.is_contiguous()` or call `.contiguous()?`.
3.  **DType Safety**: Kernels are often implemented for specific dtypes (F16, BF16, F32). Use `match tensor.dtype()` to dispatch correctly.
4.  **FFI Boundaries**: CUDA kernels are exposed via `kernels::ffi`. When adding new kernels, update the FFI layer and ensure proper pointer management using `storage_and_layout()`.
