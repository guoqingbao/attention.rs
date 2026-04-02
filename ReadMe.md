# 🚀 Attention.rs: High-Performance LLM Attention & Ops

> Efficient, cross-platform optimized kernels and operations for LLM inference in Rust, built on [Candle](https://github.com/huggingface/candle).

---

## 🔍 Overview

`attention.rs` is a collection of high-performance CUDA and Metal kernels designed for Large Language Model (LLM) inference. It provides the foundational operations required for Rust LLM inference engines like [vllm.rs](https://github.com/guoqingbao/vllm.rs) and [candle-vllm](https://github.com/EricLBuehler/candle-vllm).

### 🌟 Key Features

- ✅ **Paged Attention**: Memory-efficient KV caching with paged allocation. Supports CUDA and Metal.
- ✅ **Chunked Prefill**: Optimized attention for long sequences, avoiding memory blowup.
- ✅ **Mixture of Experts (MoE)**: Fused MoE kernels for both prefill (WMMA) and decoding (GEMV), supporting standard and FP8 weights.
- ✅ **Fused RoPE**: High-performance rotary position embeddings that fuse index selection and computation.
- ✅ **FP8 Support**: Optimized FP8 matrix multiplication with optional CUTLASS integration (SM90+).
- ✅ **Gated Delta Net (GDN)**: Specialized support for Qwen 3.5's linear attention, including Mamba-style caches.
- ✅ **GPU Sampling**: Accelerated Top-K, Nucleus (Top-P), and temperature-based sampling.
- ✅ **FlashInfer Integration**: Native support for FlashInfer attention kernels.

---

## 🛠️ Supported Components

### 1. Paged Attention
- **Cross-platform**: CUDA (V100, A100, H100) & Metal (Apple Silicon).
- **Features**: Softcapping, ALiBi slopes, sliding window, GQA/MQA.
- **Optimization**: Paged KV cache reduces memory fragmentation.

### 2. Mixture of Experts (MoE)
- Supports standard (F16/BF16) and FP8 quantized weights.
- GGUF/ISQ support for quantized experts.
- Optimized for both high-throughput prefill and low-latency decoding.

### 3. Gated Delta Net (GDN) & Mamba
- Custom kernels for Causal Conv1d and Delta Rule recurrence.
- `MambaCache` for managing per-sequence convolution and recurrent states.
- Optimized for hybrid architectures like Qwen 3.5.

### 4. Fused Operations
- **FusedRoPE**: Fuses position selection and rotary embedding.
- **GatedRMSNorm**: SiLU-gated RMS normalization.
- **L2Norm**: Optimized L2 normalization for linear attention.

---

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
attention-rs = { git = "https://github.com/guoqingbao/attention.rs" }
```

### Features

- `cuda`: Enable CUDA kernels and optimizations.
- `metal`: Enable Metal kernels for Apple Silicon.
- `flashattn`: Enable Flash Attention integration.
- `flashinfer`: Enable FlashInfer integration.
- `cutlass`: Enable CUTLASS-optimized FP8 kernels (requires CUDA).

---

## 📖 Documentation

Detailed documentation for each component can be found in the [docs/](docs/) folder:

- [Paged Attention](docs/paged_attention.md)
- [Mixture of Experts (MoE)](docs/moe.md)
- [Fused RoPE](docs/fused_rope.md)
- [FP8 Operations](docs/fp8_ops.md)
- [Gated Delta Net & Mamba](docs/mamba_gdn.md)
- [Sampling](docs/sampling.md)

---

## 📄 License

This project is licensed under the **MIT License**.

---

> 💡 **Used in [vllm.rs](https://github.com/guoqingbao/vllm.rs) and [candle-vllm](https://github.com/EricLBuehler/candle-vllm)**
