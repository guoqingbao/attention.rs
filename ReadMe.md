# 🚀 Attention.rs: High-Performance LLM Attention & Ops

> Efficient, cross-platform optimized kernels and operations for LLM inference in Rust, built on [Candle](https://github.com/huggingface/candle).

---

## 🔍 Overview

`attention.rs` is a collection of high-performance CUDA and Metal kernels designed for Large Language Model (LLM) inference. It provides the foundational operations required for Rust LLM inference engines like [vllm.rs](https://github.com/guoqingbao/vllm.rs) and [candle-vllm](https://github.com/EricLBuehler/candle-vllm).

### 🌟 Key Features

- ✅ **Native Flash Attention**: Flash-Attention-v2 style kernels supporting from SM70 to SM121 (V100 to Blackwell), with SM-specific Tensor Core optimizations and multiple KV cache quantization (FP8, Turboquant) modes.
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

### 1. Native Flash Attention

High-performance Flash-Attention-v2 style kernels with broad hardware and KV cache format support.

**Hardware Support (SM70–SM121):**

| SM | GPU | MMA Instruction | Data Type |
|----|-----|----------------|-----------|
| SM70 | V100 | `m8n8k4` Tensor Core MMA (with warp-shuffle redistribution) | FP16 |
| SM75 | T4, RTX 20xx | `m16n8k8` Tensor Core MMA (2× iteration) | FP16 |
| SM80–SM89 | A100, A10G, L40, RTX 30xx/40xx | `m16n8k16` Tensor Core MMA | BF16 |
| SM90 | H100, H200 | `m16n8k16` Tensor Core MMA + cp.async | BF16 |
| SM100–SM121 | B100, B200, GB200 | `m16n8k16` Tensor Core MMA + cp.async | BF16 |

**KV Cache Formats:**

| Format | Description | Compression | Kernels |
|--------|-------------|-------------|---------|
| BF16/FP16 | Unquantized | 1× | Prefill, Decode, Split-K Decode |
| FP8 (E4M3) | 8-bit floating point | 2× | Prefill, Decode, Split-K Decode |
| Turbo8 | FP8 K + 4-bit V (Walsh-Hadamard) | 2.6× | Prefill (TQ4), Decode, Split-K Decode |
| Turbo4 | 4-bit K + 4-bit V (Walsh-Hadamard) | 3.7× | Prefill (TQ4), Decode, Split-K Decode |
| Turbo3 | 3-bit K + 4-bit V (Walsh-Hadamard) | 4.7× | Prefill (TQ3), Decode, Split-K Decode |

All KV cache formats have dedicated Split-K decode kernels (triggered at 1024+ context length) for long-sequence performance, sharing a common float-precision reduce kernel.

**Features:** Softcapping, sliding window, GQA/MQA, paged block tables, causal masking.

### 2. Paged Attention (Legacy)
- **Cross-platform**: CUDA (V100, A100, H100) & Metal (Apple Silicon).
- **Features**: Softcapping, ALiBi slopes, sliding window, GQA/MQA.
- **Optimization**: Paged KV cache reduces memory fragmentation.

### 3. Mixture of Experts (MoE)
- Supports standard (F16/BF16) and FP8 quantized weights.
- GGUF/ISQ support for quantized experts.
- Optimized for both high-throughput prefill and low-latency decoding.

### 4. Gated Delta Net (GDN) & Mamba
- Custom kernels for Causal Conv1d and Delta Rule recurrence.
- `MambaCache` for managing per-sequence convolution and recurrent states.
- Optimized for hybrid architectures like Qwen 3.5.

### 5. Fused Operations
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
- `flash`: Enable native Flash Attention kernels (SM70–SM121, all KV cache formats).
- `metal`: Enable Metal kernels for Apple Silicon.
- `flashattn`: Enable Flash Attention (Dao-AILab) integration.
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
