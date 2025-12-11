# 🚀 Paged Attention with Chunked Prefill (GCU)

> Efficient, cross-platform chunked prefill attention for LLM inference with paged KV caching.

---

## 🔍 Overview

This project is a collection of high-performance **paged attention** implementations optimized for **large language model (LLM) inference**, supporting:

- ✅ **Paged Attention** (GPU/Metal-accelerated, memory-efficient key-value caching)
- ✅ **Fused MoE** WWMA fused MoE kernel for prefill and tiling based one for decoding
- ✅ **Chunked Prefill** for long sequences (prefill attention with kvcache, enabling `context cache` and chunked prefill)
- ✅ **Cross-platform**: CUDA (NVIDIA GPUs, `V100`, A100, H100, etc.) & Metal (Apple Silicon M1/M2/M3/M4)
- ✅ **Flexible API-first design** with clean, reusable attention operations
- ✅ **Softcapping**, **alibi slopes**, **sliding window attention**, and multi-query/grouped-query attention

---

## 🌟 Key Features

### 1. **Paged Attention (Memory-Efficient KV Caching)**
- Uses **paged memory allocation** for key/value caches to avoid padding and reduce memory waste.
- Supports **variable-length sequences** and **dynamic batching**.
- Ideal for **long-context LLM inference** (e.g., 32K+ tokens).

### 2. **Chunked Prefill with Paged Attention**
- For **prefill phase** with long sequences, splits attention into chunks (e.g., 4096 tokens) to avoid GPU memory limits.
- Avoids O(N²) memory blowup during prefill with dedicated `prefill_paged_attn` kernels on [CUDA](src/kernels/src/prefill_paged_attn.cu) and [Metal](src/metal-kernels/src/prefill_paged_attn.metal).
- Supports **`cu_seqlens_q`** to track query lengths of multiple sequences (sub-query lengths per sequence).

### 3. **Multi-Backend Support**
| Backend | Supported? | Notes |
|--------|------------|-------|
| CUDA (NVIDIA) | ✅ | Optimized CUDA kernels, supports both `native` and `flash-attn` |
| Metal (Apple M1/M2/M3/M4) | ✅ | Native Metal kernels for Apple Silicon |

### 4. **Cross-Platform API**
- Unified `PagedAttention` API across backends.
- Uses `candle-core`'s `CustomOp` for backend dispatch to backend-specific kernels.
- No need to write separate code for GPU vs. Metal.

### 5. **Advanced Attention Features**
- ✅ Softcapping (for stable attention output)
- ✅ Alibi positional embeddings (for long-range dependencies)
- ✅ Sliding window attention (for efficiency)
- ✅ Multi-Query (MQA) & Grouped-Query Attention (GQA) support
- ✅ Dynamic block size & block table tables

---

## 🛠️ How to Use

### 0. Install Flash Attention on GCU platform
```
dpkg -i flash-attn-gcu_0.1.0-1_amd64.deb
```

### 1. **Cargo.toml Dependencies**

```toml
[dependencies]
metal = { version = "0.27.0", features = ["mps"], optional = true }
candle-core = { git = "https://github.com/guoqingbao/candle.git", version = "0.8.3", rev = "42c9b42" }
candle-flash-attn = { git = "https://github.com/guoqingbao/candle.git", version = "0.8.3", optional = true, rev = "42c9b42" }
attention-rs = {git = "https://github.com/guoqingbao/attention.rs", version="0.1.1", rev = "3058c20" }

# Your project features
[features]
cuda = ["candle-core/cuda", "candle-nn/cuda", "attention-rs/cuda"]
metal = ["candle-core/metal", "candle-nn/metal", "dep:metal", "attention-rs/metal"]
flash-attn = ["cuda", "dep:candle-flash-attn", "attention-rs/flash-attn"]
```

> **Note**: Enable `cuda` or `metal` features based on your target.

### 2. **Basic Usage Example**

```rust
use candle_core::{
    candle_core::{Device, Tensor},
};
use attention_rs::{PagedAttention, InputMetadata},

// Example: 
// Step 1: Setup
let device = Device::new_cuda(0).unwrap(); // or Device::new_metal(0) for Apple Silicon
let num_heads = 32;
let head_size = 128;
let num_kv_heads = 8;
let scale = (head_size as f32).sqrt().recip();

let paged_attn = PagedAttention::new(
    num_heads,
    head_size,
    scale,
    Some(num_kv_heads),
    None,
    device,
    None,
)?;

// Input tensors (batch_size=1, seq_len=1024)
let query = Tensor::randnarrow(&query, 2, 0, 1024)?; // [1, 1024, 32*128]
let key = Tensor::narrow(&key, 2, 0, 1024)?; // [1, 256, 128]
let value = Tensor::narrow(&value, 2, 0, 1024)?; // [1, 256, 128]

// Paged cache setup (for non falsh attn)
let num_blocks = 1024;
let block_size = 64;
let num_kv_heads = 8;

let key_cache = Tensor::zeros(
    (num_blocks,
    num_kv_heads,
    head_size / 8,
    block_size,
    8,
    &device,
)?;
let value_cache = Tensor::zeros(
    num_blocks,
    num_kv_heads,
    head_size,
    block_size,
    &device,
)?;

// Slot mapping: which block to which block
let slot_mapping = Tensor::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7], (8,), &device)?;

// Context lengths per sequence
let context_lens = Tensor::from_slice(&[8], (1,), &device)?;

// Block tables: [num_sequences, max_num_blocks_per_seq]
let block_tables = Tensor::from_slice(&[0, 1], (1, 2), &device)?;

// Metadata for attention
let input_metadata = InputMetadata {
    is_prefill: true,
    slot_mapping,
    block_tables: Some(block_tables),
    context_lens: Some(context_lens),
    cu_seqlens_q: None, // only needed for chunked prefill
    cu_seqlens_k: None,
    max_seqlen_q: 1024,
    max_seqlen_k: 1024,
    max_context_len: 1024,
};

// Step 2: Run attention
let output = paged_attn.forward(
    &query,
    &key,
    &value,
    None, // attention_mask
    Some(key_cache),
    Some(value_cache),
    &input_metadata,
    None, // softcapping
)?;
```

### 3. **Chunked Prefill (for long sequences)**

For very long sequences (e.g., 32K+), use `cu_seqlens_q`** to split query tensor into chunks:

Referce usage in [vllm.rs](https://github.com/guoqingbao/vllm.rs/blob/main/src/core/runner.rs#L392)

```rust
//extra sequence length for query
let cu_seqlens_q = Tensor::from_slice(&[0, 4096, 8192, 12288], (4,), &device)?; // 3 sub sequences of length 4096, 4096, 4096 (or in total of 12288 tokens)
let input_metadata = InputMetadata {
    is_prefill: true,
    slot_mapping: slot_mapping,
    block_tables: Some(block_tables),
    context_lens: Some(context_lens),
    cu_seqlens_q: Some(cu_seqlens_q),
    cu_seqlens_k: None,
    max_seqlen_q: 4096,
    max_seqlen_k: 4096,
    max_context_len: 12288,
};
```

## 📄️ License

This project is licensed under the **MIT License**.


## 📬 Feedback & Contributions

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/guoqingbao/attention.rs/issues).

---

> 💡 **❤️ Used in [vllm.rs](https://github.com/guoqingbao/vllm.rs) and [candle-vllm](https://github.com/EricLBuehler/candle-vllm)**
