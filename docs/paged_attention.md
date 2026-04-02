# Paged Attention & Chunked Prefill

Paged Attention is the core mechanism for memory-efficient LLM inference. It allows storing Key-Value (KV) caches in non-contiguous memory pages (blocks), significantly reducing memory fragmentation and enabling higher throughput.

## Core Concepts

### 1. Paged KV Cache
Instead of allocating a single contiguous buffer for each sequence, memory is divided into **blocks** (e.g., 16 tokens). Blocks are assigned to sequences as they grow, similar to virtual memory in operating systems.

### 2. Chunked Prefill
For very long prompts (e.g., 32k tokens), the "prefill" phase can exceed GPU memory if processed as a single chunk. `attention-rs` supports **Chunked Prefill**, which splits the prompt into smaller query chunks (e.g., 512 or 1024 tokens) while attending to the previously cached KV blocks.

## Metadata: `InputMetadata`
The behavior of the attention kernels is governed by `InputMetadata`:
- `is_prefill`: If `true`, runs the prefill kernel (optimized for $S > 1$). If `false`, runs the decoding kernel ($S = 1$).
- `slot_mapping`: Maps each token in the current query to a specific "slot" in the physical KV cache.
- `block_tables`: For decoding, maps each sequence to its list of allocated blocks.
- `cu_seqlens_q`: Cumulative sequence lengths for the query, used to identify sequence boundaries in a packed batch.
- `max_context_len`: The maximum number of tokens in any sequence's history.

## Integration Example

In a real-world engine like `vllm-rs`, `PagedAttention` is integrated into a higher-level `Attention` layer.

```rust
use candle_core::{Tensor, Result, DType};
use attention_rs::{PagedAttention, InputMetadata};

pub struct Attention {
    q_proj: TensorParallelColumnLinear,
    k_proj: TensorParallelColumnLinear,
    v_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    attn: PagedAttention,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn forward(
        &self,
        xs: &Tensor,
        rotary_emb: &Option<Arc<dyn ApplyRotaryEmbedding>>,
        input_metadata: &InputMetadata,
        cache: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;

        // 1. Projections
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // 2. Reshape for attention
        let q = q.reshape((1, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((1, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((1, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        // 3. Apply rotary embeddings (using FusedRope internally)
        let (q, k) = if let Some(rotary_emb) = &rotary_emb {
            rotary_emb.apply_rotary_emb_qkv(&q, &k, positions)?
                .unwrap_or((q, k))
        } else {
            (q, k)
        };

        // 4. Execute paged attention
        let y = self.attn.forward(
            &q, &k, &v,
            None, // attention_mask
            cache.map(|(k_c, _)| k_c.clone()),
            cache.map(|(_, v_c)| v_c.clone()),
            input_metadata,
            None, // softcapping
        )?.reshape((seq_len, ()))?;

        self.o_proj.forward(&y)
    }
}
```

## Backend Implementation
- **CUDA**: `paged_attention_v2.cu` uses a thread-block level reduction to handle variable-length context efficiently.
- **Metal**: `pagedattention.metal` provides a high-performance implementation for Apple Silicon, optimized for SIMD-group reductions.
