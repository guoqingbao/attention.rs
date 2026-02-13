# Fused RoPE (Rotary Position Embedding)

Rotary Position Embedding (RoPE) is the standard method for encoding positional information in modern LLMs (Llama, Mistral, Qwen). `attention-rs` provides a **fused** implementation that is significantly faster than standard implementations.

## Why "Fused"?
In a standard implementation, applying RoPE involves:
1.  Selecting the correct rows from a global `cos`/`sin` table based on token positions (`index_select`).
2.  Multiplying the Q/K tensors by these tables.

This requires two kernel launches and intermediate memory. The **Fused RoPE** kernel in `attention-rs` does both in a single pass, reading positions and the global table directly within the RoPE computation.

## Integration in `vllm-rs`

In `vllm-rs`, the `ApplyRotaryEmbedding` trait is implemented using `FusedRope` for the CUDA backend.

```rust
use attention_rs::fused_rope::FusedRope;

impl ApplyRotaryEmbedding for RotaryEmbedding {
    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        // ... (check for partial rotary or CPU fallback) ...

        // Full rotary embedding - use fused kernel with position selection
        // Pass full cos/sin tables and positions - kernel selects on-the-fly
        // This eliminates the index_select kernel launch!
        FusedRope::apply_inplace(q, k, &self.cos, &self.sin, positions, self.is_rope_i)?;
        
        Ok(None) // In-place update
    }
}
```

## Detailed Example: Manual Usage

```rust
use candle_core::{Device, Tensor, DType};
use attention_rs::fused_rope::FusedRope;

let device = Device::new_cuda(0)?;

// 1. Prepare Q and K [batch, heads, seq, dim]
let q = Tensor::randn(0.0, 1.0, (1, 32, 128, 128), &device)?;
let k = Tensor::randn(0.0, 1.0, (1, 8, 128, 128), &device)?; 

// 2. Prepare Global Tables (precomputed for max_seq_len)
let cos = Tensor::randn(0.0, 1.0, (4096, 64), &device)?;
let sin = Tensor::randn(0.0, 1.0, (4096, 64), &device)?;

// 3. Current Token Positions
let positions = Tensor::from_slice(&[0, 1, 2, 3], (4,), &device)?;

// 4. Apply RoPE In-place
FusedRope::apply_inplace(&q, &k, &cos, &sin, &positions, false)?;
```

## Technical Specification
- **CUDA**: Uses a vectorized load/store approach to maximize memory throughput.
- **Metal**: Implements the same fusion logic using SIMD-group operations for Apple Silicon.
- **Interleaved Support**: Supports both standard Llama-style and GPT-NeoX style (interleaved) layouts.
