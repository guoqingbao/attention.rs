# Mixture of Experts (MoE)

Mixture of Experts (MoE) is a model architecture that scales capacity by using multiple "expert" networks, only a subset of which are active for any given token. `attention-rs` provides specialized kernels to make this highly efficient.

## The MoE Pipeline

To achieve high performance, `attention-rs` follows an optimized pipeline for MoE layers:

1. **Gating**: A small linear layer computes "expert scores" for each token.
2. **Top-K Softmax**: Selects the top $K$ experts and normalizes their scores.
3. **Token Sorting**: Tokens are grouped by their assigned expert ID using `ArgSortOp`.
4. **Expert GEMM**:
   - Uses `moe_gemm` (F16/BF16) or `moe_gemm_fp8` (FP8).
   - Optimized for both Prefill (throughput) and Decode (latency).

## Integration in `vllm-rs`

In `vllm-rs`, the `FusedMoe` layer uses `attention-rs` for routing and execution:

```rust
use attention_rs::moe;
use attention_rs::topk::topk_softmax;
use attention_rs::sort::ArgSortOp;

pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
    let router_logits = self.gate.forward(&xs)?;

    // 1. Select top experts
    let (mut topk_weights, topk_ids) = topk_softmax(
        &router_logits.to_dtype(DType::F32)?,
        self.num_experts_per_tok,
    )?;

    // 2. Sort tokens by expert ID to maximize GEMM efficiency
    let (expert_ids, sorted_token_ids) = if is_prefill {
        topk_ids.flatten_all()?.sort(true)? // Custom ArgSortOp::sort
    } else {
        topk_ids.flatten_all()?.sort_last_dim(true)?
    };

    // 3. Execute Expert GEMMs (e.g., gate_proj, up_proj, down_proj)
    let gate = moe::moe_gemm(
        &xs, &self.gate_w, &None,
        &sorted_token_ids, &expert_ids,
        self.num_experts_per_tok, is_prefill,
    )?;
    
    // ... MLP logic (e.g., SiLU(gate) * up) ...
    
    let ys = moe::moe_gemm(
        &down_inputs, &self.down_w, &Some(topk_weights),
        &sorted_token_ids, &expert_ids,
        self.num_experts_per_tok, is_prefill,
    )?;
    
    Ok(ys)
}
```

## FP8 MoE Support

For very large models (e.g., DeepSeek), `moe_gemm_fp8` provides block-wise quantized execution:

```rust
let output = moe::moe_gemm_fp8(
    &input,
    &expert_weights_u8,
    &expert_scales_f32,
    &Some(topk_weights),
    &sorted_token_ids,
    &expert_ids,
    num_experts_per_tok,
    block_size_n,
    block_size_k,
    is_prefill,
)?;
```

## Key Functions
- `topk_softmax`: Fused gating and normalization.
- `ArgSortOp`: Trait extending `Tensor` with high-performance `arg_sort` and `sort`.
- `moe_gemm`: Multi-expert GEMM for standard types.
- `moe_gemm_fp8`: Multi-expert GEMM for FP8 quantized experts.
