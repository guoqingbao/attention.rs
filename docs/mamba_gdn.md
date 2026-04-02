# Gated Delta Net & Mamba

The library provides specialized support for **Gated Delta Net (GDN)** and **Mamba** architectures, used in linear attention models like Qwen 3.5. These models replace standard attention with a recurrent/convolutional mechanism.

## Architecture Components

### 1. `MambaCache`
GDN/Mamba requires two types of persistent state:
- **Convolution State**: A sliding window of the last $N$ tokens (typically 4). Shape: `[max_batch, d_conv, kernel_size - 1]`.
- **Recurrent State**: A state matrix updated per-token. Shape: `[max_batch, num_heads, d_k, d_v]`.

`MambaCache` manages these states using **slot-based indexing**, where each sequence is assigned a slot index in the pre-allocated buffers.

### 2. Causal Conv1d
Optimized kernels for 1D convolution:
- `causal_conv1d_fwd`: Batched prefill (variable length) that updates the convolution state.
- `causal_conv1d_update_slots`: Decoding (single token) with slot-based state updates.

### 3. Gated Delta Rule Recurrence
The core linear attention operation:
- `gated_delta_rule_recurrence_varlen`: Batched prefill recurrence using a single CUDA launch for multiple variable-length sequences.
- `gated_delta_rule_decode_slots`: Decoding recurrence with slot-based state updates.

## Integration in `vllm-rs`

In `vllm-rs`, the `GatedDeltaNet` layer manages the hybrid forward pass using `MambaCache`.

### Prefill Pass (Batched)
```rust
// 1. Convolution Step
let mut conv_state = mamba_cache.get_batch_conv_state(layer_idx, seq_slots)?;
let x_conv = gdn::causal_conv1d_fwd(
    &mixed_qkv, &weight, bias,
    &mut conv_state, Some(cu_seqlens), true
)?;
mamba_cache.set_batch_conv_state(layer_idx, seq_slots, &conv_state)?;

// 2. Recurrence Step (Varlen)
let out = gdn::gated_delta_rule_recurrence_varlen(
    &q_scaled, &k, &v, &g, &beta,
    mamba_cache.recurrent_state_mut(layer_idx),
    seq_slots, cu_seqlens
)?;
```

### Decode Pass (Single Token)
```rust
// 1. Convolution Step
let x_conv = gdn::causal_conv1d_update_slots(
    &mixed_qkv, &weight, bias,
    mamba_cache.conv_state_mut(layer_idx),
    seq_slots, true
)?;

// 2. Recurrence Step
let out = gdn::gated_delta_rule_decode_slots(
    &q_scaled, &k, &v, &g, &beta,
    mamba_cache.recurrent_state_mut(layer_idx),
    seq_slots
)?;
```

## Specialized Ops
- **`fused_gdn_gating`**: Fuses the computation of gate $g = -\exp(A_{log}) \cdot \text{softplus}(a + dt_{bias})$ and $\beta = \text{sigmoid}(b)$ values.
- **`gated_rmsnorm_silu_mul`**: Fuses RMSNorm, SiLU gating, and multiplication: `RMSNorm(x) * SiLU(z)`. Used to combine the linear attention output with the gate branch.
- **`l2_norm_last_dim`**: Fused L2 normalization for $Q$ and $K$ tensors, ensuring unit norm before the delta rule update.
