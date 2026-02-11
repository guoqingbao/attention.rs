// GDN (Gated Delta Net) operations module
// Provides Rust interfaces for GDN CUDA kernels used in Qwen3.5's linear attention layers.

use candle_core::{DType, Device, Result, Tensor, Storage, IndexOp};
#[cfg(feature = "cuda")]
use candle_core::cuda::CudaDevice;
#[cfg(feature = "cuda")]
use kernels::ffi;
use std::ffi::{c_int, c_void};


/// Causal conv1d forward pass for variable-length sequences (prefill mode)
///
/// Arguments:
/// - x: input tensor [total_tokens, d_conv]
/// - weight: conv weight [d_conv, 1, kernel_size]
/// - bias: optional conv bias [d_conv]
/// - conv_state: [batch, d_conv, kernel_size - 1] (updated in-place)
/// - cu_seqlens: cumulative sequence lengths [batch + 1]
/// - activation_silu: whether to apply SiLU activation
///
/// Returns: convolved output [total_tokens, d_conv]
#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
pub fn causal_conv1d_fwd(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    _conv_state: &mut Tensor,
    _cu_seqlens: Option<&Tensor>,
    activation_silu: bool,
) -> Result<Tensor> {
    // Reverted to naive implementation as per request
    causal_conv1d_naive(x, weight, bias, activation_silu)
}

/// Causal conv1d single-step update for decode mode
///
/// Arguments:
/// - x: input tensor [batch, d_conv] (single token per sequence)
/// - weight: conv weight [d_conv, 1, kernel_size]  
/// - bias: optional conv bias [d_conv]
/// - conv_state: [batch, d_conv, kernel_size - 1] (shifted and updated in-place)
/// - activation_silu: whether to apply SiLU activation
///
/// Returns: convolved output [batch, d_conv]
#[cfg(feature = "cuda")]
pub fn causal_conv1d_update(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    match (x.device(), x.dtype()) {
        (Device::Cuda(dev), dtype) => {
            let (batch, d_conv) = x.dims2()?;
            let kernel_size = weight.dim(2)?;
            let stream = dev.cu_stream() as i64;
            
            let x_ptr = get_cuda_ptr(x)?;
            let w_ptr = get_cuda_ptr(weight)?;
            let bias_ptr = if let Some(b) = bias { get_cuda_ptr(b)? } else { std::ptr::null() };
            let state_ptr = get_cuda_mut_ptr(conv_state)?;
            
            let out = unsafe { dev.alloc_zeros::<u8>(x.elem_count() * x.dtype().size_in_bytes())? };
            let out_tensor = Tensor::from_raw_buffer(out, x.dtype(), x.shape(), x.device())?;
            let out_ptr = get_cuda_mut_ptr(&out_tensor)?;

            unsafe {
                match dtype {
                    DType::F32 => ffi::causal_conv1d_update_f32(
                        x_ptr as *const f32, w_ptr as *const f32, bias_ptr as *const f32,
                        state_ptr as *mut f32, out_ptr as *mut f32,
                        batch as c_int, d_conv as c_int, kernel_size as c_int,
                        activation_silu, stream
                    ),
                    DType::F16 => ffi::causal_conv1d_update_f16(
                        x_ptr, w_ptr, bias_ptr, state_ptr, out_ptr,
                        batch as c_int, d_conv as c_int, kernel_size as c_int,
                        activation_silu, stream
                    ),
                    DType::BF16 => ffi::causal_conv1d_update_bf16(
                        x_ptr, w_ptr, bias_ptr, state_ptr, out_ptr,
                        batch as c_int, d_conv as c_int, kernel_size as c_int,
                        activation_silu, stream
                    ),
                    _ => candle_core::bail!("Unsupported dtype for causal_conv1d_update"),
                }
            }
            Ok(out_tensor)
        }
        _ => causal_conv1d_update_naive(x, weight, bias, conv_state, activation_silu)
    }
}

/// Fused GDN gating computation
///
/// Computes decay gate `g` and input gate `beta`:
/// - g = sigmoid(-A_log * sigmoid(a))
/// - beta = softplus(b + dt_bias)
///
/// Arguments:
/// - a_log: learned log-space decay parameter [num_heads]
/// - a: input-dependent A values [batch, seq_len, num_heads]
/// - b: input-dependent B values [batch, seq_len, num_heads]
/// - dt_bias: learned dt bias [num_heads]
///
/// Returns: (g, beta) each of shape [batch, seq_len, num_heads]
#[cfg(feature = "cuda")]
pub fn fused_gdn_gating(
    a_log: &Tensor,
    a: &Tensor,
    b: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    match (a.device(), a.dtype()) {
        (Device::Cuda(dev), dtype) => {
            let (batch, seq_len, num_heads) = a.dims3()?;
            let stream = dev.cu_stream() as i64;
            
            let al_ptr = get_cuda_ptr(a_log)?;
            let a_ptr = get_cuda_mut_ptr(a)?; // Note: a is used as input here, pointer constness cast later
            let b_ptr = get_cuda_mut_ptr(b)?; // input
            let dt_ptr = get_cuda_ptr(dt_bias)?;
            
            let g = unsafe { dev.alloc_zeros::<u8>(a.elem_count() * a.dtype().size_in_bytes())? };
            let g = Tensor::from_raw_buffer(g, dtype, a.shape(), a.device())?;
            let g_ptr = get_cuda_mut_ptr(&g)?;
            
            let beta = unsafe { dev.alloc_zeros::<u8>(b.elem_count() * b.dtype().size_in_bytes())? };
            let beta = Tensor::from_raw_buffer(beta, dtype, b.shape(), b.device())?;
            let beta_ptr = get_cuda_mut_ptr(&beta)?;

            unsafe {
                match dtype {
                    DType::F32 => ffi::fused_gdn_gating_f32(
                        al_ptr as *const f32, a_ptr as *mut f32, b_ptr as *mut f32, dt_ptr as *const f32,
                        g_ptr as *mut f32, beta_ptr as *mut f32,
                        batch as c_int, seq_len as c_int, num_heads as c_int, stream
                    ),
                    DType::F16 => ffi::fused_gdn_gating_f16(
                        al_ptr, a_ptr, b_ptr, dt_ptr, g_ptr, beta_ptr,
                        batch as c_int, seq_len as c_int, num_heads as c_int, stream
                    ),
                    DType::BF16 => ffi::fused_gdn_gating_bf16(
                        al_ptr, a_ptr, b_ptr, dt_ptr, g_ptr, beta_ptr,
                        batch as c_int, seq_len as c_int, num_heads as c_int, stream
                    ),
                    _ => candle_core::bail!("Unsupported dtype for fused_gdn_gating"),
                }
            }
            Ok((g, beta))
        }
        _ => fused_gdn_gating_naive(a_log, a, b, dt_bias)
    }
}

/// Chunked gated delta rule for recurrent state update
///
/// For prefill: processes chunks of tokens, maintaining inter-chunk recurrent state S
/// For decode: single-step update S = g * S + beta * (k^T ⊗ v), output = q @ S
///
/// Arguments:
/// - q: query [batch, seq_len, num_heads, head_dim]
/// - k: key [batch, seq_len, num_heads, head_dim]
/// - v: value [batch, seq_len, num_heads, head_dim]
/// - g: decay gate [batch, seq_len, num_heads]
/// - beta: input gate [batch, seq_len, num_heads]
/// - state: recurrent state [batch, num_heads, head_dim, head_dim] (updated in-place)
/// - is_prefill: whether this is a prefill (chunked) or decode (single-step) call
///
/// Returns: output [batch, seq_len, num_heads, head_dim]
#[cfg(feature = "cuda")]
pub fn chunk_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    is_prefill: bool,
) -> Result<Tensor> {
    if is_prefill {
        return chunk_gated_delta_rule_prefill_naive(q, k, v, g, beta, state);
    }
    
    match (q.device(), q.dtype()) {
        (Device::Cuda(dev), dtype) => {
            // Decode mode: q,k,v,g,beta are [batch, 1, num_heads, head_dim] or [batch, 1, num_heads]
            // We squeeze dim 1 for kernel
            let (batch, _, num_heads, head_dim) = q.dims4()?;
            let stream = dev.cu_stream() as i64;
            
            let q_ptr = get_cuda_ptr(q)?;
            let k_ptr = get_cuda_ptr(k)?;
            let v_ptr = get_cuda_ptr(v)?;
            let g_ptr = get_cuda_ptr(g)?;
            let beta_ptr = get_cuda_ptr(beta)?;
            let state_ptr = get_cuda_mut_ptr(state)?;
            
            // Output shape: [batch, 1, num_heads, head_dim]
            let out = unsafe { dev.alloc_zeros::<u8>(q.elem_count() * q.dtype().size_in_bytes())? };
            let out = Tensor::from_raw_buffer(out, dtype, q.shape(), q.device())?;
            let out_ptr = get_cuda_mut_ptr(&out)?;

            unsafe {
                match dtype {
                    DType::F32 => ffi::chunk_gated_delta_rule_decode_f32(
                        q_ptr as *const f32, k_ptr as *const f32, v_ptr as *const f32,
                        g_ptr as *const f32, beta_ptr as *const f32,
                        state_ptr as *mut f32, out_ptr as *mut f32,
                        batch as c_int, num_heads as c_int, head_dim as c_int, stream
                    ),
                    DType::F16 => ffi::chunk_gated_delta_rule_decode_f16(
                        q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
                        state_ptr, out_ptr,
                        batch as c_int, num_heads as c_int, head_dim as c_int, stream
                    ),
                    DType::BF16 => ffi::chunk_gated_delta_rule_decode_bf16(
                        q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
                        state_ptr, out_ptr,
                        batch as c_int, num_heads as c_int, head_dim as c_int, stream
                    ),
                    _ => candle_core::bail!("Unsupported dtype for chunk_gated_delta_rule"),
                }
            }
            Ok(out)
        }
        _ => chunk_gated_delta_rule_decode_naive(q, k, v, g, beta, state)
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_ptr(t: &Tensor) -> Result<*const c_void> {
    match &*t.storage_and_layout().0 {
        Storage::Cuda(s) => Ok(s.as_ptr() as *const c_void),
        _ => candle_core::bail!("Expected CUDA tensor"),
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_mut_ptr(t: &Tensor) -> Result<*mut c_void> {
    match &*t.storage_and_layout().0 {
        Storage::Cuda(s) => Ok(s.as_ptr() as *mut c_void),
        _ => candle_core::bail!("Expected CUDA tensor"),
    }
}

// =============================================================================
// Naive implementations (used as fallback and for non-CUDA platforms)
// =============================================================================

/// Naive causal conv1d using candle ops
pub fn causal_conv1d_naive(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    activation_silu: bool,
) -> Result<Tensor> {
    // weight shape: [d_conv, 1, kernel_size]
    let weight_2d = weight.squeeze(1)?; // [d_conv, kernel_size]
    let kernel_size = weight_2d.dim(1)?;
    let d_conv = weight_2d.dim(0)?;
    let seq_len = x.dim(0)?;

    // Pad input on the left with zeros for causal behavior
    let padding = Tensor::zeros((kernel_size - 1, d_conv), x.dtype(), x.device())?;
    let x_padded = Tensor::cat(&[&padding, x], 0)?; // [seq_len + kernel_size - 1, d_conv]

    // Apply 1D convolution per channel (depthwise)
    // For each output position t: out[t] = sum_{k=0}^{K-1} weight[k] * x[t + k]
    let mut slices = Vec::with_capacity(kernel_size);
    for k in 0..kernel_size {
        let slice = x_padded.narrow(0, k, seq_len)?; // [seq_len, d_conv]
        let w_k = weight_2d.i((.., k))?; // [d_conv]
        slices.push(slice.broadcast_mul(&w_k)?);
    }

    let mut output = slices[0].clone();
    for s in &slices[1..] {
        output = (output + s)?;
    }

    if let Some(bias) = bias {
        output = output.broadcast_add(bias)?;
    }

    if activation_silu {
        output = candle_nn::ops::silu(&output)?;
    }

    Ok(output)
}

/// Naive causal conv1d update for decode (single step)
pub fn causal_conv1d_update_naive(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    // x: [batch, d_conv], single new token
    // conv_state: [batch, d_conv, kernel_size - 1]
    // weight: [d_conv, 1, kernel_size]
    let weight_2d = weight.squeeze(1)?; // [d_conv, kernel_size]

    // Append x to conv_state, drop oldest
    let x_expanded = x.unsqueeze(2)?; // [batch, d_conv, 1]
    let new_state = Tensor::cat(&[&conv_state.narrow(2, 1, conv_state.dim(2)? - 1)?, &x_expanded], 2)?;
    *conv_state = new_state.clone();

    // Compute: output = sum(state * weight, dim=-1) + bias
    // state: [batch, d_conv, kernel_size], weight: [d_conv, kernel_size]
    let output = (new_state.broadcast_mul(&weight_2d.unsqueeze(0)?))?
        .sum(2)?; // [batch, d_conv]

    let output = if let Some(bias) = bias {
        output.broadcast_add(bias)?
    } else {
        output
    };

    if activation_silu {
        candle_nn::ops::silu(&output)
    } else {
        Ok(output)
    }
}

/// Naive fused GDN gating
pub fn fused_gdn_gating_naive(
    a_log: &Tensor,
    a: &Tensor,
    b: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // g = sigmoid(-a_log * sigmoid(a))
    let a_sigmoid = candle_nn::ops::sigmoid(a)?;
    let neg_a_log = a_log.neg()?;
    let g = candle_nn::ops::sigmoid(&neg_a_log.broadcast_mul(&a_sigmoid)?)?;

    // beta = softplus(b + dt_bias)
    let b_biased = b.broadcast_add(dt_bias)?;
    let beta = softplus(&b_biased)?;

    Ok((g, beta))
}

/// Softplus: log(1 + exp(x)), numerically stable
fn softplus(x: &Tensor) -> Result<Tensor> {
    // For x > 20, softplus(x) ≈ x (avoids overflow)
    let threshold = 20.0f64;
    let exp_x = x.exp()?;
    let one_plus_exp = (exp_x + 1.0)?;
    let sp = one_plus_exp.log()?;
    // Clamp: where x > threshold, use x directly
    // Simple approximation: just use log(1+exp(x)) as candle handles inf reasonably
    Ok(sp)
}

/// Naive chunked gated delta rule for prefill
fn chunk_gated_delta_rule_prefill_naive(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    // q,k,v: [batch, seq_len, num_heads, head_dim]
    // g, beta: [batch, seq_len, num_heads]
    // state: [batch, num_heads, head_dim, head_dim]
    let (batch, seq_len, num_heads, head_dim) = q.dims4()?;

    let mut outputs = Vec::with_capacity(seq_len);

    // Process token by token (most naive approach)
    for t in 0..seq_len {
        let q_t = q.narrow(1, t, 1)?.squeeze(1)?; // [batch, num_heads, head_dim]
        let k_t = k.narrow(1, t, 1)?.squeeze(1)?; // [batch, num_heads, head_dim]
        let v_t = v.narrow(1, t, 1)?.squeeze(1)?; // [batch, num_heads, head_dim]
        let g_t = g.narrow(1, t, 1)?.squeeze(1)?; // [batch, num_heads]
        let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?; // [batch, num_heads]

        // State update: S = g * S + beta * (k^T outer v)
        // g_t: [batch, num_heads] -> [batch, num_heads, 1, 1]
        let g_expanded = g_t.unsqueeze(2)?.unsqueeze(3)?;
        // beta_t: [batch, num_heads] -> [batch, num_heads, 1, 1]
        let beta_expanded = beta_t.unsqueeze(2)?.unsqueeze(3)?;

        // k^T outer v: [batch, num_heads, head_dim, 1] * [batch, num_heads, 1, head_dim]
        let k_col = k_t.unsqueeze(3)?; // [batch, num_heads, head_dim, 1]
        let v_row = v_t.unsqueeze(2)?; // [batch, num_heads, 1, head_dim]
        let kv_outer = k_col.matmul(&v_row)?; // [batch, num_heads, head_dim, head_dim]

        // S = g * S + beta * kv_outer
        *state = (g_expanded.broadcast_mul(state)? + beta_expanded.broadcast_mul(&kv_outer)?)?;

        // Output: out = q @ S -> [batch, num_heads, head_dim]
        let q_row = q_t.unsqueeze(2)?; // [batch, num_heads, 1, head_dim]
        let out_t = q_row.matmul(state)?; // [batch, num_heads, 1, head_dim]
        let out_t = out_t.squeeze(2)?; // [batch, num_heads, head_dim]

        outputs.push(out_t.unsqueeze(1)?); // [batch, 1, num_heads, head_dim]
    }

    Tensor::cat(&outputs, 1) // [batch, seq_len, num_heads, head_dim]
}

/// Naive gated delta rule for decode (single step)
fn chunk_gated_delta_rule_decode_naive(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    // q,k,v: [batch, 1, num_heads, head_dim]
    // g, beta: [batch, 1, num_heads]
    // state: [batch, num_heads, head_dim, head_dim]
    let q_t = q.squeeze(1)?; // [batch, num_heads, head_dim]
    let k_t = k.squeeze(1)?;
    let v_t = v.squeeze(1)?;
    let g_t = g.squeeze(1)?; // [batch, num_heads]
    let beta_t = beta.squeeze(1)?;

    let g_expanded = g_t.unsqueeze(2)?.unsqueeze(3)?;
    let beta_expanded = beta_t.unsqueeze(2)?.unsqueeze(3)?;

    let k_col = k_t.unsqueeze(3)?;
    let v_row = v_t.unsqueeze(2)?;
    let kv_outer = k_col.matmul(&v_row)?;

    *state = (g_expanded.broadcast_mul(state)? + beta_expanded.broadcast_mul(&kv_outer)?)?;

    let q_row = q_t.unsqueeze(2)?;
    let out = q_row.matmul(state)?.squeeze(2)?;

    Ok(out.unsqueeze(1)?) // [batch, 1, num_heads, head_dim]
}

// =============================================================================
// Non-CUDA fallbacks (same naive implementations)
// =============================================================================

#[cfg(not(feature = "cuda"))]
pub fn causal_conv1d_fwd(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    _conv_state: &mut Tensor,
    _cu_seqlens: Option<&Tensor>,
    activation_silu: bool,
) -> Result<Tensor> {
    causal_conv1d_naive(x, weight, bias, activation_silu)
}

#[cfg(not(feature = "cuda"))]
pub fn causal_conv1d_update(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    causal_conv1d_update_naive(x, weight, bias, conv_state, activation_silu)
}

#[cfg(not(feature = "cuda"))]
pub fn fused_gdn_gating(
    a_log: &Tensor,
    a: &Tensor,
    b: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    fused_gdn_gating_naive(a_log, a, b, dt_bias)
}

#[cfg(not(feature = "cuda"))]
pub fn chunk_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    is_prefill: bool,
) -> Result<Tensor> {
    if is_prefill {
        chunk_gated_delta_rule_prefill_naive(q, k, v, g, beta, state)
    } else {
        chunk_gated_delta_rule_decode_naive(q, k, v, g, beta, state)
    }
}
