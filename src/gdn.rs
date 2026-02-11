// GDN (Gated Delta Net) operations module
// Provides Rust interfaces for GDN CUDA kernels used in Qwen3.5's linear attention layers.

#[cfg(feature = "cuda")]
use candle_core as candle;
use candle_core::{DType, IndexOp, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_core::{Device, Storage};
#[cfg(feature = "cuda")]
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use kernels::ffi;
#[cfg(feature = "cuda")]
use std::ffi::{c_int, c_void};

#[cfg(feature = "cuda")]
fn get_cuda_const_ptr(t: &Tensor) -> Result<*const c_void> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (Storage::Cuda(s), DType::F16) => {
            Ok(*s.as_cuda_slice::<f16>()?.device_ptr() as *const c_void)
        }
        (Storage::Cuda(s), DType::BF16) => {
            Ok(*s.as_cuda_slice::<bf16>()?.device_ptr() as *const c_void)
        }
        (Storage::Cuda(s), DType::F32) => {
            Ok(*s.as_cuda_slice::<f32>()?.device_ptr() as *const c_void)
        }
        _ => candle_core::bail!("Expected CUDA tensor with f16/bf16/f32 dtype"),
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_const_ptr_u32(t: &Tensor) -> Result<*const u32> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let (storage, _) = t.storage_and_layout();
    match &*storage {
        Storage::Cuda(s) => Ok(*s.as_cuda_slice::<u32>()?.device_ptr() as *const u32),
        _ => candle_core::bail!("Expected CUDA u32 tensor"),
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_mut_ptr(t: &Tensor) -> Result<*mut c_void> {
    Ok(get_cuda_const_ptr(t)? as *mut c_void)
}

/// Causal conv1d forward pass for variable-length sequences (prefill mode).
/// Falls back to the reference implementation that also updates per-sequence state.
#[cfg(feature = "cuda")]
pub fn causal_conv1d_fwd(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    cu_seqlens: Option<&Tensor>,
    activation_silu: bool,
) -> Result<Tensor> {
    match (x.device(), x.dtype(), cu_seqlens) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32, Some(cu)) => {
            let (total_tokens, d_conv) = x.dims2()?;
            let kernel_size = weight.dim(2)?;
            if kernel_size > 16 {
                return causal_conv1d_fwd_naive_with_state(
                    x,
                    weight,
                    bias,
                    conv_state,
                    Some(cu),
                    activation_silu,
                );
            }
            let batch = conv_state.dim(0)?;
            let out = Tensor::zeros((total_tokens, d_conv), x.dtype(), x.device())?;
            let cu_u32 = if cu.dtype() == DType::U32 {
                cu.clone()
            } else {
                cu.to_dtype(DType::U32)?
            };

            let x_ptr = get_cuda_const_ptr(x)?;
            let weight_ptr = get_cuda_const_ptr(weight)?;
            let bias_ptr = if let Some(b) = bias {
                get_cuda_const_ptr(b)?
            } else {
                std::ptr::null()
            };
            let state_ptr = get_cuda_mut_ptr(conv_state)?;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let cu_ptr = get_cuda_const_ptr_u32(&cu_u32)?;
            let stream = *dev.cu_stream() as i64;

            unsafe {
                match x.dtype() {
                    DType::F16 => ffi::causal_conv1d_fwd_f16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_ptr,
                        out_ptr,
                        cu_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream,
                    ),
                    DType::BF16 => ffi::causal_conv1d_fwd_bf16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_ptr,
                        out_ptr,
                        cu_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream,
                    ),
                    DType::F32 => ffi::causal_conv1d_fwd_f32(
                        x_ptr as *const f32,
                        weight_ptr as *const f32,
                        bias_ptr as *const f32,
                        state_ptr as *mut f32,
                        out_ptr as *mut f32,
                        cu_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            Ok(out)
        }
        _ => causal_conv1d_fwd_naive_with_state(
            x,
            weight,
            bias,
            conv_state,
            cu_seqlens,
            activation_silu,
        ),
    }
}

/// Causal conv1d single-step update for decode mode.
#[cfg(feature = "cuda")]
pub fn causal_conv1d_update(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    match (x.device(), x.dtype()) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32) => {
            let (batch, d_conv) = x.dims2()?;
            let kernel_size = weight.dim(2)?;
            let out = Tensor::zeros((batch, d_conv), x.dtype(), x.device())?;

            let x_ptr = get_cuda_const_ptr(x)?;
            let weight_ptr = get_cuda_const_ptr(weight)?;
            let bias_ptr = if let Some(b) = bias {
                get_cuda_const_ptr(b)?
            } else {
                std::ptr::null()
            };
            let state_ptr = get_cuda_mut_ptr(conv_state)?;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let stream = *dev.cu_stream() as i64;

            unsafe {
                match x.dtype() {
                    DType::F16 => ffi::causal_conv1d_update_f16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_ptr,
                        out_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream,
                    ),
                    DType::BF16 => ffi::causal_conv1d_update_bf16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_ptr,
                        out_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream,
                    ),
                    DType::F32 => ffi::causal_conv1d_update_f32(
                        x_ptr as *const f32,
                        weight_ptr as *const f32,
                        bias_ptr as *const f32,
                        state_ptr as *mut f32,
                        out_ptr as *mut f32,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            Ok(out)
        }
        _ => causal_conv1d_update_naive(x, weight, bias, conv_state, activation_silu),
    }
}

/// Fused GDN gating computation.
/// g = -exp(A_log) * softplus(a + dt_bias)
/// beta = sigmoid(b)
#[cfg(feature = "cuda")]
pub fn fused_gdn_gating(
    a_log: &Tensor,
    a: &Tensor,
    b: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    match (a.device(), a.dtype()) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32) => {
            let (batch, seq_len, heads) = a.dims3()?;
            let g = Tensor::zeros(a.shape(), a.dtype(), a.device())?;
            let beta = Tensor::zeros(a.shape(), a.dtype(), a.device())?;

            let al_ptr = get_cuda_const_ptr(a_log)?;
            let a_ptr = get_cuda_const_ptr(a)?;
            let b_ptr = get_cuda_const_ptr(b)?;
            let dt_ptr = get_cuda_const_ptr(dt_bias)?;
            let g_ptr = get_cuda_mut_ptr(&g)?;
            let beta_ptr = get_cuda_mut_ptr(&beta)?;
            let stream = *dev.cu_stream() as i64;

            unsafe {
                match a.dtype() {
                    DType::F16 => ffi::fused_gdn_gating_f16(
                        al_ptr,
                        a_ptr,
                        b_ptr,
                        dt_ptr,
                        g_ptr,
                        beta_ptr,
                        batch as c_int,
                        seq_len as c_int,
                        heads as c_int,
                        stream,
                    ),
                    DType::BF16 => ffi::fused_gdn_gating_bf16(
                        al_ptr,
                        a_ptr,
                        b_ptr,
                        dt_ptr,
                        g_ptr,
                        beta_ptr,
                        batch as c_int,
                        seq_len as c_int,
                        heads as c_int,
                        stream,
                    ),
                    DType::F32 => ffi::fused_gdn_gating_f32(
                        al_ptr as *const f32,
                        a_ptr as *const f32,
                        b_ptr as *const f32,
                        dt_ptr as *const f32,
                        g_ptr as *mut f32,
                        beta_ptr as *mut f32,
                        batch as c_int,
                        seq_len as c_int,
                        heads as c_int,
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            Ok((g, beta))
        }
        _ => fused_gdn_gating_naive(a_log, a, b, dt_bias),
    }
}

/// Chunked gated delta rule for recurrent state update.
/// Uses CUDA kernels for both prefill and decode when available.
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
    match (q.device(), q.dtype()) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32) => {
            let (batch, seq_len, heads, dim) = q.dims4()?;
            let out = Tensor::zeros((batch, seq_len, heads, dim), q.dtype(), q.device())?;

            let q_ptr = get_cuda_const_ptr(q)?;
            let k_ptr = get_cuda_const_ptr(k)?;
            let v_ptr = get_cuda_const_ptr(v)?;
            let g_ptr = get_cuda_const_ptr(g)?;
            let beta_ptr = get_cuda_const_ptr(beta)?;
            let state_ptr = get_cuda_mut_ptr(state)?;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let stream = *dev.cu_stream() as i64;

            unsafe {
                if is_prefill {
                    match q.dtype() {
                        DType::F16 => ffi::chunk_gated_delta_rule_prefill_f16(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            g_ptr,
                            beta_ptr,
                            state_ptr,
                            out_ptr,
                            batch as c_int,
                            seq_len as c_int,
                            heads as c_int,
                            dim as c_int,
                            stream,
                        ),
                        DType::BF16 => ffi::chunk_gated_delta_rule_prefill_bf16(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            g_ptr,
                            beta_ptr,
                            state_ptr,
                            out_ptr,
                            batch as c_int,
                            seq_len as c_int,
                            heads as c_int,
                            dim as c_int,
                            stream,
                        ),
                        DType::F32 => ffi::chunk_gated_delta_rule_prefill_f32(
                            q_ptr as *const f32,
                            k_ptr as *const f32,
                            v_ptr as *const f32,
                            g_ptr as *const f32,
                            beta_ptr as *const f32,
                            state_ptr as *mut f32,
                            out_ptr as *mut f32,
                            batch as c_int,
                            seq_len as c_int,
                            heads as c_int,
                            dim as c_int,
                            stream,
                        ),
                        _ => unreachable!(),
                    }
                } else {
                    if seq_len != 1 {
                        return chunk_gated_delta_rule_decode_naive(q, k, v, g, beta, state);
                    }
                    match q.dtype() {
                        DType::F16 => ffi::chunk_gated_delta_rule_decode_f16(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            g_ptr,
                            beta_ptr,
                            state_ptr,
                            out_ptr,
                            batch as c_int,
                            heads as c_int,
                            dim as c_int,
                            stream,
                        ),
                        DType::BF16 => ffi::chunk_gated_delta_rule_decode_bf16(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            g_ptr,
                            beta_ptr,
                            state_ptr,
                            out_ptr,
                            batch as c_int,
                            heads as c_int,
                            dim as c_int,
                            stream,
                        ),
                        DType::F32 => ffi::chunk_gated_delta_rule_decode_f32(
                            q_ptr as *const f32,
                            k_ptr as *const f32,
                            v_ptr as *const f32,
                            g_ptr as *const f32,
                            beta_ptr as *const f32,
                            state_ptr as *mut f32,
                            out_ptr as *mut f32,
                            batch as c_int,
                            heads as c_int,
                            dim as c_int,
                            stream,
                        ),
                        _ => unreachable!(),
                    }
                }
            }
            Ok(out)
        }
        _ => {
            if is_prefill {
                chunk_gated_delta_rule_prefill_naive(q, k, v, g, beta, state)
            } else {
                chunk_gated_delta_rule_decode_naive(q, k, v, g, beta, state)
            }
        }
    }
}

/// DeltaNet recurrent update over flattened batch-head (`BH`) dimension.
///
/// Shapes:
/// - `q`, `k`: `[bh, seq, k_dim]`
/// - `v`: `[bh, seq, v_dim]`
/// - `g`, `beta`: `[bh, seq]`
/// - `state`: `[bh, k_dim, v_dim]` (updated in place)
///
/// Note: this function expects caller-side q/k normalization to already be applied.
#[cfg(feature = "cuda")]
pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    match q.device() {
        Device::Cuda(dev) => {
            let (bh, seq_len, k_dim) = q.dims3()?;
            let (bh_k, seq_len_k, k_dim_k) = k.dims3()?;
            let (bh_v, seq_len_v, v_dim) = v.dims3()?;
            let (bh_g, seq_len_g) = g.dims2()?;
            let (bh_b, seq_len_b) = beta.dims2()?;
            let (bh_s, k_dim_s, v_dim_s) = state.dims3()?;

            if bh != bh_k
                || bh != bh_v
                || bh != bh_g
                || bh != bh_b
                || bh != bh_s
                || seq_len != seq_len_k
                || seq_len != seq_len_v
                || seq_len != seq_len_g
                || seq_len != seq_len_b
                || k_dim != k_dim_k
                || k_dim != k_dim_s
                || v_dim != v_dim_s
            {
                candle_core::bail!(
                    "gated_delta_rule_recurrence shape mismatch: \
                     q={:?}, k={:?}, v={:?}, g={:?}, beta={:?}, state={:?}",
                    q.shape(),
                    k.shape(),
                    v.shape(),
                    g.shape(),
                    beta.shape(),
                    state.shape(),
                );
            }

            let out_dtype = q.dtype();
            let state_dtype = state.dtype();

            let q_f32 = q.to_dtype(DType::F32)?.contiguous()?;
            let k_f32 = k.to_dtype(DType::F32)?.contiguous()?;
            let v_f32 = v.to_dtype(DType::F32)?.contiguous()?;
            let g_f32 = g.to_dtype(DType::F32)?.contiguous()?;
            let beta_f32 = beta.to_dtype(DType::F32)?.contiguous()?;
            let state_f32 = state.to_dtype(DType::F32)?.contiguous()?;

            let out = Tensor::zeros((bh, seq_len, v_dim), DType::F32, q.device())?;

            let q_ptr = get_cuda_const_ptr(&q_f32)? as *const f32;
            let k_ptr = get_cuda_const_ptr(&k_f32)? as *const f32;
            let v_ptr = get_cuda_const_ptr(&v_f32)? as *const f32;
            let g_ptr = get_cuda_const_ptr(&g_f32)? as *const f32;
            let beta_ptr = get_cuda_const_ptr(&beta_f32)? as *const f32;
            let state_ptr = get_cuda_mut_ptr(&state_f32)? as *mut f32;
            let out_ptr = get_cuda_mut_ptr(&out)? as *mut f32;
            let stream = *dev.cu_stream() as i64;

            unsafe {
                ffi::gated_delta_rule_recurrence(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    g_ptr,
                    beta_ptr,
                    state_ptr,
                    out_ptr,
                    bh as c_int,
                    seq_len as c_int,
                    k_dim as c_int,
                    v_dim as c_int,
                    stream,
                )
            }

            *state = state_f32.to_dtype(state_dtype)?;
            out.to_dtype(out_dtype)
        }
        _ => gated_delta_rule_recurrence_naive(q, k, v, g, beta, state),
    }
}

fn gated_delta_rule_recurrence_naive(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let out_dtype = q.dtype();
    let state_dtype = state.dtype();

    let (_bh, seq_len, _k_dim) = q.dims3()?;
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let g = g.to_dtype(DType::F32)?;
    let beta = beta.to_dtype(DType::F32)?;
    let mut s = state.to_dtype(DType::F32)?;

    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let q_t = q.narrow(1, t, 1)?.squeeze(1)?; // [bh, k_dim]
        let k_t = k.narrow(1, t, 1)?.squeeze(1)?; // [bh, k_dim]
        let v_t = v.narrow(1, t, 1)?.squeeze(1)?; // [bh, v_dim]
        let g_t = g.narrow(1, t, 1)?.squeeze(1)?; // [bh]
        let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?; // [bh]

        let decay = g_t.exp()?.unsqueeze(1)?.unsqueeze(2)?;
        s = s.broadcast_mul(&decay)?;

        let k_exp = k_t.unsqueeze(2)?; // [bh, k_dim, 1]
        let kv_mem = s.broadcast_mul(&k_exp)?.sum(1)?; // [bh, v_dim]
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_t.unsqueeze(1)?)?; // [bh, v_dim]

        let outer = k_exp.broadcast_mul(&delta.unsqueeze(1)?)?; // [bh, k_dim, v_dim]
        s = (s + outer)?;

        let y_t = s.broadcast_mul(&q_t.unsqueeze(2)?)?.sum(1)?; // [bh, v_dim]
        outputs.push(y_t.unsqueeze(1)?);
    }

    *state = s.to_dtype(state_dtype)?;
    let output_refs = outputs.iter().collect::<Vec<_>>();
    Tensor::cat(&output_refs, 1)?.to_dtype(out_dtype)
}

fn causal_conv1d_fwd_naive_with_state(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    cu_seqlens: Option<&Tensor>,
    activation_silu: bool,
) -> Result<Tensor> {
    if cu_seqlens.is_none() {
        return causal_conv1d_naive(x, weight, bias, activation_silu);
    }

    let weight_2d = weight.squeeze(1)?; // [d_conv, kernel_size]
    let kernel_size = weight_2d.dim(1)?;
    let d_conv = weight_2d.dim(0)?;
    let batch_size = conv_state.dim(0)?;
    let cu = cu_seqlens.unwrap().to_vec1::<u32>()?;
    if cu.len() != batch_size + 1 {
        candle_core::bail!(
            "causal_conv1d_fwd: cu_seqlens length {} does not match batch size {}",
            cu.len(),
            batch_size
        );
    }

    let mut outputs = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let start = cu[b] as usize;
        let end = cu[b + 1] as usize;
        let seq_len = end.saturating_sub(start);
        let seq_x = x.narrow(0, start, seq_len)?;

        let history = conv_state.i(b)?.transpose(0, 1)?; // [kernel_size - 1, d_conv]
        let x_padded = Tensor::cat(&[&history, &seq_x], 0)?; // [seq_len + kernel_size - 1, d_conv]

        let mut slices = Vec::with_capacity(kernel_size);
        for k in 0..kernel_size {
            let slice = x_padded.narrow(0, k, seq_len)?;
            let w_k = weight_2d.i((.., k))?;
            slices.push(slice.broadcast_mul(&w_k)?);
        }
        let mut seq_out = slices[0].clone();
        for s in &slices[1..] {
            seq_out = (seq_out + s)?;
        }
        if let Some(bias) = bias {
            seq_out = seq_out.broadcast_add(bias)?;
        }
        if activation_silu {
            seq_out = candle_nn::ops::silu(&seq_out)?;
        }
        outputs.push(seq_out);

        let next_history = x_padded
            .narrow(0, seq_len, kernel_size - 1)?
            .transpose(0, 1)?;
        *conv_state = conv_state.slice_assign(
            &[b..b + 1, 0..d_conv, 0..kernel_size - 1],
            &next_history.unsqueeze(0)?,
        )?;
    }

    let output_refs = outputs.iter().collect::<Vec<_>>();
    Tensor::cat(&output_refs, 0)
}

/// Naive causal conv1d using candle ops.
pub fn causal_conv1d_naive(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    activation_silu: bool,
) -> Result<Tensor> {
    let weight_2d = weight.squeeze(1)?; // [d_conv, kernel_size]
    let kernel_size = weight_2d.dim(1)?;
    let d_conv = weight_2d.dim(0)?;
    let seq_len = x.dim(0)?;

    let padding = Tensor::zeros((kernel_size - 1, d_conv), x.dtype(), x.device())?;
    let x_padded = Tensor::cat(&[&padding, x], 0)?;

    let mut slices = Vec::with_capacity(kernel_size);
    for k in 0..kernel_size {
        let slice = x_padded.narrow(0, k, seq_len)?;
        let w_k = weight_2d.i((.., k))?;
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

/// Naive causal conv1d update for decode (single step).
pub fn causal_conv1d_update_naive(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    // x: [batch, d_conv], conv_state: [batch, d_conv, kernel_size - 1]
    let weight_2d = weight.squeeze(1)?; // [d_conv, kernel_size]
    let kernel_size = weight_2d.dim(1)?;

    let x_expanded = x.unsqueeze(2)?; // [batch, d_conv, 1]
    let prev_state = conv_state.clone();
    let full_window = Tensor::cat(&[&prev_state, &x_expanded], 2)?; // [batch, d_conv, kernel_size]
    let next_state = full_window.narrow(2, 1, kernel_size - 1)?;
    *conv_state = next_state;

    let mut output = full_window
        .broadcast_mul(&weight_2d.unsqueeze(0)?)?
        .sum(2)?; // [batch, d_conv]

    if let Some(bias) = bias {
        output = output.broadcast_add(bias)?;
    }

    if activation_silu {
        candle_nn::ops::silu(&output)
    } else {
        Ok(output)
    }
}

/// Naive fused GDN gating.
pub fn fused_gdn_gating_naive(
    a_log: &Tensor,
    a: &Tensor,
    b: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // g = -exp(A_log) * softplus(a + dt_bias)
    let a_dt = a.broadcast_add(dt_bias)?;
    let g = softplus(&a_dt)?.broadcast_mul(&a_log.exp()?.neg()?)?;

    // beta = sigmoid(b)
    let beta = candle_nn::ops::sigmoid(b)?;
    Ok((g, beta))
}

/// Softplus: log(1 + exp(x)).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let exp_x = x.exp()?;
    (exp_x + 1.0)?.log()
}

/// Naive chunked gated delta rule for prefill.
fn chunk_gated_delta_rule_prefill_naive(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let (_, seq_len, _, _) = q.dims4()?;
    let mut outputs = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let q_t = q.narrow(1, t, 1)?.squeeze(1)?;
        let k_t = k.narrow(1, t, 1)?.squeeze(1)?;
        let v_t = v.narrow(1, t, 1)?.squeeze(1)?;
        let g_t = g.narrow(1, t, 1)?.squeeze(1)?;
        let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?;

        let g_expanded = g_t.unsqueeze(2)?.unsqueeze(3)?;
        let beta_expanded = beta_t.unsqueeze(2)?.unsqueeze(3)?;
        let kv_outer = k_t.unsqueeze(3)?.matmul(&v_t.unsqueeze(2)?)?;
        *state = (g_expanded.broadcast_mul(state)? + beta_expanded.broadcast_mul(&kv_outer)?)?;

        let out_t = q_t.unsqueeze(2)?.matmul(state)?.squeeze(2)?;
        outputs.push(out_t.unsqueeze(1)?);
    }

    let output_refs = outputs.iter().collect::<Vec<_>>();
    Tensor::cat(&output_refs, 1)
}

/// Naive gated delta rule for decode (single step).
fn chunk_gated_delta_rule_decode_naive(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let q_t = q.squeeze(1)?;
    let k_t = k.squeeze(1)?;
    let v_t = v.squeeze(1)?;
    let g_t = g.squeeze(1)?;
    let beta_t = beta.squeeze(1)?;

    let g_expanded = g_t.unsqueeze(2)?.unsqueeze(3)?;
    let beta_expanded = beta_t.unsqueeze(2)?.unsqueeze(3)?;
    let kv_outer = k_t.unsqueeze(3)?.matmul(&v_t.unsqueeze(2)?)?;
    *state = (g_expanded.broadcast_mul(state)? + beta_expanded.broadcast_mul(&kv_outer)?)?;

    let out = q_t.unsqueeze(2)?.matmul(state)?.squeeze(2)?;
    Ok(out.unsqueeze(1)?)
}

#[cfg(not(feature = "cuda"))]
pub fn causal_conv1d_fwd(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    cu_seqlens: Option<&Tensor>,
    activation_silu: bool,
) -> Result<Tensor> {
    causal_conv1d_fwd_naive_with_state(x, weight, bias, conv_state, cu_seqlens, activation_silu)
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

#[cfg(not(feature = "cuda"))]
pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    gated_delta_rule_recurrence_naive(q, k, v, g, beta, state)
}
