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
    let (storage, layout) = t.storage_and_layout();
    let offset = layout.start_offset();
    match (&*storage, t.dtype()) {
        (Storage::Cuda(s), DType::F16) => {
            Ok(*s.as_cuda_slice::<f16>()?.slice(offset..).device_ptr() as *const c_void)
        }
        (Storage::Cuda(s), DType::BF16) => {
            Ok(*s.as_cuda_slice::<bf16>()?.slice(offset..).device_ptr() as *const c_void)
        }
        (Storage::Cuda(s), DType::F32) => {
            Ok(*s.as_cuda_slice::<f32>()?.slice(offset..).device_ptr() as *const c_void)
        }
        _ => candle_core::bail!("Expected CUDA tensor with f16/bf16/f32 dtype"),
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_const_ptr_u32(t: &Tensor) -> Result<*const u32> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    let offset = layout.start_offset();
    match &*storage {
        Storage::Cuda(s) => {
            Ok(*s.as_cuda_slice::<u32>()?.slice(offset..).device_ptr() as *const u32)
        }
        _ => candle_core::bail!("Expected CUDA u32 tensor"),
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_const_ptr_i64(t: &Tensor) -> Result<*const i64> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    let offset = layout.start_offset();
    match &*storage {
        Storage::Cuda(s) => {
            Ok(*s.as_cuda_slice::<i64>()?.slice(offset..).device_ptr() as *const i64)
        }
        _ => candle_core::bail!("Expected CUDA i64 tensor"),
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_mut_ptr(t: &Tensor) -> Result<*mut c_void> {
    Ok(get_cuda_const_ptr(t)? as *mut c_void)
}

#[cfg(feature = "cuda")]
fn ensure_contiguous(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
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

/// Causal conv1d single-step update with slot-indexed global state.
#[cfg(feature = "cuda")]
pub fn causal_conv1d_update_slots(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    slots: &Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    match (x.device(), x.dtype()) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32) => {
            let x_c = x.contiguous()?;
            let weight_c = weight.contiguous()?;
            let bias_c = if let Some(b) = bias {
                Some(b.contiguous()?)
            } else {
                None
            };

            let (batch, d_conv) = x_c.dims2()?;
            let kernel_size = weight_c.dim(2)?;
            if slots.dtype() != DType::I64 || slots.dim(0)? != batch {
                candle_core::bail!(
                    "causal_conv1d_update_slots expects slots [batch] I64, got {:?} {:?}",
                    slots.shape(),
                    slots.dtype()
                );
            }
            let out = Tensor::zeros((batch, d_conv), x.dtype(), x.device())?;

            let x_ptr = get_cuda_const_ptr(&x_c)?;
            let weight_ptr = get_cuda_const_ptr(&weight_c)?;
            let bias_ptr = if let Some(ref b) = bias_c {
                get_cuda_const_ptr(b)?
            } else {
                std::ptr::null()
            };
            let state_ptr = get_cuda_mut_ptr(conv_state)?;
            let slots_ptr = get_cuda_const_ptr_i64(slots)?;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let stream = *dev.cu_stream() as i64;

            unsafe {
                match x.dtype() {
                    DType::F16 => ffi::causal_conv1d_update_slots_f16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream,
                    ),
                    DType::BF16 => ffi::causal_conv1d_update_slots_bf16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr,
                        batch as c_int,
                        d_conv as c_int,
                        kernel_size as c_int,
                        activation_silu,
                        stream,
                    ),
                    DType::F32 => ffi::causal_conv1d_update_slots_f32(
                        x_ptr as *const f32,
                        weight_ptr as *const f32,
                        bias_ptr as *const f32,
                        state_ptr as *mut f32,
                        slots_ptr,
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
        _ => {
            // Non-CUDA fallback: keep behavior correct for tests.
            let slots_vec = if slots.dtype() == DType::I64 {
                slots.to_vec1::<i64>()?
            } else {
                candle_core::bail!("causal_conv1d_update_slots fallback expects I64 slots");
            };
            if slots_vec.is_empty() {
                candle_core::bail!("causal_conv1d_update_slots got empty slots");
            }
            let mut gathered = Vec::with_capacity(slots_vec.len());
            for &s in &slots_vec {
                gathered.push(conv_state.i(s as usize)?);
            }
            let gathered_refs = gathered.iter().collect::<Vec<_>>();
            let mut batch_state = Tensor::stack(&gathered_refs, 0)?;
            let out =
                causal_conv1d_update_naive(x, weight, bias, &mut batch_state, activation_silu)?;
            for (i, &s) in slots_vec.iter().enumerate() {
                *conv_state = conv_state.slice_assign(
                    &[
                        s as usize..s as usize + 1,
                        0..conv_state.dim(1)?,
                        0..conv_state.dim(2)?,
                    ],
                    &batch_state.narrow(0, i, 1)?,
                )?;
            }
            Ok(out)
        }
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
                    DType::F16 => {
                        if a_log.dtype() == DType::F32 {
                            ffi::fused_gdn_gating_f16_alog_f32(
                                al_ptr as *const f32,
                                a_ptr,
                                b_ptr,
                                dt_ptr,
                                g_ptr,
                                beta_ptr,
                                batch as c_int,
                                seq_len as c_int,
                                heads as c_int,
                                stream,
                            )
                        } else {
                            ffi::fused_gdn_gating_f16(
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
                            )
                        }
                    }
                    DType::BF16 => {
                        if a_log.dtype() == DType::F32 {
                            ffi::fused_gdn_gating_bf16_alog_f32(
                                al_ptr as *const f32,
                                a_ptr,
                                b_ptr,
                                dt_ptr,
                                g_ptr,
                                beta_ptr,
                                batch as c_int,
                                seq_len as c_int,
                                heads as c_int,
                                stream,
                            )
                        } else {
                            ffi::fused_gdn_gating_bf16(
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
                            )
                        }
                    }
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

/// Fused gated RMSNorm:
/// out = RMSNorm(x; gamma, bias, eps) * SiLU(z)
/// - `x`: [rows, value_dim]
/// - `z`: [rows, value_dim]
/// - `norm_weight`: [value_dim] (full) or [group_size] (per-group/head)
/// - `norm_bias`: same rule as `norm_weight`
#[cfg(feature = "cuda")]
pub fn gated_rmsnorm_silu_mul(
    x: &Tensor,
    z: &Tensor,
    norm_weight: &Tensor,
    norm_bias: Option<&Tensor>,
    eps: f64,
    group_size: usize,
) -> Result<Tensor> {
    match (x.device(), x.dtype()) {
        (Device::Cuda(dev), DType::F16 | DType::BF16 | DType::F32) => {
            let x_c = x.contiguous()?;
            let (rows, value_dim) = x_c.dims2()?;
            let z_c = if z.dtype() == x.dtype() {
                z.contiguous()?
            } else {
                z.to_dtype(x.dtype())?.contiguous()?
            };
            let (z_rows, z_dim) = z_c.dims2()?;
            if z_rows != rows || z_dim != value_dim {
                candle_core::bail!(
                    "gated_rmsnorm_silu_mul shape mismatch: x={:?}, z={:?}",
                    x.shape(),
                    z.shape()
                );
            }
            if group_size == 0 || value_dim % group_size != 0 {
                candle_core::bail!(
                    "gated_rmsnorm_silu_mul invalid group_size={} for value_dim={}",
                    group_size,
                    value_dim
                );
            }

            let weight_len = norm_weight.dim(0)?;
            let per_group_weights = if weight_len == group_size {
                true
            } else if weight_len == value_dim {
                false
            } else {
                candle_core::bail!(
                    "gated_rmsnorm_silu_mul invalid weight shape {:?}, expected [{group_size}] or [{value_dim}]",
                    norm_weight.shape()
                );
            };

            let bias = if let Some(b) = norm_bias {
                let b_len = b.dim(0)?;
                let expected = if per_group_weights {
                    group_size
                } else {
                    value_dim
                };
                if b_len != expected {
                    candle_core::bail!(
                        "gated_rmsnorm_silu_mul invalid bias shape {:?}, expected [{expected}]",
                        b.shape()
                    );
                }
                Some(b)
            } else {
                None
            };
            let out = Tensor::zeros((rows, value_dim), x.dtype(), x.device())?;

            let x_ptr = get_cuda_const_ptr(&x_c)?;
            let z_ptr = get_cuda_const_ptr(&z_c)?;
            let w_ptr = get_cuda_const_ptr(&norm_weight)?;
            let b_ptr = if let Some(ref b) = bias {
                get_cuda_const_ptr(b)?
            } else {
                std::ptr::null()
            };
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let stream = *dev.cu_stream() as i64;
            let eps = eps as f32;

            unsafe {
                match x.dtype() {
                    DType::F16 => {
                        if norm_weight.dtype() == DType::F32 {
                            ffi::gdn_gated_rmsnorm_silu_mul_f16_wf32(
                                x_ptr,
                                z_ptr,
                                w_ptr as *const f32,
                                b_ptr as *const f32,
                                out_ptr,
                                rows as c_int,
                                value_dim as c_int,
                                group_size as c_int,
                                eps,
                                per_group_weights,
                                bias.is_some(),
                                stream,
                            )
                        } else {
                            ffi::gdn_gated_rmsnorm_silu_mul_f16(
                                x_ptr,
                                z_ptr,
                                w_ptr,
                                b_ptr,
                                out_ptr,
                                rows as c_int,
                                value_dim as c_int,
                                group_size as c_int,
                                eps,
                                per_group_weights,
                                bias.is_some(),
                                stream,
                            )
                        }
                    }
                    DType::BF16 => {
                        if norm_weight.dtype() == DType::F32 {
                            ffi::gdn_gated_rmsnorm_silu_mul_bf16_wf32(
                                x_ptr,
                                z_ptr,
                                w_ptr as *const f32,
                                b_ptr as *const f32,
                                out_ptr,
                                rows as c_int,
                                value_dim as c_int,
                                group_size as c_int,
                                eps,
                                per_group_weights,
                                bias.is_some(),
                                stream,
                            )
                        } else {
                            ffi::gdn_gated_rmsnorm_silu_mul_bf16(
                                x_ptr,
                                z_ptr,
                                w_ptr,
                                b_ptr,
                                out_ptr,
                                rows as c_int,
                                value_dim as c_int,
                                group_size as c_int,
                                eps,
                                per_group_weights,
                                bias.is_some(),
                                stream,
                            )
                        }
                    }
                    DType::F32 => ffi::gdn_gated_rmsnorm_silu_mul_f32(
                        x_ptr as *const f32,
                        z_ptr as *const f32,
                        w_ptr as *const f32,
                        b_ptr as *const f32,
                        out_ptr as *mut f32,
                        rows as c_int,
                        value_dim as c_int,
                        group_size as c_int,
                        eps,
                        per_group_weights,
                        bias.is_some(),
                        stream,
                    ),
                    _ => unreachable!(),
                }
            }
            Ok(out)
        }
        _ => gated_rmsnorm_silu_mul_naive(x, z, norm_weight, norm_bias, eps, group_size),
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

            let original_shape = state.shape().clone();
            let (bh_s, k_dim_s, v_dim_s) = if original_shape.rank() == 4 {
                let (b, h, k, v) = state.dims4()?;
                (b * h, k, v)
            } else {
                state.dims3()?
            };

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

            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;

            if q_c.dtype() != k_c.dtype() || q_c.dtype() != v_c.dtype() {
                candle_core::bail!(
                    "gated_delta_rule_recurrence dtype mismatch: q={:?} k={:?} v={:?}",
                    q_c.dtype(),
                    k_c.dtype(),
                    v_c.dtype()
                );
            }

            let out_dtype = q_c.dtype();
            let g_f32 = if g.dtype() == DType::F32 {
                ensure_contiguous(g)?
            } else {
                g.to_dtype(DType::F32)?.contiguous()?
            };
            let beta_f32 = if beta.dtype() == DType::F32 {
                ensure_contiguous(beta)?
            } else {
                beta.to_dtype(DType::F32)?.contiguous()?
            };

            if state.dtype() != DType::F32 {
                candle_core::bail!(
                    "gated_delta_rule_recurrence expects F32 state, got {:?}",
                    state.dtype()
                );
            }
            if !state.is_contiguous() {
                candle_core::bail!("gated_delta_rule_recurrence expects contiguous state");
            }
            let state_ptr = get_cuda_mut_ptr(state)? as *mut f32;

            let out = Tensor::zeros((bh, seq_len, v_dim), DType::F32, q_c.device())?;

            let q_ptr = get_cuda_const_ptr(&q_c)?;
            let k_ptr = get_cuda_const_ptr(&k_c)?;
            let v_ptr = get_cuda_const_ptr(&v_c)?;
            let g_ptr = get_cuda_const_ptr(&g_f32)? as *const f32;
            let beta_ptr = get_cuda_const_ptr(&beta_f32)? as *const f32;
            let out_ptr = get_cuda_mut_ptr(&out)? as *mut f32;
            let stream = *dev.cu_stream() as i64;

            unsafe {
                match out_dtype {
                    DType::F32 => ffi::gated_delta_rule_recurrence(
                        q_ptr as *const f32,
                        k_ptr as *const f32,
                        v_ptr as *const f32,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        out_ptr,
                        bh as c_int,
                        seq_len as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        stream,
                    ),
                    DType::F16 => ffi::gated_delta_rule_recurrence_f16(
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
                    ),
                    DType::BF16 => ffi::gated_delta_rule_recurrence_bf16(
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
                    ),
                    dt => candle_core::bail!(
                        "gated_delta_rule_recurrence unsupported dtype: {:?}",
                        dt
                    ),
                }
            }

            out.to_dtype(out_dtype)
        }
        _ => gated_delta_rule_recurrence_naive(q, k, v, g, beta, state),
    }
}

/// One-step decode recurrence with slot-indexed global state.
#[cfg(feature = "cuda")]
pub fn gated_delta_rule_decode_slots(
    q: &Tensor,    // [batch, heads, k_dim], caller-scaled if needed (e.g. * 1/sqrt(k_dim))
    k: &Tensor,    // [batch, heads, k_dim]
    v: &Tensor,    // [batch, heads, v_dim]
    g: &Tensor,    // [batch, heads]
    beta: &Tensor, // [batch, heads]
    state: &mut Tensor, // [max_batch, heads, k_dim, v_dim]
    slots: &Tensor, // [batch] i64
) -> Result<Tensor> {
    match q.device() {
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let g_c = ensure_contiguous(g)?;
            let beta_c = ensure_contiguous(beta)?;

            let (bq, hq, kq) = q.dims3()?;
            let (bk, hk, kk) = k.dims3()?;
            let (bv, hv, v_dim) = v.dims3()?;
            let (bg, hg) = g.dims2()?; // g is [batch, heads]
            let (bb, hb) = beta.dims2()?;

            let batch = bq;
            let heads = hv;
            let k_dim = kq;

            if batch != bk
                || batch != bv
                || batch != bg
                || batch != bb
                || heads != hg
                || heads != hb
                || heads != hk
                || heads != hq
                || k_dim != kk
            {
                candle_core::bail!(
                    "gated_delta_rule_decode_slots shape mismatch: q={:?}, k={:?}, v={:?}, g={:?}, beta={:?}",
                    q.shape(),
                    k.shape(),
                    v.shape(),
                    g.shape(),
                    beta.shape()
                );
            }
            if slots.dtype() != DType::I64 || slots.dim(0)? != batch {
                candle_core::bail!(
                    "gated_delta_rule_decode_slots expects slots [batch] I64, got {:?} {:?}",
                    slots.shape(),
                    slots.dtype()
                );
            }

            if q.dtype() != k.dtype()
                || q.dtype() != v.dtype()
                || q.dtype() != g.dtype()
                || q.dtype() != beta.dtype()
            {
                candle_core::bail!(
                    "gated_delta_rule_decode_slots dtype mismatch: q={:?} k={:?} v={:?} g={:?} beta={:?}",
                    q.dtype(),
                    k.dtype(),
                    v.dtype(),
                    g.dtype(),
                    beta.dtype()
                );
            }

            let slots_ptr = get_cuda_const_ptr_i64(slots)?;
            let stream = *dev.cu_stream() as i64;
            if q.dtype() == DType::F32 {
                if state.dtype() != DType::F32 {
                    candle_core::bail!(
                        "gated_delta_rule_decode_slots expects F32 state for F32 inputs, got {:?}",
                        state.dtype()
                    );
                }
                let out = Tensor::zeros((batch, heads, v_dim), DType::F32, q.device())?;
                let q_ptr = get_cuda_const_ptr(&q_c)? as *const f32;
                let k_ptr = get_cuda_const_ptr(&k_c)? as *const f32;
                let v_ptr = get_cuda_const_ptr(&v_c)? as *const f32;
                let g_ptr = get_cuda_const_ptr(&g_c)? as *const f32;
                let beta_ptr = get_cuda_const_ptr(&beta_c)? as *const f32;
                let state_ptr = get_cuda_mut_ptr(state)? as *mut f32;
                let out_ptr = get_cuda_mut_ptr(&out)? as *mut f32;

                unsafe {
                    ffi::gated_delta_rule_decode_slots_f32(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr,
                        batch as c_int,
                        heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        stream,
                    )
                }
                Ok(out)
            } else {
                if state.dtype() != DType::F32 {
                    candle_core::bail!(
                        "gated_delta_rule_decode_slots expects F32 recurrent state for {:?} inputs, got {:?}",
                        q.dtype(),
                        state.dtype()
                    );
                }
                if !state.is_contiguous() {
                    candle_core::bail!(
                        "gated_delta_rule_decode_slots expects contiguous recurrent state during CUDA execution"
                    );
                }
                // S3: use native-dtype kernel with FP32 state — no input casts needed
                let out = Tensor::zeros((batch, heads, v_dim), q.dtype(), q.device())?;
                let q_ptr = get_cuda_const_ptr(&q_c)?;
                let k_ptr = get_cuda_const_ptr(&k_c)?;
                let v_ptr = get_cuda_const_ptr(&v_c)?;
                let g_ptr = get_cuda_const_ptr(&g_c)?;
                let beta_ptr = get_cuda_const_ptr(&beta_c)?;
                let state_ptr = get_cuda_mut_ptr(state)? as *mut f32;
                let out_ptr = get_cuda_mut_ptr(&out)?;

                match q.dtype() {
                    DType::F16 => unsafe {
                        ffi::gated_delta_rule_decode_slots_f16_state_f32(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            g_ptr,
                            beta_ptr,
                            state_ptr,
                            slots_ptr,
                            out_ptr as *mut c_void,
                            batch as c_int,
                            heads as c_int,
                            k_dim as c_int,
                            v_dim as c_int,
                            stream,
                        )
                    },
                    DType::BF16 => unsafe {
                        ffi::gated_delta_rule_decode_slots_bf16_state_f32(
                            q_ptr,
                            k_ptr,
                            v_ptr,
                            g_ptr,
                            beta_ptr,
                            state_ptr,
                            slots_ptr,
                            out_ptr as *mut c_void,
                            batch as c_int,
                            heads as c_int,
                            k_dim as c_int,
                            v_dim as c_int,
                            stream,
                        )
                    },
                    dt => candle_core::bail!(
                        "gated_delta_rule_decode_slots unsupported dtype: {:?}",
                        dt
                    ),
                }
                Ok(out)
            }
        }
        _ => {
            let slots_vec = slots.to_vec1::<i64>()?;
            let mut outs = Vec::with_capacity(slots_vec.len());
            for (b, &slot) in slots_vec.iter().enumerate() {
                let q_b = q.i(b)?; // [heads, k_dim]
                let k_b = k.i(b)?; // [heads, k_dim]
                let v_b = v.i(b)?; // [heads, v_dim]
                let g_b = g.i(b)?; // [heads]
                let beta_b = beta.i(b)?; // [heads]
                let mut state_b = state.i(slot as usize)?; // [heads, k_dim, v_dim]

                // gated_delta_rule_recurrence_naive expects [bh, seq, dim] for q,k,v
                // and [bh, seq] for g,beta
                // and [bh, k_dim, v_dim] for state
                // Here, bh is heads, seq is 1.
                let out_b = gated_delta_rule_recurrence_naive(
                    &q_b.unsqueeze(1)?,    // [heads, 1, k_dim]
                    &k_b.unsqueeze(1)?,    // [heads, 1, k_dim]
                    &v_b.unsqueeze(1)?,    // [heads, 1, v_dim]
                    &g_b.unsqueeze(1)?,    // [heads, 1]
                    &beta_b.unsqueeze(1)?, // [heads, 1]
                    &mut state_b,          // [heads, k_dim, v_dim]
                )? // returns [heads, 1, v_dim]
                .squeeze(1)?; // [heads, v_dim]

                // Update the state for the specific slot
                *state = state.slice_assign(
                    &[
                        slot as usize..slot as usize + 1,
                        0..state.dim(1)?,
                        0..state.dim(2)?,
                        0..state.dim(3)?,
                    ],
                    &state_b.unsqueeze(0)?, // state_b is [heads, k_dim, v_dim], unsqueeze to [1, heads, k_dim, v_dim]
                )?;
                outs.push(out_b);
            }
            let refs = outs.iter().collect::<Vec<_>>();
            Tensor::stack(&refs, 0)
        }
    }
}

/// Fused L2 normalization over the last dimension.
/// Replaces the multi-op sequence: sumsq → sqrt → clamp → div.
/// input: [rows, dim] → output: [rows, dim] (each row normalized to unit L2 norm)
pub fn l2_norm_last_dim(input: &Tensor, eps: f64) -> Result<Tensor> {
    match input.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            let input_c = ensure_contiguous(input)?;
            let shape = input_c.shape();
            if shape.rank() < 2 {
                candle_core::bail!(
                    "l2_norm_last_dim expects at least 2D input, got {:?}",
                    shape
                );
            }
            let dim = shape.dims()[shape.rank() - 1];
            let rows = shape.elem_count() / dim;
            let output = Tensor::zeros(shape, input.dtype(), input.device())?;
            let in_ptr = get_cuda_const_ptr(&input_c)?;
            let out_ptr = get_cuda_mut_ptr(&output)?;
            let stream = *dev.cu_stream() as i64;

            match input.dtype() {
                DType::F32 => unsafe {
                    ffi::l2_norm_last_dim_f32(
                        in_ptr as *const f32,
                        out_ptr as *mut f32,
                        rows as c_int,
                        dim as c_int,
                        eps as f32,
                        stream,
                    )
                },
                DType::F16 => unsafe {
                    ffi::l2_norm_last_dim_f16(
                        in_ptr,
                        out_ptr as *mut c_void,
                        rows as c_int,
                        dim as c_int,
                        eps as f32,
                        stream,
                    )
                },
                DType::BF16 => unsafe {
                    ffi::l2_norm_last_dim_bf16(
                        in_ptr,
                        out_ptr as *mut c_void,
                        rows as c_int,
                        dim as c_int,
                        eps as f32,
                        stream,
                    )
                },
                dt => candle_core::bail!("l2_norm_last_dim: unsupported dtype {:?}", dt),
            }
            Ok(output)
        }
        _ => {
            // Fallback using candle ops
            let sumsq = input.sqr()?.sum_keepdim(input.rank() - 1)?;
            let norm = (sumsq + eps)?.sqrt()?;
            input.broadcast_div(&norm)
        }
    }
}

/// Batched variable-length recurrence: processes multiple sequences in one CUDA launch.
/// Accepts native dtype inputs (bf16/f16/f32) with FP32 state.
///
/// Shapes:
/// - `q`, `k`: `[total_tokens, num_heads, k_dim]`
/// - `v`: `[total_tokens, num_heads, v_dim]`
/// - `g`, `beta`: `[total_tokens, num_heads]`
/// - `state`: `[max_batch, num_heads, k_dim, v_dim]` (FP32, updated in place)
/// - `slots`: `[batch]` i64
/// - `cu_seqlens`: `[batch + 1]` u32
pub fn gated_delta_rule_recurrence_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
    cu_seqlens: &Tensor,
) -> Result<Tensor> {
    match q.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let g_c = ensure_contiguous(g)?;
            let beta_c = ensure_contiguous(beta)?;

            let (total_tokens, num_heads, k_dim) = q_c.dims3()?;
            let num_heads_v = v_c.dim(1)?;
            let v_dim = v_c.dim(2)?;
            let batch = slots.dim(0)?;

            if num_heads != num_heads_v {
                candle_core::bail!(
                    "gated_delta_rule_recurrence_varlen: q heads {} != v heads {}",
                    num_heads,
                    num_heads_v
                );
            }

            if state.dtype() != DType::F32 {
                candle_core::bail!(
                    "gated_delta_rule_recurrence_varlen expects FP32 state, got {:?}",
                    state.dtype()
                );
            }
            if cu_seqlens.dtype() != DType::U32 || cu_seqlens.dim(0)? != batch + 1 {
                candle_core::bail!(
                    "gated_delta_rule_recurrence_varlen expects cu_seqlens [batch+1] U32, got {:?} {:?}",
                    cu_seqlens.shape(),
                    cu_seqlens.dtype()
                );
            }

            let out = Tensor::zeros((total_tokens, num_heads, v_dim), q.dtype(), q.device())?;

            let q_ptr = get_cuda_const_ptr(&q_c)?;
            let k_ptr = get_cuda_const_ptr(&k_c)?;
            let v_ptr = get_cuda_const_ptr(&v_c)?;
            let g_ptr = get_cuda_const_ptr(&g_c)?;
            let beta_ptr = get_cuda_const_ptr(&beta_c)?;
            let state_ptr = get_cuda_mut_ptr(state)? as *mut f32;
            let slots_ptr = get_cuda_const_ptr_i64(slots)?;
            let cu_ptr = get_cuda_const_ptr_u32(cu_seqlens)?;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let stream = *dev.cu_stream() as i64;

            match q.dtype() {
                DType::F32 => unsafe {
                    ffi::gated_delta_rule_recurrence_varlen_f32(
                        q_ptr as *const f32,
                        k_ptr as *const f32,
                        v_ptr as *const f32,
                        g_ptr as *const f32,
                        beta_ptr as *const f32,
                        state_ptr,
                        slots_ptr,
                        out_ptr as *mut f32,
                        cu_ptr,
                        batch as c_int,
                        num_heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        stream,
                    )
                },
                DType::F16 => unsafe {
                    ffi::gated_delta_rule_recurrence_varlen_f16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr as *mut c_void,
                        cu_ptr,
                        batch as c_int,
                        num_heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        stream,
                    )
                },
                DType::BF16 => unsafe {
                    ffi::gated_delta_rule_recurrence_varlen_bf16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr as *mut c_void,
                        cu_ptr,
                        batch as c_int,
                        num_heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        stream,
                    )
                },
                dt => candle_core::bail!(
                    "gated_delta_rule_recurrence_varlen unsupported dtype: {:?}",
                    dt
                ),
            }
            Ok(out)
        }
        _ => gated_delta_rule_recurrence_varlen_naive(q, k, v, g, beta, state, slots, cu_seqlens),
    }
}

fn gated_delta_rule_recurrence_varlen_naive(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
    cu_seqlens: &Tensor,
) -> Result<Tensor> {
    let cu = cu_seqlens.to_vec1::<u32>()?;
    let slots_vec = slots.to_vec1::<i64>()?;
    let batch = slots_vec.len();
    if cu.len() != batch + 1 {
        candle_core::bail!(
            "gated_delta_rule_recurrence_varlen_naive: cu_seqlens length {} does not match batch size {}",
            cu.len(),
            batch
        );
    }

    let total_tokens = q.dim(0)?;
    let num_heads_q = q.dim(1)?;
    let v_dim = v.dim(2)?;
    let mut outputs = Vec::with_capacity(batch);

    for b in 0..batch {
        let start = cu[b] as usize;
        let end = cu[b + 1] as usize;
        if start >= end {
            continue;
        }
        let seq_len = end - start;

        let q_b = q.narrow(0, start, seq_len)?.transpose(0, 1)?; // [num_heads, seq_len, k_dim]
        let k_b = k.narrow(0, start, seq_len)?.transpose(0, 1)?;
        let v_b = v.narrow(0, start, seq_len)?.transpose(0, 1)?;
        let g_b = g.narrow(0, start, seq_len)?.transpose(0, 1)?;
        let beta_b = beta.narrow(0, start, seq_len)?.transpose(0, 1)?;

        let slot = slots_vec[b] as usize;
        let mut state_b = state.i(slot)?; // [num_heads, k_dim, v_dim]

        let out_b =
            gated_delta_rule_recurrence_naive(&q_b, &k_b, &v_b, &g_b, &beta_b, &mut state_b)?; // [num_heads, seq_len, v_dim]

        *state = state.slice_assign(
            &[
                slot..slot + 1,
                0..state.dim(1)?,
                0..state.dim(2)?,
                0..state.dim(3)?,
            ],
            &state_b.unsqueeze(0)?,
        )?;

        outputs.push(out_b.transpose(0, 1)?);
    }

    if outputs.is_empty() {
        return Tensor::zeros((total_tokens, num_heads_q, v_dim), q.dtype(), q.device());
    }

    let refs = outputs.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0)
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
            "gated_delta_rule_recurrence_naive shape mismatch: q={:?}, k={:?}, v={:?}, g={:?}, beta={:?}, state={:?}",
            q.shape(),
            k.shape(),
            v.shape(),
            g.shape(),
            beta.shape(),
            state.shape(),
        );
    }

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
        let delta = v_t
            .broadcast_sub(&kv_mem)?
            .broadcast_mul(&beta_t.unsqueeze(1)?)?; // [bh, v_dim]

        let outer = k_exp.broadcast_mul(&delta.unsqueeze(1)?)?; // [bh, k_dim, v_dim]
        s = (s + outer)?;

        let y_t = s.broadcast_mul(&q_t.unsqueeze(2)?)?.sum(1)?; // [bh, v_dim]
        outputs.push(y_t.unsqueeze(1)?);
    }

    *state = s.to_dtype(state_dtype)?;
    let output_refs = outputs.iter().collect::<Vec<_>>();
    Tensor::cat(&output_refs, 1)?.to_dtype(out_dtype)
}

#[cfg(not(feature = "cuda"))]
fn gated_delta_rule_decode_slots_naive(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
) -> Result<Tensor> {
    let (batch, heads, k_dim) = q.dims3()?;
    let (batch_k, heads_k, k_dim_k) = k.dims3()?;
    let (batch_v, heads_v, v_dim) = v.dims3()?;
    let (batch_g, heads_g) = g.dims2()?;
    let (batch_b, heads_b) = beta.dims2()?;
    let (state_batch, state_heads, state_k_dim, state_v_dim) = state.dims4()?;
    if batch != batch_k
        || batch != batch_v
        || batch != batch_g
        || batch != batch_b
        || heads != heads_k
        || heads != heads_v
        || heads != heads_g
        || heads != heads_b
        || k_dim != k_dim_k
        || heads != state_heads
        || k_dim != state_k_dim
        || v_dim != state_v_dim
    {
        candle_core::bail!(
            "gated_delta_rule_decode_slots_naive shape mismatch: q={:?}, k={:?}, v={:?}, g={:?}, beta={:?}, state={:?}",
            q.shape(),
            k.shape(),
            v.shape(),
            g.shape(),
            beta.shape(),
            state.shape(),
        );
    }
    if slots.dtype() != DType::I64 || slots.dim(0)? != batch {
        candle_core::bail!(
            "gated_delta_rule_decode_slots_naive expects slots [batch] I64, got {:?} {:?}",
            slots.shape(),
            slots.dtype()
        );
    }
    if batch > state_batch {
        candle_core::bail!(
            "gated_delta_rule_decode_slots_naive batch {} exceeds state capacity {}",
            batch,
            state_batch
        );
    }

    let out_dtype = q.dtype();
    let state_dtype = state.dtype();
    let slots_vec = slots.to_vec1::<i64>()?;
    let batch_state = state.index_select(slots, 0)?.to_dtype(DType::F32)?;
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let g = g.to_dtype(DType::F32)?;
    let beta = beta.to_dtype(DType::F32)?;

    let decay = g.exp()?.unsqueeze(2)?.unsqueeze(3)?;
    let mut updated_state = batch_state.broadcast_mul(&decay)?;
    let k_exp = k.unsqueeze(3)?;
    let kv_mem = updated_state.broadcast_mul(&k_exp)?.sum(2)?;
    let delta = v
        .broadcast_sub(&kv_mem)?
        .broadcast_mul(&beta.unsqueeze(2)?)?;
    let outer = k_exp.broadcast_mul(&delta.unsqueeze(2)?)?;
    updated_state = (updated_state + outer)?;
    let output = updated_state
        .broadcast_mul(&q.unsqueeze(3)?)?
        .sum(2)?
        .to_dtype(out_dtype)?;

    let updated_state = updated_state.to_dtype(state_dtype)?;
    for (i, &slot) in slots_vec.iter().enumerate() {
        *state = state.slice_assign(
            &[
                slot as usize..slot as usize + 1,
                0..heads,
                0..k_dim,
                0..v_dim,
            ],
            &updated_state.narrow(0, i, 1)?,
        )?;
    }

    Ok(output)
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
    let out_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let weight = weight.to_dtype(DType::F32)?;
    let bias = match bias {
        Some(b) => Some(b.to_dtype(DType::F32)?),
        None => None,
    };

    let weight_2d = weight.squeeze(1)?; // [d_conv, kernel_size]
    let kernel_size = weight_2d.dim(1)?;
    let d_conv = weight_2d.dim(0)?;
    let seq_len = x.dim(0)?;

    let padding = Tensor::zeros((kernel_size - 1, d_conv), x.dtype(), x.device())?;
    let x_padded = Tensor::cat(&[&padding, &x], 0)?;

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
        output = output.broadcast_add(&bias)?;
    }

    if activation_silu {
        output = candle_nn::ops::silu(&output)?;
    }

    output.to_dtype(out_dtype)
}

/// Naive causal conv1d update for decode (single step).
pub fn causal_conv1d_update_naive(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    let out_dtype = x.dtype();
    let state_dtype = conv_state.dtype();

    let x = x.to_dtype(DType::F32)?;
    let weight = weight.to_dtype(DType::F32)?;
    let bias = match bias {
        Some(b) => Some(b.to_dtype(DType::F32)?),
        None => None,
    };
    let state_f32 = conv_state.to_dtype(DType::F32)?;

    // x: [batch, d_conv], conv_state: [batch, d_conv, kernel_size - 1]
    let weight_2d = weight.squeeze(1)?; // [d_conv, kernel_size]
    let kernel_size = weight_2d.dim(1)?;

    let x_expanded = x.unsqueeze(2)?; // [batch, d_conv, 1]
    let prev_state = state_f32;
    let full_window = Tensor::cat(&[&prev_state, &x_expanded], 2)?; // [batch, d_conv, kernel_size]
    let next_state = full_window.narrow(2, 1, kernel_size - 1)?;
    *conv_state = next_state.to_dtype(state_dtype)?;

    let mut output = full_window
        .broadcast_mul(&weight_2d.unsqueeze(0)?)?
        .sum(2)?; // [batch, d_conv]

    if let Some(bias) = bias {
        output = output.broadcast_add(&bias)?;
    }

    if activation_silu {
        output = candle_nn::ops::silu(&output)?;
    }

    output.to_dtype(out_dtype)
}

/// Naive fused GDN gating.
pub fn fused_gdn_gating_naive(
    a_log: &Tensor,
    a: &Tensor,
    b: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let out_dtype = a.dtype();
    let a_f32 = a.to_dtype(DType::F32)?;
    let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
    let a_log_f32 = a_log.to_dtype(DType::F32)?;
    let b_f32 = b.to_dtype(DType::F32)?;

    // g = -exp(A_log) * softplus(a + dt_bias)
    let a_dt = a_f32.broadcast_add(&dt_bias_f32)?;
    let g = softplus(&a_dt)?.broadcast_mul(&a_log_f32.exp()?.neg()?)?;

    // beta = sigmoid(b)
    let beta = candle_nn::ops::sigmoid(&b_f32)?;
    Ok((g.to_dtype(out_dtype)?, beta.to_dtype(out_dtype)?))
}

pub fn gated_rmsnorm_silu_mul_naive(
    x: &Tensor,
    z: &Tensor,
    norm_weight: &Tensor,
    norm_bias: Option<&Tensor>,
    eps: f64,
    group_size: usize,
) -> Result<Tensor> {
    let (rows, value_dim) = x.dims2()?;
    let (z_rows, z_dim) = z.dims2()?;
    if z_rows != rows || z_dim != value_dim {
        candle_core::bail!(
            "gated_rmsnorm_silu_mul_naive shape mismatch: x={:?}, z={:?}",
            x.shape(),
            z.shape()
        );
    }
    if group_size == 0 || value_dim % group_size != 0 {
        candle_core::bail!(
            "gated_rmsnorm_silu_mul_naive invalid group_size={} for value_dim={}",
            group_size,
            value_dim
        );
    }

    let x_f32 = x.to_dtype(DType::F32)?;
    let z_gate = candle_nn::ops::silu(&z.to_dtype(DType::F32)?)?;
    let groups = value_dim / group_size;
    let per_group_weights = norm_weight.dim(0)? == group_size;
    let full_weights = norm_weight.dim(0)? == value_dim;
    if !per_group_weights && !full_weights {
        candle_core::bail!(
            "gated_rmsnorm_silu_mul_naive invalid weight shape {:?}, expected [{group_size}] or [{value_dim}]",
            norm_weight.shape()
        );
    }

    let x_grouped = x_f32.reshape((rows, groups, group_size))?;
    let variance = (&x_grouped * &x_grouped)?.mean_keepdim(2)?;
    let mut y = x_grouped.broadcast_div(&(variance + eps)?.sqrt()?)?;

    if per_group_weights {
        let w = norm_weight.to_dtype(DType::F32)?;
        y = y.broadcast_mul(&w.reshape((1, 1, group_size))?)?;
        if let Some(b) = norm_bias {
            let b = b.to_dtype(DType::F32)?;
            y = y.broadcast_add(&b.reshape((1, 1, group_size))?)?;
        }
    } else {
        let w = norm_weight
            .to_dtype(DType::F32)?
            .reshape((1, groups, group_size))?;
        y = y.broadcast_mul(&w)?;
        if let Some(b) = norm_bias {
            let b = b.to_dtype(DType::F32)?.reshape((1, groups, group_size))?;
            y = y.broadcast_add(&b)?;
        }
    }

    let y = y.reshape((rows, value_dim))?;
    (y * z_gate)?.to_dtype(x.dtype())
}

/// Softplus: log(1 + exp(x)).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let exp_x = x.exp()?;
    (exp_x + 1.0)?.log()
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
pub fn gated_rmsnorm_silu_mul(
    x: &Tensor,
    z: &Tensor,
    norm_weight: &Tensor,
    norm_bias: Option<&Tensor>,
    eps: f64,
    group_size: usize,
) -> Result<Tensor> {
    gated_rmsnorm_silu_mul_naive(x, z, norm_weight, norm_bias, eps, group_size)
}

#[cfg(not(feature = "cuda"))]
pub fn causal_conv1d_update_slots(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    slots: &Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    let slots_vec = slots.to_vec1::<i64>()?;
    let mut gathered = Vec::with_capacity(slots_vec.len());
    for &s in &slots_vec {
        gathered.push(conv_state.i(s as usize)?);
    }
    let gathered_refs = gathered.iter().collect::<Vec<_>>();
    let mut batch_state = Tensor::stack(&gathered_refs, 0)?;
    let out = causal_conv1d_update_naive(x, weight, bias, &mut batch_state, activation_silu)?;
    for (i, &s) in slots_vec.iter().enumerate() {
        *conv_state = conv_state.slice_assign(
            &[
                s as usize..s as usize + 1,
                0..conv_state.dim(1)?,
                0..conv_state.dim(2)?,
            ],
            &batch_state.narrow(0, i, 1)?,
        )?;
    }
    Ok(out)
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

#[cfg(not(feature = "cuda"))]
pub fn gated_delta_rule_decode_slots(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
) -> Result<Tensor> {
    gated_delta_rule_decode_slots_naive(q, k, v, g, beta, state, slots)
}
