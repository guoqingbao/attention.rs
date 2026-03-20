// GDN (Gated Delta Net) operations module
// Provides Rust interfaces for GDN CUDA kernels used in Qwen3.5's linear attention layers.

#[cfg(feature = "cuda")]
use candle_core as candle;
#[cfg(feature = "metal")]
use candle_core::backend::BackendStorage;
use candle_core::{DType, Result, Tensor};
#[cfg(any(feature = "cuda", feature = "metal"))]
use candle_core::{Device, Storage};
#[cfg(feature = "cuda")]
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use kernels::ffi;
#[cfg(feature = "metal")]
use metal_kernels;
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

#[cfg(feature = "metal")]
#[derive(Clone)]
struct MetalTensorSlice {
    storage: candle_core::MetalStorage,
    offset_in_bytes: usize,
}

#[cfg(feature = "metal")]
fn ensure_contiguous(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

#[cfg(feature = "metal")]
fn get_metal_slice(t: &Tensor) -> Result<MetalTensorSlice> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Metal(s) => Ok(MetalTensorSlice {
            storage: s.clone(),
            offset_in_bytes: layout.start_offset() * t.dtype().size_in_bytes(),
        }),
        _ => candle_core::bail!("Expected Metal tensor"),
    }
}

#[cfg(feature = "metal")]
fn get_metal_slice_with_dtype_size(t: &Tensor, elem_size: usize) -> Result<MetalTensorSlice> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Metal(s) => Ok(MetalTensorSlice {
            storage: s.clone(),
            offset_in_bytes: layout.start_offset() * elem_size,
        }),
        _ => candle_core::bail!("Expected Metal tensor"),
    }
}

#[cfg(feature = "metal")]
pub fn causal_conv1d_fwd(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    cu_seqlens: Option<&Tensor>,
    activation_silu: bool,
) -> Result<Tensor> {
    let cu_seqlens = cu_seqlens.ok_or_else(|| {
        candle_core::Error::msg("metal causal_conv1d_fwd requires cu_seqlens for prefill")
    })?;
    let x_c = ensure_contiguous(x)?;
    let weight_c = ensure_contiguous(weight)?;
    let bias_c = bias.map(ensure_contiguous).transpose()?;
    let cu_u32 = if cu_seqlens.dtype() == DType::U32 {
        ensure_contiguous(cu_seqlens)?
    } else {
        cu_seqlens.to_dtype(DType::U32)?.contiguous()?
    };
    if !conv_state.is_contiguous() {
        candle_core::bail!("metal causal_conv1d_fwd expects contiguous conv_state");
    }
    if x_c.dtype() != weight_c.dtype() || conv_state.dtype() != x_c.dtype() {
        candle_core::bail!(
            "metal causal_conv1d_fwd dtype mismatch: x={:?}, weight={:?}, state={:?}",
            x_c.dtype(),
            weight_c.dtype(),
            conv_state.dtype()
        );
    }

    let (total_tokens, d_conv) = x_c.dims2()?;
    let kernel_size = weight_c.dim(2)?;
    let batch = conv_state.dim(0)?;
    let out = Tensor::zeros((total_tokens, d_conv), x_c.dtype(), x_c.device())?;

    let x_m = get_metal_slice(&x_c)?;
    let weight_m = get_metal_slice(&weight_c)?;
    let bias_m = bias_c.as_ref().map(get_metal_slice).transpose()?;
    let state_m = get_metal_slice(conv_state)?;
    let out_m = get_metal_slice(&out)?;
    let cu_m = get_metal_slice_with_dtype_size(&cu_u32, std::mem::size_of::<u32>())?;

    let dev = x_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("gdn-causal-conv1d-fwd");
    metal_kernels::call_gdn_causal_conv1d_fwd(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        x_c.dtype(),
        x_m.storage.buffer(),
        x_m.offset_in_bytes,
        weight_m.storage.buffer(),
        weight_m.offset_in_bytes,
        bias_m
            .as_ref()
            .map(|b| (b.storage.buffer(), b.offset_in_bytes)),
        state_m.storage.buffer(),
        state_m.offset_in_bytes,
        out_m.storage.buffer(),
        out_m.offset_in_bytes,
        cu_m.storage.buffer(),
        cu_m.offset_in_bytes,
        batch as i32,
        d_conv as i32,
        kernel_size as i32,
        activation_silu,
    )
    .map_err(candle_core::Error::wrap)?;

    Ok(out)
}

#[cfg(feature = "metal")]
pub fn causal_conv1d_update(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    let x_c = ensure_contiguous(x)?;
    let weight_c = ensure_contiguous(weight)?;
    let bias_c = bias.map(ensure_contiguous).transpose()?;
    if !conv_state.is_contiguous() {
        candle_core::bail!("metal causal_conv1d_update expects contiguous conv_state");
    }
    if x_c.dtype() != weight_c.dtype() || conv_state.dtype() != x_c.dtype() {
        candle_core::bail!(
            "metal causal_conv1d_update dtype mismatch: x={:?}, weight={:?}, state={:?}",
            x_c.dtype(),
            weight_c.dtype(),
            conv_state.dtype()
        );
    }

    let (batch, d_conv) = x_c.dims2()?;
    let kernel_size = weight_c.dim(2)?;
    let out = Tensor::zeros((batch, d_conv), x_c.dtype(), x_c.device())?;

    let x_m = get_metal_slice(&x_c)?;
    let weight_m = get_metal_slice(&weight_c)?;
    let bias_m = bias_c.as_ref().map(get_metal_slice).transpose()?;
    let state_m = get_metal_slice(conv_state)?;
    let out_m = get_metal_slice(&out)?;

    let dev = x_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("gdn-causal-conv1d-update");
    metal_kernels::call_gdn_causal_conv1d_update(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        x_c.dtype(),
        x_m.storage.buffer(),
        x_m.offset_in_bytes,
        weight_m.storage.buffer(),
        weight_m.offset_in_bytes,
        bias_m
            .as_ref()
            .map(|b| (b.storage.buffer(), b.offset_in_bytes)),
        state_m.storage.buffer(),
        state_m.offset_in_bytes,
        out_m.storage.buffer(),
        out_m.offset_in_bytes,
        batch as i32,
        d_conv as i32,
        kernel_size as i32,
        activation_silu,
    )
    .map_err(candle_core::Error::wrap)?;

    Ok(out)
}

#[cfg(feature = "metal")]
pub fn causal_conv1d_update_slots(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    slots: &Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    let x_c = ensure_contiguous(x)?;
    let weight_c = ensure_contiguous(weight)?;
    let bias_c = bias.map(ensure_contiguous).transpose()?;
    let slots_c = if slots.dtype() == DType::I64 {
        ensure_contiguous(slots)?
    } else {
        candle_core::bail!("metal causal_conv1d_update_slots expects I64 slots");
    };
    if !conv_state.is_contiguous() {
        candle_core::bail!("metal causal_conv1d_update_slots expects contiguous conv_state");
    }
    if x_c.dtype() != weight_c.dtype() || conv_state.dtype() != x_c.dtype() {
        candle_core::bail!(
            "metal causal_conv1d_update_slots dtype mismatch: x={:?}, weight={:?}, state={:?}",
            x_c.dtype(),
            weight_c.dtype(),
            conv_state.dtype()
        );
    }

    let (batch, d_conv) = x_c.dims2()?;
    if slots_c.dim(0)? != batch {
        candle_core::bail!(
            "metal causal_conv1d_update_slots expects slots [batch], got {:?}",
            slots_c.shape()
        );
    }
    let kernel_size = weight_c.dim(2)?;
    let out = Tensor::zeros((batch, d_conv), x_c.dtype(), x_c.device())?;

    let x_m = get_metal_slice(&x_c)?;
    let weight_m = get_metal_slice(&weight_c)?;
    let bias_m = bias_c.as_ref().map(get_metal_slice).transpose()?;
    let state_m = get_metal_slice(conv_state)?;
    let slots_m = get_metal_slice_with_dtype_size(&slots_c, std::mem::size_of::<i64>())?;
    let out_m = get_metal_slice(&out)?;

    let dev = x_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("gdn-causal-conv1d-update-slots");
    metal_kernels::call_gdn_causal_conv1d_update_slots(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        x_c.dtype(),
        x_m.storage.buffer(),
        x_m.offset_in_bytes,
        weight_m.storage.buffer(),
        weight_m.offset_in_bytes,
        bias_m
            .as_ref()
            .map(|b| (b.storage.buffer(), b.offset_in_bytes)),
        state_m.storage.buffer(),
        state_m.offset_in_bytes,
        slots_m.storage.buffer(),
        slots_m.offset_in_bytes,
        out_m.storage.buffer(),
        out_m.offset_in_bytes,
        batch as i32,
        d_conv as i32,
        kernel_size as i32,
        activation_silu,
    )
    .map_err(candle_core::Error::wrap)?;

    Ok(out)
}

#[cfg(feature = "metal")]
pub fn fused_gdn_gating(
    a_log: &Tensor,
    a: &Tensor,
    b: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let a_c = ensure_contiguous(a)?;
    let b_c = if b.dtype() == a_c.dtype() {
        ensure_contiguous(b)?
    } else {
        b.to_dtype(a_c.dtype())?.contiguous()?
    };
    let dt_c = if dt_bias.dtype() == a_c.dtype() {
        ensure_contiguous(dt_bias)?
    } else {
        dt_bias.to_dtype(a_c.dtype())?.contiguous()?
    };
    let a_log_c = if a_log.dtype() == DType::F32 || a_log.dtype() == a_c.dtype() {
        ensure_contiguous(a_log)?
    } else {
        candle_core::bail!(
            "metal fused_gdn_gating expects a_log dtype {:?} or F32, got {:?}",
            a_c.dtype(),
            a_log.dtype()
        );
    };
    let (batch, seq_len, heads) = a_c.dims3()?;
    if b_c.shape() != a_c.shape() {
        candle_core::bail!(
            "metal fused_gdn_gating shape mismatch: a={:?}, b={:?}",
            a_c.shape(),
            b_c.shape()
        );
    }
    if dt_c.dim(0)? != heads || a_log_c.dim(0)? != heads {
        candle_core::bail!(
            "metal fused_gdn_gating expects head-sized a_log/dt_bias, got a_log={:?}, dt_bias={:?}, heads={heads}",
            a_log_c.shape(),
            dt_c.shape()
        );
    }
    let g = Tensor::zeros(a_c.shape(), a_c.dtype(), a_c.device())?;
    let beta = Tensor::zeros(a_c.shape(), a_c.dtype(), a_c.device())?;

    let a_log_m = get_metal_slice(&a_log_c)?;
    let a_m = get_metal_slice(&a_c)?;
    let b_m = get_metal_slice(&b_c)?;
    let dt_m = get_metal_slice(&dt_c)?;
    let g_m = get_metal_slice(&g)?;
    let beta_m = get_metal_slice(&beta)?;
    let dev = a_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("gdn-fused-gating");
    metal_kernels::call_gdn_fused_gating(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        a_c.dtype(),
        a_log_c.dtype(),
        a_log_m.storage.buffer(),
        a_log_m.offset_in_bytes,
        a_m.storage.buffer(),
        a_m.offset_in_bytes,
        b_m.storage.buffer(),
        b_m.offset_in_bytes,
        dt_m.storage.buffer(),
        dt_m.offset_in_bytes,
        g_m.storage.buffer(),
        g_m.offset_in_bytes,
        beta_m.storage.buffer(),
        beta_m.offset_in_bytes,
        (batch * seq_len * heads) as i32,
        heads as i32,
    )
    .map_err(candle_core::Error::wrap)?;
    Ok((g, beta))
}

#[cfg(feature = "metal")]
pub fn gated_rmsnorm_silu_mul(
    x: &Tensor,
    z: &Tensor,
    norm_weight: &Tensor,
    norm_bias: Option<&Tensor>,
    eps: f64,
    group_size: usize,
) -> Result<Tensor> {
    let x_c = ensure_contiguous(x)?;
    let z_c = if z.dtype() == x_c.dtype() {
        ensure_contiguous(z)?
    } else {
        z.to_dtype(x_c.dtype())?.contiguous()?
    };
    let norm_weight_c = ensure_contiguous(norm_weight)?;
    let norm_bias_c = norm_bias.map(ensure_contiguous).transpose()?;

    let (rows, value_dim) = x_c.dims2()?;
    let (z_rows, z_dim) = z_c.dims2()?;
    if z_rows != rows || z_dim != value_dim {
        candle_core::bail!(
            "metal gated_rmsnorm_silu_mul shape mismatch: x={:?}, z={:?}",
            x_c.shape(),
            z_c.shape()
        );
    }
    if group_size == 0 || value_dim % group_size != 0 {
        candle_core::bail!(
            "metal gated_rmsnorm_silu_mul invalid group_size={} for value_dim={}",
            group_size,
            value_dim
        );
    }
    let weight_len = norm_weight_c.dim(0)?;
    let per_group_weights = if weight_len == group_size {
        true
    } else if weight_len == value_dim {
        false
    } else {
        candle_core::bail!(
            "metal gated_rmsnorm_silu_mul invalid weight shape {:?}",
            norm_weight_c.shape()
        );
    };
    if let Some(ref bias_c) = norm_bias_c {
        let expected = if per_group_weights {
            group_size
        } else {
            value_dim
        };
        if bias_c.dim(0)? != expected {
            candle_core::bail!(
                "metal gated_rmsnorm_silu_mul invalid bias shape {:?}, expected [{}]",
                bias_c.shape(),
                expected
            );
        }
    }
    if !(norm_weight_c.dtype() == x_c.dtype() || norm_weight_c.dtype() == DType::F32) {
        candle_core::bail!(
            "metal gated_rmsnorm_silu_mul unsupported weight dtype {:?} for input {:?}",
            norm_weight_c.dtype(),
            x_c.dtype()
        );
    }

    let out = Tensor::zeros((rows, value_dim), x_c.dtype(), x_c.device())?;
    let x_m = get_metal_slice(&x_c)?;
    let z_m = get_metal_slice(&z_c)?;
    let w_m = get_metal_slice(&norm_weight_c)?;
    let b_m = norm_bias_c.as_ref().map(get_metal_slice).transpose()?;
    let out_m = get_metal_slice(&out)?;
    let dev = x_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("gdn-gated-rmsnorm-silu-mul");
    metal_kernels::call_gdn_gated_rmsnorm_silu_mul(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        x_c.dtype(),
        norm_weight_c.dtype(),
        x_m.storage.buffer(),
        x_m.offset_in_bytes,
        z_m.storage.buffer(),
        z_m.offset_in_bytes,
        w_m.storage.buffer(),
        w_m.offset_in_bytes,
        b_m.as_ref()
            .map(|b| (b.storage.buffer(), b.offset_in_bytes)),
        out_m.storage.buffer(),
        out_m.offset_in_bytes,
        rows as i32,
        value_dim as i32,
        group_size as i32,
        eps as f32,
        per_group_weights,
        norm_bias_c.is_some(),
    )
    .map_err(candle_core::Error::wrap)?;
    Ok(out)
}

#[cfg(feature = "metal")]
pub fn l2_norm_last_dim(input: &Tensor, eps: f64) -> Result<Tensor> {
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
    let output = Tensor::zeros(shape, input_c.dtype(), input_c.device())?;
    let in_m = get_metal_slice(&input_c)?;
    let out_m = get_metal_slice(&output)?;
    let dev = in_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("gdn-l2-norm-last-dim");
    metal_kernels::call_gdn_l2_norm_last_dim(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        input_c.dtype(),
        in_m.storage.buffer(),
        in_m.offset_in_bytes,
        out_m.storage.buffer(),
        out_m.offset_in_bytes,
        rows as i32,
        dim as i32,
        eps as f32,
    )
    .map_err(candle_core::Error::wrap)?;
    Ok(output)
}

#[cfg(feature = "metal")]
pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let q_c = ensure_contiguous(q)?;
    let k_c = ensure_contiguous(k)?;
    let v_c = ensure_contiguous(v)?;
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
    if state.dtype() != DType::F32 || !state.is_contiguous() {
        candle_core::bail!(
            "metal gated_delta_rule_recurrence expects contiguous F32 state, got {:?}",
            state.dtype()
        );
    }

    let (bh, seq_len, k_dim) = q_c.dims3()?;
    let v_dim = v_c.dim(2)?;
    let out = Tensor::zeros((bh, seq_len, v_dim), q_c.dtype(), q_c.device())?;
    let q_m = get_metal_slice(&q_c)?;
    let k_m = get_metal_slice(&k_c)?;
    let v_m = get_metal_slice(&v_c)?;
    let g_m = get_metal_slice(&g_f32)?;
    let beta_m = get_metal_slice(&beta_f32)?;
    let state_m = get_metal_slice(state)?;
    let out_m = get_metal_slice(&out)?;
    let dev = q_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("gdn-recurrence");
    metal_kernels::call_gdn_gated_delta_rule_recurrence(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        q_c.dtype(),
        q_m.storage.buffer(),
        q_m.offset_in_bytes,
        k_m.storage.buffer(),
        k_m.offset_in_bytes,
        v_m.storage.buffer(),
        v_m.offset_in_bytes,
        g_m.storage.buffer(),
        g_m.offset_in_bytes,
        beta_m.storage.buffer(),
        beta_m.offset_in_bytes,
        state_m.storage.buffer(),
        state_m.offset_in_bytes,
        out_m.storage.buffer(),
        out_m.offset_in_bytes,
        bh as i32,
        seq_len as i32,
        k_dim as i32,
        v_dim as i32,
    )
    .map_err(candle_core::Error::wrap)?;
    Ok(out)
}

#[cfg(feature = "metal")]
pub fn gated_delta_rule_decode_slots(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
) -> Result<Tensor> {
    let q_c = ensure_contiguous(q)?;
    let k_c = ensure_contiguous(k)?;
    let v_c = ensure_contiguous(v)?;
    let g_c = ensure_contiguous(g)?;
    let beta_c = ensure_contiguous(beta)?;
    let slots_c = if slots.dtype() == DType::I64 {
        ensure_contiguous(slots)?
    } else {
        candle_core::bail!("metal gated_delta_rule_decode_slots expects I64 slots");
    };
    if state.dtype() != DType::F32 || !state.is_contiguous() {
        candle_core::bail!(
            "metal gated_delta_rule_decode_slots expects contiguous F32 state, got {:?}",
            state.dtype()
        );
    }

    let (batch, heads, k_dim) = q_c.dims3()?;
    let v_dim = v_c.dim(2)?;
    let out = Tensor::zeros((batch, heads, v_dim), q_c.dtype(), q_c.device())?;
    let q_m = get_metal_slice(&q_c)?;
    let k_m = get_metal_slice(&k_c)?;
    let v_m = get_metal_slice(&v_c)?;
    let g_m = get_metal_slice(&g_c)?;
    let beta_m = get_metal_slice(&beta_c)?;
    let state_m = get_metal_slice(state)?;
    let slots_m = get_metal_slice_with_dtype_size(&slots_c, std::mem::size_of::<i64>())?;
    let out_m = get_metal_slice(&out)?;
    let dev = q_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("gdn-decode-slots");
    metal_kernels::call_gdn_gated_delta_rule_decode_slots(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        q_c.dtype(),
        q_m.storage.buffer(),
        q_m.offset_in_bytes,
        k_m.storage.buffer(),
        k_m.offset_in_bytes,
        v_m.storage.buffer(),
        v_m.offset_in_bytes,
        g_m.storage.buffer(),
        g_m.offset_in_bytes,
        beta_m.storage.buffer(),
        beta_m.offset_in_bytes,
        state_m.storage.buffer(),
        state_m.offset_in_bytes,
        slots_m.storage.buffer(),
        slots_m.offset_in_bytes,
        out_m.storage.buffer(),
        out_m.offset_in_bytes,
        batch as i32,
        heads as i32,
        k_dim as i32,
        v_dim as i32,
    )
    .map_err(candle_core::Error::wrap)?;
    Ok(out)
}

#[cfg(feature = "metal")]
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
    let q_c = ensure_contiguous(q)?;
    let k_c = ensure_contiguous(k)?;
    let v_c = ensure_contiguous(v)?;
    let g_c = ensure_contiguous(g)?;
    let beta_c = ensure_contiguous(beta)?;
    let slots_c = if slots.dtype() == DType::I64 {
        ensure_contiguous(slots)?
    } else {
        candle_core::bail!("metal gated_delta_rule_recurrence_varlen expects I64 slots");
    };
    let cu_u32 = if cu_seqlens.dtype() == DType::U32 {
        ensure_contiguous(cu_seqlens)?
    } else {
        cu_seqlens.to_dtype(DType::U32)?.contiguous()?
    };
    if state.dtype() != DType::F32 || !state.is_contiguous() {
        candle_core::bail!(
            "metal gated_delta_rule_recurrence_varlen expects contiguous F32 state, got {:?}",
            state.dtype()
        );
    }

    let (total_tokens, num_heads, k_dim) = q_c.dims3()?;
    let v_dim = v_c.dim(2)?;
    let batch = slots_c.dim(0)?;
    let out = Tensor::zeros((total_tokens, num_heads, v_dim), q_c.dtype(), q_c.device())?;

    let q_m = get_metal_slice(&q_c)?;
    let k_m = get_metal_slice(&k_c)?;
    let v_m = get_metal_slice(&v_c)?;
    let g_m = get_metal_slice(&g_c)?;
    let beta_m = get_metal_slice(&beta_c)?;
    let state_m = get_metal_slice(state)?;
    let slots_m = get_metal_slice_with_dtype_size(&slots_c, std::mem::size_of::<i64>())?;
    let out_m = get_metal_slice(&out)?;
    let cu_m = get_metal_slice_with_dtype_size(&cu_u32, std::mem::size_of::<u32>())?;
    let dev = q_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("gdn-recurrence-varlen");
    metal_kernels::call_gdn_gated_delta_rule_recurrence_varlen(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        q_c.dtype(),
        q_m.storage.buffer(),
        q_m.offset_in_bytes,
        k_m.storage.buffer(),
        k_m.offset_in_bytes,
        v_m.storage.buffer(),
        v_m.offset_in_bytes,
        g_m.storage.buffer(),
        g_m.offset_in_bytes,
        beta_m.storage.buffer(),
        beta_m.offset_in_bytes,
        state_m.storage.buffer(),
        state_m.offset_in_bytes,
        slots_m.storage.buffer(),
        slots_m.offset_in_bytes,
        out_m.storage.buffer(),
        out_m.offset_in_bytes,
        cu_m.storage.buffer(),
        cu_m.offset_in_bytes,
        batch as i32,
        num_heads as i32,
        k_dim as i32,
        v_dim as i32,
    )
    .map_err(candle_core::Error::wrap)?;
    Ok(out)
}

/// Causal conv1d forward pass for variable-length sequences (prefill mode).
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
                candle_core::bail!(
                    "causal_conv1d_fwd only supports kernel_size <= 16 on CUDA, got {}",
                    kernel_size
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
        _ => {
            candle_core::bail!(
                "Invalid tensor device {:?} for causal_conv1d_fwd",
                x.device()
            );
        }
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
        _ => {
            candle_core::bail!(
                "Invalid tensor device {:?} for causal_conv1d_update",
                x.device()
            );
        }
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
            candle_core::bail!(
                "Invalid tensor device {:?} for causal_conv1d_update_slots",
                x.device()
            );
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
        _ => {
            candle_core::bail!(
                "Invalid tensor device {:?} for fused_gdn_gating",
                a.device()
            );
        }
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
        _ => {
            candle_core::bail!(
                "Invalid tensor device {:?} for gated_rmsnorm_silu_mul",
                x.device()
            );
        }
    }
}

/// DeltaNet recurrent update over flattened batch-head (`BH`) dimension.
///
/// Shapes:
/// - `q`, `k`: `[bh, seq, k_dim]`
/// - `v`: `[bh, seq, v_dim]`
/// - `g`, `beta`: `[bh, seq]`
/// - CUDA `state`: `[bh, v_dim, k_dim]` (updated in place)
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
                let (b, h, v, k) = state.dims4()?;
                (b * h, k, v)
            } else {
                let (bh_s, v_dim_s, k_dim_s) = state.dims3()?;
                (bh_s, k_dim_s, v_dim_s)
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
            let decay_f32 = if g.dtype() == DType::F32 {
                ensure_contiguous(g)?.exp()?.contiguous()?
            } else {
                g.to_dtype(DType::F32)?.exp()?.contiguous()?
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
            let g_ptr = get_cuda_const_ptr(&decay_f32)? as *const f32;
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
        _ => {
            candle_core::bail!(
                "Invalid tensor device {:?} for gated_delta_rule_recurrence",
                q.device()
            );
        }
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
    state: &mut Tensor, // CUDA: [max_batch, heads, v_dim, k_dim]
    slots: &Tensor, // [batch] i64
) -> Result<Tensor> {
    match q.device() {
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let decay_c = ensure_contiguous(g)?.exp()?.contiguous()?;
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
                let g_ptr = get_cuda_const_ptr(&decay_c)? as *const f32;
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
                let g_ptr = get_cuda_const_ptr(&decay_c)?;
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
            candle_core::bail!(
                "Invalid tensor device {:?} for gated_delta_rule_decode_slots",
                q.device()
            );
        }
    }
}

/// Fused L2 normalization over the last dimension.
/// Replaces the multi-op sequence: sumsq → sqrt → clamp → div.
/// input: [rows, dim] → output: [rows, dim] (each row normalized to unit L2 norm)
#[cfg(feature = "cuda")]
pub fn l2_norm_last_dim(input: &Tensor, eps: f64) -> Result<Tensor> {
    match input.device() {
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
            candle_core::bail!("Invalid tensor device!");
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
/// - CUDA `state`: `[max_batch, num_heads, v_dim, k_dim]` (FP32, updated in place)
/// - `slots`: `[batch]` i64
/// - `cu_seqlens`: `[batch + 1]` u32
#[cfg(feature = "cuda")]
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
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let decay_c = ensure_contiguous(g)?.exp()?.contiguous()?;
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
            let g_ptr = get_cuda_const_ptr(&decay_c)?;
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
        _ => {
            candle_core::bail!(
                "Invalid tensor device {:?} for gated_delta_rule_recurrence_varlen",
                q.device()
            );
        }
    }
}
