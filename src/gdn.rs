// GDN (Gated Delta Net) operations module
// Provides Rust interfaces for GDN kernels used in Qwen3.5's linear attention layers.

#[cfg(feature = "cuda")]
use candle_core as candle;
#[cfg(feature = "gcu")]
use candle_core as candle;
#[cfg(feature = "metal")]
use candle_core::backend::BackendStorage;
use candle_core::{DType, Result, Tensor};
#[cfg(any(feature = "cuda", feature = "metal", feature = "gcu"))]
use candle_core::{Device, Storage};
#[cfg(feature = "cuda")]
use half::{bf16, f16};
#[cfg(feature = "gcu")]
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use kernels::ffi;
#[cfg(feature = "metal")]
use metal_kernels;
#[cfg(feature = "gcu")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::ffi::{c_int, c_void};

#[cfg(feature = "cuda")]
fn cuda_full_write_output<S: Into<candle_core::Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    // Caller must launch a CUDA kernel that writes every element before this tensor is read.
    unsafe { Tensor::empty_(shape, dtype, device) }
}

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

#[cfg(feature = "cuda")]
fn ensure_f32_contiguous(t: &Tensor) -> Result<Tensor> {
    let t = if t.dtype() == DType::F32 {
        t.clone()
    } else {
        t.to_dtype(DType::F32)?
    };
    ensure_contiguous(&t)
}

#[cfg(feature = "cuda")]
fn exp_f32_contiguous(t: &Tensor) -> Result<Tensor> {
    ensure_f32_contiguous(t)?.exp()?.contiguous()
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
    state_snapshots: Option<&Tensor>,
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
    if x_c.dtype() != weight_c.dtype() {
        candle_core::bail!(
            "metal causal_conv1d_fwd dtype mismatch: x={:?}, weight={:?}",
            x_c.dtype(),
            weight_c.dtype(),
        );
    }
    if !matches!(x_c.dtype(), DType::BF16 | DType::F32) {
        candle_core::bail!(
            "metal causal_conv1d_fwd expects BF16 or F32 x/weight, got {:?}",
            x_c.dtype()
        );
    }
    if conv_state.dtype() != DType::F32 {
        candle_core::bail!(
            "metal causal_conv1d_fwd expects F32 conv_state, got {:?}",
            conv_state.dtype()
        );
    }

    let (total_tokens, d_conv) = x_c.dims2()?;
    let kernel_size = weight_c.dim(2)?;
    let batch = conv_state.dim(0)?;
    let out = Tensor::zeros((total_tokens, d_conv), x_c.dtype(), x_c.device())?;

    let snap_c = state_snapshots.map(ensure_contiguous).transpose()?;
    if let Some(ref s) = snap_c {
        if s.dtype() != x_c.dtype() {
            candle_core::bail!(
                "metal causal_conv1d_fwd: state_snapshots dtype {:?} != x dtype {:?}",
                s.dtype(),
                x_c.dtype()
            );
        }
    }

    let x_m = get_metal_slice(&x_c)?;
    let weight_m = get_metal_slice(&weight_c)?;
    let bias_m = bias_c.as_ref().map(get_metal_slice).transpose()?;
    let state_m = get_metal_slice(conv_state)?;
    let out_m = get_metal_slice(&out)?;
    let snap_m = snap_c.as_ref().map(get_metal_slice).transpose()?;
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
        snap_m
            .as_ref()
            .map(|s| (s.storage.buffer(), s.offset_in_bytes)),
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
    if x_c.dtype() != weight_c.dtype() {
        candle_core::bail!(
            "metal causal_conv1d_update dtype mismatch: x={:?}, weight={:?}",
            x_c.dtype(),
            weight_c.dtype(),
        );
    }
    if !matches!(x_c.dtype(), DType::BF16 | DType::F32) {
        candle_core::bail!(
            "metal causal_conv1d_update expects BF16 or F32 x/weight, got {:?}",
            x_c.dtype()
        );
    }
    if conv_state.dtype() != DType::F32 {
        candle_core::bail!(
            "metal causal_conv1d_update expects F32 conv_state, got {:?}",
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
    if x_c.dtype() != weight_c.dtype() {
        candle_core::bail!(
            "metal causal_conv1d_update_slots dtype mismatch: x={:?}, weight={:?}",
            x_c.dtype(),
            weight_c.dtype(),
        );
    }
    if !matches!(x_c.dtype(), DType::BF16 | DType::F32) {
        candle_core::bail!(
            "metal causal_conv1d_update_slots expects BF16 or F32 x/weight, got {:?}",
            x_c.dtype()
        );
    }
    if conv_state.dtype() != DType::F32 {
        candle_core::bail!(
            "metal causal_conv1d_update_slots expects F32 conv_state, got {:?}",
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
    if a_log.dtype() != DType::F32 {
        candle_core::bail!(
            "metal fused_gdn_gating expects F32 a_log, got {:?}",
            a_log.dtype()
        );
    }
    if dt_bias.dtype() != DType::F32 {
        candle_core::bail!(
            "metal fused_gdn_gating expects F32 dt_bias, got {:?}",
            dt_bias.dtype()
        );
    }
    let a_log_c = ensure_contiguous(a_log)?;
    let dt_c = ensure_contiguous(dt_bias)?;
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
    let g = Tensor::zeros(a_c.shape(), DType::F32, a_c.device())?;
    let beta = Tensor::zeros(a_c.shape(), DType::F32, a_c.device())?;

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
    let g_f32 = g.exp()?;
    let beta_f32 = ensure_contiguous(beta)?;
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
    let g_c = g.exp()?;
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
pub fn gated_delta_rule_decode_slots_gqa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
    q_scale: f32,
) -> Result<Tensor> {
    let q_c = ensure_contiguous(q)?;
    let k_c = ensure_contiguous(k)?;
    let v_c = ensure_contiguous(v)?;
    let g_c = ensure_contiguous(g)?;
    let beta_c = ensure_contiguous(beta)?;

    let (batch, num_k_heads, k_dim) = q_c.dims3()?;
    let num_v_heads = v_c.dim(1)?;
    let kv_group_size = num_v_heads / num_k_heads;

    let q_exp = q_c
        .unsqueeze(2)?
        .broadcast_as((batch, num_k_heads, kv_group_size, k_dim))?
        .reshape((batch, num_v_heads, k_dim))?
        .contiguous()?;
    let k_exp = k_c
        .unsqueeze(2)?
        .broadcast_as((batch, num_k_heads, kv_group_size, k_dim))?
        .reshape((batch, num_v_heads, k_dim))?
        .contiguous()?;
    let q_scaled = (q_exp * q_scale as f64)?;

    gated_delta_rule_decode_slots(&q_scaled, &k_exp, &v_c, &g_c, &beta_c, state, slots)
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
    state_snapshots: Option<&Tensor>,
) -> Result<Tensor> {
    let q_c = ensure_contiguous(q)?;
    let k_c = ensure_contiguous(k)?;
    let v_c = ensure_contiguous(v)?;
    let g_c = g.exp()?;
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

    if let Some(snap) = state_snapshots {
        if snap.dtype() != DType::F32 {
            candle_core::bail!(
                "metal gated_delta_rule_recurrence_varlen: state_snapshots must be F32, got {:?}",
                snap.dtype()
            );
        }
    }
    let snap_c = state_snapshots.map(ensure_contiguous).transpose()?;

    let q_m = get_metal_slice(&q_c)?;
    let k_m = get_metal_slice(&k_c)?;
    let v_m = get_metal_slice(&v_c)?;
    let g_m = get_metal_slice(&g_c)?;
    let beta_m = get_metal_slice(&beta_c)?;
    let state_m = get_metal_slice(state)?;
    let slots_m = get_metal_slice_with_dtype_size(&slots_c, std::mem::size_of::<i64>())?;
    let out_m = get_metal_slice(&out)?;
    let snap_m = snap_c.as_ref().map(get_metal_slice).transpose()?;
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
        snap_m
            .as_ref()
            .map(|s| (s.storage.buffer(), s.offset_in_bytes)),
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

/// GQA variant of varlen recurrence for Metal: q/k have num_k_heads, v/g/beta/state/out have num_v_heads.
/// Fuses q_scale multiplication into the kernel to avoid separate allocation.
#[cfg(feature = "metal")]
pub fn gated_delta_rule_recurrence_varlen_gqa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
    cu_seqlens: &Tensor,
    q_scale: f32,
    state_snapshots: Option<&Tensor>,
) -> Result<Tensor> {
    let q_c = ensure_contiguous(q)?;
    let k_c = ensure_contiguous(k)?;
    let v_c = ensure_contiguous(v)?;
    let g_c = ensure_contiguous(g)?;
    let beta_c = ensure_contiguous(beta)?;
    let slots_c = if slots.dtype() == DType::I64 {
        ensure_contiguous(slots)?
    } else {
        candle_core::bail!("metal gated_delta_rule_recurrence_varlen_gqa expects I64 slots");
    };
    let cu_u32 = if cu_seqlens.dtype() == DType::U32 {
        ensure_contiguous(cu_seqlens)?
    } else {
        cu_seqlens.to_dtype(DType::U32)?.contiguous()?
    };
    if state.dtype() != DType::F32 || !state.is_contiguous() {
        candle_core::bail!(
            "metal gated_delta_rule_recurrence_varlen_gqa expects contiguous F32 state, got {:?}",
            state.dtype()
        );
    }
    if let Some(snap) = state_snapshots {
        if snap.dtype() != DType::F32 {
            candle_core::bail!(
                "metal gated_delta_rule_recurrence_varlen_gqa: state_snapshots must be F32, got {:?}",
                snap.dtype()
            );
        }
    }
    let snap_c = state_snapshots.map(ensure_contiguous).transpose()?;

    let (total_tokens, num_k_heads, k_dim) = q_c.dims3()?;
    let num_v_heads = v_c.dim(1)?;
    let v_dim = v_c.dim(2)?;
    let batch = slots_c.dim(0)?;

    if num_v_heads % num_k_heads != 0 {
        candle_core::bail!(
            "metal gated_delta_rule_recurrence_varlen_gqa: num_v_heads {} not divisible by num_k_heads {}",
            num_v_heads,
            num_k_heads
        );
    }

    let out = Tensor::zeros(
        (total_tokens, num_v_heads, v_dim),
        q_c.dtype(),
        q_c.device(),
    )?;

    let q_m = get_metal_slice(&q_c)?;
    let k_m = get_metal_slice(&k_c)?;
    let v_m = get_metal_slice(&v_c)?;
    let g_m = get_metal_slice(&g_c)?;
    let beta_m = get_metal_slice(&beta_c)?;
    let state_m = get_metal_slice(state)?;
    let slots_m = get_metal_slice_with_dtype_size(&slots_c, std::mem::size_of::<i64>())?;
    let out_m = get_metal_slice(&out)?;
    let snap_m = snap_c.as_ref().map(get_metal_slice).transpose()?;
    let cu_m = get_metal_slice_with_dtype_size(&cu_u32, std::mem::size_of::<u32>())?;
    let dev = q_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("gdn-recurrence-varlen-gqa");
    metal_kernels::call_gdn_gated_delta_rule_recurrence_varlen_gqa(
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
        snap_m
            .as_ref()
            .map(|s| (s.storage.buffer(), s.offset_in_bytes)),
        cu_m.storage.buffer(),
        cu_m.offset_in_bytes,
        batch as i32,
        num_v_heads as i32,
        num_k_heads as i32,
        k_dim as i32,
        v_dim as i32,
        q_scale,
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
    state_snapshots: Option<&Tensor>,
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
            let out = cuda_full_write_output((total_tokens, d_conv), x.dtype(), x.device())?;
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
            let snapshots_ptr = if let Some(snapshots) = state_snapshots {
                if snapshots.dtype() != x.dtype() {
                    candle_core::bail!(
                        "causal_conv1d_fwd snapshot dtype mismatch: snapshots={:?}, x={:?}",
                        snapshots.dtype(),
                        x.dtype()
                    );
                }
                if snapshots.dims() != [total_tokens, d_conv, kernel_size.saturating_sub(1)] {
                    candle_core::bail!(
                        "causal_conv1d_fwd snapshot shape {:?} != expected [{}, {}, {}]",
                        snapshots.shape(),
                        total_tokens,
                        d_conv,
                        kernel_size.saturating_sub(1)
                    );
                }
                get_cuda_mut_ptr(snapshots)?
            } else {
                std::ptr::null_mut()
            };
            let cu_ptr = get_cuda_const_ptr_u32(&cu_u32)?;
            let stream = *dev.cu_stream() as i64;

            let state_f32_ptr = state_ptr as *mut f32;
            unsafe {
                match x.dtype() {
                    DType::F16 => ffi::causal_conv1d_fwd_f16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_f32_ptr,
                        out_ptr,
                        snapshots_ptr,
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
                        state_f32_ptr,
                        out_ptr,
                        snapshots_ptr,
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
                        state_f32_ptr,
                        out_ptr as *mut f32,
                        snapshots_ptr as *mut f32,
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
            let out = cuda_full_write_output((batch, d_conv), x.dtype(), x.device())?;

            let x_ptr = get_cuda_const_ptr(x)?;
            let weight_ptr = get_cuda_const_ptr(weight)?;
            let bias_ptr = if let Some(b) = bias {
                get_cuda_const_ptr(b)?
            } else {
                std::ptr::null()
            };
            let state_ptr = get_cuda_mut_ptr(conv_state)?;
            let state_f32_ptr = state_ptr as *mut f32;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let stream = *dev.cu_stream() as i64;

            unsafe {
                match x.dtype() {
                    DType::F16 => ffi::causal_conv1d_update_f16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_f32_ptr,
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
                        state_f32_ptr,
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
                        state_f32_ptr,
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
            let state_f32_ptr = state_ptr as *mut f32;
            let slots_ptr = get_cuda_const_ptr_i64(slots)?;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let stream = *dev.cu_stream() as i64;

            unsafe {
                match x.dtype() {
                    DType::F16 => ffi::causal_conv1d_update_slots_f16(
                        x_ptr,
                        weight_ptr,
                        bias_ptr,
                        state_f32_ptr,
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
                        state_f32_ptr,
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
                        state_f32_ptr,
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
            let g = cuda_full_write_output(a.shape(), DType::F32, a.device())?;
            let beta = cuda_full_write_output(a.shape(), DType::F32, a.device())?;
            let a_log_f32 = ensure_f32_contiguous(a_log)?;
            let dt_f32 = ensure_f32_contiguous(dt_bias)?;

            let al_ptr = get_cuda_const_ptr(&a_log_f32)? as *const f32;
            let a_ptr = get_cuda_const_ptr(a)?;
            let b_ptr = get_cuda_const_ptr(b)?;
            let dt_ptr = get_cuda_const_ptr(&dt_f32)? as *const f32;
            let g_ptr = get_cuda_mut_ptr(&g)? as *mut f32;
            let beta_ptr = get_cuda_mut_ptr(&beta)? as *mut f32;
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
                        al_ptr,
                        a_ptr as *const f32,
                        b_ptr as *const f32,
                        dt_ptr,
                        g_ptr,
                        beta_ptr,
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
            let out = cuda_full_write_output((rows, value_dim), x.dtype(), x.device())?;

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
/// - CUDA `state`: `[bh, k_dim, v_dim]` (updated in place)
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
            let decay_f32 = exp_f32_contiguous(g)?;
            let beta_f32 = ensure_f32_contiguous(beta)?;

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

            let out = cuda_full_write_output((bh, seq_len, v_dim), DType::F32, q_c.device())?;

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
    state: &mut Tensor, // CUDA: [max_batch, heads, k_dim, v_dim]
    slots: &Tensor, // [batch] i64
) -> Result<Tensor> {
    match q.device() {
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let decay_c = exp_f32_contiguous(g)?;
            let beta_c = ensure_f32_contiguous(beta)?;

            let (bq, hq, kq) = q.dims3()?;
            let (bk, hk, kk) = k.dims3()?;
            let (bv, hv, v_dim) = v.dims3()?;
            let (bg, hg) = g.dims2()?;
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

            if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
                candle_core::bail!(
                    "gated_delta_rule_decode_slots dtype mismatch: q={:?} k={:?} v={:?}",
                    q.dtype(),
                    k.dtype(),
                    v.dtype()
                );
            }
            if g.dtype() != DType::F32 || beta.dtype() != DType::F32 {
                candle_core::bail!(
                    "gated_delta_rule_decode_slots expects F32 g/beta, got g={:?} beta={:?}",
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
                let g_ptr = get_cuda_const_ptr(&decay_c)? as *const f32;
                let beta_ptr = get_cuda_const_ptr(&beta_c)? as *const f32;
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
            let output = cuda_full_write_output(shape, input.dtype(), input.device())?;
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
/// When k_dim=128, v_dim=128, and max sequence length > 4096, automatically dispatches
/// to the WY32 chunk-based kernel for significantly better throughput.
///
/// Shapes:
/// - `q`, `k`: `[total_tokens, num_heads, k_dim]`
/// - `v`: `[total_tokens, num_heads, v_dim]`
/// - `g`, `beta`: `[total_tokens, num_heads]`
/// - CUDA `state`: `[max_batch, num_heads, k_dim, v_dim]` (FP32, updated in place)
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
    state_snapshots: Option<&Tensor>,
) -> Result<Tensor> {
    match q.device() {
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let decay_c = exp_f32_contiguous(g)?;
            let beta_c = ensure_f32_contiguous(beta)?;

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
            let g_ptr = get_cuda_const_ptr(&decay_c)? as *const f32;
            let beta_ptr = get_cuda_const_ptr(&beta_c)? as *const f32;
            let state_ptr = get_cuda_mut_ptr(state)? as *mut f32;
            let slots_ptr = get_cuda_const_ptr_i64(slots)?;
            let cu_ptr = get_cuda_const_ptr_u32(cu_seqlens)?;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let snapshots_ptr = if let Some(snapshots) = state_snapshots {
                if snapshots.dtype() != DType::F32 {
                    candle_core::bail!(
                        "gated_delta_rule_recurrence_varlen snapshot expects F32, got {:?}",
                        snapshots.dtype()
                    );
                }
                if snapshots.dims() != [total_tokens, num_heads, k_dim, v_dim] {
                    candle_core::bail!(
                        "gated_delta_rule_recurrence_varlen snapshot shape {:?} != expected [{}, {}, {}, {}]",
                        snapshots.shape(),
                        total_tokens,
                        num_heads,
                        k_dim,
                        v_dim
                    );
                }
                get_cuda_mut_ptr(snapshots)? as *mut f32
            } else {
                std::ptr::null_mut()
            };
            let stream = *dev.cu_stream() as i64;

            match q.dtype() {
                DType::F32 => unsafe {
                    ffi::gated_delta_rule_recurrence_varlen_f32(
                        q_ptr as *const f32,
                        k_ptr as *const f32,
                        v_ptr as *const f32,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr as *mut f32,
                        snapshots_ptr,
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
                        snapshots_ptr,
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
                        snapshots_ptr,
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

// ─── GCU implementations ────────────────────────────────────────────────────
// All GDN ops use ubridge host-kernel FFI for native GCU acceleration.

#[cfg(feature = "gcu")]
fn ensure_contiguous_gcu(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// Cached decode QSL tensor: [0, 1, 2, ..., max_batch] on device.
/// Created once per max_batch size and reused across all decode steps.
#[cfg(feature = "gcu")]
fn get_or_create_decode_qsl(max_batch: usize, device: &Device) -> Result<Tensor> {
    use std::sync::Mutex;
    static CACHE: std::sync::LazyLock<Mutex<Option<(usize, Tensor)>>> =
        std::sync::LazyLock::new(|| Mutex::new(None));

    let mut guard = CACHE.lock().unwrap();
    if let Some((cached_size, ref t)) = *guard {
        if cached_size >= max_batch {
            return Ok(t.clone());
        }
    }
    let qsl_vec: Vec<u32> = (0..=max_batch as u32).collect();
    let t = Tensor::from_vec(qsl_vec, (max_batch + 1,), device)?;
    *guard = Some((max_batch, t.clone()));
    Ok(t)
}

#[cfg(feature = "gcu")]
fn get_gcu_ptr<T: candle::gcu_backend::GcuDType>(t: &Tensor) -> Result<*mut T> {
    use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    let offset = layout.start_offset();
    match &*storage {
        Storage::Gcu(s) => {
            let slice = s.as_gcu_slice::<T>()?;
            let slice = slice.slice(offset..);
            Ok(slice.device_ptr() as *mut T)
        }
        _ => candle_core::bail!("Expected GCU tensor"),
    }
}

///
/// `slots`: [batch] i64 tensor of slot indices into the full state tensor.
#[cfg(feature = "gcu")]
pub fn causal_conv1d_fwd(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    slots: &Tensor,
    cu_seqlens: Option<&Tensor>,
    activation_silu: bool,
) -> Result<Tensor> {
    use candle::gcu_backend::ubridge::ffi;
    use candle::gcu_backend::WrapErr;

    let cu_seqlens = cu_seqlens.ok_or_else(|| {
        candle_core::Error::msg("GCU causal_conv1d_fwd requires cu_seqlens for prefill")
    })?;
    let x_c = ensure_contiguous_gcu(x)?;

    let (total_tokens, d_conv) = x_c.dims2()?;
    // Weight is pre-transposed [kernel_size, d_conv] — no per-call transpose needed
    let kernel_size = weight.dim(0)?;
    let batch = cu_seqlens.dim(0)? - 1;

    let weight_c = ensure_contiguous_gcu(weight)?;

    let dev = x_c.device().as_gcu_device()?;
    let out = Tensor::zeros((total_tokens, d_conv), x_c.dtype(), x_c.device())?;

    let stream = dev.stream_inner().expect("GCU stream") as *const c_void;

    let bias_c = bias.map(|b| ensure_contiguous_gcu(b)).transpose()?;
    let max_slots = conv_state.dim(0)?;

    match x_c.dtype() {
        DType::BF16 => unsafe {
            ffi::causal_conv1d_fwd_choreo_bf16(
                get_gcu_ptr::<bf16>(&x_c)?,
                get_gcu_ptr::<bf16>(&weight_c)?,
                if let Some(ref b) = bias_c {
                    get_gcu_ptr::<bf16>(b)?
                } else {
                    std::ptr::null_mut()
                },
                get_gcu_ptr::<bf16>(conv_state)?,
                get_gcu_ptr::<i64>(&slots)?,
                get_gcu_ptr::<u32>(&cu_seqlens)?,
                get_gcu_ptr::<bf16>(&out)?,
                total_tokens as i32,
                d_conv as i32,
                batch as i32,
                kernel_size as i32,
                if activation_silu { 1 } else { 0 },
                max_slots as i32,
                stream,
            );
        },
        dt => candle_core::bail!("GCU causal_conv1d_fwd unsupported dtype: {:?}", dt),
    }
    Ok(out)
}

#[cfg(feature = "gcu")]
pub fn causal_conv1d_update(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    let slots = Tensor::arange(0i64, x.dim(0)? as i64, x.device())?;
    causal_conv1d_update_slots(x, weight, bias, conv_state, &slots, activation_silu)
}

/// State is in kernel-native layout [max_slots, state_len, d_conv]
#[cfg(feature = "gcu")]
pub fn causal_conv1d_update_slots(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    conv_state: &mut Tensor,
    slots: &Tensor,
    activation_silu: bool,
) -> Result<Tensor> {
    use candle::gcu_backend::ubridge::ffi;
    use candle::gcu_backend::WrapErr;

    let x_c = ensure_contiguous_gcu(x)?;
    let (batch, d_conv) = x_c.dims2()?;
    // Weight is pre-transposed [kernel_size, d_conv] — no per-call transpose needed
    let kernel_size = weight.dim(0)?;
    let max_batch = conv_state.dim(0)?;

    let weight_c = ensure_contiguous_gcu(weight)?;

    let dev = x_c.device().as_gcu_device()?;
    let out = Tensor::zeros((batch, d_conv), x_c.dtype(), x_c.device())?;
    let stream = dev.stream_inner().expect("GCU stream") as *const c_void;

    let bias_c = bias.map(|b| ensure_contiguous_gcu(b)).transpose()?;

    // Decode QSL: [0, 1, ..., max_batch] pre-allocated at full capacity.
    // For actual batch <= max_batch, narrow to [0..=batch].
    let qsl = get_or_create_decode_qsl(max_batch, x_c.device())?.narrow(0, 0, batch + 1)?;

    match x_c.dtype() {
        DType::BF16 => unsafe {
            ffi::causal_conv1d_fwd_choreo_bf16(
                get_gcu_ptr::<bf16>(&x_c)?,
                get_gcu_ptr::<bf16>(&weight_c)?,
                if let Some(ref b) = bias_c {
                    get_gcu_ptr::<bf16>(b)?
                } else {
                    std::ptr::null_mut()
                },
                get_gcu_ptr::<bf16>(conv_state)?,
                get_gcu_ptr::<i64>(&slots)?,
                get_gcu_ptr::<u32>(&qsl)?,
                get_gcu_ptr::<bf16>(&out)?,
                batch as i32,
                d_conv as i32,
                batch as i32,
                kernel_size as i32,
                if activation_silu { 1 } else { 0 },
                max_batch as i32,
                stream,
            );
        },
        dt => candle_core::bail!("GCU causal_conv1d_update_slots unsupported dtype: {:?}", dt),
    }

    Ok(out)
}

/// GCU fused_gdn_gating: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)
///   a_log: [heads] F32, dt_bias: [heads] F32 (model weights)
///   a: [total, heads] BF16, b: [total, heads] BF16 (activations)
/// Returns g: [total, heads] (BF16), beta: [total, heads] (BF16).
#[cfg(feature = "gcu")]
pub fn fused_gdn_gating(
    a_log: &Tensor,
    a: &Tensor,
    b: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    use candle::gcu_backend::ubridge::ffi;
    use candle::gcu_backend::WrapErr;

    let a_c = ensure_contiguous_gcu(a)?;
    let b_c = ensure_contiguous_gcu(b)?;
    let a_log_c = ensure_contiguous_gcu(a_log)?;
    let dt_c = ensure_contiguous_gcu(dt_bias)?;

    let (total, num_heads) = a_c.dims2()?;
    let dev = a_c.device().as_gcu_device()?;
    let stream = dev.stream_inner().expect("GCU stream") as *const std::ffi::c_void;

    let g = Tensor::zeros((total, num_heads), DType::BF16, a_c.device())?;
    let beta = Tensor::zeros((total, num_heads), DType::BF16, a_c.device())?;

    unsafe {
        ffi::gdn_fused_gating_bf16(
            get_gcu_ptr::<f32>(&a_log_c)?,
            get_gcu_ptr::<bf16>(&a_c)?,
            get_gcu_ptr::<bf16>(&b_c)?,
            get_gcu_ptr::<f32>(&dt_c)?,
            get_gcu_ptr::<bf16>(&g)? as *mut bf16,
            get_gcu_ptr::<bf16>(&beta)? as *mut bf16,
            total as i32,
            num_heads as i32,
            stream,
        );
    }

    Ok((g, beta))
}

/// Input: any shape with last dim = normalization dim. Must be BF16.
#[cfg(feature = "gcu")]
pub fn l2_norm_last_dim(input: &Tensor, _eps: f64) -> Result<Tensor> {
    use candle::gcu_backend::ubridge::ffi;
    use candle::gcu_backend::WrapErr;

    let input_c = ensure_contiguous_gcu(input)?;
    let shape = input_c.shape().clone();
    let dim = *input_c.dims().last().unwrap();
    let rows = input_c.elem_count() / dim;
    let dev = input_c.device().as_gcu_device()?;
    let stream = dev.stream_inner().expect("GCU stream") as *const std::ffi::c_void;
    let output = Tensor::zeros(&shape, DType::BF16, input_c.device())?;

    unsafe {
        ffi::gdn_l2_norm_bf16(
            get_gcu_ptr::<bf16>(&input_c)?,
            get_gcu_ptr::<bf16>(&output)? as *mut bf16,
            rows as i32,
            dim as i32,
            stream,
        );
    }

    Ok(output)
}

/// GCU gated_rmsnorm_silu_mul: out = RMSNorm(x; gamma, eps) * silu(z)
///
/// The Choreo kernel fuses RMSNorm + silu(z) multiplication but does NOT
/// support a bias parameter.  When `norm_bias` is present the correct
/// formula is `out = (RMSNorm(x) * gamma + bias) * silu(z)`, which differs
/// from `RMSNorm(x) * gamma * silu(z) + bias` by a factor of `silu(z)` on
/// the bias term.  Since Qwen3.5/Qwen3Next linear-attn layers have no
/// norm.bias (verified via safetensors index), this path is never taken
/// in practice.  If a future model adds bias, the kernel must be updated
/// to fuse it correctly.
#[cfg(feature = "gcu")]
pub fn gated_rmsnorm_silu_mul(
    x: &Tensor,
    z: &Tensor,
    norm_weight: &Tensor,
    _norm_bias: Option<&Tensor>,
    _eps: f64,
    group_size: usize,
) -> Result<Tensor> {
    use candle::gcu_backend::ubridge::ffi;
    use candle::gcu_backend::WrapErr;

    let x_c = ensure_contiguous_gcu(x)?;
    let z_c = ensure_contiguous_gcu(z)?;
    let w_c = ensure_contiguous_gcu(norm_weight)?;

    let (rows, value_dim) = x_c.dims2()?;
    let num_groups = value_dim / group_size;
    let dev = x_c.device().as_gcu_device()?;
    let stream = dev.stream_inner().expect("GCU stream") as *const std::ffi::c_void;

    let total_rows = rows * num_groups;
    let output = Tensor::zeros((rows, value_dim), DType::BF16, x_c.device())?;

    unsafe {
        ffi::gdn_gated_rmsnorm_bf16(
            get_gcu_ptr::<bf16>(&x_c)?,
            get_gcu_ptr::<bf16>(&z_c)?,
            get_gcu_ptr::<f32>(&w_c)?,
            get_gcu_ptr::<bf16>(&output)? as *mut bf16,
            total_rows as i32,
            group_size as i32,
            stream,
        );
    }

    Ok(output)
}

/// GCU gated_delta_rule_recurrence
#[cfg(feature = "gcu")]
pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    use candle::gcu_backend::ubridge::ffi;
    use candle::gcu_backend::WrapErr;

    let (bh, seq_len, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;

    let q_c = ensure_contiguous_gcu(q)?;
    let k_c = ensure_contiguous_gcu(k)?;
    let v_c = ensure_contiguous_gcu(v)?;
    let g_f32 = g.exp()?.to_dtype(DType::F32)?.contiguous()?;
    let beta_f32 = beta.to_dtype(DType::F32)?.contiguous()?;
    let state_c = ensure_contiguous_gcu(state)?;

    let dev = q_c.device().as_gcu_device()?;
    let out = Tensor::zeros((bh, seq_len, v_dim), DType::F32, q_c.device())?;
    let stream = dev.stream_inner().expect("GCU stream") as *const c_void;

    match q_c.dtype() {
        DType::F16 => unsafe {
            ffi::gdn_recurrence_f16(
                get_gcu_ptr::<f16>(&q_c)?,
                get_gcu_ptr::<f16>(&k_c)?,
                get_gcu_ptr::<f16>(&v_c)?,
                get_gcu_ptr::<f32>(&g_f32)?,
                get_gcu_ptr::<f32>(&beta_f32)?,
                get_gcu_ptr::<f32>(&state_c)?,
                get_gcu_ptr::<f32>(&out)?,
                bh as i32,
                seq_len as i32,
                k_dim as i32,
                v_dim as i32,
                2,
                12,
                stream,
            );
        },
        DType::BF16 => unsafe {
            ffi::gdn_recurrence_bf16(
                get_gcu_ptr::<bf16>(&q_c)?,
                get_gcu_ptr::<bf16>(&k_c)?,
                get_gcu_ptr::<bf16>(&v_c)?,
                get_gcu_ptr::<f32>(&g_f32)?,
                get_gcu_ptr::<f32>(&beta_f32)?,
                get_gcu_ptr::<f32>(&state_c)?,
                get_gcu_ptr::<f32>(&out)?,
                bh as i32,
                seq_len as i32,
                k_dim as i32,
                v_dim as i32,
                2,
                12,
                stream,
            );
        },
        DType::F32 => unsafe {
            ffi::gdn_recurrence_f32(
                get_gcu_ptr::<f32>(&q_c)?,
                get_gcu_ptr::<f32>(&k_c)?,
                get_gcu_ptr::<f32>(&v_c)?,
                get_gcu_ptr::<f32>(&g_f32)?,
                get_gcu_ptr::<f32>(&beta_f32)?,
                get_gcu_ptr::<f32>(&state_c)?,
                get_gcu_ptr::<f32>(&out)?,
                bh as i32,
                seq_len as i32,
                k_dim as i32,
                v_dim as i32,
                2,
                12,
                stream,
            );
        },
        dt => candle_core::bail!("GCU gdn_recurrence unsupported dtype: {:?}", dt),
    }

    state.copy_(&state_c, 0)?;
    out.to_dtype(q.dtype())
}

/// GCU gated_delta_rule_decode_slots
#[cfg(feature = "gcu")]
pub fn gated_delta_rule_decode_slots(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
) -> Result<Tensor> {
    use candle::gcu_backend::ubridge::ffi;
    use candle::gcu_backend::WrapErr;

    let (batch, heads, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;
    let max_slots = state.dim(0)? as i32;

    let q_c = ensure_contiguous_gcu(q)?;
    let k_c = ensure_contiguous_gcu(k)?;
    let v_c = ensure_contiguous_gcu(v)?;
    let g_f32 = g.to_dtype(DType::F32)?.exp()?.contiguous()?;
    let beta_f32 = beta.to_dtype(DType::F32)?.contiguous()?;
    let slots_c = ensure_contiguous_gcu(slots)?;

    let dev = q_c.device().as_gcu_device()?;
    let stream = dev.stream_inner().expect("GCU stream") as *const c_void;

    let out = match q_c.dtype() {
        DType::F16 => {
            let out = Tensor::zeros((batch, heads, v_dim), q_c.dtype(), q_c.device())?;
            unsafe {
                ffi::gdn_decode_slots_f16(
                    get_gcu_ptr::<f16>(&q_c)?,
                    get_gcu_ptr::<f16>(&k_c)?,
                    get_gcu_ptr::<f16>(&v_c)?,
                    get_gcu_ptr::<f16>(&g.contiguous()?)?,
                    get_gcu_ptr::<f16>(&beta.contiguous()?)?,
                    get_gcu_ptr::<f32>(state)?,
                    get_gcu_ptr::<i64>(&slots_c)? as *const i64,
                    get_gcu_ptr::<f16>(&out)?,
                    batch as i32,
                    heads as i32,
                    k_dim as i32,
                    v_dim as i32,
                    max_slots,
                    2,
                    12,
                    stream,
                );
            }
            out
        }
        DType::BF16 => {
            let out_f32 = Tensor::zeros((batch, heads, v_dim), DType::F32, q_c.device())?;
            unsafe {
                ffi::gdn_decode_slots_bf16(
                    get_gcu_ptr::<bf16>(&q_c)?,
                    get_gcu_ptr::<bf16>(&k_c)?,
                    get_gcu_ptr::<bf16>(&v_c)?,
                    get_gcu_ptr::<f32>(&g_f32)?,
                    get_gcu_ptr::<f32>(&beta_f32)?,
                    get_gcu_ptr::<f32>(state)?,
                    get_gcu_ptr::<i64>(&slots_c)? as *const i64,
                    get_gcu_ptr::<f32>(&out_f32)?,
                    batch as i32,
                    heads as i32,
                    k_dim as i32,
                    v_dim as i32,
                    max_slots,
                    2,
                    12,
                    stream,
                );
            }
            out_f32.to_dtype(DType::BF16)?
        }
        DType::F32 => {
            let out = Tensor::zeros((batch, heads, v_dim), DType::F32, q_c.device())?;
            unsafe {
                ffi::gdn_decode_slots_f32(
                    get_gcu_ptr::<f32>(&q_c)?,
                    get_gcu_ptr::<f32>(&k_c)?,
                    get_gcu_ptr::<f32>(&v_c)?,
                    get_gcu_ptr::<f32>(&g_f32)?,
                    get_gcu_ptr::<f32>(&beta_f32)?,
                    get_gcu_ptr::<f32>(state)?,
                    get_gcu_ptr::<i64>(&slots_c)? as *const i64,
                    get_gcu_ptr::<f32>(&out)?,
                    batch as i32,
                    heads as i32,
                    k_dim as i32,
                    v_dim as i32,
                    max_slots,
                    2,
                    12,
                    stream,
                );
            }
            out
        }
        dt => candle_core::bail!("GCU gdn_decode_slots unsupported dtype: {:?}", dt),
    };

    Ok(out)
}

/// GCU GQA decode: q/k have num_k_heads, v/g/beta/state/out have num_v_heads.
/// Fuses GQA repeat + q_scale + exp(g) into a single kernel launch.
///
/// g is expected in log-space (as returned by fused_gdn_gating).
/// The kernel applies exp(g) internally.
///
/// Uses the dedicated Choreo kernel gdn_decode_slots_gqa_bf16.co which
/// performs GQA index-based repeat (no copy), q_scale fusion, and exp(g)
/// all inside a single kernel launch, eliminating ~60-70µs of overhead.
#[cfg(feature = "gcu")]
pub fn gated_delta_rule_decode_slots_gqa(
    q: &Tensor,         // [batch, num_k_heads, k_dim]
    k: &Tensor,         // [batch, num_k_heads, k_dim]
    v: &Tensor,         // [batch, num_v_heads, v_dim]
    g: &Tensor,         // [batch, num_v_heads] (log-space from fused_gdn_gating)
    beta: &Tensor,      // [batch, num_v_heads]
    state: &mut Tensor, // [max_slots, num_v_heads, k_dim, v_dim] F32
    slots: &Tensor,     // [batch] i64
    q_scale: f32,
) -> Result<Tensor> {
    use candle::gcu_backend::ubridge::ffi;
    use candle::gcu_backend::WrapErr;

    let (batch, num_k_heads, k_dim) = q.dims3()?;
    let num_v_heads = v.dim(1)?;
    let v_dim = v.dim(2)?;
    let max_slots = state.dim(0)? as i32;

    if num_v_heads % num_k_heads != 0 {
        candle_core::bail!(
            "gated_delta_rule_decode_slots_gqa: num_v_heads {} not divisible by num_k_heads {}",
            num_v_heads,
            num_k_heads
        );
    }
    if slots.dtype() != DType::I64 || slots.dim(0)? != batch {
        candle_core::bail!(
            "gated_delta_rule_decode_slots_gqa expects [batch] I64 slots, got {:?} {:?}",
            slots.shape(),
            slots.dtype()
        );
    }

    let q_c = ensure_contiguous_gcu(q)?;
    let k_c = ensure_contiguous_gcu(k)?;
    let v_c = ensure_contiguous_gcu(v)?;
    // g is log-space — kernel applies exp(g) internally
    let g_f32 = g.to_dtype(DType::F32)?.contiguous()?;
    let beta_f32 = beta.to_dtype(DType::F32)?.contiguous()?;
    let slots_c = ensure_contiguous_gcu(slots)?;

    let dev = q_c.device().as_gcu_device()?;
    let stream = dev.stream_inner().expect("GCU stream") as *const c_void;

    match q_c.dtype() {
        DType::BF16 => {
            let out_f32 = Tensor::zeros((batch, num_v_heads, v_dim), DType::F32, q_c.device())?;
            let qscale_dev = Tensor::new(&[q_scale], q_c.device())?.contiguous()?;
            unsafe {
                ffi::gdn_decode_slots_gqa_bf16(
                    get_gcu_ptr::<bf16>(&q_c)?,
                    get_gcu_ptr::<bf16>(&k_c)?,
                    get_gcu_ptr::<bf16>(&v_c)?,
                    get_gcu_ptr::<f32>(&g_f32)?,
                    get_gcu_ptr::<f32>(&beta_f32)?,
                    get_gcu_ptr::<f32>(state)?,
                    get_gcu_ptr::<i64>(&slots_c)? as *const i64,
                    get_gcu_ptr::<f32>(&qscale_dev)?,
                    get_gcu_ptr::<f32>(&out_f32)?,
                    batch as i32,
                    num_v_heads as i32,
                    num_k_heads as i32,
                    k_dim as i32,
                    v_dim as i32,
                    max_slots,
                    2,
                    12,
                    stream,
                );
            }
            out_f32.to_dtype(DType::BF16)
        }
        dt => candle_core::bail!(
            "gated_delta_rule_decode_slots_gqa: unsupported dtype {:?} on GCU (only BF16)",
            dt
        ),
    }
}

/// GCU Fused GDN Decode L2Norm + Recurrence + Post kernel.
///
/// Fuses the following into a single Choreo kernel launch:
///   1. L2Norm on Q and K (in F32, eps=1e-6)
///   2. GDN gating: g = -exp(a_log) * softplus(a + dt_bias), beta = sigmoid(b)
///   3. GQA recurrence with state update and q_scale
///   4. RMSNorm on per-head output
///   5. SiLU(z) gating multiply
///
/// Q/K have num_k_heads (GQA-indexed, not expanded).  Q/K are RAW from
/// causal_conv1d — the kernel performs L2Norm internally, eliminating
/// the separate l2_norm_last_dim kernel launches.
/// V/A/B/Z have num_v_heads dimension.
/// Expected to replace: l2_norm_last_dim(q) + l2_norm_last_dim(k)
///                      + gated_delta_rule_decode_recurrence_post_fused
#[cfg(feature = "gcu")]
pub fn gated_delta_rule_decode_recurrence_fused(
    q: &Tensor,         // [batch, num_k_heads, k_dim] raw from conv (NOT L2-normalized)
    k: &Tensor,         // [batch, num_k_heads, k_dim] raw from conv (NOT L2-normalized)
    v: &Tensor,         // [batch, num_v_heads, v_dim] raw from conv
    a: &Tensor,         // [batch, num_v_heads] activation
    b: &Tensor,         // [batch, num_v_heads] activation
    a_log: &Tensor,     // [num_v_heads] F32 model weight
    dt_bias: &Tensor,   // [num_v_heads] F32 model weight
    z: &Tensor,         // [batch, value_dim] BF16 gate
    state: &mut Tensor, // [max_slots, num_v_heads, k_dim, v_dim] F32
    slots: &Tensor,     // [batch] I64
    norm_weight: &Tensor, // [v_dim=128] F32 RMSNorm weight
    q_scale: f32,
) -> Result<Tensor> {
    use candle::gcu_backend::ubridge::ffi;
    use candle::gcu_backend::WrapErr;

    let (batch, num_k_heads, k_dim) = q.dims3()?;
    let num_v_heads = v.dim(1)?;
    let v_dim = v.dim(2)?;
    let value_dim = z.dim(1)?;
    let max_slots = state.dim(0)? as i32;

    if num_v_heads % num_k_heads != 0 {
        candle_core::bail!(
            "gated_delta_rule_decode_l2norm_recurrence_post_fused: num_v_heads {} not divisible by num_k_heads {}",
            num_v_heads,
            num_k_heads
        );
    }
    if value_dim != num_v_heads * v_dim {
        candle_core::bail!(
            "gated_delta_rule_decode_l2norm_recurrence_post_fused: value_dim {} != num_v_heads {} * v_dim {}",
            value_dim, num_v_heads, v_dim
        );
    }

    let q_c = ensure_contiguous_gcu(q)?;
    let k_c = ensure_contiguous_gcu(k)?;
    let v_c = ensure_contiguous_gcu(v)?;
    let a_c = ensure_contiguous_gcu(a)?;
    let b_c = ensure_contiguous_gcu(b)?;
    let a_log_c = ensure_contiguous_gcu(a_log)?;
    let dt_c = ensure_contiguous_gcu(dt_bias)?;
    let z_c = ensure_contiguous_gcu(z)?;
    let slots_c = ensure_contiguous_gcu(slots)?;
    let nw_c = ensure_contiguous_gcu(norm_weight)?;

    let dev = q_c.device().as_gcu_device()?;
    let stream = dev.stream_inner().expect("GCU stream") as *const c_void;

    match q_c.dtype() {
        DType::BF16 => {
            let out = Tensor::zeros((batch, value_dim), DType::BF16, q_c.device())?;
            unsafe {
                ffi::gdn_decode_recurrence_fused_bf16(
                    get_gcu_ptr::<bf16>(&q_c)?,
                    get_gcu_ptr::<bf16>(&k_c)?,
                    get_gcu_ptr::<bf16>(&v_c)?,
                    get_gcu_ptr::<bf16>(&a_c)?,
                    get_gcu_ptr::<bf16>(&b_c)?,
                    get_gcu_ptr::<f32>(&a_log_c)?,
                    get_gcu_ptr::<f32>(&dt_c)?,
                    get_gcu_ptr::<bf16>(&z_c)?,
                    get_gcu_ptr::<f32>(state)?,
                    get_gcu_ptr::<i64>(&slots_c)? as *const i64,
                    get_gcu_ptr::<f32>(&nw_c)?,
                    q_scale,
                    get_gcu_ptr::<bf16>(&out)? as *mut bf16,
                    batch as i32,
                    num_k_heads as i32,
                    num_v_heads as i32,
                    k_dim as i32,
                    v_dim as i32,
                    max_slots,
                    2,
                    12,
                    stream,
                );
            }
            Ok(out)
        }
        dt => candle_core::bail!(
            "gated_delta_rule_decode_recurrence_fused: unsupported dtype {:?} on GCU (only BF16)",
            dt
        ),
    }
}

/// GCU gated_delta_rule_recurrence_varlen
#[cfg(feature = "gcu")]
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
    use candle::gcu_backend::ubridge::ffi;
    use candle::gcu_backend::WrapErr;

    let (total_tokens, num_heads, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;
    let batch = slots.dim(0)?;
    let max_slots = state.dim(0)? as i32;

    let q_c = ensure_contiguous_gcu(q)?;
    let k_c = ensure_contiguous_gcu(k)?;
    let v_c = ensure_contiguous_gcu(v)?;
    let g_c = g
        .to_dtype(DType::F32)?
        .exp()?
        .to_dtype(q.dtype())?
        .contiguous()?;
    let beta_c = ensure_contiguous_gcu(beta)?;
    let slots_c = ensure_contiguous_gcu(slots)?;

    let dev = q_c.device().as_gcu_device()?;
    let out = Tensor::zeros((total_tokens, num_heads, v_dim), q_c.dtype(), q_c.device())?;
    let stream = dev.stream_inner().expect("GCU stream") as *const c_void;

    match q_c.dtype() {
        DType::F16 => unsafe {
            ffi::gdn_recurrence_varlen_f16(
                get_gcu_ptr::<f16>(&q_c)?,
                get_gcu_ptr::<f16>(&k_c)?,
                get_gcu_ptr::<f16>(&v_c)?,
                get_gcu_ptr::<f16>(&g_c)?,
                get_gcu_ptr::<f16>(&beta_c)?,
                get_gcu_ptr::<f32>(state)?,
                get_gcu_ptr::<i64>(&slots_c)? as *const i64,
                get_gcu_ptr::<f16>(&out)?,
                get_gcu_ptr::<u32>(&cu_seqlens)? as *const u32,
                total_tokens as i32,
                batch as i32,
                num_heads as i32,
                k_dim as i32,
                v_dim as i32,
                max_slots,
                2,
                12,
                stream,
            );
        },
        DType::BF16 => unsafe {
            ffi::gdn_recurrence_varlen_bf16(
                get_gcu_ptr::<bf16>(&q_c)?,
                get_gcu_ptr::<bf16>(&k_c)?,
                get_gcu_ptr::<bf16>(&v_c)?,
                get_gcu_ptr::<bf16>(&g_c)?,
                get_gcu_ptr::<bf16>(&beta_c)?,
                get_gcu_ptr::<f32>(state)?,
                get_gcu_ptr::<i64>(&slots_c)? as *const i64,
                get_gcu_ptr::<bf16>(&out)?,
                get_gcu_ptr::<u32>(&cu_seqlens)? as *const u32,
                total_tokens as i32,
                batch as i32,
                num_heads as i32,
                k_dim as i32,
                v_dim as i32,
                max_slots,
                2,
                12,
                stream,
            );
        },
        DType::F32 => unsafe {
            ffi::gdn_recurrence_varlen_f32(
                get_gcu_ptr::<f32>(&q_c)?,
                get_gcu_ptr::<f32>(&k_c)?,
                get_gcu_ptr::<f32>(&v_c)?,
                get_gcu_ptr::<f32>(&g_c)?,
                get_gcu_ptr::<f32>(&beta_c)?,
                get_gcu_ptr::<f32>(state)?,
                get_gcu_ptr::<i64>(&slots_c)? as *const i64,
                get_gcu_ptr::<f32>(&out)?,
                get_gcu_ptr::<u32>(&cu_seqlens)? as *const u32,
                total_tokens as i32,
                batch as i32,
                num_heads as i32,
                k_dim as i32,
                v_dim as i32,
                max_slots,
                2,
                12,
                stream,
            );
        },
        dt => candle_core::bail!("GCU gdn_recurrence_varlen unsupported dtype: {:?}", dt),
    }
    Ok(out)
}

/// GQA variant: q/k have num_k_heads, v/g/beta/state/out have num_v_heads.
/// Fuses q_scale multiplication into the kernel to avoid separate allocation.
#[cfg(feature = "cuda")]
pub fn gated_delta_rule_recurrence_varlen_gqa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
    cu_seqlens: &Tensor,
    q_scale: f32,
    state_snapshots: Option<&Tensor>,
) -> Result<Tensor> {
    match q.device() {
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let g_c = ensure_f32_contiguous(g)?;
            let beta_c = ensure_f32_contiguous(beta)?;

            let (total_tokens, num_k_heads, k_dim) = q_c.dims3()?;
            let num_v_heads = v_c.dim(1)?;
            let v_dim = v_c.dim(2)?;
            let batch = slots.dim(0)?;

            if num_v_heads % num_k_heads != 0 {
                candle_core::bail!(
                    "gated_delta_rule_recurrence_varlen_gqa: num_v_heads {} not divisible by num_k_heads {}",
                    num_v_heads,
                    num_k_heads
                );
            }

            if state.dtype() != DType::F32 {
                candle_core::bail!(
                    "gated_delta_rule_recurrence_varlen_gqa expects FP32 state, got {:?}",
                    state.dtype()
                );
            }

            let out = Tensor::zeros((total_tokens, num_v_heads, v_dim), q.dtype(), q.device())?;

            let q_ptr = get_cuda_const_ptr(&q_c)?;
            let k_ptr = get_cuda_const_ptr(&k_c)?;
            let v_ptr = get_cuda_const_ptr(&v_c)?;
            let g_ptr = get_cuda_const_ptr(&g_c)? as *const f32;
            let beta_ptr = get_cuda_const_ptr(&beta_c)? as *const f32;
            let state_ptr = get_cuda_mut_ptr(state)? as *mut f32;
            let slots_ptr = get_cuda_const_ptr_i64(slots)?;
            let cu_ptr = get_cuda_const_ptr_u32(cu_seqlens)?;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let snapshots_ptr = if let Some(snapshots) = state_snapshots {
                if snapshots.dtype() != DType::F32 {
                    candle_core::bail!(
                        "gated_delta_rule_recurrence_varlen_gqa snapshot expects F32, got {:?}",
                        snapshots.dtype()
                    );
                }
                if snapshots.dims() != [total_tokens, num_v_heads, k_dim, v_dim] {
                    candle_core::bail!(
                        "gated_delta_rule_recurrence_varlen_gqa snapshot shape {:?} != expected [{}, {}, {}, {}]",
                        snapshots.shape(),
                        total_tokens,
                        num_v_heads,
                        k_dim,
                        v_dim
                    );
                }
                get_cuda_mut_ptr(snapshots)? as *mut f32
            } else {
                std::ptr::null_mut()
            };
            let stream = *dev.cu_stream() as i64;

            match q.dtype() {
                DType::BF16 => unsafe {
                    ffi::gated_delta_rule_recurrence_varlen_gqa_bf16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr,
                        snapshots_ptr,
                        cu_ptr,
                        batch as c_int,
                        num_v_heads as c_int,
                        num_k_heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        q_scale,
                        stream,
                    )
                },
                DType::F16 => unsafe {
                    ffi::gated_delta_rule_recurrence_varlen_gqa_f16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr,
                        snapshots_ptr,
                        cu_ptr,
                        batch as c_int,
                        num_v_heads as c_int,
                        num_k_heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        q_scale,
                        stream,
                    )
                },
                DType::F32 => unsafe {
                    ffi::gated_delta_rule_recurrence_varlen_gqa_f32(
                        q_ptr as *const f32,
                        k_ptr as *const f32,
                        v_ptr as *const f32,
                        g_ptr as *const f32,
                        beta_ptr as *const f32,
                        state_ptr,
                        slots_ptr,
                        out_ptr as *mut f32,
                        snapshots_ptr,
                        cu_ptr,
                        batch as c_int,
                        num_v_heads as c_int,
                        num_k_heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        q_scale,
                        stream,
                    )
                },
                dt => candle_core::bail!(
                    "gated_delta_rule_recurrence_varlen_gqa: unsupported dtype {:?}",
                    dt
                ),
            }
            Ok(out)
        }
        _ => {
            candle_core::bail!("Invalid device for gated_delta_rule_recurrence_varlen_gqa");
        }
    }
}

/// FlashInfer SM90 persistent GDN prefill for GQA models.
/// Falls back to the regular recurrence kernel when FlashInfer is unavailable,
/// the GPU is not SM90, or parameters are unsupported (k_dim != v_dim, etc.).
///
/// Returns Ok(Some(output)) on success, Ok(None) when the kernel returned a
/// non-zero status (caller should fall back).
#[cfg(feature = "cuda")]
pub fn gated_delta_rule_prefill_flashinfer_gqa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    slots: &Tensor,
    cu_seqlens: &Tensor,
    q_scale: f32,
) -> Result<Option<Tensor>> {
    match q.device() {
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let g_c = ensure_f32_contiguous(g)?;
            let beta_c = ensure_f32_contiguous(beta)?;

            let (total_tokens, num_k_heads, k_dim) = q_c.dims3()?;
            let num_v_heads = v_c.dim(1)?;
            let v_dim = v_c.dim(2)?;
            let batch = slots.dim(0)?;

            if num_v_heads % num_k_heads != 0 {
                candle_core::bail!(
                    "gated_delta_rule_prefill_flashinfer_gqa: num_v_heads {} not divisible by num_k_heads {}",
                    num_v_heads,
                    num_k_heads
                );
            }

            let out = Tensor::zeros((total_tokens, num_v_heads, v_dim), q.dtype(), q.device())?;

            let q_ptr = get_cuda_const_ptr(&q_c)?;
            let k_ptr = get_cuda_const_ptr(&k_c)?;
            let v_ptr = get_cuda_const_ptr(&v_c)?;
            let g_ptr = get_cuda_const_ptr(&g_c)? as *const f32;
            let beta_ptr = get_cuda_const_ptr(&beta_c)? as *const f32;
            let state_ptr = get_cuda_mut_ptr(state)? as *mut f32;
            let slots_ptr = get_cuda_const_ptr_i64(slots)?;
            let cu_ptr = get_cuda_const_ptr_u32(cu_seqlens)?;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let stream = *dev.cu_stream() as i64;

            let status = match q.dtype() {
                DType::BF16 => unsafe {
                    ffi::gated_delta_rule_prefill_persistent_varlen_gqa_bf16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr,
                        cu_ptr,
                        total_tokens as c_int,
                        batch as c_int,
                        num_v_heads as c_int,
                        num_k_heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        q_scale,
                        stream,
                    )
                },
                DType::F16 => unsafe {
                    ffi::gated_delta_rule_prefill_persistent_varlen_gqa_f16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr,
                        cu_ptr,
                        total_tokens as c_int,
                        batch as c_int,
                        num_v_heads as c_int,
                        num_k_heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        q_scale,
                        stream,
                    )
                },
                dt => candle_core::bail!(
                    "gated_delta_rule_prefill_flashinfer_gqa: unsupported dtype {:?}",
                    dt
                ),
            };

            if status == 0 {
                Ok(Some(out))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// GQA decode: q/k have num_k_heads, v/g/beta/state/out have num_v_heads.
/// Fuses q_scale multiplication and exp(g) into the kernel.
/// State is FP32.
#[cfg(feature = "cuda")]
pub fn gated_delta_rule_decode_slots_gqa(
    q: &Tensor,         // [batch, num_k_heads, k_dim]
    k: &Tensor,         // [batch, num_k_heads, k_dim]
    v: &Tensor,         // [batch, num_v_heads, v_dim]
    g: &Tensor,         // [batch, num_v_heads] (log-space, NOT exp'd)
    beta: &Tensor,      // [batch, num_v_heads]
    state: &mut Tensor, // [max_batch, num_v_heads, k_dim, v_dim] FP32
    slots: &Tensor,     // [batch] i64
    q_scale: f32,
) -> Result<Tensor> {
    match q.device() {
        Device::Cuda(dev) => {
            let q_c = ensure_contiguous(q)?;
            let k_c = ensure_contiguous(k)?;
            let v_c = ensure_contiguous(v)?;
            let g_c = ensure_f32_contiguous(g)?;
            let beta_c = ensure_f32_contiguous(beta)?;

            let (batch, num_k_heads, k_dim) = q_c.dims3()?;
            let num_v_heads = v_c.dim(1)?;
            let v_dim = v_c.dim(2)?;

            if num_v_heads % num_k_heads != 0 {
                candle_core::bail!(
                    "gated_delta_rule_decode_slots_gqa: num_v_heads {} not divisible by num_k_heads {}",
                    num_v_heads,
                    num_k_heads
                );
            }
            if slots.dtype() != DType::I64 || slots.dim(0)? != batch {
                candle_core::bail!(
                    "gated_delta_rule_decode_slots_gqa expects slots [batch] I64, got {:?} {:?}",
                    slots.shape(),
                    slots.dtype()
                );
            }
            if state.dtype() != DType::F32 {
                candle_core::bail!(
                    "gated_delta_rule_decode_slots_gqa expects FP32 state, got {:?}",
                    state.dtype()
                );
            }
            if !state.is_contiguous() {
                candle_core::bail!("gated_delta_rule_decode_slots_gqa expects contiguous state");
            }

            let out = Tensor::zeros((batch, num_v_heads, v_dim), q.dtype(), q.device())?;

            let q_ptr = get_cuda_const_ptr(&q_c)?;
            let k_ptr = get_cuda_const_ptr(&k_c)?;
            let v_ptr = get_cuda_const_ptr(&v_c)?;
            let g_ptr = get_cuda_const_ptr(&g_c)? as *const f32;
            let beta_ptr = get_cuda_const_ptr(&beta_c)? as *const f32;
            let state_ptr = get_cuda_mut_ptr(state)? as *mut f32;
            let slots_ptr = get_cuda_const_ptr_i64(slots)?;
            let out_ptr = get_cuda_mut_ptr(&out)?;
            let stream = *dev.cu_stream() as i64;

            match q.dtype() {
                DType::BF16 => unsafe {
                    ffi::gated_delta_rule_decode_slots_gqa_bf16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr,
                        batch as c_int,
                        num_v_heads as c_int,
                        num_k_heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        q_scale,
                        stream,
                    )
                },
                DType::F16 => unsafe {
                    ffi::gated_delta_rule_decode_slots_gqa_f16(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        state_ptr,
                        slots_ptr,
                        out_ptr,
                        batch as c_int,
                        num_v_heads as c_int,
                        num_k_heads as c_int,
                        k_dim as c_int,
                        v_dim as c_int,
                        q_scale,
                        stream,
                    )
                },
                dt => candle_core::bail!(
                    "gated_delta_rule_decode_slots_gqa: unsupported dtype {:?}",
                    dt
                ),
            }
            Ok(out)
        }
        _ => {
            candle_core::bail!("Invalid device for gated_delta_rule_decode_slots_gqa");
        }
    }
}
