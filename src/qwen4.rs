//! Qwen4-Exp kernels: Gated Residual (Hyper-Connection) and QSA indexer.

#[cfg(feature = "cuda")]
use candle_core::{DType, Device};
use candle_core::{Result, Tensor};

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::Storage;

#[cfg(feature = "cuda")]
fn get_cuda_ptr(t: &Tensor) -> Result<*const std::ffi::c_void> {
    let (s, l) = t.storage_and_layout();
    match (&*s, t.dtype()) {
        (Storage::Cuda(c), DType::BF16) => Ok(*c
            .as_cuda_slice::<half::bf16>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const std::ffi::c_void),
        (Storage::Cuda(c), DType::F16) => Ok(*c
            .as_cuda_slice::<half::f16>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const std::ffi::c_void),
        (Storage::Cuda(c), DType::F32) => Ok(*c
            .as_cuda_slice::<f32>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const std::ffi::c_void),
        _ => candle_core::bail!(
            "qwen4 get_cuda_ptr: unsupported dtype {:?} on {:?}",
            t.dtype(),
            t.device()
        ),
    }
}

/// Kernel dtype selector: 0 = BF16, 1 = F16, 2 = F32 (matches the extern "C" ABI).
#[cfg(feature = "cuda")]
fn dtype_code(dtype: DType) -> Result<i32> {
    match dtype {
        DType::BF16 => Ok(0),
        DType::F16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!("qwen4 kernels: unsupported dtype {other:?}"),
    }
}

/// All kernel data operands must share one dtype (weights + activations).
#[cfg(feature = "cuda")]
fn check_same_dtype(op: &str, dtype: DType, tensors: &[&Tensor]) -> Result<()> {
    for t in tensors {
        if t.dtype() != dtype {
            candle_core::bail!(
                "qwen4 {op}: dtype mismatch, expected {dtype:?}, got {:?}",
                t.dtype()
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn get_cuda_ptr_mut(t: &Tensor) -> Result<*mut std::ffi::c_void> {
    Ok(get_cuda_ptr(t)? as *mut std::ffi::c_void)
}

#[cfg(feature = "cuda")]
fn get_cuda_stream(device: &Device) -> Result<i64> {
    match device {
        Device::Cuda(dev) => Ok(*dev.cu_stream() as i64),
        _ => candle_core::bail!("qwen4 kernels require CUDA device"),
    }
}

/// Hyper-Connection read (Gated Residual pre-block).
#[cfg(feature = "cuda")]
pub fn hc_read(
    hyper_input: &Tensor,
    hc_norm_weight: &Tensor,
    mix_down_weight: &Tensor,
    mix_up_weight: &Tensor,
    inject_weight: Option<&Tensor>,
    hc: usize,
    hidden: usize,
    lowrank: usize,
    eps: f32,
) -> Result<(Tensor, Option<Tensor>, Tensor)> {
    let (seq_len, hc_hidden) = hyper_input.dims2()?;
    if hc_hidden != hc * hidden {
        candle_core::bail!(
            "hc_read: expected last dim {}, got {}",
            hc * hidden,
            hc_hidden
        );
    }
    let device = hyper_input.device();
    let dtype = hyper_input.dtype();
    check_same_dtype(
        "hc_read",
        dtype,
        &[hc_norm_weight, mix_down_weight, mix_up_weight],
    )?;
    if let Some(iw) = inject_weight {
        check_same_dtype("hc_read", dtype, &[iw])?;
    }
    let dcode = dtype_code(dtype)?;
    let mixed = Tensor::zeros((seq_len, hidden), dtype, device)?;
    let normed_scratch = Tensor::zeros((seq_len, hc_hidden), dtype, device)?;
    let inject = if inject_weight.is_some() {
        Some(Tensor::zeros((seq_len, hc), dtype, device)?)
    } else {
        None
    };
    let stream = get_cuda_stream(device)?;
    let ret = unsafe {
        crate::kernels::ffi::qwen4_hc_read(
            get_cuda_ptr(hyper_input)?,
            get_cuda_ptr(hc_norm_weight)?,
            get_cuda_ptr(mix_down_weight)?,
            get_cuda_ptr(mix_up_weight)?,
            inject_weight
                .map(get_cuda_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null()),
            get_cuda_ptr_mut(&mixed)?,
            inject
                .as_ref()
                .map(get_cuda_ptr_mut)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
            get_cuda_ptr_mut(&normed_scratch)?,
            seq_len as i32,
            hc as i32,
            hidden as i32,
            lowrank as i32,
            eps,
            dcode,
            stream,
        )
    };
    if ret != 0 {
        candle_core::bail!("qwen4_hc_read CUDA error: {}", ret);
    }
    Ok((mixed, inject, normed_scratch))
}

/// Hyper-Connection write (Gated Residual post-block).
#[cfg(feature = "cuda")]
pub fn hc_write(
    hyper_input: &Tensor,
    block_out: &Tensor,
    inject_weights: &Tensor,
    hc: usize,
    hidden: usize,
) -> Result<Tensor> {
    let (seq_len, hc_hidden) = hyper_input.dims2()?;
    if hc_hidden != hc * hidden {
        candle_core::bail!(
            "hc_write: expected last dim {}, got {}",
            hc * hidden,
            hc_hidden
        );
    }
    let out = Tensor::zeros(
        (seq_len, hc_hidden),
        hyper_input.dtype(),
        hyper_input.device(),
    )?;
    check_same_dtype(
        "hc_write",
        hyper_input.dtype(),
        &[block_out, inject_weights],
    )?;
    let dcode = dtype_code(hyper_input.dtype())?;
    let stream = get_cuda_stream(hyper_input.device())?;
    let ret = unsafe {
        crate::kernels::ffi::qwen4_hc_write(
            get_cuda_ptr(hyper_input)?,
            get_cuda_ptr(block_out)?,
            get_cuda_ptr(inject_weights)?,
            get_cuda_ptr_mut(&out)?,
            seq_len as i32,
            hc as i32,
            hidden as i32,
            dcode,
            stream,
        )
    };
    if ret != 0 {
        candle_core::bail!("qwen4_hc_write CUDA error: {}", ret);
    }
    Ok(out)
}

/// Build QSA sparse attention additive mask [seq, kv_len].
#[cfg(feature = "cuda")]
pub fn qsa_indexer_mask(
    q: &Tensor,
    raw_keys: &Tensor,
    q_norm_weight: &Tensor,
    k_norm_weight: &Tensor,
    cos_table: &Tensor,
    sin_table: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    compress_ratio: usize,
    block_topk: usize,
    eps: f32,
) -> Result<Tensor> {
    let seq_len = q.dim(0)?;
    let kv_len = raw_keys.dim(0)?;
    // Kernel scores at most 512 complete blocks (stack-buffer limit), i.e.
    // 512 * compress_ratio visible tokens (2048 for the reference config) — the
    // exact token budget of Qwen3.8-Flash-Next. Longer contexts truncate block
    // scoring to the first 512 blocks (mask application is not wired up yet).
    if kv_len > 512 * compress_ratio {
        tracing::warn!(
            kv_len,
            max_supported = 512 * compress_ratio,
            "qwen4 qsa_indexer_mask: context exceeds exact block-scoring range"
        );
    }
    let device = q.device();
    check_same_dtype(
        "qsa_indexer_mask",
        q.dtype(),
        &[raw_keys, q_norm_weight, k_norm_weight],
    )?;
    let dcode = dtype_code(q.dtype())?;
    // NOTE: must be a real dense allocation — `Tensor::full` is a zero-copy
    // broadcast view over a single scalar, so passing its pointer to the kernel
    // would write out of bounds. The kernel overwrites every element anyway.
    let mask = Tensor::zeros((seq_len, kv_len), DType::F32, device)?;
    let score_scale = 1.0 / (head_dim as f32).sqrt();
    let cos_f32 = cos_table.to_dtype(DType::F32)?;
    let sin_f32 = sin_table.to_dtype(DType::F32)?;
    let stream = get_cuda_stream(device)?;
    let ret = unsafe {
        crate::kernels::ffi::qwen4_qsa_indexer_mask(
            get_cuda_ptr(q)?,
            get_cuda_ptr(raw_keys)?,
            get_cuda_ptr(q_norm_weight)?,
            get_cuda_ptr(k_norm_weight)?,
            get_cuda_ptr(&cos_f32)?,
            get_cuda_ptr(&sin_f32)?,
            get_cuda_ptr_mut(&mask)?,
            seq_len as i32,
            kv_len as i32,
            n_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            compress_ratio as i32,
            block_topk as i32,
            score_scale,
            f32::NEG_INFINITY,
            eps,
            dcode,
            stream,
        )
    };
    if ret != 0 {
        candle_core::bail!("qwen4_qsa_indexer_mask CUDA error: {}", ret);
    }
    Ok(mask)
}

#[cfg(not(feature = "cuda"))]
pub fn hc_read(
    _hyper_input: &Tensor,
    _hc_norm_weight: &Tensor,
    _mix_down_weight: &Tensor,
    _mix_up_weight: &Tensor,
    _inject_weight: Option<&Tensor>,
    _hc: usize,
    _hidden: usize,
    _lowrank: usize,
    _eps: f32,
) -> Result<(Tensor, Option<Tensor>, Tensor)> {
    candle_core::bail!("qwen4 hc_read requires cuda feature")
}

#[cfg(not(feature = "cuda"))]
pub fn hc_write(
    _hyper_input: &Tensor,
    _block_out: &Tensor,
    _inject_weights: &Tensor,
    _hc: usize,
    _hidden: usize,
) -> Result<Tensor> {
    candle_core::bail!("qwen4 hc_write requires cuda feature")
}

#[cfg(not(feature = "cuda"))]
pub fn qsa_indexer_mask(
    _q: &Tensor,
    _raw_keys: &Tensor,
    _q_norm_weight: &Tensor,
    _k_norm_weight: &Tensor,
    _cos_table: &Tensor,
    _sin_table: &Tensor,
    _n_heads: usize,
    _head_dim: usize,
    _rotary_dim: usize,
    _compress_ratio: usize,
    _block_topk: usize,
    _eps: f32,
) -> Result<Tensor> {
    candle_core::bail!("qwen4 qsa_indexer_mask requires cuda feature")
}
