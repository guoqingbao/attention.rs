//! DeepSeek V4 CUDA kernels: Hyper-Connection, per-head RMSNorm, Compressor,
//! Indexer, sparse attention, and FP8 quantization helpers.

#[cfg(feature = "cuda")]
use crate::kernels;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::Storage;
use candle_core::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
fn get_cuda_ptr(t: &Tensor) -> Result<*const core::ffi::c_void> {
    let (s, l) = t.storage_and_layout();
    match (&*s, t.dtype()) {
        (Storage::Cuda(c), DType::U8 | DType::F8E8M0 | DType::F8E4M3) => Ok(*c
            .as_cuda_slice::<u8>()?
            .slice(l.start_offset()..)
            .device_ptr()
            as *const core::ffi::c_void),
        (Storage::Cuda(c), DType::BF16) => Ok(*c
            .as_cuda_slice::<half::bf16>()?
            .slice(l.start_offset()..)
            .device_ptr()
            as *const core::ffi::c_void),
        (Storage::Cuda(c), DType::F16) => Ok(*c
            .as_cuda_slice::<half::f16>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
        (Storage::Cuda(c), DType::F32) => Ok(*c
            .as_cuda_slice::<f32>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
        (Storage::Cuda(c), DType::I64) => Ok(*c
            .as_cuda_slice::<i64>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
        (Storage::Cuda(c), DType::U32) => Ok(*c
            .as_cuda_slice::<u32>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
        _ => candle_core::bail!(
            "ds_v4 get_cuda_ptr: unsupported dtype {:?} on {:?}",
            t.dtype(),
            t.device()
        ),
    }
}

#[cfg(feature = "cuda")]
fn get_cuda_mut_ptr(t: &Tensor) -> Result<*mut core::ffi::c_void> {
    Ok(get_cuda_ptr(t)? as *mut core::ffi::c_void)
}

#[cfg(feature = "cuda")]
fn get_cuda_stream(device: &Device) -> Result<i64> {
    match device {
        Device::Cuda(dev) => Ok(*dev.cu_stream() as i64),
        _ => candle_core::bail!("ds_v4: expected CUDA device"),
    }
}

/// Bounds-checked asynchronous device-to-device copy for contiguous tensors.
///
/// Candle's current `Tensor::copy_` passes the destination layout to the
/// source storage copy.  That is only safe when both tensors have identical
/// layouts, so V4 uses this primitive for cache rows and streamed expert
/// weights where the destination is intentionally larger than the source.
pub fn copy_contiguous_into(dst: &Tensor, src: &Tensor, dst_element_offset: usize) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if dst.dtype() != src.dtype() {
            candle_core::bail!(
                "copy_contiguous_into dtype mismatch: {:?} vs {:?}",
                dst.dtype(),
                src.dtype()
            );
        }
        if !dst.is_contiguous() || !src.is_contiguous() {
            candle_core::bail!("copy_contiguous_into requires contiguous tensors");
        }
        if !dst.device().same_device(src.device()) {
            candle_core::bail!(
                "copy_contiguous_into requires tensors on the same device: {:?} vs {:?}",
                dst.device(),
                src.device()
            );
        }
        let end = dst_element_offset
            .checked_add(src.elem_count())
            .ok_or_else(|| candle_core::Error::Msg("copy offset overflow".into()))?;
        if end > dst.elem_count() {
            candle_core::bail!(
                "copy_contiguous_into range {}..{} exceeds destination elements {}",
                dst_element_offset,
                end,
                dst.elem_count()
            );
        }
        let element_bytes = dst.dtype().size_in_bytes();
        let stream = get_cuda_stream(dst.device())?;
        let ret = unsafe {
            kernels::ffi::ds_copy_device_bytes(
                get_cuda_ptr(src)?,
                get_cuda_mut_ptr(dst)?,
                src.elem_count() * element_bytes,
                dst_element_offset * element_bytes,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("copy_contiguous_into CUDA error: {}", ret);
        }
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (dst, src, dst_element_offset);
        candle_core::bail!("copy_contiguous_into requires cuda feature")
    }
}

/// Broadcast hidden states along HC dimension: [seq, dim] -> [seq, hc, dim]
pub fn hc_expand(x: &Tensor, hc: usize) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let (seq_len, dim) = (x.dim(0)?, x.dim(1)?);
        let out = Tensor::zeros((seq_len, hc, dim), x.dtype(), x.device())?;
        let stream = get_cuda_stream(x.device())?;
        unsafe {
            kernels::ffi::ds_v4_hc_expand(
                get_cuda_ptr(x)?,
                get_cuda_mut_ptr(&out)?,
                seq_len as i32,
                hc as i32,
                dim as i32,
                stream,
            );
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = hc;
        candle_core::bail!("hc_expand requires cuda feature")
    }
}

/// Compute mixing coefficients with the reference's pedantic F32 GEMM and
/// GPU-side RMS scaling.  The BF16 hidden state is promoted on-device; this
/// keeps the reduction precise without introducing a host synchronization.
///
/// x: [seq, hc*dim] BF16 (HC-expanded, flattened), hc_fn: [mix_hc, hc*dim] F32
/// Returns mixes: [seq, mix_hc] F32
pub fn hc_mixes(
    x: &Tensor,
    hc_fn: &Tensor,
    hc: usize,
    dim: usize,
    mix_hc: usize,
    eps: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let seq_len = x.dim(0)?;
        if x.dtype() != DType::BF16 || hc_fn.dtype() != DType::F32 {
            candle_core::bail!(
                "hc_mixes expects BF16 hidden/F32 weights, got {:?}/{:?}",
                x.dtype(),
                hc_fn.dtype()
            );
        }
        let mixes = Tensor::zeros((seq_len, mix_hc), DType::F32, x.device())?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_v4_hc_mixes(
                get_cuda_ptr(x)?,
                get_cuda_ptr(hc_fn)?,
                get_cuda_mut_ptr(&mixes)?,
                seq_len as i32,
                hc as i32,
                dim as i32,
                mix_hc as i32,
                eps,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("ds_v4_hc_mixes CUDA error: {}", ret);
        }
        Ok(mixes)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let (_, _, _, _, _) = (hc_fn, hc, dim, mix_hc, eps);
        candle_core::bail!("hc_mixes requires cuda feature")
    }
}

/// Fused sinkhorn + pre-output from mixes (hc=4 only).
///
/// x: [seq, hc, dim], mixes: [seq, mix_hc],
/// hc_scale: [3], hc_base: [mix_hc]
/// Returns (pre_out: [seq, dim], post: [seq, hc], comb: [seq, hc, hc])
pub fn hc_pre_from_mixes(
    x: &Tensor,
    mixes: &Tensor,
    hc_scale: &Tensor,
    hc_base: &Tensor,
    hc: usize,
    dim: usize,
    sinkhorn_iters: usize,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        let seq_len = x.dim(0)?;
        let post = Tensor::zeros((seq_len, hc), DType::F32, x.device())?;
        let comb = Tensor::zeros((seq_len, hc, hc), DType::F32, x.device())?;
        let out = Tensor::zeros((seq_len, dim), x.dtype(), x.device())?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_v4_hc_pre_from_mixes(
                get_cuda_ptr(x)?,
                get_cuda_ptr(mixes)?,
                get_cuda_ptr(hc_scale)?,
                get_cuda_ptr(hc_base)?,
                get_cuda_mut_ptr(&post)?,
                get_cuda_mut_ptr(&comb)?,
                get_cuda_mut_ptr(&out)?,
                seq_len as i32,
                hc as i32,
                dim as i32,
                sinkhorn_iters as i32,
                eps,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("ds_v4_hc_pre_from_mixes CUDA error: {}", ret);
        }
        Ok((out, post, comb))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let (_, _, _, _, _, _, _) = (mixes, hc_scale, hc_base, hc, dim, sinkhorn_iters, eps);
        candle_core::bail!("hc_pre_from_mixes requires cuda feature")
    }
}

/// Fused sinkhorn + pre-output + RMSNorm (hc=4 only).
///
/// x: [seq, hc, dim], mixes: [seq, mix_hc],
/// hc_scale: [3], hc_base: [mix_hc], norm_weight: [dim]
/// Returns (normed_out: [seq, dim], post: [seq, hc], comb: [seq, hc, hc])
pub fn hc_pre_norm_from_mixes(
    x: &Tensor,
    mixes: &Tensor,
    hc_scale: &Tensor,
    hc_base: &Tensor,
    norm_weight: &Tensor,
    hc: usize,
    dim: usize,
    sinkhorn_iters: usize,
    hc_eps: f32,
    norm_eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        let seq_len = x.dim(0)?;
        let post = Tensor::zeros((seq_len, hc), DType::F32, x.device())?;
        let comb = Tensor::zeros((seq_len, hc, hc), DType::F32, x.device())?;
        let out = Tensor::zeros((seq_len, dim), x.dtype(), x.device())?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_v4_hc_pre_norm_from_mixes(
                get_cuda_ptr(x)?,
                get_cuda_ptr(mixes)?,
                get_cuda_ptr(hc_scale)?,
                get_cuda_ptr(hc_base)?,
                get_cuda_ptr(norm_weight)?,
                get_cuda_mut_ptr(&post)?,
                get_cuda_mut_ptr(&comb)?,
                get_cuda_mut_ptr(&out)?,
                seq_len as i32,
                hc as i32,
                dim as i32,
                sinkhorn_iters as i32,
                hc_eps,
                norm_eps,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("ds_v4_hc_pre_norm_from_mixes CUDA error: {}", ret);
        }
        Ok((out, post, comb))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let (_, _, _, _, _, _, _, _, _) = (
            mixes,
            hc_scale,
            hc_base,
            norm_weight,
            hc,
            dim,
            sinkhorn_iters,
            hc_eps,
            norm_eps,
        );
        candle_core::bail!("hc_pre_norm_from_mixes requires cuda feature")
    }
}

/// Weighted sum over HC branches: out[t,d] = sum_h(pre[t,h] * x[t,h,d])
///
/// x: [seq, hc, dim], pre: [seq, hc]
/// Returns: [seq, dim]
pub fn hc_pre_output(x: &Tensor, pre: &Tensor, hc: usize, dim: usize) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let seq_len = x.dim(0)?;
        let out = Tensor::zeros((seq_len, dim), x.dtype(), x.device())?;
        let stream = get_cuda_stream(x.device())?;
        unsafe {
            kernels::ffi::ds_v4_hc_pre_output(
                get_cuda_ptr(x)?,
                get_cuda_ptr(pre)?,
                get_cuda_mut_ptr(&out)?,
                seq_len as i32,
                hc as i32,
                dim as i32,
                stream,
            );
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let (_, _, _) = (pre, hc, dim);
        candle_core::bail!("hc_pre_output requires cuda feature")
    }
}

/// Compute HC head pre-mix gates (sigmoid).
///
/// mixes: [seq, hc], hc_scale: [1], hc_base: [hc]
/// Returns pre: [seq, hc]
pub fn hc_head_pre(
    mixes: &Tensor,
    hc_scale: &Tensor,
    hc_base: &Tensor,
    hc: usize,
    eps: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let seq_len = mixes.dim(0)?;
        let pre = Tensor::zeros((seq_len, hc), DType::F32, mixes.device())?;
        let stream = get_cuda_stream(mixes.device())?;
        unsafe {
            kernels::ffi::ds_v4_hc_head_pre(
                get_cuda_ptr(mixes)?,
                get_cuda_ptr(hc_scale)?,
                get_cuda_ptr(hc_base)?,
                get_cuda_mut_ptr(&pre)?,
                seq_len as i32,
                hc as i32,
                eps,
                stream,
            );
        }
        Ok(pre)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let (_, _, _) = (hc_scale, hc_base, eps);
        candle_core::bail!("hc_head_pre requires cuda feature")
    }
}

/// HC post-mix: out[t,h,d] = post[t,h]*x[t,d] + sum_k(comb[t,k,h]*residual[t,k,d])
///
/// x: [seq, dim] (branch output), residual: [seq, hc, dim],
/// post: [seq, hc], comb: [seq, hc, hc]
/// Returns: [seq, hc, dim]
pub fn hc_post(
    x: &Tensor,
    residual: &Tensor,
    post: &Tensor,
    comb: &Tensor,
    hc: usize,
    dim: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        // Contiguous dense views are required: non-contiguous / aliased layouts
        // from upstream workspace reuse previously fed the CUDA kernels stale data.
        let x = x.contiguous()?;
        let residual = residual.contiguous()?;
        let post = post.contiguous()?;
        let comb = comb.contiguous()?;
        let seq_len = x.dim(0)?;
        // Both kernels write BF16 HC state so prefill/decode share one output dtype.
        let out = Tensor::zeros((seq_len, hc, dim), DType::BF16, x.device())?;
        let stream = get_cuda_stream(x.device())?;
        match x.dtype() {
            DType::F32 => {
                let ret = unsafe {
                    kernels::ffi::ds_v4_hc_post_f32_branch(
                        get_cuda_ptr(&x)?,
                        get_cuda_ptr(&residual)?,
                        get_cuda_ptr(&post)?,
                        get_cuda_ptr(&comb)?,
                        get_cuda_mut_ptr(&out)?,
                        seq_len as i32,
                        hc as i32,
                        dim as i32,
                        stream,
                    )
                };
                if ret != 0 {
                    candle_core::bail!("ds_v4_hc_post_f32_branch CUDA error: {}", ret);
                }
            }
            DType::BF16 => {
                let ret = unsafe {
                    kernels::ffi::ds_v4_hc_post(
                        get_cuda_ptr(&x)?,
                        get_cuda_ptr(&residual)?,
                        get_cuda_ptr(&post)?,
                        get_cuda_ptr(&comb)?,
                        get_cuda_mut_ptr(&out)?,
                        seq_len as i32,
                        hc as i32,
                        dim as i32,
                        stream,
                    )
                };
                if ret != 0 {
                    candle_core::bail!("ds_v4_hc_post CUDA error: {}", ret);
                }
            }
            dtype => candle_core::bail!("hc_post expects BF16 or F32 branch, got {dtype:?}"),
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let (_, _, _, _) = (residual, post, comb, dim);
        candle_core::bail!("hc_post requires cuda feature")
    }
}

/// Official-formula HC post in Candle (no custom CUDA). Kept for tests /
/// reference comparisons; production uses the CUDA path above.
pub fn hc_post_reference(
    x: &Tensor,
    residual: &Tensor,
    post: &Tensor,
    comb: &Tensor,
    hc: usize,
    dim: usize,
) -> Result<Tensor> {
    let seq_len = x.dim(0)?;
    // Reproduce the official BF16 branch boundary before the post multiply.
    let x = match x.dtype() {
        DType::F32 => x.to_dtype(DType::BF16)?.to_dtype(DType::F32)?,
        DType::BF16 => x.to_dtype(DType::F32)?,
        dtype => candle_core::bail!("hc_post_reference expects BF16/F32 branch, got {dtype:?}"),
    };
    let residual = residual.to_dtype(DType::F32)?.reshape((seq_len, hc, dim))?;
    let post = post.to_dtype(DType::F32)?.reshape((seq_len, hc))?;
    let comb = comb.to_dtype(DType::F32)?.reshape((seq_len, hc, hc))?;

    // post_term: [seq, hc, 1] * [seq, 1, dim] -> [seq, hc, dim]
    let post_term = post
        .unsqueeze(candle_core::D::Minus1)?
        .broadcast_mul(&x.unsqueeze(1)?)?;
    // comb: [seq, h_in, h_out, 1] * residual: [seq, h_in, 1, dim]
    // -> sum over h_in => [seq, h_out, dim]
    let residual_sum = comb
        .unsqueeze(candle_core::D::Minus1)?
        .broadcast_mul(&residual.unsqueeze(2)?)?
        .sum(1)?;
    (post_term + residual_sum)?.to_dtype(DType::BF16)
}

/// HC post-mix after an FP32 collective, matching the reference V4 path: the
/// row-parallel branch is reduced in FP32, rounded once to BF16 at the branch
/// boundary, and the final HC state is stored as BF16.
pub fn hc_post_f32_branch(
    x: &Tensor,
    residual: &Tensor,
    post: &Tensor,
    comb: &Tensor,
    hc: usize,
    dim: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if x.dtype() != DType::F32 {
            candle_core::bail!("hc_post_f32_branch expects F32 branch, got {:?}", x.dtype());
        }
        let x = x.contiguous()?;
        let residual = residual.contiguous()?;
        let post = post.contiguous()?;
        let comb = comb.contiguous()?;
        let seq_len = x.dim(0)?;
        let out = Tensor::zeros((seq_len, hc, dim), DType::BF16, x.device())?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_v4_hc_post_f32_branch(
                get_cuda_ptr(&x)?,
                get_cuda_ptr(&residual)?,
                get_cuda_ptr(&post)?,
                get_cuda_ptr(&comb)?,
                get_cuda_mut_ptr(&out)?,
                seq_len as i32,
                hc as i32,
                dim as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("ds_v4_hc_post_f32_branch CUDA error: {}", ret);
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (x, residual, post, comb, hc, dim);
        candle_core::bail!("hc_post_f32_branch requires cuda feature")
    }
}

/// Per-head RMSNorm: normalize each head independently.
///
/// x: [seq, num_heads, head_dim]
/// Returns: [seq, num_heads, head_dim]
/// ATen-order RMSNorm matching the official V4 golden.
/// x: [rows, dim] BF16, weight: [dim] F32.  All math in F32 with the same
/// (128,4) block reduce order as `torch.mean(-1)`, one BF16 round at the end.
pub fn rms_norm_v4(x: &Tensor, weight: &Tensor, dim: usize, eps: f32) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let rows = x.dim(0)?;
        let out = Tensor::zeros_like(x)?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_v4_rms_norm(
                get_cuda_ptr(x)?,
                get_cuda_ptr(weight)?,
                get_cuda_mut_ptr(&out)?,
                rows as i32,
                dim as i32,
                eps,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("ds_v4_rms_norm CUDA error: {}", ret);
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (weight, dim, eps);
        candle_core::bail!("rms_norm_v4 requires cuda feature")
    }
}

pub fn head_rms_norm(x: &Tensor, num_heads: usize, head_dim: usize, eps: f32) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let seq_len = x.dim(0)?;
        let out = Tensor::zeros_like(x)?;
        let stream = get_cuda_stream(x.device())?;
        unsafe {
            kernels::ffi::ds_v4_head_rms_norm(
                get_cuda_ptr(x)?,
                get_cuda_mut_ptr(&out)?,
                seq_len as i32,
                num_heads as i32,
                head_dim as i32,
                eps,
                stream,
            );
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let (_, _, _) = (num_heads, head_dim, eps);
        candle_core::bail!("head_rms_norm requires cuda feature")
    }
}

// ============================================================================
// Compressor kernels
// ============================================================================

fn rope_hidden_shape(x: &Tensor) -> Result<(usize, usize, usize)> {
    let dims = x.dims();
    if !(2..=3).contains(&dims.len()) {
        candle_core::bail!(
            "apply_rope_hidden expects [seq, head_dim] or [seq, heads, head_dim], got {:?}",
            dims
        );
    }
    let seq_len = dims[0];
    let head_dim = dims[dims.len() - 1];
    if seq_len == 0 || head_dim == 0 {
        candle_core::bail!(
            "apply_rope_hidden dimensions must be non-zero, got {:?}",
            dims
        );
    }
    let local_heads = x.elem_count() / (seq_len * head_dim);
    Ok((seq_len, local_heads, head_dim))
}

/// Apply RoPE in place to the final `rope_dim` values of every head.
///
/// `x` is contiguous BF16 `[seq, head_dim]` or `[seq, heads, head_dim]`.
/// `cos` and `sin` are contiguous F32 `[max_seq, rope_dim / 2]` tables.
pub fn apply_rope_hidden_inplace(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    start_pos: usize,
    rope_dim: usize,
    inverse: bool,
) -> Result<()> {
    apply_rope_hidden_strided_inplace(x, cos, sin, start_pos, 1, rope_dim, inverse)
}

/// Apply hidden-state RoPE using table rows `start_pos + token * position_stride`.
pub fn apply_rope_hidden_strided_inplace(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    start_pos: usize,
    position_stride: usize,
    rope_dim: usize,
    inverse: bool,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        let (seq_len, local_heads, head_dim) = rope_hidden_shape(x)?;
        if x.dtype() != DType::BF16 || cos.dtype() != DType::F32 || sin.dtype() != DType::F32 {
            candle_core::bail!(
                "apply_rope_hidden expects x BF16 and cos/sin F32, got {:?}, {:?}, {:?}",
                x.dtype(),
                cos.dtype(),
                sin.dtype()
            );
        }
        if !x.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
            candle_core::bail!("apply_rope_hidden expects contiguous tensors");
        }
        if rope_dim == 0 || rope_dim > head_dim || rope_dim % 2 != 0 {
            candle_core::bail!(
                "invalid rope_dim {rope_dim} for head_dim {head_dim}; it must be positive and even"
            );
        }
        if position_stride == 0 {
            candle_core::bail!("position_stride must be positive");
        }
        let last_pos = start_pos
            .checked_add((seq_len - 1).saturating_mul(position_stride))
            .ok_or_else(|| candle_core::Error::Msg("RoPE position overflow".to_string()))?;
        let required_freqs = (last_pos + 1)
            .checked_mul(rope_dim / 2)
            .ok_or_else(|| candle_core::Error::Msg("RoPE table size overflow".to_string()))?;
        if cos.elem_count() < required_freqs || sin.elem_count() < required_freqs {
            candle_core::bail!(
                "RoPE tables are too small: need {required_freqs} values, got cos={} sin={}",
                cos.elem_count(),
                sin.elem_count()
            );
        }
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_apply_rope_hidden_strided(
                get_cuda_mut_ptr(x)?,
                get_cuda_ptr(cos)? as *const f32,
                get_cuda_ptr(sin)? as *const f32,
                seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rope_dim as i32,
                start_pos as i32,
                position_stride as i32,
                if inverse { 1 } else { 0 },
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("apply_rope_hidden CUDA error: {}", ret);
        }
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (x, cos, sin, start_pos, position_stride, rope_dim, inverse);
        candle_core::bail!("apply_rope_hidden requires cuda feature")
    }
}

/// BF16 -> F32 compressor projection using the reference cuBLAS GEMM path.
/// The output is row-major `[seq_len, out_dim]` and remains device-resident.
pub fn compressor_bf16_linear_f32(
    x: &Tensor,
    weight: &Tensor,
    seq_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if x.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
            candle_core::bail!(
                "compressor BF16 GEMM expects BF16 inputs, got {:?}/{:?}",
                x.dtype(),
                weight.dtype()
            );
        }
        if !x.is_contiguous() || !weight.is_contiguous() {
            candle_core::bail!("compressor BF16 GEMM expects contiguous inputs");
        }
        let out = Tensor::zeros((seq_len, out_dim), DType::F32, x.device())?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_compressor_bf16_linear_f32(
                get_cuda_ptr(x)?,
                get_cuda_ptr(weight)?,
                get_cuda_mut_ptr(&out)? as *mut f32,
                seq_len as i32,
                in_dim as i32,
                out_dim as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("ds_compressor_bf16_linear_f32 CUDA error: {}", ret);
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (x, weight, seq_len, in_dim, out_dim);
        candle_core::bail!("compressor_bf16_linear_f32 requires cuda feature")
    }
}

/// Non-overlap compressor prefill: cuBLAS projection + fused epilogue via CUDA.
///
/// x: [seq_len, hidden_dim] BF16
/// wkv: [head_dim, hidden_dim] BF16, wgate: [head_dim, hidden_dim] BF16
/// ape: [ratio, head_dim] F32, norm: [head_dim] BF16
/// Returns: (weighted: [compressed_len, head_dim] F32, out: [compressed_len, head_dim] BF16)
pub fn compressor_nonoverlap_prefill(
    x: &Tensor,
    wkv: &Tensor,
    wgate: &Tensor,
    ape: &Tensor,
    norm: &Tensor,
    seq_len: usize,
    _hidden_dim: usize,
    head_dim: usize,
    ratio: usize,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        let compressed_len = seq_len / ratio;
        // Keep BF16 operands and accumulate into F32.  Converting both
        // operands to F32 before Candle's matmul selected the SM90 default
        // math mode (and potentially TF32); it is not the OpenInfer path.
        let scores = compressor_bf16_linear_f32(x, wgate, seq_len, _hidden_dim, head_dim)?;
        let values = compressor_bf16_linear_f32(x, wkv, seq_len, _hidden_dim, head_dim)?;

        let weighted = Tensor::zeros((compressed_len, head_dim), DType::F32, x.device())?;
        let out = Tensor::zeros((compressed_len, head_dim), DType::BF16, x.device())?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_compressor_nonoverlap_prefill_epilogue(
                get_cuda_ptr(&scores)? as *const f32,
                get_cuda_ptr(&values)? as *const f32,
                get_cuda_ptr(ape)? as *const f32,
                get_cuda_ptr(norm)?,
                get_cuda_mut_ptr(&weighted)? as *mut f32,
                get_cuda_mut_ptr(&out)?,
                seq_len as i32,
                head_dim as i32,
                ratio as i32,
                eps,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("compressor_nonoverlap_prefill_epilogue CUDA error: {}", ret);
        }
        Ok((weighted, out))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            x,
            wkv,
            wgate,
            ape,
            norm,
            seq_len,
            _hidden_dim,
            head_dim,
            ratio,
            eps,
        );
        candle_core::bail!("compressor_nonoverlap_prefill requires cuda feature")
    }
}

/// Overlap compressor prefill (ratio=4): cuBLAS projection + fused epilogue via CUDA.
///
/// Returns: (weighted: [compressed_len, head_dim] F32, out: [compressed_len, head_dim] BF16)
pub fn compressor_overlap_prefill(
    x: &Tensor,
    wkv: &Tensor,
    wgate: &Tensor,
    ape: &Tensor,
    norm: &Tensor,
    seq_len: usize,
    _hidden_dim: usize,
    head_dim: usize,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        let ratio = 4usize;
        let compressed_len = seq_len / ratio;
        // Overlap projections go to 2*head_dim and use BF16 operands with F32
        // accumulation, matching the reference layout and math mode.
        let scores = compressor_bf16_linear_f32(x, wgate, seq_len, _hidden_dim, 2 * head_dim)?;
        let values = compressor_bf16_linear_f32(x, wkv, seq_len, _hidden_dim, 2 * head_dim)?;

        let weighted = Tensor::zeros((compressed_len, head_dim), DType::F32, x.device())?;
        let out = Tensor::zeros((compressed_len, head_dim), DType::BF16, x.device())?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_compressor_overlap_prefill_epilogue(
                get_cuda_ptr(&scores)? as *const f32,
                get_cuda_ptr(&values)? as *const f32,
                get_cuda_ptr(ape)? as *const f32,
                get_cuda_ptr(norm)?,
                get_cuda_mut_ptr(&weighted)? as *mut f32,
                get_cuda_mut_ptr(&out)?,
                seq_len as i32,
                head_dim as i32,
                eps,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("compressor_overlap_prefill_epilogue CUDA error: {}", ret);
        }
        Ok((weighted, out))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            x,
            wkv,
            wgate,
            ape,
            norm,
            seq_len,
            _hidden_dim,
            head_dim,
            eps,
        );
        candle_core::bail!("compressor_overlap_prefill requires cuda feature")
    }
}

/// Non-overlap compressor decode-at: process single token and accumulate into KV/score state.
///
/// kv_state/score_state: mutable accumulation buffers (F32)
/// Returns: Option<(weighted, out)> — Some if a compressed token was emitted (start_pos+1 % ratio == 0)
pub fn compressor_nonoverlap_decode_at(
    x: &Tensor,
    wkv: &Tensor,
    wgate: &Tensor,
    ape: &Tensor,
    norm: &Tensor,
    kv_state: &Tensor,
    score_state: &Tensor,
    start_pos: usize,
    hidden_dim: usize,
    head_dim: usize,
    ratio: usize,
    state_offset: usize,
    eps: f32,
) -> Result<Option<(Tensor, Tensor)>> {
    #[cfg(feature = "cuda")]
    {
        let should_compress = (start_pos + 1) % ratio == 0;
        let weighted = should_compress
            .then(|| Tensor::zeros((1, head_dim), DType::F32, x.device()))
            .transpose()?;
        let out = should_compress
            .then(|| Tensor::zeros((1, head_dim), DType::BF16, x.device()))
            .transpose()?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_compressor_nonoverlap_decode_at(
                get_cuda_ptr(x)?,
                get_cuda_ptr(wkv)?,
                get_cuda_ptr(wgate)?,
                get_cuda_ptr(ape)? as *const f32,
                get_cuda_ptr(norm)?,
                get_cuda_mut_ptr(kv_state)? as *mut f32,
                get_cuda_mut_ptr(score_state)? as *mut f32,
                weighted
                    .as_ref()
                    .map(get_cuda_mut_ptr)
                    .transpose()?
                    .unwrap_or(std::ptr::null_mut()) as *mut f32,
                out.as_ref()
                    .map(get_cuda_mut_ptr)
                    .transpose()?
                    .unwrap_or(std::ptr::null_mut()),
                start_pos as i32,
                hidden_dim as i32,
                head_dim as i32,
                ratio as i32,
                state_offset as i32,
                eps,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("compressor_nonoverlap_decode_at CUDA error: {}", ret);
        }
        Ok(weighted.zip(out))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            x,
            wkv,
            wgate,
            ape,
            norm,
            kv_state,
            score_state,
            start_pos,
            hidden_dim,
            head_dim,
            ratio,
            state_offset,
            eps,
        );
        candle_core::bail!("compressor_nonoverlap_decode_at requires cuda feature")
    }
}

/// Overlap compressor decode-at with explicit state_offset.
pub fn compressor_overlap_decode_at(
    x: &Tensor,
    wkv: &Tensor,
    wgate: &Tensor,
    ape: &Tensor,
    norm: &Tensor,
    kv_state: &Tensor,
    score_state: &Tensor,
    start_pos: usize,
    hidden_dim: usize,
    head_dim: usize,
    state_offset: usize,
    eps: f32,
) -> Result<Option<(Tensor, Tensor)>> {
    #[cfg(feature = "cuda")]
    {
        let should_compress = (start_pos + 1) % 4 == 0;
        let weighted = should_compress
            .then(|| Tensor::zeros((1, head_dim), DType::F32, x.device()))
            .transpose()?;
        let out = should_compress
            .then(|| Tensor::zeros((1, head_dim), DType::BF16, x.device()))
            .transpose()?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_compressor_overlap_decode_at(
                get_cuda_ptr(x)?,
                get_cuda_ptr(wkv)?,
                get_cuda_ptr(wgate)?,
                get_cuda_ptr(ape)? as *const f32,
                get_cuda_ptr(norm)?,
                get_cuda_mut_ptr(kv_state)? as *mut f32,
                get_cuda_mut_ptr(score_state)? as *mut f32,
                weighted
                    .as_ref()
                    .map(get_cuda_mut_ptr)
                    .transpose()?
                    .unwrap_or(std::ptr::null_mut()) as *mut f32,
                out.as_ref()
                    .map(get_cuda_mut_ptr)
                    .transpose()?
                    .unwrap_or(std::ptr::null_mut()),
                start_pos as i32,
                hidden_dim as i32,
                head_dim as i32,
                state_offset as i32,
                eps,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("compressor_overlap_decode_at CUDA error: {}", ret);
        }
        Ok(weighted.zip(out))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            x,
            wkv,
            wgate,
            ape,
            norm,
            kv_state,
            score_state,
            start_pos,
            hidden_dim,
            head_dim,
            state_offset,
            eps,
        );
        candle_core::bail!("compressor_overlap_decode_at requires cuda feature")
    }
}

// ============================================================================
// Indexer kernels
// ============================================================================

/// Apply normalized Hadamard rotation and FP4 quantize-dequantize in place.
///
/// `x` is contiguous BF16 with `x.elem_count() == rows * groups * dim`.
/// OpenInfer's current indexer contract only supports `dim == 128`.
pub fn hadamard_fp4_quant_bf16_inplace(x: &Tensor, groups: usize, dim: usize) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if x.dtype() != DType::BF16 {
            candle_core::bail!(
                "hadamard_fp4_quant_bf16_inplace expects BF16, got {:?}",
                x.dtype()
            );
        }
        if !x.is_contiguous() {
            candle_core::bail!("hadamard_fp4_quant_bf16_inplace expects a contiguous tensor");
        }
        if groups == 0 || dim != 128 || x.elem_count() % (groups * dim) != 0 {
            candle_core::bail!(
                "invalid Hadamard layout: elements={}, groups={}, dim={} (dim must be 128)",
                x.elem_count(),
                groups,
                dim
            );
        }
        let rows = x.elem_count() / (groups * dim);
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_hadamard_fp4_quant_bf16(
                get_cuda_mut_ptr(x)?,
                rows as i32,
                groups as i32,
                dim as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("hadamard_fp4_quant_bf16 CUDA error: {}", ret);
        }
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (x, groups, dim);
        candle_core::bail!("hadamard_fp4_quant_bf16_inplace requires cuda feature")
    }
}

/// Official QAT: FP8 E4M3 + UE8M0 block scales on non-RoPE (nope) dims, in place.
///
/// Layout: BF16 `[seq_len, local_heads, head_dim]` or contiguous
/// `seq_len * local_heads * head_dim`. RoPE dims at the end stay untouched.
pub fn fp8_act_quant_nope_bf16_inplace(
    x: &Tensor,
    local_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    block_size: usize,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if x.dtype() != DType::BF16 {
            candle_core::bail!("fp8_act_quant_nope expects BF16, got {:?}", x.dtype());
        }
        if !x.is_contiguous() {
            candle_core::bail!("fp8_act_quant_nope expects a contiguous tensor");
        }
        if local_heads == 0 || head_dim == 0 || rotary_dim >= head_dim {
            candle_core::bail!(
                "invalid fp8_act_quant_nope layout: heads={}, head_dim={}, rotary={}",
                local_heads,
                head_dim,
                rotary_dim
            );
        }
        let nope = head_dim - rotary_dim;
        if nope % block_size != 0 {
            candle_core::bail!(
                "fp8_act_quant_nope: nope_dim {nope} not divisible by block_size {block_size}"
            );
        }
        let elems = x.elem_count();
        if elems % (local_heads * head_dim) != 0 {
            candle_core::bail!(
                "fp8_act_quant_nope: elements {elems} not divisible by heads*dim={}",
                local_heads * head_dim
            );
        }
        let seq_len = elems / (local_heads * head_dim);
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_fp8_act_quant_nope_bf16(
                get_cuda_ptr(x)?,
                get_cuda_mut_ptr(x)?,
                seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rotary_dim as i32,
                block_size as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("fp8_act_quant_nope CUDA error: {}", ret);
        }
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (x, local_heads, head_dim, rotary_dim, block_size);
        candle_core::bail!("fp8_act_quant_nope_bf16_inplace requires cuda feature")
    }
}

/// Apply the DeepSeek-V4 FP8 activation Q/DQ into distinct storage.
///
/// The routed and shared experts consume the same normalized activation.  An
/// in-place quantizer would therefore race the already-enqueued shared-expert
/// GEMM and permanently modify its caller's tensor.  This out-of-place form
/// fuses the copy with quantization for the common all-non-RoPE projection.
pub fn fp8_act_quant_nope_bf16(
    x: &Tensor,
    local_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    block_size: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if x.dtype() != DType::BF16 {
            candle_core::bail!("fp8_act_quant_nope expects BF16, got {:?}", x.dtype());
        }
        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        if local_heads == 0 || head_dim == 0 || rotary_dim >= head_dim {
            candle_core::bail!(
                "invalid fp8_act_quant_nope layout: heads={}, head_dim={}, rotary={}",
                local_heads,
                head_dim,
                rotary_dim
            );
        }
        let nope = head_dim - rotary_dim;
        if nope % block_size != 0 {
            candle_core::bail!(
                "fp8_act_quant_nope: nope_dim {nope} not divisible by block_size {block_size}"
            );
        }
        let elems = x.elem_count();
        if elems % (local_heads * head_dim) != 0 {
            candle_core::bail!(
                "fp8_act_quant_nope: elements {elems} not divisible by heads*dim={}",
                local_heads * head_dim
            );
        }
        let seq_len = elems / (local_heads * head_dim);
        let output = Tensor::zeros(x.shape(), DType::BF16, x.device())?;
        // Preserve the rotary tail when this helper is used on attention
        // tensors.  MoE passes rotary_dim=0, so that hot path has no copy.
        if rotary_dim != 0 {
            copy_contiguous_into(&output, &x, 0)?;
        }
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_fp8_act_quant_nope_bf16(
                get_cuda_ptr(&x)?,
                get_cuda_mut_ptr(&output)?,
                seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                rotary_dim as i32,
                block_size as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("fp8_act_quant_nope CUDA error: {}", ret);
        }
        Ok(output)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (x, local_heads, head_dim, rotary_dim, block_size);
        candle_core::bail!("fp8_act_quant_nope_bf16 requires cuda feature")
    }
}

/// Compute indexer scores for prefill: dot product between Q and compressed KV.
///
/// q: [seq_len, local_heads*head_dim] BF16
/// kv: [compressed_len, head_dim] BF16 (shared across heads)
/// weights: [local_heads, head_dim] BF16 (indexer linear weights)
/// Returns: [seq_len, compressed_len] F32 scores
pub fn indexer_scores_prefill(
    q: &Tensor,
    kv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    local_heads: usize,
    head_dim: usize,
    compressed_len: usize,
    score_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let scores = Tensor::zeros((seq_len, compressed_len), DType::F32, q.device())?;
        let stream = get_cuda_stream(q.device())?;
        let ret = unsafe {
            kernels::ffi::ds_indexer_scores_prefill(
                get_cuda_ptr(q)?,
                get_cuda_ptr(kv)?,
                get_cuda_ptr(weights)?,
                get_cuda_mut_ptr(&scores)? as *mut f32,
                seq_len as i32,
                local_heads as i32,
                head_dim as i32,
                compressed_len as i32,
                score_scale,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("indexer_scores_prefill CUDA error: {}", ret);
        }
        Ok(scores)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            q,
            kv,
            weights,
            seq_len,
            local_heads,
            head_dim,
            compressed_len,
            score_scale,
        );
        candle_core::bail!("indexer_scores_prefill requires cuda feature")
    }
}

/// Compute indexer scores for decode (single token).
///
/// q: [1, local_heads*head_dim] BF16
/// kv: [compressed_len, head_dim] BF16
/// weights: [local_heads, head_dim] BF16
/// Returns: [compressed_len] F32 scores
pub fn indexer_scores_decode(
    q: &Tensor,
    kv: &Tensor,
    weights: &Tensor,
    local_heads: usize,
    head_dim: usize,
    compressed_len: usize,
    score_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let scores = Tensor::zeros(compressed_len, DType::F32, q.device())?;
        let stream = get_cuda_stream(q.device())?;
        let ret = unsafe {
            kernels::ffi::ds_indexer_scores_decode(
                get_cuda_ptr(q)?,
                get_cuda_ptr(kv)?,
                get_cuda_ptr(weights)?,
                get_cuda_mut_ptr(&scores)? as *mut f32,
                local_heads as i32,
                head_dim as i32,
                compressed_len as i32,
                score_scale,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("indexer_scores_decode CUDA error: {}", ret);
        }
        Ok(scores)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            q,
            kv,
            weights,
            local_heads,
            head_dim,
            compressed_len,
            score_scale,
        );
        candle_core::bail!("indexer_scores_decode requires cuda feature")
    }
}

/// Top-k selection from indexer scores (prefill).
///
/// scores: [seq_len, compressed_len] F32
/// Returns: [seq_len, topk] I32 indices into compressed KV
pub fn indexer_topk_prefill(
    scores: &Tensor,
    seq_len: usize,
    compressed_len: usize,
    topk: usize,
    ratio: usize,
    offset: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let topk_idxs = Tensor::zeros((seq_len, topk), DType::U32, scores.device())?;
        let stream = get_cuda_stream(scores.device())?;
        let ret = unsafe {
            kernels::ffi::ds_indexer_topk_prefill(
                get_cuda_ptr(scores)? as *const f32,
                get_cuda_mut_ptr(&topk_idxs)? as *mut i32,
                seq_len as i32,
                compressed_len as i32,
                topk as i32,
                ratio as i32,
                offset as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("indexer_topk_prefill CUDA error: {}", ret);
        }
        Ok(topk_idxs)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (scores, seq_len, compressed_len, topk, ratio, offset);
        candle_core::bail!("indexer_topk_prefill requires cuda feature")
    }
}

/// Top-k selection from indexer scores (decode, single token).
///
/// scores: [compressed_len] F32
/// Returns: [topk] I32 indices
pub fn indexer_topk_decode(
    scores: &Tensor,
    compressed_len: usize,
    topk: usize,
    offset: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let topk_idxs = Tensor::zeros(topk, DType::U32, scores.device())?;
        let stream = get_cuda_stream(scores.device())?;
        let ret = unsafe {
            kernels::ffi::ds_indexer_topk_decode(
                get_cuda_ptr(scores)? as *const f32,
                get_cuda_mut_ptr(&topk_idxs)? as *mut i32,
                compressed_len as i32,
                topk as i32,
                offset as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("indexer_topk_decode CUDA error: {}", ret);
        }
        Ok(topk_idxs)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (scores, compressed_len, topk, offset);
        candle_core::bail!("indexer_topk_decode requires cuda feature")
    }
}

/// Generate window-based top-k indices for prefill.
///
/// Returns: [seq_len, topk] I32 — sliding window indices into KV cache
pub fn window_topk_indices(
    seq_len: usize,
    window_size: usize,
    topk: usize,
    device: &Device,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let out = Tensor::zeros((seq_len, topk), DType::U32, device)?;
        let stream = get_cuda_stream(device)?;
        let ret = unsafe {
            kernels::ffi::ds_window_topk_indices(
                get_cuda_mut_ptr(&out)? as *mut i32,
                seq_len as i32,
                window_size as i32,
                topk as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("window_topk_indices CUDA error: {}", ret);
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (seq_len, window_size, topk, device);
        candle_core::bail!("window_topk_indices requires cuda feature")
    }
}

/// Generate the fixed-width ring-buffer indices for one decode token entirely
/// on the device, including `-1` padding before the window fills.
pub fn window_topk_indices_decode(
    start_pos: usize,
    window_size: usize,
    device: &Device,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let out = Tensor::zeros((1, window_size), DType::U32, device)?;
        let stream = get_cuda_stream(device)?;
        let ret = unsafe {
            kernels::ffi::ds_window_topk_indices_decode(
                get_cuda_mut_ptr(&out)? as *mut i32,
                start_pos as i32,
                window_size as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("window_topk_indices_decode CUDA error: {}", ret);
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (start_pos, window_size, device);
        candle_core::bail!("window_topk_indices_decode requires cuda feature")
    }
}

/// Generate compressed block top-k indices for prefill.
///
/// Returns: [seq_len, compressed] I32
pub fn compress_topk_indices(
    seq_len: usize,
    compressed: usize,
    ratio: usize,
    offset: usize,
    device: &Device,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let out = Tensor::zeros((seq_len, compressed), DType::U32, device)?;
        let stream = get_cuda_stream(device)?;
        let ret = unsafe {
            kernels::ffi::ds_compress_topk_indices(
                get_cuda_mut_ptr(&out)? as *mut i32,
                seq_len as i32,
                compressed as i32,
                ratio as i32,
                offset as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("compress_topk_indices CUDA error: {}", ret);
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (seq_len, compressed, ratio, offset, device);
        candle_core::bail!("compress_topk_indices requires cuda feature")
    }
}

/// Generate contiguous compressed-cache indices for one decode token on GPU.
pub fn compress_topk_indices_decode(
    compressed: usize,
    offset: usize,
    device: &Device,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let out = Tensor::zeros((1, compressed), DType::U32, device)?;
        let stream = get_cuda_stream(device)?;
        let ret = unsafe {
            kernels::ffi::ds_compress_topk_indices_decode(
                get_cuda_mut_ptr(&out)? as *mut i32,
                compressed as i32,
                offset as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("compress_topk_indices_decode CUDA error: {}", ret);
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (compressed, offset, device);
        candle_core::bail!("compress_topk_indices_decode requires cuda feature")
    }
}

/// Concatenate two top-k index arrays along the topk dimension.
///
/// a: [seq_len, a_topk] I32, b: [seq_len, b_topk] I32
/// Returns: [seq_len, a_topk + b_topk] I32
pub fn concat_topk_indices(
    a: &Tensor,
    b: &Tensor,
    seq_len: usize,
    a_topk: usize,
    b_topk: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let out = Tensor::zeros((seq_len, a_topk + b_topk), DType::U32, a.device())?;
        let stream = get_cuda_stream(a.device())?;
        let ret = unsafe {
            kernels::ffi::ds_concat_topk_indices(
                get_cuda_ptr(a)? as *const i32,
                get_cuda_ptr(b)? as *const i32,
                get_cuda_mut_ptr(&out)? as *mut i32,
                seq_len as i32,
                a_topk as i32,
                b_topk as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("concat_topk_indices CUDA error: {}", ret);
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, b, seq_len, a_topk, b_topk);
        candle_core::bail!("concat_topk_indices requires cuda feature")
    }
}

// ============================================================================
// Attention utility kernels
// ============================================================================

/// Sparse indexed attention (works for both prefill and decode).
///
/// q: [seq_len, num_heads, head_dim] BF16
/// kv: [kv_len, head_dim] BF16 (shared across heads)
/// attn_sink: [num_heads] F32
/// topk_idxs: [seq_len, topk] I32 (-1 = invalid/skip)
/// Returns: [seq_len, num_heads, head_dim] BF16
///
/// No cudaMalloc — fully CUDA-graph compatible.
pub fn sparse_attention(
    q: &Tensor,
    kv: &Tensor,
    attn_sink: &Tensor,
    topk_idxs: &Tensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    kv_len: usize,
    topk: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let out = Tensor::zeros((seq_len, num_heads, head_dim), DType::BF16, q.device())?;
        let stream = get_cuda_stream(q.device())?;
        let ret = unsafe {
            kernels::ffi::ds_sparse_attn_dispatch(
                get_cuda_ptr(q)?,
                get_cuda_ptr(kv)?,
                get_cuda_ptr(attn_sink)? as *const core::ffi::c_void,
                get_cuda_ptr(topk_idxs)? as *const i32,
                get_cuda_mut_ptr(&out)?,
                seq_len as i32,
                num_heads as i32,
                head_dim as i32,
                kv_len as i32,
                topk as i32,
                softmax_scale,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("sparse_attention CUDA error: {}", ret);
        }
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            q,
            kv,
            attn_sink,
            topk_idxs,
            seq_len,
            num_heads,
            head_dim,
            kv_len,
            topk,
            softmax_scale,
        );
        candle_core::bail!("sparse_attention requires cuda feature")
    }
}

// ============================================================================
// Quantization utility kernels
// ============================================================================

/// Convert E8M0 scale bytes to F32 values.
///
/// E8M0 encodes a power-of-two scale: value = 2^(byte - 127).
/// This is a CPU/pure-Rust conversion suitable for small scale tensors
/// (e.g. model load time). The input must be a U8 tensor.
pub fn e8m0_scales_to_f32(scales: &Tensor) -> Result<Tensor> {
    let device = scales.device().clone();
    let shape = scales.shape().clone();
    let flat = scales.flatten_all()?.to_vec1::<u8>()?;
    let f32_vals: Vec<f32> = flat
        .iter()
        .map(|&byte| f32::from_bits((byte as u32) << 23))
        .collect();
    Tensor::from_vec(f32_vals, shape, &device)
}

// ============================================================================
// MoE hash-gate routing
// ============================================================================

/// Fused DeepSeek-V4 hash-gate routing (openinfer `deepseek_hash_gate_cuda`).
///
/// For each `(token, route)`:
/// 1. `expert = tid2eid[token_id, route]` (I64 table)
/// 2. `score = sqrt(softplus(dot(x[token], gate_weight[expert])))`
/// 3. L1-normalize scores across routes and multiply by `route_scale`
///
/// Inputs:
/// - `x`: BF16 `[seq, hidden]`
/// - `gate_weight`: BF16 `[n_experts, hidden]`
/// - `tid2eid`: I64 `[vocab, topk]`
/// - `token_ids`: U32 `[seq]`
///
/// Returns `(route_weights F32 [seq, topk], route_indices U32 [seq, topk])`.
pub fn hash_gate_route(
    x: &Tensor,
    gate_weight: &Tensor,
    tid2eid: &Tensor,
    token_ids: &Tensor,
    n_experts: usize,
    topk: usize,
    route_scale: f32,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        if x.dtype() != DType::BF16 {
            candle_core::bail!("hash_gate_route: x must be BF16, got {:?}", x.dtype());
        }
        if gate_weight.dtype() != DType::BF16 {
            candle_core::bail!(
                "hash_gate_route: gate_weight must be BF16, got {:?}",
                gate_weight.dtype()
            );
        }
        if tid2eid.dtype() != DType::I64 {
            candle_core::bail!(
                "hash_gate_route: tid2eid must be I64, got {:?}",
                tid2eid.dtype()
            );
        }
        let token_ids = if token_ids.dtype() != DType::U32 {
            token_ids.to_dtype(DType::U32)?
        } else {
            token_ids.clone()
        };
        let token_ids = token_ids.flatten_all()?.contiguous()?;
        let x = x.contiguous()?;
        let gate_weight = gate_weight.contiguous()?;
        let tid2eid = tid2eid.contiguous()?;

        let (seq_len, hidden_dim) = x.dims2()?;
        if token_ids.dim(0)? != seq_len {
            candle_core::bail!(
                "hash_gate_route: token_ids len {} != seq_len {}",
                token_ids.dim(0)?,
                seq_len
            );
        }
        let (gw_e, gw_h) = gate_weight.dims2()?;
        if gw_e != n_experts || gw_h != hidden_dim {
            candle_core::bail!(
                "hash_gate_route: gate_weight {:?} expected [{n_experts}, {hidden_dim}]",
                gate_weight.dims()
            );
        }
        if tid2eid.dim(1)? != topk {
            candle_core::bail!(
                "hash_gate_route: tid2eid topk {} != {topk}",
                tid2eid.dim(1)?
            );
        }
        let vocab_size = tid2eid.dim(0)?;

        let route_weights = Tensor::zeros((seq_len, topk), DType::F32, x.device())?;
        let route_indices = Tensor::zeros((seq_len, topk), DType::U32, x.device())?;
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_v4_hash_gate(
                get_cuda_ptr(&x)?,
                get_cuda_ptr(&gate_weight)?,
                get_cuda_ptr(&tid2eid)?,
                get_cuda_ptr(&token_ids)?,
                get_cuda_mut_ptr(&route_weights)?,
                get_cuda_mut_ptr(&route_indices)?,
                seq_len as i32,
                hidden_dim as i32,
                n_experts as i32,
                topk as i32,
                vocab_size as i32,
                route_scale,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("ds_v4_hash_gate CUDA error: {}", ret);
        }
        Ok((route_weights, route_indices))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            x,
            gate_weight,
            tid2eid,
            token_ids,
            n_experts,
            topk,
            route_scale,
        );
        candle_core::bail!("hash_gate_route requires cuda feature")
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn decode_indices_and_offset_copy_stay_on_device() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let dst = Tensor::zeros(10, DType::U32, &device)?;
        let src = Tensor::from_vec(vec![11u32, 12, 13], 3, &device)?;
        copy_contiguous_into(&dst, &src, 4)?;
        assert_eq!(dst.to_vec1::<u32>()?, vec![0, 0, 0, 0, 11, 12, 13, 0, 0, 0]);

        let early = window_topk_indices_decode(2, 5, &device)?.flatten_all()?;
        assert_eq!(early.to_vec1::<u32>()?, vec![0, 1, 2, u32::MAX, u32::MAX]);

        let wrapped = window_topk_indices_decode(6, 5, &device)?.flatten_all()?;
        assert_eq!(wrapped.to_vec1::<u32>()?, vec![2, 3, 4, 0, 1]);

        let compressed = compress_topk_indices_decode(4, 5, &device)?.flatten_all()?;
        assert_eq!(compressed.to_vec1::<u32>()?, vec![5, 6, 7, 8]);

        let scores = Tensor::from_vec(vec![1.0f32, -2.0, 4.0, 4.0, 0.0, 3.0, 4.0], 7, &device)?;
        let selected = indexer_topk_decode(&scores, 7, 5, 128)?;
        // Descending score with the serial reference's first-index tie break.
        assert_eq!(selected.to_vec1::<u32>()?, vec![130, 131, 134, 133, 128]);

        let long_scores: Vec<f32> = (0..129).map(|v| v as f32).collect();
        let long_scores = Tensor::from_vec(long_scores, 129, &device)?;
        let selected = indexer_topk_decode(&long_scores, 129, 100, 7)?;
        let expected: Vec<u32> = (29..129).rev().map(|v| (v + 7) as u32).collect();
        assert_eq!(selected.to_vec1::<u32>()?, expected);
        Ok(())
    }

    #[test]
    fn sparse_attention_skips_padding_and_keeps_sink_in_denominator() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let head_dim = 512;
        let q = Tensor::zeros((1, 1, head_dim), DType::BF16, &device)?;
        let mut kv_values = vec![3.0f32; head_dim];
        kv_values.extend(std::iter::repeat_n(6.0f32, head_dim));
        let kv = Tensor::from_vec(kv_values, (2, head_dim), &device)?.to_dtype(DType::BF16)?;
        let sink = Tensor::zeros(1, DType::F32, &device)?;
        let indices = Tensor::from_vec(vec![0u32, u32::MAX, 1], (1, 3), &device)?;

        // q.k = 0 for both valid rows and sink = 0, so the output is
        // (3 + 6) / (2 valid rows + 1 sink) = 3. Padding contributes nothing.
        let out = sparse_attention(&q, &kv, &sink, &indices, 1, 1, head_dim, 2, 3, 1.0)?;
        for value in out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()? {
            assert_eq!(value, 3.0);
        }
        Ok(())
    }

    #[test]
    fn hc_post_f32_branch_restores_reference_bf16_boundary() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let hc = 4;
        let dim = 4;
        let x = Tensor::from_vec(vec![1.003f32, -2.007, 0.499, 7.031], (1, dim), &device)?;
        let residual = Tensor::zeros((1, hc, dim), DType::BF16, &device)?;
        let post = Tensor::ones((1, hc), DType::F32, &device)?;
        let comb = Tensor::zeros((1, hc, hc), DType::F32, &device)?;

        let out = hc_post_f32_branch(&x, &residual, &post, &comb, hc, dim)?;
        let expected = x
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?[0]
            .clone();
        let actual = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for branch in actual.chunks_exact(dim) {
            assert_eq!(branch, expected.as_slice());
        }
        Ok(())
    }
}
