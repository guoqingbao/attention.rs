//! DeepSeek V4 CUDA kernels: Hyper-Connection, per-head RMSNorm, Compressor,
//! Indexer, sparse attention, and FP8 quantization helpers.

#[cfg(feature = "cuda")]
use crate::kernels;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::{DType, Storage};
use candle_core::{Device, Result, Tensor};

#[cfg(feature = "cuda")]
fn get_cuda_ptr(t: &Tensor) -> Result<*const core::ffi::c_void> {
    let (s, l) = t.storage_and_layout();
    match (&*s, t.dtype()) {
        (Storage::Cuda(c), DType::U8) => Ok(*c
            .as_cuda_slice::<u8>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
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

/// Compute mixing coefficients: matmul in candle + RMS scaling via CUDA kernel.
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
        let hc_dim = hc * dim;
        // BF16→F32 cast + matmul via candle (uses cudarc memory pool, graph-compatible)
        let x_f32 = x.to_dtype(DType::F32)?;
        let mixes = x_f32.matmul(&hc_fn.t()?)?;
        // RMS scale via CUDA kernel (no allocation)
        let stream = get_cuda_stream(x.device())?;
        let ret = unsafe {
            kernels::ffi::ds_v4_hc_scale_mixes(
                get_cuda_ptr(x)?,
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
            candle_core::bail!("ds_v4_hc_scale_mixes CUDA error: {}", ret);
        }
        let _ = hc_dim;
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
        let seq_len = x.dim(0)?;
        let out = Tensor::zeros((seq_len, hc, dim), residual.dtype(), x.device())?;
        let stream = get_cuda_stream(x.device())?;
        unsafe {
            kernels::ffi::ds_v4_hc_post(
                get_cuda_ptr(x)?,
                get_cuda_ptr(residual)?,
                get_cuda_ptr(post)?,
                get_cuda_ptr(comb)?,
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
        let (_, _, _, _) = (residual, post, comb, dim);
        candle_core::bail!("hc_post requires cuda feature")
    }
}

/// Per-head RMSNorm: normalize each head independently.
///
/// x: [seq, num_heads, head_dim]
/// Returns: [seq, num_heads, head_dim]
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

/// Non-overlap compressor prefill: matmul in candle + fused epilogue via CUDA.
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
        // BF16→F32 GEMMs: scores = x @ wgate^T, values = x @ wkv^T
        let scores = x.matmul(&wgate.t()?)?.to_dtype(DType::F32)?;
        let values = x.matmul(&wkv.t()?)?.to_dtype(DType::F32)?;

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

/// Overlap compressor prefill (ratio=4): matmul in candle + fused epilogue via CUDA.
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
        // Overlap projections go to 2*head_dim
        let scores = x.matmul(&wgate.t()?)?.to_dtype(DType::F32)?;
        let values = x.matmul(&wkv.t()?)?.to_dtype(DType::F32)?;

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
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        let weighted = Tensor::zeros((1, head_dim), DType::F32, x.device())?;
        let out = Tensor::zeros((1, head_dim), DType::BF16, x.device())?;
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
                get_cuda_mut_ptr(&weighted)? as *mut f32,
                get_cuda_mut_ptr(&out)?,
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
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    {
        let weighted = Tensor::zeros((1, head_dim), DType::F32, x.device())?;
        let out = Tensor::zeros((1, head_dim), DType::BF16, x.device())?;
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
                get_cuda_mut_ptr(&weighted)? as *mut f32,
                get_cuda_mut_ptr(&out)?,
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
