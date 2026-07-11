//! MiniMax M3 sparse-attention CUDA entry points.

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::Result;
#[cfg(feature = "cuda")]
use candle_core::{DType, Storage, Tensor};

#[cfg(feature = "cuda")]
fn cuda_ptr(t: &Tensor) -> Result<*const core::ffi::c_void> {
    let (storage, layout) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (Storage::Cuda(s), DType::F16) => Ok(*s
            .as_cuda_slice::<half::f16>()?
            .slice(layout.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
        (Storage::Cuda(s), DType::BF16) => Ok(*s
            .as_cuda_slice::<half::bf16>()?
            .slice(layout.start_offset()..)
            .device_ptr()
            as *const core::ffi::c_void),
        (Storage::Cuda(s), DType::F32) => Ok(*s
            .as_cuda_slice::<f32>()?
            .slice(layout.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
        _ => candle_core::bail!("MiniMax M3 tensors must be CUDA F16/BF16/F32"),
    }
}

#[cfg(feature = "cuda")]
fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        _ => candle_core::bail!("MiniMax M3 CUDA kernels do not support {dtype:?}"),
    }
}

#[cfg(feature = "cuda")]
fn cuda_u32_ptr(t: &Tensor) -> Result<*const core::ffi::c_void> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Cuda(s) if t.dtype() == DType::U32 => Ok(*s
            .as_cuda_slice::<u32>()?
            .slice(layout.start_offset()..)
            .device_ptr()
            as *const core::ffi::c_void),
        _ => candle_core::bail!("MiniMax M3 sequence offsets must be CUDA U32"),
    }
}

#[cfg(feature = "cuda")]
pub fn minimax_m3_indexer_prefill(
    q: &Tensor,
    k: &Tensor,
    cu_seqlens: &Tensor,
    topk_blocks: usize,
    block_size: usize,
    max_seq_len: usize,
    scale: f32,
) -> Result<Tensor> {
    let (tokens, heads, dim) = q.dims3()?;
    if k.dims2()? != (tokens, dim) || cu_seqlens.dtype() != DType::U32 {
        candle_core::bail!("MiniMax M3 indexer shape or offset mismatch");
    }
    let batch_size = cu_seqlens.dim(0)?.checked_sub(1).ok_or_else(|| {
        candle_core::Error::Msg("MiniMax M3 requires at least one sequence".into())
    })?;
    if batch_size == 0 || max_seq_len == 0 {
        candle_core::bail!("MiniMax M3 indexer received empty metadata");
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let cu_seqlens = cu_seqlens.contiguous()?;
    let output = Tensor::zeros((tokens, heads, topk_blocks), DType::U32, q.device())?;
    let dev = q.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;
    let out_ptr = {
        let (storage, layout) = output.storage_and_layout();
        match &*storage {
            Storage::Cuda(s) => Ok(*s
                .as_cuda_slice::<u32>()?
                .slice(layout.start_offset()..)
                .device_ptr() as *mut core::ffi::c_int),
            _ => candle_core::bail!("MiniMax M3 index output must be CUDA"),
        }
    }?;
    let ret = unsafe {
        crate::kernels::ffi::minimax_m3_indexer_prefill(
            cuda_ptr(&q)?,
            cuda_ptr(&k)?,
            out_ptr,
            cuda_u32_ptr(&cu_seqlens)?,
            tokens as i32,
            batch_size as i32,
            heads as i32,
            dim as i32,
            topk_blocks as i32,
            block_size as i32,
            max_seq_len as i32,
            scale,
            dtype_code(q.dtype())?,
            stream,
        )
    };
    if ret != 0 {
        candle_core::bail!("MiniMax M3 indexer CUDA error: {ret}");
    }
    Ok(output)
}

#[cfg(feature = "cuda")]
pub fn minimax_m3_sparse_attention_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    topk: &Tensor,
    cu_seqlens: &Tensor,
    max_seq_len: usize,
    scale: f32,
    block_size: usize,
) -> Result<Tensor> {
    let (tokens, num_heads, dim) = q.dims3()?;
    let (k_tokens, num_kv_heads, k_dim) = k.dims3()?;
    if (k_tokens, k_dim) != (tokens, dim)
        || v.dims3()? != (tokens, num_kv_heads, dim)
        || topk.dims3()?.0 != tokens
        || topk.dims3()?.1 != num_kv_heads
        || topk.dtype() != DType::U32
        || cu_seqlens.dtype() != DType::U32
    {
        candle_core::bail!("MiniMax M3 sparse GQA shape mismatch");
    }
    let batch_size = cu_seqlens.dim(0)?.checked_sub(1).ok_or_else(|| {
        candle_core::Error::Msg("MiniMax M3 requires at least one sequence".into())
    })?;
    if batch_size == 0 || max_seq_len == 0 {
        candle_core::bail!("MiniMax M3 sparse GQA received empty metadata");
    }
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let topk = topk.contiguous()?;
    let cu_seqlens = cu_seqlens.contiguous()?;
    let output = Tensor::zeros(q.shape(), q.dtype(), q.device())?;
    let dev = q.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;
    let out_ptr = {
        let (storage, layout) = output.storage_and_layout();
        match (&*storage, output.dtype()) {
            (Storage::Cuda(s), DType::F16) => Ok(*s
                .as_cuda_slice::<half::f16>()?
                .slice(layout.start_offset()..)
                .device_ptr()
                as *mut core::ffi::c_void),
            (Storage::Cuda(s), DType::BF16) => Ok(*s
                .as_cuda_slice::<half::bf16>()?
                .slice(layout.start_offset()..)
                .device_ptr()
                as *mut core::ffi::c_void),
            (Storage::Cuda(s), DType::F32) => Ok(*s
                .as_cuda_slice::<f32>()?
                .slice(layout.start_offset()..)
                .device_ptr()
                as *mut core::ffi::c_void),
            _ => candle_core::bail!("MiniMax M3 output must be CUDA F16/BF16/F32"),
        }
    }?;
    let topk_ptr = {
        let (storage, layout) = topk.storage_and_layout();
        match &*storage {
            Storage::Cuda(s) => Ok(*s
                .as_cuda_slice::<u32>()?
                .slice(layout.start_offset()..)
                .device_ptr() as *const core::ffi::c_int),
            _ => candle_core::bail!("MiniMax M3 top-k must be CUDA"),
        }
    }?;
    let ret = unsafe {
        crate::kernels::ffi::minimax_m3_sparse_attention_prefill(
            out_ptr,
            cuda_ptr(&q)?,
            cuda_ptr(&k)?,
            cuda_ptr(&v)?,
            topk_ptr,
            cuda_u32_ptr(&cu_seqlens)?,
            tokens as i32,
            batch_size as i32,
            num_heads as i32,
            num_kv_heads as i32,
            dim as i32,
            topk.dim(2)? as i32,
            block_size as i32,
            max_seq_len as i32,
            scale,
            dtype_code(q.dtype())?,
            stream,
        )
    };
    if ret != 0 {
        candle_core::bail!("MiniMax M3 sparse GQA CUDA error: {ret}");
    }
    Ok(output)
}

#[cfg(not(feature = "cuda"))]
pub fn minimax_m3_indexer_prefill(
    _q: &candle_core::Tensor,
    _k: &candle_core::Tensor,
    _cu_seqlens: &candle_core::Tensor,
    _topk_blocks: usize,
    _block_size: usize,
    _max_seq_len: usize,
    _scale: f32,
) -> Result<candle_core::Tensor> {
    candle_core::bail!("MiniMax M3 indexer requires CUDA")
}

#[cfg(not(feature = "cuda"))]
pub fn minimax_m3_sparse_attention_prefill(
    _q: &candle_core::Tensor,
    _k: &candle_core::Tensor,
    _v: &candle_core::Tensor,
    _topk: &candle_core::Tensor,
    _cu_seqlens: &candle_core::Tensor,
    _max_seq_len: usize,
    _scale: f32,
    _block_size: usize,
) -> Result<candle_core::Tensor> {
    candle_core::bail!("MiniMax M3 sparse attention requires CUDA")
}
