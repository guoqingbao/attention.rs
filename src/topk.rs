use candle_core as candle;
#[allow(unused_imports)]
use candle_core::backend::BackendStorage;
#[allow(unused_imports)]
use candle_core::{DType, Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;

#[cfg(feature = "cuda")]
pub fn topk_softmax(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::WrapErr;
    let (num_tokens, num_experts) = logits.dims2()?;
    let dev = logits.device().as_cuda_device()?;
    assert!(
        logits.dtype() == DType::F32,
        "Softmax topk only accept f32 inputs!"
    );

    let (logits, _) = logits.storage_and_layout();
    let logits = match &*logits {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("k_scales must be a cuda tensor"),
    };

    let token_expert_indices = unsafe { dev.alloc::<u32>(num_tokens * topk) }.w()?;
    let topk_weights = unsafe { dev.alloc::<f32>(num_tokens * topk) }.w()?;
    let topk_indices = unsafe { dev.alloc::<u32>(num_tokens * topk) }.w()?;

    let stream = *dev.cu_stream() as i64;

    unsafe {
        ffi::topk_softmax(
            *logits.device_ptr() as *const f32,
            *token_expert_indices.device_ptr() as *mut i32,
            *topk_weights.device_ptr() as *mut f32,
            *topk_indices.device_ptr() as *mut u32,
            num_experts as i32,
            num_tokens as i32,
            topk as i32,
            stream,
        )
    }

    let topk_weights = candle::CudaStorage::wrap_cuda_slice(topk_weights, dev.clone());
    let topk_weights =
        Tensor::from_storage(candle::Storage::Cuda(topk_weights), (num_tokens, topk))?;

    let topk_indices = candle::CudaStorage::wrap_cuda_slice(topk_indices, dev.clone());
    let topk_indices =
        Tensor::from_storage(candle::Storage::Cuda(topk_indices), (num_tokens, topk))?;

    Ok((topk_weights, topk_indices))
}

/// Fused sigmoid + bias + topk selection.
/// Takes raw router logits and optional bias. Returns (topk_weights, topk_indices).
/// topk_weights are original sigmoid scores (before bias), topk_indices selected from biased scores.
#[cfg(feature = "cuda")]
pub fn fused_sigmoid_topk(
    logits: &Tensor,
    bias: Option<&Tensor>,
    topk: usize,
) -> Result<(Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::WrapErr;
    let (num_tokens, _num_experts) = logits.dims2()?;
    let dev = logits.device().as_cuda_device()?;

    let logits_f32 = if logits.dtype() != DType::F32 {
        logits.to_dtype(DType::F32)?
    } else {
        logits.clone()
    };
    let logits_contig = if logits_f32.is_contiguous() {
        logits_f32
    } else {
        logits_f32.contiguous()?
    };

    let (storage, _) = logits_contig.storage_and_layout();
    let logits_cuda = match &*storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("logits must be a cuda tensor"),
    };

    let bias_ptr = if let Some(b) = bias {
        let b_f32 = if b.dtype() != DType::F32 {
            b.to_dtype(DType::F32)?
        } else {
            b.clone()
        };
        let b_contig = if b_f32.is_contiguous() {
            b_f32
        } else {
            b_f32.contiguous()?
        };
        let (bs, _) = b_contig.storage_and_layout();
        match &*bs {
            candle::Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr() as *const f32,
            _ => candle::bail!("bias must be a cuda tensor"),
        }
    } else {
        std::ptr::null()
    };

    let topk_weights = unsafe { dev.alloc::<f32>(num_tokens * topk) }.w()?;
    let topk_indices = unsafe { dev.alloc::<u32>(num_tokens * topk) }.w()?;

    let stream = *dev.cu_stream() as i64;

    unsafe {
        ffi::fused_sigmoid_topk(
            *logits_cuda.device_ptr() as *const f32,
            bias_ptr,
            *topk_weights.device_ptr() as *mut f32,
            *topk_indices.device_ptr() as *mut u32,
            _num_experts as i32,
            num_tokens as i32,
            topk as i32,
            stream,
        )
    }

    let topk_weights = candle::CudaStorage::wrap_cuda_slice(topk_weights, dev.clone());
    let topk_weights =
        Tensor::from_storage(candle::Storage::Cuda(topk_weights), (num_tokens, topk))?;

    let topk_indices = candle::CudaStorage::wrap_cuda_slice(topk_indices, dev.clone());
    let topk_indices =
        Tensor::from_storage(candle::Storage::Cuda(topk_indices), (num_tokens, topk))?;

    Ok((topk_weights, topk_indices))
}

/// Fast top-k selection from pre-computed scores (no softmax applied).
/// Returns (topk_weights, topk_indices) with shape [num_tokens, topk].
#[cfg(feature = "cuda")]
pub fn topk_select(scores: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::WrapErr;
    let (num_tokens, _num_experts) = scores.dims2()?;
    let dev = scores.device().as_cuda_device()?;
    assert!(
        scores.dtype() == DType::F32,
        "topk_select only accepts f32 inputs!"
    );

    let scores_contig = if scores.is_contiguous() {
        scores.clone()
    } else {
        scores.contiguous()?
    };

    let (storage, _) = scores_contig.storage_and_layout();
    let scores_cuda = match &*storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("scores must be a cuda tensor"),
    };

    let topk_weights = unsafe { dev.alloc::<f32>(num_tokens * topk) }.w()?;
    let topk_indices = unsafe { dev.alloc::<u32>(num_tokens * topk) }.w()?;

    let stream = *dev.cu_stream() as i64;

    unsafe {
        ffi::topk_select(
            *scores_cuda.device_ptr() as *const f32,
            *topk_weights.device_ptr() as *mut f32,
            *topk_indices.device_ptr() as *mut u32,
            _num_experts as i32,
            num_tokens as i32,
            topk as i32,
            stream,
        )
    }

    let topk_weights = candle::CudaStorage::wrap_cuda_slice(topk_weights, dev.clone());
    let topk_weights =
        Tensor::from_storage(candle::Storage::Cuda(topk_weights), (num_tokens, topk))?;

    let topk_indices = candle::CudaStorage::wrap_cuda_slice(topk_indices, dev.clone());
    let topk_indices =
        Tensor::from_storage(candle::Storage::Cuda(topk_indices), (num_tokens, topk))?;

    Ok((topk_weights, topk_indices))
}

#[cfg(not(feature = "cuda"))]
pub fn topk_softmax(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let routing_weights = candle_nn::ops::softmax_last_dim(&logits)?;
    let indices = routing_weights
        .arg_sort_last_dim(false)?
        .narrow(candle::D::Minus1, 0, topk)?
        .contiguous()?;

    let scores = routing_weights.gather(&indices, candle::D::Minus1)?;
    Ok((scores, indices))
}
