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
    let (num_tokens, _) = logits.dims2()?;
    fn cuda_fwd(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
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

        // not used
        // let token_expert_indices =
        //     candle::CudaStorage::wrap_cuda_slice(token_expert_indices, dev.clone());
        // let token_expert_indices = Tensor::from_storage(
        //     candle::Storage::Cuda(token_expert_indices),
        //     (num_tokens, topk),
        // )?;

        let topk_weights = candle::CudaStorage::wrap_cuda_slice(topk_weights, dev.clone());
        let topk_weights = crate::compat::tensor_from_storage(
            candle::Storage::Cuda(topk_weights),
            (num_tokens, topk),
        )?;

        let topk_indices = candle::CudaStorage::wrap_cuda_slice(topk_indices, dev.clone());
        let topk_indices = crate::compat::tensor_from_storage(
            candle::Storage::Cuda(topk_indices),
            (num_tokens, topk),
        )?;

        Ok((topk_weights, topk_indices))
    }

    if num_tokens > 64 {
        // fused topk faster for longer context
        cuda_fwd(logits, topk)
    } else {
        // unfused topk suitable for decoding
        let routing_weights = crate::compat::softmax_last_dim(&logits)?;
        let indices = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(candle::D::Minus1, 0, topk)?
            .contiguous()?;

        let scores = routing_weights.gather(&indices, candle::D::Minus1)?;
        Ok((scores, indices))
    }
}

#[cfg(not(feature = "cuda"))]
pub fn topk_softmax(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let routing_weights = crate::compat::softmax_last_dim(&logits)?;
    let indices = routing_weights
        .arg_sort_last_dim(false)?
        .narrow(candle::D::Minus1, 0, topk)?
        .contiguous()?;

    let scores = routing_weights.gather(&indices, candle::D::Minus1)?;
    Ok((scores, indices))
}
