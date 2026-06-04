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

#[cfg(feature = "metal")]
pub fn topk_softmax(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let routing_weights = candle_nn::ops::softmax_last_dim(&logits)?;
    let indices = routing_weights
        .arg_sort_last_dim(false)?
        .narrow(candle::D::Minus1, 0, topk)?
        .contiguous()?;

    let scores = routing_weights.gather(&indices, candle::D::Minus1)?;
    Ok((scores, indices))
}

#[cfg(feature = "gcu")]
fn gcu_topk_func<
    T: candle::gcu_backend::GcuDType + candle::gcu_backend::DeviceCopy + candle::WithDType,
>(
    input: &Tensor,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
    use candle::gcu_backend::ubridge::ffi::{topk_bf16, topk_f16, topk_f32};
    use candle::gcu_backend::WrapErr;
    use candle::Storage;
    use half::{bf16, f16};
    let dev = input.device().as_gcu_device()?;
    let (value, input_l) = input.storage_and_layout();
    let shape = input_l.shape();
    let el_count = shape.elem_count();
    let stream = dev.stream_inner().unwrap();
    let value = match &*value {
        Storage::Gcu(k) => k,
        _ => candle::bail!("tensor must be a gcu tensor"),
    };

    let rank = input_l.dims().len();
    assert!(rank <= 3);
    let value = value.as_gcu_slice::<T>()?;
    let value = value.slice(input_l.start_offset()..);
    let (chunks, dims) = if rank == 3 {
        (shape.dims()[0] * shape.dims()[1], shape.dims().to_vec())
    } else if rank == 2 {
        (
            shape.dims()[0],
            [1usize, shape.dims()[0], shape.dims()[1]].to_vec(),
        )
    } else {
        (1usize, [1usize, 1usize, shape.dims()[0]].to_vec())
    };

    let indices = dev.alloc::<u32>(chunks * k).w()?;
    let out = dev.alloc::<T>(chunks * k).w()?;
    match input.dtype() {
        DType::F16 => unsafe {
            topk_f16(
                value.device_ptr() as *mut f16,
                out.device_ptr() as *mut f16,
                indices.device_ptr() as *mut u32,
                dims[0] as i32,
                dims[1] as i32,
                dims[2] as i32,
                k as i32,
                stream as *mut core::ffi::c_void,
            );
        },
        DType::BF16 => unsafe {
            topk_bf16(
                value.device_ptr() as *mut bf16,
                out.device_ptr() as *mut bf16,
                indices.device_ptr() as *mut u32,
                dims[0] as i32,
                dims[1] as i32,
                dims[2] as i32,
                k as i32,
                stream as *mut core::ffi::c_void,
            );
        },
        DType::F32 => unsafe {
            topk_f32(
                value.device_ptr() as *mut f32,
                out.device_ptr() as *mut f32,
                indices.device_ptr() as *mut u32,
                dims[0] as i32,
                dims[1] as i32,
                dims[2] as i32,
                k as i32,
                stream as *mut core::ffi::c_void,
            );
        },
        _ => {
            panic!("not supported data type!")
        }
    }
    let s_out = candle::GcuStorage::wrap_gcu_slice(out, dev.clone());
    let s_indices = candle::GcuStorage::wrap_gcu_slice(indices, dev.clone());
    let mut out_dims = shape.dims().to_vec();
    let last_dim = out_dims.len() - 1;
    out_dims[last_dim] = k;
    Ok((
        Tensor::from_storage(candle::Storage::Gcu(s_out), out_dims.clone())?,
        Tensor::from_storage(candle::Storage::Gcu(s_indices), out_dims.clone())?,
    ))
}

#[cfg(feature = "gcu")]
pub fn topk_softmax(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    topk_softmax_impl(logits, topk, false)
}

#[cfg(feature = "gcu")]
pub fn topk_softmax_renorm(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    topk_softmax_impl(logits, topk, true)
}

#[cfg(feature = "gcu")]
fn topk_softmax_impl(
    logits: &Tensor,
    topk: usize,
    norm_topk_prob: bool,
) -> Result<(Tensor, Tensor)> {
    use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
    use candle::gcu_backend::WrapErr;
    use candle::Storage;

    let logits = logits.contiguous()?;
    let (num_tokens, num_experts) = logits.dims2()?;
    let dev = logits.device().as_gcu_device()?;
    let stream = dev.stream_inner().unwrap();

    let (logits_s, logits_l) = logits.storage_and_layout();
    let logits_ptr = match &*logits_s {
        Storage::Gcu(c) => {
            let s = c.as_gcu_slice::<f32>()?;
            let s = s.slice(logits_l.start_offset()..);
            s.device_ptr()
        }
        _ => candle::bail!("logits must be a f32 gcu tensor"),
    };

    #[cfg(feature = "aten")]
    {
        use candle::gcu_backend::ubridge::ffi::topsaten_topk_softmax_f32;

        let output = dev.alloc::<f32>(num_tokens * topk).w()?;
        let index = dev.alloc::<i32>(num_tokens * topk).w()?;
        let tei = dev.alloc::<i32>(num_tokens * topk).w()?;

        let ret = unsafe {
            topsaten_topk_softmax_f32(
                logits_ptr as *mut f32,
                output.device_ptr() as *mut f32,
                index.device_ptr() as *mut i32,
                tei.device_ptr() as *mut i32,
                num_tokens as i32,
                num_experts as i32,
                topk as i32,
                if norm_topk_prob { 1 } else { 0 },
                stream as *mut core::ffi::c_void,
            )
        };
        if ret != 0 {
            candle::bail!("topsaten_topk_softmax_f32 failed with code {ret}");
        }

        let s_out = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
        let topk_weights = Tensor::from_storage(candle::Storage::Gcu(s_out), (num_tokens, topk))?;

        let s_idx = candle::GcuStorage::wrap_gcu_slice(index, dev.clone());
        let topk_ids = Tensor::from_storage(candle::Storage::Gcu(s_idx), (num_tokens, topk))?;

        return Ok((topk_weights, topk_ids));
    }

    #[cfg(not(feature = "aten"))]
    {
        use candle::gcu_backend::ubridge::ffi::topk_softmax_f32;

        let output = dev.alloc::<f32>(num_tokens * topk).w()?;
        let index = dev.alloc::<u32>(num_tokens * topk).w()?;

        unsafe {
            topk_softmax_f32(
                logits_ptr as *mut f32,
                output.device_ptr() as *mut f32,
                index.device_ptr() as *mut i32,
                num_tokens as i32,
                num_experts as i32,
                topk as i32,
                0,
                stream as *mut core::ffi::c_void,
            );
        }

        let s_out = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
        let topk_weights = Tensor::from_storage(candle::Storage::Gcu(s_out), (num_tokens, topk))?;

        let s_idx = candle::GcuStorage::wrap_gcu_slice(index, dev.clone());
        let topk_ids = Tensor::from_storage(candle::Storage::Gcu(s_idx), (num_tokens, topk))?;

        let topk_weights = if norm_topk_prob {
            topk_weights.broadcast_div(&topk_weights.sum_keepdim(candle::D::Minus1)?)?
        } else {
            topk_weights
        };

        return Ok((topk_weights, topk_ids));
    }
}
