use candle_core as candle;
#[allow(unused_imports)]
use candle_core::backend::BackendStorage;
#[allow(unused_imports)]
use candle_core::{DType, Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;
#[cfg(feature = "metal")]
use metal_kernels;

#[cfg(feature = "metal")]
#[derive(Clone)]
struct MetalTensorSlice {
    storage: candle_core::MetalStorage,
    offset_in_bytes: usize,
}

#[cfg(feature = "metal")]
fn get_metal_slice(tensor: &Tensor) -> Result<MetalTensorSlice> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        candle::Storage::Metal(storage) => Ok(MetalTensorSlice {
            storage: storage.clone(),
            offset_in_bytes: layout.start_offset() * tensor.dtype().size_in_bytes(),
        }),
        _ => candle::bail!("expected a Metal tensor"),
    }
}

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

    let logits_contig = if logits.is_contiguous() {
        logits.clone()
    } else {
        logits.contiguous()?
    };

    let (storage, _) = logits_contig.storage_and_layout();
    let logits_cuda = match &*storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("k_scales must be a cuda tensor"),
    };

    let token_expert_indices = unsafe { dev.alloc::<u32>(num_tokens * topk) }.w()?;
    let topk_weights = unsafe { dev.alloc::<f32>(num_tokens * topk) }.w()?;
    let topk_indices = unsafe { dev.alloc::<u32>(num_tokens * topk) }.w()?;

    let stream = *dev.cu_stream() as i64;

    unsafe {
        ffi::topk_softmax(
            *logits_cuda.device_ptr() as *const f32,
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

    let bias_contig = if let Some(b) = bias {
        let b_f32 = if b.dtype() != DType::F32 {
            b.to_dtype(DType::F32)?
        } else {
            b.clone()
        };
        Some(if b_f32.is_contiguous() {
            b_f32
        } else {
            b_f32.contiguous()?
        })
    } else {
        None
    };

    let bias_ptr = if let Some(b_contig) = bias_contig.as_ref() {
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
pub fn topk_select(scores: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let (num_tokens, num_experts) = scores.dims2()?;
    if scores.dtype() != DType::F32 {
        candle::bail!("topk_select only accepts f32 inputs");
    }
    if num_tokens == 0 || num_experts == 0 || topk == 0 || topk > num_experts || topk > 32 {
        candle::bail!(
            "Metal DFlash2 topk_select requires 1 <= topk <= min(num_experts, 32), got topk={} and num_experts={}",
            topk,
            num_experts
        );
    }

    let scores = scores.contiguous()?;
    let topk_weights = Tensor::zeros((num_tokens, topk), DType::F32, scores.device())?;
    let topk_indices = Tensor::zeros((num_tokens, topk), DType::U32, scores.device())?;
    let scores_m = get_metal_slice(&scores)?;
    let weights_m = get_metal_slice(&topk_weights)?;
    let indices_m = get_metal_slice(&topk_indices)?;
    let dev = scores_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("dflash-topk-select");
    metal_kernels::call_dflash_topk_select(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        (scores_m.storage.buffer(), scores_m.offset_in_bytes),
        weights_m.storage.buffer(),
        indices_m.storage.buffer(),
        num_tokens as u32,
        num_experts as u32,
        topk as u32,
    )
    .map_err(candle_core::Error::wrap)?;
    Ok((topk_weights, topk_indices))
}

/// Fused DFlash2 candidate-path selection.
///
/// Scores and walks the K-way candidate lattice on device. The selected path
/// is returned as a GPU tensor so the caller does not synchronize once per
/// speculative position.
#[cfg(feature = "cuda")]
pub fn dflash_select_candidates(
    hidden: &Tensor,
    unary_logits: &Tensor,
    candidate_ids: &Tensor,
    predecessor_codebook: &Tensor,
    successor_codebook: &Tensor,
    anchor_token: &Tensor,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::WrapErr;

    let (sequence_len, rank) = hidden.dims2()?;
    let (candidate_rows, topk) = unary_logits.dims2()?;
    if candidate_rows != sequence_len || candidate_ids.dims2()? != (sequence_len, topk) {
        candle::bail!("DFlash2 candidate selector input shape mismatch");
    }
    if predecessor_codebook.dims2()?.1 != rank || successor_codebook.dims2()?.1 != rank {
        candle::bail!("DFlash2 codebook rank does not match hidden projection");
    }
    if anchor_token.dims1()? != 1 {
        candle::bail!("DFlash2 anchor token must have shape [1]");
    }

    let dev = hidden.device().as_cuda_device()?;
    let hidden = if hidden.dtype() == DType::F32 {
        hidden.contiguous()?
    } else {
        hidden.to_dtype(DType::F32)?.contiguous()?
    };
    let unary_logits = if unary_logits.dtype() == DType::F32 {
        unary_logits.contiguous()?
    } else {
        unary_logits.to_dtype(DType::F32)?.contiguous()?
    };
    let candidate_ids = if candidate_ids.dtype() == DType::U32 {
        candidate_ids.contiguous()?
    } else {
        candidate_ids.to_dtype(DType::U32)?.contiguous()?
    };
    let predecessor_codebook = if predecessor_codebook.dtype() == DType::F32 {
        predecessor_codebook.contiguous()?
    } else {
        predecessor_codebook.to_dtype(DType::F32)?.contiguous()?
    };
    let successor_codebook = if successor_codebook.dtype() == DType::F32 {
        successor_codebook.contiguous()?
    } else {
        successor_codebook.to_dtype(DType::F32)?.contiguous()?
    };
    let anchor_token = if anchor_token.dtype() == DType::U32 {
        anchor_token.contiguous()?
    } else {
        anchor_token.to_dtype(DType::U32)?.contiguous()?
    };

    let f32_ptr = |tensor: &Tensor| -> Result<*const f32> {
        let (storage, _) = tensor.storage_and_layout();
        match &*storage {
            candle::Storage::Cuda(storage) => {
                Ok(*storage.as_cuda_slice::<f32>()?.device_ptr() as *const f32)
            }
            _ => candle::bail!("DFlash2 selector tensors must be CUDA tensors"),
        }
    };
    let u32_ptr = |tensor: &Tensor| -> Result<*const u32> {
        let (storage, _) = tensor.storage_and_layout();
        match &*storage {
            candle::Storage::Cuda(storage) => {
                Ok(*storage.as_cuda_slice::<u32>()?.device_ptr() as *const u32)
            }
            _ => candle::bail!("DFlash2 selector tensors must be CUDA tensors"),
        }
    };

    let selected_tokens = unsafe { dev.alloc::<u32>(sequence_len) }.w()?;
    let stream = *dev.cu_stream() as i64;
    unsafe {
        let status = ffi::dflash_select_candidates(
            f32_ptr(&hidden)? as *const std::ffi::c_void,
            f32_ptr(&unary_logits)?,
            u32_ptr(&candidate_ids)?,
            f32_ptr(&predecessor_codebook)?,
            f32_ptr(&successor_codebook)?,
            u32_ptr(&anchor_token)?,
            *selected_tokens.device_ptr() as *mut u32,
            sequence_len as i32,
            rank as i32,
            topk as i32,
            stream,
        );
        if status != 0 {
            candle::bail!("DFlash2 candidate selector kernel failed with CUDA error {status}");
        }
    }
    let selected_tokens = candle::CudaStorage::wrap_cuda_slice(selected_tokens, dev.clone());
    Tensor::from_storage(
        candle::Storage::Cuda(selected_tokens),
        candle::Shape::from(sequence_len),
    )
}

#[cfg(feature = "metal")]
pub fn dflash_select_candidates(
    hidden: &Tensor,
    unary_logits: &Tensor,
    candidate_ids: &Tensor,
    predecessor_codebook: &Tensor,
    successor_codebook: &Tensor,
    anchor_token: &Tensor,
) -> Result<Tensor> {
    let (sequence_len, rank) = hidden.dims2()?;
    let (candidate_rows, topk) = unary_logits.dims2()?;
    if candidate_rows != sequence_len || candidate_ids.dims2()? != (sequence_len, topk) {
        candle::bail!("DFlash2 candidate selector input shape mismatch");
    }
    if predecessor_codebook.dims2()?.1 != rank || successor_codebook.dims2()?.1 != rank {
        candle::bail!("DFlash2 codebook rank does not match hidden projection");
    }
    if anchor_token.dims1()? != 1 {
        candle::bail!("DFlash2 anchor token must have shape [1]");
    }
    if sequence_len == 0 || rank == 0 || topk == 0 || topk > 32 {
        candle::bail!("Metal DFlash2 candidate selector requires 1 <= topk <= 32");
    }

    let hidden = hidden.to_dtype(DType::F32)?.contiguous()?;
    let unary_logits = unary_logits.to_dtype(DType::F32)?.contiguous()?;
    let candidate_ids = candidate_ids.to_dtype(DType::U32)?.contiguous()?;
    let predecessor_codebook = predecessor_codebook.to_dtype(DType::F32)?.contiguous()?;
    let successor_codebook = successor_codebook.to_dtype(DType::F32)?.contiguous()?;
    let anchor_token = anchor_token.to_dtype(DType::U32)?.contiguous()?;
    let selected_tokens = Tensor::zeros(sequence_len, DType::U32, hidden.device())?;

    let hidden_m = get_metal_slice(&hidden)?;
    let unary_m = get_metal_slice(&unary_logits)?;
    let ids_m = get_metal_slice(&candidate_ids)?;
    let predecessor_m = get_metal_slice(&predecessor_codebook)?;
    let successor_m = get_metal_slice(&successor_codebook)?;
    let anchor_m = get_metal_slice(&anchor_token)?;
    let selected_m = get_metal_slice(&selected_tokens)?;
    let dev = hidden_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("dflash-select-candidates");
    metal_kernels::call_dflash_select_candidates(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        (hidden_m.storage.buffer(), hidden_m.offset_in_bytes),
        (unary_m.storage.buffer(), unary_m.offset_in_bytes),
        (ids_m.storage.buffer(), ids_m.offset_in_bytes),
        (
            predecessor_m.storage.buffer(),
            predecessor_m.offset_in_bytes,
        ),
        (successor_m.storage.buffer(), successor_m.offset_in_bytes),
        (anchor_m.storage.buffer(), anchor_m.offset_in_bytes),
        selected_m.storage.buffer(),
        sequence_len as u32,
        rank as u32,
        topk as u32,
    )
    .map_err(candle_core::Error::wrap)?;
    Ok(selected_tokens)
}

/// Fused BF16 grouped dynamic depthwise convolution used by DFlash2.
#[cfg(feature = "cuda")]
pub fn dflash_grouped_conv_bf16(
    hidden: &Tensor,
    delta: &Tensor,
    base_kernel: &Tensor,
    block_size: usize,
    side: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::WrapErr;

    let (sequence_len, hidden_size) = hidden.dims2()?;
    let (delta_len, taps, num_groups) = delta.dims3()?;
    let (base_sides, base_taps, base_hidden) = base_kernel.dims3()?;
    if hidden.dtype() != DType::BF16
        || delta.dtype() != DType::BF16
        || base_kernel.dtype() != DType::BF16
        || delta_len != sequence_len
        || base_sides != 2
        || base_taps != taps
        || base_hidden != hidden_size
        || side > 1
        || hidden_size % num_groups != 0
    {
        candle::bail!("invalid DFlash2 grouped convolution shapes or dtypes");
    }

    let dev = hidden.device().as_cuda_device()?;
    let hidden = hidden.contiguous()?;
    let delta = delta.contiguous()?;
    let base_kernel = base_kernel.contiguous()?;
    let ptr = |tensor: &Tensor| -> Result<*const std::ffi::c_void> {
        let (storage, _) = tensor.storage_and_layout();
        match &*storage {
            candle::Storage::Cuda(storage) => {
                Ok(*storage.as_cuda_slice::<half::bf16>()?.device_ptr() as *const std::ffi::c_void)
            }
            _ => candle::bail!("DFlash2 convolution tensors must be CUDA tensors"),
        }
    };

    let output = unsafe { dev.alloc::<half::bf16>(sequence_len * hidden_size) }.w()?;
    let stream = *dev.cu_stream() as i64;
    let status = unsafe {
        ffi::dflash_grouped_conv_bf16(
            ptr(&hidden)?,
            ptr(&delta)?,
            ptr(&base_kernel)?,
            *output.device_ptr() as *mut std::ffi::c_void,
            sequence_len as i32,
            hidden_size as i32,
            num_groups as i32,
            (hidden_size / num_groups) as i32,
            taps as i32,
            block_size as i32,
            side as i32,
            stream,
        )
    };
    if status != 0 {
        candle::bail!("DFlash2 grouped convolution kernel failed with CUDA error {status}");
    }
    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Tensor::from_storage(
        candle::Storage::Cuda(output),
        candle::Shape::from((sequence_len, hidden_size)),
    )
}

/// Fused F16 grouped dynamic depthwise convolution used by DFlash2.
#[cfg(feature = "cuda")]
pub fn dflash_grouped_conv_f16(
    hidden: &Tensor,
    delta: &Tensor,
    base_kernel: &Tensor,
    block_size: usize,
    side: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::WrapErr;

    let (sequence_len, hidden_size) = hidden.dims2()?;
    let (delta_len, taps, num_groups) = delta.dims3()?;
    let (base_sides, base_taps, base_hidden) = base_kernel.dims3()?;
    if hidden.dtype() != DType::F16
        || delta.dtype() != DType::F16
        || base_kernel.dtype() != DType::F16
        || delta_len != sequence_len
        || base_sides != 2
        || base_taps != taps
        || base_hidden != hidden_size
        || side > 1
        || hidden_size % num_groups != 0
    {
        candle::bail!("invalid DFlash2 grouped convolution shapes or dtypes");
    }

    let dev = hidden.device().as_cuda_device()?;
    let hidden = hidden.contiguous()?;
    let delta = delta.contiguous()?;
    let base_kernel = base_kernel.contiguous()?;
    let ptr = |tensor: &Tensor| -> Result<*const std::ffi::c_void> {
        let (storage, _) = tensor.storage_and_layout();
        match &*storage {
            candle::Storage::Cuda(storage) => {
                Ok(*storage.as_cuda_slice::<half::f16>()?.device_ptr() as *const std::ffi::c_void)
            }
            _ => candle::bail!("DFlash2 convolution tensors must be CUDA tensors"),
        }
    };

    let output = unsafe { dev.alloc::<half::f16>(sequence_len * hidden_size) }.w()?;
    let stream = *dev.cu_stream() as i64;
    let status = unsafe {
        ffi::dflash_grouped_conv_f16(
            ptr(&hidden)?,
            ptr(&delta)?,
            ptr(&base_kernel)?,
            *output.device_ptr() as *mut std::ffi::c_void,
            sequence_len as i32,
            hidden_size as i32,
            num_groups as i32,
            (hidden_size / num_groups) as i32,
            taps as i32,
            block_size as i32,
            side as i32,
            stream,
        )
    };
    if status != 0 {
        candle::bail!("DFlash2 grouped convolution kernel failed with CUDA error {status}");
    }
    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Tensor::from_storage(
        candle::Storage::Cuda(output),
        candle::Shape::from((sequence_len, hidden_size)),
    )
}

#[cfg(feature = "metal")]
fn dflash_grouped_conv_metal(
    hidden: &Tensor,
    delta: &Tensor,
    base_kernel: &Tensor,
    block_size: usize,
    side: usize,
    expected_dtype: DType,
) -> Result<Tensor> {
    let (sequence_len, hidden_size) = hidden.dims2()?;
    let (delta_len, taps, num_groups) = delta.dims3()?;
    let (base_sides, base_taps, base_hidden) = base_kernel.dims3()?;
    if hidden.dtype() != expected_dtype
        || delta.dtype() != expected_dtype
        || base_kernel.dtype() != expected_dtype
        || delta_len != sequence_len
        || base_sides != 2
        || base_taps != taps
        || base_hidden != hidden_size
        || side > 1
        || num_groups == 0
        || sequence_len == 0
        || hidden_size == 0
        || taps == 0
        || hidden_size % num_groups != 0
        || block_size == 0
    {
        candle::bail!("invalid Metal DFlash2 grouped convolution shapes or dtypes");
    }

    let hidden = hidden.contiguous()?;
    let delta = delta.contiguous()?;
    let base_kernel = base_kernel.contiguous()?;
    let output = Tensor::zeros((sequence_len, hidden_size), expected_dtype, hidden.device())?;
    let hidden_m = get_metal_slice(&hidden)?;
    let delta_m = get_metal_slice(&delta)?;
    let base_m = get_metal_slice(&base_kernel)?;
    let output_m = get_metal_slice(&output)?;
    let dev = hidden_m.storage.device();
    let command_buffer = dev.command_buffer()?;
    command_buffer.set_label("dflash-grouped-conv");
    metal_kernels::call_dflash_grouped_conv(
        dev.device(),
        &*command_buffer,
        metal_kernels::Kernels::default(),
        expected_dtype,
        (hidden_m.storage.buffer(), hidden_m.offset_in_bytes),
        (delta_m.storage.buffer(), delta_m.offset_in_bytes),
        (base_m.storage.buffer(), base_m.offset_in_bytes),
        output_m.storage.buffer(),
        sequence_len as u32,
        hidden_size as u32,
        num_groups as u32,
        (hidden_size / num_groups) as u32,
        taps as u32,
        block_size as u32,
        side as u32,
    )
    .map_err(candle_core::Error::wrap)?;
    Ok(output)
}

#[cfg(feature = "metal")]
pub fn dflash_grouped_conv_bf16(
    hidden: &Tensor,
    delta: &Tensor,
    base_kernel: &Tensor,
    block_size: usize,
    side: usize,
) -> Result<Tensor> {
    dflash_grouped_conv_metal(hidden, delta, base_kernel, block_size, side, DType::BF16)
}

#[cfg(feature = "metal")]
pub fn dflash_grouped_conv_f16(
    hidden: &Tensor,
    delta: &Tensor,
    base_kernel: &Tensor,
    block_size: usize,
    side: usize,
) -> Result<Tensor> {
    dflash_grouped_conv_metal(hidden, delta, base_kernel, block_size, side, DType::F16)
}

/// Dispatch fused DFlash2 grouped convolution by the input tensor dtype.
#[cfg(feature = "cuda")]
pub fn dflash_grouped_conv(
    hidden: &Tensor,
    delta: &Tensor,
    base_kernel: &Tensor,
    block_size: usize,
    side: usize,
) -> Result<Tensor> {
    match hidden.dtype() {
        DType::BF16 => {
            let sm = hidden
                .device()
                .as_cuda_device()
                .ok()
                .and_then(|d| crate::cuda_utils::sm_version(d))
                .unwrap_or(0);
            if sm < 80 {
                candle::bail!(
                    "DFlash2 grouped convolution BF16 requires SM80+ (got SM{sm}); use F16 on SM70/SM75"
                );
            }
            dflash_grouped_conv_bf16(hidden, delta, base_kernel, block_size, side)
        }
        DType::F16 => dflash_grouped_conv_f16(hidden, delta, base_kernel, block_size, side),
        dtype => candle::bail!("DFlash2 grouped convolution does not support {dtype:?}"),
    }
}

#[cfg(feature = "metal")]
pub fn dflash_grouped_conv(
    hidden: &Tensor,
    delta: &Tensor,
    base_kernel: &Tensor,
    block_size: usize,
    side: usize,
) -> Result<Tensor> {
    match hidden.dtype() {
        DType::BF16 => dflash_grouped_conv_bf16(hidden, delta, base_kernel, block_size, side),
        DType::F16 => dflash_grouped_conv_f16(hidden, delta, base_kernel, block_size, side),
        dtype => candle::bail!("DFlash2 grouped convolution does not support {dtype:?}"),
    }
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

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn dflash_topk_select_orders_scores_and_ties() -> Result<()> {
        let device = Device::new_metal(0)?;
        let scores = Tensor::from_slice(
            &[1.0f32, 3.0, 2.0, 4.0, 5.0, 5.0, 1.0, 0.0],
            (2, 4),
            &device,
        )?;
        let (weights, indices) = topk_select(&scores, 2)?;
        assert_eq!(
            weights.to_vec2::<f32>()?,
            vec![vec![4.0, 3.0], vec![5.0, 5.0]]
        );
        assert_eq!(indices.to_vec2::<u32>()?, vec![vec![3, 1], vec![0, 1]]);
        Ok(())
    }

    #[test]
    fn dflash_candidate_selection_walks_previous_tokens() -> Result<()> {
        let device = Device::new_metal(0)?;
        let hidden = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], (2, 2), &device)?;
        let unary = Tensor::zeros((2, 2), DType::F32, &device)?;
        let candidate_ids = Tensor::from_slice(&[1u32, 2, 2, 3], (2, 2), &device)?;
        let predecessor = Tensor::from_slice(
            &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            (4, 2),
            &device,
        )?;
        let successor = Tensor::from_slice(
            &[0.0f32, 0.0, 2.0, 0.0, 1.0, 3.0, 1.0, 2.0],
            (4, 2),
            &device,
        )?;
        let anchor = Tensor::from_slice(&[0u32], 1, &device)?;
        let selected = dflash_select_candidates(
            &hidden,
            &unary,
            &candidate_ids,
            &predecessor,
            &successor,
            &anchor,
        )?;
        assert_eq!(selected.to_vec1::<u32>()?, vec![1, 2]);
        Ok(())
    }

    #[test]
    fn dflash_grouped_conv_respects_block_boundaries() -> Result<()> {
        let device = Device::new_metal(0)?;
        let hidden = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (4, 2),
            &device,
        )?
        .to_dtype(DType::F16)?;
        let delta = Tensor::zeros((4, 2, 1), DType::F16, &device)?;
        let base_kernel = Tensor::from_slice(
            &[1.0f32, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            (2, 2, 2),
            &device,
        )?
        .to_dtype(DType::F16)?;
        let output =
            dflash_grouped_conv(&hidden, &delta, &base_kernel, 2, 0)?.to_dtype(DType::F32)?;
        assert_eq!(
            output.to_vec2::<f32>()?,
            vec![
                vec![1.0, 2.0],
                vec![5.0, 8.0],
                vec![5.0, 6.0],
                vec![17.0, 20.0]
            ]
        );
        Ok(())
    }
}
