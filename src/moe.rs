#[cfg(all(feature = "cuda", feature = "flashinfer"))]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::quantized::QTensor;
use candle_core::{Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_dtype_code(dtype: candle_core::DType) -> Result<i32> {
    match dtype {
        candle_core::DType::F16 => Ok(0),
        candle_core::DType::BF16 => Ok(1),
        _ => candle_core::bail!("only f16/bf16 are supported for flashinfer fused moe"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_ptr_f16bf16(t: &Tensor) -> Result<*const core::ffi::c_void> {
    use candle_core as candle;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (candle::Storage::Cuda(c), candle_core::DType::F16) => {
            Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr() as *const core::ffi::c_void)
        }
        (candle::Storage::Cuda(c), candle_core::DType::BF16) => {
            Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr() as *const core::ffi::c_void)
        }
        _ => candle_core::bail!("expected CUDA f16/bf16 tensor"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_ptr_u8(t: &Tensor) -> Result<*const u8> {
    use candle_core as candle;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (candle::Storage::Cuda(c), candle_core::DType::U8) => {
            Ok(*c.as_cuda_slice::<u8>()?.device_ptr() as *const u8)
        }
        _ => candle_core::bail!("expected CUDA u8 tensor"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_ptr_f32(t: &Tensor) -> Result<*const f32> {
    use candle_core as candle;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (candle::Storage::Cuda(c), candle_core::DType::F32) => {
            Ok(*c.as_cuda_slice::<f32>()?.device_ptr() as *const f32)
        }
        _ => candle_core::bail!("expected CUDA f32 tensor"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_ptr_topk_ids_i32(t: &Tensor) -> Result<*const i32> {
    use candle_core as candle;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (candle::Storage::Cuda(c), candle_core::DType::U32) => {
            Ok(*c.as_cuda_slice::<u32>()?.device_ptr() as *const i32)
        }
        _ => candle_core::bail!("expected CUDA u32 tensor for topk ids"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn cuda_mut_ptr_f16bf16(t: &Tensor) -> Result<*mut core::ffi::c_void> {
    use candle_core as candle;
    let (storage, _) = t.storage_and_layout();
    match (&*storage, t.dtype()) {
        (candle::Storage::Cuda(c), candle_core::DType::F16) => {
            Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr() as *mut core::ffi::c_void)
        }
        (candle::Storage::Cuda(c), candle_core::DType::BF16) => {
            Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr() as *mut core::ffi::c_void)
        }
        _ => candle_core::bail!("expected CUDA f16/bf16 tensor"),
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
pub fn flashinfer_fused_moe(
    input: &Tensor,
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    gate_up_weights: &Tensor,
    down_weights: &Tensor,
) -> Result<Tensor> {
    let (num_tokens, hidden_size) = input.dims2()?;
    let (num_experts, gate_up_n, gate_up_k) = gate_up_weights.dims3()?;
    let (down_experts, down_n, down_k) = down_weights.dims3()?;
    let (topk_tokens, top_k) = topk_ids.dims2()?;
    let (topk_w_tokens, topk_w_k) = topk_weights.dims2()?;
    if !input.is_contiguous()
        || !topk_ids.is_contiguous()
        || !topk_weights.is_contiguous()
        || !gate_up_weights.is_contiguous()
        || !down_weights.is_contiguous()
    {
        candle_core::bail!("flashinfer fused moe expects contiguous tensors");
    }
    if topk_tokens != num_tokens || topk_w_tokens != num_tokens || topk_w_k != top_k {
        candle_core::bail!("flashinfer fused moe: invalid topk tensors");
    }
    if gate_up_k != hidden_size || down_experts != num_experts || down_n != hidden_size {
        candle_core::bail!("flashinfer fused moe: invalid tensor shapes for moe weights");
    }
    if gate_up_n % 2 != 0 {
        candle_core::bail!("flashinfer fused moe: gate_up second dim must be even");
    }
    if down_k * 2 != gate_up_n {
        candle_core::bail!("flashinfer fused moe: gate_up/down intermediate dims mismatch");
    }
    let input_dtype = cuda_dtype_code(input.dtype())?;
    let weight_dtype = cuda_dtype_code(gate_up_weights.dtype())?;
    if input_dtype != weight_dtype {
        candle_core::bail!("flashinfer fused moe: input and weight dtype must match");
    }
    let dev = input.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;

    let output = Tensor::zeros((num_tokens, hidden_size), input.dtype(), input.device())?;
    let status = unsafe {
        ffi::flashinfer_fused_moe_bf16(
            cuda_ptr_f16bf16(input)?,
            cuda_ptr_topk_ids_i32(topk_ids)?,
            cuda_ptr_f32(topk_weights)?,
            cuda_ptr_f16bf16(gate_up_weights)?,
            cuda_ptr_f16bf16(down_weights)?,
            cuda_mut_ptr_f16bf16(&output)?,
            num_tokens as i32,
            hidden_size as i32,
            down_k as i32,
            num_experts as i32,
            top_k as i32,
            input_dtype,
            weight_dtype,
            stream,
        )
    };
    if status != 0 {
        candle_core::bail!("flashinfer fused moe bf16 kernel failed with status {status}");
    }
    Ok(output)
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
pub fn flashinfer_fused_moe_fp8(
    input: &Tensor,
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    gate_up_weights: &Tensor,
    gate_up_scales: &Tensor,
    down_weights: &Tensor,
    down_scales: &Tensor,
) -> Result<Tensor> {
    let (num_tokens, hidden_size) = input.dims2()?;
    let (num_experts, gate_up_n, gate_up_k) = gate_up_weights.dims3()?;
    let (down_experts, down_n, down_k) = down_weights.dims3()?;
    let (gate_up_scale_experts, gate_up_scale_n, gate_up_scale_k) = gate_up_scales.dims3()?;
    let (down_scale_experts, down_scale_n, down_scale_k) = down_scales.dims3()?;
    let (topk_tokens, top_k) = topk_ids.dims2()?;
    let (topk_w_tokens, topk_w_k) = topk_weights.dims2()?;
    if !input.is_contiguous()
        || !topk_ids.is_contiguous()
        || !topk_weights.is_contiguous()
        || !gate_up_weights.is_contiguous()
        || !gate_up_scales.is_contiguous()
        || !down_weights.is_contiguous()
        || !down_scales.is_contiguous()
    {
        candle_core::bail!("flashinfer fused moe fp8 expects contiguous tensors");
    }
    if gate_up_weights.dtype() != candle_core::DType::U8
        || down_weights.dtype() != candle_core::DType::U8
        || gate_up_scales.dtype() != candle_core::DType::F32
        || down_scales.dtype() != candle_core::DType::F32
    {
        candle_core::bail!("flashinfer fused moe fp8 expects u8 weights and f32 scales");
    }
    if topk_tokens != num_tokens || topk_w_tokens != num_tokens || topk_w_k != top_k {
        candle_core::bail!("flashinfer fused moe fp8: invalid topk tensors");
    }
    if gate_up_k != hidden_size || down_experts != num_experts || down_n != hidden_size {
        candle_core::bail!("flashinfer fused moe fp8: invalid tensor shapes for moe weights");
    }
    if gate_up_n % 2 != 0 || down_k * 2 != gate_up_n {
        candle_core::bail!("flashinfer fused moe fp8: gate_up/down intermediate dims mismatch");
    }
    if gate_up_scale_experts != num_experts || down_scale_experts != num_experts {
        candle_core::bail!("flashinfer fused moe fp8: scale tensor expert dim mismatch");
    }
    if hidden_size % 128 != 0 || down_k % 128 != 0 {
        candle_core::bail!(
            "flashinfer fused moe fp8: hidden/intermediate dims must be divisible by 128"
        );
    }
    let expected_gate_up_scale_n = gate_up_n / 128;
    let expected_gate_up_scale_k = hidden_size / 128;
    let expected_down_scale_n = hidden_size / 128;
    let expected_down_scale_k = down_k / 128;
    if gate_up_scale_n != expected_gate_up_scale_n
        || gate_up_scale_k != expected_gate_up_scale_k
        || down_scale_n != expected_down_scale_n
        || down_scale_k != expected_down_scale_k
    {
        candle_core::bail!(
            "flashinfer fused moe fp8: invalid scale tensor shapes, expected gate_up=[{num_experts}, {expected_gate_up_scale_n}, {expected_gate_up_scale_k}], down=[{num_experts}, {expected_down_scale_n}, {expected_down_scale_k}]"
        );
    }
    let input_dtype = cuda_dtype_code(input.dtype())?;
    let dev = input.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;

    let output = Tensor::zeros(
        (num_tokens, hidden_size),
        candle_core::DType::BF16,
        input.device(),
    )?;
    let status = unsafe {
        ffi::flashinfer_fused_moe_fp8(
            cuda_ptr_f16bf16(input)?,
            cuda_ptr_topk_ids_i32(topk_ids)?,
            cuda_ptr_f32(topk_weights)?,
            cuda_ptr_u8(gate_up_weights)?,
            cuda_ptr_f32(gate_up_scales)?,
            cuda_ptr_u8(down_weights)?,
            cuda_ptr_f32(down_scales)?,
            cuda_mut_ptr_f16bf16(&output)?,
            num_tokens as i32,
            hidden_size as i32,
            down_k as i32,
            num_experts as i32,
            top_k as i32,
            input_dtype,
            stream,
        )
    };
    if status != 0 {
        candle_core::bail!("flashinfer fused moe fp8 kernel failed with status {status}");
    }
    Ok(output)
}

#[cfg(feature = "cuda")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::DType;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (input_rows, size_k1) = input.dims2()?;
        let size_m = if topk_weights.is_none() {
            input_rows * topk
        } else {
            input_rows
        };
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );
        let dev = input.device().as_cuda_device()?;
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                candle_core::bail!("moe_gemm_wmma only accept f16/bf16 inputs!")
            }
        };

        let (input, _) = input.storage_and_layout();
        let input = match &*input {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let (weights, _) = weights.storage_and_layout();
        let weights = match &*weights {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            *topk_weights.device_ptr() as *const f32
        } else {
            std::ptr::null() as *const f32
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }.w()?;

        let stream = *dev.cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
            if is_prefill || size_m > 128 {
                let expert_counts = dev.alloc::<u32>(num_experts).w()?;
                let expert_offsets = dev.alloc::<u32>(num_experts + 1).w()?;
                ffi::moe_gemm_wmma(
                    *input.device_ptr() as *const c_void,   // [size_m, size_k]
                    *weights.device_ptr() as *const c_void, // [num_experts, size_n, size_k]
                    *sorted_token_ids.device_ptr() as *const i32,
                    *experts_ids.device_ptr() as *const i32,
                    topk_weights_ptr,
                    *output.device_ptr() as *mut c_void, // [size_m, size_n]
                    *expert_counts.device_ptr() as *mut i32, // pre-allocated buffer [num_experts]
                    *expert_offsets.device_ptr() as *mut i32, // pre-allocated buffer [num_experts + 1]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    data_type as i32, // 0=float16, 1=bf16 (for input/output)
                    is_prefill,
                    stream as i64,
                );
            } else {
                ffi::moe_gemv(
                    *input.device_ptr() as *const c_void,   // [size_m, size_k]
                    *weights.device_ptr() as *const c_void, // [num_experts, size_n, size_k]
                    *sorted_token_ids.device_ptr() as *const i32,
                    *experts_ids.device_ptr() as *const i32,
                    topk_weights_ptr,
                    *output.device_ptr() as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    data_type as i32, // 0=float16, 1=bf16 (for input/output)
                    stream as i64,
                );
            }
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(candle::Storage::Cuda(output), (size_m, size_n))?;

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        _ => {
            candle_core::bail!("moe_gemm only accept f16/bf16 inputs!")
        }
    }
}

/// MoE GEMM with FP8 weights and block-wise scales.
///
/// # Arguments
/// * `input` - Input tensor [size_m, size_k] in F16/BF16
/// * `weights` - FP8 weights as U8 tensor [num_experts, size_n, size_k]
/// * `weight_scales` - Block-wise scales [num_experts, scale_n_dim, scale_k_dim] in F32
/// * `topk_weights` - Optional per-token gating weights [size_m]
/// * `sorted_token_ids` - Sorted token indices [size_m]
/// * `experts_ids` - Expert IDs [size_m]
/// * `topk` - Number of experts per token
/// * `block_size_n` - Block size in N dimension for scales
/// * `block_size_k` - Block size in K dimension for scales
/// * `is_prefill` - Whether this is prefill (uses WMMA) or decode (uses GEMV)
#[cfg(feature = "cuda")]
pub fn moe_gemm_fp8(
    input: &Tensor,
    weights: &Tensor,       // U8 tensor for FP8 weights
    weight_scales: &Tensor, // F32 tensor for scales
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    block_size_n: usize,
    block_size_k: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::DType;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        weight_scales: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        block_size_n: usize,
        block_size_k: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (input_rows, size_k1) = input.dims2()?;
        let size_m = if topk_weights.is_none() {
            input_rows * topk
        } else {
            input_rows
        };
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );

        // Validate weight dtype is U8 (FP8)
        assert!(
            weights.dtype() == DType::U8,
            "moe_gemm_fp8 expects U8 weights for FP8, got {:?}",
            weights.dtype()
        );

        assert!(
            weight_scales.dtype() == DType::F32,
            "moe_gemm_fp8 expects f32 scales, got {:?}",
            weight_scales.dtype()
        );

        #[cfg(feature = "cutlass")]
        let device = input.device().clone();
        let input_dtype = input.dtype();
        let dev = input.device().as_cuda_device()?;
        #[cfg(feature = "cutlass")]
        let sm_version = crate::cuda_utils::sm_version(dev).unwrap_or(0) as i32;
        let data_type = match input_dtype {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                candle_core::bail!("moe_gemm_fp8 only accepts f16/bf16 inputs!")
            }
        };

        let (input, _) = input.storage_and_layout();
        let input = match &*input {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };

        let (weights, _) = weights.storage_and_layout();
        let weights = match &*weights {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
            _ => candle::bail!("weights must be a cuda tensor"),
        };

        let (weight_scales, _) = weight_scales.storage_and_layout();
        let weight_scales = match &*weight_scales {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("weight_scales must be a cuda tensor"),
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            *topk_weights.device_ptr() as *const f32
        } else {
            std::ptr::null() as *const f32
        };

        #[cfg(feature = "cutlass")]
        let use_cutlass = sm_version >= 90 && block_size_n == 128 && block_size_k == 128;
        #[cfg(not(feature = "cutlass"))]
        let use_cutlass = false;

        if use_cutlass && (is_prefill || size_m > 128) {
            #[cfg(feature = "cutlass")]
            {
                let k_blocks = (size_k + block_size_k - 1) / block_size_k;
                let num_groups_per_row = k_blocks;
                let num_groups = (input_rows * num_groups_per_row) as i32;

                // SM100+ (Blackwell) requires column-major scale layout (UMMA::Major::MN)
                // SM90 (Hopper) requires row-major scale layout (GMMA::Major::K)
                let is_column_major_scales = sm_version >= 100;

                let input_q = Tensor::zeros((input_rows, size_k), DType::U8, &device)?;
                let input_scale = if is_column_major_scales {
                    // Column-major: allocate transposed and transpose for column-major view
                    Tensor::zeros((k_blocks, input_rows), DType::F32, &device)?.t()?
                } else {
                    // Row-major: standard contiguous layout
                    Tensor::zeros((input_rows, k_blocks), DType::F32, &device)?
                };
                let rep_a_q = Tensor::zeros((size_m, size_k), DType::U8, &device)?;
                let rep_a_scales = if is_column_major_scales {
                    Tensor::zeros((k_blocks, size_m), DType::F32, &device)?.t()?
                } else {
                    Tensor::zeros((size_m, k_blocks), DType::F32, &device)?
                };
                let rep_out = Tensor::zeros((size_m, size_n), input_dtype, &device)?;
                let output = Tensor::zeros((size_m, size_n), input_dtype, &device)?;
                let map_divisor = if topk_weights.is_none() {
                    topk as i32
                } else {
                    1
                };

                // Get scale stride for quantization kernel
                let input_scale_stride = if is_column_major_scales {
                    input_rows as i32 // Column-major stride
                } else {
                    num_groups_per_row as i32 // Row-major stride
                };

                let (input_q, _) = input_q.storage_and_layout();
                let input_q = match &*input_q {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
                    _ => candle::bail!("input_q must be a cuda tensor"),
                };
                let (input_scale, _) = input_scale.storage_and_layout();
                let input_scale = match &*input_scale {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("input_scale must be a cuda tensor"),
                };
                let (rep_a_q, _) = rep_a_q.storage_and_layout();
                let rep_a_q = match &*rep_a_q {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
                    _ => candle::bail!("rep_a_q must be a cuda tensor"),
                };
                let (rep_a_scales, _) = rep_a_scales.storage_and_layout();
                let rep_a_scales = match &*rep_a_scales {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("rep_a_scales must be a cuda tensor"),
                };
                let (rep_out, _) = rep_out.storage_and_layout();
                let rep_out = match &*rep_out {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                    _ => candle::bail!("rep_out must be a cuda tensor"),
                };
                let (output, _) = output.storage_and_layout();
                let output = match &*output {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                    _ => candle::bail!("output must be a cuda tensor"),
                };

                let expert_counts = unsafe { dev.alloc::<i32>(num_experts).w()? };
                let expert_offsets = unsafe { dev.alloc::<i32>(num_experts + 1).w()? };

                let stream = *dev.cu_stream() as i64;
                use core::ffi::c_void;
                unsafe {
                    ffi::fp8_quantize_per_token_group_launch(
                        *input.device_ptr() as *const c_void,
                        *input_q.device_ptr() as *mut c_void,
                        *input_scale.device_ptr() as *mut f32,
                        num_groups as i32,
                        128,
                        num_groups_per_row as i32,
                        input_scale_stride,
                        data_type == 0,
                        is_column_major_scales,
                        stream as i64,
                    );

                    ffi::moe_fp8_shuffle_rows_u8(
                        *input_q.device_ptr() as *const u8,
                        *sorted_token_ids.device_ptr() as *const i32,
                        *rep_a_q.device_ptr() as *mut u8,
                        input_rows as i64,
                        size_m as i64,
                        size_k as i64,
                        map_divisor,
                        stream as i64,
                    );

                    // Use strided shuffle for column-major scales (SM100+ Blackwell)
                    // or regular shuffle for row-major scales (SM90)
                    if is_column_major_scales {
                        ffi::moe_fp8_shuffle_rows_f32_strided(
                            *input_scale.device_ptr() as *const f32,
                            *sorted_token_ids.device_ptr() as *const i32,
                            *rep_a_scales.device_ptr() as *mut f32,
                            input_rows as i64,
                            size_m as i64,
                            num_groups_per_row as i64,
                            input_rows as i64, // src_row_stride (column-major)
                            size_m as i64,     // dst_row_stride (column-major)
                            map_divisor,
                            stream as i64,
                        );
                    } else {
                        ffi::moe_fp8_shuffle_rows_f32(
                            *input_scale.device_ptr() as *const f32,
                            *sorted_token_ids.device_ptr() as *const i32,
                            *rep_a_scales.device_ptr() as *mut f32,
                            input_rows as i64,
                            size_m as i64,
                            num_groups_per_row as i64,
                            map_divisor,
                            stream as i64,
                        );
                    }

                    ffi::moe_fp8_calculate_expert_offsets(
                        *experts_ids.device_ptr() as *const i32,
                        *expert_counts.device_ptr() as *mut i32,
                        *expert_offsets.device_ptr() as *mut i32,
                        num_experts as i32,
                        size_m as i32,
                        is_prefill,
                        stream as i64,
                    );

                    if data_type == 0 {
                        ffi::moe_fp8_grouped_gemm_f16(
                            *rep_a_q.device_ptr() as *const u8,
                            *weights.device_ptr() as *const u8,
                            *rep_a_scales.device_ptr() as *const f32,
                            *weight_scales.device_ptr() as *const f32,
                            *expert_offsets.device_ptr() as *const i32,
                            num_experts as i32,
                            size_m as i32,
                            size_n as i32,
                            size_k as i32,
                            block_size_n as i32,
                            block_size_k as i32,
                            sm_version as i32,
                            *rep_out.device_ptr() as *mut c_void,
                            stream as i64,
                        );
                        ffi::moe_fp8_scatter_rows_f16(
                            *rep_out.device_ptr() as *const c_void,
                            *sorted_token_ids.device_ptr() as *const i32,
                            *output.device_ptr() as *mut c_void,
                            size_m as i64,
                            size_m as i64,
                            size_n as i64,
                            topk_weights_ptr,
                            stream as i64,
                        );
                    } else {
                        ffi::moe_fp8_grouped_gemm_bf16(
                            *rep_a_q.device_ptr() as *const u8,
                            *weights.device_ptr() as *const u8,
                            *rep_a_scales.device_ptr() as *const f32,
                            *weight_scales.device_ptr() as *const f32,
                            *expert_offsets.device_ptr() as *const i32,
                            num_experts as i32,
                            size_m as i32,
                            size_n as i32,
                            size_k as i32,
                            block_size_n as i32,
                            block_size_k as i32,
                            sm_version as i32,
                            *rep_out.device_ptr() as *mut c_void,
                            stream as i64,
                        );
                        ffi::moe_fp8_scatter_rows_bf16(
                            *rep_out.device_ptr() as *const c_void,
                            *sorted_token_ids.device_ptr() as *const i32,
                            *output.device_ptr() as *mut c_void,
                            size_m as i64,
                            size_m as i64,
                            size_n as i64,
                            topk_weights_ptr,
                            stream as i64,
                        );
                    }
                }

                let output = candle::CudaStorage::wrap_cuda_slice(output.clone(), dev.clone());
                let output = Tensor::from_storage(candle::Storage::Cuda(output), (size_m, size_n))?;
                return Ok(output);
            }
        }

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }.w()?;

        let stream = *dev.cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
            if is_prefill || size_m > 128 {
                let expert_counts = dev.alloc::<u32>(num_experts).w()?;
                let expert_offsets = dev.alloc::<u32>(num_experts + 1).w()?;
                ffi::moe_gemm_wmma_fp8(
                    *input.device_ptr() as *const c_void,
                    *weights.device_ptr() as *const u8,
                    *weight_scales.device_ptr() as *const f32,
                    *sorted_token_ids.device_ptr() as *const i32,
                    *experts_ids.device_ptr() as *const i32,
                    topk_weights_ptr,
                    *output.device_ptr() as *mut c_void,
                    *expert_counts.device_ptr() as *mut i32,
                    *expert_offsets.device_ptr() as *mut i32,
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    block_size_n as i32,
                    block_size_k as i32,
                    data_type as i32,
                    is_prefill,
                    stream as i64,
                );
            } else {
                ffi::moe_gemv_fp8(
                    *input.device_ptr() as *const c_void,
                    *weights.device_ptr() as *const u8,
                    *weight_scales.device_ptr() as *const f32,
                    *sorted_token_ids.device_ptr() as *const i32,
                    *experts_ids.device_ptr() as *const i32,
                    topk_weights_ptr,
                    *output.device_ptr() as *mut c_void,
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    block_size_n as i32,
                    block_size_k as i32,
                    data_type as i32,
                    stream as i64,
                );
            }
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(candle::Storage::Cuda(output), (size_m, size_n))?;

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            weight_scales,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            block_size_n,
            block_size_k,
            is_prefill,
        ),
        DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            weight_scales,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            block_size_n,
            block_size_k,
            is_prefill,
        ),
        _ => {
            candle_core::bail!("moe_gemm_fp8 only accepts f16/bf16 inputs!")
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn moe_gemm_fp8(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: usize,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm_fp8 is not implemented on this platform!")
}

#[cfg(feature = "metal")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle_core as candle;
    use candle_core::{DType, Storage};

    let (input_rows, size_k1) = input.dims2()?;
    let size_m = if topk_weights.is_none() {
        input_rows * topk
    } else {
        input_rows
    };
    let (num_experts, size_n, size_k) = weights.dims3()?;
    assert!(
        size_k == size_k1,
        "input {:?} and weight {:?} last dim mismatch!",
        size_k1,
        size_k
    );

    let dtype = input.dtype();
    if dtype != DType::F16 && dtype != DType::BF16 {
        candle_core::bail!(
            "moe_gemm on Metal only supports f16/bf16 inputs, got {:?}",
            dtype
        );
    }

    let dev = input.device();
    let metal_dev = dev.as_metal_device()?;
    let command_buffer = metal_dev.command_buffer()?;
    let command_buffer = command_buffer.as_ref();

    let output = Tensor::zeros((size_m, size_n), dtype, dev)?;

    {
        let (input_s, input_l) = input.storage_and_layout();
        let input_s = match &*input_s {
            Storage::Metal(s) => s,
            _ => candle::bail!("input must be a metal tensor"),
        };

        let (weights_s, weights_l) = weights.storage_and_layout();
        let weights_s = match &*weights_s {
            Storage::Metal(s) => s,
            _ => candle::bail!("weights must be a metal tensor"),
        };

        let (sti_s, sti_l) = sorted_token_ids.storage_and_layout();
        let sti_s = match &*sti_s {
            Storage::Metal(s) => s,
            _ => candle::bail!("sorted_token_ids must be a metal tensor"),
        };

        let (eid_s, eid_l) = experts_ids.storage_and_layout();
        let eid_s = match &*eid_s {
            Storage::Metal(s) => s,
            _ => candle::bail!("experts_ids must be a metal tensor"),
        };

        let topk_weights_buf = if let Some(tw) = topk_weights {
            let (tw_s, tw_l) = tw.storage_and_layout();
            let tw_s = match &*tw_s {
                Storage::Metal(s) => s,
                _ => candle::bail!("topk_weights must be a metal tensor"),
            };
            Some((
                tw_s.buffer().clone(),
                tw_l.start_offset() * tw.dtype().size_in_bytes(),
            ))
        } else {
            None
        };

        let (output_s, output_l) = output.storage_and_layout();
        let output_s = match &*output_s {
            Storage::Metal(s) => s,
            _ => candle::bail!("output must be a metal tensor"),
        };

        let tw_ref = topk_weights_buf
            .as_ref()
            .map(|(buf, off)| (buf as &metal::Buffer, *off));

        metal_kernels::call_moe_gemm(
            metal_dev.device(),
            command_buffer,
            metal_kernels::Kernels::default(),
            dtype,
            input_s.buffer(),
            input_l.start_offset() * dtype.size_in_bytes(),
            weights_s.buffer(),
            weights_l.start_offset() * dtype.size_in_bytes(),
            sti_s.buffer(),
            sti_l.start_offset() * sorted_token_ids.dtype().size_in_bytes(),
            eid_s.buffer(),
            eid_l.start_offset() * experts_ids.dtype().size_in_bytes(),
            tw_ref,
            output_s.buffer(),
            output_l.start_offset() * dtype.size_in_bytes(),
            num_experts as i32,
            topk as i32,
            size_m as i32,
            size_n as i32,
            size_k as i32,
            is_prefill,
        )
        .map_err(candle_core::Error::wrap)?;
    }

    Ok(output)
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub fn moe_gemm(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm is not implemented on this platform!")
}

#[cfg(feature = "cuda")]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::quantized::GgmlDType;
    use candle_core::DType;
    use half::{bf16, f16};

    fn cuda_fwd(
        input: &Tensor,
        weights: &QTensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let (mut size_m, size_k) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k,
            size_k1,
        );
        let dev = input.device().as_cuda_device()?;

        // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5
        let gguf_dtype = match weights.dtype() {
            GgmlDType::Q8_0 => 0,
            GgmlDType::Q4K => 1,
            GgmlDType::Q2K => 2,
            GgmlDType::Q3K => 3,
            GgmlDType::Q5K => 4,
            GgmlDType::Q6K => 5,
            _ => {
                candle_core::bail!(
                    "moe_gemm_gguf `ISQ` only accept q2k, q3k, q4k, q5k, q6k or q8_0 weights!"
                )
            }
        };

        let weight_ptr = weights.device_ptr()?;

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, _) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            *topk_weights.device_ptr() as *const f32
        } else {
            std::ptr::null() as *const f32
        };

        let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };
        let (experts_ids, _) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };

        let output = unsafe { dev.alloc::<f32>(size_m * size_n) }.w()?;
        let stream = *dev.cu_stream() as i64;
        use core::ffi::c_void;

        assert!(size_k % 8 == 0, "size_k must divisible by 8");
        unsafe {
            if is_prefill {
                let input = input.to_dtype(dtype)?;
                let (input, _) = input.storage_and_layout();
                let (input_ptr, input_dtype) = match &*input {
                    candle::Storage::Cuda(c) => {
                        if dtype == DType::F16 {
                            (*c.as_cuda_slice::<f16>()?.device_ptr() as *const c_void, 0)
                        } else {
                            (*c.as_cuda_slice::<bf16>()?.device_ptr() as *const c_void, 1)
                        }
                    }
                    _ => candle::bail!("input must be a cuda tensor"),
                };
                ffi::moe_gemm_gguf_prefill(
                    input_ptr,               // [size_m or size_m/topk, size_k]
                    weight_ptr as *const u8, // [num_experts, size_n, size_k]
                    *sorted_token_ids.device_ptr() as *const i32,
                    *experts_ids.device_ptr() as *const i32,
                    topk_weights_ptr,
                    *output.device_ptr() as *mut c_void, // [size_m, size_n]
                    num_experts as i32,
                    topk as i32,
                    size_m as i32,
                    size_n as i32,
                    size_k as i32,
                    input_dtype as i32,
                    gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                    stream as i64,
                );
            } else {
                let (input, _) = input.storage_and_layout();
                let input = match &*input {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("input must be a cuda tensor"),
                };

                // Use optimized small-M kernel for batch size < 8 (decode scenarios)
                if size_m <= 8 {
                    ffi::moe_gemm_gguf_small_m(
                        *input.device_ptr() as *const f32, // [size_m or size_m/topk, size_k]
                        weight_ptr as *const c_void,       // [num_experts, size_n, size_k]
                        *sorted_token_ids.device_ptr() as *const i32,
                        *experts_ids.device_ptr() as *const i32,
                        topk_weights_ptr,
                        *output.device_ptr() as *mut c_void, // [size_m, size_n]
                        num_experts as i32,
                        topk as i32,
                        size_m as i32,
                        size_n as i32,
                        size_k as i32,
                        gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                        stream as i64,
                    );
                } else {
                    ffi::moe_gemm_gguf(
                        *input.device_ptr() as *const f32, // [size_m or size_m/topk, size_k]
                        weight_ptr as *const c_void,       // [num_experts, size_n, size_k]
                        *sorted_token_ids.device_ptr() as *const i32,
                        *experts_ids.device_ptr() as *const i32,
                        topk_weights_ptr,
                        *output.device_ptr() as *mut c_void, // [size_m, size_n]
                        num_experts as i32,
                        topk as i32,
                        size_m as i32,
                        size_n as i32,
                        size_k as i32,
                        gguf_dtype as i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5 (for weight)
                        stream as i64,
                    );
                }
            }
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from_storage(candle::Storage::Cuda(output), (size_m, size_n))?;

        Ok(output)
    }

    match input.dtype() {
        DType::F32 => cuda_fwd(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
            dtype,
        ),
        _ => {
            candle_core::bail!("moe_gemm_gguf only accept f16/bf16 inputs!")
        }
    }
}

#[cfg(feature = "metal")]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    use candle_core::DType;

    let _shape = weights.shape().dims3()?;
    let dequant = weights.dequantize_f16(input.device())?;

    let compute_dtype = if dtype == DType::F16 || dtype == DType::BF16 {
        dtype
    } else {
        DType::F16
    };

    let input_cast = if input.dtype() != compute_dtype {
        input.to_dtype(compute_dtype)?
    } else {
        input.clone()
    };

    let dequant = if dequant.dtype() != compute_dtype {
        dequant.to_dtype(compute_dtype)?
    } else {
        dequant
    };

    let result = moe_gemm(
        &input_cast,
        &dequant,
        topk_weights,
        sorted_token_ids,
        experts_ids,
        topk,
        is_prefill,
    )?;

    if result.dtype() != DType::F32 {
        result.to_dtype(DType::F32)
    } else {
        Ok(result)
    }
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub fn moe_gemm_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
    _: candle_core::DType,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm_gguf is not implemented on this platform!")
}
