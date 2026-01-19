use candle_core::quantized::QTensor;
use candle_core::{Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;

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
        let (mut size_m, size_k1) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
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
            if is_prefill || size_m > 8 {
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
        let (mut size_m, size_k1) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
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

        let dev = input.device().as_cuda_device()?;
        let data_type = match input.dtype() {
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

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }.w()?;

        let stream = *dev.cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
            if is_prefill || size_m > 8 {
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

#[cfg(not(feature = "cuda"))]
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

#[cfg(not(feature = "cuda"))]
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
