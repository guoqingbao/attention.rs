use candle_core as candle;
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
        let expert_counts = unsafe { dev.alloc::<u32>(num_experts) }.w()?;
        let expert_offsets = unsafe { dev.alloc::<u32>(num_experts + 1) }.w()?;

        let stream = *dev.cu_stream() as i64;
        use core::ffi::c_void;

        unsafe {
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

#[cfg(feature = "metal")]
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

//example:
//input [3355, 1, 2048]
//weight [128, 768, 2048]
//indices [3355, 8]
//output [3355, 8, 768]
#[cfg(feature = "gcu")]
fn indexed_moe_func<
    T: candle::gcu_backend::GcuDType + candle::gcu_backend::DeviceCopy + candle::WithDType,
>(
    input: &Tensor,
    weight: &Tensor,
    indices: &Tensor,
) -> Result<Tensor> {
    use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
    use candle::gcu_backend::ubridge::ffi::{indexed_moe_bf16, indexed_moe_f16};
    use candle::gcu_backend::WrapErr;
    use candle::Storage;
    use candle_core::DType;
    use half::{bf16, f16};
    let dev = input.device().as_gcu_device()?;
    let (input_value, input_l) = input.storage_and_layout();
    // let (out_value, out_l) = out.storage_and_layout();
    let (weight_value, weight_l) = weight.storage_and_layout();
    let (indices_value, indices_l) = indices.storage_and_layout();

    assert!(
        input.dims().len() == 3 && weight.dims().len() == 3 && indices.dims().len() == 2,
        "Invalid input dims!"
    );
    let (b1, topk) = indices.dims2()?;
    let (batch, m, k) = input.dims3()?;
    let (num_experts, n, k1) = weight.dims3()?;
    let tile_size = if batch > 12 { 128 } else { 64 };
    assert!(
        k % tile_size == 0,
        "indexed_moe: k dim must be aligned to {}!",
        tile_size
    );
    assert!(
        n % tile_size == 0,
        "indexed_moe: n dim must be aligned to {}!",
        tile_size
    );

    // let (b2, _, n1) = out_l.dims3()?;

    let out = dev.alloc::<T>(batch * topk * n).w()?;

    // assert!(n == n1, "weight and expert out should have same last dim!");
    assert!(
        b1 == batch,
        "the first dim of indices tensor should match input and expert out!"
    );
    assert!(
        k == k1,
        "weight tensor should match input tensor for matmul!"
    );
    // assert!(
    //     n == n1,
    //     "weight tensor should match output tensor for matmul!"
    // );

    // let el_count = shape.elem_count();
    let stream = dev.stream_inner().unwrap();
    let input_value = match &*input_value {
        Storage::Gcu(s) => s,
        _ => candle::bail!("tensor must be a gcu tensor"),
    };
    let input_value = input_value.as_gcu_slice::<T>()?;
    let input_value = input_value.slice(input_l.start_offset()..);

    let weight_value = match &*weight_value {
        Storage::Gcu(s) => s,
        _ => candle::bail!("tensor must be a gcu tensor"),
    };
    let weight_value = weight_value.as_gcu_slice::<T>()?;
    let weight_value = weight_value.slice(weight_l.start_offset()..);

    let indices_value = match &*indices_value {
        Storage::Gcu(s) => s,
        _ => candle::bail!("tensor must be a gcu tensor"),
    };
    let indices_value = indices_value.as_gcu_slice::<u32>()?;
    let indices_value = indices_value.slice(indices_l.start_offset()..);

    match input.dtype() {
        DType::F16 => unsafe {
            indexed_moe_f16(
                input_value.device_ptr() as *mut f16,
                weight_value.device_ptr() as *mut f16,
                out.device_ptr() as *mut f16,
                indices_value.device_ptr() as *mut u32,
                n as i32,
                k as i32,
                m as i32,
                batch as i32,
                topk as i32,
                num_experts as i32,
                stream as *mut core::ffi::c_void,
            );
        },
        DType::BF16 => unsafe {
            indexed_moe_bf16(
                input_value.device_ptr() as *mut bf16,
                weight_value.device_ptr() as *mut bf16,
                out.device_ptr() as *mut bf16,
                indices_value.device_ptr() as *mut u32,
                n as i32,
                k as i32,
                m as i32,
                batch as i32,
                topk as i32,
                num_experts as i32,
                stream as *mut core::ffi::c_void,
            );
        },
        _ => {
            panic!("not supported data type!")
        }
    }
    let s_out = candle::GcuStorage::wrap_gcu_slice(out, dev.clone());
    Ok(Tensor::from_storage(
        candle::Storage::Gcu(s_out),
        (batch, topk, n),
    )?)
}

/// GCU `moe_gemm` with the same 7-argument signature as the upstream CUDA version.
///
/// On GCU the underlying kernel uses indexed MoE (input × expert-weights selected
/// by per-token expert indices). The `sorted_token_ids` tensor is repurposed as
/// the expert-index tensor expected by the GCU kernel.
///
/// `topk_weights`, `experts_ids`, `topk`, and `is_prefill` are accepted for API
/// compatibility but the actual weighting / reduction is done by the caller in
/// `candle-vllm`'s MoE layer (same as upstream).
#[cfg(feature = "gcu")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    _experts_ids: &Tensor,
    topk: usize,
    _is_prefill: bool,
) -> Result<Tensor> {
    use candle_core::DType;
    use half::{bf16, f16};
    let _ = topk_weights;
    let _ = topk;
    match input.dtype() {
        DType::F16 => indexed_moe_func::<f16>(input, weights, sorted_token_ids),
        DType::BF16 => indexed_moe_func::<bf16>(input, weights, sorted_token_ids),
        dt => {
            candle::bail!("indexed_moe is only supported for f16 and bf16 ({dt:?})")
        }
    }
}
