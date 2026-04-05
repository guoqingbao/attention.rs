#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "metal")]
use crate::metal_kernels;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{Result, Tensor};

pub const MXFP4_BLOCK_SIZE: usize = 32;

/// FlashInfer-accelerated MXFP4 fused MoE GEMM for Blackwell (SM100+).
/// Returns Ok(output) on success, or Err if not available / not supported.
#[cfg(all(feature = "cuda", feature = "flashinfer"))]
pub fn flashinfer_mxfp4_fused_moe(
    input: &Tensor,
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    gate_up_weights: &Tensor,
    gate_up_scales: &Tensor,
    down_weights: &Tensor,
    down_scales: &Tensor,
    num_tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<Tensor> {
    use candle_core::Storage;

    let dev = input.device();
    let dtype = input.dtype();

    let sm_version = crate::cuda_utils::sm_version(dev.as_cuda_device()?).unwrap_or(0) as usize;
    if sm_version < 100 {
        candle_core::bail!("flashinfer_mxfp4_fused_moe requires Blackwell (sm100+)");
    }

    let cuda_dev = dev.as_cuda_device()?;
    let stream = *cuda_dev.cu_stream() as i64;

    let input_dtype_code: i32 = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        _ => candle_core::bail!(
            "flashinfer_mxfp4_fused_moe: unsupported input dtype {:?}",
            dtype
        ),
    };

    let output = Tensor::zeros((num_tokens, hidden_size), dtype, dev)?;

    fn flashinfer_cuda_ptr(s: &Storage, dtype: DType) -> candle_core::Result<u64> {
        match s {
            Storage::Cuda(c) => match dtype {
                DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
                DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
                DType::U8 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                DType::U32 => Ok(*c.as_cuda_slice::<u32>()?.device_ptr()),
                DType::F32 => Ok(*c.as_cuda_slice::<f32>()?.device_ptr()),
                _ => candle_core::bail!("unsupported dtype {:?}", dtype),
            },
            _ => candle_core::bail!("tensor must be on CUDA"),
        }
    }

    {
        let (input_s, _) = input.storage_and_layout();
        let input_ptr = flashinfer_cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;

        let (topk_ids_s, _) = topk_ids.storage_and_layout();
        let topk_ids_ptr = flashinfer_cuda_ptr(&topk_ids_s, DType::U32)? as *const i32;

        let (topk_weights_s, _) = topk_weights.storage_and_layout();
        let topk_weights_ptr = flashinfer_cuda_ptr(&topk_weights_s, DType::F32)? as *const f32;

        let (gate_up_w_s, _) = gate_up_weights.storage_and_layout();
        let gate_up_w_ptr = flashinfer_cuda_ptr(&gate_up_w_s, DType::U8)? as *const u8;
        let (gate_up_s_s, _) = gate_up_scales.storage_and_layout();
        let gate_up_s_ptr = flashinfer_cuda_ptr(&gate_up_s_s, DType::U8)? as *const u8;
        let (down_w_s, _) = down_weights.storage_and_layout();
        let down_w_ptr = flashinfer_cuda_ptr(&down_w_s, DType::U8)? as *const u8;
        let (down_s_s, _) = down_scales.storage_and_layout();
        let down_s_ptr = flashinfer_cuda_ptr(&down_s_s, DType::U8)? as *const u8;
        let (output_s, _) = output.storage_and_layout();
        let output_ptr = flashinfer_cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

        let status = unsafe {
            ffi::flashinfer_fused_moe_mxfp4(
                input_ptr,
                topk_ids_ptr,
                topk_weights_ptr,
                gate_up_w_ptr,
                gate_up_s_ptr,
                down_w_ptr,
                down_s_ptr,
                output_ptr,
                num_tokens as i32,
                hidden_size as i32,
                intermediate_size as i32,
                num_experts as i32,
                top_k as i32,
                input_dtype_code,
                stream,
            )
        };

        if status != 0 {
            candle_core::bail!("flashinfer_fused_moe_mxfp4 returned error code {}", status);
        }
    }

    Ok(output)
}

/// MXFP4 linear: output = input @ weight^T [+ bias]
///
/// * `input` - [M, K] in F16/BF16
/// * `weight` - [N, K/2] packed U8 (2 FP4 nibbles per byte)
/// * `scale` - [N, K/32] U8 E8M0 scales
/// * `bias` - Optional [N] in F16/BF16
///
/// Returns [M, N] in same dtype as input
#[allow(unused)]
pub fn mxfp4_matmul(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weight = if weight.is_contiguous() {
        weight.clone()
    } else {
        weight.contiguous()?
    };
    let scale = if scale.is_contiguous() {
        scale.clone()
    } else {
        scale.contiguous()?
    };

    let input_dims = input.dims();
    let weight_dims = weight.dims();

    if input_dims.len() != 2 {
        candle_core::bail!("mxfp4_matmul: expected input rank 2, got {:?}", input_dims);
    }

    let m = input_dims[0];
    let k = input_dims[1];
    let n = weight_dims[0];

    if k % MXFP4_BLOCK_SIZE != 0 {
        candle_core::bail!("mxfp4_matmul: K must be divisible by {MXFP4_BLOCK_SIZE}, got K={k}");
    }
    if weight_dims[1] != k / 2 {
        candle_core::bail!(
            "mxfp4_matmul: weight shape mismatch, expected [N, K/2]=[{}, {}], got {:?}",
            n,
            k / 2,
            weight_dims
        );
    }

    let dev = input.device();
    let dtype = input.dtype();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::Storage;

            fn cuda_ptr(s: &Storage, dtype: DType) -> candle_core::Result<u64> {
                match s {
                    Storage::Cuda(c) => match dtype {
                        DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
                        DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
                        DType::U8 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                        _ => candle_core::bail!("unsupported dtype {:?}", dtype),
                    },
                    _ => candle_core::bail!("tensor must be on CUDA"),
                }
            }

            let output = Tensor::zeros((m, n), dtype, dev)?;
            let has_bias = bias.is_some();

            {
                let (input_s, _) = input.storage_and_layout();
                let (weight_s, _) = weight.storage_and_layout();
                let (scale_s, _) = scale.storage_and_layout();
                let (output_s, _) = output.storage_and_layout();

                let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;
                let weight_ptr = cuda_ptr(&weight_s, DType::U8)? as *const u8;
                let scale_ptr = cuda_ptr(&scale_s, DType::U8)? as *const u8;
                let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                let bias_ptr = if let Some(b) = bias {
                    let (b_s, _) = b.storage_and_layout();
                    cuda_ptr(&b_s, b.dtype())? as *const std::ffi::c_void
                } else {
                    std::ptr::null()
                };

                let stream = *cuda_dev.cu_stream() as i64;

                unsafe {
                    if m < 32 {
                        match dtype {
                            DType::F16 => {
                                ffi::mxfp4_matmul_smallm_f16(
                                    input_ptr, weight_ptr, scale_ptr, bias_ptr, output_ptr,
                                    m as i32, n as i32, k as i32, has_bias, stream,
                                );
                            }
                            DType::BF16 => {
                                ffi::mxfp4_matmul_smallm_bf16(
                                    input_ptr, weight_ptr, scale_ptr, bias_ptr, output_ptr,
                                    m as i32, n as i32, k as i32, has_bias, stream,
                                );
                            }
                            _ => candle_core::bail!(
                                "mxfp4_matmul CUDA: unsupported dtype {:?}",
                                dtype
                            ),
                        }
                    } else {
                        match dtype {
                            DType::F16 => {
                                ffi::mxfp4_matmul_wmma_f16(
                                    input_ptr, weight_ptr, scale_ptr, bias_ptr, output_ptr,
                                    m as i32, n as i32, k as i32, has_bias, stream,
                                );
                            }
                            DType::BF16 => {
                                ffi::mxfp4_matmul_wmma_bf16(
                                    input_ptr, weight_ptr, scale_ptr, bias_ptr, output_ptr,
                                    m as i32, n as i32, k as i32, has_bias, stream,
                                );
                            }
                            _ => candle_core::bail!(
                                "mxfp4_matmul CUDA: unsupported dtype {:?}",
                                dtype
                            ),
                        }
                    }
                }
            }

            Ok(output)
        }

        #[cfg(feature = "metal")]
        candle_core::Device::Metal(metal_dev) => {
            use candle_core::Storage;

            let command_buffer = metal_dev.command_buffer()?;
            let command_buffer_ref = command_buffer.as_ref();

            let output = Tensor::zeros((m, n), dtype, dev)?;

            {
                let (input_s, input_l) = input.storage_and_layout();
                let input_ms = match &*input_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("input must be metal"),
                };
                let (weight_s, weight_l) = weight.storage_and_layout();
                let weight_ms = match &*weight_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("weight must be metal"),
                };
                let (scale_s, scale_l) = scale.storage_and_layout();
                let scale_ms = match &*scale_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("scale must be metal"),
                };
                let (output_s, _output_l) = output.storage_and_layout();
                let output_ms = match &*output_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("output must be metal"),
                };

                let x = (
                    input_ms.buffer(),
                    input_l.start_offset() * dtype.size_in_bytes(),
                );
                let w = (
                    weight_ms.buffer(),
                    weight_l.start_offset() * weight.dtype().size_in_bytes(),
                );
                let sc = (
                    scale_ms.buffer(),
                    scale_l.start_offset() * scale.dtype().size_in_bytes(),
                );

                if let Some(bias) = bias {
                    let bias = if bias.is_contiguous() {
                        bias.clone()
                    } else {
                        bias.contiguous()?
                    };
                    let (bias_s, bias_l) = bias.storage_and_layout();
                    let bias_ms = match &*bias_s {
                        Storage::Metal(s) => s,
                        _ => candle_core::bail!("bias must be metal"),
                    };
                    let bias_buf = (
                        bias_ms.buffer(),
                        bias_l.start_offset() * bias.dtype().size_in_bytes(),
                    );

                    metal_kernels::call_mxfp4_matmul(
                        metal_dev.device(),
                        command_buffer_ref,
                        metal_kernels::Kernels::default(),
                        dtype,
                        x,
                        w,
                        sc,
                        bias_buf,
                        output_ms.buffer(),
                        m,
                        n,
                        k,
                        true,
                    )
                    .map_err(candle_core::Error::wrap)?;
                } else {
                    let dummy_bias = (input_ms.buffer(), 0usize);

                    metal_kernels::call_mxfp4_matmul(
                        metal_dev.device(),
                        command_buffer_ref,
                        metal_kernels::Kernels::default(),
                        dtype,
                        x,
                        w,
                        sc,
                        dummy_bias,
                        output_ms.buffer(),
                        m,
                        n,
                        k,
                        false,
                    )
                    .map_err(candle_core::Error::wrap)?;
                }
            }

            Ok(output)
        }
        _ => candle_core::bail!("mxfp4_matmul: unsupported backend (need CUDA or Metal)"),
    }
}

/// MXFP4 indexed MoE GEMM: for each (token, expert_slot), compute input @ weight[expert]^T [+ bias]
///
/// * `input` - [num_tokens, K] or [num_tokens, topk, K] in F16/BF16
/// * `weights` - [num_experts, N, K/2] packed U8
/// * `weight_scales` - [num_experts, N, K/32] U8 E8M0 scales
/// * `biases` - Optional [num_experts, N] in F16/BF16
/// * `indices` - [num_tokens, topk] U32 expert indices
///
/// Returns [num_tokens, topk, N]
#[allow(unused)]
pub fn mxfp4_moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    weight_scales: &Tensor,
    biases: Option<&Tensor>,
    indices: &Tensor,
) -> Result<Tensor> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weights = if weights.is_contiguous() {
        weights.clone()
    } else {
        weights.contiguous()?
    };
    let weight_scales = if weight_scales.is_contiguous() {
        weight_scales.clone()
    } else {
        weight_scales.contiguous()?
    };
    let indices = if indices.is_contiguous() {
        indices.clone()
    } else {
        indices.contiguous()?
    };

    let indices_dims = indices.dims();
    if indices_dims.len() != 2 {
        candle_core::bail!(
            "mxfp4_moe_gemm: expected indices rank 2 [num_tokens, topk], got {:?}",
            indices_dims
        );
    }
    let num_tokens = indices_dims[0];
    let topk = indices_dims[1];

    let input_dims = input.dims();
    let (k, input_has_topk_dim) = match input_dims {
        [t, kk] => {
            if *t != num_tokens {
                candle_core::bail!(
                    "mxfp4_moe_gemm: input/indices mismatch: input tokens={t}, indices tokens={num_tokens}"
                );
            }
            (*kk, false)
        }
        [t, tk, kk] => {
            if *t != num_tokens || *tk != topk {
                candle_core::bail!(
                    "mxfp4_moe_gemm: input/indices mismatch: input={input_dims:?}, indices={indices_dims:?}"
                );
            }
            (*kk, true)
        }
        _ => candle_core::bail!(
            "mxfp4_moe_gemm: expected input rank 2 or 3, got {:?}",
            input_dims
        ),
    };

    if k % MXFP4_BLOCK_SIZE != 0 {
        candle_core::bail!("mxfp4_moe_gemm: K must be divisible by {MXFP4_BLOCK_SIZE}, got K={k}");
    }

    let w_dims = weights.dims();
    if w_dims.len() != 3 {
        candle_core::bail!(
            "mxfp4_moe_gemm: expected weights rank 3 [E, N, K/2], got {:?}",
            w_dims
        );
    }
    let num_experts = w_dims[0];
    let n = w_dims[1];

    if w_dims[2] != k / 2 {
        candle_core::bail!(
            "mxfp4_moe_gemm: weights shape mismatch, expected [E, N, K/2]=[{}, {}, {}], got {:?}",
            num_experts,
            n,
            k / 2,
            w_dims
        );
    }

    let dev = input.device();
    let dtype = input.dtype();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::Storage;

            let has_bias = biases.is_some();
            let use_fused = num_tokens >= 32;
            let output = Tensor::zeros((num_tokens, topk, n), dtype, dev)?;

            {
                fn cuda_ptr_moe(s: &Storage, dtype: DType) -> candle_core::Result<u64> {
                    match s {
                        Storage::Cuda(c) => match dtype {
                            DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
                            DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
                            DType::U8 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                            DType::U32 => Ok(*c.as_cuda_slice::<u32>()?.device_ptr()),
                            _ => candle_core::bail!("unsupported dtype {:?}", dtype),
                        },
                        _ => candle_core::bail!("tensor must be on CUDA"),
                    }
                }

                let (input_s, _) = input.storage_and_layout();
                let (weights_s, _) = weights.storage_and_layout();
                let (scales_s, _) = weight_scales.storage_and_layout();
                let (indices_s, _) = indices.storage_and_layout();
                let (output_s, _) = output.storage_and_layout();

                let input_ptr = cuda_ptr_moe(&input_s, dtype)? as *const std::ffi::c_void;
                let weights_ptr = cuda_ptr_moe(&weights_s, DType::U8)? as *const u8;
                let scales_ptr = cuda_ptr_moe(&scales_s, DType::U8)? as *const u8;
                let indices_ptr = cuda_ptr_moe(&indices_s, DType::U32)? as *const u32;
                let output_ptr = cuda_ptr_moe(&output_s, dtype)? as *mut std::ffi::c_void;

                let biases_ptr = if let Some(b) = biases {
                    let (b_s, _) = b.storage_and_layout();
                    cuda_ptr_moe(&b_s, b.dtype())? as *const std::ffi::c_void
                } else {
                    std::ptr::null()
                };

                let stream = *cuda_dev.cu_stream() as i64;

                unsafe {
                    match dtype {
                        DType::F16 => {
                            if use_fused {
                                ffi::mxfp4_moe_grouped_gemm_wmma_f16(
                                    input_ptr,
                                    weights_ptr,
                                    scales_ptr,
                                    biases_ptr,
                                    indices_ptr,
                                    output_ptr,
                                    num_tokens as i32,
                                    topk as i32,
                                    num_experts as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    input_has_topk_dim,
                                    stream,
                                );
                            } else {
                                ffi::mxfp4_indexed_moe_gemm_f16(
                                    input_ptr,
                                    weights_ptr,
                                    scales_ptr,
                                    biases_ptr,
                                    indices_ptr,
                                    output_ptr,
                                    num_tokens as i32,
                                    topk as i32,
                                    num_experts as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    input_has_topk_dim,
                                    stream,
                                );
                            }
                        }
                        DType::BF16 => {
                            if use_fused {
                                ffi::mxfp4_moe_grouped_gemm_wmma_bf16(
                                    input_ptr,
                                    weights_ptr,
                                    scales_ptr,
                                    biases_ptr,
                                    indices_ptr,
                                    output_ptr,
                                    num_tokens as i32,
                                    topk as i32,
                                    num_experts as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    input_has_topk_dim,
                                    stream,
                                );
                            } else {
                                ffi::mxfp4_indexed_moe_gemm_bf16(
                                    input_ptr,
                                    weights_ptr,
                                    scales_ptr,
                                    biases_ptr,
                                    indices_ptr,
                                    output_ptr,
                                    num_tokens as i32,
                                    topk as i32,
                                    num_experts as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    input_has_topk_dim,
                                    stream,
                                );
                            }
                        }
                        _ => {
                            candle_core::bail!("mxfp4_moe_gemm CUDA: unsupported dtype {:?}", dtype)
                        }
                    }
                }
            }

            Ok(output)
        }

        #[cfg(feature = "metal")]
        candle_core::Device::Metal(metal_dev) => {
            use candle_core::Storage;

            let reuse_topk = !input_has_topk_dim && topk <= 8;

            let command_buffer = metal_dev.command_buffer()?;
            let command_buffer_ref = command_buffer.as_ref();

            let output = Tensor::zeros((num_tokens, topk, n), dtype, dev)?;

            {
                let (input_s, input_l) = input.storage_and_layout();
                let input_ms = match &*input_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("input must be metal"),
                };
                let (weights_s, weights_l) = weights.storage_and_layout();
                let weights_ms = match &*weights_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("weights must be metal"),
                };
                let (scales_s, scales_l) = weight_scales.storage_and_layout();
                let scales_ms = match &*scales_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("weight_scales must be metal"),
                };
                let (indices_s, indices_l) = indices.storage_and_layout();
                let indices_ms = match &*indices_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("indices must be metal"),
                };
                let (output_s, _output_l) = output.storage_and_layout();
                let output_ms = match &*output_s {
                    Storage::Metal(s) => s,
                    _ => candle_core::bail!("output must be metal"),
                };

                let x = (
                    input_ms.buffer(),
                    input_l.start_offset() * dtype.size_in_bytes(),
                );
                let w = (
                    weights_ms.buffer(),
                    weights_l.start_offset() * weights.dtype().size_in_bytes(),
                );
                let sc = (
                    scales_ms.buffer(),
                    scales_l.start_offset() * weight_scales.dtype().size_in_bytes(),
                );
                let idx = (
                    indices_ms.buffer(),
                    indices_l.start_offset() * indices.dtype().size_in_bytes(),
                );

                if let Some(biases) = biases {
                    let biases = if biases.is_contiguous() {
                        biases.clone()
                    } else {
                        biases.contiguous()?
                    };
                    let (bias_s, bias_l) = biases.storage_and_layout();
                    let bias_ms = match &*bias_s {
                        Storage::Metal(s) => s,
                        _ => candle_core::bail!("biases must be metal"),
                    };
                    let bias_buf = (
                        bias_ms.buffer(),
                        bias_l.start_offset() * biases.dtype().size_in_bytes(),
                    );

                    metal_kernels::call_mxfp4_moe_gemm(
                        metal_dev.device(),
                        command_buffer_ref,
                        metal_kernels::Kernels::default(),
                        dtype,
                        x,
                        w,
                        sc,
                        bias_buf,
                        idx,
                        output_ms.buffer(),
                        num_tokens,
                        topk,
                        num_experts,
                        n,
                        k,
                        true,
                        input_has_topk_dim,
                        reuse_topk,
                    )
                    .map_err(candle_core::Error::wrap)?;
                } else {
                    let dummy_biases = (input_ms.buffer(), 0usize);

                    metal_kernels::call_mxfp4_moe_gemm(
                        metal_dev.device(),
                        command_buffer_ref,
                        metal_kernels::Kernels::default(),
                        dtype,
                        x,
                        w,
                        sc,
                        dummy_biases,
                        idx,
                        output_ms.buffer(),
                        num_tokens,
                        topk,
                        num_experts,
                        n,
                        k,
                        false,
                        input_has_topk_dim,
                        reuse_topk,
                    )
                    .map_err(candle_core::Error::wrap)?;
                }
            }

            Ok(output)
        }
        _ => candle_core::bail!("mxfp4_moe_gemm: unsupported backend (need CUDA or Metal)"),
    }
}
