#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "metal")]
use crate::metal_kernels;
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use crate::workspace::get_cutlass_workspace;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{Result, Tensor};

pub const MXFP4_BLOCK_SIZE: usize = 32;

/// Check if hardware MXFP4 (CUTLASS block-scaled tensor ops with Mxf8f6f4 schedule) is available.
/// Requires Blackwell SM100+ and the cutlass feature.
#[cfg(feature = "cuda")]
fn is_hardware_mxfp4_available(dev: &candle_core::Device) -> bool {
    if !cfg!(feature = "cutlass") {
        return false;
    }
    if let Ok(cuda_dev) = dev.as_cuda_device() {
        let sm = crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0);
        sm >= 100
    } else {
        false
    }
}

/// Pad dimension up to the nearest multiple of `align`.
#[cfg(feature = "cuda")]
fn pad_to(val: usize, align: usize) -> usize {
    (val + align - 1) / align * align
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
                        DType::F32 => Ok(*c.as_cuda_slice::<f32>()?.device_ptr()),
                        _ => candle_core::bail!("unsupported dtype {:?}", dtype),
                    },
                    _ => candle_core::bail!("tensor must be on CUDA"),
                }
            }

            let use_hardware_mxfp4 =
                cfg!(feature = "cutlass") && is_hardware_mxfp4_available(dev) && m >= 32;

            let output = Tensor::zeros((m, n), dtype, dev)?;
            let has_bias = bias.is_some();

            if use_hardware_mxfp4 {
                #[cfg(all(feature = "cuda", feature = "cutlass"))]
                {
                    // Hardware MXFP4 path: quantize activations to FP4+E8M0 -> CUTLASS Mxf8f6f4 GEMM
                    let stream = *cuda_dev.cu_stream() as i64;

                    let m_padded = pad_to(m, 128);
                    let k_scale_cols = k / MXFP4_BLOCK_SIZE;
                    let k_scale_padded = pad_to(k_scale_cols, 4);
                    let n_padded = pad_to(n, 128);

                    let act_packed = Tensor::zeros((m, k / 2), DType::U8, dev)?;
                    let act_scales = Tensor::zeros((m_padded, k_scale_cols), DType::U8, dev)?;
                    let act_scales_swizzled =
                        Tensor::zeros((m_padded, k_scale_padded), DType::U8, dev)?;

                    let weight_scales_swizzled =
                        Tensor::zeros((n_padded, k_scale_padded), DType::U8, dev)?;

                    {
                        let (input_s, _) = input.storage_and_layout();
                        let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;

                        let (act_packed_s, _) = act_packed.storage_and_layout();
                        let act_packed_ptr =
                            cuda_ptr(&act_packed_s, DType::U8)? as *mut std::ffi::c_void;

                        let (act_scales_s, _) = act_scales.storage_and_layout();
                        let act_scales_ptr =
                            cuda_ptr(&act_scales_s, DType::U8)? as *mut std::ffi::c_void;

                        let (act_scales_sw_s, _) = act_scales_swizzled.storage_and_layout();
                        let act_scales_sw_ptr =
                            cuda_ptr(&act_scales_sw_s, DType::U8)? as *mut std::ffi::c_void;

                        let (weight_s, _) = weight.storage_and_layout();
                        let weight_ptr = cuda_ptr(&weight_s, DType::U8)? as *const std::ffi::c_void;

                        let (scale_s, _) = scale.storage_and_layout();
                        let scale_ptr = cuda_ptr(&scale_s, DType::U8)? as *const std::ffi::c_void;

                        let (wscale_sw_s, _) = weight_scales_swizzled.storage_and_layout();
                        let wscale_sw_ptr =
                            cuda_ptr(&wscale_sw_s, DType::U8)? as *mut std::ffi::c_void;

                        let (output_s, _) = output.storage_and_layout();
                        let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                        unsafe {
                            // Step 1: Quantize activations to FP4 with E8M0 block scales
                            match dtype {
                                DType::F16 => ffi::mxfp4_quantize_activation_f16(
                                    input_ptr,
                                    act_packed_ptr,
                                    act_scales_ptr,
                                    act_scales_sw_ptr,
                                    m as i32,
                                    k as i32,
                                    m_padded as i32,
                                    k_scale_padded as i32,
                                    stream,
                                ),
                                DType::BF16 => ffi::mxfp4_quantize_activation_bf16(
                                    input_ptr,
                                    act_packed_ptr,
                                    act_scales_ptr,
                                    act_scales_sw_ptr,
                                    m as i32,
                                    k as i32,
                                    m_padded as i32,
                                    k_scale_padded as i32,
                                    stream,
                                ),
                                _ => candle_core::bail!(
                                    "mxfp4_matmul: unsupported dtype {:?}",
                                    dtype
                                ),
                            }

                            // Step 2: Swizzle weight E8M0 scales to CUTLASS layout
                            ffi::mxfp4_swizzle_weight_scales_e8m0(
                                scale_ptr,
                                wscale_sw_ptr,
                                n as i32,
                                k_scale_cols as i32,
                                n_padded as i32,
                                k_scale_padded as i32,
                                stream,
                            );

                            // Step 3: CUTLASS MXFP4 GEMM
                            let (ws_ptr, ws_bytes) = get_cutlass_workspace(cuda_dev, 0)?;
                            let ws_bytes = ws_bytes as i64;

                            match dtype {
                                DType::F16 => ffi::mxfp4_cutlass_gemm_f16(
                                    act_packed_ptr as *const std::ffi::c_void,
                                    weight_ptr,
                                    act_scales_sw_ptr as *const std::ffi::c_void,
                                    wscale_sw_ptr as *const std::ffi::c_void,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    ws_ptr,
                                    ws_bytes,
                                    stream,
                                ),
                                DType::BF16 => ffi::mxfp4_cutlass_gemm_bf16(
                                    act_packed_ptr as *const std::ffi::c_void,
                                    weight_ptr,
                                    act_scales_sw_ptr as *const std::ffi::c_void,
                                    wscale_sw_ptr as *const std::ffi::c_void,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    ws_ptr,
                                    ws_bytes,
                                    stream,
                                ),
                                _ => candle_core::bail!(
                                    "mxfp4_matmul: unsupported dtype {:?}",
                                    dtype
                                ),
                            }
                        }
                    }

                    if let Some(b) = bias {
                        return Ok(output.broadcast_add(b)?);
                    }
                }
            } else {
                // Software dequant path (existing kernels)
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
