#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use crate::workspace::get_cutlass_workspace;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{Result, Tensor};

pub const NVFP4_BLOCK_SIZE: usize = 16;

/// Check if hardware FP4 (CUTLASS block-scaled tensor ops) is available.
/// Requires Blackwell SM100+ and the cutlass feature.
#[cfg(feature = "cuda")]
fn is_hardware_fp4_available(dev: &candle_core::Device) -> bool {
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

/// Check if FlashInfer-ported FP4 CUTLASS path is available.
/// Requires SM100+ and the flashinfer feature (which implies cutlass).
#[cfg(feature = "cuda")]
fn is_flashinfer_fp4_available(dev: &candle_core::Device) -> bool {
    if !cfg!(feature = "flashinfer") {
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
fn pad_to(val: usize, align: usize) -> usize {
    (val + align - 1) / align * align
}

/// Pre-swizzle NVFP4 weight scales from linear [N, K/16] layout to the CUTLASS
/// 128x4 block-interleaved layout. Call once at model load time and pass the
/// result to [`nvfp4_matmul`] via `weight_scale_swizzled` to avoid re-swizzling
/// on every inference call.
///
/// * `scale` - [N, K/16] U8 FP8 E4M3 block scales (linear / row-major)
/// * `n` - number of output channels (weight rows)
/// * `k` - full K dimension (not K/16)
///
/// Returns a U8 tensor of shape [N_padded, K_scale_padded] in swizzled layout.
#[allow(unused)]
pub fn swizzle_nvfp4_weight_scales(scale: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    let scale = if scale.is_contiguous() {
        scale.clone()
    } else {
        scale.contiguous()?
    };
    let dev = scale.device();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::Storage;

            let k_scale_cols = k / NVFP4_BLOCK_SIZE;
            let k_scale_padded = pad_to(k_scale_cols, 4);
            let n_padded = pad_to(n, 128);

            let swizzled = Tensor::zeros((n_padded, k_scale_padded), DType::U8, dev)?;

            {
                let (scale_s, _) = scale.storage_and_layout();
                let scale_ptr = match &*scale_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };

                let (sw_s, _) = swizzled.storage_and_layout();
                let sw_ptr = match &*sw_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };

                let stream = *cuda_dev.cu_stream() as i64;
                unsafe {
                    ffi::nvfp4_swizzle_weight_scales(
                        scale_ptr as *const std::ffi::c_void,
                        sw_ptr as *mut std::ffi::c_void,
                        n as i32,
                        k_scale_cols as i32,
                        n_padded as i32,
                        k_scale_padded as i32,
                        stream,
                    );
                }
            }

            Ok(swizzled)
        }
        _ => candle_core::bail!("swizzle_nvfp4_weight_scales: unsupported backend (need CUDA)"),
    }
}

/// NVFP4 linear: output = input @ weight^T [+ bias]
///
/// * `input` - [M, K] in F16/BF16
/// * `weight` - [N, K/2] packed U8 (2 FP4 E2M1 nibbles per byte)
/// * `scale` - [N, K/16] U8 FP8 E4M3 block scales (linear layout)
/// * `weight_global_scale` - scalar F32 weight-side global scale
///   (from `weight_scale_2` or `1/weight_global_scale` in the checkpoint)
/// * `input_scale` - scalar F32 activation-side global scale
///   (from `input_scale` or `input_global_scale` in the checkpoint, default 1.0)
///   Used by the hardware FP4 path to pre-scale activation block scales during
///   quantization and to compute the GEMM epilogue alpha = input_scale * weight_global_scale.
///   Ignored by the software path (activations stay in FP16/BF16).
/// * `bias` - Optional [N] in F16/BF16
/// * `weight_scale_swizzled` - Optional pre-swizzled weight scales from
///   [`swizzle_nvfp4_weight_scales`]. When provided, skips per-call swizzling.
///
/// On Blackwell (SM100+) with cutlass feature: uses hardware FP4 tensor cores
/// via CUTLASS block-scaled GEMM (quantizes activations to FP4 on-the-fly).
/// On older GPUs: uses software dequant path (LUT-based FP4 decode + FMA/WMMA).
///
/// Returns [M, N] in same dtype as input

/// Like [`nvfp4_matmul`] but accepts optional pre-swizzled weight scales to
/// avoid redundant per-call swizzling on the hardware FP4 path.
#[allow(unused)]
pub fn nvfp4_matmul(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
    input_scale: f32,
    bias: Option<&Tensor>,
    is_prefill: bool,
    weight_scale_swizzled: Option<&Tensor>,
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
        candle_core::bail!("nvfp4_matmul: expected input rank 2, got {:?}", input_dims);
    }

    let m = input_dims[0];
    let k = input_dims[1];
    let n = weight_dims[0];

    if k % NVFP4_BLOCK_SIZE != 0 {
        candle_core::bail!("nvfp4_matmul: K must be divisible by {NVFP4_BLOCK_SIZE}, got K={k}");
    }
    if weight_dims[1] != k / 2 {
        candle_core::bail!(
            "nvfp4_matmul: weight shape mismatch, expected [N, K/2]=[{}, {}], got {:?}",
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

            let use_flashinfer_fp4 = cfg!(feature = "flashinfer")
                && is_flashinfer_fp4_available(dev)
                && is_prefill
                && n % 32 == 0
                && k % 32 == 0;

            let use_hardware_fp4 = !use_flashinfer_fp4
                && cfg!(feature = "cutlass")
                && is_hardware_fp4_available(dev)
                && is_prefill
                && n % 32 == 0
                && k % 32 == 0;

            let output = Tensor::zeros((m, n), dtype, dev)?;
            let has_bias = bias.is_some();

            if use_flashinfer_fp4 || use_hardware_fp4 {
                #[cfg(feature = "cutlass")]
                {
                    let stream = *cuda_dev.cu_stream() as i64;

                    let m_padded = pad_to(m, 128);
                    let k_scale_cols = k / NVFP4_BLOCK_SIZE;
                    let k_scale_padded = pad_to(k_scale_cols, 4);
                    let n_padded = pad_to(n, 128);

                    let act_packed = Tensor::zeros((m, k / 2), DType::U8, dev)?;
                    let act_scales = Tensor::zeros((m_padded, k_scale_cols), DType::U8, dev)?;
                    let act_scales_swizzled =
                        Tensor::zeros((m_padded, k_scale_padded), DType::U8, dev)?;

                    let wscale_sw_owned;
                    let wscale_sw_ref = if let Some(preswizzled) = weight_scale_swizzled {
                        preswizzled
                    } else {
                        wscale_sw_owned =
                            Tensor::zeros((n_padded, k_scale_padded), DType::U8, dev)?;
                        &wscale_sw_owned
                    };

                    let input_scale_inv = if input_scale != 0.0 {
                        1.0 / input_scale
                    } else {
                        1.0
                    };
                    let alpha = input_scale * weight_global_scale;
                    let alpha_tensor = Tensor::new(&[alpha], dev)?;

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

                        let (wscale_sw_s, _) = wscale_sw_ref.storage_and_layout();
                        let wscale_sw_ptr =
                            cuda_ptr(&wscale_sw_s, DType::U8)? as *mut std::ffi::c_void;

                        let (output_s, _) = output.storage_and_layout();
                        let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                        let (alpha_s, _) = alpha_tensor.storage_and_layout();
                        let alpha_ptr = cuda_ptr(&alpha_s, DType::F32)? as *const f32;

                        unsafe {
                            match dtype {
                                DType::F16 => ffi::nvfp4_quantize_activation_f16(
                                    input_ptr,
                                    act_packed_ptr,
                                    act_scales_ptr,
                                    act_scales_sw_ptr,
                                    input_scale_inv,
                                    m as i32,
                                    k as i32,
                                    m_padded as i32,
                                    k_scale_padded as i32,
                                    stream,
                                ),
                                DType::BF16 => ffi::nvfp4_quantize_activation_bf16(
                                    input_ptr,
                                    act_packed_ptr,
                                    act_scales_ptr,
                                    act_scales_sw_ptr,
                                    input_scale_inv,
                                    m as i32,
                                    k as i32,
                                    m_padded as i32,
                                    k_scale_padded as i32,
                                    stream,
                                ),
                                _ => candle_core::bail!(
                                    "nvfp4_matmul: unsupported dtype {:?}",
                                    dtype
                                ),
                            }

                            if weight_scale_swizzled.is_none() {
                                ffi::nvfp4_swizzle_weight_scales(
                                    scale_ptr,
                                    wscale_sw_ptr,
                                    n as i32,
                                    k_scale_cols as i32,
                                    n_padded as i32,
                                    k_scale_padded as i32,
                                    stream,
                                );
                            }

                            let (ws_ptr, ws_bytes) = get_cutlass_workspace(cuda_dev, 0)?;
                            let ws_bytes = ws_bytes as i64;

                            if use_flashinfer_fp4 {
                                // FlashInfer-ported CUTLASS path (preferred on SM100+)
                                match dtype {
                                    DType::F16 => ffi::flashinfer_nvfp4_cutlass_gemm_f16(
                                        act_packed_ptr as *const std::ffi::c_void,
                                        weight_ptr,
                                        act_scales_sw_ptr as *const std::ffi::c_void,
                                        wscale_sw_ptr as *const std::ffi::c_void,
                                        alpha_ptr,
                                        output_ptr,
                                        m as i32,
                                        n as i32,
                                        k as i32,
                                        ws_ptr,
                                        ws_bytes,
                                        stream,
                                    ),
                                    DType::BF16 => ffi::flashinfer_nvfp4_cutlass_gemm_bf16(
                                        act_packed_ptr as *const std::ffi::c_void,
                                        weight_ptr,
                                        act_scales_sw_ptr as *const std::ffi::c_void,
                                        wscale_sw_ptr as *const std::ffi::c_void,
                                        alpha_ptr,
                                        output_ptr,
                                        m as i32,
                                        n as i32,
                                        k as i32,
                                        ws_ptr,
                                        ws_bytes,
                                        stream,
                                    ),
                                    _ => candle_core::bail!(
                                        "nvfp4_matmul: unsupported dtype {:?}",
                                        dtype
                                    ),
                                }
                            } else {
                                // Existing CUTLASS path (fallback when flashinfer not enabled)
                                match dtype {
                                    DType::F16 => ffi::nvfp4_cutlass_gemm_f16(
                                        act_packed_ptr as *const std::ffi::c_void,
                                        weight_ptr,
                                        act_scales_sw_ptr as *const std::ffi::c_void,
                                        wscale_sw_ptr as *const std::ffi::c_void,
                                        alpha_ptr,
                                        output_ptr,
                                        m as i32,
                                        n as i32,
                                        k as i32,
                                        ws_ptr,
                                        ws_bytes,
                                        stream,
                                    ),
                                    DType::BF16 => ffi::nvfp4_cutlass_gemm_bf16(
                                        act_packed_ptr as *const std::ffi::c_void,
                                        weight_ptr,
                                        act_scales_sw_ptr as *const std::ffi::c_void,
                                        wscale_sw_ptr as *const std::ffi::c_void,
                                        alpha_ptr,
                                        output_ptr,
                                        m as i32,
                                        n as i32,
                                        k as i32,
                                        ws_ptr,
                                        ws_bytes,
                                        stream,
                                    ),
                                    _ => candle_core::bail!(
                                        "nvfp4_matmul: unsupported dtype {:?}",
                                        dtype
                                    ),
                                }
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
                                ffi::nvfp4_matmul_smallm_f16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    stream,
                                );
                            }
                            DType::BF16 => {
                                ffi::nvfp4_matmul_smallm_bf16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    stream,
                                );
                            }
                            _ => candle_core::bail!(
                                "nvfp4_matmul CUDA: unsupported dtype {:?}",
                                dtype
                            ),
                        }
                    } else {
                        match dtype {
                            DType::F16 => {
                                ffi::nvfp4_matmul_f16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    stream,
                                );
                            }
                            DType::BF16 => {
                                ffi::nvfp4_matmul_bf16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    stream,
                                );
                            }
                            _ => candle_core::bail!(
                                "nvfp4_matmul CUDA: unsupported dtype {:?}",
                                dtype
                            ),
                        }
                    }
                }
            }

            Ok(output)
        }
        _ => candle_core::bail!("nvfp4_matmul: unsupported backend (need CUDA)"),
    }
}
