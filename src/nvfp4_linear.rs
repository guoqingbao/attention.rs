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

pub const NVFP4_BLOCK_SIZE: usize = 16;

/// Compute online input scale from the activation tensor's max absolute value.
///
/// Device-only variant: returns a CUDA `[2]` tensor
/// (`[input_scale, input_scale_inv]` where `input_scale = amax(|x|) / 6.0`).
/// Safe under CUDA graph capture — no host sync.
#[cfg(feature = "cuda")]
pub fn compute_online_input_scale_device(input: &Tensor) -> Result<Tensor> {
    use candle_core::{DType, Storage};

    let dev = input.device();
    let dtype = input.dtype();
    let num_elements = input.elem_count();

    let cuda_dev = dev.as_cuda_device()?;
    let stream = *cuda_dev.cu_stream() as i64;

    let result_tensor = Tensor::zeros((2,), DType::F32, dev)?;
    let (result_s, _) = result_tensor.storage_and_layout();
    let result_ptr = match &*result_s {
        Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
        _ => candle_core::bail!("result must be on CUDA"),
    };

    let (input_s, _) = input.storage_and_layout();
    let input_ptr = match &*input_s {
        Storage::Cuda(c) => match dtype {
            DType::F16 => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
            DType::BF16 => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
            _ => candle_core::bail!("unsupported dtype {:?}", dtype),
        },
        _ => candle_core::bail!("tensor must be on CUDA"),
    };

    unsafe {
        match dtype {
            DType::F16 => ffi::nvfp4_compute_online_input_scale_f16(
                input_ptr as *const std::ffi::c_void,
                result_ptr as *mut f32,
                num_elements as i32,
                stream,
            ),
            DType::BF16 => ffi::nvfp4_compute_online_input_scale_bf16(
                input_ptr as *const std::ffi::c_void,
                result_ptr as *mut f32,
                num_elements as i32,
                stream,
            ),
            _ => candle_core::bail!("unsupported dtype {:?}", dtype),
        }
    }
    drop(result_s);
    drop(input_s);

    Ok(result_tensor)
}

/// Compute online input scale from the activation tensor's max absolute value.
/// Returns (input_scale, input_scale_inv) where input_scale = amax(|x|) / 6.0.
///
/// Performs a D2H sync — **not** safe under CUDA graph capture. Prefer
/// [`compute_online_input_scale_device`] + [`resolve_online_hw_scales`] inside
/// graph-captured forwards.
#[cfg(feature = "cuda")]
pub fn compute_online_input_scale(input: &Tensor) -> Result<(f32, f32)> {
    let result_tensor = compute_online_input_scale_device(input)?;
    let result_vec = result_tensor.to_vec1::<f32>()?;
    Ok((result_vec[0], result_vec[1]))
}

/// Build device `alpha` and `sf_inv` scalars from an online `[scale, inv]`
/// tensor and the checkpoint-calibrated scale. Graph-capture safe (no D2H).
#[cfg(feature = "cuda")]
pub fn resolve_online_hw_scales(
    online: &Tensor,
    calibrated_input_scale: f32,
    weight_global_scale: f32,
) -> Result<(Tensor, Tensor)> {
    use candle_core::{DType, Storage};

    let dev = online.device();
    let cuda_dev = dev.as_cuda_device()?;
    let stream = *cuda_dev.cu_stream() as i64;

    let alpha_tensor = Tensor::zeros((1,), DType::F32, dev)?;
    let sf_scale_tensor = Tensor::zeros((1,), DType::F32, dev)?;

    let (online_s, _) = online.storage_and_layout();
    let online_ptr = match &*online_s {
        Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
        _ => candle_core::bail!("online scales must be on CUDA"),
    };
    let (alpha_s, _) = alpha_tensor.storage_and_layout();
    let alpha_ptr = match &*alpha_s {
        Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
        _ => candle_core::bail!("alpha must be on CUDA"),
    };
    let (sf_s, _) = sf_scale_tensor.storage_and_layout();
    let sf_ptr = match &*sf_s {
        Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
        _ => candle_core::bail!("sf_inv must be on CUDA"),
    };

    unsafe {
        ffi::nvfp4_resolve_online_scales(
            online_ptr as *const f32,
            calibrated_input_scale,
            weight_global_scale,
            alpha_ptr as *mut f32,
            sf_ptr as *mut f32,
            stream,
        );
    }
    drop(online_s);
    drop(alpha_s);
    drop(sf_s);

    Ok((alpha_tensor, sf_scale_tensor))
}

/// Check if hardware FP4 (CUTLASS block-scaled tensor ops) is available.
/// Requires Blackwell SM100+ and the cutlass feature.
#[cfg(feature = "cuda")]
fn is_hardware_fp4_available(dev: &candle_core::Device) -> bool {
    if !cfg!(feature = "cutlass") {
        return false;
    }
    if let Ok(cuda_dev) = dev.as_cuda_device() {
        let sm = crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0);
        // SM100+ hardware NVFP4, including SM120/SM121 tensor-core FP4.
        sm >= 100
    } else {
        false
    }
}

/// Check if FlashInfer-ported FP4 CUTLASS path is available.
/// Requires SM100+ (including SM120/SM121) and the flashinfer feature.
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
#[cfg(feature = "cuda")]
fn pad_to(val: usize, align: usize) -> usize {
    (val + align - 1) / align * align
}

/// Pre-swizzle NVFP4 weight scales from linear layout to the CUTLASS
/// 128×4 block-interleaved layout required by Blackwell hardware FP4 tensor
/// cores. Call once at model load time to avoid re-swizzling on every forward.
///
/// Accepts rank-2 `[N, K/16]` (single linear layer) or rank-3 `[E, N, K/16]`
/// (batched / MoE). All dimensions are read from the tensor shape, so
/// sharded tensors work correctly without the caller adjusting sizes.
///
/// Returns a swizzled U8 tensor with the same rank (with N and K/16 padded).
#[allow(unused)]
pub fn swizzle_nvfp4_weight_scales(scale: &Tensor) -> Result<Tensor> {
    let scale = if scale.is_contiguous() {
        scale.clone()
    } else {
        scale.contiguous()?
    };

    let dims = scale.dims();
    let (num_slices, n, k_scale_cols) = match dims {
        [n, ksc] => (1, *n, *ksc),
        [e, n, ksc] => (*e, *n, *ksc),
        _ => candle_core::bail!(
            "swizzle_nvfp4_weight_scales: expected rank 2 [N, K/16] or 3 [E, N, K/16], got {:?}",
            dims
        ),
    };
    let is_batched = dims.len() == 3;
    let dev = scale.device();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::Storage;

            let k_scale_padded = pad_to(k_scale_cols, 4);
            let n_padded = pad_to(n, 128);

            let swizzled = if is_batched {
                Tensor::zeros((num_slices, n_padded, k_scale_padded), DType::U8, dev)?
            } else {
                Tensor::zeros((n_padded, k_scale_padded), DType::U8, dev)?
            };

            {
                let (scale_s, _) = scale.storage_and_layout();
                let scale_base = match &*scale_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };

                let (sw_s, _) = swizzled.storage_and_layout();
                let sw_base = match &*sw_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };

                let stream = *cuda_dev.cu_stream() as i64;
                for e in 0..num_slices {
                    let src_offset = (e * n * k_scale_cols) as u64;
                    let dst_offset = (e * n_padded * k_scale_padded) as u64;
                    unsafe {
                        ffi::nvfp4_swizzle_weight_scales(
                            (scale_base + src_offset) as *const std::ffi::c_void,
                            (sw_base + dst_offset) as *mut std::ffi::c_void,
                            n as i32,
                            k_scale_cols as i32,
                            n_padded as i32,
                            k_scale_padded as i32,
                            stream,
                        );
                    }
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
/// On SM100+ with cutlass/flashinfer: prefill uses hardware FP4 tensor cores
/// via CUTLASS block-scaled GEMM (quantizes activations to FP4 on-the-fly).
/// SM120/SM121 prefill uses FlashInfer `fp4_quantize` + `mm_fp4`. Decode uses
/// the same hardware W4A4 path by default (`XINFER_NVFP4_HW_DECODE=1`);
/// set `XINFER_NVFP4_HW_DECODE=0` for software small-M GEMV.
/// On Hopper and below: software dequant (LUT-based FP4 decode + FMA/WMMA).
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
    weight_row_scales: Option<&Tensor>,
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
                        DType::U8 | DType::F8E4M3 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                        DType::F32 => Ok(*c.as_cuda_slice::<f32>()?.device_ptr()),
                        _ => candle_core::bail!("unsupported dtype {:?}", dtype),
                    },
                    _ => candle_core::bail!("tensor must be on CUDA"),
                }
            }

            // Hardware W4A4 for prefill; decode too unless XINFER_NVFP4_HW_DECODE=0.
            let sm = crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0);
            let use_hw_gemm = is_prefill || crate::nvfp4_hw_decode();
            let use_flashinfer_fp4 = use_hw_gemm
                && cfg!(feature = "flashinfer")
                && is_flashinfer_fp4_available(dev)
                && n % 32 == 0
                && k % 32 == 0;
            // SM120/SM121: FlashInfer fp4_quantize (SWIZZLED_128x4, e4m3Max=448).
            let use_flashinfer_quant = use_flashinfer_fp4 && sm >= 120;

            let use_hardware_fp4 = use_hw_gemm
                && !use_flashinfer_fp4
                && cfg!(feature = "cutlass")
                && is_hardware_fp4_available(dev)
                && n % 32 == 0
                && k % 32 == 0;

            let output = Tensor::zeros((m, n), dtype, dev)?;
            let has_bias = bias.is_some();

            if use_flashinfer_fp4 || use_hardware_fp4 {
                #[cfg(feature = "cutlass")]
                {
                    use crate::workspace::get_nvfp4_hw_scratch;

                    let stream = *cuda_dev.cu_stream() as i64;

                    let m_padded = pad_to(m, 128);
                    let k_scale_cols = k / NVFP4_BLOCK_SIZE;
                    let k_scale_padded = pad_to(k_scale_cols, 4);
                    let n_padded = pad_to(n, 128);

                    let act_packed_bytes = m * (k / 2);
                    let act_scales_bytes = m_padded * k_scale_cols;
                    let act_scales_swizzled_bytes = m_padded * k_scale_padded;
                    let wscale_bytes = if weight_scale_swizzled.is_some() {
                        0
                    } else {
                        n_padded * k_scale_padded
                    };
                    let scratch = get_nvfp4_hw_scratch(
                        cuda_dev,
                        act_packed_bytes,
                        act_scales_bytes,
                        act_scales_swizzled_bytes,
                        wscale_bytes,
                    )?;

                    let act_packed_ptr = scratch.act_packed;
                    let act_scales_ptr = scratch.act_scales;
                    let act_scales_sw_ptr = scratch.act_scales_swizzled;

                    // Default: checkpoint-calibrated input_scale (ModelOpt joint
                    // PTQ). Optional XINFER_NVFP4_ONLINE_SCALE=1 raises the scale
                    // to max(online_amax/6, calibrated) to avoid FP4 clipping.
                    // Online path stays fully on-device (no to_vec1) so CUDA graph
                    // capture/replay remain valid.
                    let (alpha_ptr, sf_scale_ptr) = if crate::nvfp4_online_scale() {
                        // Compute online amax/6 directly into domain-stable scratch
                        // (avoids ephemeral Tensor::zeros under CUDA graph capture).
                        let (input_s, _) = input.storage_and_layout();
                        let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;
                        let num_elements = input.elem_count();
                        unsafe {
                            match dtype {
                                DType::F16 => ffi::nvfp4_compute_online_input_scale_f16(
                                    input_ptr,
                                    scratch.online_scale,
                                    num_elements as i32,
                                    stream,
                                ),
                                DType::BF16 => ffi::nvfp4_compute_online_input_scale_bf16(
                                    input_ptr,
                                    scratch.online_scale,
                                    num_elements as i32,
                                    stream,
                                ),
                                _ => candle_core::bail!(
                                    "nvfp4 online scale: unsupported dtype {:?}",
                                    dtype
                                ),
                            }
                            ffi::nvfp4_resolve_online_scales(
                                scratch.online_scale,
                                input_scale,
                                weight_global_scale,
                                scratch.alpha,
                                scratch.sf_scale,
                                stream,
                            );
                        }
                        drop(input_s);
                        (scratch.alpha as *const f32, scratch.sf_scale as *const f32)
                    } else {
                        use candle_core::cuda_backend::cudarc::driver::result;
                        let (hw_input_scale, hw_input_scale_inv) = if input_scale > 1e-12 {
                            (input_scale, 1.0 / input_scale)
                        } else {
                            (1.0, 1.0)
                        };
                        let alpha = hw_input_scale * weight_global_scale;
                        // Host→device scalar writes are graph-capture safe and keep
                        // alpha/sf addresses in the domain-scoped scratch pool.
                        unsafe {
                            result::memcpy_htod_async(
                                scratch.alpha as u64,
                                &[alpha],
                                *cuda_dev.cu_stream(),
                            )
                            .map_err(candle_core::Error::wrap)?;
                            result::memcpy_htod_async(
                                scratch.sf_scale as u64,
                                &[hw_input_scale_inv],
                                *cuda_dev.cu_stream(),
                            )
                            .map_err(candle_core::Error::wrap)?;
                        }
                        (scratch.alpha as *const f32, scratch.sf_scale as *const f32)
                    };

                    {
                        let (input_s, _) = input.storage_and_layout();
                        let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;

                        let (weight_s, _) = weight.storage_and_layout();
                        let weight_ptr =
                            cuda_ptr(&weight_s, weight.dtype())? as *const std::ffi::c_void;

                        let (scale_s, _) = scale.storage_and_layout();
                        let scale_ptr =
                            cuda_ptr(&scale_s, scale.dtype())? as *const std::ffi::c_void;

                        let wscale_sw_ptr = if let Some(preswizzled) = weight_scale_swizzled {
                            let (wscale_sw_s, _) = preswizzled.storage_and_layout();
                            let ptr = cuda_ptr(&wscale_sw_s, DType::U8)? as *mut std::ffi::c_void;
                            drop(wscale_sw_s);
                            ptr
                        } else {
                            scratch.wscale_swizzled
                        };

                        let (output_s, _) = output.storage_and_layout();
                        let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                        unsafe {
                            if use_flashinfer_quant {
                                match dtype {
                                    DType::F16 => ffi::flashinfer_nvfp4_quantize_activation_f16(
                                        input_ptr,
                                        act_packed_ptr,
                                        act_scales_sw_ptr,
                                        sf_scale_ptr,
                                        m as i32,
                                        k as i32,
                                        stream,
                                    ),
                                    DType::BF16 => ffi::flashinfer_nvfp4_quantize_activation_bf16(
                                        input_ptr,
                                        act_packed_ptr,
                                        act_scales_sw_ptr,
                                        sf_scale_ptr,
                                        m as i32,
                                        k as i32,
                                        stream,
                                    ),
                                    _ => candle_core::bail!(
                                        "nvfp4_matmul: unsupported dtype {:?}",
                                        dtype
                                    ),
                                }
                            } else {
                                match dtype {
                                    DType::F16 => ffi::nvfp4_quantize_activation_f16(
                                        input_ptr,
                                        act_packed_ptr,
                                        act_scales_ptr,
                                        act_scales_sw_ptr,
                                        sf_scale_ptr,
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
                                        sf_scale_ptr,
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
                #[cfg(not(feature = "cutlass"))]
                {
                    let _ = (use_flashinfer_fp4, use_hardware_fp4, use_flashinfer_quant);
                    candle_core::bail!("nvfp4 hardware path requires the cutlass feature");
                }
            } else {
                // Software dequant path (existing kernels)
                let (input_s, _) = input.storage_and_layout();
                let (weight_s, _) = weight.storage_and_layout();
                let (scale_s, _) = scale.storage_and_layout();
                let (output_s, _) = output.storage_and_layout();

                let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;
                let weight_ptr = cuda_ptr(&weight_s, weight.dtype())? as *const u8;
                let scale_ptr = cuda_ptr(&scale_s, scale.dtype())? as *const u8;
                let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;

                let bias_ptr = if let Some(b) = bias {
                    let (b_s, _) = b.storage_and_layout();
                    cuda_ptr(&b_s, b.dtype())? as *const std::ffi::c_void
                } else {
                    std::ptr::null()
                };

                let row_scales_ptr: *const f32 = match weight_row_scales {
                    Some(rs) => {
                        let (rs_s, _) = rs.storage_and_layout();
                        cuda_ptr(&rs_s, rs.dtype())? as *const f32
                    }
                    None => std::ptr::null(),
                };

                let stream = *cuda_dev.cu_stream() as i64;
                let force_lut = crate::nvfp4_force_lut();

                unsafe {
                    if m < 32 {
                        match dtype {
                            DType::F16 => {
                                ffi::nvfp4_matmul_smallm_f16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    row_scales_ptr,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                );
                            }
                            DType::BF16 => {
                                ffi::nvfp4_matmul_smallm_bf16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    row_scales_ptr,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
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
                                    row_scales_ptr,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
                                    stream,
                                );
                            }
                            DType::BF16 => {
                                ffi::nvfp4_matmul_bf16(
                                    input_ptr,
                                    weight_ptr,
                                    scale_ptr,
                                    weight_global_scale,
                                    row_scales_ptr,
                                    bias_ptr,
                                    output_ptr,
                                    m as i32,
                                    n as i32,
                                    k as i32,
                                    has_bias,
                                    force_lut,
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
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(metal_dev) => {
            use candle_core::Storage;

            let output = Tensor::zeros((m, n), dtype, dev)?;

            let command_buffer = metal_dev.command_buffer()?;
            let command_buffer_ref = command_buffer.as_ref();

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

                if let Some(b) = bias {
                    let b = if b.is_contiguous() {
                        b.clone()
                    } else {
                        b.contiguous()?
                    };
                    let (bias_s, bias_l) = b.storage_and_layout();
                    let bias_ms = match &*bias_s {
                        Storage::Metal(s) => s,
                        _ => candle_core::bail!("bias must be metal"),
                    };
                    let bias_buf = (
                        bias_ms.buffer(),
                        bias_l.start_offset() * b.dtype().size_in_bytes(),
                    );

                    metal_kernels::call_nvfp4_matmul(
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
                        weight_global_scale,
                        true,
                    )
                    .map_err(candle_core::Error::wrap)?;
                } else {
                    let dummy_bias = (input_ms.buffer(), 0usize);

                    metal_kernels::call_nvfp4_matmul(
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
                        weight_global_scale,
                        false,
                    )
                    .map_err(candle_core::Error::wrap)?;
                }
            }

            Ok(output)
        }
        _ => candle_core::bail!("nvfp4_matmul: unsupported backend (need CUDA or Metal)"),
    }
}

/// Repack MLX NVFP4 weights from U32 to U8 on GPU.
///
/// MLX stores FP4 E2M1 weights as U32 (8 nibbles per U32 in little-endian).
/// Our NVFP4 GEMM kernels expect U8 (2 nibbles per byte). This function
/// reinterprets the bytes on-device without CPU round-trip.
///
/// * `weight_u32` - `[rows, cols]` U32 tensor on GPU
/// * Returns `[rows, cols * 4]` U8 tensor on GPU
#[allow(unused)]
pub fn mlx_repack_u32_to_u8(weight_u32: &Tensor) -> Result<Tensor> {
    let weight_u32 = if weight_u32.is_contiguous() {
        weight_u32.clone()
    } else {
        weight_u32.contiguous()?
    };
    let dims = weight_u32.dims();
    if dims.len() != 2 {
        candle_core::bail!("mlx_repack_u32_to_u8: expected rank 2, got {:?}", dims);
    }
    let rows = dims[0];
    let u32_cols = dims[1];
    let u8_cols = u32_cols * 4;
    let dev = weight_u32.device();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::cuda_backend::cudarc::driver::DevicePtr;
            use candle_core::Storage;

            let output = Tensor::zeros((rows, u8_cols), candle_core::DType::U8, dev)?;
            let stream = *cuda_dev.cu_stream() as i64;

            {
                let (in_s, _) = weight_u32.storage_and_layout();
                let in_ptr = match &*in_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u32>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };
                let (out_s, _) = output.storage_and_layout();
                let out_ptr = match &*out_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("tensor must be on CUDA"),
                };
                unsafe {
                    ffi::mlx_nvfp4_repack_u32_to_u8(
                        in_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        rows as i32,
                        u32_cols as i32,
                        stream,
                    );
                }
            }
            Ok(output)
        }
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(_metal_dev) => {
            // Metal path: reinterpret U32 bytes as U8 via compute kernel
            mlx_repack_u32_to_u8_metal(&weight_u32, rows, u32_cols)
        }
        _ => candle_core::bail!("mlx_repack_u32_to_u8: unsupported backend (need CUDA or Metal)"),
    }
}

/// Dequantize MLX NVFP4 quantized embeddings on GPU.
///
/// * `weight_u32` - `[vocab_size, hidden_size/8]` U32 packed FP4 weights on GPU
/// * `scales` - `[vocab_size, hidden_size/16]` U8 FP8 E4M3 block scales on GPU
/// * `vocab_size` - vocabulary size
/// * `hidden_size` - embedding dimension
/// * `dtype` - output dtype (F16 or BF16)
///
/// Returns `[vocab_size, hidden_size]` dequantized embedding tensor
#[allow(unused)]
pub fn mlx_dequant_embedding(
    weight_u32: &Tensor,
    scales: &Tensor,
    vocab_size: usize,
    hidden_size: usize,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let weight_u32 = if weight_u32.is_contiguous() {
        weight_u32.clone()
    } else {
        weight_u32.contiguous()?
    };
    let scales = if scales.is_contiguous() {
        scales.clone()
    } else {
        scales.contiguous()?
    };
    let dev = weight_u32.device();

    match dev {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(cuda_dev) => {
            use candle_core::cuda_backend::cudarc::driver::DevicePtr;
            use candle_core::Storage;

            let output = Tensor::zeros((vocab_size, hidden_size), dtype, dev)?;
            let stream = *cuda_dev.cu_stream() as i64;

            {
                let (w_s, _) = weight_u32.storage_and_layout();
                let w_ptr = match &*w_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u32>()?.device_ptr(),
                    _ => candle_core::bail!("weight must be on CUDA"),
                };
                let (s_s, _) = scales.storage_and_layout();
                let s_ptr = match &*s_s {
                    Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
                    _ => candle_core::bail!("scales must be on CUDA"),
                };
                let (o_s, _) = output.storage_and_layout();
                let o_ptr = match &*o_s {
                    Storage::Cuda(c) => match dtype {
                        candle_core::DType::F16 => *c.as_cuda_slice::<half::f16>()?.device_ptr(),
                        candle_core::DType::BF16 => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
                        _ => candle_core::bail!("output dtype must be F16 or BF16"),
                    },
                    _ => candle_core::bail!("output must be on CUDA"),
                };
                unsafe {
                    match dtype {
                        candle_core::DType::F16 => {
                            ffi::mlx_nvfp4_dequant_embedding_f16(
                                w_ptr as *const std::ffi::c_void,
                                s_ptr as *const std::ffi::c_void,
                                o_ptr as *mut std::ffi::c_void,
                                vocab_size as i32,
                                hidden_size as i32,
                                stream,
                            );
                        }
                        candle_core::DType::BF16 => {
                            ffi::mlx_nvfp4_dequant_embedding_bf16(
                                w_ptr as *const std::ffi::c_void,
                                s_ptr as *const std::ffi::c_void,
                                o_ptr as *mut std::ffi::c_void,
                                vocab_size as i32,
                                hidden_size as i32,
                                stream,
                            );
                        }
                        _ => candle_core::bail!(
                            "mlx_dequant_embedding: unsupported dtype {:?}",
                            dtype
                        ),
                    }
                }
            }
            Ok(output)
        }
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(_metal_dev) => {
            mlx_dequant_embedding_metal(&weight_u32, &scales, vocab_size, hidden_size, dtype)
        }
        _ => candle_core::bail!("mlx_dequant_embedding: unsupported backend (need CUDA or Metal)"),
    }
}

#[cfg(feature = "metal")]
fn mlx_repack_u32_to_u8_metal(weight_u32: &Tensor, rows: usize, u32_cols: usize) -> Result<Tensor> {
    let u8_cols = u32_cols * 4;
    let dev = weight_u32.device();
    let output = Tensor::zeros((rows, u8_cols), candle_core::DType::U8, dev)?;

    let metal_dev = dev.as_metal_device()?;
    let command_buffer = metal_dev.command_buffer()?;
    let command_buffer_ref = command_buffer.as_ref();

    {
        use candle_core::Storage;
        let (in_s, in_l) = weight_u32.storage_and_layout();
        let in_ms = match &*in_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("tensor must be on Metal"),
        };
        let (out_s, _out_l) = output.storage_and_layout();
        let out_ms = match &*out_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("tensor must be on Metal"),
        };

        metal_kernels::call_mlx_nvfp4_repack(
            metal_dev.device(),
            command_buffer_ref,
            metal_kernels::Kernels::default(),
            (in_ms.buffer(), in_l.start_offset() * 4),
            out_ms.buffer(),
            rows,
            u32_cols,
        )
        .map_err(candle_core::Error::wrap)?;
    }
    Ok(output)
}

#[cfg(feature = "metal")]
fn mlx_dequant_embedding_metal(
    weight_u32: &Tensor,
    scales: &Tensor,
    vocab_size: usize,
    hidden_size: usize,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let dev = weight_u32.device();
    let output = Tensor::zeros((vocab_size, hidden_size), dtype, dev)?;

    let metal_dev = dev.as_metal_device()?;
    let command_buffer = metal_dev.command_buffer()?;
    let command_buffer_ref = command_buffer.as_ref();

    {
        use candle_core::Storage;
        let (w_s, w_l) = weight_u32.storage_and_layout();
        let w_ms = match &*w_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("weight must be on Metal"),
        };
        let (s_s, s_l) = scales.storage_and_layout();
        let s_ms = match &*s_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("scales must be on Metal"),
        };
        let (o_s, _o_l) = output.storage_and_layout();
        let o_ms = match &*o_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("output must be on Metal"),
        };

        metal_kernels::call_mlx_nvfp4_dequant_embedding(
            metal_dev.device(),
            command_buffer_ref,
            metal_kernels::Kernels::default(),
            dtype,
            (w_ms.buffer(), w_l.start_offset() * 4),
            (s_ms.buffer(), s_l.start_offset()),
            o_ms.buffer(),
            vocab_size,
            hidden_size,
        )
        .map_err(candle_core::Error::wrap)?;
    }
    Ok(output)
}
