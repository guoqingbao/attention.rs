#[cfg(feature = "cuda")]
use crate::kernels::ffi;
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

/// Pad dimension up to the nearest multiple of `align`.
#[cfg(feature = "cuda")]
fn pad_to(val: usize, align: usize) -> usize {
    (val + align - 1) / align * align
}

/// NVFP4 linear: output = input @ weight^T [+ bias]
///
/// * `input` - [M, K] in F16/BF16
/// * `weight` - [N, K/2] packed U8 (2 FP4 E2M1 nibbles per byte)
/// * `scale` - [N, K/16] U8 FP8 E4M3 block scales
/// * `weight_global_scale` - scalar F32 weight-side global scale
///   (from `weight_scale_2` or `1/weight_global_scale` in the checkpoint)
/// * `input_scale` - scalar F32 activation-side global scale
///   (from `input_scale` or `input_global_scale` in the checkpoint, default 1.0)
///   Used by the hardware FP4 path to pre-scale activation block scales during
///   quantization and to compute the GEMM epilogue alpha = input_scale * weight_global_scale.
///   Ignored by the software path (activations stay in FP16/BF16).
/// * `bias` - Optional [N] in F16/BF16
///
/// On Blackwell (SM100+) with cutlass feature: uses hardware FP4 tensor cores
/// via CUTLASS block-scaled GEMM (quantizes activations to FP4 on-the-fly).
/// On older GPUs: uses software dequant path (LUT-based FP4 decode + FMA/WMMA).
///
/// Returns [M, N] in same dtype as input
#[allow(unused)]
pub fn nvfp4_matmul(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
    input_scale: f32,
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

            let use_hardware_fp4 =
                is_hardware_fp4_available(dev) && m >= 32 && n % 32 == 0 && k % 32 == 0;

            let output = Tensor::zeros((m, n), dtype, dev)?;
            let has_bias = bias.is_some();

            if use_hardware_fp4 {
                // Hardware FP4 path: quantize activations -> CUTLASS block-scaled GEMM
                let stream = *cuda_dev.cu_stream() as i64;

                let m_padded = pad_to(m, 128);
                let k_scale_cols = k / NVFP4_BLOCK_SIZE;
                let k_scale_padded = pad_to(k_scale_cols, 4);
                let n_padded = pad_to(n, 128);

                let act_packed = Tensor::zeros((m, k / 2), DType::U8, dev)?;
                let act_scales = Tensor::zeros((m_padded, k_scale_cols), DType::U8, dev)?;
                let act_scales_swizzled =
                    Tensor::zeros((m_padded, k_scale_padded), DType::U8, dev)?;

                let weight_scales_swizzled =
                    Tensor::zeros((n_padded, k_scale_padded), DType::U8, dev)?;

                // GEMM alpha = input_scale * weight_global_scale
                // input_scale_inv is pre-baked into activation block scales during quantization
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

                    let (wscale_sw_s, _) = weight_scales_swizzled.storage_and_layout();
                    let wscale_sw_ptr = cuda_ptr(&wscale_sw_s, DType::U8)? as *mut std::ffi::c_void;

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
                            _ => candle_core::bail!("nvfp4_matmul: unsupported dtype {:?}", dtype),
                        }

                        ffi::nvfp4_swizzle_weight_scales(
                            scale_ptr,
                            wscale_sw_ptr,
                            n as i32,
                            k_scale_cols as i32,
                            n_padded as i32,
                            k_scale_padded as i32,
                            stream,
                        );

                        let (ws_ptr, _, _, _) =
                            crate::flashinfer::get_or_init_workspace(cuda_dev, false)?;
                        let ws_bytes = crate::flashinfer::WORKSPACE_FLOAT_SIZE as i64;

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
                            _ => candle_core::bail!("nvfp4_matmul: unsupported dtype {:?}", dtype),
                        }
                    }
                }

                if let Some(b) = bias {
                    return Ok(output.broadcast_add(b)?);
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

/// NVFP4 indexed MoE GEMM
///
/// * `input` - [num_tokens, K] or [num_tokens, topk, K]
/// * `weights` - [num_experts, N, K/2] packed U8
/// * `weight_scales` - [num_experts, N, K/16] U8 FP8 E4M3 block scales
/// * `weight_global_scales` - [num_experts] F32 per-expert global scales
/// * `input_scales` - Optional [num_experts] F32 per-expert activation global scales.
///   Used by the hardware FP4 path to compute per-expert GEMM alpha.
///   Ignored by the software path.
/// * `biases` - Optional [num_experts, N]
/// * `indices` - [num_tokens, topk] U32 expert indices
///
/// On Blackwell (SM100+) with cutlass feature: sorts tokens by expert, quantizes
/// activations to FP4 on-the-fly, and uses CUTLASS grouped GEMM with per-expert
/// alpha = input_scale[e] * weight_global_scale[e].
/// On older GPUs: uses software dequant path (LUT-based FP4 decode + FMA/WMMA).
///
/// Returns [num_tokens, topk, N]
#[allow(unused)]
pub fn nvfp4_moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    weight_scales: &Tensor,
    weight_global_scales: &Tensor,
    input_scales: Option<&Tensor>,
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
    let weight_global_scales = if weight_global_scales.is_contiguous() {
        weight_global_scales.clone()
    } else {
        weight_global_scales.contiguous()?
    };
    let indices = if indices.is_contiguous() {
        indices.clone()
    } else {
        indices.contiguous()?
    };

    let indices_dims = indices.dims();
    if indices_dims.len() != 2 {
        candle_core::bail!(
            "nvfp4_moe_gemm: expected indices rank 2 [num_tokens, topk], got {:?}",
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
                    "nvfp4_moe_gemm: input/indices mismatch: input tokens={t}, indices tokens={num_tokens}"
                );
            }
            (*kk, false)
        }
        [t, tk, kk] => {
            if *t != num_tokens || *tk != topk {
                candle_core::bail!(
                    "nvfp4_moe_gemm: input/indices mismatch: input={input_dims:?}, indices={indices_dims:?}"
                );
            }
            (*kk, true)
        }
        _ => candle_core::bail!(
            "nvfp4_moe_gemm: expected input rank 2 or 3, got {:?}",
            input_dims
        ),
    };

    if k % NVFP4_BLOCK_SIZE != 0 {
        candle_core::bail!("nvfp4_moe_gemm: K must be divisible by {NVFP4_BLOCK_SIZE}, got K={k}");
    }

    let w_dims = weights.dims();
    if w_dims.len() != 3 {
        candle_core::bail!(
            "nvfp4_moe_gemm: expected weights rank 3 [E, N, K/2], got {:?}",
            w_dims
        );
    }
    let num_experts = w_dims[0];
    let n = w_dims[1];

    if w_dims[2] != k / 2 {
        candle_core::bail!(
            "nvfp4_moe_gemm: weights shape mismatch, expected [E, N, K/2]=[{}, {}, {}], got {:?}",
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

            #[allow(unused)]
            let use_hardware_fp4 = is_hardware_fp4_available(dev) && n % 32 == 0 && k % 32 == 0;

            #[cfg(feature = "cutlass")]
            if use_hardware_fp4 && num_tokens > 0 {
                return nvfp4_moe_gemm_hardware(
                    &input,
                    &weights,
                    &weight_scales,
                    &weight_global_scales,
                    input_scales,
                    &indices,
                    num_tokens,
                    topk,
                    num_experts,
                    k,
                    n,
                    input_has_topk_dim,
                    dtype,
                    dev,
                    cuda_dev,
                );
            }

            let output = Tensor::zeros((num_tokens, topk, n), dtype, dev)?;

            {
                fn cuda_ptr_moe(s: &Storage, dtype: DType) -> candle_core::Result<u64> {
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

                let (input_s, _) = input.storage_and_layout();
                let (weights_s, _) = weights.storage_and_layout();
                let (scales_s, _) = weight_scales.storage_and_layout();
                let (gscales_s, _) = weight_global_scales.storage_and_layout();
                let (indices_s, _) = indices.storage_and_layout();
                let (output_s, _) = output.storage_and_layout();

                let input_ptr = cuda_ptr_moe(&input_s, dtype)? as *const std::ffi::c_void;
                let weights_ptr = cuda_ptr_moe(&weights_s, DType::U8)? as *const u8;
                let scales_ptr = cuda_ptr_moe(&scales_s, DType::U8)? as *const u8;
                let gscales_ptr = cuda_ptr_moe(&gscales_s, DType::F32)? as *const f32;
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
                            ffi::nvfp4_indexed_moe_gemm_f16(
                                input_ptr,
                                weights_ptr,
                                scales_ptr,
                                gscales_ptr,
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
                        DType::BF16 => {
                            ffi::nvfp4_indexed_moe_gemm_bf16(
                                input_ptr,
                                weights_ptr,
                                scales_ptr,
                                gscales_ptr,
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
                        _ => {
                            candle_core::bail!("nvfp4_moe_gemm CUDA: unsupported dtype {:?}", dtype)
                        }
                    }
                }
            }

            Ok(output)
        }
        _ => candle_core::bail!("nvfp4_moe_gemm: unsupported backend (need CUDA)"),
    }
}

/// Hardware CUTLASS MoE GEMM path for Blackwell (SM100+).
///
/// Sorts tokens by expert, quantizes activations to FP4 on-the-fly,
/// swizzles scales, and runs CUTLASS grouped GEMM with per-expert alpha.
#[cfg(all(feature = "cuda", feature = "cutlass"))]
#[allow(clippy::too_many_arguments)]
fn nvfp4_moe_gemm_hardware(
    input: &Tensor,
    weights: &Tensor,
    weight_scales: &Tensor,
    weight_global_scales: &Tensor,
    input_scales: Option<&Tensor>,
    indices: &Tensor,
    num_tokens: usize,
    topk: usize,
    num_experts: usize,
    k: usize,
    n: usize,
    _input_has_topk_dim: bool,
    dtype: DType,
    dev: &candle_core::Device,
    cuda_dev: &candle_core::cuda_backend::CudaDevice,
) -> Result<Tensor> {
    use candle_core::Storage;

    fn cuda_ptr(s: &Storage, dtype: DType) -> candle_core::Result<u64> {
        match s {
            Storage::Cuda(c) => match dtype {
                DType::F16 => Ok(*c.as_cuda_slice::<half::f16>()?.device_ptr()),
                DType::BF16 => Ok(*c.as_cuda_slice::<half::bf16>()?.device_ptr()),
                DType::U8 => Ok(*c.as_cuda_slice::<u8>()?.device_ptr()),
                DType::U32 => Ok(*c.as_cuda_slice::<u32>()?.device_ptr()),
                DType::I64 => Ok(*c.as_cuda_slice::<i64>()?.device_ptr()),
                DType::F32 => Ok(*c.as_cuda_slice::<f32>()?.device_ptr()),
                _ => candle_core::bail!("unsupported dtype {:?}", dtype),
            },
            _ => candle_core::bail!("tensor must be on CUDA"),
        }
    }

    let stream = *cuda_dev.cu_stream() as i64;
    let total_expanded = num_tokens * topk;

    // Step 1: Read indices to CPU and sort tokens by expert
    let indices_cpu = indices.to_dtype(DType::U32)?.to_vec2::<u32>()?;

    // Build expanded token list: (expert_id, original_expanded_row, source_token)
    let mut expanded: Vec<(u32, usize, usize)> = Vec::with_capacity(total_expanded);
    for t in 0..num_tokens {
        for tk in 0..topk {
            let expert_id = indices_cpu[t][tk];
            let expanded_row = t * topk + tk;
            expanded.push((expert_id, expanded_row, t));
        }
    }

    // Sort by expert_id for contiguous expert groups
    expanded.sort_by_key(|&(eid, _, _)| eid);

    // Build sorted_token_ids (which source token each sorted row comes from)
    // and scatter_ids (where each sorted row goes back to in the output)
    // Using u32 since all values are non-negative; cast to i32* at FFI boundary.
    let sorted_token_ids: Vec<u32> = expanded.iter().map(|&(_, _, src)| src as u32).collect();
    let scatter_ids: Vec<u32> = expanded.iter().map(|&(_, orig, _)| orig as u32).collect();

    // Build per-expert offsets and problem sizes
    let mut expert_offsets = vec![0u32; num_experts];
    let mut expert_counts = vec![0u32; num_experts];
    for &(eid, _, _) in &expanded {
        expert_counts[eid as usize] += 1;
    }
    let mut offset = 0u32;
    for e in 0..num_experts {
        expert_offsets[e] = offset;
        offset += expert_counts[e];
    }

    // problem_sizes: [E * 3] = (M_i, N, K) per expert
    let mut problem_sizes = vec![0u32; num_experts * 3];
    for e in 0..num_experts {
        problem_sizes[e * 3] = expert_counts[e];
        problem_sizes[e * 3 + 1] = n as u32;
        problem_sizes[e * 3 + 2] = k as u32;
    }

    // CUTLASS block-scaled grouped GEMM requires each expert's activation-scale rows
    // to start on a 128-row boundary, even though the packed activations themselves
    // remain tightly packed.
    let mut sf_offsets = vec![0u32; num_experts];
    let mut total_sf_rows = 0usize;
    for e in 0..num_experts {
        sf_offsets[e] = total_sf_rows as u32;
        total_sf_rows += pad_to(expert_counts[e] as usize, 128);
    }

    // Step 2: Compute per-expert alphas = input_scale[e] * weight_global_scale[e]
    let wgs_cpu = weight_global_scales.to_vec1::<f32>()?;
    let iscales_cpu: Vec<f32> = if let Some(is) = input_scales {
        is.to_vec1::<f32>()?
    } else {
        vec![1.0f32; num_experts]
    };
    let alphas: Vec<f32> = (0..num_experts)
        .map(|e| iscales_cpu[e] * wgs_cpu[e])
        .collect();

    // Step 3: Upload CPU arrays to GPU (u32 tensors, cast to i32* at FFI boundary)
    let sorted_token_ids_t = Tensor::from_vec(sorted_token_ids, (total_expanded,), dev)?;
    let scatter_ids_t = Tensor::from_vec(scatter_ids, (total_expanded,), dev)?;
    // Step 4: Gather input tokens sorted by expert
    let gathered = Tensor::zeros((total_expanded, k), dtype, dev)?;
    {
        let (input_s, _) = input.storage_and_layout();
        let (gathered_s, _) = gathered.storage_and_layout();
        let (stids_s, _) = sorted_token_ids_t.storage_and_layout();

        let input_ptr = cuda_ptr(&input_s, dtype)? as *const std::ffi::c_void;
        let gathered_ptr = cuda_ptr(&gathered_s, dtype)? as *mut std::ffi::c_void;
        let stids_ptr = cuda_ptr(&stids_s, DType::U32)? as *const i32;

        unsafe {
            match dtype {
                DType::F16 => ffi::nvfp4_moe_gather_f16(
                    input_ptr,
                    gathered_ptr,
                    stids_ptr,
                    total_expanded as i32,
                    k as i32,
                    stream,
                ),
                DType::BF16 => ffi::nvfp4_moe_gather_bf16(
                    input_ptr,
                    gathered_ptr,
                    stids_ptr,
                    total_expanded as i32,
                    k as i32,
                    stream,
                ),
                _ => candle_core::bail!("unsupported dtype {:?}", dtype),
            }
        }
    }

    // Step 5: Quantize gathered activations to FP4.
    // Each expert slice must be quantized with its own activation global scale.
    // Folding all per-expert input scales into the GEMM alpha is incorrect because
    // the block scales themselves must already include the activation-side global
    // scale used during FP4 quantization.
    let k_scale = k / NVFP4_BLOCK_SIZE;
    let k_scale_padded = pad_to(k_scale, 4);

    let act_packed = Tensor::zeros((total_expanded, k / 2), DType::U8, dev)?;
    let act_scales = Tensor::zeros((total_expanded, k_scale), DType::U8, dev)?;
    let act_scales_swizzled = Tensor::zeros((total_sf_rows, k_scale_padded), DType::U8, dev)?;

    {
        let (gathered_s, _) = gathered.storage_and_layout();
        let (act_packed_s, _) = act_packed.storage_and_layout();
        let (act_scales_s, _) = act_scales.storage_and_layout();
        let (act_scales_sw_s, _) = act_scales_swizzled.storage_and_layout();

        let gathered_base = cuda_ptr(&gathered_s, dtype)?;
        let act_packed_base = cuda_ptr(&act_packed_s, DType::U8)?;
        let act_scales_base = cuda_ptr(&act_scales_s, DType::U8)?;
        let act_scales_sw_base = cuda_ptr(&act_scales_sw_s, DType::U8)?;

        let gathered_row_bytes = match dtype {
            DType::F16 => k * std::mem::size_of::<half::f16>(),
            DType::BF16 => k * std::mem::size_of::<half::bf16>(),
            _ => candle_core::bail!("unsupported dtype {:?}", dtype),
        } as u64;

        for e in 0..num_experts {
            let rows = expert_counts[e] as usize;
            if rows == 0 {
                continue;
            }

            let input_scale_inv = if iscales_cpu[e] != 0.0 {
                1.0 / iscales_cpu[e]
            } else {
                1.0
            };
            let row_offset = expert_offsets[e] as u64;
            let sf_row_offset = sf_offsets[e] as u64;
            let rows_padded = pad_to(rows, 128);

            let gathered_ptr =
                (gathered_base + row_offset * gathered_row_bytes) as *const std::ffi::c_void;
            let act_packed_ptr =
                (act_packed_base + row_offset * (k as u64 / 2)) as *mut std::ffi::c_void;
            let act_scales_ptr =
                (act_scales_base + row_offset * k_scale as u64) as *mut std::ffi::c_void;
            let act_scales_sw_ptr = (act_scales_sw_base + sf_row_offset * k_scale_padded as u64)
                as *mut std::ffi::c_void;

            unsafe {
                match dtype {
                    DType::F16 => ffi::nvfp4_quantize_activation_f16(
                        gathered_ptr,
                        act_packed_ptr,
                        act_scales_ptr,
                        act_scales_sw_ptr,
                        input_scale_inv,
                        rows as i32,
                        k as i32,
                        rows_padded as i32,
                        k_scale_padded as i32,
                        stream,
                    ),
                    DType::BF16 => ffi::nvfp4_quantize_activation_bf16(
                        gathered_ptr,
                        act_packed_ptr,
                        act_scales_ptr,
                        act_scales_sw_ptr,
                        input_scale_inv,
                        rows as i32,
                        k as i32,
                        rows_padded as i32,
                        k_scale_padded as i32,
                        stream,
                    ),
                    _ => candle_core::bail!("unsupported dtype {:?}", dtype),
                }
            }
        }
    }

    let expert_offsets_t = Tensor::from_vec(expert_offsets, (num_experts,), dev)?;
    let sf_offsets_t = Tensor::from_vec(sf_offsets, (num_experts,), dev)?;
    let problem_sizes_t = Tensor::from_vec(problem_sizes, (num_experts * 3,), dev)?;
    let alphas_t = Tensor::from_vec(alphas, (num_experts,), dev)?;

    // Step 6: Swizzle weight scales for CUTLASS
    // Weight scales: [E, N, K/16] -> swizzle each expert's [N, K/16] block
    let n_padded = pad_to(n, 128);
    let weight_scales_swizzled =
        Tensor::zeros((num_experts, n_padded, k_scale_padded), DType::U8, dev)?;
    {
        let (ws_s, _) = weight_scales.storage_and_layout();
        let (wss_s, _) = weight_scales_swizzled.storage_and_layout();
        let ws_base = cuda_ptr(&ws_s, DType::U8)?;
        let wss_base = cuda_ptr(&wss_s, DType::U8)?;

        for e in 0..num_experts {
            let src_offset = (e * n * k_scale) as u64;
            let dst_offset = (e * n_padded * k_scale_padded) as u64;
            unsafe {
                ffi::nvfp4_swizzle_weight_scales(
                    (ws_base + src_offset) as *const std::ffi::c_void,
                    (wss_base + dst_offset) as *mut std::ffi::c_void,
                    n as i32,
                    k_scale as i32,
                    n_padded as i32,
                    k_scale_padded as i32,
                    stream,
                );
            }
        }
    }

    // Step 7: Run CUTLASS grouped GEMM
    let cutlass_output = Tensor::zeros((total_expanded, n), dtype, dev)?;
    {
        let (act_packed_s, _) = act_packed.storage_and_layout();
        let (weights_s, _) = weights.storage_and_layout();
        let (act_scales_sw_s, _) = act_scales_swizzled.storage_and_layout();
        let (wss_s, _) = weight_scales_swizzled.storage_and_layout();
        let (alphas_s, _) = alphas_t.storage_and_layout();
        let (eo_s, _) = expert_offsets_t.storage_and_layout();
        let (sfo_s, _) = sf_offsets_t.storage_and_layout();
        let (ps_s, _) = problem_sizes_t.storage_and_layout();
        let (out_s, _) = cutlass_output.storage_and_layout();

        let act_packed_ptr = cuda_ptr(&act_packed_s, DType::U8)? as *const std::ffi::c_void;
        let weights_ptr = cuda_ptr(&weights_s, DType::U8)? as *const std::ffi::c_void;
        let act_scales_sw_ptr = cuda_ptr(&act_scales_sw_s, DType::U8)? as *const std::ffi::c_void;
        let wss_ptr = cuda_ptr(&wss_s, DType::U8)? as *const std::ffi::c_void;
        let alphas_ptr = cuda_ptr(&alphas_s, DType::F32)? as *const f32;
        let eo_ptr = cuda_ptr(&eo_s, DType::U32)? as *const i32;
        let sfo_ptr = cuda_ptr(&sfo_s, DType::U32)? as *const i32;
        let ps_ptr = cuda_ptr(&ps_s, DType::U32)? as *const i32;
        let out_ptr = cuda_ptr(&out_s, dtype)? as *mut std::ffi::c_void;

        unsafe {
            let ret = match dtype {
                DType::F16 => ffi::nvfp4_cutlass_moe_gemm_f16(
                    out_ptr,
                    act_packed_ptr,
                    weights_ptr,
                    act_scales_sw_ptr,
                    wss_ptr,
                    alphas_ptr,
                    eo_ptr,
                    sfo_ptr,
                    ps_ptr,
                    num_experts as i32,
                    total_expanded as i32,
                    n as i32,
                    k as i32,
                    stream,
                ),
                DType::BF16 => ffi::nvfp4_cutlass_moe_gemm_bf16(
                    out_ptr,
                    act_packed_ptr,
                    weights_ptr,
                    act_scales_sw_ptr,
                    wss_ptr,
                    alphas_ptr,
                    eo_ptr,
                    sfo_ptr,
                    ps_ptr,
                    num_experts as i32,
                    total_expanded as i32,
                    n as i32,
                    k as i32,
                    stream,
                ),
                _ => candle_core::bail!("unsupported dtype {:?}", dtype),
            };
            if ret != 0 {
                candle_core::bail!("nvfp4_cutlass_moe_gemm failed with error code {}", ret);
            }
        }
    }

    // Step 8: Scatter results back to [num_tokens * topk, N] order
    let output = Tensor::zeros((total_expanded, n), dtype, dev)?;
    {
        let (cutlass_out_s, _) = cutlass_output.storage_and_layout();
        let (output_s, _) = output.storage_and_layout();
        let (scatter_s, _) = scatter_ids_t.storage_and_layout();

        let cutlass_out_ptr = cuda_ptr(&cutlass_out_s, dtype)? as *const std::ffi::c_void;
        let output_ptr = cuda_ptr(&output_s, dtype)? as *mut std::ffi::c_void;
        let scatter_ptr = cuda_ptr(&scatter_s, DType::U32)? as *const i32;

        unsafe {
            match dtype {
                DType::F16 => ffi::nvfp4_moe_scatter_f16(
                    cutlass_out_ptr,
                    output_ptr,
                    scatter_ptr,
                    total_expanded as i32,
                    n as i32,
                    stream,
                ),
                DType::BF16 => ffi::nvfp4_moe_scatter_bf16(
                    cutlass_out_ptr,
                    output_ptr,
                    scatter_ptr,
                    total_expanded as i32,
                    n as i32,
                    stream,
                ),
                _ => candle_core::bail!("unsupported dtype {:?}", dtype),
            }
        }
    }

    // Reshape to [num_tokens, topk, N]
    output.reshape((num_tokens, topk, n))
}
