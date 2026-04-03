#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{Result, Tensor};

pub const NVFP4_BLOCK_SIZE: usize = 16;

/// NVFP4 linear: output = input @ weight^T [+ bias]
///
/// * `input` - [M, K] in F16/BF16
/// * `weight` - [N, K/2] packed U8 (2 FP4 E2M1 nibbles per byte)
/// * `scale` - [N, K/16] U8 FP8 E4M3 block scales
/// * `weight_global_scale` - scalar F32 global scale
/// * `bias` - Optional [N] in F16/BF16
///
/// Returns [M, N] in same dtype as input
#[allow(unused)]
pub fn nvfp4_matmul(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    weight_global_scale: f32,
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
/// * `biases` - Optional [num_experts, N]
/// * `indices` - [num_tokens, topk] U32 expert indices
///
/// Returns [num_tokens, topk, N]
#[allow(unused)]
pub fn nvfp4_moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    weight_scales: &Tensor,
    weight_global_scales: &Tensor,
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
