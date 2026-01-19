#[cfg(feature = "cuda")]
use crate::cuda_utils;
#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "metal")]
use crate::metal_kernels;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Tensor};

#[cfg(feature = "cuda")]
fn get_cuda_slice<
    T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType,
>(
    tensor: &Tensor,
) -> Result<u64> {
    let (storage, _) = tensor.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            Ok(*slice.device_ptr() as u64)
        }
        _ => candle_core::bail!("expecting cuda tensor"),
    }
}

/// FP8 Matrix Multiplication: C = A * B^T
///
/// # Arguments
/// * `input` - Input tensor A of shape [M, K]
/// * `weight` - Weight tensor B of shape [K, N] (stored as u8)
/// * `weight_scale` - Scales for weight tensor
/// * `block_size` - [block_size_y, block_size_x] for scaling
///
/// The weight tensor is expected to be in FP8 format (e4m3).
/// FP8 Matrix Multiplication with explicit weight layout.
#[allow(unused)]
pub fn fp8_matmul(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize],
    weight_col_major: bool,
) -> Result<Tensor> {
    assert!(weight_col_major, "Only weight_col_major is supported");
    let (m, k) = input.dims2()?;
    let (k, n) = weight.dims2()?;
    let dev = input.device();
    let dtype = input.dtype();
    assert!(
        weight_scale.dtype() == DType::F32,
        "fp8_matmul expects f32 scales, got {:?}",
        weight_scale.dtype()
    );
    let scale_row_stride = (k + block_size[1] - 1) / block_size[1];

    #[cfg(feature = "cuda")]
    let sm_version = if matches!(dev, Device::Cuda(_)) {
        cuda_utils::sm_version(dev.as_cuda_device()?).unwrap_or(0) as i32
    } else {
        0
    };
    #[cfg(feature = "cuda")]
    let sm90_plus = sm_version >= 90;
    #[cfg(not(feature = "cuda"))]
    let sm90_plus = false;

    let use_cutlass = cfg!(feature = "cutlass")
        && sm90_plus
        && block_size.len() == 2
        && block_size[0] == 128
        && block_size[1] == 128;

    if input.rank() != 2 {
        candle_core::bail!("mat_a must be a 2D tensor");
    }
    if weight.rank() != 2 {
        candle_core::bail!("mat_b must be a 2D tensor");
    }

    if !input.is_contiguous() {
        candle_core::bail!("mat_a must be contiguous (row major)");
    }

    if use_cutlass && weight.stride()[0] != 1 {
        candle_core::bail!("mat_b must be a column major tensor (stride(0) == 1)");
    }

    if k != weight.dim(0)? {
        // mat_b is [K, N]
        candle_core::bail!(
            "mat_a and mat_b shapes cannot be multiplied: K={} vs mat_b.dim(0)={}",
            k,
            weight.dim(0)?
        );
    }

    if (k * input.dtype().size_in_bytes()) % 16 != 0 {
        candle_core::bail!("mat_a (K dim) must be multiple of 16 bytes");
    }
    if weight.dim(0)? % 16 != 0 {
        candle_core::bail!("mat_b (K dim) must be multiple of 16 bytes");
    }

    if weight_scale.dim(0)? != weight.dim(0)? / 128 || weight_scale.dim(1)? != weight.dim(1)? / 128
    {
        candle_core::bail!("scales_b shape mismatch");
    }

    let weight_scale_stride = weight_scale.stride();
    let weight_scale_col_major = weight_scale_stride[0] == 1;
    let weight_scale_row_major = weight_scale.is_contiguous() && weight_scale_stride[1] == 1;
    if use_cutlass && !(weight_scale_col_major || weight_scale_row_major) {
        candle_core::bail!("scales_b must be column major or contiguous row major");
    }

    if !use_cutlass {
        if weight_scale.dims1().is_ok() {
            candle_core::bail!("packed weight_scale requires CUTLASS path");
        }
    }

    let dev = input.device();
    let w_ptr = get_cuda_slice::<u8>(&weight)?;
    let ws_ptr = get_cuda_slice::<f32>(&weight_scale)?;

    let alignment = 4;
    let m_padded = (m + alignment - 1) / alignment * alignment;
    let pad_len = m_padded - m;

    let input_padded = if use_cutlass && pad_len > 0 {
        input.pad_with_zeros(0, 0, pad_len)? // Pad for TMA
    } else {
        input.clone()
    };

    let mut output = Tensor::zeros(
        (
            if use_cutlass && pad_len > 0 {
                m_padded
            } else {
                m
            },
            n,
        ),
        dtype,
        dev,
    )?;
    let cu_dev = dev.as_cuda_device()?;
    let stream = *cu_dev.cu_stream() as i64;
    let k_over_128 = (k + 127) / 128;

    let (q_ptr, s_ptr, scale_stride) = if use_cutlass {
        let input_q = Tensor::zeros((m_padded, k), DType::U8, &dev)?;
        // Create column-major scales so stride(0) == 1 (matches CUTLASS SFA layout).
        let input_scale_base = Tensor::zeros((k_over_128, m_padded), DType::F32, &dev)?;
        let input_scale = input_scale_base.t()?;
        let scale_stride = input_scale.stride()[1] as i32;

        let q_ptr = get_cuda_slice::<u8>(&input_q)? as *mut std::ffi::c_void;
        let s_ptr = get_cuda_slice::<f32>(&input_scale)? as *mut f32;
        (q_ptr, s_ptr, scale_stride)
    } else {
        (
            std::ptr::null_mut() as *mut std::ffi::c_void,
            std::ptr::null_mut() as *mut f32,
            0,
        )
    };

    let inp_ptr = if dtype == DType::F16 {
        get_cuda_slice::<half::f16>(&input_padded)?
    } else {
        get_cuda_slice::<half::bf16>(&input_padded)?
    };

    #[cfg(feature = "cutlass")]
    unsafe {
        let num_groups = m_padded * k_over_128;
        let group_size = 128;
        let num_groups_per_row = k_over_128;
        ffi::fp8_quantize_per_token_group_launch(
            inp_ptr as *const std::ffi::c_void,
            q_ptr,
            s_ptr,
            num_groups as i32,
            group_size as i32,
            num_groups_per_row as i32,
            scale_stride,
            dtype == DType::F16,
            true,
            stream as i64,
        );
    }

    match (dev, dtype) {
        #[cfg(feature = "cuda")]
        (Device::Cuda(dev), DType::F16) => {
            // GEMM
            let out_ptr = get_cuda_slice::<half::f16>(&output)?;
            unsafe {
                if use_cutlass {
                    ffi::fp8_matmul_f16_cutlass(
                        q_ptr as *const u8,
                        s_ptr as *const f32,
                        w_ptr as *const u8,
                        ws_ptr as *const f32,
                        out_ptr as *mut core::ffi::c_void,
                        m_padded as i32,
                        n as i32,
                        k as i32,
                        scale_row_stride as i32,
                        block_size[0] as i32,
                        block_size[1] as i32,
                        weight_col_major as i32,
                        sm_version,
                        stream,
                    )
                } else {
                    ffi::fp8_matmul_f16_layout(
                        inp_ptr as *const core::ffi::c_void,
                        w_ptr as *const u8,
                        ws_ptr as *const f32,
                        out_ptr as *mut core::ffi::c_void,
                        m as i32,
                        n as i32,
                        k as i32,
                        scale_row_stride as i32,
                        block_size[0] as i32,
                        block_size[1] as i32,
                        weight_col_major as i32,
                        stream,
                    )
                }
            }

            if use_cutlass && pad_len > 0 {
                output = output.narrow(0, 0, m)?;
            }
        }
        #[cfg(feature = "cuda")]
        (Device::Cuda(dev), DType::BF16) => {
            // GEMM
            let out_ptr = get_cuda_slice::<half::bf16>(&output)?;
            unsafe {
                if use_cutlass {
                    ffi::fp8_matmul_bf16_cutlass(
                        q_ptr as *const u8,
                        s_ptr as *const f32,
                        w_ptr as *const u8,
                        ws_ptr as *const f32,
                        out_ptr as *mut core::ffi::c_void,
                        m_padded as i32,
                        n as i32,
                        k as i32,
                        scale_row_stride as i32,
                        block_size[0] as i32,
                        block_size[1] as i32,
                        weight_col_major as i32,
                        sm_version,
                        stream,
                    )
                } else {
                    ffi::fp8_matmul_bf16_layout(
                        inp_ptr as *const core::ffi::c_void,
                        w_ptr as *const u8,
                        ws_ptr as *const f32,
                        out_ptr as *mut core::ffi::c_void,
                        m as i32,
                        n as i32,
                        k as i32,
                        scale_row_stride as i32,
                        block_size[0] as i32,
                        block_size[1] as i32,
                        weight_col_major as i32,
                        stream,
                    )
                }
            }

            if use_cutlass && pad_len > 0 {
                output = output.narrow(0, 0, m)?;
            }
        }
        (Device::Cuda(_), _) => candle_core::bail!("fp8_matmul requires f16 or bf16 input"),
        #[cfg(feature = "metal")]
        (Device::Metal(dev), _) => {
            let (input_storage, input_layout) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                candle_core::Storage::Metal(c) => c,
                _ => candle_core::bail!("input must be a metal tensor"),
            };
            let input_offset = input_layout.start_offset() * input.dtype().size_in_bytes();

            let (weight_storage, weight_layout) = weight.storage_and_layout();
            let weight_slice = match &*weight_storage {
                candle_core::Storage::Metal(c) => c,
                _ => candle_core::bail!("weight must be a metal tensor"),
            };
            let weight_offset = weight_layout.start_offset() * weight.dtype().size_in_bytes();

            let (scale_storage, scale_layout) = weight_scale.storage_and_layout();
            let scale_slice = match &*scale_storage {
                candle_core::Storage::Metal(c) => c,
                _ => candle_core::bail!("weight_scale must be a metal tensor"),
            };
            let scale_offset = scale_layout.start_offset() * weight_scale.dtype().size_in_bytes();

            let (output_storage, output_layout) = output.storage_and_layout();
            let output_slice = match &*output_storage {
                candle_core::Storage::Metal(c) => c,
                _ => candle_core::bail!("output allocation failed"),
            };
            let output_offset = output_layout.start_offset() * output.dtype().size_in_bytes();

            let command_buffer = dev.command_buffer()?;

            metal_kernels::call_fp8_matmul(
                dev.device(),
                &command_buffer,
                metal_kernels::Kernels::default(),
                dtype,
                input_slice.buffer(),
                input_offset,
                weight_slice.buffer(),
                weight_offset,
                scale_slice.buffer(),
                scale_offset,
                output_slice.buffer(),
                output_offset,
                m as i32,
                n as i32,
                k as i32,
                scale_row_stride as i32,
                block_size[0] as i32,
                block_size[1] as i32,
            )
            .map_err(candle_core::Error::wrap)?;
        }
        _ => candle_core::bail!("fp8_matmul only supports CUDA and Metal"),
    }

    Ok(output)
}
