#[cfg(feature = "cuda")]
use crate::cuda_utils;
#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "metal")]
use crate::metal_kernels;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Device, Result, Tensor};

/// FP8 Matrix Multiplication: C = A * B^T
///
/// # Arguments
/// * `input` - Input tensor A of shape [M, K]
/// * `weight` - Weight tensor B of shape [N, K] (stored as u8)
/// * `weight_scale` - Scales for weight tensor
/// * `block_size` - [block_size_y, block_size_x] for scaling
///
/// The weight tensor is expected to be in FP8 format (e4m3).
#[allow(unused)]
pub fn fp8_matmul(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize], // [block_size_y, block_size_x]
) -> Result<Tensor> {
    fp8_matmul_with_layout(input, weight, weight_scale, block_size, false)
}

/// FP8 Matrix Multiplication with explicit weight layout.
///
/// When `weight_col_major` is true, `weight` is expected to be stored as [K, N].
#[allow(unused)]
pub fn fp8_matmul_with_layout(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize],
    weight_col_major: bool,
) -> Result<Tensor> {
    let (m, k) = input.dims2()?;
    let (n, k_w) = if weight_col_major {
        let (k_w, n) = weight.dims2()?;
        (n, k_w)
    } else {
        weight.dims2()?
    };

    if k != k_w {
        //  println!("DEBUG: mismatch k={} k_w={} m={} n={} col={}", k, k_w, m, n, weight_col_major);
        candle_core::bail!(
            "Shape mismatch in fp8_matmul: input [{}, {}], weight [{}, {}]",
            m,
            k,
            n,
            k_w
        );
    }

    let dev = input.device();
    let dtype = input.dtype();
    assert!(
        weight_scale.dtype() == DType::F32,
        "fp8_matmul expects f32 scales, got {:?}",
        weight_scale.dtype()
    );
    let scale_row_stride = (k_w + block_size[1] - 1) / block_size[1];

    let output = Tensor::zeros((m, n), dtype, dev)?;

    if weight_col_major {
        match dev {
            Device::Cuda(_) => {}
            _ => candle_core::bail!("fp8_matmul_with_layout only supports col-major on CUDA"),
        }
    }

    let sm90_plus = matches!(dev, Device::Cuda(_))
        && cuda_utils::sm_version(dev.as_cuda_device()?).map_or(false, |sm| sm >= 90);
    let use_cutlass = weight_col_major
        && cfg!(feature = "cutlass")
        && sm90_plus
        && block_size.len() == 2
        && block_size[0] == 128
        && block_size[1] == 128;

    if !use_cutlass && weight_scale.dims1().is_ok() {
        candle_core::bail!(
            "packed weight_scale requires CUTLASS path; provide row-major scales for this dispatch"
        );
    }

    match (dev, dtype) {
        #[cfg(feature = "cuda")]
        (Device::Cuda(dev), DType::F16) => {
            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
                _ => candle_core::bail!("input must be a cuda tensor"),
            };
            let input_ptr = *input_slice.device_ptr() as *const core::ffi::c_void;

            let (weight_storage, _) = weight.storage_and_layout();
            let weight_slice = match &*weight_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
                _ => candle_core::bail!("weight must be a cuda tensor"),
            };
            let weight_ptr = *weight_slice.device_ptr() as *const u8;

            let (scale_storage, _) = weight_scale.storage_and_layout();
            let scale_slice = match &*scale_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle_core::bail!("weight_scale must be a cuda tensor"),
            };
            let weight_scale_ptr = *scale_slice.device_ptr() as *const f32;

            let (output_storage, _) = output.storage_and_layout();
            let output_slice = match &*output_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<half::f16>()?,
                _ => candle_core::bail!("output allocation failed"),
            };
            let output_ptr = *output_slice.device_ptr() as *mut core::ffi::c_void;

            let stream = *dev.cu_stream() as i64;

            unsafe {
                if use_cutlass {
                    ffi::fp8_matmul_f16_cutlass(
                        input_ptr,
                        weight_ptr,
                        weight_scale_ptr,
                        output_ptr,
                        m as i32,
                        n as i32,
                        k as i32,
                        scale_row_stride as i32,
                        block_size[0] as i32,
                        block_size[1] as i32,
                        weight_col_major as i32,
                        stream,
                    )
                } else {
                    ffi::fp8_matmul_f16_layout(
                        input_ptr,
                        weight_ptr,
                        weight_scale_ptr,
                        output_ptr,
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
        }
        #[cfg(feature = "cuda")]
        (Device::Cuda(dev), DType::BF16) => {
            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<half::bf16>()?,
                _ => candle_core::bail!("input must be a cuda tensor"),
            };
            let input_ptr = *input_slice.device_ptr() as *const core::ffi::c_void;

            let (weight_storage, _) = weight.storage_and_layout();
            let weight_slice = match &*weight_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
                _ => candle_core::bail!("weight must be a cuda tensor"),
            };
            let weight_ptr = *weight_slice.device_ptr() as *const u8;

            let (scale_storage, _) = weight_scale.storage_and_layout();
            let scale_slice = match &*scale_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle_core::bail!("weight_scale must be a cuda tensor"),
            };
            let weight_scale_ptr = *scale_slice.device_ptr() as *const f32;

            let (output_storage, _) = output.storage_and_layout();
            let output_slice = match &*output_storage {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<half::bf16>()?,
                _ => candle_core::bail!("output allocation failed"),
            };
            let output_ptr = *output_slice.device_ptr() as *mut core::ffi::c_void;

            let stream = *dev.cu_stream() as i64;

            unsafe {
                if use_cutlass {
                    ffi::fp8_matmul_bf16_cutlass(
                        input_ptr,
                        weight_ptr,
                        weight_scale_ptr,
                        output_ptr,
                        m as i32,
                        n as i32,
                        k as i32,
                        scale_row_stride as i32,
                        block_size[0] as i32,
                        block_size[1] as i32,
                        weight_col_major as i32,
                        stream,
                    )
                } else {
                    ffi::fp8_matmul_bf16_layout(
                        input_ptr,
                        weight_ptr,
                        weight_scale_ptr,
                        output_ptr,
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

/// Pack row-major FP8 weight scales into CUTLASS SFB layout.
///
/// `weight_scale` must be a contiguous F32 tensor shaped [ceil(N/by), ceil(K/bx)].
/// Returns a flat packed tensor to be used with CUTLASS paths.
#[cfg(feature = "cuda")]
pub fn pack_fp8_weight_scale_sfb(
    weight_scale: &Tensor,
    n: usize,
    k: usize,
    block_size: &[usize],
) -> Result<Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::WrapErr;

    if block_size.len() != 2 {
        candle_core::bail!("block_size must be [block_size_n, block_size_k]");
    }
    let dev = weight_scale.device().as_cuda_device()?;
    if weight_scale.dtype() != DType::F32 {
        candle_core::bail!("weight_scale must be F32");
    }

    let (scale_0, scale_1) = weight_scale.dims2()?;
    let expected_n = (n + block_size[0] - 1) / block_size[0];
    let expected_k = (k + block_size[1] - 1) / block_size[1];
    let input_k_major;

    // println!("DEBUG: pack_sfb: n={} k={} expected_n={} expected_k={} scale_dims=({}, {})", n, k, expected_n, expected_k, scale_0, scale_1);

    if scale_0 == expected_n && scale_1 == expected_k {
        input_k_major = false;
    } else if scale_0 == expected_k && scale_1 == expected_n {
        input_k_major = true;
    } else {
        candle_core::bail!(
            "weight_scale shape mismatch: expected [{}, {}] or [{}, {}], got [{}, {}]",
            expected_n,
            expected_k,
            expected_k,
            expected_n,
            scale_0,
            scale_1
        );
    }

    let packed_len = unsafe { ffi::fp8_sfb_packed_len(n as i32, k as i32) };
    if packed_len <= 0 {
        candle_core::bail!("fp8_sfb_packed_len returned 0; CUTLASS not enabled?");
    }

    let packed = unsafe { dev.alloc::<f32>(packed_len as usize) }.w()?;
    let (scale_storage, _) = weight_scale.storage_and_layout();
    let scale_slice = match &*scale_storage {
        candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("weight_scale must be a cuda tensor"),
    };

    let stream = *dev.cu_stream() as i64;
    unsafe {
        ffi::fp8_pack_sfb_scales(
            *scale_slice.device_ptr() as *const f32,
            *packed.device_ptr() as *mut f32,
            n as i32,
            k as i32,
            block_size[0] as i32,
            block_size[1] as i32,
            if input_k_major { 1 } else { 0 },
            stream,
        );
    }

    let packed = candle_core::CudaStorage::wrap_cuda_slice(packed, dev.clone());
    Tensor::from_storage(candle_core::Storage::Cuda(packed), (packed_len as usize,))
}
