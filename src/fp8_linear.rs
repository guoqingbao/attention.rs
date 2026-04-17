#[cfg(feature = "cuda")]
use crate::cuda_utils;
#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "metal")]
use crate::metal_kernels;
#[cfg(all(feature = "cuda", feature = "cutlass"))]
use crate::workspace::get_cutlass_workspace;
#[cfg(all(feature = "cuda", feature = "flashinfer"))]
use crate::workspace::get_or_init_flashinfer_fp8_workspace;
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

/// Unified FP8 matrix multiplication entry point.
///
/// Dispatches to the best available kernel based on hardware capability,
/// feature flags, and `is_prefill`:
///
/// 1. **FlashInfer** (sm90, BF16, 128x128 blocks, decode with m<=64)
/// 2. **CUTLASS** (sm90+, 128x128 blocks)
/// 3. **Fallback** (any CUDA or Metal)
///
/// # Arguments
/// * `input`  - [M, K] activation tensor (F16 or BF16)
/// * `weight` - [N, K] FP8 weight tensor (U8)
/// * `weight_scale` - [N/block_y, K/block_x] F32 scales (row-major)
/// * `weight_scale_cutlass` - Optional pre-transposed scales for CUTLASS path
/// * `block_size` - `[block_y, block_x]` quantization block dimensions
/// * `is_prefill` - true during prefill phase, affects kernel selection
pub fn fp8_matmul(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    weight_scale_cutlass: Option<&Tensor>,
    block_size: &[usize],
    is_prefill: bool,
) -> Result<Tensor> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let (m, _) = input.dims2()?;

    #[cfg(feature = "cuda")]
    let sm_version = if let Ok(cuda_dev) = input.device().as_cuda_device() {
        crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0) as usize
    } else {
        0
    };

    #[cfg(all(feature = "cuda", feature = "flashinfer"))]
    {
        // Enable only for decode phase (small M <= 64) on SM90 with BF16 and 128x128 block scales
        let use_flashinfer = !is_prefill
            && m <= 64
            && (90..100).contains(&sm_version)
            && DType::BF16 == input.dtype()
            && block_size == [128, 128];
        if use_flashinfer {
            return fp8_matmul_flashinfer(&input, weight, weight_scale);
        }
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    {
        let use_cutlass = sm_version >= 90 && block_size == [128, 128];
        if use_cutlass {
            let cutlass_scale = weight_scale_cutlass.unwrap_or(weight_scale);
            return fp8_matmul_cutlass(&input, &weight.t()?, cutlass_scale, block_size);
        }
    }

    fp8_matmul_fallback(&input, weight, weight_scale, block_size)
}

/// FP8 Matrix Multiplication: C = A * B^T (conventional path).
///
/// # Arguments
/// * `input` - Input tensor A of shape [M, K]
/// * `weight` - Weight tensor B of shape [N, K] (stored as u8)
/// * `weight_scale` - Scales for weight tensor
/// * `block_size` - [block_size_y, block_size_x] for scaling
///
/// The weight tensor is expected to be in FP8 format (e4m3).
#[allow(unused)]
pub fn fp8_matmul_fallback(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize],
) -> Result<Tensor> {
    let (m, k) = input.dims2()?;
    let (n, k_w) = weight.dims2()?;

    if k != k_w {
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
                ffi::fp8_matmul_f16(
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
                    stream,
                )
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
                ffi::fp8_matmul_bf16(
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
                    stream,
                )
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

/// FP8 Matrix Multiplication using FlashInfer/TensorRT-LLM SM90 blockwise GEMM.
///
/// This path expects Hopper-native blockwise scales in `[N/128, K/128]` layout and
/// relies on the underlying runner's small-`M` swapAB optimization for decode.
#[cfg(all(feature = "cuda", feature = "flashinfer"))]
pub fn fp8_matmul_flashinfer(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
) -> Result<Tensor> {
    let (m, k) = input.dims2()?;
    let (n, k_w) = weight.dims2()?;

    if k != k_w {
        candle_core::bail!(
            "Shape mismatch in fp8_matmul_flashinfer: input [{}, {}], weight [{}, {}]",
            m,
            k,
            n,
            k_w
        );
    }

    if input.dtype() != DType::BF16 {
        candle_core::bail!("fp8_matmul_flashinfer requires bf16 input");
    }
    if weight.dtype() != DType::U8 || weight_scale.dtype() != DType::F32 {
        candle_core::bail!("fp8_matmul_flashinfer requires u8 weights and f32 scales");
    }
    if !input.is_contiguous() {
        candle_core::bail!("fp8_matmul_flashinfer requires contiguous input");
    }
    if !weight.is_contiguous() {
        candle_core::bail!("fp8_matmul_flashinfer requires contiguous row-major weight");
    }
    if !weight_scale.is_contiguous() {
        candle_core::bail!("fp8_matmul_flashinfer requires contiguous row-major weight_scale");
    }
    if k % 128 != 0 {
        candle_core::bail!("fp8_matmul_flashinfer requires K divisible by 128");
    }
    if n % 64 != 0 {
        candle_core::bail!("fp8_matmul_flashinfer requires N divisible by 64");
    }

    let expected_scale = ((n + 127) / 128, k / 128);
    if weight_scale.dims2()? != expected_scale {
        candle_core::bail!(
            "fp8_matmul_flashinfer expects weight_scale shape [{}, {}], got {:?}",
            expected_scale.0,
            expected_scale.1,
            weight_scale.dims()
        );
    }

    let dev = input.device();
    let sm_version = cuda_utils::sm_version(dev.as_cuda_device()?).unwrap_or(0) as usize;
    if !(90..100).contains(&sm_version) {
        candle_core::bail!("fp8_matmul_flashinfer requires Hopper (sm90)");
    }

    let cu_dev = dev.as_cuda_device()?;
    let stream = *cu_dev.cu_stream() as i64;
    let m_padded = (m + 4 - 1) / 4 * 4;
    let out = unsafe { Tensor::empty_((m, n), DType::BF16, dev)? };
    let k_over_128 = k / 128;
    let input_q = unsafe { Tensor::empty_((m, k), DType::U8, dev)? };
    // FlashInfer/DeepGEMM expects scales_a to use an M-aligned leading stride.
    // Their own tests allocate [K/128, M_padded] and treat only the first M columns as live.
    let input_scale = if m_padded == m {
        unsafe { Tensor::empty_((k_over_128, m_padded), DType::F32, dev)? }
    } else {
        Tensor::zeros((k_over_128, m_padded), DType::F32, dev)?
    };
    let scale_stride = input_scale.stride()[0] as i32;
    let q_ptr = get_cuda_slice::<u8>(&input_q)? as *mut std::ffi::c_void;
    let s_ptr = get_cuda_slice::<f32>(&input_scale)? as *mut f32;
    let inp_ptr = get_cuda_slice::<half::bf16>(input)? as *const std::ffi::c_void;

    unsafe {
        let num_groups = m * k_over_128;
        ffi::fp8_quantize_per_token_group_launch(
            inp_ptr,
            q_ptr,
            s_ptr,
            num_groups as i32,
            128,
            k_over_128 as i32,
            scale_stride,
            false,
            true,
            stream,
        );
    }

    let required_ws =
        unsafe { ffi::flashinfer_fp8_blockscale_workspace_size_fp8(m as i32, n as i32, k as i32) };
    let (workspace_ptr, workspace_size) =
        get_or_init_flashinfer_fp8_workspace(cu_dev, required_ws)?;

    let weight_ptr = get_cuda_slice::<u8>(weight)? as *const std::ffi::c_void;
    let weight_scale_ptr = get_cuda_slice::<f32>(weight_scale)? as *const f32;
    let out_ptr = get_cuda_slice::<half::bf16>(&out)? as *mut std::ffi::c_void;

    let status = unsafe {
        ffi::flashinfer_fp8_blockscale_fp8(
            q_ptr as *const std::ffi::c_void,
            s_ptr as *const f32,
            weight_ptr,
            weight_scale_ptr,
            out_ptr,
            m as i32,
            n as i32,
            k as i32,
            workspace_ptr,
            workspace_size,
            stream,
        )
    };
    if status != 0 {
        candle_core::bail!("flashinfer fp8 blockscale gemm failed with status {status}");
    }

    Ok(out)
}

/// FP8 Matrix Multiplication using CUTLASS blockwise kernels (SM90+).
///
/// # Arguments
/// * `input` - Input tensor A of shape [M, K]
/// * `weight` - Weight tensor B of shape [K, N] (stored as u8, column-major)
/// * `weight_scale` - Scales for weight tensor
/// * `block_size` - [block_size_y, block_size_x] for scaling (must be [128, 128])
#[cfg(all(feature = "cuda", feature = "cutlass"))]
#[allow(unused)]
pub fn fp8_matmul_cutlass(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize],
) -> Result<Tensor> {
    if block_size.len() != 2 || block_size[0] != 128 || block_size[1] != 128 {
        candle_core::bail!("fp8_matmul_cutlass requires block_size [128, 128]");
    }

    let (m, k) = input.dims2()?;
    let (k_b, n) = weight.dims2()?;
    if k != k_b {
        candle_core::bail!(
            "mat_a and mat_b shapes cannot be multiplied: K={} vs mat_b.dim(0)={}",
            k,
            weight.dim(0)?
        );
    }

    if input.rank() != 2 {
        candle_core::bail!("mat_a must be a 2D tensor");
    }
    if weight.rank() != 2 {
        candle_core::bail!("mat_b must be a 2D tensor");
    }
    if !input.is_contiguous() {
        candle_core::bail!("mat_a must be contiguous (row major)");
    }
    if weight.stride()[0] != 1 {
        candle_core::bail!("mat_b must be a column major tensor (stride(0) == 1)");
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
    if !(weight_scale_col_major || weight_scale_row_major) {
        candle_core::bail!("scales_b must be column major or contiguous row major");
    }

    let dev = input.device();
    let dtype = input.dtype();
    let scale_row_stride = (k + block_size[1] - 1) / block_size[1];

    let sm_version = if matches!(dev, Device::Cuda(_)) {
        cuda_utils::sm_version(dev.as_cuda_device()?).unwrap_or(0) as i32
    } else {
        80
    };

    let sm90_plus = sm_version >= 90;
    if !sm90_plus {
        candle_core::bail!("fp8_matmul_cutlass requires sm90+");
    }

    if dtype != DType::F16 && dtype != DType::BF16 {
        candle_core::bail!("fp8_matmul_cutlass requires f16 or bf16 input");
    }
    if sm_version >= 100 {
        if !weight_scale_col_major {
            candle_core::bail!("scales_b must be column major for sm100+");
        }
    } else if !weight_scale_row_major {
        candle_core::bail!("scales_b must be contiguous row major for sm90");
    }

    let w_ptr = get_cuda_slice::<u8>(&weight)?;
    let ws_ptr = get_cuda_slice::<f32>(&weight_scale)?;

    let alignment = 4;
    let m_padded = (m + alignment - 1) / alignment * alignment;
    let pad_len = m_padded - m;

    let mut output =
        unsafe { Tensor::empty_((if pad_len > 0 { m_padded } else { m }, n), dtype, dev)? };
    let cu_dev = dev.as_cuda_device()?;
    let stream = *cu_dev.cu_stream() as i64;
    let k_over_128 = (k + 127) / 128;

    let input_q = if pad_len == 0 {
        unsafe { Tensor::empty_((m_padded, k), DType::U8, &dev)? }
    } else {
        Tensor::zeros((m_padded, k), DType::U8, &dev)?
    };
    let input_scale_base = if pad_len == 0 {
        unsafe { Tensor::empty_((k_over_128, m_padded), DType::F32, &dev)? }
    } else {
        Tensor::zeros((k_over_128, m_padded), DType::F32, &dev)?
    };
    let input_scale = input_scale_base.t()?;
    let scale_stride = input_scale.stride()[1] as i32;

    let q_ptr = get_cuda_slice::<u8>(&input_q)? as *mut std::ffi::c_void;
    let s_ptr = get_cuda_slice::<f32>(&input_scale)? as *mut f32;

    let inp_ptr = if dtype == DType::F16 {
        get_cuda_slice::<half::f16>(input)?
    } else {
        get_cuda_slice::<half::bf16>(input)?
    };

    unsafe {
        let num_groups = m * k_over_128;
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

    let (gemm_ws_ptr, gemm_ws_bytes) = get_cutlass_workspace(cu_dev, 0)?;
    let gemm_ws_bytes = gemm_ws_bytes as i64;

    match (dev, dtype) {
        (Device::Cuda(_), DType::F16) => {
            let out_ptr = get_cuda_slice::<half::f16>(&output)?;
            unsafe {
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
                    sm_version,
                    gemm_ws_ptr,
                    gemm_ws_bytes,
                    stream,
                )
            }
        }
        (Device::Cuda(_), DType::BF16) => {
            let out_ptr = get_cuda_slice::<half::bf16>(&output)?;
            unsafe {
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
                    sm_version,
                    gemm_ws_ptr,
                    gemm_ws_bytes,
                    stream,
                )
            }
        }
        (Device::Cuda(_), _) => candle_core::bail!("fp8_matmul_cutlass requires f16 or bf16 input"),
        _ => candle_core::bail!("fp8_matmul_cutlass only supports CUDA"),
    }

    if pad_len > 0 {
        output = output.narrow(0, 0, m)?.contiguous()?;
    }

    Ok(output)
}
