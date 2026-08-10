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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8ExecutionDomain {
    Eager,
    DecodeGraph,
    MtpGraph,
}

thread_local! {
    static FP8_EXECUTION_DOMAIN: std::cell::Cell<Fp8ExecutionDomain> =
        const { std::cell::Cell::new(Fp8ExecutionDomain::Eager) };
}

pub struct Fp8ExecutionGuard {
    previous: Fp8ExecutionDomain,
}

impl Drop for Fp8ExecutionGuard {
    fn drop(&mut self) {
        FP8_EXECUTION_DOMAIN.with(|domain| domain.set(self.previous));
    }
}

pub fn set_fp8_execution_domain(domain: Fp8ExecutionDomain) -> Fp8ExecutionGuard {
    let previous = FP8_EXECUTION_DOMAIN.with(|current| {
        let previous = current.get();
        current.set(domain);
        previous
    });
    Fp8ExecutionGuard { previous }
}

#[cfg(feature = "cuda")]
pub(crate) fn fp8_execution_domain() -> Fp8ExecutionDomain {
    FP8_EXECUTION_DOMAIN.with(|domain| domain.get())
}

#[cfg(feature = "cuda")]
fn get_cuda_slice<
    T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType,
>(
    tensor: &Tensor,
) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(c) => {
            let slice = c.as_cuda_slice::<T>()?;
            Ok(*slice.slice(layout.start_offset()..).device_ptr() as u64)
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
#[allow(unused)]
pub fn fp8_matmul(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    weight_scale_cutlass: Option<&Tensor>,
    block_size: &[usize],
    is_prefill: bool,
) -> Result<Tensor> {
    fp8_matmul_with_input_scale(
        input,
        weight,
        weight_scale,
        weight_scale_cutlass,
        block_size,
        None,
        is_prefill,
    )
}

/// FP8 matrix multiplication with an optional static activation scale.
///
/// `input_scale` is the dequantization scale used by ModelOpt static FP8
/// activation exports (normally `amax / 448`). When present, FlashInfer and
/// CUTLASS quantize every 128-element activation group with this same scale;
/// when absent, they retain the existing per-token dynamic quantization.
/// The conventional/fallback path remains W8A16 and does not quantize the
/// activation.
#[allow(unused)]
pub fn fp8_matmul_with_input_scale(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    weight_scale_cutlass: Option<&Tensor>,
    block_size: &[usize],
    input_scale: Option<f32>,
    is_prefill: bool,
) -> Result<Tensor> {
    if let Some(scale) = input_scale {
        if !scale.is_finite() || scale <= 0.0 {
            candle_core::bail!("FP8 input_scale must be finite and positive, got {scale}");
        }
    }

    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };

    #[cfg(feature = "cuda")]
    let sm_version = if let Ok(cuda_dev) = input.device().as_cuda_device() {
        crate::cuda_utils::sm_version(cuda_dev).unwrap_or(0) as usize
    } else {
        0
    };

    #[cfg(all(feature = "cuda", feature = "flashinfer"))]
    {
        let (m, _) = input.dims2()?;
        // Enable only for decode phase (small M <= 64) on SM90 with BF16 and 128x128 block scales
        let use_flashinfer = !is_prefill
            && m <= 64
            && (90..100).contains(&sm_version)
            && DType::BF16 == input.dtype()
            && block_size == [128, 128];
        if use_flashinfer {
            return fp8_matmul_flashinfer_with_input_scale(
                &input,
                weight,
                weight_scale,
                input_scale,
            );
        }
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    {
        let use_cutlass = sm_version >= 90 && block_size == [128, 128];
        if use_cutlass {
            let cutlass_scale = match weight_scale_cutlass {
                Some(s) => s.clone(),
                None => {
                    if sm_version >= 100 {
                        weight_scale.t()?
                    } else {
                        weight_scale.t()?.contiguous()?
                    }
                }
            };
            return fp8_matmul_cutlass_with_input_scale(
                &input,
                &weight.t()?,
                &cutlass_scale,
                block_size,
                input_scale,
            );
        }
    }

    fp8_matmul_fallback(&input, weight, weight_scale, block_size)
}

/// FP8 matrix multiplication for checkpoints whose activation scales use
/// UE8M0 (power-of-two) rounding, such as DeepSeek-V4-Flash.
///
/// The regular SM90 CUTLASS path computes unrounded `amax / 448` activation
/// scales. That is a different quantizer from UE8M0 QAT and causes material
/// model drift. Quantize/dequantize the BF16 activation with the same
/// power-of-two rule as the checkpoint, then multiply by the dequantized FP8
/// weights. This path is the numerical reference until the optimized CUTLASS
/// kernel accepts UE8M0 activation scales directly.
#[cfg(feature = "cuda")]
pub fn fp8_matmul_ue8m0(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize],
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 || block_size != [128, 128] {
        candle_core::bail!("UE8M0 FP8 matmul requires BF16 input and 128x128 weight blocks");
    }
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let (m, k) = input.dims2()?;
    let quantized = Tensor::zeros((m, k), DType::BF16, input.device())?;
    crate::deepseek_v4::copy_contiguous_into(&quantized, &input, 0)?;
    crate::deepseek_v4::fp8_act_quant_nope_bf16_inplace(&quantized, m, k, 0, 128)?;
    fp8_matmul_fallback(&quantized, weight, weight_scale, block_size)
}

#[cfg(not(feature = "cuda"))]
pub fn fp8_matmul_ue8m0(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize],
) -> Result<Tensor> {
    let _ = (input, weight, weight_scale, block_size);
    candle_core::bail!("UE8M0 FP8 matmul requires cuda feature")
}

/// FP8 Matrix Multiplication: C = A * B^T (conventional path).
///
/// # Arguments
/// * `input` - Input tensor A of shape [M, K]
/// * `weight` - Weight tensor B of shape [N, K] (stored as F8E4M3 or raw U8 bytes)
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
    let scale_is_e8m0 = weight_scale.dtype() == DType::F8E8M0;
    assert!(
        weight_scale.dtype() == DType::F32 || scale_is_e8m0,
        "fp8_matmul expects f32 or F8E8M0 scales, got {:?}",
        weight_scale.dtype()
    );
    let scale_row_stride = (k_w + block_size[1] - 1) / block_size[1];

    #[cfg(feature = "cuda")]
    let output = unsafe { Tensor::empty_((m, n), dtype, dev)? };
    #[cfg(not(feature = "cuda"))]
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
            let scale_ptr = match &*scale_storage {
                candle_core::Storage::Cuda(c) => {
                    if scale_is_e8m0 {
                        *c.as_cuda_slice::<u8>()?.device_ptr() as *const core::ffi::c_void
                    } else {
                        *c.as_cuda_slice::<f32>()?.device_ptr() as *const core::ffi::c_void
                    }
                }
                _ => candle_core::bail!("weight_scale must be a cuda tensor"),
            };
            let scale_dtype = if scale_is_e8m0 { 1i32 } else { 0i32 };

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
                    scale_ptr,
                    output_ptr,
                    m as i32,
                    n as i32,
                    k as i32,
                    scale_row_stride as i32,
                    block_size[0] as i32,
                    block_size[1] as i32,
                    scale_dtype,
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
            let scale_ptr = match &*scale_storage {
                candle_core::Storage::Cuda(c) => {
                    if scale_is_e8m0 {
                        *c.as_cuda_slice::<u8>()?.device_ptr() as *const core::ffi::c_void
                    } else {
                        *c.as_cuda_slice::<f32>()?.device_ptr() as *const core::ffi::c_void
                    }
                }
                _ => candle_core::bail!("weight_scale must be a cuda tensor"),
            };
            let scale_dtype = if scale_is_e8m0 { 1i32 } else { 0i32 };

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
                    scale_ptr,
                    output_ptr,
                    m as i32,
                    n as i32,
                    k as i32,
                    scale_row_stride as i32,
                    block_size[0] as i32,
                    block_size[1] as i32,
                    scale_dtype,
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
    fp8_matmul_flashinfer_with_input_scale(input, weight, weight_scale, None)
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn fp8_matmul_flashinfer_with_input_scale(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    input_scale: Option<f32>,
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
    if !matches!(weight.dtype(), DType::U8 | DType::F8E4M3) || weight_scale.dtype() != DType::F32 {
        candle_core::bail!("fp8_matmul_flashinfer requires F8E4M3/U8 weights and f32 scales");
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
    // FlashInfer/DeepGEMM expects scales_a to use an M-aligned leading stride.
    // Keep both asynchronous inputs in the per-device pool: call-local tensors
    // can be reclaimed as soon as this function returns, before the GEMM has
    // finished reading them.
    let input_q_bytes = m * k;
    let input_scales_bytes = k_over_128 * m_padded * std::mem::size_of::<f32>();
    let (q_ptr, s_ptr) = get_grouped_gemm_scratch(cu_dev, input_q_bytes, input_scales_bytes)?;
    let s_ptr = s_ptr as *mut f32;
    let scale_stride = m_padded as i32;
    let inp_ptr = get_cuda_slice::<half::bf16>(input)? as *const std::ffi::c_void;

    unsafe {
        let num_groups = m * k_over_128;
        if let Some(input_scale) = input_scale {
            ffi::fp8_quantize_per_token_group_static_launch(
                inp_ptr,
                q_ptr,
                s_ptr,
                num_groups as i32,
                128,
                k_over_128 as i32,
                scale_stride,
                false,
                true,
                input_scale,
                stream,
            );
        } else {
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
    fp8_matmul_cutlass_with_input_scale(input, weight, weight_scale, block_size, None)
}

#[cfg(all(feature = "cuda", feature = "cutlass"))]
fn fp8_matmul_cutlass_with_input_scale(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    block_size: &[usize],
    input_scale: Option<f32>,
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

    let expected_scale_dim0 = (weight.dim(0)? + block_size[0] - 1) / block_size[0];
    let expected_scale_dim1 = (weight.dim(1)? + block_size[1] - 1) / block_size[1];
    if weight_scale.dim(0)? != expected_scale_dim0 || weight_scale.dim(1)? != expected_scale_dim1 {
        candle_core::bail!(
            "scales_b shape mismatch: expected [{}, {}], got [{}, {}] for weight [{}, {}] with block_size {:?}",
            expected_scale_dim0,
            expected_scale_dim1,
            weight_scale.dim(0)?,
            weight_scale.dim(1)?,
            weight.dim(0)?,
            weight.dim(1)?,
            block_size
        );
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

    let input_q_bytes = m_padded * k;
    let input_scales_bytes = k_over_128 * m_padded * std::mem::size_of::<f32>();
    let (q_ptr, s_ptr) = get_grouped_gemm_scratch(cu_dev, input_q_bytes, input_scales_bytes)?;
    let s_ptr = s_ptr as *mut f32;
    let scale_stride = m_padded as i32;

    let inp_ptr = if dtype == DType::F16 {
        get_cuda_slice::<half::f16>(input)?
    } else {
        get_cuda_slice::<half::bf16>(input)?
    };

    unsafe {
        let num_groups = m * k_over_128;
        let group_size = 128;
        let num_groups_per_row = k_over_128;
        if let Some(input_scale) = input_scale {
            ffi::fp8_quantize_per_token_group_static_launch(
                inp_ptr as *const std::ffi::c_void,
                q_ptr,
                s_ptr,
                num_groups as i32,
                group_size as i32,
                num_groups_per_row as i32,
                scale_stride,
                dtype == DType::F16,
                true,
                input_scale,
                stream as i64,
            );
        } else {
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

    if std::env::var_os("XINFER_DSV4_SYNC_FP8").is_some() {
        input.device().synchronize()?;
    }

    if pad_len > 0 {
        output = output.narrow(0, 0, m)?.contiguous()?;
    }

    Ok(output)
}

/// Fused grouped FP8 matmul for block-diagonal linear layers.
///
/// Computes: for each group g in [0..n_groups):
///   output[g] = input[g] @ weight[g].T
///
/// where input is [n_groups, seq_len, k] and weight is [n_groups * n, k] (stacked).
///
/// Uses CUTLASS grouped GEMM (single kernel launch) on SM90+ for seq_len > 1,
/// falls back to per-group fp8_matmul otherwise.
///
/// # Arguments
/// * `input` - BF16 tensor [n_groups, seq_len, k] (contiguous, already grouped)
/// * `group_weights` - Vec of FP8 weight tensors [n, k] per group
/// * `group_scales` - Vec of F32/E8M0 scale tensors per group
/// * `group_scales_cutlass` - Vec of optional pre-transposed scale tensors per group
/// * `block_size` - [block_size_y, block_size_x]
/// * `is_prefill` - whether in prefill phase
#[allow(unused)]
pub fn fp8_grouped_matmul(
    input: &Tensor,
    group_weights: &[Tensor],
    group_scales: &[Tensor],
    group_scales_cutlass: &[Option<Tensor>],
    block_size: &[usize],
    is_prefill: bool,
) -> Result<Tensor> {
    let n_groups = group_weights.len();
    if n_groups == 0 {
        candle_core::bail!("fp8_grouped_matmul: no groups provided");
    }
    let (n_groups_in, seq_len, k) = input.dims3()?;
    if n_groups_in != n_groups {
        candle_core::bail!(
            "fp8_grouped_matmul: input groups {} != weight groups {}",
            n_groups_in,
            n_groups
        );
    }
    let n = group_weights[0].dim(0)?;

    #[cfg(feature = "cuda")]
    let sm_version = if let Ok(cuda_dev) = input.device().as_cuda_device() {
        cuda_utils::sm_version(cuda_dev).unwrap_or(0) as usize
    } else {
        0
    };
    #[cfg(not(feature = "cuda"))]
    let sm_version = 0;

    // For SM90+ CUTLASS with [128,128] block size, use fused grouped GEMM during prefill.
    // Decode uses per-group path which is CUDA graph compatible (no dynamic allocations).
    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    if is_prefill
        && sm_version >= 90
        && block_size == [128, 128]
        && group_scales[0].dtype() == candle_core::DType::F32
    {
        return fp8_grouped_matmul_cutlass(
            input,
            group_weights,
            group_scales,
            n_groups,
            seq_len,
            n,
            k,
            sm_version,
        );
    }

    // Fallback: per-group fp8_matmul (no extra allocations needed)
    let mut group_outputs = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let g_input = input.narrow(0, g, 1)?.squeeze(0)?;
        let out = fp8_matmul(
            &g_input,
            &group_weights[g],
            &group_scales[g],
            group_scales_cutlass[g].as_ref(),
            block_size,
            is_prefill,
        )?;
        group_outputs.push(out);
    }
    Tensor::cat(&group_outputs, 1)
}

/// Fused grouped GEMM using CUTLASS for SM90+ (single kernel launch).
#[cfg(all(feature = "cuda", feature = "cutlass"))]
fn fp8_grouped_matmul_cutlass(
    input: &Tensor,
    group_weights: &[Tensor],
    group_scales: &[Tensor],
    n_groups: usize,
    seq_len: usize,
    n: usize,
    k: usize,
    sm_version: usize,
) -> Result<Tensor> {
    use candle_core::cuda_backend::WrapErr;

    let device = input.device().clone();
    let total_m = n_groups * seq_len;
    let k_blocks = (k + 128 - 1) / 128;
    let is_column_major_scales = sm_version >= 100;

    // 1. Reshape input [n_groups, seq_len, k] -> [n_groups * seq_len, k] as BF16
    let input_flat = input.reshape((total_m, k))?.contiguous()?;

    // 2. Quantize BF16 input to FP8 with per-token-group scales
    let input_q = unsafe { Tensor::empty_((total_m, k), DType::U8, &device)? };
    let input_scale = if is_column_major_scales {
        unsafe { Tensor::empty_((k_blocks, total_m), DType::F32, &device)? }.t()?
    } else {
        unsafe { Tensor::empty_((total_m, k_blocks), DType::F32, &device)? }
    };

    #[cfg(feature = "cuda")]
    {
        let cuda_dev = device.as_cuda_device()?;
        let stream = *cuda_dev.cu_stream() as i64;

        let (input_storage, _input_layout) = input_flat.storage_and_layout();
        let input_ptr = match &*input_storage {
            candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
            _ => candle_core::bail!("input must be cuda"),
        };
        let (iq_storage, _) = input_q.storage_and_layout();
        let iq_ptr = match &*iq_storage {
            candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
            _ => candle_core::bail!("input_q must be cuda"),
        };
        let (is_storage, _) = input_scale.storage_and_layout();
        let is_ptr = match &*is_storage {
            candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
            _ => candle_core::bail!("input_scale must be cuda"),
        };

        let num_groups_quant = (total_m * k_blocks) as i32;
        unsafe {
            ffi::fp8_quantize_per_token_group_launch(
                input_ptr as *const std::ffi::c_void,
                iq_ptr as *mut std::ffi::c_void,
                is_ptr as *mut f32,
                num_groups_quant,
                128, // group_size
                k_blocks as i32,
                if is_column_major_scales {
                    total_m as i32
                } else {
                    k_blocks as i32
                },
                false, // is_input_f16 (BF16)
                is_column_major_scales,
                stream,
            );
        }

        // 3. Stack weights and scales (they're already contiguous per group from pre-slicing)
        let stacked_weights = Tensor::cat(group_weights, 0)?; // [n_groups * n, k]
        let stacked_scales = Tensor::cat(group_scales, 0)?; // [n_groups * scale_n, scale_k]

        // 4. Create expert_offsets: [0, seq_len, 2*seq_len, ..., n_groups*seq_len]
        let cuda_dev = device.as_cuda_device()?;
        let offsets: Vec<i32> = (0..=n_groups).map(|g| (g * seq_len) as i32).collect();
        let expert_offsets = cuda_dev.htod_sync_copy(&offsets).w()?;

        // 5. Allocate output
        let output = Tensor::zeros((total_m, n), DType::BF16, &device)?;

        let (sw_storage, _) = stacked_weights.storage_and_layout();
        let sw_ptr = match &*sw_storage {
            candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<u8>()?.device_ptr(),
            _ => candle_core::bail!("stacked_weights must be cuda"),
        };
        let (ss_storage, _) = stacked_scales.storage_and_layout();
        let ss_ptr = match &*ss_storage {
            candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr(),
            _ => candle_core::bail!("stacked_scales must be cuda"),
        };
        let eo_ptr = *expert_offsets.device_ptr();
        let (out_storage, _) = output.storage_and_layout();
        let out_ptr = match &*out_storage {
            candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<half::bf16>()?.device_ptr(),
            _ => candle_core::bail!("output must be cuda"),
        };

        unsafe {
            ffi::moe_fp8_grouped_gemm_bf16(
                iq_ptr as *const u8,
                sw_ptr as *const u8,
                is_ptr as *const f32,
                ss_ptr as *const f32,
                eo_ptr as *const i32,
                n_groups as i32,
                total_m as i32,
                n as i32,
                k as i32,
                128, // block_size_n
                128, // block_size_k
                sm_version as i32,
                out_ptr as *mut std::ffi::c_void,
                stream,
            );
        }

        // 6. Reshape output [n_groups * seq_len, n] -> [seq_len, n_groups * n]
        let output = output.reshape((n_groups, seq_len, n))?;
        let output = output.transpose(0, 1)?.contiguous()?;
        return output.reshape((seq_len, n_groups * n));
    }

    #[cfg(not(feature = "cuda"))]
    candle_core::bail!("fp8_grouped_matmul_cutlass requires CUDA")
}

/// FP8 grouped matmul via FlashInfer strided batch GEMM (CUDA graph safe).
///
/// This performs a batched FP8 GEMM using a single kernel call:
/// 1. Quantize BF16 activations to FP8 with 1x128 column-wise scales
/// 2. Run strided batched FP8 GEMM with block-wise weight scales
///
/// Quantization scratch is kept in a per-device grow-only pool. The FlashInfer
/// GEMM is asynchronous, so call-local Tensor scratch would be freed as this
/// function returns while the kernel can still be reading it.
///
/// input: [n_groups, seq_len, k] BF16 (contiguous)
/// group_weights: pre-stacked [n_groups, n, k] U8 (FP8_E4M3, contiguous)  
/// group_scales: pre-stacked [n_groups, ceil(n/128), ceil(k/128)] F32 (contiguous)
/// Returns: [n_groups, seq_len, n] BF16
#[cfg(feature = "flashinfer")]
pub fn fp8_grouped_matmul_strided(
    input: &Tensor,
    stacked_weights: &Tensor,
    stacked_scales: &Tensor,
    n_groups: usize,
    seq_len: usize,
    n: usize,
    k: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let dev = input.device().as_cuda_device()?;
        let stream = *dev.cu_stream() as i64;

        // Allocate output: [n_groups, seq_len, n] BF16
        let output = Tensor::zeros((n_groups, seq_len, n), DType::BF16, input.device())?;

        let scale_k = (k + 127) / 128;
        let total_rows = n_groups * seq_len;
        let input_q_bytes = total_rows * k;
        let input_scales_bytes = total_rows * scale_k * std::mem::size_of::<f32>();
        let (input_q_ptr, input_scales_ptr) =
            get_grouped_gemm_scratch(dev, input_q_bytes, input_scales_bytes)?;
        let input_scales_ptr = input_scales_ptr as *mut f32;

        // Get raw pointers via storage access (scoped to drop guards before Ok(output))
        let input_ptr = {
            let (s, _) = input.storage_and_layout();
            match &*s {
                candle_core::Storage::Cuda(c) => {
                    *c.as_cuda_slice::<half::bf16>()?.device_ptr() as *const std::ffi::c_void
                }
                _ => candle_core::bail!("input must be a CUDA tensor"),
            }
        };

        let output_ptr = {
            let (s, _) = output.storage_and_layout();
            match &*s {
                candle_core::Storage::Cuda(c) => {
                    *c.as_cuda_slice::<half::bf16>()?.device_ptr() as *mut std::ffi::c_void
                }
                _ => candle_core::bail!("output must be a CUDA tensor"),
            }
        };

        let weights_ptr = {
            let (s, _) = stacked_weights.storage_and_layout();
            match &*s {
                candle_core::Storage::Cuda(c) => {
                    *c.as_cuda_slice::<u8>()?.device_ptr() as *const std::ffi::c_void
                }
                _ => candle_core::bail!("weights must be a CUDA tensor"),
            }
        };

        let scales_ptr = {
            let (s, _) = stacked_scales.storage_and_layout();
            match &*s {
                candle_core::Storage::Cuda(c) => {
                    *c.as_cuda_slice::<f32>()?.device_ptr() as *const f32
                }
                _ => candle_core::bail!("scales must be a CUDA tensor"),
            }
        };

        // Step 1: Quantize BF16 input to FP8 with 1x128 column-wise scales
        let ret = unsafe {
            kernels::ffi::flashinfer_fp8_quantize_1x128(
                input_q_ptr,
                input_scales_ptr,
                input_ptr,
                total_rows as i32,
                k as i32,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("flashinfer_fp8_quantize_1x128 failed with code {}", ret);
        }

        // Step 2: Strided batched FP8 GEMM
        let ld_a = k as i32;
        let stride_a = (seq_len * k) as i32;
        let ld_b = k as i32;
        let stride_b = (n * k) as i32;
        let ld_d = n as i32;
        let stride_d = (seq_len * n) as i32;
        let stride_scales_a = (seq_len * scale_k) as i32;

        let ret = unsafe {
            kernels::ffi::flashinfer_fp8_stride_batch_gemm(
                output_ptr,
                ld_d,
                stride_d,
                input_q_ptr,
                ld_a,
                stride_a,
                weights_ptr,
                ld_b,
                stride_b,
                n_groups as i32,
                seq_len as i32,
                n as i32,
                k as i32,
                input_scales_ptr,
                stride_scales_a,
                scales_ptr,
                stream,
            )
        };
        if ret != 0 {
            candle_core::bail!("flashinfer_fp8_stride_batch_gemm failed with code {}", ret);
        }

        Ok(output)
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            input,
            stacked_weights,
            stacked_scales,
            n_groups,
            seq_len,
            n,
            k,
        );
        candle_core::bail!("fp8_grouped_matmul_strided requires cuda feature")
    }
}

/// Fused grouped FP8 GEMM using CUTLASS (no FlashInfer dependency, CUDA graph safe).
///
/// Quantizes BF16 input to FP8 in one kernel, then runs per-group CUTLASS FP8 GEMM.
/// Uses pooled scratch buffers to avoid per-call allocations (required for CUDA graph).
///
/// - input: [n_groups, seq_len, k] BF16
/// - stacked_weights: [n_groups, n, k] U8 (FP8_E4M3)
/// - stacked_scales: [n_groups, ceil(n/128), ceil(k/128)] F32
/// - Returns: [n_groups, seq_len, n] BF16
#[cfg(all(feature = "cuda", feature = "cutlass"))]
pub fn fp8_grouped_gemm_fused(
    input: &Tensor,
    stacked_weights: &Tensor,
    stacked_scales: &Tensor,
    n_groups: usize,
    seq_len: usize,
    n: usize,
    k: usize,
) -> Result<Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let dev = input.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;
    let sm_version = cuda_utils::sm_version(dev).unwrap_or(0) as i32;

    let output = Tensor::zeros((n_groups, seq_len, n), DType::BF16, input.device())?;

    let total_rows = n_groups * seq_len;
    let scale_k = (k + 127) / 128;
    let input_q_bytes = total_rows * k;
    let input_scales_bytes = total_rows * scale_k * 4; // f32

    // Get pooled scratch buffers (grow-only, never freed during execution)
    let (input_q_ptr, input_scales_ptr) =
        get_grouped_gemm_scratch(dev, input_q_bytes, input_scales_bytes)?;

    // Reshape input to [n_groups * seq_len, k] for quantization
    let input_flat = input.reshape((total_rows, k))?.contiguous()?;

    let input_ptr = {
        let (s, _) = input_flat.storage_and_layout();
        match &*s {
            candle_core::Storage::Cuda(c) => {
                *c.as_cuda_slice::<half::bf16>()?.device_ptr() as *const std::ffi::c_void
            }
            _ => candle_core::bail!("input must be a CUDA tensor"),
        }
    };

    let output_ptr = {
        let (s, _) = output.storage_and_layout();
        match &*s {
            candle_core::Storage::Cuda(c) => {
                *c.as_cuda_slice::<half::bf16>()?.device_ptr() as *mut std::ffi::c_void
            }
            _ => candle_core::bail!("output must be a CUDA tensor"),
        }
    };

    let weights_ptr = {
        let (s, _) = stacked_weights.storage_and_layout();
        match &*s {
            candle_core::Storage::Cuda(c) => {
                *c.as_cuda_slice::<u8>()?.device_ptr() as *const std::ffi::c_void
            }
            _ => candle_core::bail!("weights must be a CUDA tensor"),
        }
    };

    let scales_ptr = {
        let (s, _) = stacked_scales.storage_and_layout();
        match &*s {
            candle_core::Storage::Cuda(c) => *c.as_cuda_slice::<f32>()?.device_ptr() as *const f32,
            _ => candle_core::bail!("scales must be a CUDA tensor"),
        }
    };

    let (ws_ptr, ws_size) = crate::workspace::get_cutlass_workspace(dev, 0)?;

    let ret = unsafe {
        kernels::ffi::fp8_grouped_gemm_fused(
            input_ptr,
            input_q_ptr,
            input_scales_ptr as *mut f32,
            weights_ptr,
            scales_ptr,
            output_ptr,
            n_groups as i32,
            seq_len as i32,
            n as i32,
            k as i32,
            sm_version,
            ws_ptr,
            ws_size as i64,
            stream,
        )
    };
    if ret != 0 {
        candle_core::bail!("fp8_grouped_gemm_fused failed with code {}", ret);
    }
    if std::env::var_os("XINFER_DSV4_SYNC_FP8").is_some() {
        input.device().synchronize()?;
    }

    Ok(output)
}

/// Thread-local scratch buffer pool for grouped GEMM (avoids per-call allocations).
#[cfg(all(feature = "cuda", any(feature = "flashinfer", feature = "cutlass")))]
mod grouped_gemm_pool {
    use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};
    use candle_core::cuda_backend::WrapErr;
    use candle_core::Result;

    struct ScratchPool {
        input_q: CudaSlice<u8>,
        input_q_bytes: usize,
        input_scales: CudaSlice<u8>,
        input_scales_bytes: usize,
        device_ordinal: usize,
    }

    thread_local! {
        static POOL_EAGER: std::cell::RefCell<Option<ScratchPool>> = const { std::cell::RefCell::new(None) };
        static POOL_DECODE_GRAPH: std::cell::RefCell<Option<ScratchPool>> = const { std::cell::RefCell::new(None) };
        static POOL_MTP_GRAPH: std::cell::RefCell<Option<ScratchPool>> = const { std::cell::RefCell::new(None) };
    }

    fn get_from_pool(
        cell: &std::cell::RefCell<Option<ScratchPool>>,
        dev: &candle_core::cuda_backend::CudaDevice,
        input_q_bytes: usize,
        input_scales_bytes: usize,
    ) -> Result<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
        let mut slot = cell.borrow_mut();
        let ordinal = dev.ordinal();

        let needs_realloc = match slot.as_ref() {
            None => true,
            Some(p) => {
                p.device_ordinal != ordinal
                    || p.input_q_bytes < input_q_bytes
                    || p.input_scales_bytes < input_scales_bytes
            }
        };

        if needs_realloc {
            let old = slot.take();
            let (q_sz, s_sz) = if let Some(ref prev) = old {
                (
                    input_q_bytes.max(prev.input_q_bytes),
                    input_scales_bytes.max(prev.input_scales_bytes),
                )
            } else {
                (input_q_bytes, input_scales_bytes)
            };
            drop(old);
            let input_q = unsafe { dev.alloc::<u8>(q_sz.max(1)) }.w()?;
            let input_scales = unsafe { dev.alloc::<u8>(s_sz.max(1)) }.w()?;
            *slot = Some(ScratchPool {
                input_q,
                input_q_bytes: q_sz,
                input_scales,
                input_scales_bytes: s_sz,
                device_ordinal: ordinal,
            });
        }

        let pool = slot.as_ref().unwrap();
        Ok((
            *pool.input_q.device_ptr() as *mut std::ffi::c_void,
            *pool.input_scales.device_ptr() as *mut std::ffi::c_void,
        ))
    }

    pub fn get_grouped_gemm_scratch(
        dev: &candle_core::cuda_backend::CudaDevice,
        input_q_bytes: usize,
        input_scales_bytes: usize,
    ) -> Result<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
        match crate::fp8_linear::fp8_execution_domain() {
            crate::fp8_linear::Fp8ExecutionDomain::Eager => {
                POOL_EAGER.with(|cell| get_from_pool(cell, dev, input_q_bytes, input_scales_bytes))
            }
            crate::fp8_linear::Fp8ExecutionDomain::DecodeGraph => POOL_DECODE_GRAPH
                .with(|cell| get_from_pool(cell, dev, input_q_bytes, input_scales_bytes)),
            crate::fp8_linear::Fp8ExecutionDomain::MtpGraph => POOL_MTP_GRAPH
                .with(|cell| get_from_pool(cell, dev, input_q_bytes, input_scales_bytes)),
        }
    }
}

#[cfg(all(feature = "cuda", any(feature = "flashinfer", feature = "cutlass")))]
use grouped_gemm_pool::get_grouped_gemm_scratch;
