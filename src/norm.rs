//! Fused RMSNorm entry points.
//!
//! Candle's CUDA RMSNorm requires the activation and weight to share a dtype.
//! xinfer keeps norm weights and reduction statistics in F32 for quantized
//! checkpoints, which forces a BF16->F32 cast, the norm, and an F32->BF16 cast
//! on every invocation.  The kernels below keep the reduction and weight in
//! F32 while reading and writing activations in their native dtype, in one
//! launch.  A residual-add variant also fuses the pre-norm broadcast add.

use candle_core::{DType, Result, Tensor};

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use crate::kernels::ffi;

#[cfg(feature = "cuda")]
fn cuda_ptr<T>(tensor: &Tensor) -> Result<u64>
where
    T: candle_core::cuda_backend::cudarc::driver::DeviceRepr
        + candle_core::cuda_backend::CudaDType,
{
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(storage) => {
            let slice = storage.as_cuda_slice::<T>()?;
            Ok(*slice.slice(layout.start_offset()..).device_ptr() as u64)
        }
        _ => candle_core::bail!("norm: expected a CUDA tensor"),
    }
}

/// `y = x * rstd * weight`, computed with F32 statistics and weight.
///
/// `x` must be a contiguous 2-D CUDA tensor and `weight` a contiguous F32
/// vector of length `x.dim(1)`.  Falls back to Candle's generic RMSNorm for
/// other devices/dtypes.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if should_use_fused(x, weight) {
        let (rows, dim) = x.dims2()?;
        let out = unsafe { Tensor::empty_((rows, dim), x.dtype(), x.device())? };
        let stream = *x.device().as_cuda_device()?.cu_stream() as i64;
        unsafe {
            match x.dtype() {
                DType::F32 => ffi::attention_rms_norm_f32(
                    cuda_ptr::<f32>(x)? as *const f32,
                    cuda_ptr::<f32>(weight)? as *const f32,
                    cuda_ptr::<f32>(&out)? as *mut f32,
                    rows as i32,
                    dim as i32,
                    eps,
                    stream,
                ),
                DType::F16 => ffi::attention_rms_norm_f16(
                    cuda_ptr::<half::f16>(x)? as *const core::ffi::c_void,
                    cuda_ptr::<f32>(weight)? as *const f32,
                    cuda_ptr::<half::f16>(&out)? as *mut core::ffi::c_void,
                    rows as i32,
                    dim as i32,
                    eps,
                    stream,
                ),
                DType::BF16 => ffi::attention_rms_norm_bf16(
                    cuda_ptr::<half::bf16>(x)? as *const core::ffi::c_void,
                    cuda_ptr::<f32>(weight)? as *const f32,
                    cuda_ptr::<half::bf16>(&out)? as *mut core::ffi::c_void,
                    rows as i32,
                    dim as i32,
                    eps,
                    stream,
                ),
                dt => candle_core::bail!("rms_norm: unsupported dtype {dt:?}"),
            }
        }
        return Ok(out);
    }

    let weight = if weight.dtype() != x.dtype() {
        weight.to_dtype(x.dtype())?
    } else {
        weight.clone()
    };
    candle_nn::ops::rms_norm(x, &weight, eps)
}

/// Fused residual add + RMSNorm.
///
/// Returns `(normalized, residual_sum)` where `residual_sum` is the
/// dtype-rounded elementwise sum of `x` and `residual`, i.e. exactly the value
/// the reference `(x + residual)?` produces.  `normalized` is computed from
/// that rounded value, matching add-then-normalize semantics.
pub fn add_rms_norm(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if should_use_fused(x, weight) && residual.dims() == x.dims() {
        let (rows, dim) = x.dims2()?;
        let out = unsafe { Tensor::empty_((rows, dim), x.dtype(), x.device())? };
        let sum = unsafe { Tensor::empty_((rows, dim), x.dtype(), x.device())? };
        let residual = if residual.is_contiguous() {
            residual.clone()
        } else {
            residual.contiguous()?
        };
        let stream = *x.device().as_cuda_device()?.cu_stream() as i64;
        unsafe {
            match x.dtype() {
                DType::F32 => ffi::attention_add_rms_norm_f32(
                    cuda_ptr::<f32>(x)? as *const f32,
                    cuda_ptr::<f32>(&residual)? as *const f32,
                    cuda_ptr::<f32>(weight)? as *const f32,
                    cuda_ptr::<f32>(&out)? as *mut f32,
                    cuda_ptr::<f32>(&sum)? as *mut f32,
                    rows as i32,
                    dim as i32,
                    eps,
                    stream,
                ),
                DType::F16 => ffi::attention_add_rms_norm_f16(
                    cuda_ptr::<half::f16>(x)? as *const core::ffi::c_void,
                    cuda_ptr::<half::f16>(&residual)? as *const core::ffi::c_void,
                    cuda_ptr::<f32>(weight)? as *const f32,
                    cuda_ptr::<half::f16>(&out)? as *mut core::ffi::c_void,
                    cuda_ptr::<half::f16>(&sum)? as *mut core::ffi::c_void,
                    rows as i32,
                    dim as i32,
                    eps,
                    stream,
                ),
                DType::BF16 => ffi::attention_add_rms_norm_bf16(
                    cuda_ptr::<half::bf16>(x)? as *const core::ffi::c_void,
                    cuda_ptr::<half::bf16>(&residual)? as *const core::ffi::c_void,
                    cuda_ptr::<f32>(weight)? as *const f32,
                    cuda_ptr::<half::bf16>(&out)? as *mut core::ffi::c_void,
                    cuda_ptr::<half::bf16>(&sum)? as *mut core::ffi::c_void,
                    rows as i32,
                    dim as i32,
                    eps,
                    stream,
                ),
                dt => candle_core::bail!("add_rms_norm: unsupported dtype {dt:?}"),
            }
        }
        return Ok((out, sum));
    }

    let sum = (x + residual)?;
    let normed = rms_norm(&sum, weight, eps)?;
    Ok((normed, sum))
}

#[cfg(feature = "cuda")]
fn should_use_fused(x: &Tensor, weight: &Tensor) -> bool {
    if !matches!(x.device(), candle_core::Device::Cuda(_))
        || !matches!(x.dtype(), DType::F32 | DType::F16 | DType::BF16)
        || weight.dtype() != DType::F32
        || !weight.is_contiguous()
    {
        return false;
    }
    match (x.dims2(), weight.dims()) {
        (Ok((_, dim)), [w]) => *w == dim && x.is_contiguous(),
        _ => false,
    }
}
