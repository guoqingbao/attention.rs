#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{Result, Tensor};

/// Fused SiLU-and-Mul: given a `gate_up` tensor with last dim `2*N`, computes:
///   output[..., j] = silu(gate_up[..., j]) * gate_up[..., j + N]
///
/// This replaces: narrow(gate) + narrow(up) + contiguous + silu(gate) * up
/// saving 3 intermediate tensor allocations and kernel launches.
pub fn silu_and_mul(gate_up: &Tensor, half_dim: usize) -> Result<Tensor> {
    let gate_up = if gate_up.is_contiguous() {
        gate_up.clone()
    } else {
        gate_up.contiguous()?
    };

    let last_dim = *gate_up.dims().last().unwrap();
    if last_dim != 2 * half_dim {
        candle_core::bail!(
            "silu_and_mul: last dim {} != 2 * half_dim {}",
            last_dim,
            half_dim
        );
    }

    #[cfg(feature = "cuda")]
    {
        return silu_and_mul_cuda(&gate_up, half_dim);
    }

    #[allow(unreachable_code)]
    {
        silu_and_mul_cpu(&gate_up, half_dim)
    }
}

#[cfg(feature = "cuda")]
fn silu_and_mul_cuda(gate_up: &Tensor, half_dim: usize) -> Result<Tensor> {
    use candle_core as candle;
    use candle_core::cuda_backend::WrapErr;
    use candle_core::DType;

    let dtype = gate_up.dtype();
    let dims = gate_up.dims();
    let mut out_dims: Vec<usize> = dims.to_vec();
    *out_dims.last_mut().unwrap() = half_dim;
    let total_elems: usize = out_dims.iter().product();
    let out_shape = candle_core::Shape::from_dims(&out_dims);

    let dev = match gate_up.device() {
        candle_core::Device::Cuda(dev) => dev.clone(),
        _ => candle_core::bail!("Expected CUDA device"),
    };

    let (gu_ptr, stream) = {
        let (gu_storage, gu_layout) = gate_up.storage_and_layout();
        let gu_cuda = match &*gu_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Expected CUDA storage for gate_up"),
        };
        let stream = *gu_cuda.device.cu_stream() as i64;
        let ptr = match dtype {
            DType::F16 => {
                let s = gu_cuda.as_cuda_slice::<half::f16>()?;
                *s.slice(gu_layout.start_offset()..).device_ptr() as u64
            }
            DType::BF16 => {
                let s = gu_cuda.as_cuda_slice::<half::bf16>()?;
                *s.slice(gu_layout.start_offset()..).device_ptr() as u64
            }
            _ => candle_core::bail!("silu_and_mul: unsupported dtype {:?}", dtype),
        };
        (ptr, stream)
    };

    match dtype {
        DType::F16 => {
            let output = unsafe { dev.alloc::<half::f16>(total_elems) }.w()?;
            let out_ptr = *output.device_ptr() as *mut core::ffi::c_void;
            unsafe {
                ffi::silu_and_mul_f16(
                    gu_ptr as *const core::ffi::c_void,
                    out_ptr,
                    total_elems as i64,
                    half_dim as i64,
                    stream,
                );
            }
            let storage = candle::CudaStorage::wrap_cuda_slice(output, dev);
            Ok(Tensor::from_storage(
                candle::Storage::Cuda(storage),
                out_shape,
            )?)
        }
        DType::BF16 => {
            let output = unsafe { dev.alloc::<half::bf16>(total_elems) }.w()?;
            let out_ptr = *output.device_ptr() as *mut core::ffi::c_void;
            unsafe {
                ffi::silu_and_mul_bf16(
                    gu_ptr as *const core::ffi::c_void,
                    out_ptr,
                    total_elems as i64,
                    half_dim as i64,
                    stream,
                );
            }
            let storage = candle::CudaStorage::wrap_cuda_slice(output, dev);
            Ok(Tensor::from_storage(
                candle::Storage::Cuda(storage),
                out_shape,
            )?)
        }
        _ => unreachable!(),
    }
}

#[allow(dead_code)]
fn silu_and_mul_cpu(gate_up: &Tensor, half_dim: usize) -> Result<Tensor> {
    let gate = gate_up.narrow(candle_core::D::Minus1, 0, half_dim)?;
    let up = gate_up.narrow(candle_core::D::Minus1, half_dim, half_dim)?;
    let silu_gate = candle_nn::ops::silu(&gate)?;
    silu_gate.mul(&up)
}
