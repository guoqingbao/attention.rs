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

    #[cfg(feature = "metal")]
    {
        return silu_and_mul_metal(&gate_up, half_dim);
    }

    #[allow(unreachable_code)]
    {
        silu_and_mul_cpu(&gate_up, half_dim)
    }
}

#[cfg(feature = "metal")]
fn silu_and_mul_metal(gate_up: &Tensor, half_dim: usize) -> Result<Tensor> {
    use candle_core::DType;
    use metal::MTLSize;

    let dtype = gate_up.dtype();
    let dims = gate_up.dims();
    let mut out_dims: Vec<usize> = dims.to_vec();
    *out_dims.last_mut().unwrap() = half_dim;
    let nrows: usize = out_dims[..out_dims.len() - 1]
        .iter()
        .product::<usize>()
        .max(1);
    // The kernel writes every output element.
    let output = unsafe { Tensor::empty_(out_dims.as_slice(), dtype, gate_up.device())? };

    let device = match gate_up.device() {
        candle_core::Device::Metal(dev) => dev,
        _ => candle_core::bail!("silu_and_mul: expected Metal device"),
    };

    let vectors_ok = half_dim % 4 == 0;
    let kernel_name = match (dtype, vectors_ok) {
        (DType::F32, true) => "silu_and_mul_float_vec4",
        (DType::F16, true) => "silu_and_mul_half_vec4",
        (DType::BF16, true) => "silu_and_mul_bfloat_vec4",
        (DType::F32, false) => "silu_and_mul_float",
        (DType::F16, false) => "silu_and_mul_half",
        (DType::BF16, false) => "silu_and_mul_bfloat",
        _ => candle_core::bail!("silu_and_mul: unsupported dtype {dtype:?}"),
    };

    let kernels = crate::metal_kernels::Kernels::default();
    let pipeline = match kernels.load_pipeline(device.device(), kernel_name.to_string()) {
        Ok(p) => p,
        // Older GPUs may lack bf16 support; use the generic activation instead.
        Err(_) => return silu_and_mul_cpu(gate_up, half_dim),
    };

    let command_buffer = device.command_buffer()?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (gu_storage, gu_layout) = gate_up.storage_and_layout();
        let (out_storage, _) = output.storage_and_layout();
        let gu_ms = match &*gu_storage {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("silu_and_mul: expected Metal storage"),
        };
        let out_ms = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("silu_and_mul: expected Metal storage"),
        };
        let gu_offset = (gu_layout.start_offset() * dtype.size_in_bytes()) as u64;
        encoder.set_buffer(0, Some(gu_ms.buffer()), gu_offset);
        encoder.set_buffer(1, Some(out_ms.buffer()), 0);
        encoder.set_bytes(
            2,
            4,
            &(half_dim as i32).to_ne_bytes() as *const _ as *const _,
        );
        let threads = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let total = if vectors_ok {
            let cols4 = half_dim / 4;
            encoder.set_bytes(3, 4, &(cols4 as i32).to_ne_bytes() as *const _ as *const _);
            (nrows * cols4) as u64
        } else {
            let total = nrows * half_dim;
            encoder.set_bytes(3, 4, &(total as i32).to_ne_bytes() as *const _ as *const _);
            total as u64
        };
        let groups = MTLSize {
            width: total.div_ceil(256),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(groups, threads);
    }
    // The encoder is borrowed from the command buffer; close it so subsequent
    // candle ops can open their own encoder on the same command buffer.
    encoder.end_encoding();
    Ok(output)
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
            DType::F32 => {
                let s = gu_cuda.as_cuda_slice::<f32>()?;
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
        DType::F32 => {
            let output = unsafe { dev.alloc::<f32>(total_elems) }.w()?;
            let out_ptr = *output.device_ptr() as *mut core::ffi::c_void;
            unsafe {
                ffi::silu_and_mul_f32(
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
