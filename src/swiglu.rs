#[cfg(feature = "cuda")]
use crate::kernels::ffi;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{Result, Tensor};

/// GPT-OSS custom SwiGLU activation
///
/// Formula: output = (clamp(up, -limit, limit) + 1) * min(gate, limit) * sigmoid(min(gate, limit) * alpha)
#[allow(unused)]
pub fn gptoss_swiglu(gate: &Tensor, up: &Tensor, alpha: f32, limit: f32) -> Result<Tensor> {
    let gate = if gate.is_contiguous() {
        gate.clone()
    } else {
        gate.contiguous()?
    };
    let up = if up.is_contiguous() {
        up.clone()
    } else {
        up.contiguous()?
    };

    if gate.shape() != up.shape() {
        candle_core::bail!(
            "gptoss_swiglu: gate and up must have same shape, got {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
    }

    #[cfg(feature = "cuda")]
    {
        return gptoss_swiglu_cuda(&gate, &up, alpha, limit);
    }

    #[cfg(feature = "metal")]
    {
        return gptoss_swiglu_metal(&gate, &up, alpha, limit);
    }

    #[allow(unreachable_code)]
    {
        gptoss_swiglu_cpu(&gate, &up, alpha, limit)
    }
}

#[cfg(feature = "cuda")]
fn gptoss_swiglu_cuda(gate: &Tensor, up: &Tensor, alpha: f32, limit: f32) -> Result<Tensor> {
    use candle_core::DType;

    let n_elements = gate.elem_count();
    let dtype = gate.dtype();
    let output = Tensor::zeros(gate.shape(), dtype, gate.device())?;

    {
        let (gate_storage, gate_layout) = gate.storage_and_layout();
        let (up_storage, up_layout) = up.storage_and_layout();
        let (out_storage, _out_layout) = output.storage_and_layout();

        let gate_cuda = match &*gate_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Expected CUDA storage for gate"),
        };
        let up_cuda = match &*up_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Expected CUDA storage for up"),
        };
        let out_cuda = match &*out_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("Expected CUDA storage for output"),
        };

        let dev = &gate_cuda.device;
        let stream = *dev.cu_stream() as i64;

        match dtype {
            DType::F16 => {
                let gate_slice = gate_cuda.as_cuda_slice::<half::f16>()?;
                let up_slice = up_cuda.as_cuda_slice::<half::f16>()?;
                let out_slice = out_cuda.as_cuda_slice::<half::f16>()?;
                let gate_ptr = *gate_slice.slice(gate_layout.start_offset()..).device_ptr();
                let up_ptr = *up_slice.slice(up_layout.start_offset()..).device_ptr();
                let out_ptr = *out_slice.device_ptr();
                unsafe {
                    ffi::gptoss_swiglu_f16(
                        gate_ptr as *const core::ffi::c_void,
                        up_ptr as *const core::ffi::c_void,
                        out_ptr as *mut core::ffi::c_void,
                        n_elements as u32,
                        alpha,
                        limit,
                        stream,
                    );
                }
            }
            DType::BF16 => {
                let gate_slice = gate_cuda.as_cuda_slice::<half::bf16>()?;
                let up_slice = up_cuda.as_cuda_slice::<half::bf16>()?;
                let out_slice = out_cuda.as_cuda_slice::<half::bf16>()?;
                let gate_ptr = *gate_slice.slice(gate_layout.start_offset()..).device_ptr();
                let up_ptr = *up_slice.slice(up_layout.start_offset()..).device_ptr();
                let out_ptr = *out_slice.device_ptr();
                unsafe {
                    ffi::gptoss_swiglu_bf16(
                        gate_ptr as *const core::ffi::c_void,
                        up_ptr as *const core::ffi::c_void,
                        out_ptr as *mut core::ffi::c_void,
                        n_elements as u32,
                        alpha,
                        limit,
                        stream,
                    );
                }
            }
            DType::F32 => {
                let gate_slice = gate_cuda.as_cuda_slice::<f32>()?;
                let up_slice = up_cuda.as_cuda_slice::<f32>()?;
                let out_slice = out_cuda.as_cuda_slice::<f32>()?;
                let gate_ptr = *gate_slice.slice(gate_layout.start_offset()..).device_ptr();
                let up_ptr = *up_slice.slice(up_layout.start_offset()..).device_ptr();
                let out_ptr = *out_slice.device_ptr();
                unsafe {
                    ffi::gptoss_swiglu_f32(
                        gate_ptr as *const core::ffi::c_void,
                        up_ptr as *const core::ffi::c_void,
                        out_ptr as *mut core::ffi::c_void,
                        n_elements as u32,
                        alpha,
                        limit,
                        stream,
                    );
                }
            }
            _ => candle_core::bail!("gptoss_swiglu: unsupported dtype {:?}", dtype),
        }
    }

    Ok(output)
}

#[cfg(feature = "metal")]
fn gptoss_swiglu_metal(gate: &Tensor, up: &Tensor, alpha: f32, limit: f32) -> Result<Tensor> {
    use candle_core::DType;
    use metal::MTLSize;

    let n_elements = gate.elem_count();
    let dtype = gate.dtype();
    let output = Tensor::zeros(gate.shape(), dtype, gate.device())?;

    let device = match gate.device() {
        candle_core::Device::Metal(dev) => dev,
        _ => candle_core::bail!("Expected Metal device"),
    };

    let kernel_name = match dtype {
        DType::F16 => "gptoss_swiglu_half",
        DType::BF16 => "gptoss_swiglu_bfloat",
        DType::F32 => "gptoss_swiglu_float",
        _ => candle_core::bail!("gptoss_swiglu: unsupported dtype {:?}", dtype),
    };

    let kernels = metal_kernels::Kernels::default();
    let pipeline = kernels
        .load_pipeline(device.device(), kernel_name.to_string())
        .map_err(|e| candle_core::Error::msg(format!("Metal pipeline error: {e}")))?;

    let command_buffer = device.command_buffer()?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (gate_storage, gate_layout) = gate.storage_and_layout();
        let (up_storage, up_layout) = up.storage_and_layout();
        let (out_storage, _out_layout) = output.storage_and_layout();

        let gate_ms = match &*gate_storage {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("Expected Metal storage"),
        };
        let up_ms = match &*up_storage {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("Expected Metal storage"),
        };
        let out_ms = match &*out_storage {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("Expected Metal storage"),
        };

        let gate_offset = (gate_layout.start_offset() * dtype.size_in_bytes()) as u64;
        let up_offset = (up_layout.start_offset() * dtype.size_in_bytes()) as u64;

        encoder.set_buffer(0, Some(gate_ms.buffer()), gate_offset);
        encoder.set_buffer(1, Some(up_ms.buffer()), up_offset);
        encoder.set_buffer(2, Some(out_ms.buffer()), 0);
        encoder.set_bytes(
            3,
            4,
            &(n_elements as u32).to_ne_bytes() as *const _ as *const _,
        );
        encoder.set_bytes(4, 4, &alpha.to_ne_bytes() as *const _ as *const _);
        encoder.set_bytes(5, 4, &limit.to_ne_bytes() as *const _ as *const _);

        let threads_per_group = 256u64;
        let num_groups = ((n_elements as u64) + threads_per_group - 1) / threads_per_group;
        let grid_size = MTLSize::new(num_groups, 1, 1);
        let group_size = MTLSize::new(threads_per_group, 1, 1);
        encoder.dispatch_thread_groups(grid_size, group_size);
    }

    encoder.end_encoding();
    Ok(output)
}

fn gptoss_swiglu_cpu(gate: &Tensor, up: &Tensor, alpha: f32, limit: f32) -> Result<Tensor> {
    let gate_clamped = gate.clamp(f32::NEG_INFINITY, limit)?;
    let up_clamped = up.clamp(-limit, limit)?;

    let gate_scaled = (&gate_clamped * alpha as f64)?;
    let sigmoid_val = candle_nn::ops::sigmoid(&gate_scaled)?;
    let glu = (&gate_clamped * &sigmoid_val)?;

    let up_plus_one = (&up_clamped + 1.0)?;
    up_plus_one.mul(&glu)
}
