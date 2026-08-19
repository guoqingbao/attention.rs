//! IQ GGUF dense GEMM.
//!
//! Candle already has llama.cpp MMQ/MMVQ for Q2_K–Q8_0. IQ types currently
//! dequantize to F32 and run dense GEMM, which is both slower and a different
//! numeric contract than Q8_1 `vec_dot`. This module keeps IQ weights packed
//! and calls the native `gguf_gemm` kernel.
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Result, Tensor};

/// IQ GGUF dtypes that should use [`gguf_iq_matmul`] instead of Candle `QMatMul`.
pub fn is_iq_gguf_dtype(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::IQ2_XXS
            | GgmlDType::IQ2_XS
            | GgmlDType::IQ3_XXS
            | GgmlDType::IQ1_S
            | GgmlDType::IQ4_NL
            | GgmlDType::IQ3_S
            | GgmlDType::IQ2_S
            | GgmlDType::IQ4_XS
            | GgmlDType::IQ1_M
    )
}

fn iq_gguf_dtype_code(dtype: GgmlDType) -> Result<i32> {
    Ok(match dtype {
        GgmlDType::IQ2_XXS => 6,
        GgmlDType::IQ2_XS => 7,
        GgmlDType::IQ3_XXS => 8,
        GgmlDType::IQ4_XS => 9,
        GgmlDType::IQ1_S => 10,
        GgmlDType::IQ4_NL => 11,
        GgmlDType::IQ3_S => 12,
        GgmlDType::IQ2_S => 13,
        GgmlDType::IQ1_M => 14,
        dtype => candle_core::bail!("unsupported IQ GGUF dtype {dtype:?}"),
    })
}

/// F32 activations `[..., K]` × packed IQ weight `[N, K]` → F32 `[..., N]`.
///
/// Activations must be F32. The kernel quantizes them to Q8_1 internally.
pub fn gguf_iq_matmul(input: &Tensor, weight: &QTensor) -> Result<Tensor> {
    if !is_iq_gguf_dtype(weight.dtype()) {
        candle_core::bail!(
            "gguf_iq_matmul only supports IQ GGUF dtypes, got {:?}",
            weight.dtype()
        );
    }
    #[cfg(feature = "cuda")]
    {
        let (size_n, size_k) = weight.shape().dims2()?;
        let op = GgufIqMatMul {
            weight_ptr: weight.device_ptr()? as u64,
            size_n,
            size_k,
            dtype: iq_gguf_dtype_code(weight.dtype())?,
        };
        input.apply_op1_no_bwd(&op)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = input;
        let _ = weight;
        candle_core::bail!("gguf_iq_matmul is only implemented on CUDA")
    }
}

#[cfg(feature = "cuda")]
struct GgufIqMatMul {
    weight_ptr: u64,
    size_n: usize,
    size_k: usize,
    dtype: i32,
}

#[cfg(feature = "cuda")]
impl GgufIqMatMul {
    fn launch(
        &self,
        x: &candle_core::CudaStorage,
        x_l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use crate::kernels::ffi::gguf_gemm;
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::CudaStorage;

        if x.dtype() != DType::F32 {
            candle_core::bail!(
                "gguf_iq_matmul requires F32 activations, got {:?}",
                x.dtype()
            );
        }
        if !x_l.is_contiguous() {
            candle_core::bail!("gguf_iq_matmul requires contiguous activations: {x_l:?}");
        }
        let x_dims = x_l.shape().dims();
        let Some(&size_k) = x_dims.last() else {
            candle_core::bail!("gguf_iq_matmul requires a rank-2-or-higher input");
        };
        if size_k != self.size_k {
            candle_core::bail!(
                "gguf_iq_matmul input/weight mismatch: input K={}, weight K={}",
                size_k,
                self.size_k
            );
        }
        let size_m = x_l.shape().elem_count() / size_k;
        let mut output_dims = x_dims.to_vec();
        *output_dims.last_mut().unwrap() = self.size_n;
        let output_shape = candle_core::Shape::from(output_dims);

        let input = x.as_cuda_slice::<f32>()?;
        let input = input.slice(x_l.start_offset()..x_l.start_offset() + x_l.shape().elem_count());
        let dev = x.device().clone();
        let output = unsafe { dev.alloc::<f32>(output_shape.elem_count()) }.w()?;
        let stream = *dev.cu_stream() as i64;
        let input_ptr = *input.device_ptr() as *const f32;
        let weight_ptr = self.weight_ptr as *const std::ffi::c_void;
        let output_ptr = *output.device_ptr() as *mut f32;

        unsafe {
            gguf_gemm(
                input_ptr,
                weight_ptr,
                output_ptr,
                size_m as i32,
                self.size_n as i32,
                size_k as i32,
                self.dtype,
                stream,
            );
        }
        Ok((CudaStorage::wrap_cuda_slice(output, dev), output_shape))
    }
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for GgufIqMatMul {
    fn name(&self) -> &'static str {
        "gguf_iq_matmul"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("gguf_iq_matmul is only implemented on CUDA")
    }

    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        x_l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        self.launch(x, x_l)
    }
}
