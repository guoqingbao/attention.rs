#[cfg(feature = "metal")]
use candle::MetalStorage;
use candle_core as candle;
#[allow(unused_imports)]
use candle_core::backend::BackendStorage;
use candle_core::{DType, Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;
#[cfg(feature = "metal")]
use metal_kernels;

#[derive(Debug, Clone)]
struct CausalMask {
    sliding_window: i32,
}

impl candle::InplaceOp1 for CausalMask {
    fn name(&self) -> &'static str {
        "causal_mask"
    }

    fn cpu_fwd(&self, _: &mut candle::CpuStorage, _: &candle::Layout) -> Result<()> {
        panic!("causal_mask is not implemented on CPU!")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input: &mut candle::CudaStorage,
        input_layout: &candle::Layout,
    ) -> Result<()> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::CudaStorageSlice;
        let dev = input.device();
        let (tgt_len, tgt_len1) = input_layout.shape().dims2()?;
        assert!(
            tgt_len == tgt_len1,
            "Casual mask tensor should has same dim0 and dim1!"
        );
        use std::ffi::c_void;
        let src_ptr = match &input.slice {
            CudaStorageSlice::F32(inp) => *inp.device_ptr() as *mut c_void,
            CudaStorageSlice::F16(inp) => *inp.device_ptr() as *mut c_void,
            CudaStorageSlice::BF16(inp) => *inp.device_ptr() as *mut c_void,
            _ => {
                candle_core::bail!("Casual mask tensor should has dtype of f16, bf16 or f32!")
            }
        };
        let stream = *dev.cu_stream() as i64;

        unsafe {
            match input.dtype() {
                DType::F32 => ffi::causal_mask_f32(
                    src_ptr,
                    tgt_len as i32,
                    self.sliding_window as i32,
                    stream,
                ),
                DType::F16 => ffi::causal_mask_f16(
                    src_ptr,
                    tgt_len as i32,
                    self.sliding_window as i32,
                    stream,
                ),
                DType::BF16 => ffi::causal_mask_bf16(
                    src_ptr,
                    tgt_len as i32,
                    self.sliding_window as i32,
                    stream,
                ),
                _ => {
                    candle_core::bail!("Casual mask tensor should has dtype of f16, bf16 or f32!")
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, input: &mut MetalStorage, input_l: &candle_core::Layout) -> Result<()> {
        let dtype = input.dtype();
        let internal_type = match dtype {
            DType::F16 => metal_kernels::PagedAttentionDType::F16,
            DType::BF16 => metal_kernels::PagedAttentionDType::BF16,
            DType::F32 => metal_kernels::PagedAttentionDType::F32,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        };
        let (tgt_len, tgt_len1) = input_l.shape().dims2()?;
        assert!(
            tgt_len == tgt_len1,
            "Casual mask tensor should has same dim0 and dim1!"
        );

        let dev = input.device();

        let command_buffer = dev.command_buffer()?;
        command_buffer.set_label("causal-mask");

        metal_kernels::call_causal_mask(
            dev.device(),
            &command_buffer,
            metal_kernels::Kernels::default(),
            internal_type,
            input.buffer(),
            tgt_len as i32,
            self.sliding_window,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok(())
    }
}

/// Apply a causal (lower-triangular) attention mask in-place.
///
/// Positions above the diagonal are set to `-inf`. When `sliding_window` is
/// `Some(w)`, positions more than `w` tokens before the current position are
/// also masked out.
pub fn causal_mask(mask: &Tensor, sliding_window: Option<usize>) -> Result<()> {
    let op = CausalMask {
        sliding_window: sliding_window.unwrap_or(0) as i32,
    };
    mask.inplace_op1(&op)
}
