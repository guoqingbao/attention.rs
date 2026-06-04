use candle_core::{Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda_sort {
    use candle_core as candle;
    use candle_core::{Result, Tensor};
    use kernels::ffi;

    #[derive(Debug, Clone)]
    struct ArgSort {
        asc: bool,
        last_dim: usize,
        inplace: bool,
    }

    impl candle::CustomOp1 for ArgSort {
        fn name(&self) -> &'static str {
            "argsort"
        }

        fn cpu_fwd(
            &self,
            _storage: &candle::CpuStorage,
            _layout: &candle::Layout,
        ) -> Result<(candle::CpuStorage, candle::Shape)> {
            candle::bail!("attention_rs::sort CUDA op called on CPU storage")
        }

        fn cuda_fwd(
            &self,
            storage: &candle::CudaStorage,
            layout: &candle::Layout,
        ) -> Result<(candle::CudaStorage, candle::Shape)> {
            use candle::backend::BackendStorage;
            use candle::cuda_backend::cudarc::driver::DevicePtr;
            use candle::cuda_backend::CudaStorageSlice;
            use candle::cuda_backend::WrapErr;

            let dev = storage.device();
            let elem_count = layout.shape().elem_count();
            let ncols = self.last_dim as i32;
            let nrows = elem_count as i32 / ncols;
            let dst = unsafe { dev.alloc::<u32>(elem_count) }.w()?;

            use std::ffi::c_void;

            let src = match &storage.slice {
                CudaStorageSlice::U8(inp) => inp.device_ptr(),
                CudaStorageSlice::U32(inp) => inp.device_ptr(),
                CudaStorageSlice::I64(inp) => inp.device_ptr(),
                CudaStorageSlice::BF16(inp) => inp.device_ptr(),
                CudaStorageSlice::F16(inp) => inp.device_ptr(),
                CudaStorageSlice::F32(inp) => inp.device_ptr(),
                CudaStorageSlice::F64(inp) => inp.device_ptr(),
            };
            let src_ptr = *src as *const c_void;
            let dst_ptr = *dst.device_ptr() as *mut c_void;
            let stream = *dev.cu_stream() as i64;
            unsafe {
                if self.asc {
                    match storage.dtype() {
                        candle::DType::U8 => {
                            ffi::asort_asc_u8(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                        }
                        candle::DType::U32 => {
                            ffi::asort_asc_u32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                        }
                        candle::DType::I64 => {
                            ffi::asort_asc_i64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                        }
                        candle::DType::BF16 => ffi::asort_asc_bf16(
                            src_ptr,
                            dst_ptr,
                            nrows,
                            ncols,
                            self.inplace,
                            stream,
                        ),
                        candle::DType::F16 => {
                            ffi::asort_asc_f16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                        }
                        candle::DType::F32 => {
                            ffi::asort_asc_f32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                        }
                        candle::DType::F64 => {
                            ffi::asort_asc_f64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                        }
                    }
                } else {
                    match storage.dtype() {
                        candle::DType::U8 => {
                            ffi::asort_desc_u8(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                        }
                        candle::DType::U32 => ffi::asort_desc_u32(
                            src_ptr,
                            dst_ptr,
                            nrows,
                            ncols,
                            self.inplace,
                            stream,
                        ),
                        candle::DType::I64 => ffi::asort_desc_i64(
                            src_ptr,
                            dst_ptr,
                            nrows,
                            ncols,
                            self.inplace,
                            stream,
                        ),
                        candle::DType::BF16 => ffi::asort_desc_bf16(
                            src_ptr,
                            dst_ptr,
                            nrows,
                            ncols,
                            self.inplace,
                            stream,
                        ),
                        candle::DType::F16 => ffi::asort_desc_f16(
                            src_ptr,
                            dst_ptr,
                            nrows,
                            ncols,
                            self.inplace,
                            stream,
                        ),
                        candle::DType::F32 => ffi::asort_desc_f32(
                            src_ptr,
                            dst_ptr,
                            nrows,
                            ncols,
                            self.inplace,
                            stream,
                        ),
                        candle::DType::F64 => ffi::asort_desc_f64(
                            src_ptr,
                            dst_ptr,
                            nrows,
                            ncols,
                            self.inplace,
                            stream,
                        ),
                    }
                }
            }
            let dst_ret = candle::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(dst),
                device: dev.clone(),
            };
            Ok((dst_ret, layout.shape().clone()))
        }
    }

    pub fn arg_sort(t: &Tensor, asc: bool) -> Result<Tensor> {
        if !t.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "arg_sort" });
        }
        let last_dim = match t.dims().last() {
            Some(last_dim) => *last_dim,
            None => candle_core::bail!("empty last-dim in arg-sort"),
        };
        t.apply_op1_no_bwd(&ArgSort {
            asc,
            last_dim,
            inplace: false,
        })
    }

    pub fn sort(t: &Tensor, asc: bool) -> Result<(Tensor, Tensor)> {
        if !t.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "arg_sort" });
        }
        let last_dim = match t.dims().last() {
            Some(last_dim) => *last_dim,
            None => candle_core::bail!("empty last-dim in arg-sort"),
        };
        let sorted = t.copy()?;
        let asort = sorted.apply_op1_no_bwd(&ArgSort {
            asc,
            last_dim,
            inplace: true,
        })?;
        Ok((sorted, asort))
    }
}

pub trait ArgSortOp {
    fn arg_sort(&self, asc: bool) -> Result<Tensor>;
    fn sort(&self, asc: bool) -> Result<(Tensor, Tensor)>;
}

impl ArgSortOp for Tensor {
    fn arg_sort(&self, asc: bool) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.device().is_cuda() {
            return cuda_sort::arg_sort(self, asc);
        }

        self.arg_sort_last_dim(asc)
    }

    fn sort(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if self.device().is_cuda() {
            return cuda_sort::sort(self, asc);
        }

        self.sort_last_dim(asc)
    }
}
