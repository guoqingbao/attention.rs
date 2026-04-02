use candle_core as candle;
#[allow(unused_imports)]
use candle_core::backend::BackendStorage;
use candle_core::{DType, Result, Tensor};
#[cfg(feature = "cuda")]
use kernels::ffi;

#[derive(Debug, Clone)]
struct KvScaleUpdate {
    k_scales: Tensor,
    v_scales: Tensor,
}

impl candle::InplaceOp2 for KvScaleUpdate {
    fn name(&self) -> &'static str {
        "kvscale-update"
    }

    fn cpu_fwd(
        &self,
        _: &mut candle::CpuStorage,
        _: &candle::Layout,
        _: &candle::CpuStorage,
        _: &candle::Layout,
    ) -> Result<()> {
        panic!("kvscale-update is not implemented on CPU!")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        k: &mut candle::CudaStorage,
        k_layout: &candle::Layout,
        v: &candle::CudaStorage,
        _: &candle::Layout,
    ) -> Result<()> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::CudaStorageSlice;
        let dev = k.device();
        let _ = k_layout.shape().elem_count();

        use std::ffi::c_void;

        let (src_ptr, dst_ptr) = match (&k.slice, &v.slice) {
            (CudaStorageSlice::BF16(inp_k), CudaStorageSlice::BF16(inp_v)) => (
                *inp_k.device_ptr() as *const c_void,
                *inp_v.device_ptr() as *const c_void,
            ),
            (CudaStorageSlice::F16(inp_k), CudaStorageSlice::F16(inp_v)) => (
                *inp_k.device_ptr() as *const c_void,
                *inp_v.device_ptr() as *const c_void,
            ),
            (CudaStorageSlice::F32(inp_k), CudaStorageSlice::F32(inp_v)) => (
                *inp_k.device_ptr() as *const c_void,
                *inp_v.device_ptr() as *const c_void,
            ),
            _ => {
                panic!("Invalid dtype for kv scale update!")
            }
        };
        let stream = *dev.cu_stream() as i64;

        let (k_scales, k_scales_layout) = self.k_scales.storage_and_layout();
        let k_scales = match &*k_scales {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("k_scales must be a cuda tensor"),
        };
        let k_scales = k_scales.slice(k_scales_layout.start_offset()..);

        let (v_scales, v_scales_layout) = self.v_scales.storage_and_layout();
        let v_scales = match &*v_scales {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("v_scales must be a cuda tensor"),
        };
        let v_scales = v_scales.slice(v_scales_layout.start_offset()..);

        let k_scales_ptr = *k_scales.device_ptr() as *mut f32;
        let v_scales_ptr = *v_scales.device_ptr() as *mut f32;
        unsafe {
            let (num_tokens, num_heads, head_dim) =
                if let Ok((b, s, h, d)) = k_layout.shape().dims4() {
                    ((b * s) as i64, h as i32, d as i32)
                } else {
                    let (t, h, d) = k_layout.shape().dims3()?;
                    (t as i64, h as i32, d as i32)
                };
            match k.dtype() {
                DType::F32 => ffi::update_kv_scales_per_head_f32(
                    src_ptr,
                    dst_ptr,
                    num_tokens,
                    num_heads,
                    head_dim,
                    k_scales_ptr,
                    v_scales_ptr,
                    stream,
                ),
                DType::F16 => ffi::update_kv_scales_per_head_f16(
                    src_ptr,
                    dst_ptr,
                    num_tokens,
                    num_heads,
                    head_dim,
                    k_scales_ptr,
                    v_scales_ptr,
                    stream,
                ),
                DType::BF16 => ffi::update_kv_scales_per_head_bf16(
                    src_ptr,
                    dst_ptr,
                    num_tokens,
                    num_heads,
                    head_dim,
                    k_scales_ptr,
                    v_scales_ptr,
                    stream,
                ),
                _ => {
                    panic!("Invalid dtype for kv scale update!")
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        k: &mut candle_core::MetalStorage,
        k_l: &candle_core::Layout,
        v: &candle::MetalStorage,
        _: &candle::Layout,
    ) -> Result<()> {
        let dtype = k.dtype();
        let internal_type = match dtype {
            DType::F16 => metal_kernels::PagedAttentionDType::F16,
            DType::BF16 => metal_kernels::PagedAttentionDType::BF16,
            DType::F32 => metal_kernels::PagedAttentionDType::F32,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        };
        let _ = k_l.shape().elem_count();

        let dev = k.device();
        let (k_scales, _) = self.k_scales.storage_and_layout();
        let k_scales = match &*k_scales {
            candle::Storage::Metal(c) => c,
            _ => candle::bail!("k_scales must be a metal tensor"),
        };

        let (v_scales, _) = self.v_scales.storage_and_layout();
        let v_scales = match &*v_scales {
            candle::Storage::Metal(c) => c,
            _ => candle::bail!("v_scales must be a metal tensor"),
        };

        let command_buffer = dev.command_buffer()?;
        command_buffer.set_label("update-scales");

        let (num_tokens, num_heads, head_dim) = if let Ok((b, s, h, d)) = k_l.shape().dims4() {
            ((b * s) as i64, h as i32, d as i32)
        } else {
            let (t, h, d) = k_l.shape().dims3()?;
            (t as i64, h as i32, d as i32)
        };
        metal_kernels::call_update_scales_per_head(
            dev.device(),
            &command_buffer,
            metal_kernels::Kernels::default(),
            internal_type,
            k.buffer(),
            v.buffer(),
            num_tokens,
            num_heads,
            head_dim,
            k_scales.buffer(),
            v_scales.buffer(),
        )
        .map_err(candle_core::Error::wrap)?;

        Ok(())
    }
}

pub fn kv_scale_update(
    key: &Tensor,
    value: &Tensor,
    k_scales: &Tensor,
    v_scales: &Tensor,
) -> Result<()> {
    let op = KvScaleUpdate {
        k_scales: k_scales.to_owned(),
        v_scales: v_scales.to_owned(),
    };
    key.inplace_op2(value, &op)
}
