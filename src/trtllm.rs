use crate::cuda_utils;
use crate::flashinfer::{
    get_cuda_ptr, get_cuda_ptr_storage, get_or_init_workspace, WORKSPACE_FLOAT_SIZE,
};
use crate::kernels;
use candle_core as candle;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::WrapErr;
use candle_core::{CudaStorage, DType, Layout, Result, Storage, Tensor};

fn dtype_to_kernel_code(dtype: DType) -> i32 {
    match dtype {
        DType::U8 => 2,
        DType::BF16 => 1,
        _ => 0,
    }
}

pub struct TrtllmDecode {
    pub key_cache: Tensor,
    pub value_cache: Tensor,
    pub block_tables: Tensor,
    pub seq_lens: Tensor,
    pub cum_seq_lens_q: Option<Tensor>,
    pub max_q_len: usize,
    pub max_kv_len: usize,
    pub bmm1_scale: f32,
    pub bmm2_scale: f32,
    pub enable_pdl: bool,
}

impl candle::CustomOp1 for TrtllmDecode {
    fn name(&self) -> &'static str {
        "trtllm-decode"
    }

    fn cpu_fwd(
        &self,
        _: &candle::CpuStorage,
        _: &Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("no cpu support")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, candle::Shape)> {
        match q.dtype() {
            DType::F16 => self.cuda_fwd_impl::<half::f16>(q, q_l),
            DType::BF16 => self.cuda_fwd_impl::<half::bf16>(q, q_l),
            DType::U8 => self.cuda_fwd_impl::<u8>(q, q_l),
            _ => candle::bail!("unsupported dtype"),
        }
    }
}

impl TrtllmDecode {
    fn cuda_fwd_impl<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, candle::Shape)> {
        let dev = q.device();
        let sm = cuda_utils::sm_version(dev).unwrap_or(0);
        if sm != 100 && sm != 103 {
            candle::bail!(
                "trtllm backend currently supports only sm100/sm103, got sm{}",
                sm
            );
        }
        let (sum_seq_q, num_qo_heads, head_dim) = q_l.shape().dims3()?;
        let (num_pages, page_size, num_kv_heads, _) = self.key_cache.shape().dims4()?;
        let (batch_size, max_num_blocks_per_seq) = self.block_tables.shape().dims2()?;

        let q_ptr = get_cuda_ptr_storage(q, q_l, q.dtype())?;
        let kc_ptr = get_cuda_ptr(&self.key_cache)?;
        let vc_ptr = get_cuda_ptr(&self.value_cache)?;
        let out = unsafe { dev.alloc::<T>(q_l.shape().elem_count()) }.w()?;
        let out_ptr = *out.device_ptr() as *mut std::ffi::c_void;

        let (bt_s, bt_l) = self.block_tables.storage_and_layout();
        let bt_ptr = match &*bt_s {
            Storage::Cuda(c) => {
                let t = c.as_cuda_slice::<u32>()?.slice(bt_l.start_offset()..);
                *t.device_ptr() as *const i32
            }
            _ => candle::bail!("block_tables must be cuda"),
        };
        let (sl_s, sl_l) = self.seq_lens.storage_and_layout();
        let sl_ptr = match &*sl_s {
            Storage::Cuda(c) => {
                let t = c.as_cuda_slice::<u32>()?.slice(sl_l.start_offset()..);
                *t.device_ptr() as *const i32
            }
            _ => candle::bail!("seq_lens must be cuda"),
        };

        let cum_q_ptr = if let Some(cum_q) = &self.cum_seq_lens_q {
            let (s, l) = cum_q.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => {
                    let t = c.as_cuda_slice::<u32>()?.slice(l.start_offset()..);
                    *t.device_ptr() as *const i32
                }
                _ => candle::bail!("cum_seq_lens_q must be cuda"),
            }
        } else {
            std::ptr::null()
        };

        let (ws_float_ptr, _ws_int_ptr, _page_locked_ptr, _page_locked_size) =
            get_or_init_workspace(dev, false)?;

        unsafe {
            kernels::ffi::trtllm_decode_run_wrapper(
                out_ptr,
                q_ptr,
                kc_ptr,
                vc_ptr,
                bt_ptr,
                sl_ptr,
                cum_q_ptr,
                batch_size as i32,
                sum_seq_q as i32,
                self.max_q_len as i32,
                self.max_kv_len as i32,
                max_num_blocks_per_seq as i32,
                num_pages as i32,
                num_qo_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                page_size as i32,
                self.bmm1_scale,
                self.bmm2_scale,
                ws_float_ptr,
                WORKSPACE_FLOAT_SIZE,
                dtype_to_kernel_code(q.dtype()),
                dtype_to_kernel_code(self.key_cache.dtype()),
                dtype_to_kernel_code(q.dtype()),
                self.enable_pdl,
                -1,
                *dev.cu_stream() as i64,
            );
        }

        Ok((
            CudaStorage::wrap_cuda_slice(out, dev.clone()),
            q_l.shape().clone(),
        ))
    }
}

pub fn decode(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    cum_seq_lens_q: Option<&Tensor>,
    max_q_len: usize,
    max_kv_len: usize,
    bmm1_scale: f32,
    bmm2_scale: f32,
    enable_pdl: bool,
) -> Result<Tensor> {
    let op = TrtllmDecode {
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.clone(),
        seq_lens: seq_lens.clone(),
        cum_seq_lens_q: cum_seq_lens_q.cloned(),
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        enable_pdl,
    };
    q.apply_op1(op)
}

pub struct TrtllmContext {
    pub key_cache: Tensor,
    pub value_cache: Tensor,
    pub block_tables: Option<Tensor>,
    pub seq_lens: Tensor,
    pub cum_seq_lens_q: Tensor,
    pub cum_seq_lens_kv: Option<Tensor>,
    pub max_q_len: usize,
    pub max_kv_len: usize,
    pub bmm1_scale: f32,
    pub bmm2_scale: f32,
    pub enable_pdl: bool,
}

impl candle::CustomOp1 for TrtllmContext {
    fn name(&self) -> &'static str {
        "trtllm-context"
    }

    fn cpu_fwd(
        &self,
        _: &candle::CpuStorage,
        _: &Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("no cpu support")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, candle::Shape)> {
        match q.dtype() {
            DType::F16 => self.cuda_fwd_impl::<half::f16>(q, q_l),
            DType::BF16 => self.cuda_fwd_impl::<half::bf16>(q, q_l),
            DType::U8 => self.cuda_fwd_impl::<u8>(q, q_l),
            _ => candle::bail!("unsupported dtype"),
        }
    }
}

impl TrtllmContext {
    fn cuda_fwd_impl<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, candle::Shape)> {
        let dev = q.device();
        let sm = cuda_utils::sm_version(dev).unwrap_or(0);
        if sm != 100 && sm != 103 {
            candle::bail!(
                "trtllm backend currently supports only sm100/sm103, got sm{}",
                sm
            );
        }
        let (sum_seq_q, num_qo_heads, head_dim) = q_l.shape().dims3()?;
        let use_paged = self.block_tables.is_some();
        let (num_pages, page_size, num_kv_heads, max_num_blocks_per_seq, batch_size) = if use_paged
        {
            let (num_pages, page_size, num_kv_heads, _) = self.key_cache.shape().dims4()?;
            let (batch_size, max_num_blocks_per_seq) =
                self.block_tables.as_ref().unwrap().shape().dims2()?;
            (
                num_pages,
                page_size,
                num_kv_heads,
                max_num_blocks_per_seq,
                batch_size,
            )
        } else {
            let (_sum_seq_kv, num_kv_heads, _) = self.key_cache.shape().dims3()?;
            let batch_size = self.seq_lens.dim(0)?;
            (0usize, 0usize, num_kv_heads, 0usize, batch_size)
        };

        let q_ptr = get_cuda_ptr_storage(q, q_l, q.dtype())?;
        let kc_ptr = get_cuda_ptr(&self.key_cache)?;
        let vc_ptr = get_cuda_ptr(&self.value_cache)?;
        let out = unsafe { dev.alloc::<T>(q_l.shape().elem_count()) }.w()?;
        let out_ptr = *out.device_ptr() as *mut std::ffi::c_void;

        let bt_ptr = if let Some(block_tables) = &self.block_tables {
            let (bt_s, bt_l) = block_tables.storage_and_layout();
            match &*bt_s {
                Storage::Cuda(c) => {
                    let t = c.as_cuda_slice::<u32>()?.slice(bt_l.start_offset()..);
                    *t.device_ptr() as *const i32
                }
                _ => candle::bail!("block_tables must be cuda"),
            }
        } else {
            std::ptr::null()
        };
        let (sl_s, sl_l) = self.seq_lens.storage_and_layout();
        let sl_ptr = match &*sl_s {
            Storage::Cuda(c) => {
                let t = c.as_cuda_slice::<u32>()?.slice(sl_l.start_offset()..);
                *t.device_ptr() as *const i32
            }
            _ => candle::bail!("seq_lens must be cuda"),
        };
        let (cum_q_s, cum_q_l) = self.cum_seq_lens_q.storage_and_layout();
        let cum_q_ptr = match &*cum_q_s {
            Storage::Cuda(c) => {
                let t = c.as_cuda_slice::<u32>()?.slice(cum_q_l.start_offset()..);
                *t.device_ptr() as *const i32
            }
            _ => candle::bail!("cum_seq_lens_q must be cuda"),
        };
        let cum_kv_ptr = if let Some(cum_seq_lens_kv) = &self.cum_seq_lens_kv {
            let (cum_kv_s, cum_kv_l) = cum_seq_lens_kv.storage_and_layout();
            match &*cum_kv_s {
                Storage::Cuda(c) => {
                    let t = c.as_cuda_slice::<u32>()?.slice(cum_kv_l.start_offset()..);
                    *t.device_ptr() as *const i32
                }
                _ => candle::bail!("cum_seq_lens_kv must be cuda"),
            }
        } else {
            cum_q_ptr
        };

        let (ws_float_ptr, _ws_int_ptr, _page_locked_ptr, _page_locked_size) =
            get_or_init_workspace(dev, false)?;

        unsafe {
            kernels::ffi::trtllm_context_run_wrapper(
                out_ptr,
                q_ptr,
                kc_ptr,
                vc_ptr,
                bt_ptr,
                sl_ptr,
                cum_q_ptr,
                cum_kv_ptr,
                batch_size as i32,
                sum_seq_q as i32,
                self.max_q_len as i32,
                self.max_kv_len as i32,
                max_num_blocks_per_seq as i32,
                num_pages as i32,
                num_qo_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                page_size as i32,
                self.bmm1_scale,
                self.bmm2_scale,
                ws_float_ptr,
                WORKSPACE_FLOAT_SIZE,
                dtype_to_kernel_code(q.dtype()),
                dtype_to_kernel_code(self.key_cache.dtype()),
                dtype_to_kernel_code(q.dtype()),
                self.enable_pdl,
                -1,
                *dev.cu_stream() as i64,
            );
        }

        Ok((
            CudaStorage::wrap_cuda_slice(out, dev.clone()),
            q_l.shape().clone(),
        ))
    }
}

pub fn context(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: Option<&Tensor>,
    seq_lens: &Tensor,
    cum_seq_lens_q: &Tensor,
    cum_seq_lens_kv: Option<&Tensor>,
    max_q_len: usize,
    max_kv_len: usize,
    bmm1_scale: f32,
    bmm2_scale: f32,
    enable_pdl: bool,
) -> Result<Tensor> {
    let op = TrtllmContext {
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.cloned(),
        seq_lens: seq_lens.clone(),
        cum_seq_lens_q: cum_seq_lens_q.clone(),
        cum_seq_lens_kv: cum_seq_lens_kv.cloned(),
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        enable_pdl,
    };
    q.apply_op1(op)
}
