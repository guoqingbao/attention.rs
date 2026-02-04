use crate::kernels;
use candle_core as candle;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{sys, CudaSlice, DevicePtr};
use candle_core::cuda_backend::WrapErr;
use candle_core::{CudaStorage, DType, Layout, Result, Storage, Tensor};
use std::cell::RefCell;

/// Workspace buffer sizes for FlashInfer operations
const WORKSPACE_FLOAT_SIZE: usize = 256 * 1024 * 1024; // 256 MB
const WORKSPACE_INT_SIZE: usize = 128 * 1024 * 1024; // 128 MB

/// Static workspace buffers for FlashInfer to avoid per-call allocation
struct PinnedHostBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl PinnedHostBuffer {
    fn new(size: usize) -> Result<Self> {
        if size == 0 {
            candle::bail!("Pinned host buffer size must be > 0");
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            sys::lib()
                .cuMemAllocHost_v2(&mut ptr, size)
                .result()
                .map_err(|e| candle_core::Error::Msg(format!("cuMemAllocHost_v2 failed: {e:?}")))?
        }
        if ptr.is_null() {
            candle::bail!("cuMemAllocHost_v2 returned null pointer");
        }
        Ok(Self { ptr, size })
    }

    fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                sys::lib()
                    .cuMemFreeHost(self.ptr)
                    .result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("cuMemAllocHost_v2 failed: {e:?}"))
                    })
                    .unwrap();
            }
        }
    }
}

struct FlashInferWorkspace {
    float_buffer: CudaSlice<u8>,
    int_buffer: CudaSlice<u8>,
    pinned_host: PinnedHostBuffer,
    device_ordinal: usize,
}

thread_local! {
    static WORKSPACE: RefCell<Option<FlashInferWorkspace>> = const { RefCell::new(None) };
    static WORKSPACE_GRAPH: RefCell<Option<FlashInferWorkspace>> = const { RefCell::new(None) };
}

fn get_cuda_ptr(t: &Tensor) -> Result<*const core::ffi::c_void> {
    let (s, l) = t.storage_and_layout();
    match (&*s, t.dtype()) {
        (Storage::Cuda(c), DType::U8) => {
            let ptr = *c
                .as_cuda_slice::<u8>()?
                .slice(l.start_offset()..)
                .device_ptr();
            Ok(ptr as *const core::ffi::c_void)
        }
        (Storage::Cuda(c), DType::BF16) => {
            let ptr = *c
                .as_cuda_slice::<half::bf16>()?
                .slice(l.start_offset()..)
                .device_ptr();
            Ok(ptr as *const core::ffi::c_void)
        }
        (Storage::Cuda(c), DType::F16) => {
            let ptr = *c
                .as_cuda_slice::<half::f16>()?
                .slice(l.start_offset()..)
                .device_ptr();
            Ok(ptr as *const core::ffi::c_void)
        }
        _ => candle::bail!("Tensor must be on CUDA and have U8, BF16, or F16 dtype"),
    }
}

fn get_cuda_ptr_storage(
    s: &CudaStorage,
    l: &Layout,
    dtype: DType,
) -> Result<*const core::ffi::c_void> {
    match dtype {
        DType::U8 => {
            let ptr = *s
                .as_cuda_slice::<u8>()?
                .slice(l.start_offset()..)
                .device_ptr();
            Ok(ptr as *const core::ffi::c_void)
        }
        DType::BF16 => {
            let ptr = *s
                .as_cuda_slice::<half::bf16>()?
                .slice(l.start_offset()..)
                .device_ptr();
            Ok(ptr as *const core::ffi::c_void)
        }
        DType::F16 => {
            let ptr = *s
                .as_cuda_slice::<half::f16>()?
                .slice(l.start_offset()..)
                .device_ptr();
            Ok(ptr as *const core::ffi::c_void)
        }
        _ => candle::bail!("Tensor must be on CUDA and have U8, BF16, or F16 dtype"),
    }
}

fn get_or_init_workspace(
    dev: &candle_core::cuda_backend::CudaDevice,
    for_cuda_graph: bool,
) -> Result<(
    *mut std::ffi::c_void,
    *mut std::ffi::c_void,
    *mut std::ffi::c_void,
    usize,
)> {
    let ws_cell = if for_cuda_graph {
        &WORKSPACE_GRAPH
    } else {
        &WORKSPACE
    };
    ws_cell.with(|ws| {
        let mut ws = ws.borrow_mut();
        let ordinal = dev.ordinal();

        // Check if we need to (re)initialize the workspace
        let needs_init = match ws.as_ref() {
            None => true,
            Some(existing) => existing.device_ordinal != ordinal,
        };

        if needs_init {
            let float_buffer = unsafe { dev.alloc::<u8>(WORKSPACE_FLOAT_SIZE) }.w()?;
            let int_buffer = unsafe { dev.alloc::<u8>(WORKSPACE_INT_SIZE) }.w()?;
            let pinned_host = PinnedHostBuffer::new(WORKSPACE_INT_SIZE)?;
            *ws = Some(FlashInferWorkspace {
                float_buffer,
                int_buffer,
                pinned_host,
                device_ordinal: ordinal,
            });
        }

        let workspace = ws.as_ref().unwrap();
        Ok((
            *workspace.float_buffer.device_ptr() as *mut std::ffi::c_void,
            *workspace.int_buffer.device_ptr() as *mut std::ffi::c_void,
            workspace.pinned_host.as_ptr(),
            workspace.pinned_host.size(),
        ))
    })
}

pub fn append_kv_cache(
    k: &Tensor,
    v: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    indices: &Tensor,
    indptr: &Tensor,
    last_len: &Tensor,
    batch_indices: Option<&Tensor>,
    positions: Option<&Tensor>,
) -> Result<()> {
    let op = FlashInferAppend {
        k: k.clone(),
        v: v.clone(),
        k_cache: k_cache.clone(),
        v_cache: v_cache.clone(),
        indices: indices.clone(),
        indptr: indptr.clone(),
        last_len: last_len.clone(),
        batch_indices: batch_indices.cloned(),
        positions: positions.cloned(),
    };
    k.apply_op1(op)?;
    Ok(())
}

pub struct FlashInferAppend {
    pub k: Tensor,
    pub v: Tensor,
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub indices: Tensor,
    pub indptr: Tensor,
    pub last_len: Tensor,
    pub batch_indices: Option<Tensor>,
    pub positions: Option<Tensor>,
}

impl candle::CustomOp1 for FlashInferAppend {
    fn name(&self) -> &'static str {
        "flashinfer-append"
    }

    fn cpu_fwd(
        &self,
        _s: &candle::CpuStorage,
        _l: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("cpu not implemented for flash-infer")
    }

    fn cuda_fwd(
        &self,
        _s: &candle::CudaStorage,
        _l: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        let k_ptr = &self.k;
        let v_ptr = &self.v;
        let kc_ptr = &self.k_cache;
        let vc_ptr = &self.v_cache;
        let indices_ptr = &self.indices;
        let indptr_ptr = &self.indptr;
        let last_len_ptr = &self.last_len;

        let dev = _s.device();

        // Correctly handle dims (k_ptr is [total_tokens, num_heads, head_dim])
        let (nnz, num_heads, head_dim) = k_ptr.dims3()?;

        // Determine batch size from indptr
        let batch_size = indptr_ptr.dim(0)? - 1;

        let (_, page_size, _, _) = kc_ptr.shape().dims4()?;

        let batch_indices_ptr = if let Some(t) = &self.batch_indices {
            let (t, t_l) = t.storage_and_layout();
            let t = match &*t {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(t_l.start_offset()..),
                _ => candle::bail!("batch_indices must be cuda"),
            };
            *t.device_ptr() as *const i32
        } else {
            std::ptr::null()
        };

        let positions_ptr = if let Some(t) = &self.positions {
            let (t, t_l) = t.storage_and_layout();
            let t = match &*t {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(t_l.start_offset()..),
                _ => candle::bail!("positions must be cuda"),
            };
            *t.device_ptr() as *const i32
        } else {
            std::ptr::null()
        };

        // 0: F16, 1: BF16, 2: U8 (FP8)
        let data_type = match self.k_cache.dtype() {
            DType::U8 => 2,
            DType::BF16 => 1,
            _ => 0,
        };

        let kc_ptr = get_cuda_ptr(kc_ptr)?;
        let vc_ptr = get_cuda_ptr(vc_ptr)?;
        let k_data_ptr = get_cuda_ptr(k_ptr)?;
        let v_data_ptr = get_cuda_ptr(v_ptr)?;

        let (indices_s, indices_l) = indices_ptr.storage_and_layout();
        let indices = match &*indices_s {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(indices_l.start_offset()..),
            _ => candle::bail!("indices must be cuda"),
        };

        let (indptr_s, indptr_l) = indptr_ptr.storage_and_layout();
        let indptr = match &*indptr_s {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(indptr_l.start_offset()..),
            _ => candle::bail!("indptr must be cuda"),
        };

        let (last_len_s, last_len_l) = last_len_ptr.storage_and_layout();
        let last_len = match &*last_len_s {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(last_len_l.start_offset()..),
            _ => candle::bail!("last_len must be cuda"),
        };

        unsafe {
            kernels::ffi::flashinfer_append_kv_cache(
                kc_ptr,
                vc_ptr,
                k_data_ptr,
                v_data_ptr,
                *indices.device_ptr() as *const i32,
                *indptr.device_ptr() as *const i32,
                *last_len.device_ptr() as *const i32,
                batch_indices_ptr,
                positions_ptr,
                nnz as i32,
                batch_size as i32,
                num_heads as i32,
                head_dim as i32,
                page_size as i32,
                data_type,
                *dev.cu_stream() as i64,
            );
        }

        let out = unsafe { dev.alloc::<half::f16>(1) }.w()?;
        let out_shape = candle::Shape::from(());
        Ok((CudaStorage::wrap_cuda_slice(out, dev.clone()), out_shape))
    }
}

pub struct FlashInferDecodeWithPlan {
    pub key_cache: Tensor,
    pub value_cache: Tensor,
    pub indices: Tensor,
    pub indptr: Tensor, // Device tensor for paged_kv
    pub last_len: Tensor,
    pub block_size: usize,
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sm_scale: f32,
    pub plan_info: Vec<i64>, // length 10
}

impl candle::CustomOp1 for FlashInferDecodeWithPlan {
    fn name(&self) -> &'static str {
        "flashinfer-decode-with-plan"
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

impl FlashInferDecodeWithPlan {
    fn cuda_fwd_impl<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, candle::Shape)> {
        if self.plan_info.len() != 10 {
            candle::bail!(
                "flashinfer decode plan_info must have length 10, got {}",
                self.plan_info.len()
            );
        }

        let dev = q.device();
        let (batch_size, _, _) = q_l.shape().dims3()?;

        let kc_ptr = get_cuda_ptr(&self.key_cache)?;
        let vc_ptr = get_cuda_ptr(&self.value_cache)?;

        let (indices, indices_l) = self.indices.storage_and_layout();
        let indices_ptr = match &*indices {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(indices_l.start_offset()..),
            _ => candle::bail!("indices must be cuda"),
        };

        let (indptr, indptr_l) = self.indptr.storage_and_layout();
        let indptr_ptr = match &*indptr {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(indptr_l.start_offset()..),
            _ => candle::bail!("indptr must be cuda"),
        };

        let (last_len, last_len_l) = self.last_len.storage_and_layout();
        let last_len_ptr = match &*last_len {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(last_len_l.start_offset()..),
            _ => candle::bail!("last_len must be cuda"),
        };

        let q_ptr = get_cuda_ptr_storage(q, q_l, q.dtype())?;

        // 0: F16, 1: BF16, 2: U8 (FP8)
        let data_type = match self.key_cache.dtype() {
            DType::U8 => 2,
            DType::BF16 => 1,
            _ => 0,
        };

        let out = unsafe { dev.alloc::<T>(q_l.shape().elem_count()) }.w()?;
        let out_ptr = *out.device_ptr() as *mut std::ffi::c_void;
        let (ws_float_ptr, ws_int_ptr, _page_locked_ptr, _page_locked_size) =
            get_or_init_workspace(dev, true)?;

        unsafe {
            kernels::ffi::flashinfer_decode_run_wrapper(
                out_ptr,
                q_ptr,
                kc_ptr,
                vc_ptr,
                *indices_ptr.device_ptr() as *const i32,
                *indptr_ptr.device_ptr() as *const i32,
                *last_len_ptr.device_ptr() as *const i32,
                batch_size as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                self.head_dim as i32,
                self.block_size as i32,
                self.sm_scale,
                ws_float_ptr,
                WORKSPACE_FLOAT_SIZE,
                ws_int_ptr,
                WORKSPACE_INT_SIZE,
                self.plan_info.as_ptr(),
                data_type,
                *dev.cu_stream() as i64,
            );
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, q_l.shape().clone()))
    }
}

pub fn decode_with_plan(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    indices: &Tensor,
    indptr: &Tensor,
    last_len: &Tensor,
    block_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
    plan_info: &[i64],
) -> Result<Tensor> {
    let op = FlashInferDecodeWithPlan {
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        indices: indices.clone(),
        indptr: indptr.clone(),
        last_len: last_len.clone(),
        block_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
        plan_info: plan_info.to_vec(),
    };
    q.apply_op1(op)
}

pub fn decode_plan(
    dev: &candle_core::Device,
    kv_dtype: DType,
    indptr_host: &[u32],
    batch_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    enable_cuda_graph: bool,
) -> Result<Vec<i64>> {
    let dev = dev.as_cuda_device()?;

    if indptr_host.len() != batch_size + 1 {
        candle::bail!(
            "indptr_host length must be batch_size+1 ({}), got {}",
            batch_size + 1,
            indptr_host.len()
        );
    }

    // 0: F16, 1: BF16, 2: U8 (FP8)
    let data_type = match kv_dtype {
        DType::U8 => 2,
        DType::BF16 => 1,
        _ => 0,
    };

    let (ws_float_ptr, ws_int_ptr, page_locked_ptr, page_locked_size) =
        get_or_init_workspace(dev, enable_cuda_graph)?;

    let mut plan_info = [0i64; 10];
    unsafe {
        kernels::ffi::flashinfer_decode_plan_wrapper(
            indptr_host.as_ptr().cast(),
            batch_size as i32,
            num_qo_heads as i32,
            num_kv_heads as i32,
            head_dim as i32,
            page_size as i32,
            ws_float_ptr,
            WORKSPACE_FLOAT_SIZE,
            ws_int_ptr,
            WORKSPACE_INT_SIZE,
            page_locked_ptr,
            page_locked_size,
            enable_cuda_graph,
            data_type,
            plan_info.as_mut_ptr(),
            *dev.cu_stream() as i64,
        );
    }

    Ok(plan_info.to_vec())
}

pub struct FlashInferPrefill {
    pub key_cache: Tensor,
    pub value_cache: Tensor,
    pub indices: Tensor,
    pub indptr: Tensor,        // Device tensor for paged_kv
    pub indptr_host: Vec<u32>, // Host data for planning
    pub last_len: Tensor,
    pub q_cu_seqlens: Tensor,        // Device tensor for kernel params
    pub q_cu_seqlens_host: Vec<u32>, // Host data for planning
    pub total_num_rows: u32,         // Total tokens (from host)
    pub block_size: usize,
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sm_scale: f32,
}

impl candle::CustomOp1 for FlashInferPrefill {
    fn name(&self) -> &'static str {
        "flashinfer-prefill"
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

impl FlashInferPrefill {
    fn cuda_fwd_impl<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, candle::Shape)> {
        let dev = q.device();
        let (_total_tokens, _num_heads, _head_dim) = q_l.shape().dims3()?;

        let kc_ptr = get_cuda_ptr(&self.key_cache)?;
        let vc_ptr = get_cuda_ptr(&self.value_cache)?;

        let (indices, indices_l) = self.indices.storage_and_layout();
        let indices_ptr = match &*indices {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(indices_l.start_offset()..),
            _ => candle::bail!("indices must be cuda"),
        };

        let (indptr, indptr_l) = self.indptr.storage_and_layout();
        let indptr_ptr = match &*indptr {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(indptr_l.start_offset()..),
            _ => candle::bail!("indptr must be cuda"),
        };

        let (last_len, last_len_l) = self.last_len.storage_and_layout();
        let last_len_ptr = match &*last_len {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(last_len_l.start_offset()..),
            _ => candle::bail!("last_len must be cuda"),
        };

        let (q_lens, q_lens_l) = self.q_cu_seqlens.storage_and_layout();
        let q_lens_ptr = match &*q_lens {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.slice(q_lens_l.start_offset()..),
            _ => candle::bail!("q_cu_seqlens must be cuda"),
        };

        let q_ptr = get_cuda_ptr_storage(q, q_l, q.dtype())?;

        // 0: F16, 1: BF16, 2: U8 (FP8)
        let data_type = match self.key_cache.dtype() {
            DType::U8 => 2,
            DType::BF16 => 1,
            _ => 0,
        };

        let out = unsafe { dev.alloc::<T>(q_l.shape().elem_count()) }.w()?;
        let out_ptr = *out.device_ptr() as *mut std::ffi::c_void;

        let batch_size = self.q_cu_seqlens_host.len() - 1;

        // Use static workspace buffers
        let (ws_float_ptr, ws_int_ptr, page_locked_ptr, page_locked_size) =
            get_or_init_workspace(dev, false)?;

        unsafe {
            kernels::ffi::flashinfer_prefill_wrapper(
                out_ptr,
                q_ptr,
                *q_lens_ptr.device_ptr() as *const i32,
                self.q_cu_seqlens_host.as_ptr().cast(), // Host pointer for planning
                self.total_num_rows as i32,
                kc_ptr,
                vc_ptr,
                *indices_ptr.device_ptr() as *const i32,
                *indptr_ptr.device_ptr() as *const i32,
                self.indptr_host.as_ptr().cast(), // Host pointer for planning
                *last_len_ptr.device_ptr() as *const i32,
                batch_size as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                self.head_dim as i32,
                self.block_size as i32,
                self.sm_scale,
                ws_float_ptr,
                WORKSPACE_FLOAT_SIZE,
                ws_int_ptr,
                WORKSPACE_INT_SIZE,
                page_locked_ptr,
                page_locked_size,
                false,
                data_type,
                *dev.cu_stream() as i64,
            );
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, q_l.shape().clone()))
    }
}

pub fn prefill(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    indices: &Tensor,
    indptr: &Tensor,
    indptr_host: &[u32], // Host slice for planning
    last_len: &Tensor,
    q_cu_seqlens: &Tensor,
    q_cu_seqlens_host: &[u32], // Host slice for planning
    total_num_rows: u32,       // Total tokens (from host)
    block_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sm_scale: f32,
) -> Result<Tensor> {
    let op = FlashInferPrefill {
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        indices: indices.clone(),
        indptr: indptr.clone(),
        indptr_host: indptr_host.to_vec(),
        last_len: last_len.clone(),
        q_cu_seqlens: q_cu_seqlens.clone(),
        q_cu_seqlens_host: q_cu_seqlens_host.to_vec(),
        total_num_rows,
        block_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        sm_scale,
    };
    q.apply_op1(op)
}
