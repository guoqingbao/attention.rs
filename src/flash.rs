use candle_core as candle;
use candle_core::{DType, Result, Tensor};

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use std::ffi::c_int;

#[cfg(feature = "cuda")]
fn scale_gpu_ptr(scale: Option<&Tensor>) -> Result<*const f32> {
    match scale {
        Some(t) => {
            let (s, l) = t.storage_and_layout();
            let s = match &*s {
                candle::Storage::Cuda(c) => c,
                _ => candle::bail!("scale tensor must be CUDA"),
            };
            let slice = s.as_cuda_slice::<f32>()?;
            Ok(*slice.slice(l.start_offset()..).device_ptr() as *const f32)
        }
        None => Ok(std::ptr::null()),
    }
}

#[cfg(feature = "cuda")]
fn gpu_ptr_u32(t: &Tensor) -> Result<*const u32> {
    let (s, l) = t.storage_and_layout();
    let s = match &*s {
        candle::Storage::Cuda(c) => c,
        _ => candle::bail!("tensor must be CUDA"),
    };
    let slice = s.as_cuda_slice::<u32>()?;
    Ok(*slice.slice(l.start_offset()..).device_ptr() as *const u32)
}

#[cfg(feature = "cuda")]
fn get_cuda_stream(dev: &candle::CudaDevice) -> i64 {
    use candle::cuda_backend::cudarc::driver::sys;
    let stream: sys::CUstream = *dev.cu_stream();
    stream as i64
}

#[cfg(feature = "cuda")]
fn ptr_from_tensor(t: &Tensor) -> Result<*const std::ffi::c_void> {
    let (storage, layout) = t.storage_and_layout();
    let cuda_storage = match &*storage {
        candle::Storage::Cuda(c) => c,
        _ => candle::bail!("expected CUDA tensor"),
    };
    let offset = layout.start_offset();

    match t.dtype() {
        DType::BF16 => {
            let slice = cuda_storage.as_cuda_slice::<half::bf16>()?;
            let slice = slice.slice(offset..);
            Ok(*slice.device_ptr() as *const std::ffi::c_void)
        }
        DType::F16 => {
            let slice = cuda_storage.as_cuda_slice::<half::f16>()?;
            let slice = slice.slice(offset..);
            Ok(*slice.device_ptr() as *const std::ffi::c_void)
        }
        DType::U8 => {
            let slice = cuda_storage.as_cuda_slice::<u8>()?;
            let slice = slice.slice(offset..);
            Ok(*slice.device_ptr() as *const std::ffi::c_void)
        }
        DType::F32 => {
            let slice = cuda_storage.as_cuda_slice::<f32>()?;
            let slice = slice.slice(offset..);
            Ok(*slice.device_ptr() as *const std::ffi::c_void)
        }
        DType::U32 => {
            let slice = cuda_storage.as_cuda_slice::<u32>()?;
            let slice = slice.slice(offset..);
            Ok(*slice.device_ptr() as *const std::ffi::c_void)
        }
        dt => candle::bail!("unsupported dtype for ptr_from_tensor: {dt:?}"),
    }
}

#[cfg(feature = "cuda")]
pub fn flash_reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    slot_mapping: &Tensor,
) -> Result<()> {
    let dev = match key.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_reshape_and_cache requires CUDA tensors"),
    };
    let stream = get_cuda_stream(dev);

    let (num_tokens, num_kv_heads, head_dim) = key.dims3()?;
    let block_size = key_cache.dim(1)?;

    let key_ptr = ptr_from_tensor(key)?;
    let value_ptr = ptr_from_tensor(value)?;
    let key_cache_ptr = ptr_from_tensor(key_cache)? as *mut std::ffi::c_void;
    let value_cache_ptr = ptr_from_tensor(value_cache)? as *mut std::ffi::c_void;

    let slot_ptr = {
        let (s, l) = slot_mapping.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c,
            _ => candle::bail!("slot_mapping must be CUDA"),
        };
        let slice = s.as_cuda_slice::<i64>()?;
        *slice.slice(l.start_offset()..).device_ptr() as *const i64
    };

    let is_fp8 = key_cache.dtype() == DType::U8;

    if is_fp8 {
        let ks_ptr = scale_gpu_ptr(k_scale)?;
        let vs_ptr = scale_gpu_ptr(v_scale)?;
        unsafe {
            kernels::ffi::call_flash_reshape_and_cache_fp8_kv(
                key_ptr,
                value_ptr,
                key_cache_ptr,
                value_cache_ptr,
                slot_ptr,
                num_tokens as u32,
                num_kv_heads as u32,
                head_dim as u32,
                block_size as u32,
                ks_ptr,
                vs_ptr,
                stream,
            );
        }
    } else {
        unsafe {
            kernels::ffi::call_flash_reshape_and_cache_bf16(
                key_ptr,
                value_ptr,
                key_cache_ptr,
                value_cache_ptr,
                slot_ptr,
                num_tokens as u32,
                num_kv_heads as u32,
                head_dim as u32,
                block_size as u32,
                stream,
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn flash_prefill(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    context_lens: &Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    softcap: f32,
    sliding_window: Option<usize>,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    cu_seqlens_q: Option<&Tensor>,
    max_seqlen_q: usize,
) -> Result<Tensor> {
    let dev = match query.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_prefill requires CUDA"),
    };
    let stream = get_cuda_stream(dev);

    let q_len = query.dim(0)?;
    let block_size = key_cache.dim(1)?;

    let o = Tensor::zeros_like(query)?;

    let q_ptr = ptr_from_tensor(query)?;
    let kc_ptr = ptr_from_tensor(key_cache)?;
    let vc_ptr = ptr_from_tensor(value_cache)?;
    let o_ptr = ptr_from_tensor(&o)? as *mut std::ffi::c_void;

    let is_fp8 = key_cache.dtype() == DType::U8;
    let sw = sliding_window.unwrap_or(0) as u32;

    let block_table_stride = block_table.dim(1)? as u32;
    let bt_ptr = {
        let (s, l) = block_table.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("block_table must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };

    let (cu_ptr, cl_ptr, num_seqs, actual_max_q_len) = if let Some(cu) = cu_seqlens_q {
        let ns = cu.dim(0)? - 1;
        (
            gpu_ptr_u32(cu)?,
            gpu_ptr_u32(context_lens)?,
            ns,
            max_seqlen_q,
        )
    } else {
        let cu_t = Tensor::from_vec(vec![0u32, q_len as u32], 2, query.device())?;
        (
            gpu_ptr_u32(&cu_t)?,
            gpu_ptr_u32(context_lens)?,
            1usize,
            q_len,
        )
    };

    if is_fp8 {
        let ks_ptr = scale_gpu_ptr(k_scale)?;
        let vs_ptr = scale_gpu_ptr(v_scale)?;
        let fp8_cache_stride = (key_cache.dim(1)? * key_cache.dim(2)? * key_cache.dim(3)?) as u64;
        unsafe {
            kernels::ffi::call_flash_prefill_paged_fp8(
                q_ptr,
                kc_ptr,
                vc_ptr,
                o_ptr,
                bt_ptr,
                block_table_stride,
                cu_ptr,
                cl_ptr,
                num_seqs as u32,
                actual_max_q_len as u32,
                num_q_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                block_size as u32,
                sw,
                1,
                scale,
                softcap,
                ks_ptr,
                vs_ptr,
                fp8_cache_stride,
                stream,
            );
        }
    } else {
        unsafe {
            kernels::ffi::call_flash_prefill_paged(
                q_ptr,
                kc_ptr,
                vc_ptr,
                o_ptr,
                bt_ptr,
                block_table_stride,
                cu_ptr,
                cl_ptr,
                num_seqs as u32,
                actual_max_q_len as u32,
                num_q_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                block_size as u32,
                sw,
                1,
                scale,
                softcap,
                stream,
            );
        }
    }

    Ok(o)
}

pub const SPLIT_K_THRESHOLD: usize = 1024;
pub const NUM_SPLITS: u32 = 8;
pub const TQ_NUM_SPLITS: u32 = 16;

#[cfg(feature = "cuda")]
pub fn flash_decode(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    output: &Tensor,
    max_context_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    softcap: f32,
    sliding_window: Option<usize>,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    workspace: Option<&Tensor>,
) -> Result<Tensor> {
    let dev = match query.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_decode requires CUDA"),
    };
    let stream = get_cuda_stream(dev);

    let num_seqs = query.dim(0)?;
    let block_size = key_cache.dim(1)?;
    let q_stride = (num_q_heads * head_dim) as u32;

    let q_ptr = ptr_from_tensor(query)?;
    let kc_ptr = ptr_from_tensor(key_cache)?;
    let vc_ptr = ptr_from_tensor(value_cache)?;
    let o_ptr = ptr_from_tensor(output)? as *mut std::ffi::c_void;

    let bt_ptr = {
        let (s, l) = block_tables.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("block_tables must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };
    let cl_ptr = {
        let (s, l) = context_lens.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("context_lens must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };

    let max_blocks_per_seq = block_tables.dim(1)? as u32;
    let sw = sliding_window.unwrap_or(0) as u32;
    let is_fp8 = key_cache.dtype() == DType::U8;
    let use_splitk = max_context_len >= SPLIT_K_THRESHOLD && workspace.is_some();

    // GQA disabled for native flash path: shared memory overflow at higher ratios
    // (e.g. GQA=8 with HDIM=256 needs 64KB smem, exceeding 48KB limit).
    // Each CTA handles one Q head; kv_head = q_head / gqa_ratio is computed inside kernel.
    let effective_gqa: usize = 1;

    if is_fp8 {
        let fp8_cache_stride = (block_size * num_kv_heads * head_dim) as u64;
        let ks_ptr = scale_gpu_ptr(k_scale)?;
        let vs_ptr = scale_gpu_ptr(v_scale)?;

        if use_splitk {
            let ws = workspace.unwrap();
            let ws_ptr = ptr_from_tensor(ws)? as *mut std::ffi::c_void;
            unsafe {
                kernels::ffi::call_flash_decode_paged_splitk_fp8(
                    q_ptr,
                    kc_ptr,
                    vc_ptr,
                    ws_ptr,
                    bt_ptr,
                    cl_ptr,
                    max_blocks_per_seq,
                    num_q_heads as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                    block_size as u32,
                    scale,
                    num_seqs as u32,
                    NUM_SPLITS,
                    q_stride,
                    softcap,
                    ks_ptr,
                    vs_ptr,
                    fp8_cache_stride,
                    sw,
                    effective_gqa as u32,
                    stream,
                );
                kernels::ffi::call_flash_decode_paged_reduce(
                    ws_ptr as *const std::ffi::c_void,
                    o_ptr,
                    num_q_heads as u32,
                    head_dim as u32,
                    NUM_SPLITS,
                    num_seqs as u32,
                    stream,
                );
            }
        } else {
            unsafe {
                kernels::ffi::call_flash_decode_paged_fp8(
                    q_ptr,
                    kc_ptr,
                    vc_ptr,
                    o_ptr,
                    bt_ptr,
                    cl_ptr,
                    max_blocks_per_seq,
                    num_q_heads as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                    block_size as u32,
                    scale,
                    num_seqs as u32,
                    q_stride,
                    sw,
                    softcap,
                    ks_ptr,
                    vs_ptr,
                    fp8_cache_stride,
                    effective_gqa as u32,
                    stream,
                );
            }
        }
    } else {
        if use_splitk {
            let ws = workspace.unwrap();
            let ws_ptr = ptr_from_tensor(ws)? as *mut std::ffi::c_void;
            unsafe {
                kernels::ffi::call_flash_decode_paged_splitk(
                    q_ptr,
                    kc_ptr,
                    vc_ptr,
                    ws_ptr,
                    bt_ptr,
                    cl_ptr,
                    max_blocks_per_seq,
                    num_q_heads as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                    block_size as u32,
                    scale,
                    num_seqs as u32,
                    NUM_SPLITS,
                    q_stride,
                    softcap,
                    sw,
                    effective_gqa as u32,
                    stream,
                );
                kernels::ffi::call_flash_decode_paged_reduce(
                    ws_ptr as *const std::ffi::c_void,
                    o_ptr,
                    num_q_heads as u32,
                    head_dim as u32,
                    NUM_SPLITS,
                    num_seqs as u32,
                    stream,
                );
            }
        } else {
            unsafe {
                kernels::ffi::call_flash_decode_paged(
                    q_ptr,
                    kc_ptr,
                    vc_ptr,
                    o_ptr,
                    bt_ptr,
                    cl_ptr,
                    max_blocks_per_seq,
                    num_q_heads as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                    block_size as u32,
                    scale,
                    num_seqs as u32,
                    q_stride,
                    sw,
                    softcap,
                    effective_gqa as u32,
                    stream,
                );
            }
        }
    }

    Ok(output.clone())
}

// ============================================================================
// TurboQuant k8v4 wrappers
// ============================================================================

#[cfg(feature = "cuda")]
pub fn flash_tq_store_k8v4(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    slot_mapping: &Tensor,
    k_scale: Option<&Tensor>,
) -> Result<()> {
    let dev = match key.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_tq_store requires CUDA tensors"),
    };
    let stream = get_cuda_stream(dev);

    let (num_tokens, num_kv_heads, head_dim) = key.dims3()?;
    let block_size = key_cache.dim(1)?;

    let key_ptr = ptr_from_tensor(key)?;
    let value_ptr = ptr_from_tensor(value)?;
    let kc_ptr = ptr_from_tensor(key_cache)? as *mut std::ffi::c_void;
    let va_ptr = ptr_from_tensor(v_absmax)? as *mut std::ffi::c_void;
    let vq_ptr = ptr_from_tensor(v_quant)? as *mut std::ffi::c_void;

    let slot_ptr = {
        let (s, l) = slot_mapping.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c,
            _ => candle::bail!("slot_mapping must be CUDA"),
        };
        let slice = s.as_cuda_slice::<i64>()?;
        *slice.slice(l.start_offset()..).device_ptr() as *const i64
    };

    let ks_ptr = scale_gpu_ptr(k_scale)?;

    unsafe {
        kernels::ffi::call_flash_tq_store_k8v4(
            key_ptr,
            value_ptr,
            kc_ptr,
            va_ptr,
            vq_ptr,
            slot_ptr,
            num_tokens as u32,
            num_kv_heads as u32,
            head_dim as u32,
            block_size as u32,
            ks_ptr,
            stream,
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn flash_tq_decode_k8v4(
    query: &Tensor,
    key_cache: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    output: &Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    softcap: f32,
    k_scale: Option<&Tensor>,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    let dev = match query.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_tq_decode requires CUDA"),
    };
    let stream = get_cuda_stream(dev);

    let num_seqs = query.dim(0)?;
    let block_size = key_cache.dim(1)?;
    let q_stride = (num_q_heads * head_dim) as u32;

    let q_ptr = ptr_from_tensor(query)?;
    let kc_ptr = ptr_from_tensor(key_cache)?;
    let va_ptr = ptr_from_tensor(v_absmax)?;
    let vq_ptr = ptr_from_tensor(v_quant)?;
    let o_ptr = ptr_from_tensor(output)? as *mut std::ffi::c_void;

    let bt_ptr = {
        let (s, l) = block_tables.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("block_tables must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };
    let cl_ptr = {
        let (s, l) = context_lens.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("context_lens must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };

    let max_blocks_per_seq = block_tables.dim(1)? as u32;
    let ks_ptr = scale_gpu_ptr(k_scale)?;
    let sw = sliding_window.unwrap_or(0) as u32;

    unsafe {
        kernels::ffi::call_flash_tq_decode_k8v4(
            q_ptr,
            kc_ptr,
            va_ptr,
            vq_ptr,
            o_ptr,
            bt_ptr,
            cl_ptr,
            max_blocks_per_seq,
            num_q_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            block_size as u32,
            scale,
            num_seqs as u32,
            q_stride,
            softcap,
            ks_ptr,
            sw,
            stream,
        );
    }

    Ok(output.clone())
}

#[cfg(feature = "cuda")]
pub fn flash_tq_decode_k8v4_splitk(
    query: &Tensor,
    key_cache: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    output: &Tensor,
    max_context_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    softcap: f32,
    k_scale: Option<&Tensor>,
    workspace: Option<&Tensor>,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    let use_splitk = max_context_len >= SPLIT_K_THRESHOLD && workspace.is_some();

    if !use_splitk {
        return flash_tq_decode_k8v4(
            query,
            key_cache,
            v_absmax,
            v_quant,
            block_tables,
            context_lens,
            output,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            softcap,
            k_scale,
            sliding_window,
        );
    }

    let dev = match query.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_tq_decode_k8v4_splitk requires CUDA"),
    };
    let stream = get_cuda_stream(dev);

    let num_seqs = query.dim(0)?;
    let block_size = key_cache.dim(1)?;
    let q_stride = (num_q_heads * head_dim) as u32;

    let q_ptr = ptr_from_tensor(query)?;
    let kc_ptr = ptr_from_tensor(key_cache)?;
    let va_ptr = ptr_from_tensor(v_absmax)?;
    let vq_ptr = ptr_from_tensor(v_quant)?;

    let bt_ptr = {
        let (s, l) = block_tables.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("block_tables must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };
    let cl_ptr = {
        let (s, l) = context_lens.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("context_lens must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };

    let max_blocks_per_seq = block_tables.dim(1)? as u32;
    let ks_ptr = scale_gpu_ptr(k_scale)?;
    let sw = sliding_window.unwrap_or(0) as u32;
    let o_ptr = ptr_from_tensor(output)? as *mut std::ffi::c_void;

    let ws = workspace.unwrap();
    let ws_ptr = ptr_from_tensor(ws)? as *mut std::ffi::c_void;

    unsafe {
        kernels::ffi::call_flash_tq_decode_k8v4_splitk(
            q_ptr,
            kc_ptr,
            va_ptr,
            vq_ptr,
            ws_ptr,
            bt_ptr,
            cl_ptr,
            max_blocks_per_seq,
            num_q_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            block_size as u32,
            scale,
            TQ_NUM_SPLITS,
            num_seqs as u32,
            q_stride,
            softcap,
            ks_ptr,
            sw,
            stream,
        );
        kernels::ffi::call_flash_decode_paged_reduce(
            ws_ptr as *const std::ffi::c_void,
            o_ptr,
            num_q_heads as u32,
            head_dim as u32,
            TQ_NUM_SPLITS,
            num_seqs as u32,
            stream,
        );
    }

    Ok(output.clone())
}

// ============================================================================
// TurboQuant turbo4: 4-bit K + 4-bit V
// ============================================================================

#[cfg(feature = "cuda")]
pub fn flash_tq4_store(
    key: &Tensor,
    value: &Tensor,
    k_absmax: &Tensor,
    k_quant: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    slot_mapping: &Tensor,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    let dev = match key.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_tq4_store requires CUDA"),
    };
    let stream = get_cuda_stream(dev);
    let num_tokens = key.dim(0)?;

    let k_ptr = ptr_from_tensor(key)?;
    let v_ptr = ptr_from_tensor(value)?;
    let ka_ptr = ptr_from_tensor(k_absmax)? as *mut std::ffi::c_void;
    let kq_ptr = ptr_from_tensor(k_quant)? as *mut std::ffi::c_void;
    let va_ptr = ptr_from_tensor(v_absmax)? as *mut std::ffi::c_void;
    let vq_ptr = ptr_from_tensor(v_quant)? as *mut std::ffi::c_void;
    let slot_ptr = {
        let (s, l) = slot_mapping.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<i64>()?,
            _ => candle::bail!("slot_mapping must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const i64
    };

    unsafe {
        kernels::ffi::call_flash_tq4_store(
            k_ptr,
            v_ptr,
            ka_ptr,
            kq_ptr,
            va_ptr,
            vq_ptr,
            slot_ptr,
            num_tokens as u32,
            num_kv_heads as u32,
            head_dim as u32,
            block_size as u32,
            stream,
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn flash_tq4_decode(
    query: &Tensor,
    k_absmax: &Tensor,
    k_quant: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    output: &Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_context_len: usize,
    scale: f32,
    softcap: f32,
    sliding_window: Option<usize>,
    workspace: Option<&Tensor>,
) -> Result<Tensor> {
    let dev = match query.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_tq4_decode requires CUDA"),
    };
    let stream = get_cuda_stream(dev);

    let num_seqs = query.dim(0)?;
    let block_size_from_absmax = {
        let dims = k_absmax.dims();
        if dims.len() >= 2 {
            dims[1]
        } else {
            16
        }
    };
    let q_stride = (num_q_heads * head_dim) as u32;

    let q_ptr = ptr_from_tensor(query)?;
    let ka_ptr = ptr_from_tensor(k_absmax)?;
    let kq_ptr = ptr_from_tensor(k_quant)?;
    let va_ptr = ptr_from_tensor(v_absmax)?;
    let vq_ptr = ptr_from_tensor(v_quant)?;
    let o_ptr = ptr_from_tensor(output)? as *mut std::ffi::c_void;

    let bt_ptr = {
        let (s, l) = block_tables.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("block_tables must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };
    let cl_ptr = {
        let (s, l) = context_lens.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("context_lens must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };
    let max_blocks_per_seq = block_tables.dim(1)? as u32;
    let sw = sliding_window.unwrap_or(0) as u32;

    let use_splitk = max_context_len >= SPLIT_K_THRESHOLD && workspace.is_some();

    if use_splitk {
        let ws = workspace.unwrap();
        let ws_ptr = ptr_from_tensor(ws)? as *mut std::ffi::c_void;
        unsafe {
            kernels::ffi::call_flash_tq4_decode_splitk(
                q_ptr,
                ka_ptr,
                kq_ptr,
                va_ptr,
                vq_ptr,
                ws_ptr,
                bt_ptr,
                cl_ptr,
                max_blocks_per_seq,
                num_q_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                block_size_from_absmax as u32,
                scale,
                TQ_NUM_SPLITS,
                num_seqs as u32,
                q_stride,
                softcap,
                sw,
                stream,
            );
            kernels::ffi::call_flash_decode_paged_reduce(
                ws_ptr as *const std::ffi::c_void,
                o_ptr,
                num_q_heads as u32,
                head_dim as u32,
                TQ_NUM_SPLITS,
                num_seqs as u32,
                stream,
            );
        }
    } else {
        unsafe {
            kernels::ffi::call_flash_tq4_decode(
                q_ptr,
                ka_ptr,
                kq_ptr,
                va_ptr,
                vq_ptr,
                o_ptr,
                bt_ptr,
                cl_ptr,
                max_blocks_per_seq,
                num_q_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                block_size_from_absmax as u32,
                scale,
                num_seqs as u32,
                q_stride,
                softcap,
                sw,
                stream,
            );
        }
    }

    Ok(output.clone())
}

// ============================================================================
// TurboQuant turbo3: 3-bit K + 4-bit V
// ============================================================================

#[cfg(feature = "cuda")]
pub fn flash_tq3_store(
    key: &Tensor,
    value: &Tensor,
    k_absmax: &Tensor,
    k_quant: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    slot_mapping: &Tensor,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<()> {
    let dev = match key.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_tq3_store requires CUDA"),
    };
    let stream = get_cuda_stream(dev);
    let num_tokens = key.dim(0)?;

    let k_ptr = ptr_from_tensor(key)?;
    let v_ptr = ptr_from_tensor(value)?;
    let ka_ptr = ptr_from_tensor(k_absmax)? as *mut std::ffi::c_void;
    let kq_ptr = ptr_from_tensor(k_quant)? as *mut std::ffi::c_void;
    let va_ptr = ptr_from_tensor(v_absmax)? as *mut std::ffi::c_void;
    let vq_ptr = ptr_from_tensor(v_quant)? as *mut std::ffi::c_void;
    let slot_ptr = {
        let (s, l) = slot_mapping.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<i64>()?,
            _ => candle::bail!("slot_mapping must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const i64
    };

    unsafe {
        kernels::ffi::call_flash_tq3_store(
            k_ptr,
            v_ptr,
            ka_ptr,
            kq_ptr,
            va_ptr,
            vq_ptr,
            slot_ptr,
            num_tokens as u32,
            num_kv_heads as u32,
            head_dim as u32,
            block_size as u32,
            stream,
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn flash_tq3_decode(
    query: &Tensor,
    k_absmax: &Tensor,
    k_quant: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    output: &Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_context_len: usize,
    scale: f32,
    softcap: f32,
    sliding_window: Option<usize>,
    workspace: Option<&Tensor>,
) -> Result<Tensor> {
    let dev = match query.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_tq3_decode requires CUDA"),
    };
    let stream = get_cuda_stream(dev);

    let num_seqs = query.dim(0)?;
    let block_size_from_absmax = {
        let dims = k_absmax.dims();
        if dims.len() >= 2 {
            dims[1]
        } else {
            16
        }
    };
    let q_stride = (num_q_heads * head_dim) as u32;

    let q_ptr = ptr_from_tensor(query)?;
    let ka_ptr = ptr_from_tensor(k_absmax)?;
    let kq_ptr = ptr_from_tensor(k_quant)?;
    let va_ptr = ptr_from_tensor(v_absmax)?;
    let vq_ptr = ptr_from_tensor(v_quant)?;
    let o_ptr = ptr_from_tensor(output)? as *mut std::ffi::c_void;

    let bt_ptr = {
        let (s, l) = block_tables.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("block_tables must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };
    let cl_ptr = {
        let (s, l) = context_lens.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("context_lens must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };
    let max_blocks_per_seq = block_tables.dim(1)? as u32;
    let sw = sliding_window.unwrap_or(0) as u32;

    let use_splitk = max_context_len >= SPLIT_K_THRESHOLD && workspace.is_some();

    if use_splitk {
        let ws = workspace.unwrap();
        let ws_ptr = ptr_from_tensor(ws)? as *mut std::ffi::c_void;
        unsafe {
            kernels::ffi::call_flash_tq3_decode_splitk(
                q_ptr,
                ka_ptr,
                kq_ptr,
                va_ptr,
                vq_ptr,
                ws_ptr,
                bt_ptr,
                cl_ptr,
                max_blocks_per_seq,
                num_q_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                block_size_from_absmax as u32,
                scale,
                TQ_NUM_SPLITS,
                num_seqs as u32,
                q_stride,
                softcap,
                sw,
                stream,
            );
            kernels::ffi::call_flash_decode_paged_reduce(
                ws_ptr as *const std::ffi::c_void,
                o_ptr,
                num_q_heads as u32,
                head_dim as u32,
                TQ_NUM_SPLITS,
                num_seqs as u32,
                stream,
            );
        }
    } else {
        unsafe {
            kernels::ffi::call_flash_tq3_decode(
                q_ptr,
                ka_ptr,
                kq_ptr,
                va_ptr,
                vq_ptr,
                o_ptr,
                bt_ptr,
                cl_ptr,
                max_blocks_per_seq,
                num_q_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                block_size_from_absmax as u32,
                scale,
                num_seqs as u32,
                q_stride,
                softcap,
                sw,
                stream,
            );
        }
    }

    Ok(output.clone())
}

// ============================================================================
// TurboQuant 4-bit prefill (turbo4 only: 4-bit K + 4-bit V)
// ============================================================================

#[cfg(feature = "cuda")]
pub fn flash_tq4_prefill(
    query: &Tensor,
    k_absmax: &Tensor,
    k_quant: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    block_table: &Tensor,
    context_lens: &Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    softcap: f32,
    sliding_window: Option<usize>,
    block_size: usize,
    cu_seqlens_q: Option<&Tensor>,
    max_seqlen_q: usize,
) -> Result<Tensor> {
    let dev = match query.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_tq4_prefill requires CUDA"),
    };
    let stream = get_cuda_stream(dev);

    let q_len = query.dim(0)?;
    let o = Tensor::zeros_like(query)?;

    let q_ptr = ptr_from_tensor(query)?;
    let ka_ptr = ptr_from_tensor(k_absmax)?;
    let kq_ptr = ptr_from_tensor(k_quant)?;
    let va_ptr = ptr_from_tensor(v_absmax)?;
    let vq_ptr = ptr_from_tensor(v_quant)?;
    let o_ptr = ptr_from_tensor(&o)? as *mut std::ffi::c_void;

    let sw = sliding_window.unwrap_or(0) as u32;

    let block_table_stride = block_table.dim(1)? as u32;
    let bt_ptr = {
        let (s, l) = block_table.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("block_table must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };

    let (cu_ptr, cl_ptr, num_seqs, actual_max_q_len) = if let Some(cu) = cu_seqlens_q {
        let ns = cu.dim(0)? - 1;
        (
            gpu_ptr_u32(cu)?,
            gpu_ptr_u32(context_lens)?,
            ns,
            max_seqlen_q,
        )
    } else {
        let cu_t = Tensor::from_vec(vec![0u32, q_len as u32], 2, query.device())?;
        (
            gpu_ptr_u32(&cu_t)?,
            gpu_ptr_u32(context_lens)?,
            1usize,
            q_len,
        )
    };

    unsafe {
        kernels::ffi::call_flash_tq4_prefill(
            q_ptr,
            ka_ptr,
            kq_ptr,
            va_ptr,
            vq_ptr,
            o_ptr,
            bt_ptr,
            block_table_stride,
            cu_ptr,
            cl_ptr,
            num_seqs as u32,
            actual_max_q_len as u32,
            num_q_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            block_size as u32,
            sw,
            1,
            scale,
            softcap,
            stream,
        );
    }

    Ok(o)
}

#[cfg(feature = "cuda")]
pub fn flash_tq3_prefill(
    query: &Tensor,
    k_absmax: &Tensor,
    k_quant: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    block_table: &Tensor,
    context_lens: &Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    softcap: f32,
    sliding_window: Option<usize>,
    block_size: usize,
    cu_seqlens_q: Option<&Tensor>,
    max_seqlen_q: usize,
) -> Result<Tensor> {
    let dev = match query.device() {
        candle::Device::Cuda(d) => d,
        _ => candle::bail!("flash_tq3_prefill requires CUDA"),
    };
    let stream = get_cuda_stream(dev);

    let q_len = query.dim(0)?;
    let o = Tensor::zeros_like(query)?;

    let q_ptr = ptr_from_tensor(query)?;
    let ka_ptr = ptr_from_tensor(k_absmax)?;
    let kq_ptr = ptr_from_tensor(k_quant)?;
    let va_ptr = ptr_from_tensor(v_absmax)?;
    let vq_ptr = ptr_from_tensor(v_quant)?;
    let o_ptr = ptr_from_tensor(&o)? as *mut std::ffi::c_void;

    let sw = sliding_window.unwrap_or(0) as u32;

    let block_table_stride = block_table.dim(1)? as u32;
    let bt_ptr = {
        let (s, l) = block_table.storage_and_layout();
        let s = match &*s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("block_table must be CUDA"),
        };
        *s.slice(l.start_offset()..).device_ptr() as *const c_int
    };

    let (cu_ptr, cl_ptr, num_seqs, actual_max_q_len) = if let Some(cu) = cu_seqlens_q {
        let ns = cu.dim(0)? - 1;
        (
            gpu_ptr_u32(cu)?,
            gpu_ptr_u32(context_lens)?,
            ns,
            max_seqlen_q,
        )
    } else {
        let cu_t = Tensor::from_vec(vec![0u32, q_len as u32], 2, query.device())?;
        (
            gpu_ptr_u32(&cu_t)?,
            gpu_ptr_u32(context_lens)?,
            1usize,
            q_len,
        )
    };

    unsafe {
        kernels::ffi::call_flash_tq3_prefill(
            q_ptr,
            ka_ptr,
            kq_ptr,
            va_ptr,
            vq_ptr,
            o_ptr,
            bt_ptr,
            block_table_stride,
            cu_ptr,
            cl_ptr,
            num_seqs as u32,
            actual_max_q_len as u32,
            num_q_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            block_size as u32,
            sw,
            1,
            scale,
            softcap,
            stream,
        );
    }

    Ok(o)
}
