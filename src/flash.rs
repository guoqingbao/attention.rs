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

    let seqlens_q = if let Some(cu) = cu_seqlens_q {
        cu.to_vec1::<u32>()?
    } else {
        vec![0u32, q_len as u32]
    };

    let num_seqs = seqlens_q.len() - 1;

    for seq_idx in 0..num_seqs {
        let q_start = seqlens_q[seq_idx] as usize;
        let q_end = seqlens_q[seq_idx + 1] as usize;
        let seq_q_len = q_end - q_start;
        if seq_q_len == 0 {
            continue;
        }

        let cl_vec = context_lens.to_vec1::<u32>()?;
        let kv_len = cl_vec[seq_idx] as usize;

        let bt_row = block_table.narrow(0, seq_idx, 1)?.squeeze(0)?;
        let bt_row_ptr = {
            let (s, l) = bt_row.storage_and_layout();
            let s = match &*s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("block_table row must be CUDA"),
            };
            *s.slice(l.start_offset()..).device_ptr() as *const c_int
        };

        let q_offset = kv_len.saturating_sub(seq_q_len);
        let q_slice_ptr = unsafe {
            (q_ptr as *const u8).add(q_start * num_q_heads * head_dim * 2)
                as *const std::ffi::c_void
        };
        let o_slice_ptr = unsafe {
            (o_ptr as *mut u8).add(q_start * num_q_heads * head_dim * 2) as *mut std::ffi::c_void
        };

        let sw = sliding_window.unwrap_or(0) as u32;

        if is_fp8 {
            let ks_ptr = scale_gpu_ptr(k_scale)?;
            let vs_ptr = scale_gpu_ptr(v_scale)?;
            let fp8_cache_stride =
                (key_cache.dim(1)? * key_cache.dim(2)? * key_cache.dim(3)?) as u64;
            unsafe {
                kernels::ffi::call_flash_prefill_paged_fp8(
                    q_slice_ptr,
                    kc_ptr,
                    vc_ptr,
                    o_slice_ptr,
                    bt_row_ptr,
                    seq_q_len as u32,
                    kv_len as u32,
                    q_offset as u32,
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
                    q_slice_ptr,
                    kc_ptr,
                    vc_ptr,
                    o_slice_ptr,
                    bt_row_ptr,
                    seq_q_len as u32,
                    kv_len as u32,
                    q_offset as u32,
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
    }

    Ok(o)
}

const SPLIT_K_THRESHOLD: usize = 1024;
pub const NUM_SPLITS: u32 = 8;

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
                    stream,
                );
            }
        }
    }

    Ok(output.clone())
}
