//! GCU flash-attention wrappers.
//!
//! All FA paths use Aten topsfa (`topsfa_flash_attn_*`). The old `libflashkernels`
//! host path has been removed — enable feature `aten` (or `flashattn`, which
//! implies `aten`) to compile these kernels in.
pub mod ffi;
pub mod param;
use candle_core as candle;
use candle_core::{DType, Result, Storage, Tensor};
use std::os::raw::c_void;
use std::ptr;

#[macro_export]
macro_rules! gcu_device_ptr {
    ($tensor:expr, $ty:ty) => {{
        let (s, c_l) = $tensor.storage_and_layout();
        match &*s {
            Storage::Gcu(c) => {
                let c = c.as_gcu_slice::<$ty>().unwrap();
                let c = c.slice(c_l.start_offset()..);
                c.device_ptr()
            }
            _ => panic!("Must be a GCU tensor!"),
        }
    }};
}

fn dtype_code(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F16 => Ok(4),
        DType::BF16 => Ok(5),
        DType::F32 => Ok(8),
        dt => candle_core::bail!("unsupported dtype for GCU flash attention ({dt:?})"),
    }
}

/// Avoid allocating a fresh buffer when the tensor is already contiguous
/// (required: realloc / host sync during CUDA/GCU graph capture/replay).
fn maybe_contiguous(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// Grow-only float scratch for flash FA softmax_lse.
/// Per-call `dev.alloc` + free inside graph capture breaks replay.
mod flash_lse_pool {
    use candle_core::gcu_backend::ubridge::device_ptr::DevicePtr;
    use candle_core::gcu_backend::WrapErr;
    use candle_core::Result;
    use std::cell::RefCell;

    struct LsePool {
        buf: candle_core::gcu_backend::ubridge::gcu_slice::GcuSlice<f32>,
        elems: usize,
        device_id: usize,
    }

    thread_local! {
        static POOL: RefCell<Option<LsePool>> = const { RefCell::new(None) };
    }

    pub fn get_softmax_lse_ptr(
        dev: &candle_core::gcu_backend::GcuDevice,
        needed_elems: usize,
    ) -> Result<*mut std::ffi::c_void> {
        let needed = needed_elems.max(1);
        POOL.with(|cell| {
            let mut slot = cell.borrow_mut();
            let device_id = dev.id();
            let needs_realloc = match slot.as_ref() {
                None => true,
                Some(p) => p.device_id != device_id || p.elems < needed,
            };
            if needs_realloc {
                #[cfg(feature = "graph")]
                {
                    use candle_core::gcu_backend::ubridge::gcu_device::GcuDevice as RawDevice;
                    use candle_core::gcu_backend::ubridge::gcu_slice::driv::topsStreamCaptureStatus;
                    let capturing = RawDevice::capture_status(dev.stream_inner().expect("stream"))
                        == Ok(topsStreamCaptureStatus::topsStreamCaptureStatusActive);
                    if capturing {
                        if let Some(p) = slot.as_ref() {
                            if p.elems < needed {
                                candle_core::bail!(
                                    "flash FA LSE scratch too small during graph capture \
                                     (have {}, need {}) — pre-size for max batch before capture",
                                    p.elems,
                                    needed
                                );
                            }
                        }
                    }
                }
                let old = slot.take();
                let elems = old.as_ref().map(|p| p.elems.max(needed)).unwrap_or(needed);
                drop(old);
                let buf = dev.alloc::<f32>(elems).w()?;
                *slot = Some(LsePool {
                    buf,
                    elems,
                    device_id,
                });
            }
            let pool = slot.as_ref().unwrap();
            Ok(pool.buf.device_ptr() as *mut std::ffi::c_void)
        })
    }
}

use flash_lse_pool::get_softmax_lse_ptr;

/// During graph warmup/capture, oversize decode LSE for max planned batch (32).
fn flash_lse_elems(batch_or_tokens: usize, seqlen_q: usize, num_heads: usize) -> usize {
    let base = batch_or_tokens * seqlen_q * num_heads;
    #[cfg(feature = "graph")]
    {
        if candle_core::is_param_cache_enabled() {
            return base.max(32 * seqlen_q * num_heads);
        }
    }
    base
}

fn require_aten_fa() -> Result<()> {
    #[cfg(feature = "aten")]
    {
        Ok(())
    }
    #[cfg(not(feature = "aten"))]
    {
        candle_core::bail!(
            "GCU flash attention requires feature `aten` (topsfa). \
             The libflashkernels path has been removed."
        )
    }
}

fn flash_attn_func<
    T: candle::gcu_backend::GcuDType + candle::gcu_backend::DeviceCopy + candle::WithDType,
>(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    softmax_scale: f32,
    alibi_slopes: &Option<Tensor>,
    softcap: Option<f32>,
    is_causal: bool,
) -> Result<Tensor> {
    #[cfg(not(feature = "aten"))]
    {
        let _ = (
            query,
            key,
            value,
            softmax_scale,
            alibi_slopes,
            softcap,
            is_causal,
        );
        return require_aten_fa().map(|_| unreachable!());
    }
    #[cfg(feature = "aten")]
    {
        use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
        use candle::gcu_backend::WrapErr;

        let (batch, seqlen_q, attention_heads, head_size) = query.shape().dims4().unwrap();
        let (_, seqlen_k, num_heads_k, _) = key.shape().dims4().unwrap();
        let dev = query.device().as_gcu_device()?;
        let query_ptr = gcu_device_ptr!(query, T);
        let key_ptr = gcu_device_ptr!(key, T);
        let value_ptr = gcu_device_ptr!(value, T);

        let output = dev.alloc::<T>(query.elem_count()).w()?;
        let stream = dev.stream_inner().expect("Unable to obtain stream");
        let softmax_lse =
            get_softmax_lse_ptr(dev, flash_lse_elems(batch, seqlen_q, attention_heads))?;
        let code = dtype_code(query.dtype())?;

        let ret = unsafe {
            candle::gcu_backend::ubridge::ffi::topsfa_flash_attn_fwd(
                output.device_ptr() as *mut c_void,
                softmax_lse as *mut c_void,
                query_ptr as *const c_void,
                key_ptr as *const c_void,
                value_ptr as *const c_void,
                alibi_slopes
                    .as_ref()
                    .map_or(ptr::null(), |t| gcu_device_ptr!(t, T) as *const c_void),
                batch as i32,
                seqlen_q as i32,
                seqlen_k as i32,
                attention_heads as i32,
                num_heads_k as i32,
                head_size as i32,
                softmax_scale,
                softcap.unwrap_or(0.0),
                is_causal as i32,
                -1,
                -1,
                code,
                stream as *const c_void,
            )
        };
        if ret != 0 {
            candle_core::bail!("topsfa_flash_attn_fwd failed with code {ret}");
        }
        let output = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
        Tensor::from_storage(candle::Storage::Gcu(output), query.shape())
    }
}

pub fn flash_attn(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    flash_attn_alibi(query, key, value, softmax_scale, &None, None, causal)
}

pub fn flash_attn_alibi(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    softmax_scale: f32,
    alibi_slopes: &Option<Tensor>,
    softcap: Option<f32>,
    causal: bool,
) -> Result<Tensor> {
    use half::{bf16, f16};
    match query.dtype() {
        DType::F16 => flash_attn_func::<f16>(
            query,
            key,
            value,
            softmax_scale,
            alibi_slopes,
            softcap,
            causal,
        ),
        DType::BF16 => flash_attn_func::<bf16>(
            query,
            key,
            value,
            softmax_scale,
            alibi_slopes,
            softcap,
            causal,
        ),
        dt => {
            candle::bail!("flash_attn is only supported for f16 and bf16 ({dt:?})")
        }
    }
}

fn flash_attn_varlen_func<
    T: candle::gcu_backend::GcuDType + candle::gcu_backend::DeviceCopy + candle::WithDType,
>(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    block_table: &Option<Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    alibi_slopes: &Option<Tensor>,
    window_size_left: Option<i32>,
    window_size_right: Option<i32>,
    causal: bool,
) -> Result<Tensor> {
    #[cfg(not(feature = "aten"))]
    {
        let _ = (
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            softcap,
            alibi_slopes,
            window_size_left,
            window_size_right,
            causal,
        );
        return require_aten_fa().map(|_| unreachable!());
    }
    #[cfg(feature = "aten")]
    {
        use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
        use candle_core::gcu_backend::WrapErr;

        // topsfa: window=(-1,-1) + is_causal. flashkernels' window_right=0 convention NaNs.
        let window_left = window_size_left.unwrap_or(-1);
        let window_right = window_size_right.unwrap_or(-1);

        let query = maybe_contiguous(query)?;
        let key = maybe_contiguous(key)?;
        let value = maybe_contiguous(value)?;
        let cu_seqlens_q = maybe_contiguous(cu_seqlens_q)?;
        let cu_seqlens_k = maybe_contiguous(cu_seqlens_k)?;
        let block_table = match block_table {
            Some(bt) => Some(maybe_contiguous(bt)?),
            None => None,
        };

        let total_q = query.shape().dim(0)?;
        let num_heads = query.shape().dim(1)?;
        let head_size = query.shape().dim(2)?;
        let total_k = key.shape().dim(0)?;
        let num_heads_k = key.shape().dim(1)?;
        let batch = cu_seqlens_q.dim(0)? - 1;

        let dev = query.device().as_gcu_device()?;
        let output = dev.alloc::<T>(query.elem_count()).w()?;
        let softmax_lse = get_softmax_lse_ptr(dev, flash_lse_elems(total_q, 1, num_heads))?;
        let stream = dev.stream_inner().expect("Unable to obtain stream");
        let code = dtype_code(query.dtype())?;

        let (has_bt, bt_batch, bt_max_blocks) = if let Some(bt) = &block_table {
            (1i32, bt.dim(0)? as i32, bt.dim(1)? as i32)
        } else {
            (0, 0, 0)
        };

        let ret = unsafe {
            candle::gcu_backend::ubridge::ffi::topsfa_flash_attn_varlen_fwd(
                output.device_ptr() as *mut c_void,
                softmax_lse as *mut c_void,
                gcu_device_ptr!(query, T) as *const c_void,
                gcu_device_ptr!(key, T) as *const c_void,
                gcu_device_ptr!(value, T) as *const c_void,
                gcu_device_ptr!(cu_seqlens_q, u32) as *const c_void,
                gcu_device_ptr!(cu_seqlens_k, u32) as *const c_void,
                ptr::null(),
                block_table
                    .as_ref()
                    .map_or(ptr::null(), |t| gcu_device_ptr!(t, u32) as *const c_void),
                alibi_slopes
                    .as_ref()
                    .map_or(ptr::null(), |t| gcu_device_ptr!(t, T) as *const c_void),
                total_q as i32,
                total_k as i32,
                batch as i32,
                num_heads as i32,
                num_heads_k as i32,
                head_size as i32,
                max_seqlen_q as i32,
                max_seqlen_k as i32,
                softmax_scale,
                softcap.unwrap_or(0.0),
                causal as i32,
                window_left,
                window_right,
                has_bt,
                bt_batch,
                bt_max_blocks,
                0,
                0,
                code,
                stream as *const c_void,
            )
        };
        if ret != 0 {
            candle_core::bail!("topsfa_flash_attn_varlen_fwd failed with code {ret}");
        }

        let output = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
        Tensor::from_storage(candle::Storage::Gcu(output), query.shape())
    }
}

pub fn flash_attn_varlen(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    block_table: &Option<Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    alibi_slopes: &Option<Tensor>,
    window_size_left: Option<i32>,
    window_size_right: Option<i32>,
    causal: bool,
) -> Result<Tensor> {
    use half::{bf16, f16};
    match key.dtype() {
        DType::F16 => flash_attn_varlen_func::<f16>(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            softcap,
            alibi_slopes,
            window_size_left,
            window_size_right,
            causal,
        ),
        DType::BF16 => flash_attn_varlen_func::<bf16>(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            softcap,
            alibi_slopes,
            window_size_left,
            window_size_right,
            causal,
        ),
        dt => {
            candle::bail!("flash_attn_varlen is only supported for f16 and bf16 ({dt:?})")
        }
    }
}

/// Paged-KV varlen flash attention.
fn flash_attn_varlen_paged_func<
    T: candle::gcu_backend::GcuDType + candle::gcu_backend::DeviceCopy + candle::WithDType,
>(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    cu_seqlens_q: &Tensor,
    seqused_k: &Tensor,
    block_table: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    window_size_left: Option<i32>,
    window_size_right: Option<i32>,
    causal: bool,
) -> Result<Tensor> {
    #[cfg(not(feature = "aten"))]
    {
        let _ = (
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            softcap,
            window_size_left,
            window_size_right,
            causal,
        );
        return require_aten_fa().map(|_| unreachable!());
    }
    #[cfg(feature = "aten")]
    {
        use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
        use candle_core::gcu_backend::WrapErr;

        let query = maybe_contiguous(query)?;
        let key_cache = maybe_contiguous(key_cache)?;
        let value_cache = maybe_contiguous(value_cache)?;
        let cu_seqlens_q = maybe_contiguous(cu_seqlens_q)?;
        let seqused_k = maybe_contiguous(seqused_k)?;
        let block_table = maybe_contiguous(block_table)?;

        let total_q = query.dim(0)?;
        let num_heads = query.dim(1)?;
        let head_size = query.dim(2)?;
        let num_heads_k = key_cache.dim(2)?;
        let num_blocks = key_cache.dim(0)?;
        let page_block_size = key_cache.dim(1)?;
        let batch = cu_seqlens_q.dim(0)? - 1;

        let window_left = window_size_left.unwrap_or(-1);
        let window_right = window_size_right.unwrap_or(-1);

        let dev = query.device().as_gcu_device()?;
        let output = dev.alloc::<T>(query.elem_count()).w()?;
        let softmax_lse = get_softmax_lse_ptr(dev, flash_lse_elems(total_q, 1, num_heads))?;
        let stream = dev.stream_inner().expect("Unable to obtain stream");
        let code = dtype_code(query.dtype())?;

        let ret = unsafe {
            candle::gcu_backend::ubridge::ffi::topsfa_flash_attn_varlen_fwd(
                output.device_ptr() as *mut c_void,
                softmax_lse as *mut c_void,
                gcu_device_ptr!(query, T) as *const c_void,
                gcu_device_ptr!(key_cache, T) as *const c_void,
                gcu_device_ptr!(value_cache, T) as *const c_void,
                gcu_device_ptr!(cu_seqlens_q, u32) as *const c_void,
                ptr::null(),
                gcu_device_ptr!(seqused_k, u32) as *const c_void,
                gcu_device_ptr!(block_table, u32) as *const c_void,
                ptr::null(),
                total_q as i32,
                0,
                batch as i32,
                num_heads as i32,
                num_heads_k as i32,
                head_size as i32,
                max_seqlen_q as i32,
                max_seqlen_k as i32,
                softmax_scale,
                softcap.unwrap_or(0.0),
                causal as i32,
                window_left,
                window_right,
                1,
                block_table.dim(0)? as i32,
                block_table.dim(1)? as i32,
                num_blocks as i32,
                page_block_size as i32,
                code,
                stream as *const c_void,
            )
        };
        if ret != 0 {
            candle_core::bail!("topsfa_flash_attn_varlen_fwd (paged) failed with code {ret}");
        }

        let output = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
        Tensor::from_storage(candle::Storage::Gcu(output), query.shape())
    }
}

/// varlen + paged KV cache + seqused_k + block_table.
pub fn flash_attn_varlen_paged(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    cu_seqlens_q: &Tensor,
    seqused_k: &Tensor,
    block_table: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    window_size_left: Option<i32>,
    window_size_right: Option<i32>,
    causal: bool,
) -> Result<Tensor> {
    use half::{bf16, f16};
    match query.dtype() {
        DType::F16 => flash_attn_varlen_paged_func::<f16>(
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            softcap,
            window_size_left,
            window_size_right,
            causal,
        ),
        DType::BF16 => flash_attn_varlen_paged_func::<bf16>(
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            softcap,
            window_size_left,
            window_size_right,
            causal,
        ),
        dt => candle::bail!("flash_attn_varlen_paged is only supported for f16 and bf16 ({dt:?})"),
    }
}

fn flash_attn_kvcache_func<
    T: candle::gcu_backend::GcuDType + candle::gcu_backend::DeviceCopy + candle::WithDType,
>(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    context_lens: &Tensor,
    block_table: &Option<Tensor>,
    softmax_scale: f32,
    softcap: Option<f32>,
    alibi_slopes: &Option<Tensor>,
    window_size_left: Option<i32>,
    window_size_right: Option<i32>,
    causal: bool,
) -> Result<Tensor> {
    #[cfg(not(feature = "aten"))]
    {
        let _ = (
            query,
            key_cache,
            value_cache,
            context_lens,
            block_table,
            softmax_scale,
            softcap,
            alibi_slopes,
            window_size_left,
            window_size_right,
            causal,
        );
        return require_aten_fa().map(|_| unreachable!());
    }
    #[cfg(feature = "aten")]
    {
        use candle::gcu_backend::ubridge::device_ptr::DevicePtr;
        use candle_core::gcu_backend::WrapErr;
        assert!(query.dim(0)? == context_lens.dim(0)?, "shape mismatch");

        let batch = query.dim(0)?;
        let seqlen_q = query.dim(1)?;
        let q_num_heads = query.dim(2)?;
        let head_size = query.dim(3)?;
        let kv_num_heads = key_cache.dim(2)?;
        let max_kernel_batches = (2 * 12 + kv_num_heads - 1) / kv_num_heads;
        let alloc_batches = batch.max(max_kernel_batches);

        let window_left = window_size_left.unwrap_or(-1);
        let window_right = window_size_right.unwrap_or(-1);

        let query = maybe_contiguous(query)?;
        let context_lens = maybe_contiguous(context_lens)?;
        let block_table = match block_table {
            Some(bt) => Some(maybe_contiguous(bt)?),
            None => None,
        };

        let dev = query.device().as_gcu_device()?;
        let softmax_lse =
            get_softmax_lse_ptr(dev, flash_lse_elems(alloc_batches, seqlen_q, q_num_heads))?;
        let stream = dev.stream_inner().expect("Unable to obtain stream");
        let code = dtype_code(query.dtype())?;

        let (num_blocks, page_block_size, max_num_blocks_per_seq, has_block_table) =
            if let Some(bt) = &block_table {
                (
                    key_cache.dim(0)? as i32,
                    key_cache.dim(1)? as i32,
                    bt.dim(1)? as i32,
                    1i32,
                )
            } else {
                (0, key_cache.dim(1)? as i32, 0, 0)
            };
        let seqlen_k = if has_block_table != 0 {
            max_num_blocks_per_seq * page_block_size
        } else {
            key_cache.dim(1)? as i32
        };

        let output = dev
            .alloc::<T>(alloc_batches * seqlen_q * q_num_heads * head_size)
            .w()?;

        let ret = unsafe {
            candle::gcu_backend::ubridge::ffi::topsfa_flash_attn_fwd_kvcache(
                output.device_ptr() as *mut c_void,
                softmax_lse as *mut c_void,
                gcu_device_ptr!(query, T) as *const c_void,
                gcu_device_ptr!(key_cache, T) as *const c_void,
                gcu_device_ptr!(value_cache, T) as *const c_void,
                gcu_device_ptr!(context_lens, u32) as *const c_void,
                block_table
                    .as_ref()
                    .map_or(ptr::null(), |t| gcu_device_ptr!(t, u32) as *const c_void),
                alibi_slopes
                    .as_ref()
                    .map_or(ptr::null(), |t| gcu_device_ptr!(t, T) as *const c_void),
                batch as i32,
                seqlen_q as i32,
                q_num_heads as i32,
                kv_num_heads as i32,
                head_size as i32,
                seqlen_k,
                page_block_size,
                num_blocks,
                max_num_blocks_per_seq,
                softmax_scale,
                softcap.unwrap_or(0.0),
                causal as i32,
                window_left,
                window_right,
                0,
                0,
                has_block_table,
                code,
                stream as *const c_void,
            )
        };
        if ret != 0 {
            candle_core::bail!("topsfa_flash_attn_fwd_kvcache failed with code {ret}");
        }

        let output = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
        Tensor::from_storage(candle::Storage::Gcu(output), query.shape())
    }
}

pub fn flash_attn_with_kvcache(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    cu_seqlens_k: &Tensor,
    block_table: &Option<Tensor>,
    softmax_scale: f32,
    softcap: Option<f32>,
    alibi_slopes: &Option<Tensor>,
    window_size_left: Option<i32>,
    window_size_right: Option<i32>,
    causal: bool,
) -> Result<Tensor> {
    use half::{bf16, f16};
    match query.dtype() {
        DType::F16 => flash_attn_kvcache_func::<f16>(
            query,
            key_cache,
            value_cache,
            cu_seqlens_k,
            block_table,
            softmax_scale,
            softcap,
            alibi_slopes,
            window_size_left,
            window_size_right,
            causal,
        ),
        DType::BF16 => flash_attn_kvcache_func::<bf16>(
            query,
            key_cache,
            value_cache,
            cu_seqlens_k,
            block_table,
            softmax_scale,
            softcap,
            alibi_slopes,
            window_size_left,
            window_size_right,
            causal,
        ),
        dt => candle::bail!("flash_attn_with_kvcache is only supported for f16 and bf16 ({dt:?})"),
    }
}
