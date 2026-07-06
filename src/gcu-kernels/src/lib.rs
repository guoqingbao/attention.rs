pub mod ffi;
pub mod param;
use candle_core as candle;
use candle_core::{DType, Result, Storage, Tensor};
#[cfg(feature = "flashattn")]
use ffi::{flash_attn_fwd_host, fwd_kvcache_attn_host, varlen_attn_fwd_host};
use param::{build_flash_params, build_kvcache_params, build_varlen_params, dim3};
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

    #[cfg(feature = "flashattn")]
    {
        let params = build_flash_params(
            query,
            key,
            value,
            softmax_scale,
            alibi_slopes,
            softcap,
            is_causal,
        )?;
        let logsumexp = dev.alloc::<f32>(batch * attention_heads * seqlen_q).w()?;

        unsafe {
            flash_attn_fwd_host(
                dim3 { x: 2, y: 1, z: 1 },
                dim3 { x: 12, y: 1, z: 1 },
                query_ptr as *const c_void,
                key_ptr as *const c_void,
                value_ptr as *const c_void,
                alibi_slopes
                    .as_ref()
                    .map_or(ptr::null_mut(), |t| gcu_device_ptr!(t, T) as *const c_void),
                output.device_ptr() as *mut c_void,
                logsumexp.device_ptr() as *mut c_void,
                ptr::null(),
                ptr::null(),
                params,
                stream,
            );
        }
        let output = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
        return Tensor::from_storage(candle::Storage::Gcu(output), query.shape());
    }

    #[cfg(feature = "aten")]
    {
        let dtype_code: i32 = match T::DTYPE {
            DType::F16 => 4,
            DType::BF16 => 5,
            _ => -1,
        };
        if dtype_code > 0 {
            let softmax_lse = dev.alloc::<f32>(batch * attention_heads * seqlen_q).w()?;
            let ret = unsafe {
                candle::gcu_backend::ubridge::ffi::topsfa_flash_attn_fwd(
                    output.device_ptr() as *mut c_void,
                    softmax_lse.device_ptr() as *mut c_void,
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
                    if is_causal { 1 } else { 0 },
                    -1,
                    0,
                    dtype_code,
                    stream as *const c_void,
                )
            };
            if ret != 0 {
                candle::bail!("topsfa_flash_attn_fwd failed with code {ret}");
            }
            let output = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
            return Tensor::from_storage(candle::Storage::Gcu(output), query.shape());
        }
    }

    candle_core::bail!("Flash attention is not available in this build!")
}

pub fn flash_attn(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    softmax_scale: f32,
    alibi_slopes: &Option<Tensor>,
    softcap: Option<f32>,
    causal: bool,
) -> Result<Tensor> {
    use half::{bf16, f16};
    match key.dtype() {
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
    use candle_core::gcu_backend::ubridge::device_ptr::DevicePtr;
    use candle_core::gcu_backend::WrapErr;

    let dev = query.device().as_gcu_device()?;
    let query_ptr = gcu_device_ptr!(query, T);
    let key_ptr = gcu_device_ptr!(key, T);
    let value_ptr = gcu_device_ptr!(value, T);

    let total_q = query.shape().dim(0)?;
    let num_heads = query.shape().dim(1)?;
    let head_size = query.shape().dim(2)?;
    let total_k = key.shape().dim(0)?;
    let num_heads_k = key.shape().dim(1)?;
    let batch = cu_seqlens_q.dim(0)? - 1;

    let output = dev.alloc::<T>(query.elem_count()).w()?;
    let stream = dev.stream_inner().expect("Unable to obtain stream");
    #[cfg(feature = "flashattn")]
    {
        let params = build_varlen_params(
            query,
            key,
            value,
            cu_seqlens_q,
            block_table,
            alibi_slopes,
            max_seqlen_q as i32,
            max_seqlen_k as i32,
            softmax_scale,
            softcap,
            window_size_left,
            window_size_right,
            causal,
        )?;
        let softmax_lse = dev.alloc::<f32>(total_q * num_heads).w()?;
        unsafe {
            varlen_attn_fwd_host(
                dim3 { x: 2, y: 1, z: 1 },
                dim3 { x: 12, y: 1, z: 1 },
                output.device_ptr() as *mut c_void,
                softmax_lse.device_ptr() as *mut c_void,
                ptr::null(),
                query_ptr as *const c_void,
                key_ptr as *const c_void,
                value_ptr as *const c_void,
                gcu_device_ptr!(cu_seqlens_q, u32) as *const c_void,
                gcu_device_ptr!(cu_seqlens_k, u32) as *const c_void,
                ptr::null(),
                ptr::null(),
                block_table
                    .as_ref()
                    .map_or(ptr::null(), |t| gcu_device_ptr!(t, u32) as *const c_void),
                alibi_slopes
                    .as_ref()
                    .map_or(ptr::null(), |t| gcu_device_ptr!(t, T) as *const c_void),
                params,
                stream,
            );
        }

        let output = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
        return Tensor::from_storage(candle::Storage::Gcu(output), query.shape());
    }

    #[cfg(feature = "aten")]
    {
        let dtype_code: i32 = match T::DTYPE {
            DType::F16 => 4,
            DType::BF16 => 5,
            _ => -1,
        };
        if dtype_code > 0 {
            let softmax_lse = dev.alloc::<f32>(total_q * num_heads).w()?;

            let (has_bt, bt_batch, bt_max_blocks) = if let Some(bt) = block_table {
                let dims = bt.dims();
                (1i32, dims[0] as i32, dims[1] as i32)
            } else {
                (0i32, 0i32, 0i32)
            };

            let ret = unsafe {
                candle::gcu_backend::ubridge::ffi::topsfa_flash_attn_varlen_fwd(
                    output.device_ptr() as *mut c_void,
                    softmax_lse.device_ptr() as *mut c_void,
                    query_ptr as *const c_void,
                    key_ptr as *const c_void,
                    value_ptr as *const c_void,
                    gcu_device_ptr!(cu_seqlens_q, u32) as *const c_void,
                    gcu_device_ptr!(cu_seqlens_k, u32) as *const c_void,
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
                    if causal { 1 } else { 0 },
                    window_size_left.unwrap_or(-1),
                    window_size_right.unwrap_or(0),
                    has_bt,
                    bt_batch,
                    bt_max_blocks,
                    dtype_code,
                    stream as *const c_void,
                )
            };
            if ret != 0 {
                candle::bail!("topsfa_flash_attn_varlen_fwd failed with code {ret}");
            }
            let output = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
            return Tensor::from_storage(candle::Storage::Gcu(output), query.shape());
        }
    }

    candle_core::bail!("Flash attention is not available in this build!")
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
    use candle_core::gcu_backend::ubridge::device_ptr::DevicePtr;
    use candle_core::gcu_backend::WrapErr;
    assert!(query.dim(0)? == context_lens.dim(0)?, "shape mismatch");

    let dev = query.device().as_gcu_device()?;
    let query_ptr = gcu_device_ptr!(query, T);
    let key_cache_ptr = gcu_device_ptr!(key_cache, T);
    let value_cache_ptr = gcu_device_ptr!(value_cache, T);

    let batch = query.dim(0)?;
    let seqlen_q = query.dim(1)?;
    let q_num_heads = query.dim(2)?;
    let head_size = query.dim(3)?;
    let kv_num_heads = key_cache.dim(2)?;
    let stream = dev.stream_inner().expect("Unable to obtain stream");
    #[cfg(feature = "flashattn")]
    {
        let params = build_kvcache_params(
            query,
            key_cache,
            value_cache,
            &None,
            &None,
            &Some(context_lens.to_owned()),
            &None,
            &None,
            &None,
            &None,
            block_table,
            alibi_slopes,
            softmax_scale,
            softcap,
            causal,
            window_size_left,
            window_size_right,
            true,
            0,
            dim3 { x: 2, y: 1, z: 1 },
            dim3 { x: 12, y: 1, z: 1 },
        )?;

        let max_kernel_batches = (2 * 12 + kv_num_heads - 1) / kv_num_heads;
        let alloc_batches = batch.max(max_kernel_batches);

        let output = dev
            .alloc::<T>(alloc_batches * seqlen_q * q_num_heads * head_size)
            .w()?;
        let softmax_lse = dev
            .alloc::<f32>(alloc_batches * seqlen_q * q_num_heads)
            .w()?;
        unsafe {
            fwd_kvcache_attn_host(
                dim3 { x: 2, y: 1, z: 1 },
                dim3 { x: 12, y: 1, z: 1 },
                output.device_ptr() as *mut c_void,
                softmax_lse.device_ptr() as *mut c_void,
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                gcu_device_ptr!(context_lens, u32) as *const c_void,
                ptr::null(),
                ptr::null(),
                block_table
                    .as_ref()
                    .map_or(ptr::null(), |t| gcu_device_ptr!(t, u32) as *const c_void),
                alibi_slopes
                    .as_ref()
                    .map_or(ptr::null(), |t| gcu_device_ptr!(t, T) as *const c_void),
                params,
                stream,
            );
        }

        let output = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
        return Tensor::from_storage(candle::Storage::Gcu(output), query.shape());
    }

    #[cfg(feature = "aten")]
    {
        let dtype_code: i32 = match T::DTYPE {
            DType::F16 => 4,
            DType::BF16 => 5,
            _ => -1,
        };
        if dtype_code > 0 {
            let seqlen_k = key_cache.dim(1)?;
            let (has_bt, num_blocks, page_block_size, max_num_blocks_per_seq) =
                if let Some(bt) = block_table {
                    let bt_dims = bt.dims();
                    (
                        1i32,
                        key_cache.dim(0)? as i32,
                        seqlen_k as i32,
                        bt_dims[1] as i32,
                    )
                } else {
                    (0i32, 0i32, 0i32, 0i32)
                };

            let output = dev
                .alloc::<T>(batch * seqlen_q * q_num_heads * head_size)
                .w()?;
            let softmax_lse = dev.alloc::<f32>(batch * q_num_heads * seqlen_q).w()?;

            let ret = unsafe {
                candle::gcu_backend::ubridge::ffi::topsfa_flash_attn_fwd_kvcache(
                    output.device_ptr() as *mut c_void,
                    softmax_lse.device_ptr() as *mut c_void,
                    query_ptr as *const c_void,
                    key_cache_ptr as *const c_void,
                    value_cache_ptr as *const c_void,
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
                    seqlen_k as i32,
                    page_block_size,
                    num_blocks,
                    max_num_blocks_per_seq,
                    softmax_scale,
                    softcap.unwrap_or(0.0),
                    if causal { 1 } else { 0 },
                    window_size_left.unwrap_or(-1),
                    window_size_right.unwrap_or(0),
                    1,
                    0,
                    has_bt,
                    dtype_code,
                    stream as *const c_void,
                )
            };
            if ret != 0 {
                candle::bail!("topsfa_flash_attn_fwd_kvcache failed with code {ret}");
            }
            let output = candle::GcuStorage::wrap_gcu_slice(output, dev.clone());
            return Tensor::from_storage(candle::Storage::Gcu(output), query.shape());
        }
    }

    candle_core::bail!("Flash attention is not available in this build!")
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
        dt => {
            candle::bail!("flash_attn_kvcache is only supported for f16 and bf16 ({dt:?})")
        }
    }
}
