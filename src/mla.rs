#[cfg(feature = "cuda")]
use crate::kernels;
#[cfg(all(feature = "cuda", feature = "flashinfer"))]
use crate::workspace::{get_or_init_workspace, WORKSPACE_FLOAT_SIZE};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
#[cfg(feature = "cuda")]
use candle_core::DType;
#[cfg(feature = "cuda")]
use candle_core::Storage;
use candle_core::{Result, Tensor};

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
const WORKSPACE_INT_SIZE: usize = 128 * 1024 * 1024;

#[cfg(feature = "cuda")]
fn get_cuda_ptr(t: &Tensor) -> Result<*const core::ffi::c_void> {
    let (s, l) = t.storage_and_layout();
    match (&*s, t.dtype()) {
        (Storage::Cuda(c), DType::U8) => Ok(*c
            .as_cuda_slice::<u8>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
        (Storage::Cuda(c), DType::BF16) => Ok(*c
            .as_cuda_slice::<half::bf16>()?
            .slice(l.start_offset()..)
            .device_ptr()
            as *const core::ffi::c_void),
        (Storage::Cuda(c), DType::F16) => Ok(*c
            .as_cuda_slice::<half::f16>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
        (Storage::Cuda(c), DType::F32) => Ok(*c
            .as_cuda_slice::<f32>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
        (Storage::Cuda(c), DType::U32) => Ok(*c
            .as_cuda_slice::<u32>()?
            .slice(l.start_offset()..)
            .device_ptr() as *const core::ffi::c_void),
        _ => candle_core::bail!(
            "get_cuda_ptr: unsupported dtype {:?} on {:?}",
            t.dtype(),
            t.device()
        ),
    }
}

#[cfg(feature = "cuda")]
fn dtype_to_u32(dtype: DType) -> u32 {
    match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        _ => 0,
    }
}

#[cfg(all(feature = "cuda", feature = "flashinfer"))]
fn mla_flashinfer_dtype(dtype: DType) -> candle_core::Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        _ => candle_core::bail!("FlashInfer MLA only supports F16 and BF16, got {:?}", dtype),
    }
}

/// Write per-token ckv and k_pe into paged MLA cache using slot_mapping.
///
/// ckv: [num_tokens, kv_lora_rank]
/// k_pe: [num_tokens, kpe_head_dim]
/// ckv_cache: [num_blocks, block_size, kv_lora_rank]
/// kpe_cache: [num_blocks, block_size, kpe_head_dim]
/// slot_mapping: [num_tokens] (i64)
pub fn concat_and_cache_mla(
    ckv: &Tensor,
    k_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        let num_tokens = ckv.dim(0)? as i32;
        let kv_lora_rank = ckv.dim(1)? as i32;
        let kpe_head_dim = k_pe.dim(1)? as i32;
        let block_size = ckv_cache.dim(1)? as i32;
        let ckv_stride = ckv.stride()[0] as i32;
        let kpe_stride = k_pe.stride()[0] as i32;
        let dtype = dtype_to_u32(ckv.dtype());

        let ckv_ptr = get_cuda_ptr(ckv)?;
        let kpe_ptr = get_cuda_ptr(k_pe)?;
        let ckv_cache_ptr = get_cuda_ptr(ckv_cache)? as *mut core::ffi::c_void;
        let kpe_cache_ptr = get_cuda_ptr(kpe_cache)? as *mut core::ffi::c_void;

        let (slot_s, slot_l) = slot_mapping.storage_and_layout();
        let slot_ptr = match &*slot_s {
            candle_core::Storage::Cuda(c) => {
                let ptr = *c
                    .as_cuda_slice::<i64>()?
                    .slice(slot_l.start_offset()..)
                    .device_ptr();
                ptr as *const i64
            }
            _ => candle_core::bail!("slot_mapping must be on CUDA"),
        };

        let dev = ckv.device().as_cuda_device()?;
        let stream = *dev.cu_stream() as i64;

        unsafe {
            kernels::ffi::concat_and_cache_mla(
                ckv_ptr,
                kpe_ptr,
                ckv_cache_ptr,
                kpe_cache_ptr,
                slot_ptr,
                num_tokens,
                kv_lora_rank,
                kpe_head_dim,
                block_size,
                ckv_stride,
                kpe_stride,
                stream,
                dtype,
            );
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    {
        let _ = (ckv, k_pe, ckv_cache, kpe_cache, slot_mapping);
        candle_core::bail!("concat_and_cache_mla not yet implemented for Metal")
    }
}

/// Partition size for split-K decode (must match CUDA kernel constant).
#[cfg(feature = "cuda")]
const MLA_PARTITION_SIZE: usize = 128;

/// Fused MLA paged attention decode (non-FlashInfer).
///
/// Uses split-K partitioned approach for long contexts to maintain stable
/// throughput regardless of context length.
///
/// q_absorbed: [num_seqs, num_heads, kv_lora_rank]
/// q_pe:       [num_seqs, num_heads, qk_rope_head_dim]
/// ckv_cache:  [num_blocks, block_size, kv_lora_rank]
/// kpe_cache:  [num_blocks, block_size, qk_rope_head_dim]
/// block_tables: [num_seqs, max_num_blocks_per_seq] (i32)
/// context_lens: [num_seqs] (i32)
/// Returns: [num_seqs, num_heads, kv_lora_rank]
#[cfg(feature = "cuda")]
pub fn mla_paged_decode(
    q_absorbed: &Tensor,
    q_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let num_seqs = q_absorbed.dim(0)?;
    let num_heads = q_absorbed.dim(1)?;
    let kv_lora_rank = q_absorbed.dim(2)?;
    let qk_rope_head_dim = q_pe.dim(2)?;
    let block_size = ckv_cache.dim(1)? as i32;
    let max_num_blocks_per_seq = block_tables.dim(1)? as i32;
    let dtype = dtype_to_u32(q_absorbed.dtype());

    let output = Tensor::zeros(
        (num_seqs, num_heads, kv_lora_rank),
        q_absorbed.dtype(),
        q_absorbed.device(),
    )?;

    let max_ctx = (max_num_blocks_per_seq as usize) * (block_size as usize);
    let max_partitions = (max_ctx + MLA_PARTITION_SIZE - 1) / MLA_PARTITION_SIZE;
    let use_partitioned = max_partitions > 1;

    let out_ptr = get_cuda_ptr(&output)? as *mut core::ffi::c_void;
    let q_abs_ptr = get_cuda_ptr(q_absorbed)?;
    let q_pe_ptr = get_cuda_ptr(q_pe)?;
    let ckv_cache_ptr = get_cuda_ptr(ckv_cache)?;
    let kpe_cache_ptr = get_cuda_ptr(kpe_cache)?;

    let bt_ptr = get_cuda_ptr(block_tables)? as *const core::ffi::c_int;
    let cl_ptr = get_cuda_ptr(context_lens)? as *const core::ffi::c_int;

    let dev = q_absorbed.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;

    if use_partitioned {
        let tmp_out = Tensor::zeros(
            &[num_seqs, num_heads, max_partitions, kv_lora_rank],
            DType::F32,
            q_absorbed.device(),
        )?;
        let tmp_max = Tensor::zeros(
            &[num_seqs, num_heads, max_partitions],
            DType::F32,
            q_absorbed.device(),
        )?;
        let tmp_sum = Tensor::zeros(
            &[num_seqs, num_heads, max_partitions],
            DType::F32,
            q_absorbed.device(),
        )?;

        let tmp_out_ptr = get_cuda_ptr(&tmp_out)? as *mut core::ffi::c_void;
        let tmp_max_ptr = get_cuda_ptr(&tmp_max)? as *mut core::ffi::c_void;
        let tmp_sum_ptr = get_cuda_ptr(&tmp_sum)? as *mut core::ffi::c_void;

        unsafe {
            kernels::ffi::mla_paged_attention_decode(
                out_ptr,
                q_abs_ptr,
                q_pe_ptr,
                ckv_cache_ptr,
                kpe_cache_ptr,
                bt_ptr,
                cl_ptr,
                scale,
                num_seqs as i32,
                num_heads as i32,
                kv_lora_rank as i32,
                qk_rope_head_dim as i32,
                block_size,
                max_num_blocks_per_seq,
                dtype,
                stream,
                tmp_out_ptr,
                tmp_max_ptr,
                tmp_sum_ptr,
                1,
            );
        }
    } else {
        unsafe {
            kernels::ffi::mla_paged_attention_decode(
                out_ptr,
                q_abs_ptr,
                q_pe_ptr,
                ckv_cache_ptr,
                kpe_cache_ptr,
                bt_ptr,
                cl_ptr,
                scale,
                num_seqs as i32,
                num_heads as i32,
                kv_lora_rank as i32,
                qk_rope_head_dim as i32,
                block_size,
                max_num_blocks_per_seq,
                dtype,
                stream,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
            );
        }
    }
    Ok(output)
}

/// Fused MLA paged attention prefill (non-FlashInfer).
///
/// q_absorbed: [total_tokens, num_heads, kv_lora_rank]
/// q_pe:       [total_tokens, num_heads, qk_rope_head_dim]
/// ckv_cache:  [num_blocks, block_size, kv_lora_rank]
/// kpe_cache:  [num_blocks, block_size, qk_rope_head_dim]
/// block_tables: [num_seqs, max_num_blocks_per_seq] (i32)
/// context_lens: [num_seqs] (i32)
/// cu_seqlens_q: [num_seqs + 1] (i32)
/// Returns: [total_tokens, num_heads, kv_lora_rank]
#[cfg(feature = "cuda")]
pub fn mla_paged_prefill(
    q_absorbed: &Tensor,
    q_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    cu_seqlens_q: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let total_tokens = q_absorbed.dim(0)?;
    let num_heads = q_absorbed.dim(1)?;
    let kv_lora_rank = q_absorbed.dim(2)?;
    let qk_rope_head_dim = q_pe.dim(2)?;
    let block_size = ckv_cache.dim(1)? as i32;
    let max_num_blocks_per_seq = block_tables.dim(1)? as i32;
    let num_seqs = context_lens.dim(0)?;
    let dtype = dtype_to_u32(q_absorbed.dtype());

    let output = Tensor::zeros(
        (total_tokens, num_heads, kv_lora_rank),
        q_absorbed.dtype(),
        q_absorbed.device(),
    )?;

    let out_ptr = get_cuda_ptr(&output)? as *mut core::ffi::c_void;
    let q_abs_ptr = get_cuda_ptr(q_absorbed)?;
    let q_pe_ptr = get_cuda_ptr(q_pe)?;
    let ckv_cache_ptr = get_cuda_ptr(ckv_cache)?;
    let kpe_cache_ptr = get_cuda_ptr(kpe_cache)?;

    let bt_ptr = get_cuda_ptr(block_tables)? as *const core::ffi::c_int;
    let cl_ptr = get_cuda_ptr(context_lens)? as *const core::ffi::c_int;
    let cu_ptr = get_cuda_ptr(cu_seqlens_q)? as *const core::ffi::c_int;

    let dev = q_absorbed.device().as_cuda_device()?;
    let stream = *dev.cu_stream() as i64;

    unsafe {
        kernels::ffi::mla_paged_attention_prefill(
            out_ptr,
            q_abs_ptr,
            q_pe_ptr,
            ckv_cache_ptr,
            kpe_cache_ptr,
            bt_ptr,
            cl_ptr,
            cu_ptr,
            scale,
            num_seqs as i32,
            num_heads as i32,
            kv_lora_rank as i32,
            qk_rope_head_dim as i32,
            block_size,
            max_num_blocks_per_seq,
            dtype,
            stream,
        );
    }
    Ok(output)
}

/// MLA decode plan: CPU-side work estimation producing DecodePlanInfo (10 i64 values).
#[cfg(feature = "flashinfer")]
pub fn mla_decode_plan(
    dev: &candle_core::Device,
    dtype: DType,
    indptr_host: &[u32],
    batch_size: usize,
    num_qo_heads: usize,
    page_size: usize,
    enable_cuda_graph: bool,
) -> Result<Vec<i64>> {
    let dev = dev.as_cuda_device()?;
    if indptr_host.len() != batch_size + 1 {
        candle_core::bail!(
            "mla_decode_plan: indptr_host length must be batch_size+1 ({}), got {}",
            batch_size + 1,
            indptr_host.len()
        );
    }

    let data_type = match dtype {
        DType::BF16 => 1u32,
        _ => 0u32,
    };

    let (ws_float_ptr, ws_int_ptr, page_locked_ptr, page_locked_size) =
        get_or_init_workspace(dev, enable_cuda_graph)?;

    let mut plan_info = [0i64; 10];
    unsafe {
        kernels::ffi::flashinfer_mla_decode_plan_wrapper(
            indptr_host.as_ptr() as *const i32,
            batch_size as i32,
            num_qo_heads as i32,
            page_size as i32,
            ws_float_ptr,
            WORKSPACE_FLOAT_SIZE as i64,
            ws_int_ptr,
            WORKSPACE_INT_SIZE as i64,
            page_locked_ptr,
            page_locked_size as i64,
            enable_cuda_graph,
            data_type,
            plan_info.as_mut_ptr(),
            *dev.cu_stream() as i64,
        );
    }
    Ok(plan_info.to_vec())
}

/// MLA decode run: execute FlashInfer MLA decode using pre-computed plan.
///
/// q_nope: [batch_size, num_qo_heads, HEAD_DIM_CKV]  (absorbed)
/// q_pe:   [batch_size, num_qo_heads, HEAD_DIM_KPE]
/// ckv_cache: [num_blocks, block_size, HEAD_DIM_CKV]
/// kpe_cache: [num_blocks, block_size, HEAD_DIM_KPE]
/// Returns: output [batch_size, num_qo_heads, HEAD_DIM_CKV]
#[cfg(feature = "flashinfer")]
pub fn mla_decode_run(
    q_nope: &Tensor,
    q_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    indptr: &Tensor,
    indices: &Tensor,
    last_len: &Tensor,
    batch_size: usize,
    num_qo_heads: usize,
    page_size: usize,
    sm_scale: f32,
    rope_scale: f32,
    rope_theta: f32,
    plan_info: &[i64],
    enable_cuda_graph: bool,
) -> Result<Tensor> {
    let dev = q_nope.device().as_cuda_device()?;
    let dtype = q_nope.dtype();
    let head_dim_ckv = q_nope.dim(2)?;

    let output = Tensor::zeros(
        (batch_size, num_qo_heads, head_dim_ckv),
        dtype,
        q_nope.device(),
    )?;

    let q_nope_ptr = get_cuda_ptr(q_nope)?;
    let q_pe_ptr = get_cuda_ptr(q_pe)?;
    let ckv_cache_ptr = get_cuda_ptr(ckv_cache)?;
    let kpe_cache_ptr = get_cuda_ptr(kpe_cache)?;
    let o_ptr = get_cuda_ptr(&output)? as *mut core::ffi::c_void;

    let indptr_ptr = get_cuda_ptr(indptr)? as *const i32;
    let indices_ptr = get_cuda_ptr(indices)? as *const i32;
    let last_len_ptr = get_cuda_ptr(last_len)? as *const i32;

    let data_type = mla_flashinfer_dtype(dtype)?;

    let (ws_float_ptr, ws_int_ptr, _page_locked_ptr, _page_locked_size) =
        get_or_init_workspace(dev, enable_cuda_graph)?;

    unsafe {
        kernels::ffi::flashinfer_mla_decode_run_wrapper(
            o_ptr,
            q_nope_ptr,
            q_pe_ptr,
            ckv_cache_ptr,
            kpe_cache_ptr,
            indptr_ptr,
            indices_ptr,
            last_len_ptr,
            batch_size as i32,
            num_qo_heads as i32,
            page_size as i32,
            sm_scale,
            rope_scale,
            rope_theta,
            ws_float_ptr,
            WORKSPACE_FLOAT_SIZE as i64,
            ws_int_ptr,
            WORKSPACE_INT_SIZE as i64,
            plan_info.as_ptr(),
            data_type,
            *dev.cu_stream() as i64,
        );
    }
    Ok(output)
}

/// MLA prefill plan: CPU-side load-balancing scheduling producing MLAPlanInfo (18 i64 values).
#[cfg(feature = "flashinfer")]
pub fn mla_prefill_plan(
    dev: &candle_core::Device,
    qo_indptr_host: &[u32],
    kv_indptr_host: &[u32],
    kv_len_arr_host: &[u32],
    batch_size: usize,
    num_heads: usize,
    head_dim_ckv: usize,
    causal: bool,
) -> Result<Vec<i64>> {
    let dev = dev.as_cuda_device()?;
    if qo_indptr_host.len() != batch_size + 1 {
        candle_core::bail!(
            "mla_prefill_plan: qo_indptr_host length must be batch_size+1 ({}), got {}",
            batch_size + 1,
            qo_indptr_host.len()
        );
    }
    if kv_indptr_host.len() != batch_size + 1 {
        candle_core::bail!(
            "mla_prefill_plan: kv_indptr_host length must be batch_size+1 ({}), got {}",
            batch_size + 1,
            kv_indptr_host.len()
        );
    }
    if kv_len_arr_host.len() != batch_size {
        candle_core::bail!(
            "mla_prefill_plan: kv_len_arr_host length must be batch_size ({}), got {}",
            batch_size,
            kv_len_arr_host.len()
        );
    }

    let (ws_float_ptr, ws_int_ptr, page_locked_ptr, page_locked_size) =
        get_or_init_workspace(dev, false)?;

    let mut plan_info = [0i64; 18];
    unsafe {
        kernels::ffi::flashinfer_mla_prefill_plan_wrapper(
            qo_indptr_host.as_ptr() as *const i32,
            kv_indptr_host.as_ptr() as *const i32,
            kv_len_arr_host.as_ptr() as *const i32,
            batch_size as i32,
            num_heads as i32,
            head_dim_ckv as i32,
            causal,
            ws_float_ptr,
            WORKSPACE_FLOAT_SIZE as i64,
            ws_int_ptr,
            WORKSPACE_INT_SIZE as i64,
            page_locked_ptr,
            page_locked_size as i64,
            plan_info.as_mut_ptr(),
            *dev.cu_stream() as i64,
        );
    }
    Ok(plan_info.to_vec())
}

/// MLA prefill run: execute fused FlashInfer MLA prefill using pre-computed plan.
///
/// q_nope: [total_tokens, num_heads, HEAD_DIM_CKV]  (absorbed)
/// q_pe:   [total_tokens, num_heads, HEAD_DIM_KPE]
/// ckv_cache: [num_blocks, block_size, HEAD_DIM_CKV]
/// kpe_cache: [num_blocks, block_size, HEAD_DIM_KPE]
/// kv_indices: [total_pages] (i32)
/// Returns: output [total_tokens, num_heads, HEAD_DIM_CKV]
#[cfg(feature = "flashinfer")]
pub fn mla_prefill_run(
    q_nope: &Tensor,
    q_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    kv_indices: &Tensor,
    num_heads: usize,
    page_size: usize,
    sm_scale: f32,
    plan_info: &[i64],
    causal: bool,
) -> Result<Tensor> {
    let dev = q_nope.device().as_cuda_device()?;
    let dtype = q_nope.dtype();
    let total_tokens = q_nope.dim(0)?;
    let head_dim_ckv = q_nope.dim(2)?;

    let output = Tensor::zeros(
        (total_tokens, num_heads, head_dim_ckv),
        dtype,
        q_nope.device(),
    )?;

    let q_nope_ptr = get_cuda_ptr(q_nope)?;
    let q_pe_ptr = get_cuda_ptr(q_pe)?;
    let ckv_cache_ptr = get_cuda_ptr(ckv_cache)?;
    let kpe_cache_ptr = get_cuda_ptr(kpe_cache)?;
    let o_ptr = get_cuda_ptr(&output)? as *mut core::ffi::c_void;
    let kv_indices_ptr = get_cuda_ptr(kv_indices)? as *const i32;

    let data_type = mla_flashinfer_dtype(dtype)?;

    let (ws_float_ptr, ws_int_ptr, _page_locked_ptr, _page_locked_size) =
        get_or_init_workspace(dev, false)?;

    unsafe {
        kernels::ffi::flashinfer_mla_prefill_run_wrapper(
            o_ptr,
            q_nope_ptr,
            q_pe_ptr,
            ckv_cache_ptr,
            kpe_cache_ptr,
            kv_indices_ptr,
            num_heads as i32,
            page_size as i32,
            sm_scale,
            ws_float_ptr,
            WORKSPACE_FLOAT_SIZE as i64,
            ws_int_ptr,
            WORKSPACE_INT_SIZE as i64,
            plan_info.as_ptr(),
            causal,
            data_type,
            *dev.cu_stream() as i64,
        );
    }
    Ok(output)
}
