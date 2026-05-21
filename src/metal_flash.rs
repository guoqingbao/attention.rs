use candle_core::{self as candle, DType, Device, Result, Tensor};

fn metal_dtype(dt: DType) -> metal_kernels::PagedAttentionDType {
    match dt {
        DType::F16 => metal_kernels::PagedAttentionDType::F16,
        DType::BF16 => metal_kernels::PagedAttentionDType::BF16,
        DType::F32 => metal_kernels::PagedAttentionDType::F32,
        _ => metal_kernels::PagedAttentionDType::BF16,
    }
}

fn get_metal_device(t: &Tensor) -> Result<&candle::MetalDevice> {
    match t.device() {
        Device::Metal(d) => Ok(d),
        _ => candle::bail!("expected Metal device"),
    }
}

pub const SPLIT_K_THRESHOLD: usize = 8192;
pub const NUM_SPLITS: i32 = 8;

pub fn flash_reshape_and_cache_metal(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    slot_mapping: &Tensor,
) -> Result<()> {
    let device = get_metal_device(key)?;
    let (num_tokens, num_kv_heads, head_dim) = key.dims3()?;
    let block_size = key_cache.dim(1)?;
    let ty = metal_dtype(key.dtype());

    let (key_s, key_l) = key.storage_and_layout();
    let key_s = match &*key_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (val_s, val_l) = value.storage_and_layout();
    let val_s = match &*val_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let kc_s = match &*kc_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (vc_s, vc_l) = value_cache.storage_and_layout();
    let vc_s = match &*vc_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (sm_s, sm_l) = slot_mapping.storage_and_layout();
    let sm_s = match &*sm_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };

    let ks_storage = if let Some(t) = k_scale {
        let (s, _) = t.storage_and_layout();
        match &*s {
            candle::Storage::Metal(ms) => Some(ms.clone()),
            _ => None,
        }
    } else {
        None
    };
    let vs_storage = if let Some(t) = v_scale {
        let (s, _) = t.storage_and_layout();
        match &*s {
            candle::Storage::Metal(ms) => Some(ms.clone()),
            _ => None,
        }
    } else {
        None
    };

    let command_buffer = device.command_buffer()?;
    command_buffer.set_label("flash-reshape-cache");

    metal_kernels::flash_reshape_and_cache_metal(
        device.device(),
        &command_buffer,
        metal_kernels::Kernels::default(),
        ty,
        key_s.buffer(),
        key_l.start_offset() * key.dtype().size_in_bytes(),
        val_s.buffer(),
        val_l.start_offset() * value.dtype().size_in_bytes(),
        kc_s.buffer(),
        kc_l.start_offset() * key_cache.dtype().size_in_bytes(),
        vc_s.buffer(),
        vc_l.start_offset() * value_cache.dtype().size_in_bytes(),
        sm_s.buffer(),
        sm_l.start_offset() * slot_mapping.dtype().size_in_bytes(),
        num_tokens as i32,
        num_kv_heads as i32,
        head_dim as i32,
        block_size as i32,
        ks_storage,
        vs_storage,
    )
    .map_err(|e| candle::Error::Msg(format!("flash_reshape_and_cache_metal: {e}")))?;

    Ok(())
}

pub fn flash_prefill_metal(
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
    _max_seqlen_q: usize,
) -> Result<Tensor> {
    let device = get_metal_device(query)?;
    let q_len = query.dim(0)?;
    let block_size = key_cache.dim(1)? as i32;
    let ty = metal_dtype(query.dtype());

    let output = Tensor::zeros_like(query)?;

    let num_seqs: i32;
    let query_start_len: Tensor;
    if let Some(cu) = cu_seqlens_q {
        num_seqs = (cu.dim(0)? - 1) as i32;
        query_start_len = cu.clone();
    } else {
        num_seqs = 1;
        query_start_len = Tensor::from_vec(vec![0u32, q_len as u32], 2, query.device())?;
    }

    let block_table_stride = block_table.dim(1)? as i32;
    let total_num_blocks = key_cache.dim(0)? as i32;
    let kv_block_stride = (block_size as usize * num_kv_heads * head_dim) as i32;
    let kv_head_stride = head_dim as i32;
    let sw = sliding_window.unwrap_or(0) as i32;

    {
        let (q_s, q_l) = query.storage_and_layout();
        let q_s = match &*q_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (kc_s, kc_l) = key_cache.storage_and_layout();
        let kc_s = match &*kc_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (vc_s, vc_l) = value_cache.storage_and_layout();
        let vc_s = match &*vc_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (bt_s, bt_l) = block_table.storage_and_layout();
        let bt_s = match &*bt_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (cl_s, cl_l) = context_lens.storage_and_layout();
        let cl_s = match &*cl_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (o_s, _o_l) = output.storage_and_layout();
        let o_s = match &*o_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (qsl_s, qsl_l) = query_start_len.storage_and_layout();
        let qsl_s = match &*qsl_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };

        let ks_storage = if let Some(t) = k_scale {
            let (s, _) = t.storage_and_layout();
            match &*s {
                candle::Storage::Metal(ms) => Some(ms.clone()),
                _ => None,
            }
        } else {
            None
        };
        let vs_storage = if let Some(t) = v_scale {
            let (s, _) = t.storage_and_layout();
            match &*s {
                candle::Storage::Metal(ms) => Some(ms.clone()),
                _ => None,
            }
        } else {
            None
        };

        let command_buffer = device.command_buffer()?;
        command_buffer.set_label("flash-prefill");

        metal_kernels::flash_attention_prefill(
            device.device(),
            &command_buffer,
            metal_kernels::Kernels::default(),
            ty,
            o_s.buffer(),
            q_s.buffer(),
            q_l.start_offset() * query.dtype().size_in_bytes(),
            kc_s.buffer(),
            kc_l.start_offset() * key_cache.dtype().size_in_bytes(),
            vc_s.buffer(),
            vc_l.start_offset() * value_cache.dtype().size_in_bytes(),
            bt_s.buffer(),
            bt_l.start_offset() * block_table.dtype().size_in_bytes(),
            cl_s.buffer(),
            cl_l.start_offset() * context_lens.dtype().size_in_bytes(),
            qsl_s.buffer(),
            qsl_l.start_offset() * query_start_len.dtype().size_in_bytes(),
            None, // alibi
            ks_storage,
            vs_storage,
            None, // sinks
            num_kv_heads as i32,
            scale,
            block_table_stride,
            num_seqs,
            num_q_heads as i32,
            q_len as i32,
            head_dim as i32,
            block_size,
            softcap,
            (num_q_heads * head_dim) as i32,
            sw,
            total_num_blocks,
            kv_block_stride,
            kv_head_stride,
        )
        .map_err(|e| candle::Error::Msg(format!("flash_attention_prefill: {e}")))?;
    }

    Ok(output)
}

pub fn flash_decode_metal(
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
) -> Result<Tensor> {
    let device = get_metal_device(query)?;
    let num_seqs = query.dim(0)? as i32;
    let block_size = key_cache.dim(1)? as i32;
    let ty = metal_dtype(query.dtype());

    let (q_s, q_l) = query.storage_and_layout();
    let q_s = match &*q_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let kc_s = match &*kc_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (vc_s, vc_l) = value_cache.storage_and_layout();
    let vc_s = match &*vc_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (bt_s, bt_l) = block_tables.storage_and_layout();
    let bt_s = match &*bt_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (cl_s, cl_l) = context_lens.storage_and_layout();
    let cl_s = match &*cl_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (o_s, _o_l) = output.storage_and_layout();
    let o_s = match &*o_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };

    let ks_storage = if let Some(t) = k_scale {
        let (s, _) = t.storage_and_layout();
        match &*s {
            candle::Storage::Metal(ms) => Some(ms.clone()),
            _ => None,
        }
    } else {
        None
    };
    let vs_storage = if let Some(t) = v_scale {
        let (s, _) = t.storage_and_layout();
        match &*s {
            candle::Storage::Metal(ms) => Some(ms.clone()),
            _ => None,
        }
    } else {
        None
    };

    let max_blocks_per_seq = block_tables.dim(1)? as i32;
    let q_stride = (num_q_heads * head_dim) as i32;
    let sw = sliding_window.unwrap_or(0) as i32;

    let use_splitk = max_context_len >= SPLIT_K_THRESHOLD;

    let command_buffer = device.command_buffer()?;
    command_buffer.set_label("flash-decode");

    if use_splitk {
        let num_splits = NUM_SPLITS;
        // Allocate temp buffers via device.new_buffer (no command encoding)
        let tmp_out_elems = num_seqs as usize * num_q_heads * num_splits as usize * head_dim;
        let tmp_meta_elems = num_seqs as usize * num_q_heads * num_splits as usize;
        let tmp_out_buf = device.new_buffer(tmp_out_elems, candle::DType::F32, "splitk-out")?;
        let tmp_max_buf = device.new_buffer(tmp_meta_elems, candle::DType::F32, "splitk-max")?;
        let tmp_sum_buf = device.new_buffer(tmp_meta_elems, candle::DType::F32, "splitk-sum")?;

        metal_kernels::flash_attention_decode_splitk(
            device.device(),
            &command_buffer,
            metal_kernels::Kernels::default(),
            ty,
            o_s.buffer(),
            &tmp_out_buf,
            &tmp_max_buf,
            &tmp_sum_buf,
            q_s.buffer(),
            q_l.start_offset() * query.dtype().size_in_bytes(),
            kc_s.buffer(),
            kc_l.start_offset() * key_cache.dtype().size_in_bytes(),
            vc_s.buffer(),
            vc_l.start_offset() * value_cache.dtype().size_in_bytes(),
            bt_s.buffer(),
            bt_l.start_offset() * block_tables.dtype().size_in_bytes(),
            cl_s.buffer(),
            cl_l.start_offset() * context_lens.dtype().size_in_bytes(),
            ks_storage,
            vs_storage,
            num_q_heads as i32,
            num_kv_heads as i32,
            scale,
            softcap,
            block_size,
            num_seqs,
            head_dim as i32,
            max_blocks_per_seq,
            q_stride,
            sw,
            num_splits,
        )
        .map_err(|e| candle::Error::Msg(format!("flash_attention_decode_splitk: {e}")))?;
    } else {
        metal_kernels::flash_attention_decode(
            device.device(),
            &command_buffer,
            metal_kernels::Kernels::default(),
            ty,
            o_s.buffer(),
            q_s.buffer(),
            q_l.start_offset() * query.dtype().size_in_bytes(),
            kc_s.buffer(),
            kc_l.start_offset() * key_cache.dtype().size_in_bytes(),
            vc_s.buffer(),
            vc_l.start_offset() * value_cache.dtype().size_in_bytes(),
            bt_s.buffer(),
            bt_l.start_offset() * block_tables.dtype().size_in_bytes(),
            cl_s.buffer(),
            cl_l.start_offset() * context_lens.dtype().size_in_bytes(),
            ks_storage,
            vs_storage,
            num_q_heads as i32,
            num_kv_heads as i32,
            scale,
            softcap,
            block_size,
            max_context_len as i32,
            num_seqs,
            head_dim as i32,
            max_blocks_per_seq,
            q_stride,
            sw,
        )
        .map_err(|e| candle::Error::Msg(format!("flash_attention_decode: {e}")))?;
    }

    Ok(output.clone())
}

pub fn flash_tq_store_k8v4_metal(
    value: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let device = get_metal_device(value)?;
    let (num_tokens, num_kv_heads, head_dim) = value.dims3()?;
    let block_size = {
        let dims = v_absmax.dims();
        if dims.len() >= 2 {
            dims[1]
        } else {
            32
        }
    };
    let ty = metal_dtype(value.dtype());

    let (v_s, v_l) = value.storage_and_layout();
    let v_s = match &*v_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (va_s, _) = v_absmax.storage_and_layout();
    let va_s = match &*va_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (vq_s, _) = v_quant.storage_and_layout();
    let vq_s = match &*vq_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (sm_s, sm_l) = slot_mapping.storage_and_layout();
    let sm_s = match &*sm_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };

    let command_buffer = device.command_buffer()?;
    command_buffer.set_label("tq-store-k8v4");

    metal_kernels::flash_tq_store_k8v4(
        device.device(),
        &command_buffer,
        metal_kernels::Kernels::default(),
        ty,
        v_s.buffer(),
        v_l.start_offset() * value.dtype().size_in_bytes(),
        va_s.buffer(),
        vq_s.buffer(),
        sm_s.buffer(),
        sm_l.start_offset() * slot_mapping.dtype().size_in_bytes(),
        num_tokens as i32,
        num_kv_heads as i32,
        head_dim as i32,
        block_size as i32,
    )
    .map_err(|e| candle::Error::Msg(format!("flash_tq_store_k8v4: {e}")))?;

    Ok(())
}

pub fn flash_tq_decode_k8v4_metal(
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
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    let device = get_metal_device(query)?;
    let num_seqs = query.dim(0)? as i32;
    let block_size = key_cache.dim(1)? as i32;
    let ty = metal_dtype(query.dtype());

    let (q_s, q_l) = query.storage_and_layout();
    let q_s = match &*q_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let kc_s = match &*kc_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (va_s, _) = v_absmax.storage_and_layout();
    let va_s = match &*va_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (vq_s, _) = v_quant.storage_and_layout();
    let vq_s = match &*vq_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (bt_s, bt_l) = block_tables.storage_and_layout();
    let bt_s = match &*bt_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (cl_s, cl_l) = context_lens.storage_and_layout();
    let cl_s = match &*cl_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (o_s, _) = output.storage_and_layout();
    let o_s = match &*o_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };

    let ks_storage = if let Some(t) = k_scale {
        let (s, _) = t.storage_and_layout();
        match &*s {
            candle::Storage::Metal(ms) => Some(ms.clone()),
            _ => None,
        }
    } else {
        None
    };

    let max_blocks_per_seq = block_tables.dim(1)? as i32;
    let q_stride = (num_q_heads * head_dim) as i32;
    let sw = sliding_window.unwrap_or(0) as i32;

    let command_buffer = device.command_buffer()?;
    command_buffer.set_label("tq-decode-k8v4");

    metal_kernels::flash_tq_decode_k8v4(
        device.device(),
        &command_buffer,
        metal_kernels::Kernels::default(),
        ty,
        o_s.buffer(),
        q_s.buffer(),
        q_l.start_offset() * query.dtype().size_in_bytes(),
        kc_s.buffer(),
        kc_l.start_offset() * key_cache.dtype().size_in_bytes(),
        va_s.buffer(),
        vq_s.buffer(),
        bt_s.buffer(),
        bt_l.start_offset() * block_tables.dtype().size_in_bytes(),
        cl_s.buffer(),
        cl_l.start_offset() * context_lens.dtype().size_in_bytes(),
        ks_storage,
        num_q_heads as i32,
        num_kv_heads as i32,
        scale,
        softcap,
        block_size,
        num_seqs,
        head_dim as i32,
        max_blocks_per_seq,
        q_stride,
        sw,
    )
    .map_err(|e| candle::Error::Msg(format!("flash_tq_decode_k8v4: {e}")))?;

    Ok(output.clone())
}

pub fn flash_tq4_store_metal(
    key: &Tensor,
    value: &Tensor,
    k_absmax: &Tensor,
    k_quant: &Tensor,
    v_absmax: &Tensor,
    v_quant: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let device = get_metal_device(key)?;
    let (num_tokens, num_kv_heads, head_dim) = key.dims3()?;
    let block_size = k_absmax.dim(1)?;
    let ty = metal_dtype(key.dtype());

    let (k_s, k_l) = key.storage_and_layout();
    let k_s = match &*k_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (v_s, v_l) = value.storage_and_layout();
    let v_s = match &*v_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (ka_s, _) = k_absmax.storage_and_layout();
    let ka_s = match &*ka_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (kq_s, _) = k_quant.storage_and_layout();
    let kq_s = match &*kq_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (va_s, _) = v_absmax.storage_and_layout();
    let va_s = match &*va_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (vq_s, _) = v_quant.storage_and_layout();
    let vq_s = match &*vq_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (sm_s, sm_l) = slot_mapping.storage_and_layout();
    let sm_s = match &*sm_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };

    let command_buffer = device.command_buffer()?;
    command_buffer.set_label("tq4-store");

    metal_kernels::flash_tq4_store(
        device.device(),
        &command_buffer,
        metal_kernels::Kernels::default(),
        ty,
        k_s.buffer(),
        k_l.start_offset() * key.dtype().size_in_bytes(),
        v_s.buffer(),
        v_l.start_offset() * value.dtype().size_in_bytes(),
        ka_s.buffer(),
        kq_s.buffer(),
        va_s.buffer(),
        vq_s.buffer(),
        sm_s.buffer(),
        sm_l.start_offset() * slot_mapping.dtype().size_in_bytes(),
        num_tokens as i32,
        num_kv_heads as i32,
        head_dim as i32,
        block_size as i32,
    )
    .map_err(|e| candle::Error::Msg(format!("flash_tq4_store: {e}")))?;

    Ok(())
}

pub fn flash_tq4_decode_metal(
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
    scale: f32,
    softcap: f32,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    let device = get_metal_device(query)?;
    let num_seqs = query.dim(0)? as i32;
    let block_size = k_absmax.dim(1)? as i32;
    let ty = metal_dtype(query.dtype());

    let (q_s, q_l) = query.storage_and_layout();
    let q_s = match &*q_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (ka_s, _) = k_absmax.storage_and_layout();
    let ka_s = match &*ka_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (kq_s, _) = k_quant.storage_and_layout();
    let kq_s = match &*kq_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (va_s, _) = v_absmax.storage_and_layout();
    let va_s = match &*va_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (vq_s, _) = v_quant.storage_and_layout();
    let vq_s = match &*vq_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (bt_s, bt_l) = block_tables.storage_and_layout();
    let bt_s = match &*bt_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (cl_s, cl_l) = context_lens.storage_and_layout();
    let cl_s = match &*cl_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };
    let (o_s, _) = output.storage_and_layout();
    let o_s = match &*o_s {
        candle::Storage::Metal(s) => s.clone(),
        _ => candle::bail!("Metal"),
    };

    let max_blocks_per_seq = block_tables.dim(1)? as i32;
    let q_stride = (num_q_heads * head_dim) as i32;
    let sw = sliding_window.unwrap_or(0) as i32;

    let command_buffer = device.command_buffer()?;
    command_buffer.set_label("tq4-decode");

    metal_kernels::flash_tq4_decode(
        device.device(),
        &command_buffer,
        metal_kernels::Kernels::default(),
        ty,
        o_s.buffer(),
        q_s.buffer(),
        q_l.start_offset() * query.dtype().size_in_bytes(),
        ka_s.buffer(),
        kq_s.buffer(),
        va_s.buffer(),
        vq_s.buffer(),
        bt_s.buffer(),
        bt_l.start_offset() * block_tables.dtype().size_in_bytes(),
        cl_s.buffer(),
        cl_l.start_offset() * context_lens.dtype().size_in_bytes(),
        scale,
        softcap,
        num_q_heads as i32,
        num_kv_heads as i32,
        block_size,
        num_seqs,
        head_dim as i32,
        max_blocks_per_seq,
        q_stride,
        sw,
    )
    .map_err(|e| candle::Error::Msg(format!("flash_tq4_decode: {e}")))?;

    Ok(output.clone())
}

pub fn flash_tq4_prefill_metal(
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
    cu_seqlens_q: Option<&Tensor>,
    _max_seqlen_q: usize,
) -> Result<Tensor> {
    let device = get_metal_device(query)?;
    let q_len = query.dim(0)?;
    let block_size = k_absmax.dim(1)? as i32;
    let ty = metal_dtype(query.dtype());

    let output = Tensor::zeros_like(query)?;

    let num_seqs: i32;
    let query_start_len: Tensor;
    if let Some(cu) = cu_seqlens_q {
        num_seqs = (cu.dim(0)? - 1) as i32;
        query_start_len = cu.clone();
    } else {
        num_seqs = 1;
        query_start_len = Tensor::from_vec(vec![0u32, q_len as u32], 2, query.device())?;
    }

    let block_table_stride = block_table.dim(1)? as i32;
    let sw = sliding_window.unwrap_or(0) as i32;

    {
        let (q_s, q_l) = query.storage_and_layout();
        let q_s = match &*q_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (ka_s, _) = k_absmax.storage_and_layout();
        let ka_s = match &*ka_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (kq_s, _) = k_quant.storage_and_layout();
        let kq_s = match &*kq_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (va_s, _) = v_absmax.storage_and_layout();
        let va_s = match &*va_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (vq_s, _) = v_quant.storage_and_layout();
        let vq_s = match &*vq_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (bt_s, bt_l) = block_table.storage_and_layout();
        let bt_s = match &*bt_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (cl_s, cl_l) = context_lens.storage_and_layout();
        let cl_s = match &*cl_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (o_s, _o_l) = output.storage_and_layout();
        let o_s = match &*o_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };
        let (qsl_s, qsl_l) = query_start_len.storage_and_layout();
        let qsl_s = match &*qsl_s {
            candle::Storage::Metal(s) => s.clone(),
            _ => candle::bail!("Metal"),
        };

        let command_buffer = device.command_buffer()?;
        command_buffer.set_label("tq4-prefill");

        metal_kernels::flash_tq4_prefill(
            device.device(),
            &command_buffer,
            metal_kernels::Kernels::default(),
            ty,
            o_s.buffer(),
            q_s.buffer(),
            q_l.start_offset() * query.dtype().size_in_bytes(),
            ka_s.buffer(),
            kq_s.buffer(),
            va_s.buffer(),
            vq_s.buffer(),
            bt_s.buffer(),
            bt_l.start_offset() * block_table.dtype().size_in_bytes(),
            cl_s.buffer(),
            cl_l.start_offset() * context_lens.dtype().size_in_bytes(),
            qsl_s.buffer(),
            qsl_l.start_offset() * query_start_len.dtype().size_in_bytes(),
            num_kv_heads as i32,
            scale,
            block_table_stride,
            num_seqs,
            num_q_heads as i32,
            q_len as i32,
            head_dim as i32,
            block_size,
            softcap,
            (num_q_heads * head_dim) as i32,
            sw,
        )
        .map_err(|e| candle::Error::Msg(format!("flash_tq4_prefill: {e}")))?;
    }

    Ok(output)
}
