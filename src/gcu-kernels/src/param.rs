use candle_core::{DType, Device, Result, Tensor};
use std::os::raw::c_void;
use std::ptr;

#[repr(C)]
pub struct dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(i32)]
#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum topsopDataType {
    TOPSOP_DATA_NONE = -1, // -1

    TOPSOP_DATA_I8 = 0, // 0
    TOPSOP_DATA_U8,     // 1
    TOPSOP_DATA_I16,    // 2
    TOPSOP_DATA_U16,    // 3
    TOPSOP_DATA_FP16,   // 4
    TOPSOP_DATA_BF16,   // 5
    TOPSOP_DATA_I32,    // 6
    TOPSOP_DATA_U32,    // 7
    TOPSOP_DATA_FP32,   // 8
    TOPSOP_DATA_EF32,   // 9
    TOPSOP_DATA_TF32,   // 10
    TOPSOP_DATA_I64,    // 11
    TOPSOP_DATA_U64,    // 12
    TOPSOP_DATA_F64,    // 13
    TOPSOP_DATA_PRED,   // 14
    TOPSOP_DATA_I4,     // 15

    TOPSOP_DATA_CI8,   // 16
    TOPSOP_DATA_CU8,   // 17
    TOPSOP_DATA_CI16,  // 18
    TOPSOP_DATA_CU16,  // 19
    TOPSOP_DATA_CFP16, // 20
    TOPSOP_DATA_CBF16, // 21
    TOPSOP_DATA_CI32,  // 22
    TOPSOP_DATA_CU32,  // 23
    TOPSOP_DATA_CFP32, // 24
    TOPSOP_DATA_CEF32, // 25
    TOPSOP_DATA_CTF32, // 26
    TOPSOP_DATA_CI64,  // 27
    TOPSOP_DATA_CU64,  // 28
    TOPSOP_DATA_CF64,  // 29
    TOPSOP_DATA_CPRED, // 30
    TOPSOP_DATA_CI4,   // 31

    TOPSOP_DATA_FP8E4M3, // 32
    TOPSOP_DATA_FP8E5M2, // 33
}

#[repr(C)]
pub struct VARLEN_ATTN_FWD_OP_PARAS {
    pub data_type: i32,
    pub max_seqlen_q: i32,
    pub max_seqlen_k: i32,
    pub p_dropout: f32,
    pub softmax_scale: f32,
    pub zero_tensors: bool,
    pub is_causal: bool,
    pub window_size_en: bool,
    pub window_size_left: i32,
    pub window_size_right: i32,
    pub softcap: f32,
    pub return_softmax: bool,
    pub seqused_k_en: bool,
    pub leftpad_k_en: bool,
    pub bt_en: bool,
    pub alibi_en: bool,
    pub alibi_rank: i32,
    pub batch: i32,
    pub total_q: i32,
    pub total_k: i32,
    pub q_num_heads: i32,
    pub kv_num_heads: i32,
    pub qk_head_size: i32,
    pub v_head_size: i32,
    pub max_num_blocks_per_seq: i32,
    pub block_size: i32,
    pub num_blocks: i32,
    pub kvcache_layout: i32,
    pub q_seq_sub: i32,
    pub kv_seq_sub: i32,
    pub shared_size: i32,
    pub mode: i32,
    pub thread_group: i32,
    pub rand_seed: i32,
    pub rand_offset: i32,
    pub rng_state_ptr: *mut u64,
    pub q_heads: i32,
    pub k_heads: i32,
    pub v_heads: i32,
    pub o_heads: i32,
    pub packed_type: i32,
    pub is_varlen: bool,
    pub use_fuse: bool,
    pub q_head_dim: i32,
    pub k_head_dim: i32,
    pub v_head_dim: i32,
    pub bt_stride: i32,
    pub k_stride0: i64,
    pub v_stride0: i64,
}

#[repr(C)]
pub struct FWD_KVCACHE_ATTN_OP_PARAS {
    pub data_type: i32,
    pub softmax_scale: f32,
    pub is_causal: bool,
    pub window_size_en: bool,
    pub window_size_left: i32,
    pub window_size_right: i32,
    pub softcap: f32,
    pub num_splits: i32,
    pub is_rotary_interleaved: bool,
    pub q_rotary_mode: i32,
    pub seqlens_k_en: bool,
    pub kv_en: bool,
    pub cache_batch_idx_en: bool,
    pub rotary_en: bool,
    pub leftpad_k_en: bool,
    pub bt_en: bool,
    pub alibi_en: bool,
    pub alibi_rank: i32,
    pub seqlen_knew: i32,
    pub seqlen_ro: i32,
    pub rotary_dim: i32,
    pub page_block_size: i32,
    pub seqlen_k: i32,
    pub num_blocks: i32,
    pub max_num_blocks_per_seq: i32,
    pub batch: i32,
    pub seqlen_q: i32,
    pub q_num_heads: i32,
    pub head_size: i32,
    pub batch_size_cache: i32,
    pub kv_num_heads: i32,
    pub q_seq_sub: i32,
    pub kv_seq_sub: i32,
    pub shared_size: i32,
    pub mode: i32,
    pub thread_group: i32,
    pub workspace: *mut c_void,
    pub q_stride0: i32,
    pub o_stride0: i32,
    pub SMALL_SIZE: i32,
}

pub fn build_flash_params(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    softmax_scale: f32,
    alibi_slopes: &Option<Tensor>,
    softcap: Option<f32>,
    is_causal: bool,
) -> Result<VARLEN_ATTN_FWD_OP_PARAS> {
    // Extract dimensions
    let (batch, seqlen_q, attention_heads, head_size) = query.shape().dims4()?;
    let (_batch_k, seqlen_k, key_value_heads, head_size_k) = key.shape().dims4()?; // same shape for key/value
    let window_size_left = -1;
    let window_size_right = -1;
    // Determine whether to use fused kernel
    let use_fuse = head_size <= 192
        && head_size_k <= 128
        && is_causal
        && alibi_slopes.is_none()
        && softcap.unwrap_or(0.0) <= 0.0;
    let data_type = match query.dtype() {
        DType::F16 => topsopDataType::TOPSOP_DATA_FP16,
        DType::BF16 => topsopDataType::TOPSOP_DATA_BF16,
        _ => candle_core::bail!("Unsupport data type for flash attention!"),
    };
    let mut params = VARLEN_ATTN_FWD_OP_PARAS {
        data_type: data_type as i32, // map Tensor dtype to TOPSOP_DATA_FP16 / BF16
        max_seqlen_q: seqlen_q as i32,
        max_seqlen_k: seqlen_k as i32,
        batch: batch as i32,
        total_q: (batch * seqlen_q) as i32,
        total_k: (batch * seqlen_k) as i32,
        q_num_heads: attention_heads as i32,
        kv_num_heads: key_value_heads as i32,
        q_heads: attention_heads as i32,
        k_heads: key_value_heads as i32,
        v_heads: key_value_heads as i32,
        o_heads: attention_heads as i32,
        qk_head_size: head_size as i32,
        v_head_size: value.shape().dim(3)? as i32,
        q_head_dim: head_size as i32,
        k_head_dim: head_size as i32,
        v_head_dim: value.shape().dim(3)? as i32,
        is_causal,
        p_dropout: 0.0f32,
        softmax_scale,
        softcap: softcap.unwrap_or(0.0),
        window_size_en: !(window_size_left == -1 && window_size_right == -1),
        window_size_left,
        window_size_right,
        is_varlen: false,
        use_fuse,
        packed_type: 0,
        zero_tensors: false,
        return_softmax: false,
        seqused_k_en: false,
        leftpad_k_en: false,
        bt_en: false,
        alibi_en: alibi_slopes.is_some(),
        alibi_rank: alibi_slopes
            .as_ref()
            .map_or(0, |t| t.shape().dims().len() as i32),
        thread_group: 1, // will compute later
        shared_size: 0,  // will compute later
        rand_seed: 0,
        rand_offset: 0,
        rng_state_ptr: ptr::null_mut(),
        mode: 0,
        kvcache_layout: 0,
        q_seq_sub: if use_fuse { 32 } else { 64 },
        kv_seq_sub: if use_fuse { 384 } else { 512 },
        // not used for attention without kvcache
        block_size: 0,
        bt_stride: 0,
        k_stride0: 0,
        v_stride0: 0,
        max_num_blocks_per_seq: 0,
        num_blocks: 0,
    };

    // Compute shared_size
    let pingpong = 2;
    let bpe = 2; // assuming FP16
    params.shared_size =
        params.kv_seq_sub * (params.qk_head_size + params.v_head_size) * pingpong * 3 * bpe;

    Ok(params)
}

pub fn build_varlen_params(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    cu_seqlens_q: &Tensor,
    block_table: &Option<Tensor>,
    alibi_slopes: &Option<Tensor>,
    max_seqlen_q: i32,
    max_seqlen_k: i32,
    softmax_scale: f32,
    softcap: Option<f32>,
    window_size_left: Option<i32>,
    window_size_right: Option<i32>,
    is_causal: bool,
) -> Result<VARLEN_ATTN_FWD_OP_PARAS> {
    let batch = cu_seqlens_q.shape().dim(0)? as i32 - 1;
    let (total_q, q_num_heads, qk_head_size) = query.dims3()?;
    let mut v_head_size = value.shape().dim(2)?;
    let mut total_k = key.shape().dim(0)? as i32;
    let mut kv_num_heads = key.shape().dim(1)?;
    // Determine block table logic
    let bt_en = block_table.is_some();
    let seqused_k_en = false;
    let leftpad_k_en = false;
    let alibi_en = alibi_slopes.is_some();
    let alibi_rank = if let Some(alibi) = alibi_slopes {
        alibi.shape().dims().len() as i32
    } else {
        0
    };

    let return_softmax = false;
    let zero_tensors = false;
    // Window size logic
    let mut window_size_left = window_size_left.unwrap_or(-1);
    let mut window_size_right = window_size_right.unwrap_or(-1);
    let mut window_size_en = true;
    if window_size_left >= max_seqlen_k {
        window_size_left = -1;
    }
    if window_size_right >= max_seqlen_k {
        window_size_right = -1;
    }
    if window_size_left == -1 && window_size_right == -1 {
        window_size_en = false;
    }
    if is_causal {
        window_size_right = 0;
        if !window_size_en && window_size_left == -1 && window_size_right == 0 {
            window_size_en = false;
        }
    }

    // Softcap logic
    let (softcap_param, softmax_scale_param) = if softcap.unwrap_or(0.0) > 0.0 {
        (
            softmax_scale / softcap.unwrap_or(0.0),
            softcap.unwrap_or(0.0),
        )
    } else {
        (0.0, softmax_scale)
    };

    // Determine thread group logic (simplified, can be adjusted)
    let thread_group = 1; // default fallback

    // Shared memory size estimation
    let kv_seq_sub = 512;
    let q_seq_sub = 64;
    let bpe = 2; // fp16
    let pingpong = 2;
    let shared_size = kv_seq_sub * (qk_head_size + v_head_size) * bpe * 3 * pingpong;
    let data_type = match query.dtype() {
        DType::F16 => topsopDataType::TOPSOP_DATA_FP16,
        DType::BF16 => topsopDataType::TOPSOP_DATA_BF16,
        _ => candle_core::bail!("Unsupport data type for flash attention!"),
    };
    // println!("query: {:?}, key {:?}, value {:?}", query.shape(), key.shape(), value.shape());

    let mut num_blocks = 0;
    let mut block_size = 0;
    if bt_en {
        v_head_size = value.shape().dim(3)?;
        kv_num_heads = key.shape().dim(2)?;
        num_blocks = key.shape().dim(0)?;
        block_size = key.shape().dim(1)?;
        if let Some(bt) = block_table {
            total_k = batch * (block_size as i32) * (bt.shape().dim(1)? as i32);
        }
    }

    let mut params = VARLEN_ATTN_FWD_OP_PARAS {
        is_varlen: true,
        use_fuse: false,
        packed_type: 0,
        data_type: data_type as i32,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout: 0.0f32,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        return_softmax,
        seqused_k_en,
        leftpad_k_en,
        bt_en,
        bt_stride: block_table.as_ref().map_or(0, |bt| bt.stride()[0] as i32),
        alibi_en,
        alibi_rank,
        batch,
        total_q: total_q as i32,
        q_num_heads: q_num_heads as i32,
        q_heads: q_num_heads as i32,
        k_heads: kv_num_heads as i32,
        v_heads: kv_num_heads as i32,
        o_heads: q_num_heads as i32,
        qk_head_size: qk_head_size as i32,
        v_head_size: v_head_size as i32,
        total_k: total_k as i32,
        kv_num_heads: kv_num_heads as i32,
        max_num_blocks_per_seq: block_table
            .as_ref()
            .map_or(0, |bt| bt.shape().dim(1).unwrap() as i32),
        block_size: block_size as i32,
        num_blocks: num_blocks as i32,
        kvcache_layout: 1, //0: flash, 1: paged attn
        q_seq_sub: q_seq_sub as i32,
        kv_seq_sub: kv_seq_sub as i32,
        shared_size: shared_size as i32,
        mode: 0,
        thread_group,
        rand_seed: 0,
        rand_offset: 0,
        q_head_dim: qk_head_size as i32,
        k_head_dim: qk_head_size as i32,
        v_head_dim: v_head_size as i32,
        k_stride0: key.stride()[0] as i64,
        v_stride0: value.stride()[0] as i64,
        softmax_scale: softmax_scale_param,
        softcap: softcap_param,
        rng_state_ptr: ptr::null_mut(),
        window_size_en,
    };

    Ok(params)
}
