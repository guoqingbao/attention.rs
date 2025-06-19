use core::ffi::{c_int, c_long, c_void};
#[allow(dead_code)]
extern "C" {
    pub fn call_reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        slot_mapping: *const c_long,

        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        x: c_int,
        key_stride: c_int,
        value_stride: c_int,
        dtype: u32,
        stream: i64,
    );

    pub fn paged_attention_v1(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,

        dtype: u32,
        softscapping: f32,
        stream: i64,
    );

    pub fn paged_attention_v2(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,

        dtype: u32,
        softscapping: f32,
        stream: i64,
    );

    pub fn asort_asc_f32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_f16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_bf16(
        x: *const c_void,
        dst: *const c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_f64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_u8(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_u32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_i64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_f32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_f16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_bf16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_f64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_u8(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_u32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_i64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
}
