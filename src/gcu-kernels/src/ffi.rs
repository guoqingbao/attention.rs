use crate::param::{dim3, FWD_KVCACHE_ATTN_OP_PARAS, VARLEN_ATTN_FWD_OP_PARAS};
use candle_core::gcu_backend::ubridge::gcu_device::driv;
use core::ffi::{c_void, c_int};
#[link(name = "flashkernels")] // links with libflashkernels.so
extern "C" {
    pub fn flash_attn_fwd_host(
        numBlocks: dim3,
        numThreads: dim3,
        q: *const std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        alibi_slopes: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        logsumexp: *mut std::ffi::c_void,
        s_dmask: *const std::ffi::c_void,
        rng_state: *const std::ffi::c_void,
        para: VARLEN_ATTN_FWD_OP_PARAS,
        stream: driv::topsStream_t,
    ) -> i32;

    pub fn varlen_attn_fwd_host(
        numBlocks: dim3,
        numDimBlocks: dim3,
        output: *mut std::ffi::c_void,
        softmax_lse: *mut std::ffi::c_void,
        p: *const std::ffi::c_void,
        query: *const std::ffi::c_void,
        key: *const std::ffi::c_void,
        value: *const std::ffi::c_void,
        cu_seqlens_q: *const std::ffi::c_void,
        cu_seqlens_k: *const std::ffi::c_void,
        seqused_k: *const std::ffi::c_void,
        leftpad_k: *const std::ffi::c_void,
        block_table: *const std::ffi::c_void,
        alibi_slopes: *const std::ffi::c_void,
        params: VARLEN_ATTN_FWD_OP_PARAS,
        stream: driv::topsStream_t,
    ) -> i32;

    pub fn varlen_kvpacked_attn_fwd_host(
        numBlocks: dim3,
        dimBlocks: dim3,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        p: *const c_void,
        rng_state: *const c_void,
        q: *const c_void,
        kv: *const c_void,
        cu_seqlens_q: *const c_void,
        cu_seqlens_k: *const c_void,
        alibi_slopes: *const c_void,
        params: VARLEN_ATTN_FWD_OP_PARAS,
        stream: driv::topsStream_t,
    ) -> i32;

    pub fn varlen_qkvpacked_attn_fwd_host(
        numBlocks: dim3,
        dimBlocks: dim3,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        p: *const c_void,
        qkv: *const c_void,
        cu_seqlens: *const c_void,
        alibi_slopes: *const c_void,
        params: VARLEN_ATTN_FWD_OP_PARAS,
        stream: driv::topsStream_t,
    ) -> i32;

    pub fn fwd_kvcache_attn_host(
        dimBlocks: dim3,
        dimThreads: dim3,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        query: *const c_void,
        kcache: *const c_void,
        vcache: *const c_void,
        key: *const c_void,
        value: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        seqlens_k: *const c_void,
        cache_batch_idx: *const c_void,
        leftpad_k: *const c_void,
        block_table: *const c_void,
        alibi_slopes: *const c_void,
        params: FWD_KVCACHE_ATTN_OP_PARAS,
        stream: driv::topsStream_t,
    ) -> i32;

    pub fn reshape_and_cache_flash_host(
        dimBlocks: dim3,
        dimThreads: dim3,
        key: *const c_void,          // [num_tokens, num_heads, head_size]
        value: *const c_void,        // [num_tokens, num_heads, head_size]
        slot_mapping: *const c_void, // [num_tokens]
        key_cache: *const c_void,    // [num_blocks, block_size, num_heads, head_size]
        value_cache: *const c_void,  // [num_blocks, block_size, num_heads, head_size]
        dataType: i32,
        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        num_blocks: c_int,
        block_size: c_int,
        key_stride: c_int,
        value_stride: c_int,
        block_stride: c_int,
        page_stride: c_int,
        head_stride: c_int,
        stream: driv::topsStream_t,
    );
}
