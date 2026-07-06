use crate::param::{dim3, FWD_KVCACHE_ATTN_OP_PARAS, VARLEN_ATTN_FWD_OP_PARAS};
use candle_core::gcu_backend::ubridge::gcu_device::driv;
use core::ffi::{c_int, c_void};
#[cfg(feature = "flashattn")]
extern "C" {
    #[link(name = "flashkernels")]
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

    #[link(name = "flashkernels")]
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

    #[link(name = "flashkernels")]
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

    #[link(name = "flashkernels")]
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

    #[link(name = "flashkernels")]
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
}
