use core::ffi::{c_int, c_long, c_void};
#[allow(dead_code)]
extern "C" {
    pub fn call_reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_scale: *const c_void,
        v_scale: *const c_void,
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

    pub fn call_reshape_and_cache_flash(
        key: *const c_void,         // [num_tokens, num_heads, head_size]
        value: *const c_void,       // [num_tokens, num_heads, head_size]
        key_cache: *const c_void,   // [num_blocks, block_size, num_heads, head_size]
        value_cache: *const c_void, // [num_blocks, block_size, num_heads, head_size]
        k_scale: *const c_void,
        v_scale: *const c_void,
        slot_mapping: *const c_long, // [num_tokens]

        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        key_stride: c_int,
        value_stride: c_int,
        block_stride: c_int,
        page_stride: c_int,
        head_stride: c_int,
        dtype: u32,
        stream: i64,
    );

    pub fn paged_attention_v1(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_scale: *const c_void,
        v_scale: *const c_void,
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
        sliding_window: c_int,
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
        k_scale: *const c_void,
        v_scale: *const c_void,
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
        sliding_window: c_int,
        stream: i64,
    );

    pub fn paged_attention_prefill(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_scale: *const c_void,
        v_scale: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        num_query_tokens: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        num_blocks: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,

        dtype: u32,
        softscapping: f32,

        o_stride_tokens: c_int,
        query_start_len: *const u32,
        sinks: *const f32,
        sliding_window: c_int,
        stream: i64,
    );

    // Optimized prefill kernel with shared memory tiling (for large KV cache)
    pub fn paged_attention_prefill_opt(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        k_scale: *const c_void,
        v_scale: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        num_query_tokens: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        num_blocks: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,

        dtype: u32,
        softscapping: f32,

        o_stride_tokens: c_int,
        query_start_len: *const u32,
        sinks: *const f32,
        sliding_window: c_int,
        stream: i64,
    );

    pub fn update_kv_scales_per_head_f32(
        k: *const c_void,
        v: *const c_void,
        num_tokens: c_long,
        num_heads: c_int,
        head_dim: c_int,
        k_scales: *const f32,
        v_scales: *const f32,
        stream: i64,
    );

    pub fn update_kv_scales_per_head_f16(
        k: *const c_void,
        v: *const c_void,
        num_tokens: c_long,
        num_heads: c_int,
        head_dim: c_int,
        k_scales: *const f32,
        v_scales: *const f32,
        stream: i64,
    );

    pub fn update_kv_scales_per_head_bf16(
        k: *const c_void,
        v: *const c_void,
        num_tokens: c_long,
        num_heads: c_int,
        head_dim: c_int,
        k_scales: *const f32,
        v_scales: *const f32,
        stream: i64,
    );

    pub fn marlin_4bit_f16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    pub fn marlin_4bit_bf16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    pub fn marlin_awq_4bit_f16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    pub fn marlin_awq_4bit_bf16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );
    pub fn gptq_repack(
        weight: *const c_void,
        result: *const c_void,
        m: c_int,
        n: c_int,
        stream: i64,
    );

    pub fn awq_repack(
        weight: *const c_void,
        result: *const c_void,
        k: c_int,
        n: c_int,
        bits: c_int,
        stream: i64,
    );

    pub fn gemm_half_q_half_alt(
        a: *const c_void,
        weight: *const u32,
        qzeros: *const u32,
        scales: *const c_void,
        g_idx: *const i32,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        bit: i32,
        stream: i64,
    );

    pub fn copy_blocks_bf16(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block: i32,
        stream: i64,
    );

    pub fn copy_blocks_f16(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block: i32,
        stream: i64,
    );

    pub fn copy_blocks_f32(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block: i32,
        stream: i64,
    );

    pub fn copy_blocks_u8(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block: i32,
        stream: i64,
    );

    pub fn mamba_scatter_rows_f16(
        src: *const c_void,
        dst: *mut c_void,
        slots: *const c_long,
        num_rows: i32,
        row_elems: i32,
        src_row_stride: i64,
        dst_row_stride: i64,
        stream: i64,
    );
    pub fn mamba_scatter_rows_bf16(
        src: *const c_void,
        dst: *mut c_void,
        slots: *const c_long,
        num_rows: i32,
        row_elems: i32,
        src_row_stride: i64,
        dst_row_stride: i64,
        stream: i64,
    );
    pub fn mamba_scatter_rows_f32(
        src: *const c_void,
        dst: *mut c_void,
        slots: *const c_long,
        num_rows: i32,
        row_elems: i32,
        src_row_stride: i64,
        dst_row_stride: i64,
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

    pub fn causal_mask_f32(d_out: *mut c_void, tgt_len: i32, sliding_window: i32, stream: i64);
    pub fn causal_mask_f16(d_out: *mut c_void, tgt_len: i32, sliding_window: i32, stream: i64);
    pub fn causal_mask_bf16(d_out: *mut c_void, tgt_len: i32, sliding_window: i32, stream: i64);

    // for unquntized models (without wmma)
    pub fn moe_gemm(
        input: *const c_void,   // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input)
        stream: i64,
    );

    // for unquntized models
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void,      // device pointer [size_m, size_n]
        expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
        expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        is_prefill: bool,
        stream: i64,
    );

    pub fn moe_gemm_gguf(
        input: *const f32,      // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    // Optimized kernel for small M (batch size 1-8) with input caching
    pub fn moe_gemm_gguf_small_m(
        input: *const f32,      // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    pub fn moe_gemm_gguf_prefill(
        input: *const c_void, // input [size_m, size_k]
        weights: *const u8,   // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,   //must be host ptr
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        input_dtype: i32, // 0=f16, 1=bf16 (for inputs)
        gguf_dtype: i32,  //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    // MoE GEMV for decode phase (optimized for small batch sizes M <= 8)
    pub fn moe_gemv(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void, // device pointer [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        stream: i64,
    );

    // MoE GEMV for decode phase with transposed weights [num_experts, size_k, size_n]
    pub fn moe_gemv_transposed(
        input: *const c_void,   // input [size_m or size_m / topk, size_k]
        weights: *const c_void, // weights [num_experts, size_k, size_n] - transposed layout
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input)
        stream: i64,
    );

    // MoE GEMM WMMA with FP8 weights and block-wise scales
    pub fn moe_gemm_wmma_fp8(
        input: *const c_void,      // [size_m, size_k] in half/bf16
        weights: *const u8,        // [num_experts, size_n, size_k] FP8 as uint8_t
        weight_scales: *const f32, // [num_experts, scale_n_dim, scale_k_dim]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut c_void,      // [size_m, size_n]
        expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
        expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        block_size_n: i32,
        block_size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        is_prefill: bool,
        stream: i64,
    );

    // MoE GEMV with FP8 weights and block-wise scales (for decode phase)
    pub fn moe_gemv_fp8(
        input: *const c_void,      // [size_m, size_k]
        weights: *const u8,        // [num_experts, size_n, size_k] FP8
        weight_scales: *const f32, // [num_experts, scale_n_dim, scale_k_dim]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut c_void, // [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        block_size_n: i32,
        block_size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        stream: i64,
    );

    pub fn topk_softmax(
        gating_output: *const f32,      // in： [num_tokens, num_experts]
        token_expert_indices: *mut i32, // out: [num_tokens, topk]
        topk_weights: *mut f32,         // out: [num_tokens, topk]
        topk_indices: *mut u32,         // out: [num_tokens, topk]
        num_experts: i32,
        num_tokens: i32,
        topk: i32,
        stream: i64,
    );

    pub fn sampling_f32(
        logits_d: *const f32,
        out_tokens_d: *mut i32,
        B: i32,
        V: i32,
        K: i32,
        temperature: f32,
        top_p: f32,
        seed: u64,
        token_pos: u64,
        stream: i64,
    );

    pub fn sampling_f16(
        logits_d: *const c_void,
        out_tokens_d: *mut i32,
        B: i32,
        V: i32,
        K: i32,
        temperature: f32,
        top_p: f32,
        seed: u64,
        token_pos: u64,
        stream: i64,
    );

    pub fn sampling_bf16(
        logits_d: *const c_void,
        out_tokens_d: *mut i32,
        B: i32,
        V: i32,
        K: i32,
        temperature: f32,
        top_p: f32,
        seed: u64,
        token_pos: u64,
        stream: i64,
    );

    // Fused Rotary Position Embedding (RoPE) kernels - with position selection
    // Non-interleaved versions - support GQA, fuses index_select
    pub fn fused_rope_f32(
        q: *mut f32,
        k: *mut f32,
        cos: *const f32,
        sin: *const f32,
        positions: *const i64, // Position indices [seq_len]
        q_bh: u32,
        k_bh: u32,
        seq_len: u32,
        d: u32,
        stream: i64,
    );

    pub fn fused_rope_f16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        q_bh: u32,
        k_bh: u32,
        seq_len: u32,
        d: u32,
        stream: i64,
    );

    pub fn fused_rope_bf16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        q_bh: u32,
        k_bh: u32,
        seq_len: u32,
        d: u32,
        stream: i64,
    );

    // Interleaved versions - support GQA, fuses index_select
    pub fn fused_rope_i_f32(
        q: *mut f32,
        k: *mut f32,
        cos: *const f32,
        sin: *const f32,
        positions: *const i64,
        q_bh: u32,
        k_bh: u32,
        seq_len: u32,
        d: u32,
        stream: i64,
    );

    pub fn fused_rope_i_f16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        q_bh: u32,
        k_bh: u32,
        seq_len: u32,
        d: u32,
        stream: i64,
    );

    pub fn fused_rope_i_bf16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        q_bh: u32,
        k_bh: u32,
        seq_len: u32,
        d: u32,
        stream: i64,
    );

    // Token-major variants for packed [num_tokens, num_heads, head_dim] tensors.
    pub fn fused_rope_tok_major_f32(
        q: *mut f32,
        k: *mut f32,
        cos: *const f32,
        sin: *const f32,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        d: u32,
        stream: i64,
    );

    pub fn fused_rope_tok_major_f16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        d: u32,
        stream: i64,
    );

    pub fn fused_rope_tok_major_bf16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        d: u32,
        stream: i64,
    );

    pub fn fused_rope_i_tok_major_f32(
        q: *mut f32,
        k: *mut f32,
        cos: *const f32,
        sin: *const f32,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        d: u32,
        stream: i64,
    );

    pub fn fused_rope_i_tok_major_f16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        d: u32,
        stream: i64,
    );

    pub fn fused_rope_i_tok_major_bf16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        d: u32,
        stream: i64,
    );

    pub fn fused_rope_partial_tok_major_f32(
        q: *mut f32,
        k: *mut f32,
        cos: *const f32,
        sin: *const f32,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        rotary_d: u32,
        full_d: u32,
        stream: i64,
    );

    pub fn fused_rope_partial_tok_major_f16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        rotary_d: u32,
        full_d: u32,
        stream: i64,
    );

    pub fn fused_rope_partial_tok_major_bf16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        rotary_d: u32,
        full_d: u32,
        stream: i64,
    );

    pub fn fused_rope_i_partial_tok_major_f32(
        q: *mut f32,
        k: *mut f32,
        cos: *const f32,
        sin: *const f32,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        rotary_d: u32,
        full_d: u32,
        stream: i64,
    );

    pub fn fused_rope_i_partial_tok_major_f16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        rotary_d: u32,
        full_d: u32,
        stream: i64,
    );

    pub fn fused_rope_i_partial_tok_major_bf16(
        q: *mut c_void,
        k: *mut c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const i64,
        num_tokens: u32,
        q_heads: u32,
        k_heads: u32,
        rotary_d: u32,
        full_d: u32,
        stream: i64,
    );

    pub fn fp8_matmul_f16(
        input: *const c_void,     // [M, K]
        weight: *const u8,        // [N, K]
        weight_scale: *const f32, // [N, K] (block-wise)
        output: *mut c_void,      // [M, N]
        m: c_int,
        n: c_int,
        k: c_int,
        scale_row_stride: c_int,
        block_size_y: c_int,
        block_size_x: c_int,
        stream: i64,
    );

    pub fn fp8_matmul_f16_cutlass(
        input_q: *const u8,
        input_scale: *const f32,
        weight: *const u8,
        weight_scale: *const f32,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        scale_row_stride: c_int,
        block_size_y: c_int,
        block_size_x: c_int,
        sm_version: c_int,
        workspace: *mut c_void,
        workspace_bytes: i64,
        stream: i64,
    );

    pub fn fp8_matmul_bf16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const f32,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        scale_row_stride: c_int,
        block_size_y: c_int,
        block_size_x: c_int,
        stream: i64,
    );

    pub fn fp8_matmul_bf16_cutlass(
        input_q: *const u8,
        input_scale: *const f32,
        weight: *const u8,
        weight_scale: *const f32,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        scale_row_stride: c_int,
        block_size_y: c_int,
        block_size_x: c_int,
        sm_version: c_int,
        workspace: *mut c_void,
        workspace_bytes: i64,
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_fp8_blockscale_workspace_size_bf16(m: c_int, n: c_int, k: c_int) -> usize;

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_fp8_blockscale_bf16(
        input: *const c_void,
        weight: *const c_void,
        weight_scale: *const f32,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: i64,
    ) -> c_int;

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_fp8_blockscale_workspace_size_fp8(m: c_int, n: c_int, k: c_int) -> usize;

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_fp8_blockscale_fp8(
        input: *const c_void,
        input_scale: *const f32,
        weight: *const c_void,
        weight_scale: *const f32,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: i64,
    ) -> c_int;

    pub fn moe_fp8_calculate_expert_offsets(
        expert_ids: *const i32,
        expert_counts: *mut i32,
        expert_offsets: *mut i32,
        num_experts: c_int,
        size_m: c_int,
        is_prefill: bool,
        stream: i64,
    );

    pub fn moe_fp8_shuffle_rows_u8(
        input: *const u8,
        dst2src_map: *const i32,
        output: *mut u8,
        num_src_rows: i64,
        num_dst_rows: i64,
        num_cols: i64,
        map_divisor: c_int,
        stream: i64,
    );

    pub fn moe_fp8_shuffle_rows_f32(
        input: *const f32,
        dst2src_map: *const i32,
        output: *mut f32,
        num_src_rows: i64,
        num_dst_rows: i64,
        num_cols: i64,
        map_divisor: c_int,
        stream: i64,
    );

    // Strided version for column-major scale tensors (SM100+ Blackwell)
    pub fn moe_fp8_shuffle_rows_f32_strided(
        input: *const f32,
        dst2src_map: *const i32,
        output: *mut f32,
        num_src_rows: i64,
        num_dst_rows: i64,
        num_cols: i64,
        src_row_stride: i64,
        dst_row_stride: i64,
        map_divisor: c_int,
        stream: i64,
    );

    pub fn moe_fp8_scatter_rows_f16(
        input: *const c_void,
        src2dst_map: *const i32,
        output: *mut c_void,
        num_src_rows: i64,
        num_dst_rows: i64,
        num_cols: i64,
        weights: *const f32,
        stream: i64,
    );

    pub fn moe_fp8_scatter_rows_bf16(
        input: *const c_void,
        src2dst_map: *const i32,
        output: *mut c_void,
        num_src_rows: i64,
        num_dst_rows: i64,
        num_cols: i64,
        weights: *const f32,
        stream: i64,
    );

    pub fn moe_fp8_grouped_gemm_f16(
        a: *const u8,
        b: *const u8,
        a_scales: *const f32,
        b_scales: *const f32,
        expert_offsets: *const i32,
        num_experts: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        block_size_n: c_int,
        block_size_k: c_int,
        sm_version: c_int,
        out: *mut c_void,
        stream: i64,
    );

    pub fn moe_fp8_grouped_gemm_bf16(
        a: *const u8,
        b: *const u8,
        a_scales: *const f32,
        b_scales: *const f32,
        expert_offsets: *const i32,
        num_experts: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        block_size_n: c_int,
        block_size_k: c_int,
        sm_version: c_int,
        out: *mut c_void,
        stream: i64,
    );

    pub fn fp8_quantize_per_token_group_launch(
        input: *const c_void,
        output_q: *mut c_void,
        output_s: *mut f32,
        num_groups: c_int,
        group_size: c_int,
        num_groups_per_row: c_int,
        scale_stride: c_int,
        is_input_f16: bool,
        is_column_major_stats: bool,
        stream: i64,
    );

    pub fn flashinfer_fp8_quantize_q_per_head(
        input: *const c_void,
        output_q: *mut c_void,
        output_scale: *mut f32,
        numel: i64,
        num_heads: c_int,
        head_dim: c_int,
        is_input_f16: bool,
        stream: i64,
    );

    // FlashInfer wrappers
    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_append_kv_cache(
        k_data_ptr: *const c_void,
        v_data_ptr: *const c_void,
        new_k_ptr: *const c_void,
        new_v_ptr: *const c_void,
        paged_kv_indices: *const i32,
        paged_kv_indptr: *const i32,
        paged_kv_last_len: *const i32,
        batch_indices: *const i32, // Pre-constructed in Rust
        positions: *const i32,     // Pre-constructed in Rust
        nnz: i32,                  // Total tokens to append
        batch_size: i32,
        num_heads: i32,
        head_dim: i32,
        page_size: i32,
        k_scale_ptr: *const f32,
        v_scale_ptr: *const f32,
        is_input_f16: bool,
        data_type: i32,
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_decode_plan_wrapper(
        indptr_host: *const i32,     // Host pointer for planning
        qo_indptr_host: *const i32,  // Host pointer for fp8 decode plan
        kv_len_arr_host: *const i32, // Host pointer for fp8 decode plan
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace_float: *mut c_void,
        workspace_float_size: usize,
        workspace_int: *mut c_void,
        workspace_int_size: usize,
        page_locked_int_buffer: *mut c_void,
        page_locked_int_size: usize,
        enable_cuda_graph: bool,
        data_type: i32,
        out_data_type: i32,
        plan_info_out: *mut i64, // length 10
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_decode_plan_wrapper_fp8(
        indptr_host: *mut i32,
        qo_indptr_host: *mut i32,
        kv_len_arr_host: *mut i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace_float: *mut c_void,
        workspace_float_size: usize,
        workspace_int: *mut c_void,
        workspace_int_size: usize,
        page_locked_int_buffer: *mut c_void,
        page_locked_int_size: usize,
        enable_cuda_graph: bool,
        data_type: i32,
        out_data_type: i32,
        plan_info_out: *mut i64, // length 9
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_decode_run_wrapper(
        out_ptr: *mut c_void,
        q_ptr: *const c_void,
        k_data: *const c_void,
        v_data: *const c_void,
        indices: *const i32,
        indptr: *const i32, // Device pointer for paged_kv
        last_len: *const i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        sm_scale: f32,
        k_scale_ptr: *const f32,
        v_scale_ptr: *const f32,
        workspace_float: *mut c_void,
        workspace_float_size: usize,
        workspace_int: *mut c_void,
        workspace_int_size: usize,
        plan_info_vec: *const i64, // length 10
        window_left: i32,
        logits_soft_cap: f32,
        data_type: i32,
        out_data_type: i32,
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_decode_run_wrapper_fp8(
        out_ptr: *mut c_void,
        q_ptr: *mut c_void,
        k_data: *mut c_void,
        v_data: *mut c_void,
        indices: *const i32,
        indptr: *const i32,
        last_len: *const i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        sm_scale: f32,
        k_scale_ptr: *const f32,
        v_scale_ptr: *const f32,
        workspace_float: *mut c_void,
        workspace_float_size: usize,
        workspace_int: *mut c_void,
        workspace_int_size: usize,
        plan_info_vec: *const i64, // length 9
        data_type: i32,
        out_data_type: i32,
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_prefill_plan_wrapper(
        q_cu_seqlens_host: *const i32,
        indptr_host: *const i32,
        kv_len_arr_host: *const i32,
        total_num_rows: i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        enable_cuda_graph: bool,
        window_left: i32,
        out_data_type: i32,
        workspace_float: *mut c_void,
        workspace_float_size: usize,
        workspace_int: *mut c_void,
        workspace_int_size: usize,
        page_locked_buffer: *mut c_void,
        page_locked_size: usize,
        plan_info_out: *mut i64,
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_prefill_run_wrapper(
        out_ptr: *mut c_void,
        q_ptr: *const c_void,
        q_cu_seqlens: *const i32,
        total_num_rows: i32,
        k_data: *const c_void,
        v_data: *const c_void,
        indices: *const i32,
        indptr: *const i32,
        last_len: *const i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        sm_scale: f32,
        k_scale_ptr: *const f32,
        v_scale_ptr: *const f32,
        workspace_float: *mut c_void,
        workspace_float_size: usize,
        workspace_int: *mut c_void,
        workspace_int_size: usize,
        window_left: i32,
        logits_soft_cap: f32,
        data_type: i32,
        out_data_type: i32,
        plan_info_vec: *const i64,
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_prefill_ragged_wrapper(
        out_ptr: *mut c_void,
        q_ptr: *const c_void,
        q_cu_seqlens: *const i32,       // Device pointer
        kv_cu_seqlens: *const i32,      // Device pointer
        q_cu_seqlens_host: *const i32,  // Host pointer
        kv_cu_seqlens_host: *const i32, // Host pointer
        total_num_rows: i32,            // Total query rows
        total_kv_rows: i32,             // Total kv rows
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        sm_scale: f32,
        k_scale_ptr: *const f32,
        v_scale_ptr: *const f32,
        workspace_float: *mut c_void,
        workspace_float_size: usize,
        workspace_int: *mut c_void,
        workspace_int_size: usize,
        page_locked_int_buffer: *mut c_void,
        page_locked_int_size: usize,
        enable_cuda_graph: bool,
        data_type: i32,
        out_data_type: i32,
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_prefill_wrapper_fp8(
        out_ptr: *mut c_void,
        q_ptr: *const c_void,
        q_cu_seqlens: *const i32,
        q_cu_seqlens_host: *const i32,
        kv_len_arr_host: *const i32,
        total_num_rows: i32,
        k_data: *const c_void,
        v_data: *const c_void,
        indices: *const i32,
        indptr: *const i32,
        indptr_host: *const i32,
        last_len: *const i32,
        batch_size: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        sm_scale: f32,
        k_scale_ptr: *const f32,
        v_scale_ptr: *const f32,
        workspace_float: *mut c_void,
        workspace_float_size: usize,
        workspace_int: *mut c_void,
        workspace_int_size: usize,
        page_locked_int_buffer: *mut c_void,
        page_locked_int_size: usize,
        enable_cuda_graph: bool,
        data_type: i32,
        out_data_type: i32,
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_fused_moe_bf16(
        input: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        gate_up_weights: *const c_void,
        down_weights: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_size: i32,
        intermediate_size: i32,
        num_experts: i32,
        top_k: i32,
        input_dtype: i32,
        weight_dtype: i32,
        stream: i64,
    ) -> i32;

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_fused_moe_fp8(
        input: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        gate_up_weights: *const u8,
        gate_up_scales: *const f32,
        down_weights: *const u8,
        down_scales: *const f32,
        output: *mut c_void,
        num_tokens: i32,
        hidden_size: i32,
        intermediate_size: i32,
        num_experts: i32,
        top_k: i32,
        input_dtype: i32,
        stream: i64,
    ) -> i32;

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_fused_moe_mxfp4(
        input: *const c_void,
        topk_ids: *const i32,
        topk_weights: *const f32,
        gate_up_weights: *const u8,
        gate_up_scales: *const u8,
        down_weights: *const u8,
        down_scales: *const u8,
        output: *mut c_void,
        num_tokens: i32,
        hidden_size: i32,
        intermediate_size: i32,
        num_experts: i32,
        top_k: i32,
        input_dtype: i32,
        stream: i64,
    ) -> i32;

    // NVFP4 activation quantization (SM100+ / Blackwell)
    pub fn nvfp4_quantize_activation_f16(
        input: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        swizzled_scales: *mut c_void,
        input_scale_inv: f32,
        M: i32,
        K: i32,
        M_padded: i32,
        K_scale_padded: i32,
        stream: i64,
    );

    pub fn nvfp4_quantize_activation_bf16(
        input: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        swizzled_scales: *mut c_void,
        input_scale_inv: f32,
        M: i32,
        K: i32,
        M_padded: i32,
        K_scale_padded: i32,
        stream: i64,
    );

    pub fn nvfp4_quantize_activation_grouped_f16(
        input: *const c_void,
        output: *mut c_void,
        swizzled_scales: *mut c_void,
        input_scale_invs: *const f32,
        expert_offsets: *const i32,
        sf_offsets: *const i32,
        total_rows: i32,
        num_experts: i32,
        K: i32,
        K_scale_padded: i32,
        stream: i64,
    );

    pub fn nvfp4_quantize_activation_grouped_bf16(
        input: *const c_void,
        output: *mut c_void,
        swizzled_scales: *mut c_void,
        input_scale_invs: *const f32,
        expert_offsets: *const i32,
        sf_offsets: *const i32,
        total_rows: i32,
        num_experts: i32,
        K: i32,
        K_scale_padded: i32,
        stream: i64,
    );

    pub fn nvfp4_moe_build_metadata(
        expert_offsets: *const i32,
        weight_global_scales: *const f32,
        input_scales: *const f32,
        sf_offsets: *mut i32,
        problem_sizes: *mut i32,
        alphas: *mut f32,
        input_scale_invs: *mut f32,
        num_experts: i32,
        N: i32,
        K: i32,
        stream: i64,
    );

    pub fn nvfp4_swizzle_weight_scales(
        linear_scales: *const c_void,
        swizzled_scales: *mut c_void,
        rows: i32,
        cols: i32,
        rows_padded: i32,
        cols_padded: i32,
        stream: i64,
    );

    pub fn nvfp4_moe_gather_f16(
        input: *const c_void,
        output: *mut c_void,
        sorted_token_ids: *const i32,
        total_expanded: i32,
        K: i32,
        map_divisor: i32,
        stream: i64,
    );

    pub fn nvfp4_moe_gather_bf16(
        input: *const c_void,
        output: *mut c_void,
        sorted_token_ids: *const i32,
        total_expanded: i32,
        K: i32,
        map_divisor: i32,
        stream: i64,
    );

    pub fn nvfp4_moe_scatter_f16(
        input: *const c_void,
        output: *mut c_void,
        scatter_ids: *const i32,
        total_expanded: i32,
        N: i32,
        stream: i64,
    );

    pub fn nvfp4_moe_scatter_bf16(
        input: *const c_void,
        output: *mut c_void,
        scatter_ids: *const i32,
        total_expanded: i32,
        N: i32,
        stream: i64,
    );

    // CUTLASS hardware FP4 GEMM (SM100+ / Blackwell)
    pub fn nvfp4_cutlass_gemm_f16(
        input: *const c_void,
        weight: *const c_void,
        input_sf: *const c_void,
        weight_sf: *const c_void,
        global_sf: *const f32,
        output: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        workspace: *mut c_void,
        workspace_bytes: i64,
        stream: i64,
    );

    pub fn nvfp4_cutlass_gemm_bf16(
        input: *const c_void,
        weight: *const c_void,
        input_sf: *const c_void,
        weight_sf: *const c_void,
        global_sf: *const f32,
        output: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        workspace: *mut c_void,
        workspace_bytes: i64,
        stream: i64,
    );

    pub fn nvfp4_cutlass_moe_gemm_f16(
        output: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        a_blockscale: *const c_void,
        b_blockscales: *const c_void,
        alphas: *const f32,
        expert_offsets: *const i32,
        sf_offsets: *const i32,
        problem_sizes: *const i32,
        num_experts: i32,
        total_tokens: i32,
        N: i32,
        K: i32,
        workspace: *mut c_void,
        workspace_bytes: i64,
        stream: i64,
    ) -> i32;

    pub fn nvfp4_cutlass_moe_gemm_bf16(
        output: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        a_blockscale: *const c_void,
        b_blockscales: *const c_void,
        alphas: *const f32,
        expert_offsets: *const i32,
        sf_offsets: *const i32,
        problem_sizes: *const i32,
        num_experts: i32,
        total_tokens: i32,
        N: i32,
        K: i32,
        workspace: *mut c_void,
        workspace_bytes: i64,
        stream: i64,
    ) -> i32;

    pub fn causal_conv1d_fwd_f32(
        x: *const f32,
        weight: *const f32,
        bias: *const f32,
        conv_state: *mut f32,
        out: *mut f32,
        cu_seqlens: *const u32,
        batch: c_int,
        d_conv: c_int,
        kernel_size: c_int,
        silu: bool,
        stream: i64,
    );
    pub fn causal_conv1d_fwd_f16(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        conv_state: *mut c_void,
        out: *mut c_void,
        cu_seqlens: *const u32,
        batch: c_int,
        d_conv: c_int,
        kernel_size: c_int,
        silu: bool,
        stream: i64,
    );
    pub fn causal_conv1d_fwd_bf16(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        conv_state: *mut c_void,
        out: *mut c_void,
        cu_seqlens: *const u32,
        batch: c_int,
        d_conv: c_int,
        kernel_size: c_int,
        silu: bool,
        stream: i64,
    );

    pub fn causal_conv1d_update_f32(
        x: *const f32,
        weight: *const f32,
        bias: *const f32,
        conv_state: *mut f32,
        out: *mut f32,
        batch: c_int,
        d_conv: c_int,
        kernel_size: c_int,
        silu: bool,
        stream: i64,
    );
    pub fn causal_conv1d_update_f16(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        conv_state: *mut c_void,
        out: *mut c_void,
        batch: c_int,
        d_conv: c_int,
        kernel_size: c_int,
        silu: bool,
        stream: i64,
    );
    pub fn causal_conv1d_update_bf16(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        conv_state: *mut c_void,
        out: *mut c_void,
        batch: c_int,
        d_conv: c_int,
        kernel_size: c_int,
        silu: bool,
        stream: i64,
    );

    pub fn causal_conv1d_update_slots_f32(
        x: *const f32,
        weight: *const f32,
        bias: *const f32,
        conv_state: *mut f32,
        slots: *const c_long,
        out: *mut f32,
        batch: c_int,
        d_conv: c_int,
        kernel_size: c_int,
        silu: bool,
        stream: i64,
    );
    pub fn causal_conv1d_update_slots_f16(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        conv_state: *mut c_void,
        slots: *const c_long,
        out: *mut c_void,
        batch: c_int,
        d_conv: c_int,
        kernel_size: c_int,
        silu: bool,
        stream: i64,
    );
    pub fn causal_conv1d_update_slots_bf16(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        conv_state: *mut c_void,
        slots: *const c_long,
        out: *mut c_void,
        batch: c_int,
        d_conv: c_int,
        kernel_size: c_int,
        silu: bool,
        stream: i64,
    );

    pub fn gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        bh: c_int,
        seq_len: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );
    pub fn gated_delta_rule_recurrence_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        bh: c_int,
        seq_len: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );
    pub fn gated_delta_rule_recurrence_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        bh: c_int,
        seq_len: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );

    pub fn gated_delta_rule_decode_slots_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut f32,
        batch: c_int,
        heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );
    pub fn gated_delta_rule_decode_slots_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        state: *mut c_void,
        slots: *const i64,
        out: *mut c_void,
        batch: c_int,
        heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );
    pub fn gated_delta_rule_decode_slots_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        state: *mut c_void,
        slots: *const i64,
        out: *mut c_void,
        batch: c_int,
        heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );
    pub fn gated_delta_rule_decode_slots_f16_state_f32(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        batch: c_int,
        heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );
    pub fn gated_delta_rule_decode_slots_bf16_state_f32(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        batch: c_int,
        heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );

    pub fn fused_gdn_gating_f32(
        al: *const f32,
        a: *const f32,
        b: *const f32,
        dt: *const f32,
        g: *mut f32,
        beta: *mut f32,
        bat: c_int,
        seq: c_int,
        h: c_int,
        s: i64,
    );
    pub fn fused_gdn_gating_f16(
        al: *const c_void,
        a: *const c_void,
        b: *const c_void,
        dt: *const c_void,
        g: *mut c_void,
        beta: *mut c_void,
        bat: c_int,
        seq: c_int,
        h: c_int,
        s: i64,
    );
    pub fn fused_gdn_gating_bf16(
        al: *const c_void,
        a: *const c_void,
        b: *const c_void,
        dt: *const c_void,
        g: *mut c_void,
        beta: *mut c_void,
        bat: c_int,
        seq: c_int,
        h: c_int,
        s: i64,
    );
    pub fn fused_gdn_gating_f16_alog_f32(
        al: *const f32,
        a: *const c_void,
        b: *const c_void,
        dt: *const c_void,
        g: *mut c_void,
        beta: *mut c_void,
        bat: c_int,
        seq: c_int,
        h: c_int,
        s: i64,
    );
    pub fn fused_gdn_gating_bf16_alog_f32(
        al: *const f32,
        a: *const c_void,
        b: *const c_void,
        dt: *const c_void,
        g: *mut c_void,
        beta: *mut c_void,
        bat: c_int,
        seq: c_int,
        h: c_int,
        s: i64,
    );

    pub fn gdn_gated_rmsnorm_silu_mul_f32(
        x: *const f32,
        z: *const f32,
        gamma: *const f32,
        bias: *const f32,
        out: *mut f32,
        rows: c_int,
        value_dim: c_int,
        group_size: c_int,
        eps: f32,
        per_group_weights: bool,
        has_bias: bool,
        s: i64,
    );
    pub fn gdn_gated_rmsnorm_silu_mul_f16(
        x: *const c_void,
        z: *const c_void,
        gamma: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        rows: c_int,
        value_dim: c_int,
        group_size: c_int,
        eps: f32,
        per_group_weights: bool,
        has_bias: bool,
        s: i64,
    );
    pub fn gdn_gated_rmsnorm_silu_mul_bf16(
        x: *const c_void,
        z: *const c_void,
        gamma: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        rows: c_int,
        value_dim: c_int,
        group_size: c_int,
        eps: f32,
        per_group_weights: bool,
        has_bias: bool,
        s: i64,
    );
    pub fn gdn_gated_rmsnorm_silu_mul_f16_wf32(
        x: *const c_void,
        z: *const c_void,
        gamma: *const f32,
        bias: *const f32,
        out: *mut c_void,
        rows: c_int,
        value_dim: c_int,
        group_size: c_int,
        eps: f32,
        per_group_weights: bool,
        has_bias: bool,
        s: i64,
    );
    pub fn gdn_gated_rmsnorm_silu_mul_bf16_wf32(
        x: *const c_void,
        z: *const c_void,
        gamma: *const f32,
        bias: *const f32,
        out: *mut c_void,
        rows: c_int,
        value_dim: c_int,
        group_size: c_int,
        eps: f32,
        per_group_weights: bool,
        has_bias: bool,
        s: i64,
    );

    // Fused L2 norm (last dim)
    pub fn l2_norm_last_dim_f32(
        input: *const f32,
        output: *mut f32,
        rows: c_int,
        dim: c_int,
        eps: f32,
        stream: i64,
    );
    pub fn l2_norm_last_dim_f16(
        input: *const c_void,
        output: *mut c_void,
        rows: c_int,
        dim: c_int,
        eps: f32,
        stream: i64,
    );
    pub fn l2_norm_last_dim_bf16(
        input: *const c_void,
        output: *mut c_void,
        rows: c_int,
        dim: c_int,
        eps: f32,
        stream: i64,
    );

    // Batched varlen recurrence (native dtype inputs, FP32 state)
    pub fn gated_delta_rule_recurrence_varlen_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut f32,
        cu_seqlens: *const u32,
        batch: c_int,
        num_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );
    pub fn gated_delta_rule_recurrence_varlen_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        cu_seqlens: *const u32,
        batch: c_int,
        num_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );
    pub fn gated_delta_rule_recurrence_varlen_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        cu_seqlens: *const u32,
        batch: c_int,
        num_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );

    // =========================================================================
    // MXFP4 GEMM
    // =========================================================================

    pub fn mxfp4_matmul_smallm_f16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const u8,
        bias: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        stream: i64,
    );

    pub fn mxfp4_matmul_smallm_bf16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const u8,
        bias: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        stream: i64,
    );

    pub fn mxfp4_matmul_f16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const u8,
        bias: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        stream: i64,
    );

    pub fn mxfp4_matmul_bf16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const u8,
        bias: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        stream: i64,
    );

    pub fn mxfp4_matmul_wmma_f16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const u8,
        bias: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        stream: i64,
    );

    pub fn mxfp4_matmul_wmma_bf16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const u8,
        bias: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        stream: i64,
    );

    pub fn mxfp4_indexed_moe_gemm_f16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const c_void,
        indices: *const u32,
        output: *mut c_void,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: i64,
    );

    pub fn mxfp4_indexed_moe_gemm_bf16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const c_void,
        indices: *const u32,
        output: *mut c_void,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: i64,
    );

    pub fn mxfp4_get_max_smem_optin() -> c_int;

    pub fn mxfp4_moe_grouped_gemm_f16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const c_void,
        indices: *const u32,
        output: *mut c_void,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: i64,
    );

    pub fn mxfp4_moe_grouped_gemm_bf16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const c_void,
        indices: *const u32,
        output: *mut c_void,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: i64,
    );

    pub fn mxfp4_moe_grouped_gemm_wmma_f16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const c_void,
        indices: *const u32,
        output: *mut c_void,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: i64,
    );

    // =========================================================================
    // MXFP4 CUTLASS hardware FP4 GEMM (SM100+ / Blackwell)
    // =========================================================================

    pub fn mxfp4_cutlass_gemm_f16(
        input: *const c_void,
        weight: *const c_void,
        input_sf: *const c_void,
        weight_sf: *const c_void,
        output: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        workspace: *mut c_void,
        workspace_bytes: i64,
        stream: i64,
    );

    pub fn mxfp4_cutlass_gemm_bf16(
        input: *const c_void,
        weight: *const c_void,
        input_sf: *const c_void,
        weight_sf: *const c_void,
        output: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
        workspace: *mut c_void,
        workspace_bytes: i64,
        stream: i64,
    );

    // MXFP4 activation quantization (SM100+ / Blackwell)
    pub fn mxfp4_quantize_activation_f16(
        input: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        swizzled_scales: *mut c_void,
        M: i32,
        K: i32,
        M_padded: i32,
        K_scale_padded: i32,
        stream: i64,
    );

    pub fn mxfp4_quantize_activation_bf16(
        input: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        swizzled_scales: *mut c_void,
        M: i32,
        K: i32,
        M_padded: i32,
        K_scale_padded: i32,
        stream: i64,
    );

    pub fn mxfp4_swizzle_weight_scales_e8m0(
        linear_scales: *const c_void,
        swizzled_scales: *mut c_void,
        rows: i32,
        cols: i32,
        rows_padded: i32,
        cols_padded: i32,
        stream: i64,
    );

    pub fn mxfp4_moe_grouped_gemm_wmma_bf16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        biases: *const c_void,
        indices: *const u32,
        output: *mut c_void,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: i64,
    );

    // ======================================================================
    // NVFP4 GEMM kernels (block_size=16, FP8 E4M3 block scales + F32 global)
    // ======================================================================

    pub fn nvfp4_matmul_smallm_f16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const u8,
        weight_global_scale: f32,
        bias: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        stream: i64,
    );

    pub fn nvfp4_matmul_smallm_bf16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const u8,
        weight_global_scale: f32,
        bias: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        stream: i64,
    );

    pub fn nvfp4_matmul_f16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const u8,
        weight_global_scale: f32,
        bias: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        stream: i64,
    );

    pub fn nvfp4_matmul_bf16(
        input: *const c_void,
        weight: *const u8,
        weight_scale: *const u8,
        weight_global_scale: f32,
        bias: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        stream: i64,
    );

    pub fn nvfp4_indexed_moe_gemm_f16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        weight_global_scales: *const f32,
        biases: *const c_void,
        indices: *const u32,
        output: *mut c_void,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: i64,
    );

    pub fn nvfp4_indexed_moe_gemm_bf16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        weight_global_scales: *const f32,
        biases: *const c_void,
        indices: *const u32,
        output: *mut c_void,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        input_has_topk_dim: bool,
        stream: i64,
    );

    pub fn nvfp4_moe_gemm_wmma_f16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        weight_global_scales: *const f32,
        sorted_token_ids: *const i32,
        expert_offsets: *const i32,
        topk_weights: *const f32,
        output: *mut c_void,
        num_experts: c_int,
        topk: c_int,
        size_m: c_int,
        size_n: c_int,
        size_k: c_int,
        input_has_topk_dim: bool,
        stream: i64,
    );

    pub fn nvfp4_moe_gemm_wmma_bf16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        weight_global_scales: *const f32,
        sorted_token_ids: *const i32,
        expert_offsets: *const i32,
        topk_weights: *const f32,
        output: *mut c_void,
        num_experts: c_int,
        topk: c_int,
        size_m: c_int,
        size_n: c_int,
        size_k: c_int,
        input_has_topk_dim: bool,
        stream: i64,
    );

    // ======================================================================
    // GPT-OSS SwiGLU kernels
    // ======================================================================

    pub fn gptoss_swiglu_f16(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        n: u32,
        alpha: f32,
        limit: f32,
        stream: i64,
    );

    pub fn gptoss_swiglu_bf16(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        n: u32,
        alpha: f32,
        limit: f32,
        stream: i64,
    );

    pub fn gptoss_swiglu_f32(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        n: u32,
        alpha: f32,
        limit: f32,
        stream: i64,
    );

    // =========================================================================
    // MLA (Multi-head Latent Attention) cache update
    // =========================================================================

    pub fn concat_and_cache_mla(
        ckv: *const c_void,
        k_pe: *const c_void,
        ckv_cache: *mut c_void,
        kpe_cache: *mut c_void,
        slot_mapping: *const i64,
        num_tokens: c_int,
        kv_lora_rank: c_int,
        kpe_head_dim: c_int,
        block_size: c_int,
        ckv_stride: c_int,
        kpe_stride: c_int,
        stream: i64,
        dtype: u32,
    );

    // =========================================================================
    // Fused MLA paged attention (non-FlashInfer)
    // =========================================================================

    pub fn mla_paged_attention_decode(
        out: *mut c_void,
        q_abs: *const c_void,
        q_pe: *const c_void,
        ckv_cache: *const c_void,
        kpe_cache: *const c_void,
        block_tables: *const c_int,
        context_lens: *const c_int,
        scale: f32,
        num_seqs: c_int,
        num_heads: c_int,
        kv_lora_rank: c_int,
        qk_rope_head_dim: c_int,
        block_size: c_int,
        max_num_blocks_per_seq: c_int,
        dtype: u32,
        stream: i64,
        tmp_out_buf: *mut c_void,
        tmp_max_buf: *mut c_void,
        tmp_sum_buf: *mut c_void,
        use_partitioned: c_int,
    );

    pub fn mla_paged_attention_prefill(
        out: *mut c_void,
        q_abs: *const c_void,
        q_pe: *const c_void,
        ckv_cache: *const c_void,
        kpe_cache: *const c_void,
        block_tables: *const c_int,
        context_lens: *const c_int,
        cu_seqlens_q: *const c_int,
        scale: f32,
        num_seqs: c_int,
        num_heads: c_int,
        kv_lora_rank: c_int,
        qk_rope_head_dim: c_int,
        block_size: c_int,
        max_num_blocks_per_seq: c_int,
        dtype: u32,
        stream: i64,
    );

    // =========================================================================
    // FlashInfer MLA decode (plan + run)
    // =========================================================================

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_mla_decode_plan_wrapper(
        kv_indptr_host: *const i32,
        batch_size: c_int,
        num_qo_heads: c_int,
        page_size: c_int,
        float_workspace: *mut c_void,
        float_workspace_size: i64,
        int_workspace: *mut c_void,
        int_workspace_size: i64,
        page_locked_buffer: *mut c_void,
        page_locked_size: i64,
        enable_cuda_graph: bool,
        dtype: u32,
        plan_info_out: *mut i64,
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_mla_decode_run_wrapper(
        o: *mut c_void,
        q_nope: *const c_void,
        q_pe: *const c_void,
        ckv_cache: *const c_void,
        kpe_cache: *const c_void,
        kv_indptr: *const i32,
        kv_indices: *const i32,
        kv_last_page_len: *const i32,
        batch_size: c_int,
        num_qo_heads: c_int,
        page_size: c_int,
        sm_scale: f32,
        rope_scale: f32,
        rope_theta: f32,
        float_workspace: *mut c_void,
        float_workspace_size: i64,
        int_workspace: *mut c_void,
        int_workspace_size: i64,
        plan_info: *const i64,
        dtype: u32,
        stream: i64,
    );

    // =========================================================================
    // FlashInfer MLA prefill (plan + run)
    // =========================================================================

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_mla_prefill_plan_wrapper(
        qo_indptr_host: *const i32,
        kv_indptr_host: *const i32,
        kv_len_arr_host: *const i32,
        batch_size: c_int,
        num_heads: c_int,
        head_dim_ckv: c_int,
        causal: bool,
        float_workspace: *mut c_void,
        float_workspace_size: i64,
        int_workspace: *mut c_void,
        int_workspace_size: i64,
        page_locked_buffer: *mut c_void,
        page_locked_size: i64,
        plan_info_out: *mut i64,
        stream: i64,
    );

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_mla_prefill_run_wrapper(
        o: *mut c_void,
        q_nope: *const c_void,
        q_pe: *const c_void,
        ckv_cache: *const c_void,
        kpe_cache: *const c_void,
        kv_indices: *const i32,
        num_heads: c_int,
        page_size: c_int,
        sm_scale: f32,
        float_workspace: *mut c_void,
        float_workspace_size: i64,
        int_workspace: *mut c_void,
        int_workspace_size: i64,
        plan_info: *const i64,
        causal: bool,
        dtype: u32,
        stream: i64,
    );

    // =========================================================================
    // TRT-LLM cubin loader callback (FlashInfer export interface)
    // =========================================================================

    #[cfg(feature = "trtllm")]
    pub fn FlashInferSetCubinCallback(
        callback: Option<unsafe extern "C" fn(*const std::ffi::c_char, *const std::ffi::c_char)>,
    );

    #[cfg(feature = "trtllm")]
    pub fn FlashInferSetCurrentCubin(binary: *const std::ffi::c_char, size: std::ffi::c_int);
}
