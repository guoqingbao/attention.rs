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

    pub fn update_kv_scales_f32(
        k: *const c_void,
        v: *const c_void,
        elements: c_long,
        k_scales: *const f32,
        v_scales: *const f32,
        stream: i64,
    );

    pub fn update_kv_scales_f16(
        k: *const c_void,
        v: *const c_void,
        elements: c_long,
        k_scales: *const f32,
        v_scales: *const f32,
        stream: i64,
    );

    pub fn update_kv_scales_bf16(
        k: *const c_void,
        v: *const c_void,
        elements: c_long,
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
        gating_output: *const f32,        // in： [num_tokens, num_experts]
        token_expert_indices: *const i32, // out: [num_tokens, topk]
        topk_weights: *const f32,         // out: [num_tokens, topk]
        topk_indices: *const u32,         // out: [num_tokens, topk]
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
        stream: i64,
    );

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
}
