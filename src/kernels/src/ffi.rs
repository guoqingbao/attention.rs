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

    pub fn call_convert_to_fp8(
        input: *const c_void,
        output: *mut c_void,
        scales_out: *mut f32,
        num_tokens: c_int,
        num_heads: c_int,
        head_dim: c_int,
        dtype: u32,
        fixed_scale: f32,
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

    // Dense GEMM for ordinary GGUF tensors. The weights remain in their
    // original Q/K/IQ block format; only the activation is quantized to Q8_1.
    pub fn gguf_gemm(
        input: *const f32,
        weights: *const c_void,
        output: *mut f32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32,
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

    pub fn moe_w2_pack_from_mxfp4(
        packed: *const u8,
        scales_in: *const u8,
        planes: *mut u8,
        scales_out: *mut u8,
        e: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    pub fn moe_w2_unpack_to_fp8(
        planes: *const u8,
        scales: *const u8,
        out_w: *mut u8,
        out_s: *mut f32,
        e: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    pub fn moe_w2_unpack_by_ids_to_fp8(
        planes: *const u8,
        scales: *const u8,
        expert_ids: *const i32,
        out_w: *mut u8,
        out_s: *mut f32,
        u: i32,
        n: i32,
        k: i32,
        stream: i64,
    );

    pub fn moe_w2_dequantize_activation_fp8(
        input_q: *const u8,
        input_scales: *const f32,
        output: *mut c_void,
        rows: i32,
        k: i32,
        stream: i64,
    );

    pub fn moe_w2_swiglu_clamp_bf16(
        gate_up: *const c_void,
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        limit: f32,
        stream: i64,
    );

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

    // Grouped MoE WNA16 (compressed-tensors pack-quantized) kernel.
    // Weights are dense INT{4,8} values packed along K into uint32 words;
    // scales are symmetric per-group and kept in F32 for dequant precision.
    pub fn moe_gemm_wmma_wna16(
        input: *const c_void,
        weights: *const u32,
        weight_scales: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut c_void,
        expert_counts: *mut i32,
        expert_offsets: *mut i32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        bits: i32,
        group_size: i32,
        zero_point: i32,
        data_type: i32,
        is_prefill: bool,
        stream: i64,
    );

    // Decode GEMV for compressed-tensors WNA16. One warp computes one
    // output row and reuses the routed input vector from shared memory.
    pub fn moe_gemv_wna16(
        input: *const c_void,
        weights: *const u32,
        weight_scales: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut c_void,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        bits: i32,
        group_size: i32,
        zero_point: i32,
        data_type: i32,
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

    pub fn topk_select(
        scores: *const f32,     // in: [num_tokens, num_experts]
        topk_weights: *mut f32, // out: [num_tokens, topk]
        topk_indices: *mut u32, // out: [num_tokens, topk]
        num_experts: i32,
        num_tokens: i32,
        topk: i32,
        stream: i64,
    );

    pub fn fused_sigmoid_topk(
        logits: *const f32,     // in: [num_tokens, num_experts]
        bias: *const f32,       // in: [num_experts] or null
        topk_weights: *mut f32, // out: [num_tokens, topk] (original sigmoid scores)
        topk_indices: *mut u32, // out: [num_tokens, topk]
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
        input: *const c_void,        // [M, K]
        weight: *const u8,           // [N, K]
        weight_scale: *const c_void, // [N, K] (block-wise)
        output: *mut c_void,         // [M, N]
        m: c_int,
        n: c_int,
        k: c_int,
        scale_row_stride: c_int,
        block_size_y: c_int,
        block_size_x: c_int,
        scale_dtype: c_int, // 0=f32, 1=e8m0
        stream: i64,
    );

    pub fn fp8_matmul_f16_channelwise(
        input: *const c_void,     // [M, K]
        weight: *const u8,        // [N, K]
        weight_scale: *const f32, // [N, 1], one scale per output row
        output: *mut c_void,      // [M, N]
        m: c_int,
        n: c_int,
        k: c_int,
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
        weight_scale: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        scale_row_stride: c_int,
        block_size_y: c_int,
        block_size_x: c_int,
        scale_dtype: c_int, // 0=f32, 1=e8m0
        stream: i64,
    );

    pub fn fp8_matmul_bf16_channelwise(
        input: *const c_void,     // [M, K]
        weight: *const u8,        // [N, K]
        weight_scale: *const f32, // [N, 1], one scale per output row
        output: *mut c_void,      // [M, N]
        m: c_int,
        n: c_int,
        k: c_int,
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

    pub fn calculate_expert_offsets(
        expert_ids: *const i32,
        expert_counts: *mut i32,
        expert_offsets: *mut i32,
        num_experts: c_int,
        size_m: c_int,
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

    pub fn moe_fp8_scatter_rows_f32_to_f16(
        input: *const f32,
        src2dst_map: *const i32,
        output: *mut c_void,
        num_src_rows: i64,
        num_dst_rows: i64,
        num_cols: i64,
        weights: *const f32,
        stream: i64,
    );

    pub fn moe_fp8_scatter_rows_f32_to_bf16(
        input: *const f32,
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

    pub fn moe_fp8_grouped_gemm_f32(
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
        out: *mut f32,
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

    pub fn fp8_quantize_per_token_group_static_launch(
        input: *const c_void,
        output_q: *mut c_void,
        output_s: *mut f32,
        num_groups: c_int,
        group_size: c_int,
        num_groups_per_row: c_int,
        scale_stride: c_int,
        is_input_f16: bool,
        is_column_major_stats: bool,
        input_scale: f32,
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
    pub fn has_flashinfer_fp8_e4m3() -> bool;

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
    pub fn flashinfer_prefill_plan_fp8_fa2(
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
    pub fn flashinfer_prefill_run_fp8_fa2(
        out_ptr: *mut c_void,
        q_ptr: *mut c_void,
        q_cu_seqlens: *mut i32,
        total_num_rows: i32,
        k_data: *mut c_void,
        v_data: *mut c_void,
        indices: *mut i32,
        indptr: *mut i32,
        last_len: *mut i32,
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

    // NVFP4 online input scale computation
    pub fn nvfp4_compute_online_input_scale_f16(
        input: *const c_void,
        output: *mut f32,
        num_elements: i32,
        stream: i64,
    );

    pub fn nvfp4_compute_online_input_scale_bf16(
        input: *const c_void,
        output: *mut f32,
        num_elements: i32,
        stream: i64,
    );

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

    pub fn nvfp4_moe_compute_online_scales_f16(
        input: *const c_void,
        expert_offsets: *const i32,
        weight_global_scales: *const f32,
        alphas: *mut f32,
        input_scale_invs: *mut f32,
        num_experts: i32,
        K: i32,
        stream: i64,
    );

    pub fn nvfp4_moe_compute_online_scales_bf16(
        input: *const c_void,
        expert_offsets: *const i32,
        weight_global_scales: *const f32,
        alphas: *mut f32,
        input_scale_invs: *mut f32,
        num_experts: i32,
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

    // FlashInfer-style CUTLASS NVFP4 GEMM (SM100+, flashinfer feature)
    pub fn flashinfer_nvfp4_cutlass_gemm_f16(
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

    pub fn flashinfer_nvfp4_cutlass_gemm_bf16(
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

    pub fn nvfp4_cutlass_moe_gemm_f32(
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
        state_snapshots: *mut f32,
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
        conv_state: *mut f32,
        out: *mut c_void,
        state_snapshots: *mut c_void,
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
        conv_state: *mut f32,
        out: *mut c_void,
        state_snapshots: *mut c_void,
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
        conv_state: *mut f32,
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
        conv_state: *mut f32,
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
        conv_state: *mut f32,
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
        conv_state: *mut f32,
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
        g: *const f32,
        beta: *const f32,
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
        g: *const f32,
        beta: *const f32,
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
        g: *const f32,
        beta: *const f32,
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
        g: *const f32,
        beta: *const f32,
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
        al: *const f32,
        a: *const c_void,
        b: *const c_void,
        dt: *const f32,
        g: *mut f32,
        beta: *mut f32,
        bat: c_int,
        seq: c_int,
        h: c_int,
        s: i64,
    );
    pub fn fused_gdn_gating_bf16(
        al: *const f32,
        a: *const c_void,
        b: *const c_void,
        dt: *const f32,
        g: *mut f32,
        beta: *mut f32,
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
        state_snapshots: *mut f32,
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
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        state_snapshots: *mut f32,
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
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        state_snapshots: *mut f32,
        cu_seqlens: *const u32,
        batch: c_int,
        num_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        stream: i64,
    );

    pub fn gated_delta_rule_decode_slots_gqa_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        batch: c_int,
        num_v_heads: c_int,
        num_k_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        q_scale: f32,
        stream: i64,
    );
    pub fn gated_delta_rule_decode_slots_gqa_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        batch: c_int,
        num_v_heads: c_int,
        num_k_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        q_scale: f32,
        stream: i64,
    );
    pub fn gated_delta_rule_decode_slots_gqa_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut f32,
        batch: c_int,
        num_v_heads: c_int,
        num_k_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        q_scale: f32,
        stream: i64,
    );

    // Grouped-Query varlen recurrence (num_k_heads != num_v_heads, fused q_scale)
    pub fn gated_delta_rule_recurrence_varlen_gqa_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        state_snapshots: *mut f32,
        cu_seqlens: *const u32,
        batch: c_int,
        num_v_heads: c_int,
        num_k_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        q_scale: f32,
        stream: i64,
    );
    pub fn gated_delta_rule_recurrence_varlen_gqa_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        state_snapshots: *mut f32,
        cu_seqlens: *const u32,
        batch: c_int,
        num_v_heads: c_int,
        num_k_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        q_scale: f32,
        stream: i64,
    );
    pub fn gated_delta_rule_recurrence_varlen_gqa_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut f32,
        state_snapshots: *mut f32,
        cu_seqlens: *const u32,
        batch: c_int,
        num_v_heads: c_int,
        num_k_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        q_scale: f32,
        stream: i64,
    );

    // Persistent GQA varlen prefill (H in registers, 128 threads per CTA)
    pub fn gated_delta_rule_prefill_persistent_varlen_gqa_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        cu_seqlens: *const u32,
        total_tokens: c_int,
        batch: c_int,
        num_v_heads: c_int,
        num_k_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        q_scale: f32,
        stream: i64,
    ) -> c_int;
    pub fn gated_delta_rule_prefill_persistent_varlen_gqa_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut c_void,
        cu_seqlens: *const u32,
        total_tokens: c_int,
        batch: c_int,
        num_v_heads: c_int,
        num_k_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        q_scale: f32,
        stream: i64,
    ) -> c_int;

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
        topk_weights: *const f32,
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
        topk_weights: *const f32,
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
        force_lut: bool,
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
        force_lut: bool,
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
        force_lut: bool,
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
        force_lut: bool,
        stream: i64,
    );

    pub fn nvfp4_indexed_moe_gemm_f16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        weight_global_scales: *const f32,
        biases: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        output: *mut c_void,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        input_has_topk_dim: bool,
        force_lut: bool,
        stream: i64,
    );

    pub fn nvfp4_indexed_moe_gemm_bf16(
        input: *const c_void,
        weights: *const u8,
        weight_scales: *const u8,
        weight_global_scales: *const f32,
        biases: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        output: *mut c_void,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        n: c_int,
        k: c_int,
        has_bias: bool,
        input_has_topk_dim: bool,
        force_lut: bool,
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
        force_lut: bool,
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
        force_lut: bool,
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
    // Fused SiLU-and-Mul kernel
    // =========================================================================

    pub fn silu_and_mul_f16(
        gate_up: *const c_void,
        output: *mut c_void,
        total_elems: i64,
        N: i64,
        stream: i64,
    );

    pub fn silu_and_mul_bf16(
        gate_up: *const c_void,
        output: *mut c_void,
        total_elems: i64,
        N: i64,
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
        num_blocks: c_int,
        ckv_stride: c_int,
        kpe_stride: c_int,
        stream: i64,
        dtype: u32,
    ) -> c_int;

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
        max_context_len: c_int,
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
        total_tokens: c_int,
        dtype: u32,
        stream: i64,
    );

    // =========================================================================
    // Sparse MLA attention (DSA — DeepSeek Sparse Attention)
    // =========================================================================

    pub fn mla_sparse_attention_prefill(
        out: *mut c_void,
        q_abs: *const c_void,
        q_pe: *const c_void,
        ckv_cache: *const c_void,
        kpe_cache: *const c_void,
        block_tables: *const c_int,
        context_lens: *const c_int,
        cu_seqlens_q: *const c_int,
        topk_indices: *const c_int,
        scale: f32,
        num_seqs: c_int,
        num_heads: c_int,
        kv_lora_rank: c_int,
        qk_rope_head_dim: c_int,
        block_size: c_int,
        max_num_blocks_per_seq: c_int,
        topk: c_int,
        total_tokens: c_int,
        dtype: u32,
        stream: i64,
    );

    pub fn mla_sparse_attention_decode(
        out: *mut c_void,
        q_abs: *const c_void,
        q_pe: *const c_void,
        ckv_cache: *const c_void,
        kpe_cache: *const c_void,
        block_tables: *const c_int,
        context_lens: *const c_int,
        topk_indices: *const c_int,
        scale: f32,
        num_seqs: c_int,
        num_heads: c_int,
        kv_lora_rank: c_int,
        qk_rope_head_dim: c_int,
        block_size: c_int,
        max_num_blocks_per_seq: c_int,
        topk: c_int,
        dtype: u32,
        stream: i64,
    );

    // =========================================================================
    // DSA Lightning Indexer (fused score + causal + topk)
    // =========================================================================

    pub fn dsa_lightning_indexer_prefill(
        q: *const c_void,
        k: *const c_void,
        weights: *const c_void,
        topk_out: *mut c_int,
        seq_len: c_int,
        n_heads: c_int,
        head_dim: c_int,
        topk: c_int,
        score_scale: f32,
        stream: i64,
    ) -> c_int;

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

    // =========================================================================
    // Native flash attention (flash feature)
    // dtype: 0 = f16, 1 = bf16 (BF16 kernels only on SM80+)
    // =========================================================================

    #[cfg(feature = "flash")]
    pub fn call_flash_prefill_paged(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        o: *mut c_void,
        block_tables: *const c_int,
        block_table_stride: u32,
        cu_seqlens_q: *const u32,
        context_lens: *const u32,
        num_seqs: u32,
        max_q_len: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        cache_block_size: u32,
        sliding_window: u32,
        causal: u32,
        inv_sqrt_d: f32,
        softcap: f32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_prefill_paged_fp8(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        o: *mut c_void,
        block_tables: *const c_int,
        block_table_stride: u32,
        cu_seqlens_q: *const u32,
        context_lens: *const u32,
        num_seqs: u32,
        max_q_len: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        cache_block_size: u32,
        sliding_window: u32,
        causal: u32,
        inv_sqrt_d: f32,
        softcap: f32,
        k_scale_ptr: *const f32,
        v_scale_ptr: *const f32,
        fp8_cache_stride: u64,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_decode_paged(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        o: *mut c_void,
        block_tables: *const c_int,
        seq_lens: *const c_int,
        max_blocks_per_seq: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        num_seqs: u32,
        q_stride: u32,
        sliding_window: u32,
        softcap: f32,
        gqa_ratio: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_decode_paged_splitk(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        workspace: *mut c_void,
        block_tables: *const c_int,
        seq_lens: *const c_int,
        max_blocks_per_seq: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        num_seqs: u32,
        num_splits: u32,
        q_stride: u32,
        softcap: f32,
        sliding_window: u32,
        gqa_ratio: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_decode_paged_reduce(
        workspace: *const c_void,
        o: *mut c_void,
        num_q_heads: u32,
        head_dim: u32,
        num_splits: u32,
        num_seqs: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_decode_paged_fp8(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        o: *mut c_void,
        block_tables: *const c_int,
        seq_lens: *const c_int,
        max_blocks_per_seq: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        num_seqs: u32,
        q_stride: u32,
        sliding_window: u32,
        softcap: f32,
        k_scale_ptr: *const f32,
        v_scale_ptr: *const f32,
        fp8_cache_stride: u64,
        gqa_ratio: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_decode_paged_splitk_fp8(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        workspace: *mut c_void,
        block_tables: *const c_int,
        seq_lens: *const c_int,
        max_blocks_per_seq: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        num_seqs: u32,
        num_splits: u32,
        q_stride: u32,
        softcap: f32,
        k_scale_ptr: *const f32,
        v_scale_ptr: *const f32,
        fp8_cache_stride: u64,
        sliding_window: u32,
        gqa_ratio: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *mut c_void,
        value_cache: *mut c_void,
        slot_mapping: *const i64,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        cache_block_size: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_reshape_and_cache_fp8_kv(
        key: *const c_void,
        value: *const c_void,
        key_cache: *mut c_void,
        value_cache: *mut c_void,
        slot_mapping: *const i64,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        cache_block_size: u32,
        k_scale_ptr: *const f32,
        v_scale_ptr: *const f32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq_store_k8v4(
        key: *const c_void,
        value: *const c_void,
        key_cache: *mut c_void,
        v_absmax: *mut c_void,
        v_quant: *mut c_void,
        slot_mapping: *const i64,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        k_scale_ptr: *const f32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq_decode_k8v4(
        q: *const c_void,
        k_cache: *const c_void,
        v_absmax: *const c_void,
        v_quant: *const c_void,
        o: *mut c_void,
        block_tables: *const c_int,
        seq_lens: *const c_int,
        max_blocks_per_seq: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        num_seqs: u32,
        q_stride: u32,
        softcap: f32,
        k_scale_ptr: *const f32,
        sliding_window: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq_decode_k8v4_splitk(
        q: *const c_void,
        k_cache: *const c_void,
        v_absmax: *const c_void,
        v_quant: *const c_void,
        workspace: *mut c_void,
        block_tables: *const c_int,
        seq_lens: *const c_int,
        max_blocks_per_seq: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        num_splits: u32,
        num_seqs: u32,
        q_stride: u32,
        softcap: f32,
        k_scale_ptr: *const f32,
        sliding_window: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq4_store(
        key: *const c_void,
        value: *const c_void,
        k_absmax: *mut c_void,
        k_quant: *mut c_void,
        v_absmax: *mut c_void,
        v_quant: *mut c_void,
        slot_mapping: *const i64,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq4_decode(
        q: *const c_void,
        k_absmax: *const c_void,
        k_quant: *const c_void,
        v_absmax: *const c_void,
        v_quant: *const c_void,
        o: *mut c_void,
        block_tables: *const c_int,
        seq_lens: *const c_int,
        max_blocks_per_seq: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        num_seqs: u32,
        q_stride: u32,
        softcap: f32,
        sliding_window: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq4_decode_splitk(
        q: *const c_void,
        k_absmax: *const c_void,
        k_quant: *const c_void,
        v_absmax: *const c_void,
        v_quant: *const c_void,
        workspace: *mut c_void,
        block_tables: *const c_int,
        seq_lens: *const c_int,
        max_blocks_per_seq: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        num_splits: u32,
        num_seqs: u32,
        q_stride: u32,
        softcap: f32,
        sliding_window: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq3_store(
        key: *const c_void,
        value: *const c_void,
        k_absmax: *mut c_void,
        k_quant: *mut c_void,
        v_absmax: *mut c_void,
        v_quant: *mut c_void,
        slot_mapping: *const i64,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq3_decode(
        q: *const c_void,
        k_absmax: *const c_void,
        k_quant: *const c_void,
        v_absmax: *const c_void,
        v_quant: *const c_void,
        o: *mut c_void,
        block_tables: *const c_int,
        seq_lens: *const c_int,
        max_blocks_per_seq: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        num_seqs: u32,
        q_stride: u32,
        softcap: f32,
        sliding_window: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq3_decode_splitk(
        q: *const c_void,
        k_absmax: *const c_void,
        k_quant: *const c_void,
        v_absmax: *const c_void,
        v_quant: *const c_void,
        workspace: *mut c_void,
        block_tables: *const c_int,
        seq_lens: *const c_int,
        max_blocks_per_seq: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        num_splits: u32,
        num_seqs: u32,
        q_stride: u32,
        softcap: f32,
        sliding_window: u32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq4_prefill(
        q: *const c_void,
        k_absmax: *const c_void,
        k_quant: *const c_void,
        v_absmax: *const c_void,
        v_quant: *const c_void,
        o: *mut c_void,
        block_tables: *const c_int,
        block_table_stride: u32,
        cu_seqlens_q: *const u32,
        context_lens: *const u32,
        num_seqs: u32,
        max_q_len: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        cache_block_size: u32,
        sliding_window: u32,
        causal: u32,
        inv_sqrt_d: f32,
        softcap: f32,
        dtype: i32,
        stream: i64,
    );

    #[cfg(feature = "flash")]
    pub fn call_flash_tq3_prefill(
        q: *const c_void,
        k_absmax: *const c_void,
        k_quant: *const c_void,
        v_absmax: *const c_void,
        v_quant: *const c_void,
        o: *mut c_void,
        block_tables: *const c_int,
        block_table_stride: u32,
        cu_seqlens_q: *const u32,
        context_lens: *const u32,
        num_seqs: u32,
        max_q_len: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        cache_block_size: u32,
        sliding_window: u32,
        causal: u32,
        inv_sqrt_d: f32,
        softcap: f32,
        dtype: i32,
        stream: i64,
    );

    // MLX NVFP4 utility kernels
    pub fn mlx_nvfp4_repack_u32_to_u8(
        input: *const c_void,
        output: *mut c_void,
        num_rows: c_int,
        num_u32_cols: c_int,
        stream: i64,
    );

    pub fn mlx_nvfp4_dequant_embedding_f16(
        weight_u32: *const c_void,
        scales: *const c_void,
        output: *mut c_void,
        vocab_size: c_int,
        hidden_size: c_int,
        stream: i64,
    );

    pub fn mlx_nvfp4_dequant_embedding_bf16(
        weight_u32: *const c_void,
        scales: *const c_void,
        output: *mut c_void,
        vocab_size: c_int,
        hidden_size: c_int,
        stream: i64,
    );

    // DeepSeek V4 Hyper-Connection kernels
    pub fn ds_v4_hc_expand(
        x: *const c_void,
        out: *mut c_void,
        seq_len: c_int,
        hc: c_int,
        dim: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_v4_hc_mixes(
        x: *const c_void,
        hc_fn: *const c_void,
        mixes: *mut c_void,
        seq_len: c_int,
        hc: c_int,
        dim: c_int,
        mix_hc: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_v4_hc_scale_mixes(
        x: *const c_void,
        mixes: *mut c_void,
        seq_len: c_int,
        hc: c_int,
        dim: c_int,
        mix_hc: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_v4_hc_pre_from_mixes(
        x: *const c_void,
        mixes: *const c_void,
        hc_scale: *const c_void,
        hc_base: *const c_void,
        post: *mut c_void,
        comb: *mut c_void,
        out: *mut c_void,
        seq_len: c_int,
        hc: c_int,
        dim: c_int,
        sinkhorn_iters: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_v4_hc_pre_norm_from_mixes(
        x: *const c_void,
        mixes: *const c_void,
        hc_scale: *const c_void,
        hc_base: *const c_void,
        norm_weight: *const c_void,
        post: *mut c_void,
        comb: *mut c_void,
        out: *mut c_void,
        seq_len: c_int,
        hc: c_int,
        dim: c_int,
        sinkhorn_iters: c_int,
        hc_eps: f32,
        norm_eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_v4_hc_pre_output(
        x: *const c_void,
        pre: *const c_void,
        out: *mut c_void,
        seq_len: c_int,
        hc: c_int,
        dim: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_v4_hc_head_pre(
        mixes: *const c_void,
        hc_scale: *const c_void,
        hc_base: *const c_void,
        pre: *mut c_void,
        seq_len: c_int,
        hc: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_v4_hc_post(
        x: *const c_void,
        residual: *const c_void,
        post: *const c_void,
        comb: *const c_void,
        out: *mut c_void,
        seq_len: c_int,
        hc: c_int,
        dim: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_v4_hc_post_f32_branch(
        x: *const c_void,
        residual: *const c_void,
        post: *const c_void,
        comb: *const c_void,
        out: *mut c_void,
        seq_len: c_int,
        hc: c_int,
        dim: c_int,
        stream: i64,
    ) -> c_int;

    // DeepSeek V4 per-head RMSNorm
    pub fn ds_v4_head_rms_norm(
        x: *const c_void,
        out: *mut c_void,
        seq_len: c_int,
        num_heads: c_int,
        head_dim: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_v4_rms_norm(
        x: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        rows: c_int,
        dim: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    // ======== DeepSeek V4 Attention kernels (from ds_attention.cu) ========

    pub fn ds_sparse_attn_dispatch(
        q: *const c_void,
        kv: *const c_void,
        attn_sink: *const c_void,
        topk_idxs: *const c_int,
        out: *mut c_void,
        seq_len: c_int,
        num_heads: c_int,
        head_dim: c_int,
        kv_len: c_int,
        topk: c_int,
        softmax_scale: f32,
        stream: i64,
    ) -> c_int;

    // FlashMLA-ABI FP8 FOOTER pack (584 B/token)
    pub fn ds_fp8_kv_pack_footer(
        src_bf16: *const c_void,
        dst_u8: *mut c_void,
        num_tokens: c_int,
        page_block_size: c_int,
        src_stride: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_fp8_kv_pack_rows(
        src_bf16: *const c_void,
        dst_page_u8: *mut c_void,
        num_tokens: c_int,
        page_block_size: c_int,
        src_stride: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_fp8_kv_bytes_per_token() -> c_int;

    // FlashMLA SM90 sparse MLA
    pub fn flashmla_dsv4_supported(num_heads: c_int) -> c_int;
    pub fn flashmla_dsv4_bytes_per_token() -> c_int;
    pub fn flashmla_dsv4_decode_workspace_bytes(
        batch_size: c_int,
        s_q: c_int,
        num_heads: c_int,
        num_sm_parts_out: *mut c_int,
        tile_meta_bytes_out: *mut usize,
        num_splits_bytes_out: *mut usize,
        lse_accum_bytes_out: *mut usize,
        o_accum_bytes_out: *mut usize,
    ) -> c_int;

    pub fn flashmla_dsv4_sparse_decode(
        q_bf16: *const c_void,
        kv_fp8: *const c_void,
        indices: *const c_int,
        topk_length: *const c_int,
        attn_sink: *const f32,
        out_bf16: *mut c_void,
        lse: *mut f32,
        extra_kv_fp8: *const c_void,
        extra_indices: *const c_int,
        extra_topk_length: *const c_int,
        tile_scheduler_metadata: *mut c_int,
        num_splits: *mut c_int,
        lse_accum: *mut f32,
        o_accum: *mut f32,
        batch_size: c_int,
        s_q: c_int,
        num_heads: c_int,
        topk: c_int,
        num_blocks: c_int,
        page_block_size: c_int,
        extra_num_blocks: c_int,
        extra_page_block_size: c_int,
        extra_topk: c_int,
        num_sm_parts: c_int,
        sm_scale: f32,
        stream: i64,
    ) -> c_int;

    pub fn flashmla_dsv4_sparse_prefill(
        q_bf16: *const c_void,
        kv_bf16: *const c_void,
        indices: *const c_int,
        attn_sink: *const f32,
        topk_length: *const c_int,
        out_bf16: *mut c_void,
        lse: *mut f32,
        max_logits: *mut f32,
        s_q: c_int,
        s_kv: c_int,
        num_heads: c_int,
        topk: c_int,
        sm_scale: f32,
        stream: i64,
    ) -> c_int;

    // FlashInfer SM120 sparse MLA
    pub fn flashinfer_dsv4_sparse_sm120_compiled() -> c_int;
    pub fn flashinfer_dsv4_sparse_sm120_supported(num_heads: c_int, topk: c_int) -> c_int;
    pub fn flashinfer_dsv4_sparse_decode_sm120(
        q_bf16: *const c_void,
        kv_fp8: *const c_void,
        indices: *const c_int,
        topk_length: *const c_int,
        attn_sink: *const f32,
        out_bf16: *mut c_void,
        out_lse: *mut f32,
        mid_out_bf16: *mut c_void,
        mid_lse: *mut f32,
        extra_kv_fp8: *const c_void,
        extra_indices: *const c_int,
        extra_topk_length: *const c_int,
        num_tokens: c_int,
        num_heads: c_int,
        topk: c_int,
        num_splits: c_int,
        page_block_size: c_int,
        extra_topk: c_int,
        extra_page_block_size: c_int,
        chunks_per_block_override: c_int,
        sm_scale: f32,
        stream: i64,
    ) -> c_int;

    // ======== DeepSeek V4 Compressor kernels (from ds_compressor.cu) ========

    pub fn ds_apply_rope_hidden(
        x: *mut c_void,
        cos_cache: *const f32,
        sin_cache: *const f32,
        seq_len: c_int,
        local_heads: c_int,
        head_dim: c_int,
        rotary_dim: c_int,
        start_pos: c_int,
        inverse: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_apply_rope_hidden_strided(
        x: *mut c_void,
        cos_cache: *const f32,
        sin_cache: *const f32,
        seq_len: c_int,
        local_heads: c_int,
        head_dim: c_int,
        rotary_dim: c_int,
        start_pos: c_int,
        position_stride: c_int,
        inverse: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_compressor_bf16_linear_f32(
        x: *const c_void,
        weight: *const c_void,
        out: *mut f32,
        seq_len: c_int,
        in_dim: c_int,
        out_dim: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_compressor_nonoverlap_prefill_epilogue(
        scores: *const f32,
        values: *const f32,
        ape: *const f32,
        norm: *const c_void,
        weighted: *mut f32,
        out: *mut c_void,
        seq_len: c_int,
        head_dim: c_int,
        ratio: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_compressor_overlap_prefill_epilogue(
        scores: *const f32,
        values: *const f32,
        ape: *const f32,
        norm: *const c_void,
        weighted: *mut f32,
        out: *mut c_void,
        seq_len: c_int,
        head_dim: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_compressor_nonoverlap_decode_at(
        x: *const c_void,
        wkv: *const c_void,
        wgate: *const c_void,
        ape: *const f32,
        norm: *const c_void,
        kv_state: *mut f32,
        score_state: *mut f32,
        weighted: *mut f32,
        out: *mut c_void,
        start_pos: c_int,
        hidden_dim: c_int,
        head_dim: c_int,
        ratio: c_int,
        state_offset: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_compressor_overlap_decode_at(
        x: *const c_void,
        wkv: *const c_void,
        wgate: *const c_void,
        ape: *const f32,
        norm: *const c_void,
        kv_state: *mut f32,
        score_state: *mut f32,
        weighted: *mut f32,
        out: *mut c_void,
        start_pos: c_int,
        hidden_dim: c_int,
        head_dim: c_int,
        state_offset: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    // ======== DeepSeek V4 Indexer kernels (from ds_indexer.cu) ========

    pub fn ds_hadamard_fp4_quant_bf16(
        x: *mut c_void,
        rows: c_int,
        groups: c_int,
        dim: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_copy_device_bytes(
        src: *const c_void,
        dst: *mut c_void,
        bytes: usize,
        dst_byte_offset: usize,
        stream: i64,
    ) -> c_int;

    pub fn ds_fp8_act_quant_nope_bf16(
        input: *const c_void,
        output: *mut c_void,
        seq_len: c_int,
        local_heads: c_int,
        head_dim: c_int,
        rotary_dim: c_int,
        block_size: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_indexer_scores_decode(
        q: *const c_void,
        kv: *const c_void,
        weights: *const c_void,
        scores: *mut f32,
        local_heads: c_int,
        head_dim: c_int,
        compressed_len: c_int,
        score_scale: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_indexer_topk_prefill(
        scores: *const f32,
        topk_idxs: *mut c_int,
        seq_len: c_int,
        compressed_len: c_int,
        topk: c_int,
        ratio: c_int,
        offset: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_indexer_topk_decode(
        scores: *const f32,
        topk_idxs: *mut c_int,
        compressed_len: c_int,
        topk: c_int,
        offset: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_indexer_mask_scores_by_pos(
        scores: *mut f32,
        compressed_len: c_int,
        positions: *const i64,
        ratio: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_indexer_mask_scores_prefill_by_pos(
        scores: *mut f32,
        seq_len: c_int,
        compressed_len: c_int,
        positions: *const i64,
        ratio: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_indexer_topk_prefill_from_pos(
        scores: *const f32,
        topk_idxs: *mut c_int,
        seq_len: c_int,
        compressed_len: c_int,
        topk: c_int,
        positions: *const i64,
        ratio: c_int,
        offset: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_window_topk_indices_prefill_from_pos(
        out: *mut c_int,
        positions: *const i64,
        seq_len: c_int,
        window_size: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_window_topk_indices_chrono_from_pos(
        out: *mut c_int,
        positions: *const i64,
        seq_len: c_int,
        window_size: c_int,
        gather_start: c_int,
        gather_len: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_compress_topk_indices_prefill_from_pos(
        out: *mut c_int,
        positions: *const i64,
        seq_len: c_int,
        compressed: c_int,
        offset: c_int,
        ratio: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_write_window_rows_from_pos(
        cache: *mut c_void,
        rows: *const c_void,
        positions: *const i64,
        seq_len: c_int,
        window_size: c_int,
        head_dim: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_gather_ring_chrono(
        out: *mut c_void,
        cache: *const c_void,
        start_abs: c_int,
        n: c_int,
        window_size: c_int,
        head_dim: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_compressor_overlap_prefill_cont_epilogue(
        scores: *const f32,
        values: *const f32,
        ape: *const f32,
        norm: *const c_void,
        weighted: *mut f32,
        out: *mut c_void,
        state_kv: *const f32,
        state_scores: *const f32,
        bulk_rows: c_int,
        seq_len: c_int,
        chunk_start: c_int,
        head_dim: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_compressor_nonoverlap_prefill_cont_epilogue(
        scores: *const f32,
        values: *const f32,
        ape: *const f32,
        norm: *const c_void,
        weighted: *mut f32,
        out: *mut c_void,
        state_kv: *const f32,
        state_scores: *const f32,
        bulk_rows: c_int,
        seq_len: c_int,
        chunk_start: c_int,
        head_dim: c_int,
        ratio: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_concat_topk_indices(
        a: *const c_int,
        b: *const c_int,
        out: *mut c_int,
        seq_len: c_int,
        a_topk: c_int,
        b_topk: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_window_topk_indices(
        out: *mut c_int,
        seq_len: c_int,
        window_size: c_int,
        topk: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_window_topk_indices_decode(
        out: *mut c_int,
        start_pos: c_int,
        window_size: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_window_topk_indices_decode_from_pos(
        out: *mut c_int,
        positions: *const c_void,
        window_size: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_v4_hc_prewarm(max_elements: c_int, stream: i64) -> c_int;

    pub fn ds_v4_indexer_fp4_prewarm(max_elems: c_int, stream: i64) -> c_int;

    pub fn ds_v4_compressor_prewarm(stream: i64) -> c_int;

    pub fn ds_apply_rope_hidden_from_pos(
        x: *mut c_void,
        cos_cache: *const c_void,
        sin_cache: *const c_void,
        positions: *const c_void,
        seq_len: c_int,
        local_heads: c_int,
        head_dim: c_int,
        rotary_dim: c_int,
        position_offset: c_int,
        inverse: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_compressor_nonoverlap_decode_at_graph(
        x: *const c_void,
        wkv: *const c_void,
        wgate: *const c_void,
        ape: *const c_void,
        norm: *const c_void,
        kv_state: *mut f32,
        score_state: *mut f32,
        weighted: *mut f32,
        out: *mut c_void,
        positions: *const c_void,
        hidden_dim: c_int,
        head_dim: c_int,
        ratio: c_int,
        state_offset: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_compressor_overlap_decode_at_graph(
        x: *const c_void,
        wkv: *const c_void,
        wgate: *const c_void,
        ape: *const c_void,
        norm: *const c_void,
        kv_state: *mut f32,
        score_state: *mut f32,
        weighted: *mut f32,
        out: *mut c_void,
        positions: *const c_void,
        hidden_dim: c_int,
        head_dim: c_int,
        state_offset: c_int,
        eps: f32,
        stream: i64,
    ) -> c_int;

    pub fn ds_write_kv_row_from_pos(
        cache: *mut c_void,
        token: *const c_void,
        positions: *const c_void,
        window_size: c_int,
        head_dim: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_write_compressed_row_from_pos(
        cache: *mut c_void,
        row: *const c_void,
        positions: *const c_void,
        window_size: c_int,
        head_dim: c_int,
        ratio: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_write_indexer_row_from_pos(
        cache: *mut c_void,
        row: *const c_void,
        positions: *const c_void,
        head_dim: c_int,
        ratio: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_compress_topk_indices(
        out: *mut c_int,
        seq_len: c_int,
        compressed: c_int,
        ratio: c_int,
        offset: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_compress_topk_indices_decode(
        out: *mut c_int,
        compressed: c_int,
        offset: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_compress_topk_indices_decode_from_pos(
        out: *mut c_int,
        positions: *const c_void,
        compressed: c_int,
        offset: c_int,
        ratio: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_sparse_indices_to_local(
        input: *const c_void,
        output: *mut c_void,
        count: c_int,
        offset: c_int,
        stream: i64,
    ) -> c_int;

    pub fn ds_indexer_scores_prefill(
        q: *const c_void,
        kv: *const c_void,
        weights: *const c_void,
        scores: *mut f32,
        seq_len: c_int,
        local_heads: c_int,
        head_dim: c_int,
        compressed_len: c_int,
        score_scale: f32,
        stream: i64,
    ) -> c_int;

    // =========================================================================
    // FP8 grouped/strided GEMM (flashinfer + cutlass features)
    // =========================================================================

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_fp8_quantize_1x128(
        mat_quant: *mut c_void,
        scales: *mut f32,
        mat: *const c_void,
        shape_x: c_int,
        shape_y: c_int,
        stream: i64,
    ) -> c_int;

    #[cfg(feature = "flashinfer")]
    pub fn flashinfer_fp8_stride_batch_gemm(
        mat_d: *mut c_void,
        ld_d: c_int,
        stride_d: c_int,
        mat_a: *mut c_void,
        ld_a: c_int,
        stride_a: c_int,
        mat_b: *const c_void,
        ld_b: c_int,
        stride_b: c_int,
        num_problems: c_int,
        shape_m: c_int,
        shape_n: c_int,
        shape_k: c_int,
        scales_a: *mut f32,
        stride_scales_a: c_int,
        scales_b: *const f32,
        stream: i64,
    ) -> c_int;

    #[cfg(feature = "cutlass")]
    pub fn fp8_grouped_gemm_fused(
        input: *const c_void,
        input_q: *mut c_void,
        input_scales: *mut f32,
        weights: *const c_void,
        weight_scales: *const f32,
        output: *mut c_void,
        n_groups: c_int,
        seq_len: c_int,
        n: c_int,
        k: c_int,
        sm_version: c_int,
        workspace: *mut c_void,
        workspace_bytes: i64,
        stream: i64,
    ) -> c_int;

    pub fn fast_topk_select(
        scores: *const f32,
        indices_out: *mut i32,
        batch: c_int,
        seq_len: c_int,
        topk: c_int,
        stream: i64,
    ) -> c_int;

    /// DeepSeek V4 fused hash-gate: tid2eid lookup + expert-row dots +
    /// sqrt(softplus) + L1 normalize * route_scale.
    /// x/gate_weight: BF16; tid2eid: I64 [vocab, topk]; token_ids: U32 [seq];
    /// route_weights: F32 [seq, topk]; route_indices: U32 [seq, topk].
    pub fn ds_v4_hash_gate(
        x: *const c_void,
        gate_weight: *const c_void,
        tid2eid: *const c_void,
        token_ids: *const c_void,
        route_weights: *mut c_void,
        route_indices: *mut c_void,
        seq_len: c_int,
        hidden_dim: c_int,
        n_experts: c_int,
        topk: c_int,
        vocab_size: c_int,
        route_scale: f32,
        stream: i64,
    ) -> c_int;

}
