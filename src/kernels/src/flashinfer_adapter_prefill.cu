#include "flashinfer_common.cuh"

#if defined(FLASHINFER_ENABLE_FP8_E4M3)
extern "C" {
void flashinfer_prefill_wrapper_fp8(
    void* out_ptr,
    void* q_ptr,
    int32_t* q_cu_seqlens,
    int32_t* q_cu_seqlens_host,
    int32_t* kv_len_arr_host,
    int32_t total_num_rows,
    void* k_data, void* v_data,
    int32_t* indices,
    int32_t* indptr,
    int32_t* indptr_host,
    int32_t* last_len,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    float sm_scale,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    void* page_locked_int_buffer,
    size_t page_locked_int_size,
    bool enable_cuda_graph,
    int32_t data_type,
    int32_t out_data_type,
    cudaStream_t stream
);
void flashinfer_prefill_ragged_wrapper_fp8(
    void* out_ptr,
    void* q_ptr,
    int32_t* q_cu_seqlens,
    int32_t* kv_cu_seqlens,
    int32_t* q_cu_seqlens_host,
    int32_t* kv_cu_seqlens_host,
    int32_t total_num_rows,
    int32_t total_kv_rows,
    void* k_ptr,
    void* v_ptr,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    float sm_scale,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    void* page_locked_int_buffer,
    size_t page_locked_int_size,
    bool enable_cuda_graph,
    int32_t data_type,
    int32_t out_data_type,
    cudaStream_t stream
);
}
#endif

template <typename T>
__global__ void scale_output_inplace_kernel(T* out, int64_t numel, float scale) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; idx < numel; idx += stride) {
        float x = static_cast<float>(out[idx]);
        out[idx] = static_cast<T>(x * scale);
    }
}

extern "C" {

void flashinfer_prefill_ragged_wrapper(
    void* out_ptr,
    void* q_ptr,
    int32_t* q_cu_seqlens,
    int32_t* kv_cu_seqlens,
    int32_t* q_cu_seqlens_host,
    int32_t* kv_cu_seqlens_host,
    int32_t total_num_rows,
    int32_t total_kv_rows,
    void* k_ptr,
    void* v_ptr,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    float sm_scale,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    void* page_locked_int_buffer,
    size_t page_locked_int_size,
    bool enable_cuda_graph,
    int32_t data_type,
    int32_t out_data_type,
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    if (data_type == 2) {
#if defined(FLASHINFER_ENABLE_FP8_E4M3)
        flashinfer_prefill_ragged_wrapper_fp8(
            out_ptr, q_ptr, q_cu_seqlens, kv_cu_seqlens, q_cu_seqlens_host, kv_cu_seqlens_host,
            total_num_rows, total_kv_rows, k_ptr, v_ptr, batch_size, num_qo_heads, num_kv_heads,
            head_dim, sm_scale, k_scale_ptr, v_scale_ptr, workspace_float, workspace_float_size,
            workspace_int, workspace_int_size, page_locked_int_buffer, page_locked_int_size,
            enable_cuda_graph, data_type, out_data_type, stream
        );
#endif
        return;
    }
    if (page_locked_int_buffer == nullptr || page_locked_int_size < workspace_int_size) {
        return;
    }
    if (q_cu_seqlens_host == nullptr || kv_cu_seqlens_host == nullptr ||
        q_cu_seqlens == nullptr || kv_cu_seqlens == nullptr) {
        return;
    }
    const float rope_scale = 1.0f;
    const float rope_theta = 10000.0f;
#if defined(SM_90_PASS)
    std::vector<int32_t> kv_len_host(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        kv_len_host[i] = kv_cu_seqlens_host[i + 1] - kv_cu_seqlens_host[i];
    }
    PrefillPlanSM90Info plan_info;
    PrefillSM90Plan<int32_t>(
        workspace_float, workspace_float_size,
        workspace_int, page_locked_int_buffer, workspace_int_size,
        plan_info,
        q_cu_seqlens_host, kv_cu_seqlens_host, kv_len_host.data(),
        total_num_rows, batch_size,
        num_qo_heads, num_kv_heads, head_dim, head_dim, 1,
        true, enable_cuda_graph,
        (out_data_type == 1 ? sizeof(nv_bfloat16) : sizeof(half)),
        stream
    );
    using IdType = int32_t;
    auto run_ragged_sm90 = [&](auto dtype_val) {
        using DTypeKV = decltype(dtype_val);
        using DTypeQ = DTypeKV;
        using DTypeOut = DTypeKV;
        BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeOut, IdType> params;
        FillSM90RaggedParams<DTypeQ, DTypeKV, DTypeOut, IdType>(
            params, q_ptr, k_ptr, v_ptr, out_ptr,
            num_qo_heads, num_kv_heads, head_dim, total_num_rows, total_kv_rows, sm_scale,
            workspace_int, plan_info);
        using AttentionType = DefaultAttentionAlias<false, false, false, false>;
        DISPATCH_HEAD_DIM_SM90(head_dim, HEAD_DIM, {
            if (plan_info.same_schedule_for_all_heads) {
                BatchPrefillWithRaggedKVCacheDispatched<
                    HEAD_DIM, HEAD_DIM, MaskMode::kCausal, false, true, AttentionType>(
                    params, false, stream);
            } else {
                BatchPrefillWithRaggedKVCacheDispatched<
                    HEAD_DIM, HEAD_DIM, MaskMode::kCausal, false, false, AttentionType>(
                    params, false, stream);
            }
        });
    };
    if (data_type == 1) {
        run_ragged_sm90(cutlass::bfloat16_t{});
    } else {
        run_ragged_sm90(cutlass::half_t{});
    }
#else
    auto run_ragged = [&](auto dtype_val) {
        using DTypeKV = decltype(dtype_val);
        using DTypeQ = DTypeKV;
        using DTypeOut = DTypeKV;
        using IdType = int32_t;
        PrefillPlanInfo plan_info;
        PrefillPlan<int32_t>(
            workspace_float, workspace_float_size,
            workspace_int, page_locked_int_buffer, workspace_int_size,
            plan_info,
            q_cu_seqlens_host, kv_cu_seqlens_host, total_num_rows,
            batch_size, num_qo_heads, num_kv_heads, head_dim, head_dim, 1,
            enable_cuda_graph, sizeof(DTypeOut),
            -1, 0, false, 0, stream
        );
        using ParamsType = BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeOut, IdType>;
        ParamsType params(
            (DTypeQ*)q_ptr, (DTypeKV*)k_ptr, (DTypeKV*)v_ptr, nullptr,
            q_cu_seqlens, kv_cu_seqlens, nullptr, nullptr, nullptr,
            (DTypeOut*)out_ptr, nullptr, nullptr,
            num_qo_heads, num_kv_heads,
            num_qo_heads * head_dim, head_dim,
            num_kv_heads * head_dim, head_dim,
            -1, 0.0f, sm_scale, rope_scale, rope_theta
        );
        params.request_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.request_indices_offset);
        params.qo_tile_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_tile_indices_offset);
        params.kv_tile_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_tile_indices_offset);
        params.o_indptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.o_indptr_offset);
        params.kv_chunk_size_ptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_chunk_size_ptr_offset);
        params.max_total_num_rows = plan_info.total_num_rows;
        params.padded_batch_size = plan_info.padded_batch_size;
        params.partition_kv = plan_info.split_kv;
        params.merge_indptr = nullptr;
        params.block_valid_mask = nullptr;
        if (plan_info.split_kv) {
            params.merge_indptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.merge_indptr_offset);
            if (plan_info.enable_cuda_graph) {
                params.block_valid_mask = GetPtrFromBaseOffset<bool>(workspace_int, plan_info.block_valid_mask_offset);
            }
        }
        params.total_num_rows = nullptr;
        if (plan_info.enable_cuda_graph) {
            params.total_num_rows = GetPtrFromBaseOffset<uint32_t>(workspace_int, plan_info.total_num_rows_offset);
        }
        DTypeOut* tmp_v = nullptr;
        float* tmp_s = nullptr;
        if (plan_info.split_kv) {
            tmp_v = GetPtrFromBaseOffset<DTypeOut>(workspace_float, plan_info.v_offset);
            tmp_s = GetPtrFromBaseOffset<float>(workspace_float, plan_info.s_offset);
        }
        using AttentionType = DefaultAttentionAlias<false, false, false, false>;
        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
            DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
                BatchPrefillWithRaggedKVCacheDispatched<
                    CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
                    PosEncodingMode::kNone, false, MaskMode::kCausal,
                    AttentionType, ParamsType>(
                    params, tmp_v, tmp_s, false, stream
                );
            });
        });
    };
    if (data_type == 1) {
        run_ragged(nv_bfloat16{});
    } else {
        run_ragged(half{});
    }
#endif
#endif
}

// ============================================================================
// Separated prefill plan + run (plan computed once per forward, reused by layers)
// Plan output: int64_t[16] tagged vector
//   [0] = tag: 0 = non-SM90 PrefillPlanInfo (15 values at [1..15])
//              1 = SM90 PrefillPlanSM90Info (9 values at [1..9])
// ============================================================================

void flashinfer_prefill_plan_wrapper(
    int32_t* q_cu_seqlens_host,
    int32_t* indptr_host,
    int32_t* kv_len_arr_host,
    int32_t total_num_rows,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    bool enable_cuda_graph,
    int32_t window_left,
    int32_t out_data_type,
    void* workspace_float, size_t workspace_float_size,
    void* workspace_int, size_t workspace_int_size,
    void* page_locked_buffer, size_t page_locked_size,
    int64_t* plan_info_out,
    cudaStream_t stream)
{
#ifdef USE_FLASHINFER
    if (page_locked_buffer == nullptr || page_locked_size < workspace_int_size) {
        fprintf(stderr, "[flashinfer][prefill_plan] page_locked_buffer too small\n");
        return;
    }
    if (q_cu_seqlens_host == nullptr || indptr_host == nullptr || kv_len_arr_host == nullptr) {
        fprintf(stderr, "[flashinfer][prefill_plan] host pointers are null\n");
        return;
    }

#if defined(SM_90_PASS)
    {
        PrefillPlanSM90Info plan_info;
        PrefillSM90Plan<int32_t>(
            workspace_float, workspace_float_size,
            workspace_int, page_locked_buffer, workspace_int_size,
            plan_info,
            q_cu_seqlens_host, indptr_host, kv_len_arr_host,
            total_num_rows, batch_size,
            num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
            true, enable_cuda_graph,
            (out_data_type == 1 ? sizeof(nv_bfloat16) : sizeof(half)),
            stream
        );
        if (plan_info_out != nullptr) {
            plan_info_out[0] = 1; // tag: SM90
            auto vec = plan_info.ToVector();
            for (int i = 0; i < 9; ++i) {
                plan_info_out[1 + i] = vec[i];
            }
        }
    }
#else
    {
        PrefillPlanInfo plan_info;
        PrefillPlan<int32_t>(
            workspace_float, workspace_float_size,
            workspace_int, page_locked_buffer, workspace_int_size,
            plan_info,
            q_cu_seqlens_host, indptr_host, total_num_rows,
            batch_size, num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
            enable_cuda_graph, (out_data_type == 1 ? sizeof(nv_bfloat16) : sizeof(half)),
            window_left > 0 ? window_left : -1, 0, false, 0,
            stream
        );
        if (!ValidatePrefillPlanInfoBounds(
                plan_info, workspace_float_size, workspace_int_size)) {
            return;
        }
        if (plan_info_out != nullptr) {
            plan_info_out[0] = 0; // tag: non-SM90
            auto vec = plan_info.ToVector();
            for (int i = 0; i < 15; ++i) {
                plan_info_out[1 + i] = vec[i];
            }
        }
    }
#endif
#endif
}

void flashinfer_prefill_run_wrapper(
    void* out_ptr,
    void* q_ptr,
    int32_t* q_cu_seqlens,
    int32_t total_num_rows,
    void* k_data, void* v_data,
    int32_t* indices,
    int32_t* indptr,
    int32_t* last_len,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    float sm_scale,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    void* workspace_float, size_t workspace_float_size,
    void* workspace_int, size_t workspace_int_size,
    int32_t window_left,
    float logits_soft_cap,
    int32_t data_type,
    int32_t out_data_type,
    const int64_t* plan_info_vec,
    cudaStream_t stream)
{
#ifdef USE_FLASHINFER
    if (plan_info_vec == nullptr) {
        fprintf(stderr, "[flashinfer][prefill_run] plan_info is null\n");
        return;
    }
    const float rope_scale = 1.0f;
    const float rope_theta = 10000.0f;
    int64_t tag = plan_info_vec[0];
    const int64_t* plan_data = plan_info_vec + 1;

    if (data_type == 2) {
#if defined(SM_90_PASS) && defined(FLASHINFER_ENABLE_FP8_E4M3)
        extern void flashinfer_prefill_run_fp8(
            void*, void*, int32_t, void*, void*, int32_t*,
            int32_t, int32_t, int32_t, int32_t, float,
            const float*, const float*, void*, int32_t,
            const int64_t*, cudaStream_t);
        flashinfer_prefill_run_fp8(
            out_ptr, q_ptr, total_num_rows, k_data, v_data, indices,
            num_qo_heads, num_kv_heads, head_dim, page_size, sm_scale,
            k_scale_ptr, v_scale_ptr, workspace_int, out_data_type,
            plan_info_vec, stream);
#elif defined(FLASHINFER_ENABLE_FP8_E4M3)
        if (tag != 0) {
            fprintf(stderr, "[flashinfer][prefill_run] FA2 FP8 path but plan tag=%lld (expected 0)\n", (long long)tag);
            return;
        }
        auto run_fp8_fa2 = [&](auto dtype_q_val) {
            using DTypeQ = decltype(dtype_q_val);
            using DTypeKV = uint8_t;
            using DTypeOut = DTypeQ;
            using IdType = int32_t;

            PrefillPlanInfo plan_info;
            std::vector<int64_t> vec(plan_data, plan_data + 15);
            plan_info.FromVector(vec);

            DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                paged_kv_t<DTypeKV, IdType> paged_kv(
                    num_kv_heads, page_size, head_dim, batch_size, QKVLayout::kNHD,
                    (DTypeKV*)k_data, (DTypeKV*)v_data,
                    indices, indptr, last_len
                );

                using ParamsType = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeOut, IdType>;
                ParamsType params(
                    (DTypeQ*)q_ptr, paged_kv, nullptr, q_cu_seqlens,
                    nullptr, nullptr,
                    (DTypeOut*)out_ptr, nullptr, nullptr,
                    num_qo_heads, num_qo_heads * head_dim, head_dim,
                    window_left > 0 ? window_left : -1, logits_soft_cap, sm_scale, rope_scale, rope_theta
                );

                params.request_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.request_indices_offset);
                params.qo_tile_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_tile_indices_offset);
                params.kv_tile_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_tile_indices_offset);
                params.o_indptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.o_indptr_offset);
                params.kv_chunk_size_ptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_chunk_size_ptr_offset);
                params.max_total_num_rows = plan_info.total_num_rows;
                params.padded_batch_size = plan_info.padded_batch_size;
                params.partition_kv = plan_info.split_kv;
                params.merge_indptr = nullptr;
                params.block_valid_mask = nullptr;
                if (plan_info.split_kv) {
                    params.merge_indptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.merge_indptr_offset);
                    if (plan_info.enable_cuda_graph) {
                        params.block_valid_mask = GetPtrFromBaseOffset<bool>(workspace_int, plan_info.block_valid_mask_offset);
                    }
                }
                params.total_num_rows = nullptr;
                if (plan_info.enable_cuda_graph) {
                    params.total_num_rows = GetPtrFromBaseOffset<uint32_t>(workspace_int, plan_info.total_num_rows_offset);
                }

                DTypeOut* tmp_v = nullptr;
                float* tmp_s = nullptr;
                if (plan_info.split_kv) {
                    tmp_v = GetPtrFromBaseOffset<DTypeOut>(workspace_float, plan_info.v_offset);
                    tmp_s = GetPtrFromBaseOffset<float>(workspace_float, plan_info.s_offset);
                }

                DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
                    using AttentionType = DefaultAttentionAlias<false, false, false, false>;
                    BatchPrefillWithPagedKVCacheDispatched<
                        CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
                        PosEncodingMode::kNone, false, MaskMode::kCausal,
                        AttentionType, ParamsType>(
                        params, tmp_v, tmp_s, false, stream
                    );
                });
            });
        };
        if (out_data_type == 1) {
            run_fp8_fa2(nv_bfloat16{});
        } else {
            run_fp8_fa2(half{});
        }
#else
        fprintf(stderr, "[flashinfer][prefill_run] FP8 prefill requires SM89+ with FP8 support\n");
#endif
        return;
    }

#if defined(SM_90_PASS)
    if (tag != 1) {
        fprintf(stderr, "[flashinfer][prefill_run] SM90 build but plan tag=%lld (expected 1)\n", (long long)tag);
        return;
    }
    {
        PrefillPlanSM90Info plan_info;
        std::vector<int64_t> vec(plan_data, plan_data + 9);
        plan_info.FromVector(vec);

        auto run_sm90 = [&](auto dtype_val) {
            using DTypeKV = decltype(dtype_val);
            using DTypeQ = DTypeKV;
            using DTypeOut = DTypeKV;
            using IdType = int32_t;

            BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeOut, IdType> params;
            FillSM90PagedParams<DTypeQ, DTypeKV, DTypeOut, IdType>(
                params, q_ptr, k_data, v_data, out_ptr,
                num_qo_heads, num_kv_heads, head_dim, page_size,
                total_num_rows, sm_scale, indices, workspace_int, window_left, logits_soft_cap, plan_info);

            bool use_swa = (window_left > 0);
            using AttentionNoSWA = DefaultAttentionAlias<false, false, false, false>;
            using AttentionSWA   = DefaultAttentionAlias<false, true,  false, false>;
            DISPATCH_HEAD_DIM_SM90(head_dim, HEAD_DIM, {
                if (use_swa) {
                    if (plan_info.same_schedule_for_all_heads) {
                        BatchPrefillWithPagedKVCacheDispatched<
                            HEAD_DIM, HEAD_DIM, MaskMode::kCausal, true, true, AttentionSWA>(
                            params, false, stream);
                    } else {
                        BatchPrefillWithPagedKVCacheDispatched<
                            HEAD_DIM, HEAD_DIM, MaskMode::kCausal, true, false, AttentionSWA>(
                            params, false, stream);
                    }
                } else {
                    if (plan_info.same_schedule_for_all_heads) {
                        BatchPrefillWithPagedKVCacheDispatched<
                            HEAD_DIM, HEAD_DIM, MaskMode::kCausal, false, true, AttentionNoSWA>(
                            params, false, stream);
                    } else {
                        BatchPrefillWithPagedKVCacheDispatched<
                            HEAD_DIM, HEAD_DIM, MaskMode::kCausal, false, false, AttentionNoSWA>(
                            params, false, stream);
                    }
                }
            });
        };
        if (data_type == 1) {
            run_sm90(cutlass::bfloat16_t{});
        } else {
            run_sm90(cutlass::half_t{});
        }
    }
#else
    if (tag != 0) {
        fprintf(stderr, "[flashinfer][prefill_run] non-SM90 build but plan tag=%lld (expected 0)\n", (long long)tag);
        return;
    }
    {
        PrefillPlanInfo plan_info;
        std::vector<int64_t> vec(plan_data, plan_data + 15);
        plan_info.FromVector(vec);
        if (!ValidatePrefillPlanInfoBounds(
                plan_info, workspace_float_size, workspace_int_size)) {
            return;
        }

        auto run_prefill = [&](auto dtype_kv_val) {
            using DTypeKV = decltype(dtype_kv_val);
            using DTypeQ = DTypeKV;
            using DTypeOut = DTypeKV;
            using IdType = int32_t;

            DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                paged_kv_t<DTypeKV, IdType> paged_kv(
                    num_kv_heads, page_size, head_dim, batch_size, QKVLayout::kNHD,
                    (DTypeKV*)k_data, (DTypeKV*)v_data,
                    indices, indptr, last_len
                );

                using ParamsType = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeOut, IdType>;
                ParamsType params(
                    (DTypeQ*)q_ptr, paged_kv, nullptr, q_cu_seqlens,
                    nullptr, nullptr,
                    (DTypeOut*)out_ptr, nullptr, nullptr,
                    num_qo_heads, num_qo_heads * head_dim, head_dim,
                    window_left > 0 ? window_left : -1, logits_soft_cap, sm_scale, rope_scale, rope_theta
                );

                params.request_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.request_indices_offset);
                params.qo_tile_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_tile_indices_offset);
                params.kv_tile_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_tile_indices_offset);
                params.o_indptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.o_indptr_offset);
                params.kv_chunk_size_ptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_chunk_size_ptr_offset);
                params.max_total_num_rows = plan_info.total_num_rows;
                params.padded_batch_size = plan_info.padded_batch_size;
                params.partition_kv = plan_info.split_kv;
                params.merge_indptr = nullptr;
                params.block_valid_mask = nullptr;
                if (plan_info.split_kv) {
                    params.merge_indptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.merge_indptr_offset);
                    if (plan_info.enable_cuda_graph) {
                        params.block_valid_mask = GetPtrFromBaseOffset<bool>(workspace_int, plan_info.block_valid_mask_offset);
                    }
                }
                params.total_num_rows = nullptr;
                if (plan_info.enable_cuda_graph) {
                    params.total_num_rows = GetPtrFromBaseOffset<uint32_t>(workspace_int, plan_info.total_num_rows_offset);
                }

                DTypeOut* tmp_v = nullptr;
                float* tmp_s = nullptr;
                if (plan_info.split_kv) {
                    tmp_v = GetPtrFromBaseOffset<DTypeOut>(workspace_float, plan_info.v_offset);
                    tmp_s = GetPtrFromBaseOffset<float>(workspace_float, plan_info.s_offset);
                }

                bool use_swa = (window_left > 0);
                DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
                    if (use_swa) {
                        using AttentionType = DefaultAttentionAlias<false, true, false, false>;
                        BatchPrefillWithPagedKVCacheDispatched<
                            CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
                            PosEncodingMode::kNone, false, MaskMode::kCausal,
                            AttentionType, ParamsType>(
                            params, tmp_v, tmp_s, false, stream
                        );
                    } else {
                        using AttentionType = DefaultAttentionAlias<false, false, false, false>;
                        BatchPrefillWithPagedKVCacheDispatched<
                            CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
                            PosEncodingMode::kNone, false, MaskMode::kCausal,
                            AttentionType, ParamsType>(
                            params, tmp_v, tmp_s, false, stream
                        );
                    }
                });
            });
        };
        if (data_type == 1) {
            run_prefill(nv_bfloat16{});
        } else {
            run_prefill(half{});
        }
    }
#endif
#endif
}

}
