#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <algorithm>

#ifdef USE_FLASHINFER
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

using namespace flashinfer;

#endif

extern "C" {

void flashinfer_append_kv_cache(
    void* k_data_ptr,
    void* v_data_ptr,
    void* new_k_ptr,
    void* new_v_ptr,
    int32_t* paged_kv_indices,
    int32_t* paged_kv_indptr,
    int32_t* paged_kv_last_len,
    int32_t* batch_indices, // Pre-constructed in Rust
    int32_t* positions,     // Pre-constructed in Rust
    int32_t nnz,            // Total tokens to append
    int32_t batch_size,
    int32_t num_heads,
    int32_t head_dim,
    int32_t page_size,
    int32_t data_type,
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    if (data_type == 2) {
        // FP8 KV cache is not supported by this wrapper yet.
        return;
    }

    auto run = [&](auto dtype_val) {
        using DType = decltype(dtype_val);
        paged_kv_t<DType, int32_t> paged_kv(
            num_heads, page_size, head_dim, batch_size, QKVLayout::kNHD,
            (DType*)k_data_ptr, (DType*)v_data_ptr,
            paged_kv_indices, paged_kv_indptr, paged_kv_last_len
        );
        
        if (batch_size > 0 && batch_indices && positions) {
             // Prefill append (Ragged)
             size_t stride_n = num_heads * head_dim;
             size_t stride_h = head_dim;
             
             AppendPagedKVCache(paged_kv, (DType*)new_k_ptr, (DType*)new_v_ptr, 
                                batch_indices, positions, nnz,
                                stride_n, stride_h, stride_n, stride_h, 
                                stream);
        } else {
             // Decode append (Batch)
             AppendPagedKVCacheDecode(paged_kv, (DType*)new_k_ptr, (DType*)new_v_ptr, stream);
        }
    };

    if (data_type == 1) {
        run(nv_bfloat16(0));
    } else {
        run(half(0));
    }
#endif
}

void flashinfer_decode_plan_wrapper(
    int32_t* indptr_host,      // Host pointer for planning
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    void* page_locked_int_buffer,
    size_t page_locked_int_size,
    bool enable_cuda_graph,
    int32_t data_type,
    int64_t* plan_info_out,     // length 10
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    if (data_type == 2) {
        // FP8 KV cache is not supported by this wrapper yet.
        return;
    }
    if (page_locked_int_buffer == nullptr || page_locked_int_size < workspace_int_size) {
        return;
    }

    auto run_plan = [&](auto dtype_kv_val) {
        using DTypeKV = decltype(dtype_kv_val);
        using DTypeQ = DTypeKV;
        using DTypeOut = DTypeKV;
        using IdType = int32_t;

        uint32_t group_size = num_qo_heads / num_kv_heads;

        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
            DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, {
                using AttentionType = DefaultAttention<false, false, false, false>;
                using ParamsType = BatchDecodeParams<DTypeQ, DTypeKV, DTypeOut, IdType>;

                DecodePlanInfo plan_info;
                DecodePlan<HEAD_DIM, PosEncodingMode::kNone, AttentionType, ParamsType>(
                    workspace_float, workspace_float_size,
                    workspace_int, page_locked_int_buffer, workspace_int_size,
                    plan_info,
                    indptr_host, batch_size, num_qo_heads, page_size, enable_cuda_graph, stream,
                    BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
                        GROUP_SIZE, HEAD_DIM, PosEncodingMode::kNone,
                        AttentionType, ParamsType>
                );

                if (plan_info_out != nullptr) {
                    plan_info_out[0] = plan_info.padded_batch_size;
                    plan_info_out[1] = plan_info.v_offset;
                    plan_info_out[2] = plan_info.s_offset;
                    plan_info_out[3] = plan_info.request_indices_offset;
                    plan_info_out[4] = plan_info.kv_tile_indices_offset;
                    plan_info_out[5] = plan_info.o_indptr_offset;
                    plan_info_out[6] = plan_info.block_valid_mask_offset;
                    plan_info_out[7] = plan_info.kv_chunk_size_ptr_offset;
                    plan_info_out[8] = plan_info.enable_cuda_graph;
                    plan_info_out[9] = plan_info.split_kv;
                }
            });
        });
    };

    if (data_type == 1) {
        run_plan(nv_bfloat16{});
    } else {
        run_plan(half{});
    }
#endif
}

void flashinfer_decode_run_wrapper(
    void* out_ptr,
    void* q_ptr,
    void* k_data, void* v_data,
    int32_t* indices,
    int32_t* indptr,           // Device pointer for paged_kv
    int32_t* last_len,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    float sm_scale,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    const int64_t* plan_info_vec, // length 10
    int32_t data_type,
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    const float rope_scale = 1.0f;
    const float rope_theta = 10000.0f;

    if (data_type == 2) {
        // FP8 KV cache is not supported by this wrapper yet.
        return;
    }
    if (plan_info_vec == nullptr) {
        return;
    }

    auto run_decode = [&](auto dtype_kv_val) {
        using DTypeKV = decltype(dtype_kv_val);
        using DTypeQ = DTypeKV;
        using DTypeOut = DTypeKV;
        using IdType = int32_t;
        
        uint32_t group_size = num_qo_heads / num_kv_heads;

        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
            DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, {
                paged_kv_t<DTypeKV, IdType> paged_kv(
                    num_kv_heads, page_size, head_dim, batch_size, QKVLayout::kNHD,
                    (DTypeKV*)k_data, (DTypeKV*)v_data,
                    indices, indptr, last_len
                );

                DecodePlanInfo plan_info;
                std::vector<int64_t> vec(plan_info_vec, plan_info_vec + 10);
                plan_info.FromVector(vec);

                using AttentionType = DefaultAttention<false, false, false, false>;
                using ParamsType = BatchDecodeParams<DTypeQ, DTypeKV, DTypeOut, IdType>;

                ParamsType params(
                    (DTypeQ*)q_ptr, nullptr /* q_rope_offset */, paged_kv, (DTypeOut*)out_ptr,
                    nullptr /* lse */, nullptr /* alibi */, num_qo_heads,
                    num_qo_heads * head_dim /* q_stride_n */, head_dim /* q_stride_h */,
                    -1 /* window_left */, 0.0f /* logits_cap */, sm_scale, rope_scale, rope_theta
                );
                
                params.request_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.request_indices_offset);
                params.kv_tile_indices = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_tile_indices_offset);
                params.o_indptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.o_indptr_offset);
                params.kv_chunk_size_ptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_chunk_size_ptr_offset);
                params.partition_kv = plan_info.split_kv;
                params.padded_batch_size = plan_info.padded_batch_size;
                params.block_valid_mask = nullptr;
                if (plan_info.split_kv && plan_info.enable_cuda_graph) {
                    params.block_valid_mask = GetPtrFromBaseOffset<bool>(workspace_int, plan_info.block_valid_mask_offset);
                }
                
                DTypeOut* tmp_v = nullptr;
                float* tmp_s = nullptr;
                if (plan_info.split_kv) {
                    tmp_v = GetPtrFromBaseOffset<DTypeOut>(workspace_float, plan_info.v_offset);
                    tmp_s = GetPtrFromBaseOffset<float>(workspace_float, plan_info.s_offset);
                }

                BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, PosEncodingMode::kNone,
                     AttentionType, ParamsType>(
                     params, tmp_v, tmp_s, false /* pdl */, stream
                );
            });
        });
    };

    if (data_type == 1) {
        run_decode(nv_bfloat16{});
    } else {
        run_decode(half{});
    }
#endif
}

void flashinfer_prefill_wrapper(
    void* out_ptr,
    void* q_ptr,
    int32_t* q_cu_seqlens,      // Device pointer for kernel params
    int32_t* q_cu_seqlens_host, // Host pointer for planning (avoids D2H copy)
    int32_t total_num_rows,     // Total tokens (from host to avoid D2H + read)
    void* k_data, void* v_data,
    int32_t* indices,
    int32_t* indptr,            // Device pointer for paged_kv
    int32_t* indptr_host,       // Host pointer for planning (avoids D2H copy)
    int32_t* last_len,
    int32_t batch_size,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    float sm_scale,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    void* page_locked_int_buffer,
    size_t page_locked_int_size,
    bool enable_cuda_graph,
    int32_t data_type,
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    const float rope_scale = 1.0f;
    const float rope_theta = 10000.0f;

    if (data_type == 2) {
        // FP8 KV cache is not supported by this wrapper yet.
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

            PrefillPlanInfo plan_info;
            if (page_locked_int_buffer == nullptr || page_locked_int_size < workspace_int_size) {
                return;
            }
            void* page_locked_buffer = page_locked_int_buffer;

            // Use host pointers directly - no D2H copy needed
            PrefillPlan<int32_t>(
                workspace_float, workspace_float_size,
                workspace_int, page_locked_buffer, workspace_int_size,
                plan_info,
                q_cu_seqlens_host, indptr_host, total_num_rows,
                batch_size, num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
                enable_cuda_graph, sizeof(DTypeOut),
                -1 /* window_left */, 0 /* fixed_split_size */, false /* disable_split_kv */, 0,
                stream
            );

            using ParamsType = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeOut, IdType>;
            ParamsType params(
                (DTypeQ*)q_ptr, paged_kv, nullptr /* custom_mask */, q_cu_seqlens,
                nullptr /* mask indptr */, nullptr /* q rope offset */,
                (DTypeOut*)out_ptr, nullptr /* lse */, nullptr /* alibi */,
                num_qo_heads, num_qo_heads * head_dim /* q_stride_n */, head_dim /* q_stride_h */,
                -1 /* window */, 0.0f /* logits_cap */, sm_scale, rope_scale, rope_theta
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
            
            using AttentionType = DefaultAttention<false, false, false, false>;

            DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
                BatchPrefillWithPagedKVCacheDispatched<
                    CTA_TILE_Q, HEAD_DIM, HEAD_DIM, 
                    PosEncodingMode::kNone, false, MaskMode::kCausal,
                    AttentionType,
                    ParamsType>(
                    params, tmp_v, tmp_s, false /* pdl */, stream
                );
            });

            // Should not free static buffer
           // cudaFreeHost(page_locked_buffer);
        });
    };

    if (data_type == 1) {
        run_prefill(nv_bfloat16{});
    } else {
        run_prefill(half{});
    }
#endif
}

}
