#include <cuda_runtime.h>
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

// Static page-locked buffer for FlashInfer planning (4MB, allocated once)
static void* g_page_locked_buffer = nullptr;
constexpr size_t PAGE_LOCKED_BUFFER_SIZE = 4 * 1024 * 1024;

static void* get_page_locked_buffer() {
    if (g_page_locked_buffer == nullptr) {
        cudaMallocHost(&g_page_locked_buffer, PAGE_LOCKED_BUFFER_SIZE);
    }
    return g_page_locked_buffer;
}
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
    bool is_fp8,
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
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

    if (is_fp8) {
        run(uint8_t(0));
    } else {
        run(half(0));
    }
#endif
}

void flashinfer_decode_wrapper(
    void* out_ptr,
    void* q_ptr,
    void* k_data, void* v_data,
    int32_t* indices,
    int32_t* indptr,           // Device pointer for paged_kv
    int32_t* indptr_host,      // Host pointer for planning (avoids D2H copy)
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
    bool is_fp8,
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    const float rope_scale = 1.0f;
    const float rope_theta = 10000.0f;

    auto run_decode = [&](auto dtype_kv_val) {
        using DTypeKV = decltype(dtype_kv_val);
        using DTypeQ = half;
        using DTypeOut = half;
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
                void* page_locked_buffer = get_page_locked_buffer(); 

                using AttentionType = DefaultAttention<false, false, false, false>;
                using ParamsType = BatchDecodeParams<DTypeQ, DTypeKV, DTypeOut, IdType>;

                // Use host pointer directly - no D2H copy needed
                DecodePlan<HEAD_DIM, PosEncodingMode::kNone, AttentionType, ParamsType>(
                    workspace_float, workspace_float_size,
                    workspace_int, page_locked_buffer, workspace_int_size,
                    plan_info,
                    indptr_host, batch_size, num_qo_heads, page_size, false /* graph */, stream,
                    BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
                        GROUP_SIZE, HEAD_DIM, PosEncodingMode::kNone,
                        AttentionType, ParamsType>
                );

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
                params.block_valid_mask = GetPtrFromBaseOffset<bool>(workspace_int, plan_info.block_valid_mask_offset);
                params.partition_kv = plan_info.split_kv;
                params.padded_batch_size = plan_info.padded_batch_size;
                
                DTypeOut* tmp_v = GetPtrFromBaseOffset<DTypeOut>(workspace_float, plan_info.v_offset);
                float* tmp_s = GetPtrFromBaseOffset<float>(workspace_float, plan_info.s_offset);

                BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, PosEncodingMode::kNone,
                     AttentionType, ParamsType>(
                     params, tmp_v, tmp_s, false /* pdl */, stream
                );
                
                // Should not free static buffer
                //cudaFreeHost(page_locked_buffer);
            });
        });
    };

    if (is_fp8) {
#ifndef NO_FP8_KVCACHE
        run_decode(__nv_fp8_e4m3{});
#endif
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
    bool enable_cuda_graph,
    bool is_fp8,
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    const float rope_scale = 1.0f;
    const float rope_theta = 10000.0f;

    auto run_prefill = [&](auto dtype_kv_val) {
        using DTypeKV = decltype(dtype_kv_val);
        using DTypeQ = half;
        using DTypeOut = half;
        using IdType = int32_t;

        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
            paged_kv_t<DTypeKV, IdType> paged_kv(
                num_kv_heads, page_size, head_dim, batch_size, QKVLayout::kNHD,
                (DTypeKV*)k_data, (DTypeKV*)v_data,
                indices, indptr, last_len
            );

            PrefillPlanInfo plan_info;
            void* page_locked_buffer = get_page_locked_buffer();

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
            params.merge_indptr = GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.merge_indptr_offset);
            params.block_valid_mask = GetPtrFromBaseOffset<bool>(workspace_int, plan_info.block_valid_mask_offset);
            params.max_total_num_rows = plan_info.total_num_rows;
            params.padded_batch_size = plan_info.padded_batch_size;
            params.partition_kv = plan_info.split_kv;
            params.total_num_rows = GetPtrFromBaseOffset<uint32_t>(workspace_int, plan_info.total_num_rows_offset);

            DTypeOut* tmp_v = GetPtrFromBaseOffset<DTypeOut>(workspace_float, plan_info.v_offset);
            float* tmp_s = GetPtrFromBaseOffset<float>(workspace_float, plan_info.s_offset);
            
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

    if (is_fp8) {
#ifndef NO_FP8_KVCACHE
        run_prefill(__nv_fp8_e4m3{});
#endif
    } else {
        run_prefill(half{});
    }
#endif
}

}