/*
 * FA2-style FP8 prefill fallback for head_dim >= 256 on SM90.
 * Compiled WITHOUT SM_90_PASS so the FA2 BatchPrefillPagedParams is used.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>
#include <algorithm>

#ifdef USE_FLASHINFER
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/pos_enc.cuh>
using namespace flashinfer;

template <bool use_custom_mask, bool use_sliding_window, bool use_logits_soft_cap, bool use_alibi>
using DefaultAttentionAlias =
    DefaultAttention<use_custom_mask, use_sliding_window, use_logits_soft_cap, use_alibi>;

static inline bool ValidateWorkspaceOffset(
    const char* name, int64_t offset, size_t workspace_size, const char* ws_type) {
    if (offset < 0 || (size_t)offset > workspace_size) {
        fprintf(stderr, "[flashinfer][fp8_fa2] %s offset %lld is out of bounds for %s workspace size %zu\n",
                name, (long long)offset, ws_type, workspace_size);
        return false;
    }
    return true;
}

static inline bool ValidatePrefillPlanInfoBounds(
    const PrefillPlanInfo& plan_info,
    size_t float_workspace_size,
    size_t int_workspace_size) {
    return ValidateWorkspaceOffset(
               "request_indices", plan_info.request_indices_offset, int_workspace_size, "int") &&
           ValidateWorkspaceOffset(
               "qo_tile_indices", plan_info.qo_tile_indices_offset, int_workspace_size, "int") &&
           ValidateWorkspaceOffset(
               "kv_tile_indices", plan_info.kv_tile_indices_offset, int_workspace_size, "int") &&
           ValidateWorkspaceOffset(
               "merge_indptr", plan_info.merge_indptr_offset, int_workspace_size, "int") &&
           ValidateWorkspaceOffset(
               "o_indptr", plan_info.o_indptr_offset, int_workspace_size, "int") &&
           ValidateWorkspaceOffset(
               "kv_chunk_size_ptr", plan_info.kv_chunk_size_ptr_offset, int_workspace_size, "int") &&
           ValidateWorkspaceOffset(
               "total_num_rows", plan_info.total_num_rows_offset, int_workspace_size, "int") &&
           ValidateWorkspaceOffset(
               "v", plan_info.v_offset, float_workspace_size, "float") &&
           ValidateWorkspaceOffset(
               "s", plan_info.s_offset, float_workspace_size, "float") &&
           ValidateWorkspaceOffset(
               "block_valid_mask", plan_info.block_valid_mask_offset, int_workspace_size, "int");
}
#endif

#if defined(USE_FLASHINFER)

extern "C" {

void flashinfer_prefill_run_fp8_fa2(
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
    int32_t out_data_type,
    const int64_t* plan_info_vec,
    cudaStream_t stream)
{
    #if defined(FLASHINFER_ENABLE_FP8_E4M3)
    if (plan_info_vec == nullptr) {
        fprintf(stderr, "[flashinfer][prefill_fp8_fa2] plan_info is null\n");
        return;
    }
    const float rope_scale = 1.0f;
    const float rope_theta = 10000.0f;
    int64_t tag = plan_info_vec[0];
    if (tag != 0) {
        fprintf(stderr, "[flashinfer][prefill_fp8_fa2] expected tag=0 (FA2 plan), got tag=%lld\n", (long long)tag);
        return;
    }
    const int64_t* plan_data = plan_info_vec + 1;

    auto run_fp8_fa2 = [&](auto dtype_q_val) {
        using DTypeQ = decltype(dtype_q_val);
        using DTypeKV = __nv_fp8_e4m3;
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
    #endif //FLASHINFER_ENABLE_FP8_E4M3
}

void flashinfer_prefill_plan_fp8_fa2(
    const int32_t* q_cu_seqlens_host,
    const int32_t* indptr_host,
    const int32_t* kv_len_arr_host,
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
    void* page_locked_int_buffer, size_t page_locked_int_size,
    int64_t* plan_info_out,
    int64_t stream_i64)
{
    #if defined(FLASHINFER_ENABLE_FP8_E4M3)
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_i64);

    PrefillPlanInfo plan_info;
    PrefillPlan<int32_t>(
        workspace_float, workspace_float_size,
        workspace_int, page_locked_int_buffer, workspace_int_size,
        plan_info,
        const_cast<int32_t*>(q_cu_seqlens_host),
        const_cast<int32_t*>(indptr_host),
        static_cast<uint32_t>(total_num_rows),
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_qo_heads),
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(page_size),
        enable_cuda_graph,
        static_cast<uint32_t>(out_data_type == 1 ? sizeof(nv_bfloat16) : sizeof(half)),
        window_left > 0 ? window_left : -1, 0, false, static_cast<int64_t>(0),
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
    #endif //FLASHINFER_ENABLE_FP8_E4M3
}

} // extern "C"
#endif // USE_FLASHINFER 
