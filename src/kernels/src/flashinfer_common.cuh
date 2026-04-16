#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>

#ifdef USE_FLASHINFER
    #include <flashinfer/attention/decode.cuh>
    #include <flashinfer/attention/scheduler.cuh>
    #if defined(SM_90_PASS)
        #include <flashinfer/attention/hopper/prefill_sm90.cuh>
        #include <flashinfer/attention/hopper/variants.cuh>
        #include <flashinfer/attention/hopper/default_params.cuh>
    #else
        #include <flashinfer/attention/prefill.cuh>
        #include <flashinfer/attention/default_prefill_params.cuh>
        #include <flashinfer/attention/variants.cuh>
    #endif

#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/utils.cuh>

#if defined(SM_90_PASS)
#include <cutlass/numeric_types.h>
#endif
#include <flashinfer/pos_enc.cuh>
using namespace flashinfer;

#if !defined(SM_90_PASS)
static inline bool ValidateWorkspaceOffset(
    const char* name,
    int64_t offset,
    size_t workspace_size,
    const char* workspace_kind) {
    if (offset < 0 || static_cast<size_t>(offset) >= workspace_size) {
        fprintf(stderr,
                "[flashinfer][prefill_plan] %s offset=%lld exceeds %s workspace size=%zu\n",
                name,
                static_cast<long long>(offset),
                workspace_kind,
                workspace_size);
        return false;
    }
    return true;
}

static inline bool ValidatePrefillPlanInfoBounds(
    const PrefillPlanInfo& plan_info,
    size_t float_workspace_size,
    size_t int_workspace_size) {
    return ValidateWorkspaceOffset(
               "total_num_rows", plan_info.total_num_rows_offset, int_workspace_size, "int") &&
           ValidateWorkspaceOffset(
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
               "v", plan_info.v_offset, float_workspace_size, "float") &&
           ValidateWorkspaceOffset(
               "s", plan_info.s_offset, float_workspace_size, "float") &&
           ValidateWorkspaceOffset(
               "block_valid_mask", plan_info.block_valid_mask_offset, int_workspace_size, "int");
}
#endif

#if !defined(SM_90_PASS)
template <bool use_custom_mask, bool use_sliding_window, bool use_logits_soft_cap, bool use_alibi>
using DefaultAttentionAlias =
    DefaultAttention<use_custom_mask, use_sliding_window, use_logits_soft_cap, use_alibi>;
using DefaultDecodeAttention = DefaultAttentionAlias<false, false, false, false>;
#else // SM_90_PASS defined (Hopper FA3 path)
template <bool use_custom_mask, bool use_sliding_window, bool use_logits_soft_cap, bool use_alibi>
using DefaultAttentionAlias = DefaultAttention<use_logits_soft_cap>;

struct DefaultDecodeAttention {
    static constexpr bool use_softmax = true;
    uint32_t kv_len;
    uint32_t window_left;
    float sm_scale_log2;
    float soft_cap_pre_tanh_scale;
    bool use_logits_soft_cap;

    template <typename Params>
    __device__ __host__ DefaultDecodeAttention(const Params& params, uint32_t batch_idx,
                                               uint8_t* smem_ptr) {
        (void)smem_ptr;
        kv_len = params.get_kv_len(batch_idx);
        window_left = (params.window_left >= 0) ? params.window_left : kv_len;
        use_logits_soft_cap = params.logits_soft_cap > 0.f;
        if (use_logits_soft_cap) {
            soft_cap_pre_tanh_scale = params.sm_scale / params.logits_soft_cap;
            sm_scale_log2 = math::log2e * params.logits_soft_cap;
        } else {
            soft_cap_pre_tanh_scale = 0.f;
            sm_scale_log2 = params.sm_scale * math::log2e;
        }
    }

    template <typename Params, typename T>
    __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                                 uint32_t qo_idx, uint32_t kv_idx,
                                                 uint32_t qo_head_idx, uint32_t kv_head_idx) {
        if (use_logits_soft_cap) {
            logits = math::tanh(logits * soft_cap_pre_tanh_scale);
        }
        return logits;
    }

    template <typename Params>
    __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
        return (kv_idx + 1 + window_left >= kv_len + qo_idx);
    }

    template <typename Params, typename T, typename T_M>
    __device__ __forceinline__ T OutputTransform(const Params& params, T output, uint32_t batch_idx,
                                                 uint32_t qo_idx, uint32_t qo_head_idx, T_M& m,
                                                 float& d, float scale) {
        float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;
        return output * d_rcp;
    }
};
#endif

#if defined(SM_90_PASS)
template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
static inline void FillSM90PagedParams(
    BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>& params,
    void* q_ptr,
    void* k_data,
    void* v_data,
    void* out_ptr,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t page_size,
    int64_t nnz_qo,
    float sm_scale,
    IdType* indices,
    void* workspace_int,
    int window_left,
    float logits_soft_cap,
    const PrefillPlanSM90Info& plan_info
) {
    params.q_ptr = static_cast<DTypeQ*>(q_ptr);
    params.k_ptr = static_cast<DTypeKV*>(k_data);
    params.v_ptr = static_cast<DTypeKV*>(v_data);
    params.o_ptr = static_cast<DTypeO*>(out_ptr);
    params.lse_ptr = nullptr;
    params.q_stride_n = static_cast<int64_t>(num_qo_heads) * head_dim;
    params.q_stride_h = head_dim;
    params.o_stride_n = params.q_stride_n;
    params.o_stride_h = params.q_stride_h;
    params.k_stride_n = static_cast<int64_t>(num_kv_heads) * head_dim;
    params.k_stride_h = head_dim;
    params.v_stride_n = params.k_stride_n;
    params.v_stride_h = params.k_stride_h;
    params.k_page_stride = static_cast<int64_t>(page_size) * num_kv_heads * head_dim;
    params.v_page_stride = params.k_page_stride;
    params.nnz_qo = nnz_qo;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.group_size = num_qo_heads / num_kv_heads;
    params.page_size = page_size;
    params.window_left = window_left > 0 ? window_left : -1;
    params.causal = true;
    params.additional_params.logits_soft_cap = logits_soft_cap;
    params.additional_params.sm_scale = sm_scale;
    params.additional_params.maybe_prefix_len_ptr = nullptr;
    params.additional_params.maybe_token_pos_in_items_ptr = nullptr;
    params.additional_params.token_pos_in_items_len = 0;
    params.additional_params.maybe_max_item_len_ptr = nullptr;

    params.qo_tile_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_tile_indices_offset);
    params.qo_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_indptr_offset);
    params.kv_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_indptr_offset);
    params.qo_lens =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_len_offset);
    params.kv_lens =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_len_offset);
    params.head_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.head_indices_offset);
    params.work_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.work_indptr_offset);
    params.batch_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.batch_indices_offset);
    params.kv_indices = indices;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
static inline void FillSM90RaggedParams(
    BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>& params,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr,
    void* out_ptr,
    int32_t num_qo_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int64_t nnz_qo,
    int64_t nnz_kv,
    float sm_scale,
    void* workspace_int,
    const PrefillPlanSM90Info& plan_info
) {
    params.q_ptr = static_cast<DTypeQ*>(q_ptr);
    params.k_ptr = static_cast<DTypeKV*>(k_ptr);
    params.v_ptr = static_cast<DTypeKV*>(v_ptr);
    params.o_ptr = static_cast<DTypeO*>(out_ptr);
    params.lse_ptr = nullptr;
    params.q_stride_n = static_cast<int64_t>(num_qo_heads) * head_dim;
    params.q_stride_h = head_dim;
    params.o_stride_n = params.q_stride_n;
    params.o_stride_h = params.q_stride_h;
    params.k_stride_n = static_cast<int64_t>(num_kv_heads) * head_dim;
    params.k_stride_h = head_dim;
    params.v_stride_n = params.k_stride_n;
    params.v_stride_h = params.k_stride_h;
    params.nnz_qo = nnz_qo;
    params.nnz_kv = nnz_kv;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.group_size = num_qo_heads / num_kv_heads;
    params.window_left = -1;
    params.causal = true;
    params.additional_params.logits_soft_cap = 0.0f;
    params.additional_params.sm_scale = sm_scale;
    params.additional_params.maybe_prefix_len_ptr = nullptr;
    params.additional_params.maybe_token_pos_in_items_ptr = nullptr;
    params.additional_params.token_pos_in_items_len = 0;
    params.additional_params.maybe_max_item_len_ptr = nullptr;

    params.qo_tile_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_tile_indices_offset);
    params.qo_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_indptr_offset);
    params.kv_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_indptr_offset);
    params.qo_lens =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.qo_len_offset);
    params.kv_lens =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.kv_len_offset);
    params.head_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.head_indices_offset);
    params.work_indptr =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.work_indptr_offset);
    params.batch_indices =
        GetPtrFromBaseOffset<IdType>(workspace_int, plan_info.batch_indices_offset);
}
#endif

#endif // USE_FLASHINFER

#ifdef USE_FLASHINFER
static inline bool IsSupportedDecodeGroupSize(uint32_t group_size) {
    return group_size == 1 || group_size == 2 || group_size == 3 || group_size == 4 || group_size == 5 ||
           group_size == 6 || group_size == 8 || group_size == 16 || group_size == 32 || group_size == 64;
}

static inline bool IsSupportedDecodeHeadDimForGroupSize(uint32_t group_size, uint32_t head_dim) {
    if (group_size == 64) {
        return head_dim <= 128;
    }
    return true;
}
#endif

#if defined(SM_90_PASS)
#define DISPATCH_HEAD_DIM_SM90(HEAD_DIM_VALUE, HEAD_DIM, ...) \
    if ((HEAD_DIM_VALUE) == 64) {                              \
        constexpr uint32_t HEAD_DIM = 64;                      \
        __VA_ARGS__;                                           \
    } else if ((HEAD_DIM_VALUE) == 128) {                      \
        constexpr uint32_t HEAD_DIM = 128;                     \
        __VA_ARGS__;                                           \
    } else if ((HEAD_DIM_VALUE) == 256) {                      \
        constexpr uint32_t HEAD_DIM = 256;                     \
        __VA_ARGS__;                                           \
    } else {                                                   \
        return;                                                \
    }
#endif
