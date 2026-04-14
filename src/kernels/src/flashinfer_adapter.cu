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

// Decode kernels require the non-hopper attention variant interface, but the
// standard and hopper variant headers conflict if included together in one TU.
// Keep a local decode-only variant so SM90 builds can use FlashInfer decode
// planning/run (including split-kv metadata) without pulling standard variants.
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

#endif // Flashinfer

#ifdef USE_FLASHINFER
static inline bool IsSupportedDecodeGroupSize(uint32_t group_size) {
    return group_size == 1 || group_size == 2 || group_size == 3 || group_size == 4 || group_size == 5 ||
           group_size == 6 || group_size == 8 || group_size == 16 || group_size == 32 || group_size == 64;
}

static inline bool IsSupportedDecodeHeadDimForGroupSize(uint32_t group_size, uint32_t head_dim) {
    // group_size=64 can exceed 1024 threads for HEAD_DIM=256 in decode kernels.
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
void flashinfer_fp8_quantize_kv_scalar(const void* k_in, const void* v_in,
                                       void* k_out, void* v_out, int64_t numel,
                                       const float* k_scale, const float* v_scale,
                                       bool is_input_f16, int64_t stream_);
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
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    bool is_input_f16,
    int32_t data_type,
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    if (data_type == 2) {
        #if defined(FLASHINFER_ENABLE_FP8_E4M3)
        if (!k_scale_ptr || !v_scale_ptr) {
            fprintf(stderr, "[flashinfer][append_kv_fp8] k_scale or v_scale is null\n");
            return;
        }
        extern void flashinfer_fp8_quantize_kv_per_head(
            const void*, const void*, void*, void*, int64_t,
            int, int, const float*, const float*, bool, int64_t);
        void* k_fp8_ptr = nullptr;
        void* v_fp8_ptr = nullptr;
        int64_t numel = static_cast<int64_t>(nnz) * num_heads * head_dim;
        if (cudaMallocAsync(&k_fp8_ptr, static_cast<size_t>(numel) * sizeof(uint8_t), stream) != cudaSuccess ||
            cudaMallocAsync(&v_fp8_ptr, static_cast<size_t>(numel) * sizeof(uint8_t), stream) != cudaSuccess) {
            fprintf(stderr, "[flashinfer][append_kv_fp8] cudaMallocAsync failed\n");
            if (k_fp8_ptr) cudaFreeAsync(k_fp8_ptr, stream);
            if (v_fp8_ptr) cudaFreeAsync(v_fp8_ptr, stream);
            return;
        }
        flashinfer_fp8_quantize_kv_per_head(
            new_k_ptr, new_v_ptr, k_fp8_ptr, v_fp8_ptr, numel,
            num_heads, head_dim, k_scale_ptr, v_scale_ptr, is_input_f16, (int64_t)stream
        );
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[flashinfer][append_kv_fp8] KV quantize launch error: %s\n", cudaGetErrorString(err));
                fflush(stderr);
            }
        }

        paged_kv_t<uint8_t, int32_t> paged_kv(
            num_heads, page_size, head_dim, batch_size, QKVLayout::kNHD,
            (uint8_t*)k_data_ptr, (uint8_t*)v_data_ptr,
            paged_kv_indices, paged_kv_indptr, paged_kv_last_len
        );
        if (batch_size > 0 && batch_indices && positions) {
            size_t stride_n = num_heads * head_dim;
            size_t stride_h = head_dim;
            AppendPagedKVCache(
                paged_kv, (uint8_t*)k_fp8_ptr, (uint8_t*)v_fp8_ptr,
                batch_indices, positions, nnz,
                stride_n, stride_h, stride_n, stride_h, stream
            );
        } else {
            AppendPagedKVCacheDecode(paged_kv, (uint8_t*)k_fp8_ptr, (uint8_t*)v_fp8_ptr, stream);
        }
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[flashinfer][append_kv_fp8] append launch error: %s\n", cudaGetErrorString(err));
                fflush(stderr);
            }
        }

        if (k_fp8_ptr) cudaFreeAsync(k_fp8_ptr, stream);
        if (v_fp8_ptr) cudaFreeAsync(v_fp8_ptr, stream);
        #endif
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
    int32_t* qo_indptr_host,   // Host pointer for fp8 decode planning (optional)
    int32_t* kv_len_arr_host,  // Host pointer for fp8 decode planning (optional)
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
    int32_t out_data_type,
    int64_t* plan_info_out,     // length 10
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    if (num_kv_heads <= 0 || num_qo_heads <= 0 || (num_qo_heads % num_kv_heads) != 0) {
        fprintf(stderr,
                "[flashinfer][decode_plan] invalid head config qo_heads=%d kv_heads=%d\n",
                num_qo_heads, num_kv_heads);
        return;
    }
    uint32_t group_size = static_cast<uint32_t>(num_qo_heads / num_kv_heads);
    if (!IsSupportedDecodeGroupSize(group_size)) {
        fprintf(stderr,
                "[flashinfer][decode_plan] unsupported group_size=%u (supported: 1,2,3,4,6,8,16,32,64)\n",
                group_size);
        return;
    }
    if (!IsSupportedDecodeHeadDimForGroupSize(group_size, static_cast<uint32_t>(head_dim))) {
        fprintf(stderr,
                "[flashinfer][decode_plan] unsupported combination group_size=%u head_dim=%d (group_size=64 requires head_dim<=128)\n",
                group_size, head_dim);
        return;
    }
    if (page_locked_int_buffer == nullptr || page_locked_int_size < workspace_int_size) {
        return;
    }
    if (data_type == 2) {
        #if defined(FLASHINFER_ENABLE_FP8_E4M3)
        auto run_plan_fp8 = [&](auto dtype_q_val) {
            using DTypeQ = decltype(dtype_q_val);
            using DTypeKV = uint8_t;
            using DTypeOut = DTypeQ;
            using IdType = int32_t;

            DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, {
                    using AttentionType = DefaultDecodeAttention;
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

        if (out_data_type == 1) {
            run_plan_fp8(nv_bfloat16{});
        } else {
            run_plan_fp8(half{});
        }
        #endif
        return;
    }
    auto run_plan = [&](auto dtype_kv_val) {
        using DTypeKV = decltype(dtype_kv_val);
        using DTypeQ = DTypeKV;
        using DTypeOut = DTypeKV;
        using IdType = int32_t;

        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
            DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, {
                using AttentionType = DefaultDecodeAttention;
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
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    const int64_t* plan_info_vec, // length 10
    int32_t window_left,
    float logits_soft_cap,
    int32_t data_type,
    int32_t out_data_type,
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    if (num_kv_heads <= 0 || num_qo_heads <= 0 || (num_qo_heads % num_kv_heads) != 0) {
        fprintf(stderr,
                "[flashinfer][decode_run] invalid head config qo_heads=%d kv_heads=%d\n",
                num_qo_heads, num_kv_heads);
        return;
    }
    uint32_t group_size = static_cast<uint32_t>(num_qo_heads / num_kv_heads);
    if (!IsSupportedDecodeGroupSize(group_size)) {
        fprintf(stderr,
                "[flashinfer][decode_run] unsupported group_size=%u (supported: 1,2,3,4,6,8,16,32,64)\n",
                group_size);
        return;
    }
    if (!IsSupportedDecodeHeadDimForGroupSize(group_size, static_cast<uint32_t>(head_dim))) {
        fprintf(stderr,
                "[flashinfer][decode_run] unsupported combination group_size=%u head_dim=%d (group_size=64 requires head_dim<=128)\n",
                group_size, head_dim);
        return;
    }
    const float rope_scale = 1.0f;
    const float rope_theta = 10000.0f;
    if (data_type == 2) {
        #if defined(FLASHINFER_ENABLE_FP8_E4M3)
        if (plan_info_vec == nullptr) {
            fprintf(stderr, "[flashinfer][decode_run] plan_info_vec is null\n");
            return;
        }
        auto run_decode_fp8 = [&](auto dtype_q_val) {
            using DTypeQ = decltype(dtype_q_val);
            using DTypeKV = uint8_t;
            using DTypeOut = DTypeQ;
            using IdType = int32_t;

            DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                paged_kv_t<DTypeKV, IdType> paged_kv(
                    num_kv_heads, page_size, head_dim, batch_size, QKVLayout::kNHD,
                    (DTypeKV*)k_data, (DTypeKV*)v_data,
                    indices, indptr, last_len
                );

                DecodePlanInfo plan_info;
                std::vector<int64_t> vec(plan_info_vec, plan_info_vec + 10);
                plan_info.FromVector(vec);

                using AttentionType = DefaultDecodeAttention;
                using ParamsType = BatchDecodeParams<DTypeQ, DTypeKV, DTypeOut, IdType>;

                ParamsType params(
                    (DTypeQ*)q_ptr, nullptr /* q_rope_offset */, paged_kv, (DTypeOut*)out_ptr,
                    nullptr /* lse */, nullptr /* alibi */, num_qo_heads,
                    num_qo_heads * head_dim /* q_stride_n */, head_dim /* q_stride_h */,
                    window_left > 0 ? window_left : -1 /* window_left */, logits_soft_cap /* logits_cap */, sm_scale, rope_scale, rope_theta
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
        };

        if (out_data_type == 1) {
            run_decode_fp8(nv_bfloat16{});
        } else {
            run_decode_fp8(half{});
        }
        #endif
        return;
    }
    if (plan_info_vec == nullptr) {
        fprintf(stderr, "[flashinfer][decode_run] plan_info_vec is null\n");
        return;
    }
    auto run_decode = [&](auto dtype_kv_val) {
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

            DecodePlanInfo plan_info;
            std::vector<int64_t> vec(plan_info_vec, plan_info_vec + 10);
            plan_info.FromVector(vec);

            using AttentionType = DefaultDecodeAttention;
            using ParamsType = BatchDecodeParams<DTypeQ, DTypeKV, DTypeOut, IdType>;

            ParamsType params(
                (DTypeQ*)q_ptr, nullptr /* q_rope_offset */, paged_kv, (DTypeOut*)out_ptr,
                nullptr /* lse */, nullptr /* alibi */, num_qo_heads,
                num_qo_heads * head_dim /* q_stride_n */, head_dim /* q_stride_h */,
                window_left > 0 ? window_left : -1 /* window_left */, logits_soft_cap /* logits_cap */, sm_scale, rope_scale, rope_theta
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
    };

    if (data_type == 1) {
        run_decode(nv_bfloat16{});
    } else {
        run_decode(half{});
    }
#endif
}

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
            true /* causal */, enable_cuda_graph,
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
        // FA2 path for FP8 prefill (Blackwell and other non-Hopper SM89+ GPUs)
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
