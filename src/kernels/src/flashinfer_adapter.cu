#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
        // #if !defined(NO_FP8_KVCACHE)
        //     #include <flashinfer/attention/hopper/quantization/prefill_sm90.cuh>
        // #endif
    #endif

#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/page.cuh>

#if defined(SM_90_PASS)
#include <cutlass/numeric_types.h>
#endif
#include <flashinfer/pos_enc.cuh>
using namespace flashinfer;

#if !defined(SM_90_PASS)
template <bool use_custom_mask, bool use_sliding_window, bool use_logits_soft_cap, bool use_alibi>
using DefaultAttentionAlias =
    DefaultAttention<use_custom_mask, use_sliding_window, use_logits_soft_cap, use_alibi>;
using DefaultDecodeAttention = DefaultAttentionAlias<false, false, false, false>;
#else
template <bool use_custom_mask, bool use_sliding_window, bool use_logits_soft_cap, bool use_alibi>
using DefaultAttentionAlias = DefaultAttention<use_logits_soft_cap>;

// Decode kernels require the non-hopper attention variant interface, but the
// standard and hopper variant headers conflict if included together in one TU.
// Keep a local decode-only variant so SM90 builds can use FlashInfer decode
// planning/run (including split-kv metadata) without pulling standard variants.
struct DefaultDecodeAttention {
    static constexpr bool use_softmax = true;
    uint32_t kv_len;
    float sm_scale_log2;

    template <typename Params>
    __device__ __host__ DefaultDecodeAttention(const Params& params, uint32_t batch_idx,
                                               uint8_t* smem_ptr) {
        (void)smem_ptr;
        kv_len = params.get_kv_len(batch_idx);
        sm_scale_log2 = params.sm_scale * math::log2e;
    }

    template <typename Params, typename T>
    __device__ __forceinline__ T LogitsTransform(const Params& params, T logits, uint32_t batch_idx,
                                                 uint32_t qo_idx, uint32_t kv_idx,
                                                 uint32_t qo_head_idx, uint32_t kv_head_idx) {
        return logits;
    }

    template <typename Params>
    __device__ __forceinline__ bool LogitsMask(const Params& params, uint32_t batch_idx,
                                               uint32_t qo_idx, uint32_t kv_idx,
                                               uint32_t qo_head_idx, uint32_t kv_head_idx) {
        return true;
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
    params.kv_indices = indices;
}
#endif

#endif // Flashinfer

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
    // if (data_type == 2) {
    //     flashinfer_append_kv_cache_fp8(
    //         k_data_ptr,
    //         v_data_ptr,
    //         new_k_ptr,
    //         new_v_ptr,
    //         paged_kv_indices,
    //         paged_kv_indptr,
    //         paged_kv_last_len,
    //         batch_indices,
    //         positions,
    //         nnz,
    //         batch_size,
    //         num_heads,
    //         head_dim,
    //         page_size,
    //         k_scale_ptr,
    //         v_scale_ptr,
    //         is_input_f16,
    //         data_type,
    //         stream);
    //     return;
    // }

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
    if (page_locked_int_buffer == nullptr || page_locked_int_size < workspace_int_size) {
        return;
    }
    // if (data_type == 2) {
    //     flashinfer_decode_plan_wrapper_fp8(
    //         indptr_host,
    //         qo_indptr_host,
    //         kv_len_arr_host,
    //         batch_size,
    //         num_qo_heads,
    //         num_kv_heads,
    //         head_dim,
    //         page_size,
    //         workspace_float,
    //         workspace_float_size,
    //         workspace_int,
    //         workspace_int_size,
    //         page_locked_int_buffer,
    //         page_locked_int_size,
    //         enable_cuda_graph,
    //         data_type,
    //         out_data_type,
    //         plan_info_out,
    //         stream);
    //     return;
    // }
    auto run_plan = [&](auto dtype_kv_val) {
        using DTypeKV = decltype(dtype_kv_val);
        using DTypeQ = DTypeKV;
        using DTypeOut = DTypeKV;
        using IdType = int32_t;

        uint32_t group_size = num_qo_heads / num_kv_heads;

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
    int32_t data_type,
    int32_t out_data_type,
    cudaStream_t stream
) {
#ifdef USE_FLASHINFER
    const float rope_scale = 1.0f;
    const float rope_theta = 10000.0f;
    // if (data_type == 2) {
    //     flashinfer_decode_run_wrapper_fp8(
    //         out_ptr,
    //         q_ptr,
    //         k_data, v_data,
    //         indices,
    //         indptr,
    //         last_len,
    //         batch_size,
    //         num_qo_heads,
    //         num_kv_heads,
    //         head_dim,
    //         page_size,
    //         sm_scale,
    //         k_scale_ptr,
    //         v_scale_ptr,
    //         workspace_float,
    //         workspace_float_size,
    //         workspace_int,
    //         workspace_int_size,
    //         plan_info_vec,
    //         data_type,
    //         out_data_type,
    //         stream);
    //     return;
    // }
    if (plan_info_vec == nullptr) {
        fprintf(stderr, "[flashinfer][decode_run] plan_info_vec is null\n");
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

                using AttentionType = DefaultDecodeAttention;
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
    int32_t* kv_len_arr_host,   // Host pointer for kv lengths (fp8 sm90 plan)
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
    const float rope_scale = 1.0f;
    const float rope_theta = 10000.0f;
    // if (data_type == 2) {
    //     flashinfer_prefill_wrapper_fp8(
    //         out_ptr,
    //         q_ptr,
    //         q_cu_seqlens,
    //         q_cu_seqlens_host,
    //         kv_len_arr_host,
    //         total_num_rows,
    //         k_data, v_data,
    //         indices,
    //         indptr,
    //         indptr_host,
    //         last_len,
    //         batch_size,
    //         num_qo_heads,
    //         num_kv_heads,
    //         head_dim,
    //         page_size,
    //         sm_scale,
    //         k_scale_ptr,
    //         v_scale_ptr,
    //         workspace_float,
    //         workspace_float_size,
    //         workspace_int,
    //         workspace_int_size,
    //         page_locked_int_buffer,
    //         page_locked_int_size,
    //         enable_cuda_graph,
    //         data_type,
    //         out_data_type,
    //         stream);
    //     return;
    // }

#if defined(SM_90_PASS)
    if (page_locked_int_buffer == nullptr || page_locked_int_size < workspace_int_size) {
        return;
    }
    if (q_cu_seqlens_host == nullptr || indptr_host == nullptr || kv_len_arr_host == nullptr) {
        return;
    }
    {
        using IdType = int32_t;

        PrefillPlanSM90Info plan_info;
        PrefillSM90Plan<int32_t>(
            workspace_float, workspace_float_size,
            workspace_int, page_locked_int_buffer, workspace_int_size,
            plan_info,
            q_cu_seqlens_host, indptr_host, kv_len_arr_host,
            total_num_rows, batch_size,
            num_qo_heads, num_kv_heads, head_dim, head_dim, page_size,
            true /* causal */, enable_cuda_graph,
            (out_data_type == 1 ? sizeof(nv_bfloat16) : sizeof(half)),
            stream
        );

        auto run_sm90 = [&]() {
            if (out_data_type != data_type) {
                return;
            }
            auto run_non_fp8 = [&](auto dtype_val) {
                using DTypeKV = decltype(dtype_val);
                using DTypeQ = DTypeKV;
                using DTypeOut = DTypeKV;

                BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeOut, IdType> params;
                FillSM90PagedParams<DTypeQ, DTypeKV, DTypeOut, IdType>(
                    params, q_ptr, k_data, v_data, out_ptr,
                    num_qo_heads, num_kv_heads, head_dim, page_size,
                    total_num_rows, sm_scale, indices, workspace_int, plan_info);

                using AttentionType = DefaultAttentionAlias<false, false, false, false>;
                DISPATCH_HEAD_DIM_SM90(head_dim, HEAD_DIM, {
                    if (plan_info.same_schedule_for_all_heads) {
                        BatchPrefillWithPagedKVCacheDispatched<
                            HEAD_DIM, HEAD_DIM, MaskMode::kCausal, false, true, AttentionType>(
                            params, false, stream);
                    } else {
                        BatchPrefillWithPagedKVCacheDispatched<
                            HEAD_DIM, HEAD_DIM, MaskMode::kCausal, false, false, AttentionType>(
                            params, false, stream);
                    }
                });
            };

            if (data_type == 1) {
                run_non_fp8(cutlass::bfloat16_t{});
            } else {
                run_non_fp8(cutlass::half_t{});
            }
        };

        run_sm90();
    }
    return;
#else
    if (page_locked_int_buffer == nullptr || page_locked_int_size < workspace_int_size) {
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
            
            using AttentionType = DefaultAttentionAlias<false, false, false, false>;

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
#endif
}

}
