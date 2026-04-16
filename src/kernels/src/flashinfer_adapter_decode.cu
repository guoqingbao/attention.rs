#include "flashinfer_common.cuh"

#if defined(FLASHINFER_ENABLE_FP8_E4M3)
extern "C" {
void flashinfer_fp8_quantize_kv_scalar(const void* k_in, const void* v_in,
                                       void* k_out, void* v_out, int64_t numel,
                                       const float* k_scale, const float* v_scale,
                                       bool is_input_f16, int64_t stream_);
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
    int32_t* batch_indices,
    int32_t* positions,
    int32_t nnz,
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
             size_t stride_n = num_heads * head_dim;
             size_t stride_h = head_dim;
             
             AppendPagedKVCache(paged_kv, (DType*)new_k_ptr, (DType*)new_v_ptr,
                                batch_indices, positions, nnz,
                                stride_n, stride_h, stride_n, stride_h, 
                                stream);
        } else {
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
    int32_t* indptr_host,
    int32_t* qo_indptr_host,
    int32_t* kv_len_arr_host,
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
    int64_t* plan_info_out,
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
    void* workspace_float,
    size_t workspace_float_size,
    void* workspace_int,
    size_t workspace_int_size,
    const int64_t* plan_info_vec,
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
                    (DTypeQ*)q_ptr, nullptr, paged_kv, (DTypeOut*)out_ptr,
                    nullptr, nullptr, num_qo_heads,
                    num_qo_heads * head_dim, head_dim,
                    window_left > 0 ? window_left : -1, logits_soft_cap, sm_scale, rope_scale, rope_theta
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
                        params, tmp_v, tmp_s, false, stream
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
                (DTypeQ*)q_ptr, nullptr, paged_kv, (DTypeOut*)out_ptr,
                nullptr, nullptr, num_qo_heads,
                num_qo_heads * head_dim, head_dim,
                window_left > 0 ? window_left : -1, logits_soft_cap, sm_scale, rope_scale, rope_theta
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
                    params, tmp_v, tmp_s, false, stream
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

}
