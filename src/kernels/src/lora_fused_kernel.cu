// Fused batched LoRA shrink/expand GEMM for multi-tenant serving.
//
// This is the attention-rs side of the xinfer LoRA fused kernel. It computes the
// low-rank delta `delta = scale * (x @ A) @ B` for a batch of rows, each row
// may belong to a different LoRA tenant (slot), in two fused phases over a
// COMPACTED set of the slots:
//
//   shrink:  buf[row, s, k] = scale[slot[s]] * sum_c x[row, c] * A[slot[s], c, k]
//   expand:  out[row, o]    = sum_k buf[row, s, k] * B[slot[s], k, o]
//
// where `s = row_slot[row]` is the compacted slot index for that row (0..num_active),
// and `slot[s]` is the actual bank slot. The compaction means the scratch is
// `[rows, num_active, rank]` (only the active tenants, not the full bank), which
// is the vLLM Punica layout.
//
// The `slot_map` (u32, per row) holds the compacted slot index, with a
// sentinel (the `LORA_FUSED_BASE_SLOT` = 0xFFFFFFFF) for base-model rows (zero
// delta). The `active_slots` (u32, length num_active) maps compacted index ->
// actual bank slot. The A/B are the stacked banks: A is `[num_slots, in_features,
// rank]`, B is `[num_slots, rank, out_features]`. The `scale` is `[num_slots]`
// (f32).
//
// The accumulation is f32 (no precision loss in the low-rank product), matching
// the vLLM Punica choice. The x/A/B may be f16/bf16/f32 (the model compute
// dtype); the `out` is written in its own dtype.
//
// One kernel launch does the shrink for all rows; a second does the expand. The
// scratch `buf` is `[rows, num_active, rank]` f32, pre-allocated by the caller
// (the xinfer side owns the scratch pool). The `out` is the fresh delta tensor,
// pre-allocated by the caller (the xinfer side adds it to the base output).

#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>

// The sentinel slot_map value for a base-model row (no LoRA delta).
#define LORA_FUSED_BASE_SLOT 0xFFFFFFFFu

template <typename T>
__global__ void lora_shrink_kernel(
    const T* __restrict__ x,        // [rows, in_features]
    const T* __restrict__ a,        // [num_slots, in_features, rank]
    const float* __restrict__ scale, // [num_slots]
    const unsigned int* __restrict__ slot_map, // [rows] (compacted slot, or the sentinel)
    const unsigned int* __restrict__ active_slots, // [num_active] (compacted -> bank slot)
    float* __restrict__ buf,        // [rows, num_active, rank] (f32 scratch)
    const int rows, const int in_features, const int rank, const int num_active)
{
    // One thread per (row, rank) element; the compacted slot is looked up per row.
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || k >= rank) return;

    const unsigned int s = slot_map[row];
    if (s == LORA_FUSED_BASE_SLOT) {
        // The base-model row: zero this buf entry (the expand skips it via the slot_map).
        buf[((int64_t)row * num_active) * rank + k] = 0.0f;
        return;
    }
    const int bank_slot = (int)active_slots[s];
    const T* x_row = x + (int64_t)row * in_features;
    const T* a_slot = a + ((int64_t)bank_slot * in_features) * rank + k;
    float acc = 0.0f;
    for (int c = 0; c < in_features; c++) {
        acc += (float)x_row[c] * (float)a_slot[(int64_t)c * rank];
    }
    buf[((int64_t)row * num_active + s) * rank + k] = acc * scale[bank_slot];
}

template <typename T>
__global__ void lora_expand_kernel(
    const float* __restrict__ buf,   // [rows, num_active, rank] (f32 scratch)
    const T* __restrict__ b,         // [num_slots, rank, out_features]
    const unsigned int* __restrict__ slot_map, // [rows] (compacted slot, or the SENTINEL)
    const unsigned int* __restrict__ active_slots, // [num_active]
    T* __restrict__ out,            // [rows, out_features] (the delta, written fresh)
    const int rows, const int rank, const int out_features, const int num_active)
{
    // One thread per (row, out) element; the compacted slot is looked up per row.
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || o >= out_features) return;

    const unsigned int s = slot_map[row];
    if (s == LORA_FUSED_BASE_SLOT) {
        // The base-model row: the delta is zero.
        out[(int64_t)row * out_features + o] = (T)0.0f;
        return;
    }

    const int bank_slot = (int)active_slots[s];
    const float* buf_row = buf + (int64_t)row * num_active * rank + (int64_t)s * rank;
    const T* b_slot = b + ((int64_t)bank_slot * rank) * out_features + o;
    float acc = 0.0f;
    for (int r = 0; r < rank; r++) {
        acc += buf_row[r] * (float)b_slot[(int64_t)r * out_features];
    }
    out[(int64_t)row * out_features + o] = (T)acc;
}

// ---- The FFI launchers (the one per dtype) ---------------------------------
// The signature carries the compacted `slot_map` + the `active_slots` mapping +
// the `num_active` count (not the full `num_slots`), which is what makes the
// scratch compact.

extern "C" void lora_shrink_f16(
    const void* x, const void* a, const void* scale, const void* slot_map,
    const void* active_slots, void* buf,
    const int rows, const int in_features, const int rank, const int num_active,
    cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((rows + block.x - 1) / block.x, (rank + block.y - 1) / block.y);
    lora_shrink_kernel<__half><<<grid, block, 0, stream>>>(
        (const __half*)x, (const __half*)a, (const float*)scale,
        (const unsigned int*)slot_map, (const unsigned int*)active_slots,
        (float*)buf, rows, in_features, rank, num_active);
}

extern "C" void lora_shrink_bf16(
    const void* x, const void* a, const void* scale, const void* slot_map,
    const void* active_slots, void* buf,
    const int rows, const int in_features, const int rank, const int num_active,
    cudaStream_t stream)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const dim3 block(16, 16);
    const dim3 grid((rows + block.x - 1) / block.x, (rank + block.y - 1) / block.y);
    lora_shrink_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)a, (const float*)scale,
        (const unsigned int*)slot_map, (const unsigned int*)active_slots,
        (float*)buf, rows, in_features, rank, num_active);
#endif
}

extern "C" void lora_shrink_f32(
    const void* x, const void* a, const void* scale, const void* slot_map,
    const void* active_slots, void* buf,
    const int rows, const int in_features, const int rank, const int num_active,
    cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((rows + block.x - 1) / block.x, (rank + block.y - 1) / block.y);
    lora_shrink_kernel<float><<<grid, block, 0, stream>>>(
        (const float*)x, (const float*)a, (const float*)scale,
        (const unsigned int*)slot_map, (const unsigned int*)active_slots,
        (float*)buf, rows, in_features, rank, num_active);
}

extern "C" void lora_expand_f16(
    const void* buf, const void* b, const void* slot_map, const void* active_slots,
    void* out,
    const int rows, const int rank, const int out_features, const int num_active,
    cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((rows + block.x - 1) / block.x, (out_features + block.y - 1) / block.y);
    lora_expand_kernel<__half><<<grid, block, 0, stream>>>(
        (const float*)buf, (const __half*)b, (const unsigned int*)slot_map,
        (const unsigned int*)active_slots, (__half*)out, rows, rank, out_features, num_active);
}

extern "C" void lora_expand_bf16(
    const void* buf, const void* b, const void* slot_map, const void* active_slots,
    void* out,
    const int rows, const int rank, const int out_features, const int num_active,
    cudaStream_t stream)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const dim3 block(16, 16);
    const dim3 grid((rows + block.x - 1) / block.x, (out_features + block.y - 1) / block.y);
    lora_expand_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        (const float*)buf, (const __nv_bfloat16*)b, (const unsigned int*)slot_map,
        (const unsigned int*)active_slots, (__nv_bfloat16*)out, rows, rank, out_features, num_active);
#endif
}

extern "C" void lora_expand_f32(
    const void* buf, const void* b, const void* slot_map, const void* active_slots,
    void* out,
    const int rows, const int rank, const int out_features, const int num_active,
    cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((rows + block.x - 1) / block.x, (out_features + block.y - 1) / block.y);
    lora_expand_kernel<float><<<grid, block, 0, stream>>>(
        (const float*)buf, (const float*)b, (const unsigned int*)slot_map,
        (const unsigned int*)active_slots, (float*)out, rows, rank, out_features, num_active);
}

// ============================================================================
// Stacked (additive multi-adapter) shrink/expand.
//
// A row may carry a LIST of compacted slots (the stacked adapters). The CSR
// layout:
//   slot_offsets: [rows+1] u32  -- row r's slots are slot_ids[slot_offsets[r] ..
//                                   slot_offsets[r+1]]
//   slot_ids:     [total] u32   -- compacted slot index per (row, adapter) pair
//   active_slots: [num_active] u32 -- compacted index -> bank slot
//
// The buf is [rows, max_stack, rank] f32; only the first stack_count[row]
// entries along the middle dim are written by the shrink + read by the expand.
// The shrink computes, for each (row, j) pair:
//   buf[row, j, k] = scale[bank_slot] * sum_c x[row, c] * A[bank_slot, c, k]
// where bank_slot = active_slots[slot_ids[slot_offsets[row] + j]].
// The expand sums over the row's stack:
//   out[row, o] = sum_j sum_k buf[row, j, k] * B[bank_slot(j), k, o]
// which is the additive composition base + sum_j scaling_j * (x @ A_j) @ B_j.
// ============================================================================

template <typename T>
__global__ void lora_shrink_stacked_kernel(
    const T* __restrict__ x,        // [rows, in_features]
    const T* __restrict__ a,        // [num_slots, in_features, rank]
    const float* __restrict__ scale, // [num_slots]
    const unsigned int* __restrict__ slot_offsets, // [rows+1]
    const unsigned int* __restrict__ slot_ids,     // [total]
    const unsigned int* __restrict__ active_slots, // [num_active]
    float* __restrict__ buf,        // [rows, max_stack, rank]
    const int rows, const int in_features, const int rank, const int max_stack)
{
    // One thread per (row, max, k) element; m < max_stack; the (row, m) pair
    // is valid only when m < the row's actual stack count.
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (row >= rows || m >= max_stack || k >= rank) return;

    const int stack_count = (int)(slot_offsets[row + 1] - slot_offsets[row]);
    if (m >= stack_count) return; // beyond this row's actual stack

    const unsigned int compact = slot_ids[slot_offsets[row] + m];
    const int bank_slot = (int)active_slots[compact];
    const T* x_row = x + (int64_t)row * in_features;
    const T* a_slot = a + ((int64_t)bank_slot * in_features) * rank + k;
    float acc = 0.0f;
    for (int c = 0; c < in_features; c++) {
        acc += (float)x_row[c] * (float)a_slot[(int64_t)c * rank];
    }
    buf[((int64_t)row * max_stack + m) * rank + k] = acc * scale[bank_slot];
}

template <typename T>
__global__ void lora_expand_stacked_kernel(
    const float* __restrict__ buf,   // [rows, max_stack, rank]
    const T* __restrict__ b,         // [num_slots, rank, out_features]
    const unsigned int* __restrict__ slot_offsets, // [rows+1]
    const unsigned int* __restrict__ slot_ids,     // [total]
    const unsigned int* __restrict__ active_slots, // [num_active]
    T* __restrict__ out,             // [rows, out_features]
    const int rows, const int rank, const int out_features, const int max_stack)
{
    // One thread per (row, out) element; the row's stack is summed in-kernel.
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || o >= out_features) return;

    const int stack_count = (int)(slot_offsets[row + 1] - slot_offsets[row]);
    float acc = 0.0f;
    for (int m = 0; m < stack_count; m++) {
        const unsigned int compact = slot_ids[slot_offsets[row] + m];
        const int bank_slot = (int)active_slots[compact];
        const float* buf_row = buf + ((int64_t)row * max_stack + m) * rank;
        const T* b_slot = b + ((int64_t)bank_slot * rank) * out_features + o;
        for (int r = 0; r < rank; r++) {
            acc += buf_row[r] * (float)b_slot[(int64_t)r * out_features];
        }
    }
    out[(int64_t)row * out_features + o] = (T)acc;
}

// ---- The stacked FFI launchers (the one per dtype) -------------------------

extern "C" void lora_shrink_stacked_f16(
    const void* x, const void* a, const void* scale,
    const void* slot_offsets, const void* slot_ids, const void* active_slots,
    void* buf,
    const int rows, const int in_features, const int rank, const int max_stack,
    cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid(
        (rows + block.x - 1) / block.x,
        (max_stack + block.y - 1) / block.y,
        (rank + block.z - 1) / block.z);
    lora_shrink_stacked_kernel<__half><<<grid, block, 0, stream>>>(
        (const __half*)x, (const __half*)a, (const float*)scale,
        (const unsigned int*)slot_offsets, (const unsigned int*)slot_ids,
        (const unsigned int*)active_slots, (float*)buf,
        rows, in_features, rank, max_stack);
}

extern "C" void lora_shrink_stacked_bf16(
    const void* x, const void* a, const void* scale,
    const void* slot_offsets, const void* slot_ids, const void* active_slots,
    void* buf,
    const int rows, const int in_features, const int rank, const int max_stack,
    cudaStream_t stream)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const dim3 block(16, 16, 1);
    const dim3 grid(
        (rows + block.x - 1) / block.x,
        (max_stack + block.y - 1) / block.y,
        (rank + block.z - 1) / block.z);
    lora_shrink_stacked_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)a, (const float*)scale,
        (const unsigned int*)slot_offsets, (const unsigned int*)slot_ids,
        (const unsigned int*)active_slots, (float*)buf,
        rows, in_features, rank, max_stack);
#endif
}

extern "C" void lora_shrink_stacked_f32(
    const void* x, const void* a, const void* scale,
    const void* slot_offsets, const void* slot_ids, const void* active_slots,
    void* buf,
    const int rows, const int in_features, const int rank, const int max_stack,
    cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid(
        (rows + block.x - 1) / block.x,
        (max_stack + block.y - 1) / block.y,
        (rank + block.z - 1) / block.z);
    lora_shrink_stacked_kernel<float><<<grid, block, 0, stream>>>(
        (const float*)x, (const float*)a, (const float*)scale,
        (const unsigned int*)slot_offsets, (const unsigned int*)slot_ids,
        (const unsigned int*)active_slots, (float*)buf,
        rows, in_features, rank, max_stack);
}

extern "C" void lora_expand_stacked_f16(
    const void* buf, const void* b,
    const void* slot_offsets, const void* slot_ids, const void* active_slots,
    void* out,
    const int rows, const int rank, const int out_features, const int max_stack,
    cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((rows + block.x - 1) / block.x, (out_features + block.y - 1) / block.y);
    lora_expand_stacked_kernel<__half><<<grid, block, 0, stream>>>(
        (const float*)buf, (const __half*)b,
        (const unsigned int*)slot_offsets, (const unsigned int*)slot_ids,
        (const unsigned int*)active_slots, (__half*)out,
        rows, rank, out_features, max_stack);
}

extern "C" void lora_expand_stacked_bf16(
    const void* buf, const void* b,
    const void* slot_offsets, const void* slot_ids, const void* active_slots,
    void* out,
    const int rows, const int rank, const int out_features, const int max_stack,
    cudaStream_t stream)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const dim3 block(16, 16);
    const dim3 grid((rows + block.x - 1) / block.x, (out_features + block.y - 1) / block.y);
    lora_expand_stacked_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        (const float*)buf, (const __nv_bfloat16*)b,
        (const unsigned int*)slot_offsets, (const unsigned int*)slot_ids,
        (const unsigned int*)active_slots, (__nv_bfloat16*)out,
        rows, rank, out_features, max_stack);
#endif
}

extern "C" void lora_expand_stacked_f32(
    const void* buf, const void* b,
    const void* slot_offsets, const void* slot_ids, const void* active_slots,
    void* out,
    const int rows, const int rank, const int out_features, const int max_stack,
    cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((rows + block.x - 1) / block.x, (out_features + block.y - 1) / block.y);
    lora_expand_stacked_kernel<float><<<grid, block, 0, stream>>>(
        (const float*)buf, (const float*)b,
        (const unsigned int*)slot_offsets, (const unsigned int*)slot_ids,
        (const unsigned int*)active_slots, (float*)out,
        rows, rank, out_features, max_stack);
}