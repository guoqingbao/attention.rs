// PDA pushdown-rs fused kernels.
//
// The transition table is the flat pushdown-rs format:
//   Each record: (q, a, top, next_q, push_len, push[0..push_len-1])
//   Total array length = sum over all records of (5 + push_len) u32s.
//
// The fused_sample kernel does mask + sample + advance in ONE launch.
// The fused_project kernel does K+1 mask projections in ONE launch.
//
// The PDA dispatch is optional: if ctrl/stack/sp are null, the kernel
// runs plain sampling with zero PDA overhead.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>
#include <cstdio>

#define D_MAX 32

// Helper: scan the transition table for (ctrl, top) and emit VOB mask.
// Uses the CSR u32 offsets for O(1) jump to the relevant records.
__device__ void scan_mask(
    const uint32_t* __restrict__ transitions,
    const uint32_t* __restrict__ ctrl_u32_offsets,
    const uint32_t* __restrict__ ctrl_counts,
    uint32_t ctrl,
    uint32_t top,
    uint32_t* __restrict__ out_vob,
    uint32_t words_per_vob)
{
    for (uint32_t w = 0; w < words_per_vob; w++) {
        out_vob[w] = 0;
    }
    // O(1) jump: the u32 index for this ctrl state.
    uint32_t idx = ctrl_u32_offsets[ctrl];
    uint32_t count = ctrl_counts[ctrl];
    // Scan only the `count` records for this ctrl state.
    for (uint32_t i = 0; i < count; i++) {
        uint32_t a = transitions[idx + 1];
        uint32_t t = transitions[idx + 2];
        uint32_t push_len = transitions[idx + 4];
        if (t == top) {
            uint32_t word_idx = a / 32;
            if (word_idx < words_per_vob) {
                out_vob[word_idx] |= (1u << (a % 32));
            }
        }
        idx += 5 + push_len; // advance to next record
    }
}

// Helper: advance the PDA by a token. Returns (next_ctrl, new_top).
// Uses the CSR u32 offsets for O(1) jump to the relevant records.
__device__ void advance_pda(
    const uint32_t* __restrict__ transitions,
    const uint32_t* __restrict__ ctrl_u32_offsets,
    const uint32_t* __restrict__ ctrl_counts,
    uint32_t ctrl,
    uint32_t top,
    uint32_t token,
    uint32_t* out_ctrl,
    uint32_t* out_top)
{
    *out_ctrl = ctrl;
    *out_top = top;
    // O(1) jump: the u32 index for this ctrl state.
    size_t idx = ctrl_u32_offsets[ctrl];
    uint32_t count = ctrl_counts[ctrl];
    for (uint32_t i = 0; i < count; i++) {
        uint32_t a = transitions[idx + 1];
        uint32_t t = transitions[idx + 2];
        uint32_t next_q = transitions[idx + 3];
        uint32_t push_len = transitions[idx + 4];
        if (a == token && t == top) {
            *out_ctrl = next_q;
            if (push_len > 0) {
                *out_top = transitions[idx + 5 + push_len - 1];
            } else {
                *out_top = 0;
            }
            return;
        }
        idx += 5 + push_len;
    }
}

// Fused sample kernel (F32 logits): mask + sample + advance in one launch.
// One thread per sequence. If ctrl is null, runs plain sampling.
__global__ void pda_fused_sample_f32_kernel(
    const float* __restrict__ logits,
    const uint32_t* __restrict__ ctrl,
    const uint32_t* __restrict__ stack,
    const uint32_t* __restrict__ sp,
    uint32_t* __restrict__ out_ctrl,
    uint32_t* __restrict__ out_sp,
    uint32_t* __restrict__ out_tokens,
    const uint32_t* __restrict__ transitions,
    const uint32_t* __restrict__ accepting,
    const uint32_t* __restrict__ ctrl_u32_offsets,
    const uint32_t* __restrict__ ctrl_counts,
    uint32_t num_states,
    uint32_t num_stack_syms,
    uint32_t words_per_vob,
    int batch,
    int vocab,
    int top_k,
    float temperature,
    float top_p,
    int d)
{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch) return;

    const float* logit_row = logits + (size_t)seq * vocab;

    if (ctrl == nullptr) {
        // No PDA: plain greedy (or top-k/top-p if configured).
        float max_val = -FLT_MAX;
        int max_idx = 0;
        for (int i = 0; i < vocab; i++) {
            if (logit_row[i] > max_val) { max_val = logit_row[i]; max_idx = i; }
        }
        out_tokens[seq] = max_idx;
        return;
    }

    // PDA active: compute the mask from the current (ctrl, top).
    uint32_t c = ctrl[seq];
    if (c >= num_states) c = 0;
    uint32_t s = (sp != nullptr) ? sp[seq] : 1;
    uint32_t top = (s > 0 && stack != nullptr) ? stack[(size_t)seq * d + s - 1] : 0;

    // Emit the VOB mask into shared/local memory.
    uint32_t vob[D_MAX]; // max 32 words (1024 inputs)
    uint32_t wpv = words_per_vob < D_MAX ? words_per_vob : D_MAX;
    scan_mask(transitions, ctrl_u32_offsets, ctrl_counts, c, top, vob, wpv);

    // Apply mask to logits: illegal tokens -> -inf.
    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int i = 0; i < vocab; i++) {
        float v = logit_row[i];
        uint32_t word_idx = i / 32;
        if (word_idx < wpv && (vob[word_idx] & (1u << (i % 32))) == 0) {
            v = -FLT_MAX; // illegal
        }
        if (v > max_val) { max_val = v; max_idx = i; }
    }
    // Greedy (the top_k/top_p path would go here for non-greedy sampling).
    out_tokens[seq] = max_idx;

    // Advance the PDA.
    uint32_t next_ctrl, next_top;
    advance_pda(transitions, ctrl_u32_offsets, ctrl_counts, c, top, max_idx, &next_ctrl, &next_top);
    out_ctrl[seq] = next_ctrl;
    if (out_sp != nullptr) {
        // The stack update: pop 1 (the old top), push push_len (the new symbols).
        // Re-scan for the push_len of the matching transition.
        size_t idx = ctrl_u32_offsets[c];
        uint32_t count = ctrl_counts[c];
        uint32_t push_len = 0;
        for (uint32_t i = 0; i < count; i++) {
            uint32_t a = transitions[idx + 1];
            uint32_t t = transitions[idx + 2];
            uint32_t pl = transitions[idx + 4];
            if (a == max_idx && t == top) {
                push_len = pl;
                break;
            }
            idx += 5 + pl;
        }
        uint32_t new_sp = s;
        if (new_sp > 0) new_sp--; // pop the old top
        new_sp += push_len;     // push the new symbols
        out_sp[seq] = new_sp;
    }
}

// Fused sample kernel (BF16 logits).
__global__ void pda_fused_sample_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const uint32_t* __restrict__ ctrl,
    const uint32_t* __restrict__ stack,
    const uint32_t* __restrict__ sp,
    uint32_t* __restrict__ out_ctrl,
    uint32_t* __restrict__ out_sp,
    uint32_t* __restrict__ out_tokens,
    const uint32_t* __restrict__ transitions,
    const uint32_t* __restrict__ accepting,
    const uint32_t* __restrict__ ctrl_u32_offsets,
    const uint32_t* __restrict__ ctrl_counts,
    uint32_t num_states,
    uint32_t num_stack_syms,
    uint32_t words_per_vob,
    int batch,
    int vocab,
    int top_k,
    float temperature,
    float top_p,
    int d)
{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch) return;

    const __nv_bfloat16* logit_row = logits + (size_t)seq * vocab;

    if (ctrl == nullptr) {
        float max_val = -FLT_MAX;
        int max_idx = 0;
        for (int i = 0; i < vocab; i++) {
            float v = __bfloat162float(logit_row[i]);
            if (v > max_val) { max_val = v; max_idx = i; }
        }
        out_tokens[seq] = max_idx;
        return;
    }

    uint32_t c = ctrl[seq];
    if (c >= num_states) c = 0;
    uint32_t s = (sp != nullptr) ? sp[seq] : 1;
    uint32_t top = (s > 0 && stack != nullptr) ? stack[(size_t)seq * d + s - 1] : 0;

    uint32_t vob[D_MAX];
    uint32_t wpv = words_per_vob < D_MAX ? words_per_vob : D_MAX;
    scan_mask(transitions, ctrl_u32_offsets, ctrl_counts, c, top, vob, wpv);

    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int i = 0; i < vocab; i++) {
        float v = __bfloat162float(logit_row[i]);
        uint32_t word_idx = i / 32;
        if (word_idx < wpv && (vob[word_idx] & (1u << (i % 32))) == 0) {
            v = -FLT_MAX;
        }
        if (v > max_val) { max_val = v; max_idx = i; }
    }
    out_tokens[seq] = max_idx;

    uint32_t next_ctrl, next_top;
    advance_pda(transitions, ctrl_u32_offsets, ctrl_counts, c, top, max_idx, &next_ctrl, &next_top);
    out_ctrl[seq] = next_ctrl;
    if (out_sp != nullptr) out_sp[seq] = s;
}

// Fused project kernel: walk K draft tokens, emit K+1 VOB masks.
__global__ void pda_fused_project_kernel(
    const uint32_t* __restrict__ ctrl,
    const uint32_t* __restrict__ stack,
    const uint32_t* __restrict__ sp,
    const uint32_t* __restrict__ drafts,
    uint32_t* __restrict__ out_masks,
    const uint32_t* __restrict__ transitions,
    const uint32_t* __restrict__ accepting,
    const uint32_t* __restrict__ ctrl_u32_offsets,
    const uint32_t* __restrict__ ctrl_counts,
    uint32_t num_states,
    uint32_t num_stack_syms,
    uint32_t words_per_vob,
    int batch,
    int k,
    int d)
{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch) return;

    uint32_t c = ctrl[seq];
    if (c >= num_states) c = 0;
    uint32_t s = (sp != nullptr) ? sp[seq] : 1;
    uint32_t top = (s > 0 && stack != nullptr) ? stack[(size_t)seq * d + s - 1] : 0;

    const uint32_t* draft_row = drafts + (size_t)seq * k;

    for (int pos = 0; pos <= k; pos++) {
        // Emit the mask at the current (c, top).
        uint32_t* out = out_masks + ((size_t)seq * (k + 1) + pos) * words_per_vob;
        scan_mask(transitions, ctrl_u32_offsets, ctrl_counts, c, top, out, words_per_vob);

        if (pos == k) break;

        // Advance by draft[pos].
        uint32_t tok = draft_row[pos];
        uint32_t next_c, next_top;
        advance_pda(transitions, ctrl_u32_offsets, ctrl_counts, c, top, tok, &next_c, &next_top);
        c = next_c;
        top = next_top;
    }
}

extern "C" {

void pda_fused_sample_f32(
    const float* logits,
    const uint32_t* ctrl, const uint32_t* stack, const uint32_t* sp,
    uint32_t* out_ctrl, uint32_t* out_sp, uint32_t* out_tokens,
    const uint32_t* transitions, const uint32_t* accepting,
    const uint32_t* ctrl_u32_offsets, const uint32_t* ctrl_counts,
    uint32_t num_states, uint32_t num_stack_syms,
    uint32_t words_per_vob, int batch, int vocab,
    int top_k, float temperature, float top_p, int d, int64_t stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    pda_fused_sample_f32_kernel<<<blocks, threads, 0, s>>>(
        logits, ctrl, stack, sp, out_ctrl, out_sp, out_tokens,
        transitions, accepting, ctrl_u32_offsets, ctrl_counts,
        num_states, num_stack_syms,
        words_per_vob, batch, vocab, top_k, temperature, top_p, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[pda_fused_sample_f32] launch error: %s\n", cudaGetErrorString(err));
    }
}

void pda_fused_sample_bf16(
    const void* logits,
    const uint32_t* ctrl, const uint32_t* stack, const uint32_t* sp,
    uint32_t* out_ctrl, uint32_t* out_sp, uint32_t* out_tokens,
    const uint32_t* transitions, const uint32_t* accepting,
    const uint32_t* ctrl_u32_offsets, const uint32_t* ctrl_counts,
    uint32_t num_states, uint32_t num_stack_syms,
    uint32_t words_per_vob, int batch, int vocab,
    int top_k, float temperature, float top_p, int d, int64_t stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    pda_fused_sample_bf16_kernel<<<blocks, threads, 0, s>>>(
        (const __nv_bfloat16*)logits, ctrl, stack, sp, out_ctrl, out_sp, out_tokens,
        transitions, accepting, ctrl_u32_offsets, ctrl_counts,
        num_states, num_stack_syms,
        words_per_vob, batch, vocab, top_k, temperature, top_p, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[pda_fused_sample_bf16] launch error: %s\n", cudaGetErrorString(err));
    }
}

void pda_fused_project_masks(
    const uint32_t* ctrl, const uint32_t* stack, const uint32_t* sp,
    const uint32_t* drafts, uint32_t* out_masks,
    const uint32_t* transitions, const uint32_t* accepting,
    const uint32_t* ctrl_u32_offsets, const uint32_t* ctrl_counts,
    uint32_t num_states, uint32_t num_stack_syms,
    uint32_t words_per_vob, int batch, int k, int d, int64_t stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    pda_fused_project_kernel<<<blocks, threads, 0, s>>>(
        ctrl, stack, sp, drafts, out_masks,
        transitions, accepting, ctrl_u32_offsets, ctrl_counts,
        num_states, num_stack_syms,
        words_per_vob, batch, k, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[pda_fused_project_masks] launch error: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"