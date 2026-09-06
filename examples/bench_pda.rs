//! PDA step-time benchmark (GPU). Measures the per-decode-step cost of the
//! GPU-resident PDA (fused_sample: mask + sample + advance) at various batch sizes.
//!
//! Run on a GPU host:
//!   CUDA_COMPUTE_CAP=120 cargo run --features cuda --example bench_pda
//!
//! DISCARDABLE QA artifact (not part of the permanent suite).

#![cfg(feature = "cuda")]

use attention_rs::pda::{PdaPushdownTable, PdaSampling};
use candle_core::{Device, DType, Tensor};
use std::time::Instant;

/// Build a small synthetic PDA table (3 ctrl states) sized to a given vocab.
fn make_table(dev: &Device, vocab: usize) -> PdaPushdownTable {
    let num_states = 3u32;
    let num_inputs = vocab as u32;
    let num_stack_syms = 2u32;
    // Transition flat array:
    //   Record 0: (q=0, a=1, top=0, next_q=1, push_len=1, push=[1])
    //   Record 1: (q=1, a=2, top=1, next_q=2, push_len=1, push=[2])
    let transitions = vec![
        0, 1, 0, 1, 1, 1, // record 0: 6 u32s
        1, 2, 1, 2, 1, 2, // record 1: 6 u32s
    ];
    let accepting = vec![2u32];
    PdaPushdownTable::upload(
        transitions,
        accepting,
        num_states,
        num_inputs,
        num_stack_syms,
        2, // num_transitions
        0, // start_state
        0, // start_stack
        8, // max_stack_depth
        dev,
    ).expect("table upload")
}

fn main() {
    let dev = Device::new_cuda(0).expect("CUDA device");
    let vocab = 4096;
    let table = make_table(&dev, vocab);
    let batch = 32;
    let d = table.max_stack_depth as usize;

    // Allocate the per-batch tensors.
    let logits = Tensor::zeros((batch, vocab), DType::F32, &dev).unwrap();
    let ctrl = Tensor::zeros((batch,), DType::U32, &dev).unwrap();
    let stack = Tensor::zeros((batch, d), DType::U32, &dev).unwrap();
    let sp = Tensor::from_vec(vec![1u32; batch], (batch,), &dev).unwrap();

    // Warmup.
    for _ in 0..10 {
        let _ = table.fused_sample(&logits, &ctrl, &stack, &sp, &PdaSampling::Greedy).unwrap();
    }
    dev.as_cuda_device().unwrap().synchronize().unwrap();

    // Measure fused_sample.
    let iters = 1000;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = table.fused_sample(&logits, &ctrl, &stack, &sp, &PdaSampling::Greedy).unwrap();
    }
    dev.as_cuda_device().unwrap().synchronize().unwrap();
    let elapsed = start.elapsed();
    let us_per_step = elapsed.as_micros() / iters as u128;
    println!(
        "PDA fused_sample: batch={}, vocab={}, {} us/step ({} iters)",
        batch, vocab, us_per_step, iters
    );

    // Measure fused_project (drafting).
    let k = 16;
    let drafts = Tensor::zeros((batch, k), DType::U32, &dev).unwrap();
    for _ in 0..10 {
        let _ = table.fused_project(&ctrl, &stack, &sp, &drafts).unwrap();
    }
    dev.as_cuda_device().unwrap().synchronize().unwrap();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = table.fused_project(&ctrl, &stack, &sp, &drafts).unwrap();
    }
    dev.as_cuda_device().unwrap().synchronize().unwrap();
    let elapsed = start.elapsed();
    let us_per_proj = elapsed.as_micros() / iters as u128;
    println!(
        "PDA fused_project: batch={}, k={}, vocab={}, {} us/step ({} iters)",
        batch, k, vocab, us_per_proj, iters
    );
}