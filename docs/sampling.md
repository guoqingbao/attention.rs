# GPU Sampling

Sampling is the final step of LLM inference, where a token ID is selected from the output logits. Standard CPU-based sampling requires transferring logits from the GPU, which is slow. `attention-rs` provides **fully fused GPU sampling kernels**.

## Core Algorithms
- **Temperature Scaling**: `logits / temperature`.
- **Top-K Filtering**: Keeps only the $K$ tokens with the highest probabilities.
- **Nucleus (Top-P) Sampling**: Keeps tokens such that cumulative probability exceeds $P$.

## Fused Sampler
The `Sampler` struct provides a `sample_cuda` method that performs softmax, filtering, and random selection entirely on the GPU.

## Integration in `vllm-rs`

In `vllm-rs`, the `LogitsProcessor` uses the fast GPU sampler for compatible strategies.

```rust
use attention_rs::sampler::Sampler;

pub struct LogitsProcessor {
    // ...
    fast_sampler: Arc<Mutex<Sampler>>,
}

impl LogitsProcessor {
    pub fn sample(&self, logits: &Tensor, sampling: &Sampling) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        {
            if let Some((k, p, t)) = get_params(sampling) {
                let seed = self.rng.lock().next_u64();
                let sampler = self.fast_sampler.lock().unwrap();
                
                // This call happens entirely on the GPU
                // Only the resulting u32 token IDs are transferred back.
                return sampler.sample_cuda(logits, k, p, t, seed);
            }
        }
        
        // Fallback to CPU sampling for complex cases...
    }
}
```

## Detailed Example

```rust
use candle_core::{Device, Tensor, DType};
use attention_rs::sampler::Sampler;

let device = Device::new_cuda(0)?;
let sampler = Sampler::new();

// Batch size 4, Vocab size 128000
let logits = Tensor::randn(0.0, 1.0, (4, 128000), &device)?.to_dtype(DType::F16)?;

// Sampling parameters
let top_k = 50;
let top_p = 0.95;
let temperature = 0.7;
let seed = 42;

let sampled_tokens = sampler.sample_cuda(
    &logits, top_k, top_p, temperature, seed,
)?;

println!("Sampled tokens: {:?}", sampled_tokens); // Vec<u32> length 4
```

## Performance Benefits
- **Zero Latency**: No need to sync large logit tensors to the CPU.
- **High Throughput**: Can sample for thousands of sequences in parallel.
- **Deterministic**: Fully supports seeding for reproducible generation.
