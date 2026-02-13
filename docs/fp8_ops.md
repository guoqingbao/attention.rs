# FP8 Operations & CUTLASS

FP8 (8-bit floating point) offers 2x the throughput of FP16 while maintaining high accuracy. `attention-rs` provides state-of-the-art FP8 kernels, including integration with NVIDIA CUTLASS for modern architectures.

## Supported Formats
- **E4M3**: Used for weights and activations.
- **Block-wise Scaling**: Supports fine-grained scaling (e.g., $128 \times 128$ blocks) to preserve dynamic range in large models.

## MATMUL Paths

1.  **Conventional (`fp8_matmul`)**: Optimized CUDA/Metal kernels for general-purpose FP8 matrix-vector or matrix-matrix multiplication.
2.  **CUTLASS (`fp8_matmul_cutlass`)**: Specialized for NVIDIA **Hopper (H100/H200)** and **Blackwell**. Uses CUTLASS Grouped GEMM for maximum Tensor Core utilization.

## Integration in `vllm-rs`

In `vllm-rs`, the `LnFp8` (Linear FP8) layer selects the best kernel based on the GPU's Compute Capability.

```rust
use attention_rs::fp8_linear::{fp8_matmul, fp8_matmul_cutlass};

impl LnFp8 {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_2d = x.reshape((m, k))?;

        #[cfg(feature = "cutlass")]
        let out = if self.sm_version >= 90 {
            // High-performance Hopper path
            fp8_matmul_cutlass(
                &x_2d,
                &self.weight.t()?,
                &self.weight_scale,
                &self.weight_block_size,
            )?
        } else {
            // Standard CUDA path
            fp8_matmul(
                &x_2d,
                &self.weight,
                &self.weight_scale,
                &self.weight_block_size,
            )?
        };
        
        // ...
    }
}
```

## Detailed Example

```rust
use candle_core::{Device, Tensor, DType};
use attention_rs::fp8_linear::fp8_matmul;

let device = Device::new_cuda(0)?;

// 1. Activation (F16/BF16)
let input = Tensor::randn(0.0, 1.0, (128, 4096), &device)?;

// 2. Weights (FP8 E4M3, stored in U8)
let weight_fp8 = Tensor::zeros((4096, 4096), DType::U8, &device)?;

// 3. Block-wise Scales (F32)
// Scale factor for each block (e.g., 1x128)
let weight_scale = Tensor::randn(0.0, 1.0, (4096, 32), &device)?;

// 4. Perform MatMul
let output = fp8_matmul(
    &input,
    &weight_fp8,
    &weight_scale,
    &[1, 128], // [m_block, k_block]
)?;
```

## Technical Details
- **sm_version**: Use `attention_rs::cuda_utils::sm_version()` to detect GPU capabilities.
- **Quantization**: Activations are typically dynamically quantized to FP8 per-token before calling the matmul.
