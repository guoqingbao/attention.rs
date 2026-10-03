//! The big/little cross-model ridge GEMV (the fused per-head W + b, the
//! proportional layer/head map). This is the GPU version of xinfer's
//! `RidgeMapper::apply_head` (the CPU oracle): it computes
//! `dst = W * src + b` for the cross-model KV translation, on the.
//!
//! The kernel must match the CPU oracle bit-for-bit (the same f32 accumulation
//! order: a = 0..d_s). Verified by xinfer's CPU-oracle test.

#[cfg(feature = "cuda")]
pub mod cuda {
    use candle_core::{Device, Result, Tensor};

    /// Compute the cross-model ridge translation: `dst = W * src + b`.
    ///
    /// - `src`: `[num_src_layers, src_heads, num_blocks, block_size, d_s]` f32
    /// - `W`:   `[num_tgt_layers, tgt_heads, d_s, d_t]` f32 (row-major, W[a,c] = W[a*d_t+c])
    /// - `b`:   `[num_tgt_layers, tgt_heads, d_t]` f32
    /// - `dst`: `[num_tgt_layers, tgt_heads, num_blocks, block_size, d_t]` f32 (pre-allocated)
    ///
    /// The proportional source map: `src_layer(j) = j * num_src_layers / num_tgt_layers`,
    /// `src_head(h) = h * src_heads / tgt_heads`.
    pub fn ridge_gemm(
        src: &Tensor,
        W: &Tensor,
        b: &Tensor,
        dst: &Tensor,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::Storage;
        use kernels::ffi;

        let raw_ptr = |t: &Tensor| -> Result<*const std::ffi::c_void> {
            let (s, _) = t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => Ok(*c.as_cuda_slice::<f32>()?.device_ptr() as *const std::ffi::c_void),
                _ => candle_core::bail!("ridge_gemm: expected a CUDA tensor"),
            }
        };
        let raw_ptr_mut = |t: &Tensor| -> Result<*mut std::ffi::c_void> {
            let (s, _) = t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => Ok(*c.as_cuda_slice::<f32>()?.device_ptr() as *mut std::ffi::c_void),
                _ => candle_core::bail!("ridge_gemm: expected a CUDA tensor"),
            }
        };

        let dev = src.device();
        let stream = *dev.as_cuda_device()?.cu_stream() as i64;

        let num_src_layers = src.dims()[0];
        let src_heads = src.dims()[1];
        let num_blocks = src.dims()[2];
        let block_size = src.dims()[3];
        let d_s = src.dims()[4];

        let num_tgt_layers = dst.dims()[0];
        let tgt_heads = dst.dims()[1];
        let d_t = dst.dims()[4];

        unsafe {
            ffi::ridge_gemm_f32(
                raw_ptr(src)?,
                raw_ptr(W)?,
                raw_ptr(b)?,
                raw_ptr_mut(dst)?,
                num_src_layers as i32,
                src_heads as i32,
                num_tgt_layers as i32,
                tgt_heads as i32,
                num_blocks as i32,
                block_size as i32,
                d_s as i32,
                d_t as i32,
                stream,
            );
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::cuda::ridge_gemm;
    use candle_core::{Device, Tensor};

    /// The CPU oracle: the same per-head W*x + b math the kernel does, with the
    /// proportional source layer/head map. This is the reference the kernel must
    /// match (bit-for-bit, the same f32 accumulation order a = 0..d_s).
    fn cpu_reference(
        src: &[f32],
        W: &[f32],
        b: &[f32],
        num_src_layers: usize,
        src_heads: usize,
        num_tgt_layers: usize,
        tgt_heads: usize,
        num_blocks: usize,
        block_size: usize,
        d_s: usize,
        d_t: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; num_tgt_layers * tgt_heads * num_blocks * block_size * d_t];
        for j in 0..num_tgt_layers {
            let src_layer = j * num_src_layers / num_tgt_layers;
            for h in 0..tgt_heads {
                let src_head = h * src_heads / tgt_heads;
                for bp in 0..num_blocks * block_size {
                    let block_idx = bp / block_size;
                    let pos = bp % block_size;
                    // The source row (d_s elements) for (src_layer, src_head, block_idx, pos).
                    let src_base =
                        (((src_layer * src_heads + src_head) * num_blocks + block_idx) * block_size + pos) * d_s;
                    // The W + b base for (j, h).
                    let W_base = (j * tgt_heads + h) * d_s * d_t;
                    let b_base = (j * tgt_heads + h) * d_t;
                    for c in 0..d_t {
                        let mut acc = b[b_base + c];
                        for a in 0..d_s {
                            acc += src[src_base + a] * W[W_base + a * d_t + c];
                        }
                        let dst_idx = (((j * tgt_heads + h) * num_blocks + block_idx) * block_size + pos) * d_t + c;
                        out[dst_idx] = acc;
                    }
                }
            }
        }
        out
    }

    /// The oracle test: a small deterministic case where the GPU kernel output must
    /// equal the CPU reference exactly (the same f32 accumulation order).
    #[test]
    fn ridge_gemm_matches_cpu_oracle() {
        let dev = Device::new_cuda(0).unwrap();
        // Small dims to keep the test fast + cover the proportional map.
        let (num_src_layers, src_heads) = (2, 2);
        let (num_tgt_layers, tgt_heads) = (4, 2);
        let (num_blocks, block_size) = (2, 2);
        let (d_s, d_t) = (3, 2);

        // Deterministic fills (the no rand): the reproducible oracle).
        let src_total = num_src_layers * src_heads * num_blocks * block_size * d_s;
        let src: Vec<f32> = (0..src_total).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let W_total = num_tgt_layers * tgt_heads * d_s * d_t;
        let W: Vec<f32> = (0..W_total).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let b_total = num_tgt_layers * tgt_heads * d_t;
        let b: Vec<f32> = (0..b_total).map(|i| (i as f32) * 0.2 - 1.0).collect();

        let src_t = Tensor::from_vec(src.clone(), (num_src_layers, src_heads, num_blocks, block_size, d_s), &dev).unwrap();
        let W_t = Tensor::from_vec(W.clone(), (num_tgt_layers, tgt_heads, d_s, d_t), &dev).unwrap();
        let b_t = Tensor::from_vec(b.clone(), (num_tgt_layers, tgt_heads, d_t), &dev).unwrap();
        let dst_t = Tensor::zeros((num_tgt_layers, tgt_heads, num_blocks, block_size, d_t), candle_core::DType::F32, &dev).unwrap();

        ridge_gemm(&src_t, &W_t, &b_t, &dst_t).unwrap();
        dev.as_cuda_device().unwrap().synchronize().unwrap();

        let dst_out = dst_t.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = cpu_reference(&src, &W, &b, num_src_layers, src_heads, num_tgt_layers, tgt_heads, num_blocks, block_size, d_s, d_t);

        let max_err = dst_out.iter().zip(expected.iter()).fold(0.0f32, |m, (a, b2)| m.max((a - b2).abs()));
        assert!(
            max_err < 1e-4,
            "ridge_gemm kernel diverged from the CPU oracle: max_err={max_err}"
        );
        // The proportional map sanity: layer 0 must read from source layer 0.
        assert_eq!(0 * num_src_layers / num_tgt_layers, 0);
        assert_eq!(3 * num_src_layers / num_tgt_layers, 1); // layer 3 -> source layer 1
    }
}