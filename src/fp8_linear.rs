use candle_core::{DType, Result, Tensor};

/// Block-wise FP8 matrix multiplication (software fallback for non-CUDA platforms).
///
/// Dequantizes the FP8 (U8) weight using per-block scales, then performs
/// standard matmul in the input's dtype.
///
/// - `input`:  (M, K) in f16/bf16/f32
/// - `weight`: (N, K) in U8 (FP8-E4M3 stored as u8)
/// - `weight_scale`: (ceil(N/by), ceil(K/bx)) in f32
/// - `weight_block_size`: [by, bx]
pub fn fp8_matmul(
    input: &Tensor,
    weight: &Tensor,
    weight_scale: &Tensor,
    weight_block_size: &[usize],
) -> Result<Tensor> {
    if weight_block_size.len() != 2 {
        candle_core::bail!("fp8_matmul: weight_block_size must have 2 elements");
    }
    let (n, k) = weight.dims2()?;
    let by = weight_block_size[0];
    let bx = weight_block_size[1];

    let w_f32 = weight.to_dtype(DType::F32)?;

    let scale_rows = (n + by - 1) / by;
    let scale_cols = (k + bx - 1) / bx;

    let scale = weight_scale.reshape((scale_rows, scale_cols))?;
    let scale_expanded = scale
        .unsqueeze(2)?
        .expand((scale_rows, scale_cols, bx))?
        .reshape((scale_rows, scale_cols * bx))?
        .narrow(1, 0, k)?
        .unsqueeze(1)?
        .expand((scale_rows, by, k))?
        .reshape((scale_rows * by, k))?
        .narrow(0, 0, n)?;

    let w_dequant = (w_f32 * scale_expanded)?.to_dtype(input.dtype())?;

    input.matmul(&w_dequant.t()?)
}

/// Cutlass-accelerated FP8 matmul stub — falls back to `fp8_matmul` on non-CUDA.
///
/// On CUDA with SM>=90 the upstream uses cutlass kernels; here we just delegate
/// to the software path.
///
/// Note: the cutlass variant expects weight already transposed (K, N),
/// while the standard variant expects (N, K). We transpose back before delegating.
pub fn fp8_matmul_cutlass(
    input: &Tensor,
    weight_transposed: &Tensor,
    weight_scale: &Tensor,
    weight_block_size: &[usize],
) -> Result<Tensor> {
    let weight = weight_transposed.t()?;
    fp8_matmul(input, &weight, weight_scale, weight_block_size)
}
