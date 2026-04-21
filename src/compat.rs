use candle_core::{DType, Device, Result, Shape, Storage, Tensor};

/// Numerically-stable softmax over the last dimension.
///
/// Delegates to `candle_nn::ops::softmax_last_dim` when `candle-upstream` or
/// `candle-custom` is active; falls back to a pure-tensor implementation when
/// only `candle-upstream-core-only` is enabled.
pub fn softmax_last_dim(t: &Tensor) -> Result<Tensor> {
    #[cfg(any(feature = "candle-upstream", feature = "candle-custom"))]
    {
        candle_nn::ops::softmax_last_dim(t)
    }
    #[cfg(not(any(feature = "candle-upstream", feature = "candle-custom")))]
    {
        let max = t.max_keepdim(candle_core::D::Minus1)?;
        let shifted = t.broadcast_sub(&max)?;
        let exp = shifted.exp()?;
        let sum = exp.sum_keepdim(candle_core::D::Minus1)?;
        exp.broadcast_div(&sum)
    }
}

/// SiLU (Sigmoid Linear Unit) activation: `x * sigmoid(x)`.
pub fn silu(t: &Tensor) -> Result<Tensor> {
    #[cfg(any(feature = "candle-upstream", feature = "candle-custom"))]
    {
        candle_nn::ops::silu(t)
    }
    #[cfg(not(any(feature = "candle-upstream", feature = "candle-custom")))]
    {
        let sig = sigmoid(t)?;
        t.mul(&sig)
    }
}

/// Sigmoid activation: `1 / (1 + exp(-x))`.
pub fn sigmoid(t: &Tensor) -> Result<Tensor> {
    #[cfg(any(feature = "candle-upstream", feature = "candle-custom"))]
    {
        candle_nn::ops::sigmoid(t)
    }
    #[cfg(not(any(feature = "candle-upstream", feature = "candle-custom")))]
    {
        let neg = t.neg()?;
        let exp_neg = neg.exp()?;
        let one_plus = (exp_neg + 1.0)?;
        let ones = Tensor::ones_like(&one_plus)?;
        ones.div(&one_plus)
    }
}

/// Create a Tensor from raw Storage.
///
/// The custom fork's signature is `(storage, shape) -> Result<Tensor>`.
/// Upstream's signature is `(storage, shape, BackpropOp, bool) -> Tensor`.
#[cfg(feature = "candle-custom")]
pub fn tensor_from_storage<S: Into<Shape>>(storage: Storage, shape: S) -> Result<Tensor> {
    Tensor::from_storage(storage, shape)
}

#[cfg(not(feature = "candle-custom"))]
pub fn tensor_from_storage<S: Into<Shape>>(storage: Storage, shape: S) -> Result<Tensor> {
    Ok(Tensor::from_storage(
        storage,
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

/// Allocate an uninitialized GPU tensor.
///
/// The custom fork provides `Tensor::empty_` (true uninitialized allocation).
/// Upstream does not expose it, so we fall back to `Tensor::zeros` (safe, slightly slower).
#[cfg(feature = "candle-custom")]
pub unsafe fn tensor_empty_uninit<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    Tensor::empty_(shape, dtype, device)
}

#[cfg(not(feature = "candle-custom"))]
pub fn tensor_empty_uninit<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    Tensor::zeros(shape, dtype, device)
}

/// Zero-fill a tensor in-place.
///
/// The custom fork provides `Tensor::zero_()`.
/// Upstream provides `Tensor::zero_set()`.
#[cfg(feature = "candle-custom")]
pub fn tensor_zero_inplace(tensor: &Tensor) -> Result<()> {
    tensor.zero_()?;
    Ok(())
}

#[cfg(not(feature = "candle-custom"))]
pub fn tensor_zero_inplace(tensor: &Tensor) -> Result<()> {
    tensor.zero_set()
}
