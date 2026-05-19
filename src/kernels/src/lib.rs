pub mod ffi;

#[cfg(flashinfer_fp8_kvcache)]
pub const HAS_FLASHINFER_FP8_KVCACHE: bool = true;
#[cfg(not(flashinfer_fp8_kvcache))]
pub const HAS_FLASHINFER_FP8_KVCACHE: bool = false;
