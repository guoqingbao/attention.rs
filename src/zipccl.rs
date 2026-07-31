//! ZipCCL: lossless BF16/F16 exponent compression for NCCL collectives.
//! Based on arXiv:2604.27844 (top-7 exponent coding + bitplane packing).
//!
//! Compression is CUDA-kernel based. The exact zero-point count is read back
//! for the metadata exchange used by variable-count NCCL sends. xInfer uses
//! this path only for prefill; decode remains on native NCCL/CUDA graphs.

use std::sync::atomic::{AtomicBool, Ordering};

/// Global switch. Also honored via env `XINFER_ZIPCCL=1` so runner subprocesses
/// inherit the flag from the parent (AllReduce runs inside runners).
pub static ZIPCCL_ENABLED: AtomicBool = AtomicBool::new(false);

pub const ZIPCCL_ENV: &str = "XINFER_ZIPCCL";

pub fn set_enabled(enabled: bool) {
    ZIPCCL_ENABLED.store(enabled, Ordering::Relaxed);
    if enabled {
        std::env::set_var(ZIPCCL_ENV, "1");
    } else {
        std::env::remove_var(ZIPCCL_ENV);
    }
}

pub fn is_enabled() -> bool {
    if ZIPCCL_ENABLED.load(Ordering::Relaxed) {
        return true;
    }
    match std::env::var_os(ZIPCCL_ENV) {
        Some(v) => v == "1" || v == "true" || v == "TRUE",
        None => false,
    }
}

#[cfg(feature = "cuda")]
pub mod cuda {
    use crate::kernels::ffi;
    use candle_core::cuda_backend::cudarc::driver::{
        result as driver_result, CudaSlice, DevicePtr, DeviceSlice,
    };
    use candle_core::cuda_backend::{CudaDevice, WrapErr};
    use candle_core::{DType, Result};

    fn check_cuda(op: &str, status: i32) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            candle_core::bail!("{op} launch failed with CUDA error code {status}")
        }
    }

    pub fn dtype_id(dtype: DType) -> Result<i32> {
        match dtype {
            DType::BF16 => Ok(0),
            DType::F16 => Ok(1),
            other => candle_core::bail!("zipccl supports BF16/F16 only, got {other:?}"),
        }
    }

    /// Fixed NCCL transfer size for `n` elements — host-only, no device access.
    pub fn xfer_bytes(n: usize, dtype: DType) -> Result<usize> {
        let id = dtype_id(dtype)?;
        Ok(unsafe { ffi::zipccl_xfer_bytes_dtype(n as i32, id) as usize })
    }

    pub fn max_compressed_bytes(n: usize, dtype: DType) -> Result<usize> {
        xfer_bytes(n, dtype)
    }

    /// Read the dynamic zero-point count written by the compressor and return
    /// the exact aligned payload size used by the variable-count collective.
    /// This is the metadata exchange required by ZipCCL: the payload size is
    /// not a function of `(n, dtype)` alone.
    pub fn compressed_bytes(
        device: &CudaDevice,
        compressed_ptr: u64,
        n: usize,
        dtype: DType,
    ) -> Result<usize> {
        // The count is produced by a kernel on the device stream.  A plain
        // host memcpy is not ordered with that stream, so synchronize before
        // consuming the metadata.
        device.synchronize().map_err(candle_core::Error::wrap)?;
        let mut header = [0u8; 4];
        unsafe {
            driver_result::memcpy_dtoh_sync(&mut header, (compressed_ptr + 4) as _)
                .map_err(candle_core::Error::wrap)?;
        }
        let zp_count = u32::from_ne_bytes(header) as usize;
        compressed_bytes_from_zp_count(n, dtype, zp_count)
    }

    pub fn compressed_bytes_from_zp_count(
        n: usize,
        dtype: DType,
        zp_count: usize,
    ) -> Result<usize> {
        let static_bytes =
            unsafe { ffi::zipccl_static_bytes_dtype(n as i32, dtype_id(dtype)?) as usize };
        let blocks = num_blocks(n).max(0) as usize;
        Ok((static_bytes + blocks * std::mem::size_of::<u32>() + zp_count + 15) & !15)
    }

    pub fn num_blocks(n: usize) -> i32 {
        unsafe { ffi::zipccl_num_blocks(n as i32) }
    }

    /// Compress into a device buffer of size [`xfer_bytes`].
    /// Returns `(buffer, xfer_bytes)` — size is host-computed, never read back from GPU.
    pub fn compress_buf(
        device: &CudaDevice,
        input_ptr: u64,
        n: usize,
        dtype: DType,
        stream: i64,
    ) -> Result<(CudaSlice<u8>, usize)> {
        let id = dtype_id(dtype)?;
        let bytes = xfer_bytes(n, dtype)?;
        let buf = unsafe { device.alloc::<u8>(bytes) }.w()?;
        unsafe {
            check_cuda(
                "zipccl_compress",
                ffi::zipccl_compress(
                    input_ptr as *const core::ffi::c_void,
                    *buf.device_ptr() as *mut core::ffi::c_void,
                    n as i32,
                    id,
                    std::ptr::null(),
                    stream,
                ),
            )?;
        }
        Ok((buf, bytes))
    }

    /// Compress into a caller-provided buffer (must be ≥ xfer_bytes). Device-only.
    pub fn compress_into(
        input_ptr: u64,
        output: &CudaSlice<u8>,
        n: usize,
        dtype: DType,
        stream: i64,
    ) -> Result<usize> {
        let id = dtype_id(dtype)?;
        let bytes = xfer_bytes(n, dtype)?;
        unsafe {
            check_cuda(
                "zipccl_compress",
                ffi::zipccl_compress(
                    input_ptr as *const core::ffi::c_void,
                    *output.device_ptr() as *mut core::ffi::c_void,
                    n as i32,
                    id,
                    std::ptr::null(),
                    stream,
                ),
            )?;
        }
        Ok(bytes)
    }

    /// Compress using a per-payload GPU histogram to select the seven most
    /// frequent exponents. `hist` must contain 256 `u32`s and `top7` 8 bytes.
    pub fn compress_into_dynamic<O: DevicePtr<u8>>(
        input_ptr: u64,
        output: &O,
        n: usize,
        dtype: DType,
        hist: &CudaSlice<u32>,
        top7: &CudaSlice<u8>,
        stream: i64,
    ) -> Result<usize> {
        let id = dtype_id(dtype)?;
        let bytes = xfer_bytes(n, dtype)?;
        if hist.len() < 256 || top7.len() < 7 {
            candle_core::bail!("zipccl dynamic compression scratch is too small");
        }
        unsafe {
            check_cuda(
                "zipccl_compress_dynamic",
                ffi::zipccl_compress_dynamic(
                    input_ptr as *const core::ffi::c_void,
                    *output.device_ptr() as *mut core::ffi::c_void,
                    n as i32,
                    id,
                    *hist.device_ptr() as *mut core::ffi::c_void,
                    *top7.device_ptr() as *mut core::ffi::c_void,
                    stream,
                ),
            )?;
        }
        Ok(bytes)
    }

    pub fn compress_into_with_top7<O: DevicePtr<u8>>(
        input_ptr: u64,
        output: &O,
        n: usize,
        dtype: DType,
        top7: &CudaSlice<u8>,
        stream: i64,
    ) -> Result<usize> {
        let id = dtype_id(dtype)?;
        let bytes = xfer_bytes(n, dtype)?;
        if top7.len() < 7 {
            candle_core::bail!("zipccl top7 scratch is too small");
        }
        unsafe {
            check_cuda(
                "zipccl_compress_top7",
                ffi::zipccl_compress(
                    input_ptr as *const core::ffi::c_void,
                    *output.device_ptr() as *mut core::ffi::c_void,
                    n as i32,
                    id,
                    *top7.device_ptr() as *const core::ffi::c_void,
                    stream,
                ),
            )?;
        }
        Ok(bytes)
    }

    pub fn decompress_buf(
        compressed: &CudaSlice<u8>,
        output_ptr: u64,
        n: usize,
        dtype: DType,
        stream: i64,
    ) -> Result<()> {
        let id = dtype_id(dtype)?;
        unsafe {
            check_cuda(
                "zipccl_decompress",
                ffi::zipccl_decompress(
                    *compressed.device_ptr() as *const core::ffi::c_void,
                    output_ptr as *mut core::ffi::c_void,
                    n as i32,
                    id,
                    stream,
                ),
            )?;
        }
        Ok(())
    }

    pub fn decompress_ptr(
        compressed_ptr: u64,
        output_ptr: u64,
        n: usize,
        dtype: DType,
        stream: i64,
    ) -> Result<()> {
        let id = dtype_id(dtype)?;
        unsafe {
            check_cuda(
                "zipccl_decompress",
                ffi::zipccl_decompress(
                    compressed_ptr as *const core::ffi::c_void,
                    output_ptr as *mut core::ffi::c_void,
                    n as i32,
                    id,
                    stream,
                ),
            )?;
        }
        Ok(())
    }

    pub fn decompress_add_f32_ptr(
        compressed_ptr: u64,
        output_ptr: u64,
        n: usize,
        dtype: DType,
        stream: i64,
    ) -> Result<()> {
        let id = dtype_id(dtype)?;
        unsafe {
            check_cuda(
                "zipccl_decompress_add_f32",
                ffi::zipccl_decompress_add_f32(
                    compressed_ptr as *const core::ffi::c_void,
                    output_ptr as *mut core::ffi::c_void,
                    n as i32,
                    id,
                    stream,
                ),
            )?;
        }
        Ok(())
    }
}
