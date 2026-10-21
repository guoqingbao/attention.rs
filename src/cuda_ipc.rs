//! The CUDA IPC (inter-process communication) primitives for the big/little
//! cross-model KV transfer. These are CUDA runtime API calls (not kernels), so
//! they don't need the kernel build system. They enable GPU->GPU memory sharing
//! between the small model's runner process and the large model's runner process
//! (the no host round-trip for the translated KV).
//!
//! The flow:
//! 1. The small model's runner allocates the translated KV buffer with
//!    `cuda_malloc` (the driver-allocated memory, the CUDA IPC requirement).
//! 2. The ridge kernel writes into it (the in-GPU translation).
//! 3. `cuda_ipc_get_mem_handle` produces a 64-byte handle (the IPC handle).
//! 4. The handle is sent to the large model's runner (via the engine).
//! 5. `cuda_ipc_open_mem_handle` maps the handle into the large model's runner's
//!    address space (the GPU pointer to the small model's buffer).
//! 6. `cuda_memcpy` (D2D) copies the translated KV into the large model's pool
//!    (GPU->GPU, the no host hop).
//! 7. `cuda_ipc_close_mem_handle` + `cuda_free` clean up.

use std::os::raw::c_void;

/// The CUDA IPC memory handle (the 64-byte opaque handle, the CUDA requirement).
/// Serialized as raw bytes (the  derive fails on the fixed-size array).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CudaIpcMemHandle {
    pub status: i32,
    pub reserved: [u8; 5],
    pub bytes: [u8; 48],
}

impl CudaIpcMemHandle {
    /// Serialize the handle to a 64-byte buffer (the IPC message payload).
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0..4].copy_from_slice(&self.status.to_le_bytes());
        out[4..9].copy_from_slice(&self.reserved);
        out[9..57].copy_from_slice(&self.bytes);
        out
    }

    /// Deserialize the handle from a 64-byte buffer (the IPC message payload).
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        let mut status_bytes = [0u8; 4];
        status_bytes.copy_from_slice(&bytes[0..4]);
        let mut reserved = [0u8; 5];
        reserved.copy_from_slice(&bytes[4..9]);
        let mut data = [0u8; 48];
        data.copy_from_slice(&bytes[9..57]);
        Self {
            status: i32::from_le_bytes(status_bytes), reserved, bytes: data}
    }
}

impl Default for CudaIpcMemHandle {
    fn default() -> Self {
        Self { status: 0, reserved: [0; 5], bytes: [0; 48] }
    }
}

extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaIpcGetMemHandle(handle: *mut CudaIpcMemHandle, devPtr: *mut c_void) -> i32;
    fn cudaIpcOpenMemHandle(devPtr: *mut *mut c_void, handle: CudaIpcMemHandle, flags: u32) -> i32;
    fn cudaIpcCloseMemHandle(devPtr: *mut c_void) -> i32;
}

/// The cudaMemcpy kinds (the D2D = 3, the H2D = 1, the D2H = 2).
pub const CUDA_MEMCPY_D2D: i32 = 3;
pub const CUDA_MEMCPY_H2D: i32 = 1;
pub const CUDA_MEMCPY_D2H: i32 = 2;

/// Allocate a GPU buffer with `cudaMalloc` (the driver-allocated memory, the
/// CUDA IPC requirement). Returns the raw pointer.
pub fn cuda_malloc(size: usize) -> Result<*mut c_void, String> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let code = unsafe { cudaMalloc(&mut ptr, size) };
    if code != 0 {
        return Err(format!("cudaMalloc failed (code {code})"));
    }
    Ok(ptr)
}

/// Free a GPU buffer allocated with `cuda_malloc`.
pub fn cuda_free(ptr: *mut c_void) -> Result<(), String> {
    if ptr.is_null() {
        return Ok(());
    }
    let code = unsafe { cudaFree(ptr) };
    if code != 0 {
        return Err(format!("cudaFree failed (code {code})"));
    }
    Ok(())
}

/// Copy GPU memory (the D2D for the cross-process KV transfer).
pub fn cuda_memcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> Result<(), String> {
    let code = unsafe { cudaMemcpy(dst, src, count, kind) };
    if code != 0 {
        return Err(format!("cudaMemcpy failed (code {code})"));
    }
    Ok(())
}

/// Get the CUDA IPC handle for a GPU buffer (the 64-byte handle, the cross-process
/// memory sharing).
pub fn cuda_ipc_get_mem_handle(ptr: *mut c_void) -> Result<CudaIpcMemHandle, String> {
    let mut handle = CudaIpcMemHandle::default();
    let code = unsafe { cudaIpcGetMemHandle(&mut handle, ptr) };
    if code != 0 {
        return Err(format!("cudaIpcGetMemHandle failed (code {code})"));
    }
    Ok(handle)
}

/// Open a CUDA IPC handle (the map the remote process's GPU buffer into this
/// process's address space). Returns the mapped GPU pointer.
pub fn cuda_ipc_open_mem_handle(handle: &CudaIpcMemHandle) -> Result<*mut c_void, String> {
    let mut dev_ptr: *mut c_void = std::ptr::null_mut();
    let code = unsafe { cudaIpcOpenMemHandle(&mut dev_ptr, *handle, 0) };
    if code != 0 {
        return Err(format!("cudaIpcOpenMemHandle failed (code {code})"));
    }
    Ok(dev_ptr)
}

/// Close a CUDA IPC handle (the unmap the remote process's GPU buffer).
pub fn cuda_ipc_close_mem_handle(ptr: *mut c_void) -> Result<(), String> {
    if ptr.is_null() {
        return Ok(());
    }
    let code = unsafe { cudaIpcCloseMemHandle(ptr) };
    if code != 0 {
        return Err(format!("cudaIpcCloseMemHandle failed (code {code})"));
    }
    Ok(())
}