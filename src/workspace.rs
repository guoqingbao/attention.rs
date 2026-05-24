//! Unified workspace buffer management for FlashInfer, CUTLASS, and other GPU operations.
//!
//! This module provides a centralized workspace allocation strategy that eliminates
//! redundant buffer allocations across different modules (moe.rs, fp8_linear.rs,
//! nvfp4_linear.rs, mxfp4_linear.rs, flashinfer.rs).
//!
//! # Workspace Layout (when `flashinfer` feature is enabled)
//!
//! ```text
//! FLASHINFER FLOAT BUFFER (512 MiB total):
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  FlashInfer Plan Float (256 MiB)  │  FlashInfer Scratch (256 MiB)
//! │  [0..256 MiB)                     │  [256 MiB..512 MiB)        │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! INT BUFFER (128 MiB):
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  FlashInfer Plan Int (128 MiB)                                  │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! PINNED HOST BUFFER (128 MiB):
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Page-locked host memory for FlashInfer scheduler              │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Workspace Layout (when `flashinfer` feature is disabled)
//!
//! CUTLASS keeps a separate per-thread device workspace. When `flashinfer` is
//! enabled, CUTLASS does not alias the FlashInfer float buffer because that
//! shared path is not stable for the FP8 CUTLASS GEMM kernels.
//!
//! # Thread Safety
//!
//! Workspaces are thread-local to avoid synchronization overhead. Each thread
//! maintains its own workspace buffers, which are lazily initialized on first use.

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{sys, CudaSlice, DevicePtr};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::WrapErr;
use candle_core::Result;

/// Total size of the float workspace buffer (512 MiB).
pub const WORKSPACE_FLOAT_SIZE: usize = 512 * 1024 * 1024;

/// Size of the int workspace buffer (128 MiB).
pub const WORKSPACE_INT_SIZE: usize = 128 * 1024 * 1024;

/// Size of the FlashInfer plan region within the float buffer.
pub const FLASHINFER_PLAN_FLOAT_SIZE: usize = 256 * 1024 * 1024;

/// Offset where GEMM scratch space begins within the float buffer.
pub const GEMM_SCRATCH_FLOAT_OFFSET: usize = FLASHINFER_PLAN_FLOAT_SIZE;

/// Size of the GEMM scratch region (used by CUTLASS operations).
pub const GEMM_SCRATCH_FLOAT_SIZE: usize = 256 * 1024 * 1024;

/// Fallback workspace size when flashinfer is disabled.
#[cfg(feature = "cuda")]
pub const CUTLASS_WORKSPACE_FALLBACK_SIZE: usize = 512 * 1024 * 1024;

/// Describes a region within a workspace buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceRegion {
    pub offset: usize,
    pub size: usize,
}

/// Collection of all workspace regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceRegions {
    pub plan_float: WorkspaceRegion,
    pub gemm_scratch_float: WorkspaceRegion,
    pub plan_int: WorkspaceRegion,
}

/// Pre-computed workspace region definitions.
pub const WORKSPACE_REGIONS: WorkspaceRegions = WorkspaceRegions {
    plan_float: WorkspaceRegion {
        offset: 0,
        size: FLASHINFER_PLAN_FLOAT_SIZE,
    },
    gemm_scratch_float: WorkspaceRegion {
        offset: GEMM_SCRATCH_FLOAT_OFFSET,
        size: GEMM_SCRATCH_FLOAT_SIZE,
    },
    plan_int: WorkspaceRegion {
        offset: 0,
        size: WORKSPACE_INT_SIZE,
    },
};

/// Returns the workspace regions after validating they don't overlap.
pub fn workspace_regions() -> WorkspaceRegions {
    debug_assert!(
        WORKSPACE_REGIONS.plan_float.offset + WORKSPACE_REGIONS.plan_float.size
            <= WORKSPACE_FLOAT_SIZE
    );
    debug_assert!(
        WORKSPACE_REGIONS.gemm_scratch_float.offset
            >= WORKSPACE_REGIONS.plan_float.offset + WORKSPACE_REGIONS.plan_float.size
    );
    debug_assert!(
        WORKSPACE_REGIONS.gemm_scratch_float.offset + WORKSPACE_REGIONS.gemm_scratch_float.size
            <= WORKSPACE_FLOAT_SIZE
    );
    debug_assert!(
        WORKSPACE_REGIONS.plan_int.offset + WORKSPACE_REGIONS.plan_int.size <= WORKSPACE_INT_SIZE
    );
    WORKSPACE_REGIONS
}

/// Adds an offset to a base pointer, returning a pointer to the specified region.
pub fn add_workspace_offset(
    base: *mut std::ffi::c_void,
    region: WorkspaceRegion,
) -> *mut std::ffi::c_void {
    debug_assert!(!base.is_null());
    unsafe { (base as *mut u8).add(region.offset) as *mut std::ffi::c_void }
}

// ============================================================================
// CUDA-specific workspace management
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// Page-locked host buffer for FlashInfer scheduler operations.
    pub struct PinnedHostBuffer {
        ptr: *mut std::ffi::c_void,
        size: usize,
    }

    impl PinnedHostBuffer {
        pub fn new(size: usize) -> Result<Self> {
            if size == 0 {
                candle_core::bail!("Pinned host buffer size must be > 0");
            }
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            unsafe {
                sys::lib()
                    .cuMemAllocHost_v2(&mut ptr, size)
                    .result()
                    .map_err(|e| {
                        candle_core::Error::Msg(format!("cuMemAllocHost_v2 failed: {e:?}"))
                    })?
            }
            if ptr.is_null() {
                candle_core::bail!("cuMemAllocHost_v2 returned null pointer");
            }
            Ok(Self { ptr, size })
        }

        pub fn as_ptr(&self) -> *mut std::ffi::c_void {
            self.ptr
        }

        pub fn size(&self) -> usize {
            self.size
        }
    }

    impl Drop for PinnedHostBuffer {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe {
                    let _ = sys::lib().cuMemFreeHost(self.ptr).result();
                }
            }
        }
    }

    // SAFETY: The pinned buffer pointer is allocated per-thread and only accessed
    // within that thread's context.
    unsafe impl Send for PinnedHostBuffer {}

    /// Primary workspace for FlashInfer operations.
    pub struct FlashInferWorkspace {
        pub float_buffer: CudaSlice<u8>,
        pub int_buffer: CudaSlice<u8>,
        pub pinned_host: PinnedHostBuffer,
        pub device_ordinal: usize,
    }

    /// Dedicated CUTLASS workspace, kept separate from FlashInfer buffers.
    #[cfg(feature = "cutlass")]
    pub struct CutlassWorkspace {
        pub buffer: CudaSlice<u8>,
        pub size: usize,
        pub device_ordinal: usize,
    }

    /// Specialized FP8 workspace for FlashInfer blockscale operations.
    #[cfg(feature = "flashinfer")]
    pub struct FlashInferFp8Workspace {
        pub buffer: CudaSlice<u8>,
        pub size: usize,
        pub device_ordinal: usize,
    }

    thread_local! {
        /// Primary FlashInfer workspace (normal operations).
        #[cfg(feature = "flashinfer")]
        pub static WORKSPACE: std::cell::RefCell<Option<FlashInferWorkspace>> = const { std::cell::RefCell::new(None) };

        /// FlashInfer workspace for CUDA graph captures (separate to avoid interference).
        #[cfg(feature = "flashinfer")]
        pub static WORKSPACE_GRAPH: std::cell::RefCell<Option<FlashInferWorkspace>> = const { std::cell::RefCell::new(None) };

        /// Dedicated CUTLASS workspace.
        #[cfg(all(feature = "cuda", feature = "cutlass"))]
        pub static CUTLASS_WORKSPACE: std::cell::RefCell<Option<CutlassWorkspace>> = const { std::cell::RefCell::new(None) };

        /// Specialized FP8 blockscale workspace.
        #[cfg(feature = "flashinfer")]
        pub static FLASHINFER_FP8_WORKSPACE: std::cell::RefCell<Option<FlashInferFp8Workspace>> = const { std::cell::RefCell::new(None) };
    }

    /// Initializes or retrieves the FlashInfer workspace for the given device.
    ///
    /// Returns (float_ptr, int_ptr, pinned_host_ptr, pinned_host_size).
    #[cfg(feature = "flashinfer")]
    pub fn get_or_init_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        for_cuda_graph: bool,
    ) -> Result<(
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
        usize,
    )> {
        let ws_cell = if for_cuda_graph {
            &WORKSPACE_GRAPH
        } else {
            &WORKSPACE
        };
        ws_cell.with(|ws| {
            let mut ws = ws.borrow_mut();
            let ordinal = dev.ordinal();

            let needs_init = match ws.as_ref() {
                None => true,
                Some(existing) => existing.device_ordinal != ordinal,
            };

            if needs_init {
                let float_buffer = unsafe { dev.alloc::<u8>(WORKSPACE_FLOAT_SIZE) }.w()?;
                let int_buffer = unsafe { dev.alloc::<u8>(WORKSPACE_INT_SIZE) }.w()?;
                let pinned_host = PinnedHostBuffer::new(WORKSPACE_INT_SIZE)?;
                *ws = Some(FlashInferWorkspace {
                    float_buffer,
                    int_buffer,
                    pinned_host,
                    device_ordinal: ordinal,
                });
            }

            let workspace = ws.as_ref().unwrap();
            Ok((
                *workspace.float_buffer.device_ptr() as *mut std::ffi::c_void,
                *workspace.int_buffer.device_ptr() as *mut std::ffi::c_void,
                workspace.pinned_host.as_ptr(),
                workspace.pinned_host.size(),
            ))
        })
    }

    /// Returns workspace pointers for FlashInfer plan operations.
    ///
    /// Returns (float_ptr, float_size, int_ptr, int_size, page_locked_ptr, page_locked_size).
    #[cfg(feature = "flashinfer")]
    pub fn get_plan_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        for_cuda_graph: bool,
    ) -> Result<(
        *mut std::ffi::c_void,
        usize,
        *mut std::ffi::c_void,
        usize,
        *mut std::ffi::c_void,
        usize,
    )> {
        let regions = workspace_regions();
        let (float_ptr, int_ptr, page_locked_ptr, page_locked_size) =
            get_or_init_workspace(dev, for_cuda_graph)?;
        Ok((
            add_workspace_offset(float_ptr, regions.plan_float),
            regions.plan_float.size,
            add_workspace_offset(int_ptr, regions.plan_int),
            regions.plan_int.size,
            page_locked_ptr,
            page_locked_size,
        ))
    }

    /// Returns the FlashInfer GEMM scratch region from the shared float buffer.
    ///
    /// This remains available for FlashInfer-owned scratch usage, but CUTLASS uses
    /// a separate dedicated workspace.
    #[cfg(feature = "flashinfer")]
    pub fn get_gemm_scratch_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
    ) -> Result<(*mut std::ffi::c_void, usize)> {
        let regions = workspace_regions();
        let (float_ptr, _, _, _) = get_or_init_workspace(dev, false)?;
        Ok((
            add_workspace_offset(float_ptr, regions.gemm_scratch_float),
            regions.gemm_scratch_float.size,
        ))
    }

    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    fn get_or_init_cutlass_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        required_size: usize,
    ) -> Result<(*mut std::ffi::c_void, usize)> {
        CUTLASS_WORKSPACE.with(|cell| {
            let mut slot = cell.borrow_mut();
            let ordinal = dev.ordinal();
            let alloc_size = required_size.max(CUTLASS_WORKSPACE_FALLBACK_SIZE).max(1);
            let needs_init = match slot.as_ref() {
                None => true,
                Some(existing) => existing.device_ordinal != ordinal || existing.size < alloc_size,
            };

            if needs_init {
                let buffer = unsafe { dev.alloc::<u8>(alloc_size) }.w()?;
                *slot = Some(CutlassWorkspace {
                    buffer,
                    size: alloc_size,
                    device_ordinal: ordinal,
                });
            }

            let ws = slot.as_ref().unwrap();
            Ok((*ws.buffer.device_ptr() as *mut std::ffi::c_void, ws.size))
        })
    }

    /// Returns the dedicated CUTLASS workspace.
    ///
    /// Even when `flashinfer` is enabled, CUTLASS does not alias the FlashInfer
    /// float buffer because that shared path is not stable for the FP8 CUTLASS GEMM kernels.
    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    pub fn get_cutlass_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        required_size: usize,
    ) -> Result<(*mut std::ffi::c_void, usize)> {
        get_or_init_cutlass_workspace(dev, required_size)
    }

    /// Alias for CUTLASS workspace in MoE context.
    #[cfg(all(feature = "cuda", feature = "cutlass"))]
    pub fn get_moe_cutlass_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        required_size: usize,
    ) -> Result<(*mut std::ffi::c_void, usize)> {
        get_cutlass_workspace(dev, required_size)
    }

    /// Thread-local pool for MoE NVFP4 hardware activation buffers.
    ///
    /// The Blackwell CUTLASS path allocates several large temporary tensors
    /// (gathered, act_packed, act_scales, rep_out, output) that scale with
    /// `size_m = num_tokens × topk`. By pooling these as grow-only raw
    /// `CudaSlice` buffers, we avoid per-call `cudaMalloc`/`cudaFree` overhead
    /// and reduce peak memory from allocation-deallocation hysteresis.
    #[cfg(feature = "cutlass")]
    pub struct MoeActivationPool {
        pub gathered: CudaSlice<u8>,
        pub gathered_bytes: usize,
        pub act_packed: CudaSlice<u8>,
        pub act_packed_bytes: usize,
        pub act_scales: CudaSlice<u8>,
        pub act_scales_bytes: usize,
        pub rep_out: CudaSlice<u8>,
        pub rep_out_bytes: usize,
        pub metadata: CudaSlice<u8>,
        pub metadata_bytes: usize,
        pub device_ordinal: usize,
    }

    #[cfg(feature = "cutlass")]
    thread_local! {
        pub static MOE_ACTIVATION_POOL: std::cell::RefCell<Option<MoeActivationPool>> = const { std::cell::RefCell::new(None) };
    }

    #[cfg(feature = "cutlass")]
    pub struct MoePoolBuffers {
        pub gathered_ptr: u64,
        pub act_packed_ptr: u64,
        pub act_scales_ptr: u64,
        pub rep_out_ptr: u64,
        pub metadata_ptr: u64,
    }

    #[cfg(feature = "cutlass")]
    pub fn get_moe_activation_pool(
        dev: &candle_core::cuda_backend::CudaDevice,
        gathered_bytes: usize,
        act_packed_bytes: usize,
        act_scales_bytes: usize,
        rep_out_bytes: usize,
        _output_bytes: usize,
        metadata_bytes: usize,
    ) -> Result<MoePoolBuffers> {
        MOE_ACTIVATION_POOL.with(|cell| {
            let mut slot = cell.borrow_mut();
            let ordinal = dev.ordinal();

            let needs_realloc = match slot.as_ref() {
                None => true,
                Some(p) => {
                    p.device_ordinal != ordinal
                        || p.gathered_bytes < gathered_bytes
                        || p.act_packed_bytes < act_packed_bytes
                        || p.act_scales_bytes < act_scales_bytes
                        || p.rep_out_bytes < rep_out_bytes
                        || p.metadata_bytes < metadata_bytes
                }
            };

            if needs_realloc {
                let old = slot.take();
                let (g_sz, ap_sz, as_sz, r_sz, m_sz) = if let Some(ref prev) = old {
                    (
                        gathered_bytes.max(prev.gathered_bytes),
                        act_packed_bytes.max(prev.act_packed_bytes),
                        act_scales_bytes.max(prev.act_scales_bytes),
                        rep_out_bytes.max(prev.rep_out_bytes),
                        metadata_bytes.max(prev.metadata_bytes),
                    )
                } else {
                    (
                        gathered_bytes,
                        act_packed_bytes,
                        act_scales_bytes,
                        rep_out_bytes,
                        metadata_bytes,
                    )
                };
                drop(old);
                let gathered = unsafe { dev.alloc::<u8>(g_sz.max(1)) }.w()?;
                let act_packed = unsafe { dev.alloc::<u8>(ap_sz.max(1)) }.w()?;
                let act_scales = unsafe { dev.alloc::<u8>(as_sz.max(1)) }.w()?;
                let rep_out = unsafe { dev.alloc::<u8>(r_sz.max(1)) }.w()?;
                let metadata = unsafe { dev.alloc::<u8>(m_sz.max(1)) }.w()?;
                *slot = Some(MoeActivationPool {
                    gathered,
                    gathered_bytes: g_sz,
                    act_packed,
                    act_packed_bytes: ap_sz,
                    act_scales,
                    act_scales_bytes: as_sz,
                    rep_out,
                    rep_out_bytes: r_sz,
                    metadata,
                    metadata_bytes: m_sz,
                    device_ordinal: ordinal,
                });
            }

            let pool = slot.as_ref().unwrap();
            Ok(MoePoolBuffers {
                gathered_ptr: *pool.gathered.device_ptr(),
                act_packed_ptr: *pool.act_packed.device_ptr(),
                act_scales_ptr: *pool.act_scales.device_ptr(),
                rep_out_ptr: *pool.rep_out.device_ptr(),
                metadata_ptr: *pool.metadata.device_ptr(),
            })
        })
    }

    /// Returns the FlashInfer FP8 blockscale workspace.
    ///
    /// This is a separate workspace used by the TRT-LLM blockscale FP8 GEMM runners,
    /// which have different allocation requirements than the standard GEMM scratch.
    #[cfg(feature = "flashinfer")]
    pub fn get_or_init_flashinfer_fp8_workspace(
        dev: &candle_core::cuda_backend::CudaDevice,
        required_size: usize,
    ) -> Result<(*mut std::ffi::c_void, usize)> {
        FLASHINFER_FP8_WORKSPACE.with(|cell| {
            let mut slot = cell.borrow_mut();
            let ordinal = dev.ordinal();

            let needs_init = match slot.as_ref() {
                None => true,
                Some(existing) => {
                    existing.device_ordinal != ordinal || existing.size < required_size
                }
            };

            if needs_init {
                let alloc_size = required_size.max(1);
                let buffer = unsafe { dev.alloc::<u8>(alloc_size) }.w()?;
                *slot = Some(FlashInferFp8Workspace {
                    buffer,
                    size: alloc_size,
                    device_ordinal: ordinal,
                });
            }

            let ws = slot.as_ref().unwrap();
            Ok((*ws.buffer.device_ptr() as *mut std::ffi::c_void, ws.size))
        })
    }
}

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_regions_do_not_overlap_and_fit() {
        let regions = workspace_regions();
        assert_eq!(regions.plan_float.offset, 0);
        assert_eq!(regions.plan_int.offset, 0);
        assert_eq!(regions.gemm_scratch_float.size, GEMM_SCRATCH_FLOAT_SIZE);
        assert!(regions.plan_float.size <= WORKSPACE_FLOAT_SIZE);
        assert!(regions.plan_int.size <= WORKSPACE_INT_SIZE);
        assert_eq!(
            regions.plan_float.offset + regions.plan_float.size,
            regions.gemm_scratch_float.offset
        );
        assert!(
            regions.gemm_scratch_float.offset + regions.gemm_scratch_float.size
                <= WORKSPACE_FLOAT_SIZE
        );
    }

    #[test]
    fn constants_are_consistent() {
        assert_eq!(WORKSPACE_FLOAT_SIZE, 512 * 1024 * 1024);
        assert_eq!(WORKSPACE_INT_SIZE, 128 * 1024 * 1024);
        assert_eq!(FLASHINFER_PLAN_FLOAT_SIZE, 256 * 1024 * 1024);
        assert_eq!(GEMM_SCRATCH_FLOAT_SIZE, 256 * 1024 * 1024);
        assert_eq!(GEMM_SCRATCH_FLOAT_OFFSET, FLASHINFER_PLAN_FLOAT_SIZE);
    }
}
