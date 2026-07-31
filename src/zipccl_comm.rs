//! ZipCCL-compressed NCCL AllReduce / AllGather.
//!
//! Prefill path:
//! - Exact compressed sizes are exchanged before variable-count NCCL sends.
//! - No `cudaMalloc` occurs inside NCCL groups; persistent scratch is reused.
//! - Decode is intentionally kept out of this module because metadata readback
//!   is not CUDA-graph safe.
//!
//! AllGather = compress → exchange dynamic sizes → variable-count send/recv →
//! decompress. ReduceScatter uses the same variable-count exchange followed by
//! a local FP32 reduction; AllReduce then performs a variable-count AllGather.

#[cfg(all(feature = "cuda", feature = "nccl"))]
use crate::zipccl;
#[cfg(all(feature = "cuda", feature = "nccl"))]
use candle_core::cuda_backend::cudarc::driver::result as driver_result;
#[cfg(all(feature = "cuda", feature = "nccl"))]
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DeviceSlice};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use candle_core::cuda_backend::cudarc::nccl::safe::{group_end, group_start, Comm, ReduceOp};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use candle_core::cuda_backend::{CudaDevice, WrapErr};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use candle_core::{DType, Result};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use half::{bf16, f16};
#[cfg(all(feature = "cuda", feature = "nccl"))]
use std::cell::RefCell;

#[cfg(all(feature = "cuda", feature = "nccl"))]
pub fn zipccl_active() -> bool {
    zipccl::is_enabled()
}

/// Native NCCL communicators used by the hierarchical multi-node path.
///
/// The global communicator remains the bootstrap communicator.  `local`
/// contains the GPUs in this node, while `inter_node` contains the same local
/// GPU rank across all nodes.  Keeping this state here means model layers keep
/// the same `&Comm` interface as native NCCL operations.
#[cfg(all(feature = "cuda", feature = "nccl"))]
struct HierarchicalTopology {
    global_world: usize,
    global_rank: usize,
    local_world: usize,
    num_nodes: usize,
    local: Comm,
    inter_node: Comm,
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
thread_local! {
    static HIERARCHICAL_TOPOLOGY: RefCell<Option<HierarchicalTopology>> =
        const { RefCell::new(None) };
}

/// Create the local and cross-node NCCL subcommunicators once during runner
/// initialization.  `num_shards` and `node_rank` come from the existing
/// `RunnerInitRequest.econfig`; no model or collective API changes are needed.
#[cfg(all(feature = "cuda", feature = "nccl"))]
pub fn configure_topology(
    global: &Comm,
    num_nodes: usize,
    node_rank: usize,
    local_world: usize,
) -> Result<()> {
    if num_nodes <= 1 {
        HIERARCHICAL_TOPOLOGY.with(|slot| *slot.borrow_mut() = None);
        return Ok(());
    }
    if local_world == 0
        || num_nodes
            .checked_mul(local_world)
            .is_none_or(|world| world != global.world_size())
        || node_rank >= num_nodes
    {
        candle_core::bail!(
            "invalid ZipCCL topology: global_world={}, num_nodes={}, node_rank={}, local_world={}",
            global.world_size(),
            num_nodes,
            node_rank,
            local_world
        );
    }

    let local_rank = global.rank() % local_world;
    let expected_rank = node_rank * local_world + local_rank;
    if expected_rank != global.rank() {
        candle_core::bail!(
            "ZipCCL rank mapping is not node-major: rank={}, expected={}",
            global.rank(),
            expected_rank
        );
    }

    // All ranks call the two splits in the same order.  The first groups a
    // node, the second groups matching local ranks across nodes.
    let local = global
        .split(node_rank as i32, local_rank as i32)
        .map_err(candle_core::Error::debug)?;
    let inter_node = global
        .split(local_rank as i32, node_rank as i32)
        .map_err(candle_core::Error::debug)?;
    if local.rank() != local_rank
        || local.world_size() != local_world
        || inter_node.rank() != node_rank
        || inter_node.world_size() != num_nodes
    {
        candle_core::bail!(
            "ZipCCL communicator split mismatch: local rank/size={}/{}, expected={}/{}; inter rank/size={}/{}, expected={}/{}",
            local.rank(),
            local.world_size(),
            local_rank,
            local_world,
            inter_node.rank(),
            inter_node.world_size(),
            node_rank,
            num_nodes
        );
    }

    HIERARCHICAL_TOPOLOGY.with(|slot| {
        *slot.borrow_mut() = Some(HierarchicalTopology {
            global_world: global.world_size(),
            global_rank: global.rank(),
            local_world,
            num_nodes,
            local,
            inter_node,
        });
    });
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn topology_matches(global: &Comm) -> bool {
    HIERARCHICAL_TOPOLOGY.with(|slot| {
        slot.borrow().as_ref().is_some_and(|topology| {
            topology.global_world == global.world_size() && topology.global_rank == global.rank()
        })
    })
}

/// Metadata exchange is worthwhile for large transfers, but it dominates
/// batch-1 decode collectives. Keep the adaptive switcher conservative for
/// inference; callers can lower this with XINFER_ZIPCCL_MIN_ELEMENTS when the
/// interconnect is slower than the GPU or when larger batches are used.
#[cfg(all(feature = "cuda", feature = "nccl"))]
pub fn eligible(n: usize) -> bool {
    const DEFAULT_MIN_ELEMENTS: usize = 131_072;
    let min_elements = std::env::var("XINFER_ZIPCCL_MIN_ELEMENTS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MIN_ELEMENTS);
    n >= min_elements
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn stream_of(dev: &CudaDevice) -> i64 {
    *dev.cu_stream() as i64
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn memcpy_dtod(dev: &CudaDevice, dst: u64, src: u64, bytes: usize) -> Result<()> {
    unsafe {
        driver_result::memcpy_dtod_async(dst, src, bytes, *dev.cu_stream())
            .map_err(candle_core::Error::wrap)?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn check_kernel(op: &str, status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        candle_core::bail!("{op} launch failed with CUDA error code {status}")
    }
}

/// Persistent scratch for ZipCCL collectives. Grown only when capacity is
/// insufficient; never allocated inside an NCCL group.
#[cfg(all(feature = "cuda", feature = "nccl"))]
struct ZipcclScratch {
    device_ordinal: usize,
    /// Max elements (full tensor `n`) this scratch supports.
    max_n: usize,
    world: usize,
    dtype: DType,
    /// send_pack: world * xfer(chunk) bytes — outbound compressed chunks.
    send_pack: CudaSlice<u8>,
    /// recv_pack: world * xfer(chunk) bytes — inbound compressed chunks.
    recv_pack: CudaSlice<u8>,
    /// f32 accumulator for reduce-scatter local sum (chunk elems).
    f32_acc: CudaSlice<f32>,
    /// reduced chunk in half precision (chunk elems).
    reduced_half: CudaSlice<u16>,
    /// all-gather compress buffer: xfer(chunk) bytes.
    ag_comp: CudaSlice<u8>,
    /// all-gather for VocabParallel path: compress full `n` + gather world*xfer(n).
    ag_full_comp: CudaSlice<u8>,
    /// Persistent receive storage for the full-tensor AllGather path.
    ag_full_recv: CudaSlice<u8>,
    /// Device-side dynamic zero-point counts, collected in one kernel launch.
    counts: CudaSlice<u32>,
    /// Reusable device buffers for the small NCCL metadata AllGather.
    size_send: CudaSlice<u32>,
    size_recv: CudaSlice<u32>,
    /// Scratch for per-payload exponent histogram selection.
    exponent_hist: CudaSlice<u32>,
    top7: CudaSlice<u8>,
    top7_valid: bool,
    payloads_since_calibration: usize,
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
impl ZipcclScratch {
    fn chunk_xfer(max_n: usize, world: usize, dtype: DType) -> Result<usize> {
        let chunk = (max_n / world).max(1);
        zipccl::cuda::xfer_bytes(chunk, dtype)
    }

    fn full_xfer(max_n: usize, dtype: DType) -> Result<usize> {
        zipccl::cuda::xfer_bytes(max_n.max(1), dtype)
    }

    fn new(dev: &CudaDevice, max_n: usize, world: usize, dtype: DType) -> Result<Self> {
        let max_n = max_n.max(world); // ensure chunk >= 1
        let chunk = max_n / world;
        let cx = Self::chunk_xfer(max_n, world, dtype)?;
        let fx = Self::full_xfer(max_n, dtype)?;
        Ok(Self {
            device_ordinal: dev.ordinal(),
            max_n,
            world,
            dtype,
            send_pack: unsafe { dev.alloc::<u8>(world * cx) }.w()?,
            recv_pack: unsafe { dev.alloc::<u8>(world * cx) }.w()?,
            f32_acc: unsafe { dev.alloc::<f32>(chunk.max(1)) }.w()?,
            reduced_half: unsafe { dev.alloc::<u16>(chunk.max(1)) }.w()?,
            ag_comp: unsafe { dev.alloc::<u8>(cx) }.w()?,
            ag_full_comp: unsafe { dev.alloc::<u8>(fx) }.w()?,
            ag_full_recv: unsafe { dev.alloc::<u8>(world * fx) }.w()?,
            counts: unsafe { dev.alloc::<u32>(world.max(1)) }.w()?,
            size_send: unsafe { dev.alloc::<u32>(world.max(1)) }.w()?,
            size_recv: unsafe { dev.alloc::<u32>(world.max(1) * world.max(1)) }.w()?,
            exponent_hist: unsafe { dev.alloc::<u32>(256) }.w()?,
            top7: unsafe { dev.alloc::<u8>(8) }.w()?,
            top7_valid: false,
            payloads_since_calibration: 0,
        })
    }

    fn ensure(&mut self, dev: &CudaDevice, n: usize, world: usize, dtype: DType) -> Result<()> {
        if self.device_ordinal == dev.ordinal()
            && self.world == world
            && self.dtype == dtype
            && self.max_n >= n
        {
            return Ok(());
        }
        // Grow (or recreate). Prefer doing this outside graph capture; decode
        // graphs use fixed `n` so first ensure during warmup sticks.
        let saved_top7 = if self.top7_valid && self.dtype == dtype {
            Some((
                dev.dtoh_sync_copy(&self.top7.slice(0..8)).w()?,
                self.payloads_since_calibration,
            ))
        } else {
            None
        };
        *self = Self::new(dev, n.max(self.max_n), world, dtype)?;
        if let Some((top7, payloads_since_calibration)) = saved_top7 {
            dev.htod_copy_into(top7, &mut self.top7).w()?;
            self.top7_valid = true;
            self.payloads_since_calibration = payloads_since_calibration;
        }
        Ok(())
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
pub fn reset_prefill_calibration() {
    ZIPCCL_SCRATCH.with(|cell| {
        if let Some(scratch) = cell.borrow_mut().as_mut() {
            scratch.top7_valid = false;
            scratch.payloads_since_calibration = 0;
        }
    });
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn compress_calibrated<O: DevicePtr<u8>>(
    scratch: &ZipcclScratch,
    input_ptr: u64,
    output: &O,
    n: usize,
    dtype: DType,
    stream: i64,
) -> Result<bool> {
    // A table from one layer is not representative of every layer, but a
    // full histogram for every payload costs more than it saves.  Refresh it
    // every eight all-reduce-sized payloads (24 compressed payloads: two
    // reduce-scatter chunks plus one all-gather per all-reduce).
    const CALIBRATION_INTERVAL: usize = 24;
    let recalibrate = std::env::var("XINFER_ZIPCCL_RECALIBRATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let refresh = !scratch.top7_valid
        || scratch.payloads_since_calibration >= CALIBRATION_INTERVAL
        || recalibrate;
    if !refresh {
        zipccl::cuda::compress_into_with_top7(input_ptr, output, n, dtype, &scratch.top7, stream)?;
    } else {
        zipccl::cuda::compress_into_dynamic(
            input_ptr,
            output,
            n,
            dtype,
            &scratch.exponent_hist,
            &scratch.top7,
            stream,
        )?;
    }
    Ok(refresh)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn compressed_sizes(
    dev: &CudaDevice,
    packed_ptr: u64,
    stride: usize,
    count: usize,
    n: usize,
    dtype: DType,
    counts: &CudaSlice<u32>,
) -> Result<Vec<usize>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if count > counts.len() {
        candle_core::bail!(
            "ZipCCL metadata buffer too small: count {} > {}",
            count,
            counts.len()
        );
    }
    let stream = stream_of(dev);
    unsafe {
        check_kernel(
            "zipccl_extract_counts",
            crate::kernels::ffi::zipccl_extract_counts(
                packed_ptr as *const _,
                stride as i32,
                *counts.device_ptr() as *mut _,
                count as i32,
                stream,
            ),
        )?;
    }
    // One synchronization covers every compression kernel and the metadata
    // extraction kernel issued by this collective.
    unsafe {
        driver_result::stream::synchronize(*dev.cu_stream()).map_err(candle_core::Error::wrap)?;
    }
    let zp_counts = dev.dtoh_sync_copy(&counts.slice(0..count)).w()?;
    zp_counts
        .into_iter()
        .map(|count| zipccl::cuda::compressed_bytes_from_zp_count(n, dtype, count as usize))
        .collect()
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
thread_local! {
    static ZIPCCL_SCRATCH: RefCell<Option<ZipcclScratch>> = const { RefCell::new(None) };
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn with_scratch<R>(
    dev: &CudaDevice,
    n: usize,
    world: usize,
    dtype: DType,
    f: impl FnOnce(&mut ZipcclScratch) -> Result<R>,
) -> Result<R> {
    ZIPCCL_SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(ZipcclScratch::new(dev, n, world, dtype)?);
        } else {
            slot.as_mut().unwrap().ensure(dev, n, world, dtype)?;
        }
        f(slot.as_mut().unwrap())
    })
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn exchange_sizes(
    comm: &Comm,
    dev: &CudaDevice,
    local: &[usize],
    size_send: &CudaSlice<u32>,
    size_recv: &mut CudaSlice<u32>,
) -> Result<Vec<usize>> {
    let mut local_u32 = Vec::with_capacity(local.len());
    for &size in local {
        local_u32.push(u32::try_from(size).map_err(|_| {
            candle_core::Error::Msg(format!("ZipCCL payload is too large: {size} bytes"))
        })?);
    }
    if local.len() > size_send.len() || local.len() * comm.world_size() > size_recv.len() {
        candle_core::bail!("ZipCCL metadata scratch is too small");
    }
    unsafe {
        driver_result::memcpy_htod_sync(*size_send.device_ptr() as u64, &local_u32)
            .map_err(candle_core::Error::wrap)?;
    }
    let send = size_send.slice(0..local.len());
    let mut recv = size_recv.slice_mut(0..local.len() * comm.world_size());
    comm.all_gather(&send, &mut recv)
        .map_err(candle_core::Error::debug)?;
    unsafe {
        driver_result::stream::synchronize(*dev.cu_stream()).map_err(candle_core::Error::wrap)?;
    }
    let mut sizes = vec![0u32; local.len() * comm.world_size()];
    unsafe {
        driver_result::memcpy_dtoh_sync(&mut sizes, *size_recv.device_ptr() as u64)
            .map_err(candle_core::Error::wrap)?;
    }
    Ok(sizes.into_iter().map(|size| size as usize).collect())
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn variable_all_gather_into(
    comm: &Comm,
    dev: &CudaDevice,
    compressed: &CudaSlice<u8>,
    local_size: usize,
    sizes: &[usize],
    gathered: &mut CudaSlice<u8>,
) -> Result<usize> {
    let world = comm.world_size();
    if sizes.len() != world {
        candle_core::bail!(
            "ZipCCL size exchange returned {} entries for world size {}",
            sizes.len(),
            world
        );
    }
    if local_size > compressed.len() {
        candle_core::bail!(
            "ZipCCL local payload {} exceeds buffer {}",
            local_size,
            compressed.len()
        );
    }
    let mut offsets = vec![0usize; world];
    for rank in 1..world {
        offsets[rank] = offsets[rank - 1] + sizes[rank - 1];
    }
    let total = offsets[world - 1] + sizes[world - 1];
    if total > gathered.len() {
        candle_core::bail!(
            "ZipCCL gathered payload {} exceeds persistent buffer {}",
            total,
            gathered.len()
        );
    }
    memcpy_dtod(
        dev,
        *gathered.device_ptr() as u64 + offsets[comm.rank()] as u64,
        *compressed.device_ptr() as u64,
        local_size,
    )?;

    group_start().map_err(candle_core::Error::debug)?;
    for peer in 0..world {
        if peer == comm.rank() {
            continue;
        }
        let send = compressed.slice(0..local_size);
        let mut recv = gathered.slice_mut(offsets[peer]..offsets[peer] + sizes[peer]);
        comm.send(&send, peer as i32)
            .map_err(candle_core::Error::debug)?;
        comm.recv(&mut recv, peer as i32)
            .map_err(candle_core::Error::debug)?;
    }
    group_end().map_err(candle_core::Error::debug)?;
    Ok(total)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn all_gather_zipped(
    comm: &Comm,
    dev: &CudaDevice,
    src_ptr: u64,
    n: usize,
    dtype: DType,
) -> Result<CudaSlice<u16>> {
    let world = comm.world_size();
    let stream = stream_of(dev);
    if n == 0 {
        candle_core::bail!("ZipCCL all_gather does not support an empty tensor");
    }

    with_scratch(dev, n, world, dtype, |scratch| {
        let comp_len = scratch.ag_full_comp.len();
        let refreshed =
            compress_calibrated(scratch, src_ptr, &scratch.ag_full_comp, n, dtype, stream)?;
        if refreshed {
            scratch.top7_valid = true;
            scratch.payloads_since_calibration = 0;
        } else {
            scratch.payloads_since_calibration += 1;
        }
        let local_size = compressed_sizes(
            dev,
            *scratch.ag_full_comp.device_ptr() as u64,
            comp_len,
            1,
            n,
            dtype,
            &scratch.counts,
        )?[0];
        let sizes = exchange_sizes(
            comm,
            dev,
            &[local_size],
            &scratch.size_send,
            &mut scratch.size_recv,
        )?;
        variable_all_gather_into(
            comm,
            dev,
            &scratch.ag_full_comp,
            local_size,
            &sizes,
            &mut scratch.ag_full_recv,
        )?;

        // Final output — same pattern as native AllGather (mempool-friendly).
        let out = unsafe { dev.alloc::<u16>(n * world) }.w()?;
        let out_ptr = *out.device_ptr() as u64;
        let base = *scratch.ag_full_recv.device_ptr() as u64;
        for r in 0..world {
            zipccl::cuda::decompress_ptr(
                base + sizes[..r].iter().sum::<usize>() as u64,
                out_ptr + (r * n * 2) as u64,
                n,
                dtype,
                stream,
            )?;
        }
        Ok(out)
    })
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn all_reduce_bf16_hierarchical(
    topology: &HierarchicalTopology,
    dev: &CudaDevice,
    src: &impl DevicePtr<bf16>,
    n: usize,
) -> Result<CudaSlice<bf16>> {
    if n == 0 || n % topology.local_world != 0 {
        candle_core::bail!(
            "hierarchical ZipCCL all_reduce requires n divisible by local world (n={}, local_world={})",
            n,
            topology.local_world
        );
    }
    let local_chunk = n / topology.local_world;
    let mut local_reduced = unsafe { dev.alloc::<bf16>(local_chunk) }.w()?;
    topology
        .local
        .reduce_scatter(src, &mut local_reduced, &ReduceOp::Sum)
        .map_err(candle_core::Error::debug)?;

    // Only this communicator crosses nodes.  The payload is already reduced
    // over all GPUs on this node, so the compressed exchange is exactly one
    // local-rank lane per node rather than a global all-to-all.
    let inter_u16 = all_reduce_zipped(
        &topology.inter_node,
        dev,
        *local_reduced.device_ptr() as u64,
        local_chunk,
        DType::BF16,
    )?;
    let inter_reduced =
        unsafe { std::mem::transmute::<CudaSlice<u16>, CudaSlice<bf16>>(inter_u16) };
    let mut out = unsafe { dev.alloc::<bf16>(n) }.w()?;
    topology
        .local
        .all_gather(&inter_reduced, &mut out)
        .map_err(candle_core::Error::debug)?;
    Ok(out)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn all_reduce_f16_hierarchical(
    topology: &HierarchicalTopology,
    dev: &CudaDevice,
    src: &impl DevicePtr<f16>,
    n: usize,
) -> Result<CudaSlice<f16>> {
    if n == 0 || n % topology.local_world != 0 {
        candle_core::bail!(
            "hierarchical ZipCCL all_reduce requires n divisible by local world (n={}, local_world={})",
            n,
            topology.local_world
        );
    }
    let local_chunk = n / topology.local_world;
    let mut local_reduced = unsafe { dev.alloc::<f16>(local_chunk) }.w()?;
    topology
        .local
        .reduce_scatter(src, &mut local_reduced, &ReduceOp::Sum)
        .map_err(candle_core::Error::debug)?;
    let inter_u16 = all_reduce_zipped(
        &topology.inter_node,
        dev,
        *local_reduced.device_ptr() as u64,
        local_chunk,
        DType::F16,
    )?;
    let inter_reduced = unsafe { std::mem::transmute::<CudaSlice<u16>, CudaSlice<f16>>(inter_u16) };
    let mut out = unsafe { dev.alloc::<f16>(n) }.w()?;
    topology
        .local
        .all_gather(&inter_reduced, &mut out)
        .map_err(candle_core::Error::debug)?;
    Ok(out)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn all_gather_bf16_hierarchical(
    topology: &HierarchicalTopology,
    dev: &CudaDevice,
    src: &impl DevicePtr<bf16>,
    n: usize,
) -> Result<CudaSlice<bf16>> {
    if n == 0 {
        candle_core::bail!("hierarchical ZipCCL all_gather does not support an empty tensor");
    }
    let inter_u16 = all_gather_zipped(
        &topology.inter_node,
        dev,
        *src.device_ptr() as u64,
        n,
        DType::BF16,
    )?;
    let inter_payload =
        unsafe { std::mem::transmute::<CudaSlice<u16>, CudaSlice<bf16>>(inter_u16) };
    let lane_len = n * topology.num_nodes;
    let mut local_gathered = unsafe { dev.alloc::<bf16>(lane_len * topology.local_world) }.w()?;
    topology
        .local
        .all_gather(&inter_payload, &mut local_gathered)
        .map_err(candle_core::Error::debug)?;

    // local all-gather produces [local_rank][node]. Native global NCCL order
    // is [node][local_rank], so transpose the small rank/node index on-device.
    let out = unsafe { dev.alloc::<bf16>(n * topology.global_world) }.w()?;
    for local_rank in 0..topology.local_world {
        for node in 0..topology.num_nodes {
            memcpy_dtod(
                dev,
                *out.device_ptr() as u64
                    + ((node * topology.local_world + local_rank) * n * 2) as u64,
                *local_gathered.device_ptr() as u64
                    + ((local_rank * topology.num_nodes + node) * n * 2) as u64,
                n * 2,
            )?;
        }
    }
    Ok(out)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn all_gather_f16_hierarchical(
    topology: &HierarchicalTopology,
    dev: &CudaDevice,
    src: &impl DevicePtr<f16>,
    n: usize,
) -> Result<CudaSlice<f16>> {
    if n == 0 {
        candle_core::bail!("hierarchical ZipCCL all_gather does not support an empty tensor");
    }
    let inter_u16 = all_gather_zipped(
        &topology.inter_node,
        dev,
        *src.device_ptr() as u64,
        n,
        DType::F16,
    )?;
    let inter_payload = unsafe { std::mem::transmute::<CudaSlice<u16>, CudaSlice<f16>>(inter_u16) };
    let lane_len = n * topology.num_nodes;
    let mut local_gathered = unsafe { dev.alloc::<f16>(lane_len * topology.local_world) }.w()?;
    topology
        .local
        .all_gather(&inter_payload, &mut local_gathered)
        .map_err(candle_core::Error::debug)?;
    let out = unsafe { dev.alloc::<f16>(n * topology.global_world) }.w()?;
    for local_rank in 0..topology.local_world {
        for node in 0..topology.num_nodes {
            memcpy_dtod(
                dev,
                *out.device_ptr() as u64
                    + ((node * topology.local_world + local_rank) * n * 2) as u64,
                *local_gathered.device_ptr() as u64
                    + ((local_rank * topology.num_nodes + node) * n * 2) as u64,
                n * 2,
            )?;
        }
    }
    Ok(out)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
pub fn all_gather_bf16_zipped(
    comm: &Comm,
    dev: &CudaDevice,
    src: &impl DevicePtr<bf16>,
    n: usize,
) -> Result<CudaSlice<bf16>> {
    if topology_matches(comm) {
        return HIERARCHICAL_TOPOLOGY.with(|slot| {
            let topology = slot.borrow();
            all_gather_bf16_hierarchical(topology.as_ref().unwrap(), dev, src, n)
        });
    }
    let out_u16 = all_gather_zipped(comm, dev, *src.device_ptr() as u64, n, DType::BF16)?;
    Ok(unsafe { std::mem::transmute::<CudaSlice<u16>, CudaSlice<bf16>>(out_u16) })
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
pub fn all_gather_f16_zipped(
    comm: &Comm,
    dev: &CudaDevice,
    src: &impl DevicePtr<f16>,
    n: usize,
) -> Result<CudaSlice<f16>> {
    if topology_matches(comm) {
        return HIERARCHICAL_TOPOLOGY.with(|slot| {
            let topology = slot.borrow();
            all_gather_f16_hierarchical(topology.as_ref().unwrap(), dev, src, n)
        });
    }
    let out_u16 = all_gather_zipped(comm, dev, *src.device_ptr() as u64, n, DType::F16)?;
    Ok(unsafe { std::mem::transmute::<CudaSlice<u16>, CudaSlice<f16>>(out_u16) })
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn all_reduce_zipped(
    comm: &Comm,
    dev: &CudaDevice,
    src_ptr: u64,
    n: usize,
    dtype: DType,
) -> Result<CudaSlice<u16>> {
    let world = comm.world_size();
    let rank = comm.rank();
    let stream = stream_of(dev);

    if world <= 1 {
        let dst = unsafe { dev.alloc::<u16>(n) }.w()?;
        memcpy_dtod(dev, *dst.device_ptr() as u64, src_ptr, n * 2)?;
        return Ok(dst);
    }

    if n == 0 || n % world != 0 {
        candle_core::bail!(
            "ZipCCL all_reduce requires a non-empty tensor divisible by world size (n={}, world={})",
            n,
            world
        );
    }

    let chunk = n / world;
    let elem = 2usize;
    let id = zipccl::cuda::dtype_id(dtype)?;

    with_scratch(dev, n, world, dtype, |scratch| {
        let max_chunk = scratch.max_n / world;
        let max_xfer = zipccl::cuda::xfer_bytes(max_chunk.max(1), dtype)?;

        // --- Compress each outbound chunk into send_pack (no alloc) ---
        for p in 0..world {
            let ptr = src_ptr + (p * chunk * elem) as u64;
            let start = p * max_xfer;
            let dst_slice = scratch.send_pack.slice(start..start + max_xfer);
            let refreshed = compress_calibrated(scratch, ptr, &dst_slice, chunk, dtype, stream)?;
            drop(dst_slice);
            if refreshed {
                scratch.top7_valid = true;
                scratch.payloads_since_calibration = 0;
            } else {
                scratch.payloads_since_calibration += 1;
            }
        }
        let local_sizes = compressed_sizes(
            dev,
            *scratch.send_pack.device_ptr() as u64,
            max_xfer,
            world,
            chunk,
            dtype,
            &scratch.counts,
        )?;

        // The first collective is the paper's metadata exchange.  The matrix
        // is rank-major: sizes[src * world + dst].
        let sizes = exchange_sizes(
            comm,
            dev,
            &local_sizes,
            &scratch.size_send,
            &mut scratch.size_recv,
        )?;

        // --- Variable-count compressed ReduceScatter (NCCL send/recv) ---
        let mut recv_offsets = vec![0usize; world];
        let mut recv_sizes = vec![0usize; world];
        let mut recv_total = 0usize;
        for src in 0..world {
            if src == rank {
                continue;
            }
            recv_offsets[src] = recv_total;
            recv_sizes[src] = sizes[src * world + rank];
            recv_total += recv_sizes[src];
        }
        group_start().map_err(candle_core::Error::debug)?;
        for peer in 0..world {
            if peer == rank {
                continue;
            }
            let send_start = peer * max_xfer;
            let send_view = scratch
                .send_pack
                .slice(send_start..send_start + local_sizes[peer]);
            let mut recv_view = scratch
                .recv_pack
                .slice_mut(recv_offsets[peer]..recv_offsets[peer] + recv_sizes[peer]);
            comm.send(&send_view, peer as i32)
                .map_err(candle_core::Error::debug)?;
            comm.recv(&mut recv_view, peer as i32)
                .map_err(candle_core::Error::debug)?;
        }
        group_end().map_err(candle_core::Error::debug)?;

        // --- Local FP32 sum (device kernels only; memset via async kernel path) ---
        // Clear accumulator with async memset (graph-safe).
        unsafe {
            driver_result::memset_d8_async(
                *scratch.f32_acc.device_ptr() as u64,
                0,
                chunk * std::mem::size_of::<f32>(),
                *dev.cu_stream(),
            )
            .map_err(candle_core::Error::wrap)?;
        }
        unsafe {
            check_kernel(
                "zipccl_half_to_f32",
                crate::kernels::ffi::zipccl_half_to_f32(
                    (src_ptr + (rank * chunk * elem) as u64) as *const _,
                    *scratch.f32_acc.device_ptr() as *mut _,
                    chunk as i32,
                    id,
                    stream,
                ),
            )?;
        }

        for src in 0..world {
            if src == rank {
                continue;
            }
            let cptr = *scratch.recv_pack.device_ptr() as u64 + recv_offsets[src] as u64;
            unsafe {
                check_kernel(
                    "zipccl_decompress_add_f32",
                    crate::kernels::ffi::zipccl_decompress_add_f32(
                        cptr as *const _,
                        *scratch.f32_acc.device_ptr() as *mut _,
                        chunk as i32,
                        id,
                        stream,
                    ),
                )?;
            }
        }

        unsafe {
            check_kernel(
                "zipccl_f32_to_half",
                crate::kernels::ffi::zipccl_f32_to_half(
                    *scratch.f32_acc.device_ptr() as *const _,
                    *scratch.reduced_half.device_ptr() as *mut _,
                    chunk as i32,
                    id,
                    stream,
                ),
            )?;
        }

        // --- Compressed variable-count AllGather of reduced chunks ---
        let refreshed = compress_calibrated(
            scratch,
            *scratch.reduced_half.device_ptr() as u64,
            &scratch.ag_comp,
            chunk,
            dtype,
            stream,
        )?;
        if refreshed {
            scratch.top7_valid = true;
            scratch.payloads_since_calibration = 0;
        } else {
            scratch.payloads_since_calibration += 1;
        }
        let reduced_size = compressed_sizes(
            dev,
            *scratch.ag_comp.device_ptr() as u64,
            scratch.ag_comp.len(),
            1,
            chunk,
            dtype,
            &scratch.counts,
        )?[0];
        let reduced_sizes = exchange_sizes(
            comm,
            dev,
            &[reduced_size],
            &scratch.size_send,
            &mut scratch.size_recv,
        )?;
        variable_all_gather_into(
            comm,
            dev,
            &scratch.ag_comp,
            reduced_size,
            &reduced_sizes,
            &mut scratch.recv_pack,
        )?;

        let out = unsafe { dev.alloc::<u16>(n) }.w()?;
        let out_ptr = *out.device_ptr() as u64;
        let base = *scratch.recv_pack.device_ptr() as u64;
        for r in 0..world {
            if r == rank {
                memcpy_dtod(
                    dev,
                    out_ptr + (r * chunk * 2) as u64,
                    *scratch.reduced_half.device_ptr() as u64,
                    chunk * 2,
                )?;
            } else {
                zipccl::cuda::decompress_ptr(
                    base + (reduced_sizes[..r].iter().sum::<usize>()) as u64,
                    out_ptr + (r * chunk * 2) as u64,
                    chunk,
                    dtype,
                    stream,
                )?;
            }
        }
        Ok(out)
    })
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
pub fn all_reduce_bf16_zipped(
    comm: &Comm,
    dev: &CudaDevice,
    src: &impl DevicePtr<bf16>,
    n: usize,
) -> Result<CudaSlice<bf16>> {
    if topology_matches(comm) {
        return HIERARCHICAL_TOPOLOGY.with(|slot| {
            let topology = slot.borrow();
            all_reduce_bf16_hierarchical(topology.as_ref().unwrap(), dev, src, n)
        });
    }
    let out = all_reduce_zipped(comm, dev, *src.device_ptr() as u64, n, DType::BF16)?;
    Ok(unsafe { std::mem::transmute::<CudaSlice<u16>, CudaSlice<bf16>>(out) })
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
pub fn all_reduce_f16_zipped(
    comm: &Comm,
    dev: &CudaDevice,
    src: &impl DevicePtr<f16>,
    n: usize,
) -> Result<CudaSlice<f16>> {
    if topology_matches(comm) {
        return HIERARCHICAL_TOPOLOGY.with(|slot| {
            let topology = slot.borrow();
            all_reduce_f16_hierarchical(topology.as_ref().unwrap(), dev, src, n)
        });
    }
    let out = all_reduce_zipped(comm, dev, *src.device_ptr() as u64, n, DType::F16)?;
    Ok(unsafe { std::mem::transmute::<CudaSlice<u16>, CudaSlice<f16>>(out) })
}

#[cfg(test)]
mod tests {
    use candle_core::backend::BackendDevice;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::{CudaDevice, WrapErr};
    use candle_core::DType;
    use half::{bf16, f16};

    fn round_trip<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + Copy>(
        input: &[T],
        dtype: DType,
    ) -> candle_core::Result<Vec<T>> {
        let dev = CudaDevice::new(0)?;
        let input_dev = dev.htod_sync_copy(input).w()?;
        let transfer_bytes = crate::zipccl::cuda::xfer_bytes(input.len(), dtype)?;
        let compressed = unsafe { dev.alloc::<u8>(transfer_bytes) }.w()?;
        let hist = unsafe { dev.alloc::<u32>(256) }.w()?;
        let top7 = unsafe { dev.alloc::<u8>(8) }.w()?;
        crate::zipccl::cuda::compress_into_dynamic(
            *input_dev.device_ptr() as u64,
            &compressed,
            input.len(),
            dtype,
            &hist,
            &top7,
            *dev.cu_stream() as i64,
        )?;
        assert_eq!(
            transfer_bytes,
            crate::zipccl::cuda::xfer_bytes(input.len(), dtype)?
        );
        let output_dev = unsafe { dev.alloc::<T>(input.len()) }.w()?;
        crate::zipccl::cuda::decompress_buf(
            &compressed,
            *output_dev.device_ptr() as u64,
            input.len(),
            dtype,
            *dev.cu_stream() as i64,
        )?;
        dev.synchronize()?;
        Ok(dev.dtoh_sync_copy(&output_dev).w()?)
    }

    #[test]
    fn zipccl_bf16_round_trip_preserves_bits() -> candle_core::Result<()> {
        let input: Vec<bf16> = (0..1024)
            .map(|i| {
                // Include a distribution outside the hard-coded top-7
                // exponents.  A codec that silently truncates zero-point
                // exponents must fail this exact round-trip.
                if i % 2 == 0 {
                    bf16::from_bits(0x2f80) // approximately 1e-10
                } else {
                    bf16::from_f32(((i % 17) as f32 - 8.0) / 8.0)
                }
            })
            .collect();
        let output = round_trip(&input, DType::BF16)?;
        assert_eq!(input, output);
        Ok(())
    }

    #[test]
    fn zipccl_f16_round_trip_preserves_bits() -> candle_core::Result<()> {
        let input: Vec<f16> = (0..1024)
            .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) / 8.0))
            .collect();
        let output = round_trip(&input, DType::F16)?;
        assert_eq!(input, output);
        Ok(())
    }

    #[test]
    fn zipccl_rejects_unsupported_dtype_and_non_divisible_reduce() {
        assert!(crate::zipccl::cuda::dtype_id(DType::F32).is_err());
        assert_eq!(crate::zipccl::cuda::num_blocks(0), 0);
        assert_eq!(crate::zipccl::cuda::num_blocks(257), 2);
    }

    #[test]
    fn zipccl_two_rank_all_reduce_matches_native_nccl() -> candle_core::Result<()> {
        use candle_core::cuda_backend::cudarc::nccl::safe::{Comm, ReduceOp};

        let n = 1024;
        let id = candle_core::cuda_backend::cudarc::nccl::safe::Id::new()
            .map_err(candle_core::Error::debug)?;
        let threads: Vec<_> = (0..2)
            .map(|rank| {
                std::thread::spawn(move || {
                    let dev = CudaDevice::new_with_stream(rank).unwrap();
                    let comm = Comm::from_rank(dev.cuda_device(), rank, 2, id).unwrap();
                    let input: Vec<bf16> = (0..n)
                        .map(|i| {
                            if i % 2 == 0 {
                                bf16::from_bits(if rank == 0 { 0x2f80 } else { 0x3080 })
                            } else if rank == 0 {
                                bf16::from_f32(i as f32 / 32.0)
                            } else {
                                bf16::from_f32(-(i as f32) / 64.0)
                            }
                        })
                        .collect();
                    let src = dev.htod_sync_copy(&input).unwrap();
                    let zipped = super::all_reduce_bf16_zipped(&comm, &dev, &src, n).unwrap();
                    dev.synchronize().unwrap();
                    let got = dev.dtoh_sync_copy(&zipped).unwrap();

                    let mut native = unsafe { dev.alloc::<bf16>(n) }.unwrap();
                    comm.all_reduce(&src, &mut native, &ReduceOp::Sum).unwrap();
                    dev.synchronize().unwrap();
                    let expected = dev.dtoh_sync_copy(&native).unwrap();
                    (got, expected)
                })
            })
            .collect();
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        for (rank, (got, expected)) in results.iter().into_iter().enumerate() {
            if let Some((i, (g, e))) = got
                .iter()
                .zip(expected.iter())
                .enumerate()
                .find(|(_, (g, e))| g != e)
            {
                panic!("rank {rank} mismatch at {i}: got {g:?}, native {e:?}");
            }
        }
        assert_eq!(results[0].0, results[1].0);
        Ok(())
    }

    #[test]
    fn zipccl_two_rank_all_gather_matches_native_nccl() -> candle_core::Result<()> {
        use candle_core::cuda_backend::cudarc::nccl::safe::{Comm, Id};

        let n = 513;
        let id = Id::new().map_err(candle_core::Error::debug)?;
        let threads: Vec<_> = (0..2)
            .map(|rank| {
                std::thread::spawn(move || {
                    let dev = CudaDevice::new_with_stream(rank).unwrap();
                    let comm = Comm::from_rank(dev.cuda_device(), rank, 2, id).unwrap();
                    let input: Vec<bf16> = (0..n)
                        .map(|i| {
                            if i % 3 == 0 {
                                bf16::from_bits(0x2f80 + rank as u16)
                            } else {
                                bf16::from_f32((rank as f32 + 1.0) * i as f32 / 17.0)
                            }
                        })
                        .collect();
                    let src = dev.htod_sync_copy(&input).unwrap();
                    let zipped = super::all_gather_bf16_zipped(&comm, &dev, &src, n).unwrap();
                    dev.synchronize().unwrap();
                    let got = dev.dtoh_sync_copy(&zipped).unwrap();

                    let mut native = unsafe { dev.alloc::<bf16>(n * 2) }.unwrap();
                    comm.all_gather(&src, &mut native).unwrap();
                    dev.synchronize().unwrap();
                    let expected = dev.dtoh_sync_copy(&native).unwrap();
                    (got, expected)
                })
            })
            .collect();
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(results[0].0, results[0].1);
        assert_eq!(results[1].0, results[1].1);
        assert_eq!(results[0].0, results[1].0);
        Ok(())
    }

    #[test]
    fn zipccl_two_node_hierarchy_matches_native_nccl() -> candle_core::Result<()> {
        use candle_core::cuda_backend::cudarc::nccl::safe::{Comm, Id, ReduceOp};

        // Model the smallest multi-node topology: one GPU per node. This
        // exercises both ncclCommSplit lanes and the hierarchical dispatch;
        // the 2x8 production topology uses the same grouping with
        // local_world=8.
        let id = Id::new().map_err(candle_core::Error::debug)?;
        let threads: Vec<_> = (0..2)
            .map(|rank| {
                std::thread::spawn(move || {
                    let dev = CudaDevice::new_with_stream(rank).unwrap();
                    let comm = Comm::from_rank(dev.cuda_device(), rank, 2, id).unwrap();
                    super::configure_topology(&comm, 2, rank, 1).unwrap();

                    let reduce_input: Vec<bf16> = (0..1024)
                        .map(|i| bf16::from_f32((rank as f32 + 1.0) * i as f32 / 32.0))
                        .collect();
                    let reduce_src = dev.htod_sync_copy(&reduce_input).unwrap();
                    let zipped_reduce =
                        super::all_reduce_bf16_zipped(&comm, &dev, &reduce_src, reduce_input.len())
                            .unwrap();
                    dev.synchronize().unwrap();
                    let got_reduce = dev.dtoh_sync_copy(&zipped_reduce).unwrap();
                    let mut native_reduce =
                        unsafe { dev.alloc::<bf16>(reduce_input.len()) }.unwrap();
                    comm.all_reduce(&reduce_src, &mut native_reduce, &ReduceOp::Sum)
                        .unwrap();
                    dev.synchronize().unwrap();
                    let expected_reduce = dev.dtoh_sync_copy(&native_reduce).unwrap();

                    let gather_input: Vec<bf16> = (0..513)
                        .map(|i| bf16::from_f32(rank as f32 + i as f32 / 17.0))
                        .collect();
                    let gather_src = dev.htod_sync_copy(&gather_input).unwrap();
                    let zipped_gather =
                        super::all_gather_bf16_zipped(&comm, &dev, &gather_src, gather_input.len())
                            .unwrap();
                    dev.synchronize().unwrap();
                    let got_gather = dev.dtoh_sync_copy(&zipped_gather).unwrap();
                    let mut native_gather =
                        unsafe { dev.alloc::<bf16>(gather_input.len() * 2) }.unwrap();
                    comm.all_gather(&gather_src, &mut native_gather).unwrap();
                    dev.synchronize().unwrap();
                    let expected_gather = dev.dtoh_sync_copy(&native_gather).unwrap();
                    (got_reduce, expected_reduce, got_gather, expected_gather)
                })
            })
            .collect();
        let results: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(results[0].0, results[0].1);
        assert_eq!(results[1].0, results[1].1);
        assert_eq!(results[0].2, results[0].3);
        assert_eq!(results[1].2, results[1].3);
        assert_eq!(results[0].2, results[1].2);
        Ok(())
    }
}
