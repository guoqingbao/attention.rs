#[allow(unused_imports)]
use candle_core::{backend::BackendDevice, Device, Result, Storage, Tensor};
use std::collections::HashMap;

pub fn swap_blocks(
    src: &Tensor,
    dst: &Tensor,
    block_mapping: &HashMap<usize, usize>,
) -> Result<()> {
    use candle_core::DType;
    use half::{bf16, f16};
    #[cfg(feature = "cuda")]
    fn call_fwd<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::WithDType,
    >(
        src: &Tensor,
        dst: &Tensor,
        block_mapping: &HashMap<usize, usize>,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{result, CudaSlice, DevicePtr};
        use std::slice;
        let block_size_elements = src.elem_count() / src.dim(0)?;
        let (src_storage, _) = src.storage_and_layout();
        let (dst_storage, _) = dst.storage_and_layout();
        let dtype_size = src.dtype().size_in_bytes();

        match (src.device(), dst.device()) {
            (Device::Cpu, Device::Cuda(dst_dev)) => {
                let Storage::Cpu(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Cuda(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };
                let cpu_num_blocks = src.dim(0)?;
                let gpu_num_blocks = dst.dim(0)?;

                let dst_ptr = *dst_storage.as_cuda_slice::<T>()?.device_ptr();
                let src_slice: &[T] = src_storage.as_slice()?;

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset: usize = src_block_number * block_size_elements;
                    assert!(
                        *src_block_number < cpu_num_blocks,
                        "Invalid cpu block {} / {}",
                        src_block_number,
                        cpu_num_blocks
                    );
                    assert!(
                        *dst_block_number < gpu_num_blocks,
                        "Invalid gpu block {} / {}",
                        dst_block_number,
                        gpu_num_blocks
                    );

                    assert!(
                        src_offset + block_size_elements <= src_slice.len(),
                        "Invalid cpu kvcache block {} for offload",
                        src_block_number
                    );

                    let dst_offset: u64 = (dst_block_number * block_size_elements * dtype_size)
                        .try_into()
                        .unwrap();
                    let dst_slice: std::mem::ManuallyDrop<CudaSlice<T>> = unsafe {
                        let slice = dst_dev.upgrade_device_ptr(
                            dst_ptr.wrapping_add(dst_offset),
                            block_size_elements * dtype_size,
                        );
                        std::mem::ManuallyDrop::new(slice)
                    };

                    unsafe {
                        result::memcpy_htod_async(
                            *dst_slice.device_ptr(),
                            &src_slice[src_offset..src_offset + block_size_elements],
                            *dst_dev.cu_stream(),
                        )
                        .map_err(candle_core::Error::wrap)?
                    }
                }
                dst_dev.synchronize()
            }
            (Device::Cuda(src_dev), Device::Cpu) => {
                let Storage::Cuda(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Cpu(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };
                let gpu_num_blocks = src.dim(0)?;
                let cpu_num_blocks = dst.dim(0)?;

                let src_ptr = src_storage
                    .as_cuda_slice::<T>()
                    .map_err(candle_core::Error::wrap)?
                    .device_ptr();
                let dst_slice: &[T] = dst_storage.as_slice().map_err(candle_core::Error::wrap)?;
                let ptr = dst_slice.as_ptr() as *mut u8;

                for (src_block_number, dst_block_number) in block_mapping {
                    assert!(
                        *src_block_number < gpu_num_blocks,
                        "Invalid gpu block {} / {}",
                        src_block_number,
                        gpu_num_blocks
                    );
                    assert!(
                        *dst_block_number < cpu_num_blocks,
                        "Invalid cpu block {} / {}",
                        dst_block_number,
                        cpu_num_blocks
                    );

                    let src_offset: u64 = (src_block_number * block_size_elements * dtype_size)
                        .try_into()
                        .unwrap();
                    let dst_offset: usize = (dst_block_number * block_size_elements * dtype_size)
                        .try_into()
                        .unwrap();
                    let dst_slice = unsafe {
                        slice::from_raw_parts_mut(
                            ptr.wrapping_add(dst_offset),
                            block_size_elements * dtype_size,
                        )
                    };

                    let src_slice = unsafe {
                        let slice: CudaSlice<T> = src_dev.upgrade_device_ptr(
                            src_ptr.wrapping_add(src_offset),
                            block_size_elements * dtype_size,
                        );
                        std::mem::ManuallyDrop::new(slice)
                    };

                    unsafe {
                        result::memcpy_dtoh_async(
                            dst_slice,
                            *src_slice.device_ptr(),
                            *src_dev.cu_stream(),
                        )
                        .map_err(candle_core::Error::wrap)?;
                    }
                }
                src_dev.synchronize()
            }
            //PD remote kvcache transfer
            (Device::Cuda(src_dev), Device::Cuda(dst_dev)) => {
                let Storage::Cuda(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Cuda(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };
                let remote_num_blocks = src.dim(0)?;
                let local_num_blocks = dst.dim(0)?;

                let src_ptr = *src_storage.as_cuda_slice::<T>()?.device_ptr();
                let dst_ptr = *dst_storage.as_cuda_slice::<T>()?.device_ptr();

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset: usize = src_block_number * block_size_elements;
                    assert!(
                        *src_block_number < remote_num_blocks,
                        "Invalid remote block {} / {}",
                        src_block_number,
                        remote_num_blocks
                    );
                    assert!(
                        *dst_block_number < local_num_blocks,
                        "Invalid local block {} / {}",
                        dst_block_number,
                        local_num_blocks
                    );

                    assert!(
                        src_offset + block_size_elements <= src.elem_count(),
                        "Invalid src kvcache block {} for transfer",
                        src_block_number
                    );

                    let src_offset: u64 = (src_block_number * block_size_elements * dtype_size)
                        .try_into()
                        .unwrap();
                    let src_slice: std::mem::ManuallyDrop<CudaSlice<T>> = unsafe {
                        let slice = src_dev.upgrade_device_ptr(
                            src_ptr.wrapping_add(src_offset),
                            block_size_elements * dtype_size,
                        );
                        std::mem::ManuallyDrop::new(slice)
                    };

                    let dst_offset: u64 = (dst_block_number * block_size_elements * dtype_size)
                        .try_into()
                        .unwrap();
                    let dst_slice: std::mem::ManuallyDrop<CudaSlice<T>> = unsafe {
                        let slice = dst_dev.upgrade_device_ptr(
                            dst_ptr.wrapping_add(dst_offset),
                            block_size_elements * dtype_size,
                        );
                        std::mem::ManuallyDrop::new(slice)
                    };

                    unsafe {
                        result::memcpy_dtod_async(
                            *dst_slice.device_ptr(),
                            *src_slice.device_ptr(),
                            block_size_elements * dtype_size,
                            *dst_dev.cu_stream(),
                        )
                        .map_err(candle_core::Error::wrap)?
                    }
                }
                dst_dev.synchronize()
            }
            (src, dst) => {
                candle_core::bail!("Tensors must be on either the GPU or CPU to swap, or GPU-GPU transfer, got {src:?} (src) and {dst:?} (dst).")
            }
        }
    }

    #[cfg(feature = "metal")]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        // `Copy` trait is needed for std::ptr::copy_nonoverlapping
        src: &Tensor,
        dst: &Tensor,
        block_mapping: &HashMap<usize, usize>,
    ) -> Result<()> {
        use metal::{self, MTLStorageMode};
        let block_size_elements = src.elem_count() / src.dim(0)?;
        let (src_storage, _) = src.storage_and_layout();
        let (dst_storage, _) = dst.storage_and_layout();
        let dtype_size = src.dtype().size_in_bytes();
        let block_size_bytes = block_size_elements * dtype_size;

        match (src.device(), dst.device()) {
            // Case 1: CPU -> Metal (Host to Device)
            (Device::Cpu, Device::Metal(_)) => {
                let Storage::Cpu(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Metal(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };

                let src_slice: &[T] = src_storage.as_slice()?;
                let dst_buffer = dst_storage.buffer(); // Get the underlying metal::Buffer

                // Get a CPU-writable pointer to the Metal buffer's contents.
                // This is valid for Shared and Managed storage modes.
                let dst_ptr = dst_buffer.contents() as *mut T;
                if dst_ptr.is_null() {
                    candle_core::bail!(
                        "Failed to get Metal buffer contents. Buffer might be device-private (not Shared or Managed)."
                    );
                }
                let is_managed = dst_buffer.storage_mode() == MTLStorageMode::Managed;

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset_elements = src_block_number * block_size_elements;
                    let dst_offset_elements = dst_block_number * block_size_elements;

                    // Bounds checks
                    assert!(src_offset_elements + block_size_elements <= src_slice.len());
                    assert!(
                        (dst_offset_elements * dtype_size) + block_size_bytes
                            <= dst_buffer.length() as usize
                    );

                    let src_ptr_offset = unsafe { src_slice.as_ptr().add(src_offset_elements) };
                    let dst_ptr_offset = unsafe { dst_ptr.add(dst_offset_elements) };

                    // Perform a simple CPU-side memory copy.
                    // On UMA, this directly writes to the memory the GPU will use.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_ptr_offset,
                            dst_ptr_offset,
                            block_size_elements,
                        );
                    }

                    if is_managed {
                        // If memory is Managed (not Shared), we must notify Metal of the CPU-side change.
                        dst_buffer.did_modify_range(metal::NSRange {
                            location: (dst_offset_elements * dtype_size) as u64,
                            length: block_size_bytes as u64,
                        });
                    }
                }
                // For Shared memory (default on Apple Silicon), no explicit sync is needed.
                Ok(())
            }

            // Case 2: Metal -> CPU (Device to Host)
            (Device::Metal(_), Device::Cpu) => {
                let Storage::Metal(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Cpu(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };

                let src_buffer = src_storage.buffer();
                let dst_slice: &[T] = dst_storage.as_slice()?;

                // Get a mutable pointer to the CPU slice's backing data
                let dst_ptr_mut = dst_slice.as_ptr() as *mut T;

                // Get a CPU-readable pointer to the Metal buffer's contents.
                let src_ptr = src_buffer.contents() as *const T;
                if src_ptr.is_null() {
                    candle_core::bail!(
                        "Failed to get Metal buffer contents. Buffer might be device-private."
                    );
                }

                // NOTE: If storage is Managed, a GPU-side synchronization
                // (e.g., blit_encoder.synchronize_resource) might be needed
                // before this read to ensure visibility.
                // For Shared memory, it's coherent.

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset_elements = src_block_number * block_size_elements;
                    let dst_offset_elements = dst_block_number * block_size_elements;

                    // Bounds checks
                    assert!(
                        (src_offset_elements * dtype_size) + block_size_bytes
                            <= src_buffer.length() as usize
                    );
                    assert!(dst_offset_elements + block_size_elements <= dst_slice.len());

                    let src_ptr_offset = unsafe { src_ptr.add(src_offset_elements) };
                    let dst_ptr_offset = unsafe { dst_ptr_mut.add(dst_offset_elements) };

                    // Perform a simple CPU-side memory copy.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_ptr_offset,
                            dst_ptr_offset,
                            block_size_elements,
                        );
                    }
                }
                Ok(())
            }

            // Case 3: Metal -> Metal (Device to Device)
            (Device::Metal(_), Device::Metal(dst_dev)) => {
                let Storage::Metal(src_storage) = &*src_storage else {
                    candle_core::bail!("Invalid source kvcache storage!")
                };
                let Storage::Metal(dst_storage) = &*dst_storage else {
                    candle_core::bail!("Invalid dst kvcache storage!")
                };

                let src_buffer = src_storage.buffer();
                let dst_buffer = dst_storage.buffer();

                // This is the Metal equivalent of a D2D async copy.
                // We use a Blit Command Encoder to schedule GPU-side copies.

                // Use the *destination* device's command queue.
                let command_queue = dst_dev.new_command_queue();
                let command_buffer = command_queue.new_command_buffer();
                let blit_encoder = command_buffer.new_blit_command_encoder();

                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset_bytes =
                        (src_block_number * block_size_elements * dtype_size) as u64;
                    let dst_offset_bytes =
                        (dst_block_number * block_size_elements * dtype_size) as u64;

                    // Bounds checks
                    assert!(src_offset_bytes + block_size_bytes as u64 <= src_buffer.length());
                    assert!(dst_offset_bytes + block_size_bytes as u64 <= dst_buffer.length());

                    // Schedule the GPU-side copy
                    blit_encoder.copy_from_buffer(
                        src_buffer,
                        src_offset_bytes,
                        dst_buffer,
                        dst_offset_bytes,
                        block_size_bytes as u64,
                    );
                }

                // Finish encoding and commit the commands to the GPU
                blit_encoder.end_encoding();
                command_buffer.commit();

                // The CUDA code synchronizes, so we wait for the copy to complete.
                command_buffer.wait_until_completed();

                Ok(())
            }
            (src, dst) => {
                candle_core::bail!("Tensors must be on either the Metal GPU or CPU to swap, or Metal-Metal transfer, got {src:?} (src) and {dst:?} (dst).")
            }
        }
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        _: &Tensor,
        _: &Tensor,
        _: &HashMap<usize, usize>,
    ) -> candle_core::Result<()> {
        candle_core::bail!("swap_blocks is not implemented on this platform.")
    }

    match src.dtype() {
        DType::F16 => call_fwd::<f16>(src, dst, block_mapping),
        DType::BF16 => call_fwd::<bf16>(src, dst, block_mapping),
        DType::U8 => call_fwd::<u8>(src, dst, block_mapping),
        _ => {
            candle_core::bail!("swap_blocks only accept f16/bf16/u8 kvcache dtypes!")
        }
    }
}

pub fn clear_blocks(cache: &Tensor, block_ids: &Vec<u32>) -> Result<()> {
    use candle_core::DType;
    use half::{bf16, f16};
    #[cfg(feature = "cuda")]
    fn call_fwd<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::WithDType,
    >(
        cache: &Tensor,
        block_ids: &Vec<u32>,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{
            result, CudaSlice, DevicePtr, DevicePtrMut,
        };
        let block_size_elements = cache.elem_count() / cache.dim(0)?;
        let (cache_storage, _) = cache.storage_and_layout();
        let dtype_size = cache.dtype().size_in_bytes();
        let dst_dev = cache.device().as_cuda_device()?;

        let Storage::Cuda(cache_storage) = &*cache_storage else {
            candle_core::bail!("Invalid kvcache storage!")
        };

        let num_blocks = cache.dim(0)?;

        let cache_ptr = *cache_storage.as_cuda_slice::<T>()?.device_ptr();

        for block_number in block_ids {
            let src_offset: usize = *block_number as usize * block_size_elements;
            assert!(
                *block_number < num_blocks as u32,
                "Invalid gpu block {} / {}",
                block_number,
                num_blocks
            );

            let mut src_slice: std::mem::ManuallyDrop<CudaSlice<T>> = unsafe {
                let slice = dst_dev.upgrade_device_ptr(
                    cache_ptr.wrapping_add(src_offset as u64),
                    block_size_elements * dtype_size,
                );
                std::mem::ManuallyDrop::new(slice)
            };

            unsafe {
                result::memset_d8_sync(
                    *src_slice.device_ptr_mut(),
                    0,
                    block_size_elements * dtype_size,
                )
                .map_err(candle_core::Error::wrap)?
            }
        }

        Ok(())
    }

    #[cfg(feature = "metal")]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        cache: &Tensor,
        block_ids: &Vec<u32>,
    ) -> Result<()> {
        let block_size_elements = cache.elem_count() / cache.dim(0)?;
        let (cache_storage, _) = cache.storage_and_layout();
        let dtype_size = cache.dtype().size_in_bytes();

        let Storage::Metal(cache_storage) = &*cache_storage else {
            candle_core::bail!("Invalid kvcache storage!")
        };

        let cache_buffer = cache_storage.buffer(); // Get the underlying metal::Buffer
        let num_blocks = cache.dim(0)?;

        let cache_ptr = cache_buffer.contents() as *mut T;
        if cache_ptr.is_null() {
            candle_core::bail!(
                "Failed to get Metal buffer contents. Buffer might be device-private (not Shared or Managed)."
            );
        }

        for block_number in block_ids {
            let src_offset: usize = (*block_number as usize) * block_size_elements;
            assert!(
                (*block_number as usize) < num_blocks,
                "Invalid gpu block {} / {}",
                block_number,
                num_blocks
            );

            let dst_ptr = unsafe { cache_ptr.add(src_offset) };

            // Perform a simple CPU-side memory copy.
            // On UMA, this directly writes to the memory the GPU will use.
            unsafe {
                std::ptr::write_bytes(dst_ptr, 0, block_size_elements * dtype_size);
            }
        }
        Ok(())
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    fn call_fwd<T: candle_core::WithDType + Copy>(
        _: &Tensor,
        _: &Vec<u32>,
    ) -> candle_core::Result<()> {
        candle_core::bail!("clear_blocks is not implemented on this platform.")
    }

    match cache.dtype() {
        DType::F16 => call_fwd::<f16>(cache, block_ids),
        DType::BF16 => call_fwd::<bf16>(cache, block_ids),
        DType::U8 => call_fwd::<u8>(cache, block_ids),
        _ => {
            candle_core::bail!("clear_blocks only accept f16/bf16/u8 kvcache dtypes!")
        }
    }
}
