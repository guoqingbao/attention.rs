#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::sys;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::CudaDevice;

#[cfg(feature = "cuda")]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

#[cfg(feature = "cuda")]
static SM_CACHE: OnceLock<Mutex<HashMap<usize, Option<i32>>>> = OnceLock::new();

#[cfg(feature = "cuda")]
pub fn compute_capability(dev: &CudaDevice) -> Option<(i32, i32)> {
    let major = dev
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .ok()?;
    let minor = dev
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .ok()?;
    Some((major, minor))
}

#[cfg(feature = "cuda")]
pub fn sm_version(dev: &CudaDevice) -> Option<i32> {
    // Key by the address of this device instance
    let key = (dev as *const CudaDevice) as usize;

    let cache = SM_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Fast path: return cached value if present
    if let Some(v) = cache.lock().unwrap().get(&key).copied() {
        return v;
    }

    // Compute outside the lock to keep the critical section small
    let computed = compute_capability(dev).map(|(major, minor)| major * 10 + minor);

    // Store (a second thread might store first; that is fine)
    cache.lock().unwrap().insert(key, computed);

    computed
}
