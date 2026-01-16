#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::sys;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::CudaDevice;

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
    compute_capability(dev).map(|(major, minor)| major * 10 + minor)
}
