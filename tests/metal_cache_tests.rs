use candle_core::{DType, Device, Tensor};
use attention_rs::paged_attention::reshape_and_cache;

#[test]
fn test_metal_flash_reshape_and_cache_stride() {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    if !device.is_metal() {
        return;
    }

    // [num_tokens, num_kv_heads, head_dim] -> [61, 8, 128]
    // Let's create a non-contiguous key and value tensor.
    // We create a larger tensor and slice it.
    let qkv = Tensor::zeros((61, 24, 128), DType::F16, &device).unwrap();
    // Slice to extract key: [61, 8, 128]
    let key = qkv.narrow(1, 8, 8).unwrap();
    let value = qkv.narrow(1, 16, 8).unwrap();

    let num_blocks = 96;
    let block_size = 32;
    let num_kv_heads = 8;
    let head_dim = 128;

    let key_cache = Tensor::zeros((num_blocks, block_size, num_kv_heads, head_dim), DType::F16, &device).unwrap();
    let value_cache = Tensor::zeros((num_blocks, block_size, num_kv_heads, head_dim), DType::F16, &device).unwrap();

    // Create a U32 slot mapping, which should be promoted to I64 correctly.
    let slot_mapping = Tensor::arange(0u32, 61u32, &device).unwrap();

    // Run reshape_and_cache
    let res = reshape_and_cache(
        &key,
        &value,
        &key_cache,
        &value_cache,
        None,
        None,
        &slot_mapping,
    );

    assert!(res.is_ok(), "reshape_and_cache failed on Metal: {:?}", res.err());

    // Pull cache to CPU and verify no NaNs
    let k_cpu = key_cache.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let nan_count = k_cpu.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, 0, "KV Cache has NaN values!");
}
