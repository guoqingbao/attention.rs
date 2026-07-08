use attention_rs::paged_attention::reshape_and_cache;
use candle_core::{DType, Device, Tensor};

#[test]
fn test_metal_flash_reshape_and_cache_stride() {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    if !device.is_metal() {
        return;
    }

    // Let's create separate contiguous tensors first.
    let key = Tensor::zeros((61, 8, 128), DType::F16, &device).unwrap();
    let value = Tensor::zeros((61, 8, 128), DType::F16, &device).unwrap();

    let num_blocks = 96;
    let block_size = 32;
    let num_kv_heads = 8;
    let head_dim = 128;

    let x = 8;
    let key_cache = Tensor::zeros(
        (num_blocks, num_kv_heads, head_dim / x, block_size, x),
        DType::F16,
        &device,
    )
    .unwrap();
    let value_cache = Tensor::zeros(
        (num_blocks, num_kv_heads, head_dim, block_size),
        DType::F16,
        &device,
    )
    .unwrap();

    // Create an I64 slot mapping, as expected by reshape_and_cache.
    let slot_mapping = Tensor::arange(0i64, 61i64, &device).unwrap();

    let res = reshape_and_cache(
        &key,
        &value,
        &key_cache,
        &value_cache,
        None,
        None,
        &slot_mapping,
    );

    assert!(
        res.is_ok(),
        "reshape_and_cache failed on Metal: {:?}",
        res.err()
    );

    // Pull cache to CPU and verify no NaNs
    let k_cpu = key_cache
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let nan_count = k_cpu.iter().filter(|x| x.is_nan()).count();
    assert_eq!(nan_count, 0, "KV Cache has NaN values!");
}
