#include <metal_stdlib>
using namespace metal;

// Fused SiLU-and-multiply over a packed gate/up projection:
//   dst[r, j] = silu(src[r, j]) * src[r, half_dim + j]
// One pass over the packed buffer replaces narrow + silu + mul (and the
// contiguous copies the split used to make).

template <typename T, typename T4>
[[kernel]] void silu_and_mul_vec4(
    device const T *src [[buffer(0)]],
    device T *dst [[buffer(1)]],
    device const int &half_dim [[buffer(2)]],
    device const int &cols4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const int row = gid / uint(cols4);
  const int col = (gid % uint(cols4)) * 4;
  device const T4 *gate =
      (device const T4 *)(src + (long)row * 2 * half_dim + col);
  device const T4 *up =
      (device const T4 *)(src + (long)row * 2 * half_dim + half_dim + col);
  device T4 *out = (device T4 *)(dst + (long)row * half_dim + col);

  T4 g = *gate;
  T4 u = *up;
  T4 r;
  for (int i = 0; i < 4; i++) {
    float x = float(g[i]);
    r[i] = T((x / (1.0f + exp(-x))) * float(u[i]));
  }
  *out = r;
}

template <typename T>
[[kernel]] void silu_and_mul_scalar(
    device const T *src [[buffer(0)]],
    device T *dst [[buffer(1)]],
    device const int &half_dim [[buffer(2)]],
    device const int &total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= total) {
    return;
  }
  const long row = gid / uint(half_dim);
  const long col = gid % uint(half_dim);
  const long base = row * 2 * half_dim + col;
  float x = float(src[base]);
  dst[gid] = T((x / (1.0f + exp(-x))) * float(src[base + half_dim]));
}

#define instantiate_silu_and_mul(type, type4)                               \
  template [[host_name("silu_and_mul_" #type "_vec4")]] [[kernel]] void     \
  silu_and_mul_vec4<type, type4>(device const type *src [[buffer(0)]],      \
                                 device type *dst [[buffer(1)]],            \
                                 device const int &half_dim [[buffer(2)]],  \
                                 device const int &cols4 [[buffer(3)]],     \
                                 uint gid [[thread_position_in_grid]]);     \
  template [[host_name("silu_and_mul_" #type)]] [[kernel]] void             \
  silu_and_mul_scalar<type>(device const type *src [[buffer(0)]],           \
                            device type *dst [[buffer(1)]],                 \
                            device const int &half_dim [[buffer(2)]],       \
                            device const int &total [[buffer(3)]],          \
                            uint gid [[thread_position_in_grid]]);

instantiate_silu_and_mul(float, float4);
instantiate_silu_and_mul(half, half4);
#if defined(__HAVE_BFLOAT__)
instantiate_silu_and_mul(bfloat, bfloat4);
#endif
