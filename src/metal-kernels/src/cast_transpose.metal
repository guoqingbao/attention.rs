#include <metal_stdlib>
using namespace metal;

// Fused cast + transpose between the projection layout (seq, heads, dim) and
// the attention-kernel layout (heads, seq, dim). Both sides are contiguous
// along dim, so each thread moves one aligned 4-element vector on each side.

template <typename T, typename U, typename T4, typename U4, bool SHD_TO_HSD>
[[kernel]] void cast_transpose(device const T *src [[buffer(0)]],
                               device U *dst [[buffer(1)]],
                               device const int &S [[buffer(2)]],
                               device const int &H [[buffer(3)]],
                               device const int &D [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
  const int d4 = D / 4;
  const int d = int(gid % uint(d4)) * 4;
  const int rest = int(gid) / d4;
  const int s = rest % S;
  const int h = rest / S;

  long src_off;
  long dst_off;
  if (SHD_TO_HSD) {
    src_off = ((long)s * H + h) * D + d;
    dst_off = ((long)h * S + s) * D + d;
  } else {
    src_off = ((long)h * S + s) * D + d;
    dst_off = ((long)s * H + h) * D + d;
  }

  T4 v = *(device const T4 *)(src + src_off);
  U4 r;
  for (int i = 0; i < 4; i++) {
    r[i] = U(v[i]);
  }
  *(device U4 *)(dst + dst_off) = r;
}

#define instantiate_cast_transpose_shd_hsd(t, t4, u, u4)                    \
  template [[host_name("cast_transpose_" #t "_to_" #u "_shd_hsd")]]          \
  [[kernel]] void cast_transpose<t, u, t4, u4, true>(                        \
      device const t *src [[buffer(0)]], device u *dst [[buffer(1)]],        \
      device const int &S [[buffer(2)]], device const int &H [[buffer(3)]],  \
      device const int &D [[buffer(4)]], uint gid [[thread_position_in_grid]]);

#define instantiate_cast_transpose_hsd_shd(t, t4, u, u4)                    \
  template [[host_name("cast_transpose_" #t "_to_" #u "_hsd_shd")]]          \
  [[kernel]] void cast_transpose<t, u, t4, u4, false>(                       \
      device const t *src [[buffer(0)]], device u *dst [[buffer(1)]],        \
      device const int &S [[buffer(2)]], device const int &H [[buffer(3)]],  \
      device const int &D [[buffer(4)]], uint gid [[thread_position_in_grid]]);

instantiate_cast_transpose_shd_hsd(float, float4, half, half4)
instantiate_cast_transpose_shd_hsd(half, half4, half, half4)
instantiate_cast_transpose_hsd_shd(half, half4, float, float4)
instantiate_cast_transpose_hsd_shd(half, half4, half, half4)
#if defined(__HAVE_BFLOAT__)
instantiate_cast_transpose_shd_hsd(bfloat, bfloat4, half, half4)
instantiate_cast_transpose_hsd_shd(half, half4, bfloat, bfloat4)
#endif
