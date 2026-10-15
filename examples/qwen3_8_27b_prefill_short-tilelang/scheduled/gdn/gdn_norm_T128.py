"""Scheduled HIR: GDN gated RMSNorm at prompt length 128.

Implements model.py Qwen3_8DeltaNetLayer.gated_delta_net, the Qwen3_5RMSNormGated part:
  rf = f32(read); ms = mean(rf^2) over the 128 values of a head;
  normed = bf16(bf16(rf * rsqrt(ms + 1e-6)) * gamma); y = bf16(f32(normed) * silu(f32(z))).
Runtime kernel: runtime/kernels/gdn.py make_gated_norm(128, 16640) (`gdn_gated_norm`, RW = 1,
WARPS = 4); it reads the plain scan's bf16 read-out (T, 48, 128) and z = in-projection
columns [10240, 16384) through a strided view of the (T, 16640) buffer.

Decomposition (as the kernel): one CTA per 8 consecutive (t, head) rows of 128 values:
CTA b = 6t + hb (hb fastest; tilelang grid (6T,)) owns token t, heads 8hb .. 8hb + 7.
128 threads = 4 warps x 2 half-warps x 16 lanes; half-warp s of warp w serves row
2w + s; lane l owns channels 8l .. 8l + 7 (16-byte loads of o, z; its 8 gamma values in
registers). Per row: sum of rf^2 over the row (in-thread + across the 16 lanes),
* (1/128), rsqrt(ms + 1e-6), bf16(rf * r), bf16(* gamma), * zf / (exp(-zf) + 1) in f32,
landed in bf16, row store.

Differences to the kernel:
- Row sum: the kernel reproduces torch's summation order for rf.square().mean(-1)
  (chunks of 4 as ((x0^2 + x1^2) + x2^2) + x3^2, shfl_xor 8, 4, 2, 1 per chunk, then the
  lane's two chunks), with IEEE adds/multiplies (no FMA contraction), so it is bitwise
  equal to the HIR. TileFoundry has no shuffle and no way to state the summation order:
  the sum is one tf.reduce over the lane-split axis (same terms, the order is the
  lowering's; equal up to f32 rounding order).
- No FMA contraction: the kernel uses T.ieee_add / T.ieee_mul / T.ieee_div; this
  program states plain binary ops and does not state a contraction policy.
- Gate: the kernel computes n2 * silu(z) with ex2.approx / rcp.approx for the 8 values of
  a lane and one branch per thread redoes n2 * (z / (1 + expf(-z))) in IEEE f32 for every
  value whose fast result is within 16 f32 ulps of a bf16 rounding midpoint or whose
  z < -16 (checked over all finite bf16 (n2, z) pairs); its output is bitwise that of the
  exact formula stated here. The kernel's bf16 roundings are packed vector casts.
- ms = sum * 0.0078125 is the kernel's * (1/128) (a literal `(1.0 / 128.0)` inside an
  expression is refused: "dimension uses unsupported binary operation BinaryKind.DIV").
- Result: the bf16 y (T, 48, 128) itself (one tensor result, no packing).
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

T_ = 128
HV = 48
D = 128
PROJ = 16640
Z_OFF = 10240
RPC = 8
INV_D = 0.0078125
EPS = 1e-6


@module(entry="gdn_norm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", T_ * 6), Topology("thread", 128)))
class SCHED_GDN_NORM_T128:
    @func
    def gdn_norm(o: Tensor[(T_, HV, D), "bf16"], proj: Tensor[(T_, PROJ), "bf16"],
                 gamma: Tensor[(D,), "bf16"]) -> Tensor[(T_, HV, D), "bf16"]:
        with Mesh(("cta",), layout=(T_, 6), names=("t", "hb")) as cta:
            out = tf.zeros(Tensor[(T_, HV, D), "bf16"])
            h0 = cta.hb * RPC
            with Mesh(("thread",), layout=(4, 2, 16), names=("warp", "half", "ln")) as th:
                ot = tf.schedule((tf.reshape(o[cta.t:cta.t + 1, h0:h0 + RPC, :], new_shape=(RPC, D)), tf.empty(Tensor[(RPC, D), "bf16", ((4 @ th.warp, 2 @ th.half, 16 @ th.ln, 8), (0, 0, 0, 1)), "rmem"])), op=T.copy())
                zt = tf.schedule((tf.reshape(proj[cta.t:cta.t + 1, Z_OFF + h0 * D:Z_OFF + h0 * D + RPC * D], new_shape=(RPC, D)), tf.empty(Tensor[(RPC, D), "bf16", ((4 @ th.warp, 2 @ th.half, 16 @ th.ln, 8), (0, 0, 0, 1)), "rmem"])), op=T.copy())
                gm = tf.cast(tf.schedule((tf.reshape(gamma, new_shape=(1, D)), tf.empty(Tensor[(1, D), "bf16", ((1, 16 @ th.ln, 8), (0, 0, 1)), "rmem"])), op=T.copy()), dtype="f32")
                rf = tf.cast(ot, dtype="f32")
                rr = tf.rsqrt(tf.reduce(rf * rf, axes=(-1,), keepdim=True, kind="sum") * INV_D + EPS)
                nb = tf.cast(tf.cast(rf * rr, dtype="bf16"), dtype="f32")
                n2 = tf.cast(tf.cast(nb * gm, dtype="bf16"), dtype="f32")
                zf = tf.cast(zt, dtype="f32")
                y = tf.cast(n2 * (zf / (tf.exp(tf.neg(zf)) + 1.0)), dtype="bf16")
                yo = tf.schedule((y, tf.empty(Tensor[(RPC, D), "bf16", (RPC, D)])), op=T.copy())
                out = tf.insert_slice(out, tf.reshape(yo, new_shape=(1, RPC, D)), (cta.t, h0, 0))
            return out
