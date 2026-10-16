"""Scheduled HIR: gate_up at prompt length 1024 (hand warp-specialized kernel).

Implements: Qwen3_8*Layer.swiglu: silu(matmul(x, w_gate, NK)) * matmul(x, w_up, NK)
Runtime kernel: runtime/kernels/gemm_ws.py gemm_ws_kernel(1024, 17408, 5120, cfg, "swiglu2"),
  cfg = WsConfig(BN=176, stages=3, grid=132, pk=True) (tag ws128x176x64_s3_p132_pk; runtime/plan.py ws_choice(1024, "gate_up"))
Finalized TIR: tir/gemm/gate_up_T1024.py

Structure (as the kernel): persistent, 132 CTAs (one per SM); CTA p walks the tile list
u = p, p + 132, p + 264, ... < 792, M fastest (tile u = rows (u % 8) * 128, columns (u // 8) * 176),
stated as the tile loop `for u in range(cta.p, UNITS, NSM)`.
384 threads: warpgroup 0 = producer, warp 0 issues every TMA load (the A tile 128 x 64 and
the W_gate and W_up tiles 176 x 64 into 3-deep 128B-swizzled rings).
Warpgroups 1-2 = consumers: WGMMA m64n176k16 SS, W read K-major through a bitcast view,
warpgroup g owns rows [64g, 64g+64), two f32 accumulators (gate, up), k16 steps ascending; then
the SwiGLU epilogue g = bf16(acc_g), u = bf16(acc_u), bf16(silu(g)), bf16(. * u) in the
accumulator registers, the tile staged in shared memory and TMA-stored by a consumer thread.

Differences to the kernel:
- Synchronisation is not stated: the full/empty mbarrier ring (full count 32, empty count 8),
  the wgmma wait per k step, the named barriers of the consumer warpgroups and setmaxnreg
  40 / 232; TileFoundry derives the order from the data flow.
- N is 17408; the program is stated over 17424 = 99 whole tiles: weight rows >= 17408 are TMA out-of-bounds zero fill and output columns >= 17408 are clipped by the TMA store (as in the kernel), so the valid region is the authored computation.
- Persistent loop: the kernel's ring slot sequence continues across a CTA's tiles (slot =
  (it * KT + k) % stages), the producer runs ahead into the next tile while the consumers
  run the epilogue, and a tile's TMA store is waited for only before the staging tile is
  written again. Here the ring is declared in the tile loop (slot = k % stages restarts at
  each tile); the overlap across tiles is left to lowering.
- pk: the kernel's epilogue casts are packed bf16x2 conversions (T.copy fragment ->
  fragment) and the SwiGLU reciprocal is a non-volatile rcp.approx.ftz; same roundings. Stated as plain casts and g / (exp(-g) + 1).
- Staging tile: plain (128, 176) bf16 (the kernel swizzles it only when BN is a multiple of 64).
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 1024
N = 17424
K = 5120
BM = 128
BN = 176
BK = 64
STAGES = 3
GM = M // BM
GN = N // BN
UNITS = GM * GN
NSM = 132


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", NSM), Topology("thread", 384)))
class SCHED_GATE_UP_T1024:
    @func
    def gemm(a: Tensor[(M, K), "bf16"], w_gate: Tensor[(N, K), "bf16"], w_up: Tensor[(N, K), "bf16"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(NSM,), names=("p",)) as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=176, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                          a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                out = tf.zeros(Tensor[(M, N), "bf16"])
                for u in range(cta.p, UNITS, NSM):
                    m = (u % GM) * BM
                    n = (u // GM) * BN
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        acc_g = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, 22, 4 @ _compute.lane4, 2), (11264, 16, 8, 1, 512, 128, 64)), "rmem"])
                        acc_u = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, 22, 4 @ _compute.lane4, 2), (11264, 16, 8, 1, 512, 128, 64)), "rmem"])
                    for k in tf.tile(0, K, BK):
                        with threads[0, :32] as _loader:
                            lhs_ring = tf.empty(Tensor[(BM, BK), "bf16", Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                            lhs = tf.schedule((a[m:m + BM, k], lhs_ring), op=T.copy_async_tensor())
                            gate_ring = tf.empty(Tensor[(BN, BK), "bf16", Layout(((22, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                            rhs_0 = tf.schedule((w_gate[n:n + BN, k], gate_ring), op=T.copy_async_tensor())
                            up_ring = tf.empty(Tensor[(BN, BK), "bf16", Layout(((22, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                            rhs_1 = tf.schedule((w_up[n:n + BN, k], up_ring), op=T.copy_async_tensor())
                        with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                            rhs_t0 = tf.bitcast(rhs_0, Layout(((4, 16), (22, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))
                            acc_g = tf.schedule((acc_g, lhs, rhs_t0), op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                            rhs_t1 = tf.bitcast(rhs_1, Layout(((4, 16), (22, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))
                            acc_u = tf.schedule((acc_u, lhs, rhs_t1), op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        g = tf.cast(tf.cast(acc_g, dtype="bf16"), dtype="f32")
                        u = tf.cast(tf.cast(acc_u, dtype="bf16"), dtype="f32")
                        act = tf.cast(g / (tf.exp(tf.neg(g)) + 1.0), dtype="bf16")
                        tile_out = tf.cast(tf.cast(act, dtype="f32") * u, dtype="bf16")
                        staged = tf.schedule((tile_out, tf.empty(Tensor[(BM, BN), "bf16", (BM, BN), "smem"])), op=T.copy())
                    with threads[1, :32] as _storer:
                        tile = tf.schedule((staged, tf.empty(Tensor[(BM, BN), "bf16", (BM, BN)])), op=T.copy_async_tensor())
                        out = tf.insert_slice(out, tile, (m, n))
                return out
