"""Scheduled HIR: gdn_in at prompt length 512 (hand warp-specialized kernel).

Implements: Qwen3_8DeltaNetLayer.gated_delta_net: matmul(x, w_qkv|w_z|w_b|w_a, NK) (packed rows, padded to 16640)
Runtime kernel: runtime/kernels/gemm_ws.py gemm_ws_kernel(512, 16640, 5120, cfg, "store"),
  cfg = WsConfig(BN=256, stages=3, grid=132) (tag ws128x256x64_s3_p132; runtime/plan.py ws_choice(512, "gdn_in"))
Finalized TIR: tir/gemm/gdn_in_T512.py

Structure (as the kernel): persistent, 132 CTAs (one per SM); CTA p walks the tile list
u = p, p + 132, p + 264, ... < 260, M fastest (tile u = rows (u % 4) * 128, columns (u // 4) * 256),
stated as the tile loop `for u in range(cta.p, UNITS, NSM)`.
384 threads: warpgroup 0 = producer, warp 0 issues every TMA load (the A tile 128 x 64 and
the W tile 256 x 64 into 3-deep 128B-swizzled rings).
Warpgroups 1-2 = consumers: WGMMA m64n256k16 SS, W read K-major through a bitcast view,
warpgroup g owns rows [64g, 64g+64), one f32 accumulator, k16 steps ascending; then
bf16(acc) staged in shared memory and TMA-stored by a consumer thread.

Differences to the kernel:
- Synchronisation is not stated: the full/empty mbarrier ring (full count 32, empty count 8),
  the wgmma wait per k step, the named barriers of the consumer warpgroups and setmaxnreg
  40 / 232; TileFoundry derives the order from the data flow.
- Persistent loop: the kernel's ring slot sequence continues across a CTA's tiles (slot =
  (it * KT + k) % stages), the producer runs ahead into the next tile while the consumers
  run the epilogue, and a tile's TMA store is waited for only before the staging tile is
  written again. Here the ring is declared in the tile loop (slot = k % stages restarts at
  each tile); the overlap across tiles is left to lowering.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 512
N = 16640
K = 5120
BM = 128
BN = 256
BK = 64
STAGES = 3
GM = M // BM
GN = N // BN
UNITS = GM * GN
NSM = 132


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", NSM), Topology("thread", 384)))
class SCHED_GDN_IN_T512:
    @func
    def gemm(a: Tensor[(M, K), "bf16"], w: Tensor[(N, K), "bf16"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(NSM,), names=("p",)) as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=256, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                          a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                out = tf.zeros(Tensor[(M, N), "bf16"])
                for u in range(cta.p, UNITS, NSM):
                    m = (u % GM) * BM
                    n = (u // GM) * BN
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, 32, 4 @ _compute.lane4, 2), (16384, 16, 8, 1, 512, 128, 64)), "rmem"])
                    for k in tf.tile(0, K, BK):
                        with threads[0, :32] as _loader:
                            lhs_ring = tf.empty(Tensor[(BM, BK), "bf16", Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                            lhs = tf.schedule((a[m:m + BM, k], lhs_ring), op=T.copy_async_tensor())
                            rhs_ring = tf.empty(Tensor[(BN, BK), "bf16", Layout(((32, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                            rhs_0 = tf.schedule((w[n:n + BN, k], rhs_ring), op=T.copy_async_tensor())
                        with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                            rhs_t0 = tf.bitcast(rhs_0, Layout(((4, 16), (32, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))
                            acc = tf.schedule((acc, lhs, rhs_t0), op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        tile_out = tf.cast(acc, dtype="bf16")
                        staged = tf.schedule((tile_out, tf.empty(Tensor[(BM, BN), "bf16", Layout(((16, 8), (4, 64)), ((512, 64), (8192, 1))) | Swizzle(3, 4, 3), "smem"])), op=T.copy())
                    with threads[1, :32] as _storer:
                        tile = tf.schedule((staged, tf.empty(Tensor[(BM, BN), "bf16", (BM, BN)])), op=T.copy_async_tensor())
                        out = tf.insert_slice(out, tile, (m, n))
                return out
