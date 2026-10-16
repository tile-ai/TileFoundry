"""Scheduled HIR: o_proj at prompt length 512 (hand warp-specialized kernel).

Implements: Qwen3_8*Layer.layer: hidden + matmul(gated, w_out|w_o, NK)
Runtime kernel: runtime/kernels/gemm_ws.py gemm_ws_kernel(512, 5120, 6144, cfg, "residual"),
  cfg = WsConfig(BN=160, stages=4, pk=True, io=True) (tag ws128x160x64_s4_pk_io; runtime/plan.py ws_choice(512, "o_proj"))
Finalized TIR: tir/gemm/o_proj_T512.py

Structure (as the kernel): one CTA per (128 x 160) output tile, 128 CTAs, tile order M fastest
(CTA b = rows (b % 4) * 128, columns (b // 4) * 160).
384 threads: warpgroup 0 = producer, warp 0 issues every TMA load (the A tile 128 x 64 and
the W tile 160 x 64 into 4-deep 128B-swizzled rings);
warp 1 (IO warp) TMA-loads the residual tile R and issues the output TMA store.
Warpgroups 1-2 = consumers: WGMMA m64n160k16 SS, W read K-major through a bitcast view,
warpgroup g owns rows [64g, 64g+64), one f32 accumulator, k16 steps ascending; then
the residual epilogue bf16(R + bf16(acc)) with R read from the IO warp's shared tile, the tile
staged in shared memory and TMA-stored by the IO warp.

Differences to the kernel:
- Synchronisation is not stated: the full/empty mbarrier ring (full count 32, empty count 8),
  the wgmma wait per k step, the named barriers of the consumer warpgroups and setmaxnreg
  40 / 232; TileFoundry derives the order from the data flow.
- pk: the kernel's epilogue casts are packed bf16x2 conversions (T.copy fragment ->
  fragment); same roundings. Stated as plain casts.
- io: the kernel's IO warp TMA-loads R into the output staging tile itself, the consumers
  read R from it and overwrite it with the result, and the IO warp stores that tile; here R
  lands in its own shared tile and the result in a second one (both TMA ops on warp 1).
- Staging tile: plain (128, 160) bf16 (the kernel swizzles it only when BN is a multiple of 64).
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 512
N = 5120
K = 6144
BM = 128
BN = 160
BK = 64
STAGES = 4
GM = M // BM
GN = N // BN
UNITS = GM * GN


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", UNITS), Topology("thread", 384)))
class SCHED_O_PROJ_T512:
    @func
    def gemm(a: Tensor[(M, K), "bf16"], w: Tensor[(N, K), "bf16"], residual: Tensor[(M, N), "bf16"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(GN, GM), names=("bn", "bm")) as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=160, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                          a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                out = tf.zeros(Tensor[(M, N), "bf16"])
                m = cta.bm * BM
                n = cta.bn * BN
                with threads[0, 32:64] as _io:
                    r_s = tf.schedule((residual[m:m + BM, n:n + BN], tf.empty(Tensor[(BM, BN), "bf16", (BM, BN), "smem"])), op=T.copy_async_tensor())
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    acc = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, 20, 4 @ _compute.lane4, 2), (10240, 16, 8, 1, 512, 128, 64)), "rmem"])
                for k in tf.tile(0, K, BK):
                    with threads[0, :32] as _loader:
                        lhs_ring = tf.empty(Tensor[(BM, BK), "bf16", Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        lhs = tf.schedule((a[m:m + BM, k], lhs_ring), op=T.copy_async_tensor())
                        rhs_ring = tf.empty(Tensor[(BN, BK), "bf16", Layout(((20, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        rhs_0 = tf.schedule((w[n:n + BN, k], rhs_ring), op=T.copy_async_tensor())
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        rhs_t0 = tf.bitcast(rhs_0, Layout(((4, 16), (20, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))
                        acc = tf.schedule((acc, lhs, rhs_t0), op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    mixed = tf.cast(acc, dtype="bf16")
                    r_tile = tf.schedule((r_s, tf.empty(Tensor[(BM, BN), "bf16", ((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, 20, 4 @ _compute.lane4, 2), (10240, 16, 8, 1, 512, 128, 64)), "rmem"])), op=T.copy())
                    tile_out = tf.cast(tf.cast(r_tile, dtype="f32") + tf.cast(mixed, dtype="f32"), dtype="bf16")
                    staged = tf.schedule((tile_out, tf.empty(Tensor[(BM, BN), "bf16", (BM, BN), "smem"])), op=T.copy())
                with threads[0, 32:64] as _io:
                    tile = tf.schedule((staged, tf.empty(Tensor[(BM, BN), "bf16", (BM, BN)])), op=T.copy_async_tensor())
                    out = tf.insert_slice(out, tile, (m, n))
                return out
