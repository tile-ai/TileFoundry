"""Scheduled HIR: gate_up at prompt length 256 (hand warp-specialized kernel).

Implements: Qwen3_8*Layer.swiglu: silu(matmul(x, w_gate, NK)) * matmul(x, w_up, NK)
Runtime kernel: runtime/kernels/gemm_ws.py gemm_ws_kernel(256, 17408, 5120, cfg, "swiglu2"),
  cfg = WsConfig(BN=144, stages=4, amerge=True, pk=True) (tag ws128x144x64_s4_am_pk; runtime/plan.py ws_choice(256, "gate_up"))
Finalized TIR: tir/gemm/gate_up_T256.py

Structure (as the kernel): one CTA per (128 x 144) output tile, 242 CTAs, tile order M fastest
(CTA b = rows (b % 2) * 128, columns (b // 2) * 144).
384 threads: warpgroup 0 = producer, warp 0 issues every TMA load (the A tile 128 x 64 and
the W_gate and W_up tiles 144 x 64 into 4-deep 128B-swizzled rings).
Warpgroups 1-2 = consumers: WGMMA m64n144k16 SS, W read K-major through a bitcast view,
warpgroup g owns rows [64g, 64g+64), two f32 accumulators (gate, up), k16 steps ascending; then
the SwiGLU epilogue g = bf16(acc_g), u = bf16(acc_u), bf16(silu(g)), bf16(. * u) in the
accumulator registers, the tile staged in shared memory and TMA-stored by a consumer thread.

Differences to the kernel:
- Synchronisation is not stated: the full/empty mbarrier ring (full count 32, empty count 8),
  the wgmma wait per k step, the named barriers of the consumer warpgroups and setmaxnreg
  40 / 232; TileFoundry derives the order from the data flow.
- N is 17408; the program is stated over 17424 = 121 whole tiles: weight rows >= 17408 are TMA out-of-bounds zero fill and output columns >= 17408 are clipped by the TMA store (as in the kernel), so the valid region is the authored computation.
- pk: the kernel's epilogue casts are packed bf16x2 conversions (T.copy fragment ->
  fragment) and the SwiGLU reciprocal is a non-volatile rcp.approx.ftz; same roundings. Stated as plain casts and g / (exp(-g) + 1).
- amerge: the kernel compiles with tl.enable_aggressive_shared_memory_merge, so the output
  staging tile aliases the A ring (one tile per CTA); here the staging tile is its own buffer.
- Staging tile: plain (128, 144) bf16 (the kernel swizzles it only when BN is a multiple of 64).
"""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

M = 256
N = 17424
K = 5120
BM = 128
BN = 144
BK = 64
STAGES = 4
GM = M // BM
GN = N // BN
UNITS = GM * GN


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", UNITS), Topology("thread", 384)))
class SCHED_GATE_UP_T256:
    @func
    def gemm(a: Tensor[(M, K), "bf16"], w_gate: Tensor[(N, K), "bf16"], w_up: Tensor[(N, K), "bf16"]) -> Tensor[(M, N), "bf16"]:
        with Mesh(("cta",), layout=(GN, GM), names=("bn", "bm")) as cta:
            with Mesh(("thread",), layout=(3, 128), names=("warpgroup", "participant")) as threads:
                wgmma = T.cuda.sm90.Wgmma(n=144, dtype="bf16", form=T.cuda.sm90.Form.SS,
                                          a_major=T.cuda.sm90.Major.K, b_major=T.cuda.sm90.Major.K)
                out = tf.zeros(Tensor[(M, N), "bf16"])
                m = cta.bm * BM
                n = cta.bn * BN
                with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                    acc_g = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, 18, 4 @ _compute.lane4, 2), (9216, 16, 8, 1, 512, 128, 64)), "rmem"])
                    acc_u = tf.zeros(Tensor[(BM, BN), "f32", ((2 @ _compute.group, 4 @ _compute.warp, 2, 8 @ _compute.lane8, 18, 4 @ _compute.lane4, 2), (9216, 16, 8, 1, 512, 128, 64)), "rmem"])
                for k in tf.tile(0, K, BK):
                    with threads[0, :32] as _loader:
                        lhs_ring = tf.empty(Tensor[(BM, BK), "bf16", Layout(((2, 8, 8), (4, 16)), ((4096, 512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        lhs = tf.schedule((a[m:m + BM, k], lhs_ring), op=T.copy_async_tensor())
                        gate_ring = tf.empty(Tensor[(BN, BK), "bf16", Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        rhs_0 = tf.schedule((w_gate[n:n + BN, k], gate_ring), op=T.copy_async_tensor())
                        up_ring = tf.empty(Tensor[(BN, BK), "bf16", Layout(((18, 8), (4, 16)), ((512, 64), (16, 1))) | Swizzle(3, 4, 3), "smem"], buffers=STAGES)
                        rhs_1 = tf.schedule((w_up[n:n + BN, k], up_ring), op=T.copy_async_tensor())
                    with Mesh(threads[1:3, :], layout=(2, 4, 8, 4), names=("group", "warp", "lane8", "lane4")) as _compute:
                        rhs_t0 = tf.bitcast(rhs_0, Layout(((4, 16), (18, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))
                        acc_g = tf.schedule((acc_g, lhs, rhs_t0), op=T.tiled_mma(atom=wgmma), repeat=(2, 1, 4))
                        rhs_t1 = tf.bitcast(rhs_1, Layout(((4, 16), (18, 8)), ((16, 1), (512, 64))) | Swizzle(3, 4, 3))
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
