"""The programs a round reported `performance` could not answer, and the controls.

From `tileops-loop-state-v2/round-2-mha-decode-paged`: `findings.json` is sha256
41bdc13b13927159c9a83a4d82205f9415aeb75b9d835f29a2fb5490bcdb4c79 and its three
reproducers are sha256 76dd3567ce8f429ea95a8be115bdb8f03026e511326c1915895f245ea7db3d34,
0cfa30ae409d1036d77c1959c0b16e17efc062787464e1d601fba925bec2d58d and
218cd703420a51817e24bc3681a90bba2e04ba7a3337fbf05bdf0d8a4b205bcb. A shape means
something only beside the control it differs from, so `GmemSquare` is the control
for two of these and each pair differs in exactly one respect.
"""

from tilefoundry import func, module
from tilefoundry.dsl import DimVar, Mesh, Tensor, Topology, tf
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")

CTAS = 132
N = CTAS * 128

B, H, KV, D = 1, 16, 512, 128
SPLIT, WARPS, LANES = 8, 4, 32


@module(entry="kernel", target=_H200, topologies=(Topology("cta", CTAS),))
class GmemSquare:
    """The control both storage and comparison findings are stated against.

    One row per CTA of an H200's 132. The round shipped its two findings in two
    files and each carried its own copy of this, and one program is not made into
    two by being written down twice.
    """

    @func
    def kernel(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as m:
            placed = tf.reshard(x, (N @ m.block,), "gmem")
            return tf.reshard(tf.square(placed), (N @ m.block,), "gmem")


@module(entry="kernel", target=_H200, topologies=(Topology("cta", CTAS),))
class LocalTier:
    """TF-LOCAL-STORAGE-UNMODELLED: the same program with one rmem intermediate.

    The target states `memory.register.bandwidth` as unavailable, because NVIDIA
    publishes no static figure. Stating the movement the provenance gate asks an
    authored program to state must not therefore cost the program its answer:
    the bytes are recorded, and what nobody published a rate for is left untimed.
    """

    @func
    def kernel(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as m:
            local = tf.reshard(x, (N @ m.block,), "rmem")
            return tf.reshard(tf.square(local), (N @ m.block,), "gmem")


@module(entry="kernel", target=_H200, topologies=(Topology("cta", CTAS),))
class Compare:
    """TF-TIMELINE-BOOL: the same program with one comparison feeding a select.

    Every masked attention has this shape -- a variable-length cache masked by
    comparing a position against a length -- and a comparison is predicate work
    at a rate the target states, not floating-point work at a `bool` rate nobody
    publishes.
    """

    @func
    def kernel(
        x: Tensor[(N,), "f32"], limit: Tensor[(N,), "f32"]
    ) -> Tensor[(N,), "f32"]:
        with Mesh(("cta",), layout=(CTAS,), names=("block",)) as m:
            placed = tf.reshard(x, (N @ m.block,), "gmem")
            bound = tf.reshard(limit, (N @ m.block,), "gmem")
            kept = tf.where(
                tf.cmp_lt(placed, bound), placed, tf.full_like(placed, value=0.0)
            )
            return tf.reshard(tf.square(kept), (N @ m.block,), "gmem")


@module(entry="kernel", target=_H200, topologies=(Topology("cta", B * H * SPLIT),))
class Levels:
    """TF-MESH-LEVELS: the CTA half of the placement, on its own.

    A warp-specialized kernel is placed per CTA *and* per lane. This says only
    the first half, and is the control the two forms that say both are read
    against: all three place the same CTAs and predict the same time, because
    the lane structure is a fact about what happens inside one of them.
    """

    @func
    def kernel(x: Tensor[(B, KV, H, D), "f16"]) -> Tensor[(B, KV, H, D), "f16"]:
        with Mesh(("cta",), layout=(B, H, SPLIT), names=("seq", "head", "split")) as g:
            t = tf.reshard(x, (B @ g.seq, KV @ g.split, H @ g.head, D), "rmem")
            return tf.reshard(
                tf.square(t), (B @ g.seq, KV @ g.split, H @ g.head, D), "gmem"
            )


@module(entry="kernel", target=_H200,
        topologies=(Topology("cta", B * H * SPLIT), Topology("thread", WARPS * LANES)))
class LevelsOnOneMesh:
    """TF-MESH-LEVELS: both halves, on one Mesh naming both levels.

    The axes are handed to the levels left to right and the boundary is where
    their extents multiply to the CTA count, so `(seq, head, split)` are the
    CTAs and `(warp, lane)` are the threads inside one. This is the form the
    round's own scaffold template used.
    """

    @func
    def kernel(x: Tensor[(B, KV, H, D), "f16"]) -> Tensor[(B, KV, H, D), "f16"]:
        with Mesh(
            ("cta", "thread"),
            layout=(B, H, SPLIT, WARPS, LANES),
            names=("seq", "head", "split", "warp", "lane"),
        ) as m:
            t = tf.reshard(
                x, (B @ m.seq, KV @ (m.split, m.warp), H @ m.head, D @ m.lane), "rmem"
            )
            return tf.reshard(
                tf.square(t),
                (B @ m.seq, KV @ (m.split, m.warp), H @ m.head, D @ m.lane),
                "gmem",
            )


@module(entry="kernel", target=_H200,
        topologies=(Topology("cta", B * H * SPLIT), Topology("thread", WARPS * LANES)))
class LevelsNested:
    """TF-MESH-LEVELS: both halves, as two single-level Meshes nested in one another.

    Which level distributes which is not something a layout can be read for, so
    it is taken from the scopes the layout is written inside. The lane mesh is
    entered within the CTA mesh, and a layout naming axes of both says the CTA
    owns a tile and the lane owns part of that tile -- the same placement the
    one-Mesh form states, said the other way round.
    """

    @func
    def kernel(x: Tensor[(B, KV, H, D), "f16"]) -> Tensor[(B, KV, H, D), "f16"]:
        with Mesh(("cta",), layout=(B, H, SPLIT), names=("seq", "head", "split")) as g:
            with Mesh(("thread",), layout=(WARPS, LANES), names=("warp", "lane")) as m:
                t = tf.reshard(
                    x,
                    (B @ g.seq, KV @ (g.split, m.warp), H @ g.head, D @ m.lane),
                    "rmem",
                )
                return tf.reshard(
                    tf.square(t),
                    (B @ g.seq, KV @ (g.split, m.warp), H @ g.head, D @ m.lane),
                    "gmem",
                )


_CROSS_SCOPE_GRID = 128
_CROSS_SCOPE_HEAD = 128
_CROSS_SCOPE_HEADS = 16
_CROSS_SCOPE_TILE = 256
_CROSS_SCOPE_STAGES = 2
_CROSS_SCOPE_SEQ = DimVar("seq_len", 0, 8193)


@module(
    entry="cross_scope",
    target=_H200,
    topologies=(Topology("cta", _CROSS_SCOPE_GRID),),
)
class _CrossScopePerformance:
    """A cross-scope consumer and the control that keeps both calls nested."""

    @func
    def stage(
        x: Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"],
        out: Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"],
    ) -> Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"]:
        with Mesh(
            ("cta",), layout=(_CROSS_SCOPE_HEADS, 8), names=("strip", "tile")
        ) as mesh:
            acc = out
            for position in tile(_CROSS_SCOPE_SEQ, _CROSS_SCOPE_TILE):
                base = position + 0
                placed = tf.reshard(
                    x[:, base : base + _CROSS_SCOPE_TILE, :],
                    (
                        _CROSS_SCOPE_HEADS @ mesh.strip,
                        _CROSS_SCOPE_TILE @ mesh.tile,
                        _CROSS_SCOPE_HEAD,
                    ),
                    "smem",
                )
                acc = tf.insert_slice(acc, placed + placed, (0, base, 0))
            return acc

    @func
    def cross_scope(
        x: Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"],
        inner: Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"],
        outer: Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"],
    ) -> Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"]:
        with Mesh(("cta",), layout=(_CROSS_SCOPE_GRID,), names=("unit",)) as _mesh:
            result = x
            for _batch in range(1):
                for _stage in range(_CROSS_SCOPE_STAGES):
                    result = stage(result, inner)
                result = stage(result, outer)
            return result

    @func
    def nested_control(
        x: Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"],
        inner: Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"],
        outer: Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"],
    ) -> Tensor[(_CROSS_SCOPE_HEADS, _CROSS_SCOPE_SEQ, _CROSS_SCOPE_HEAD), "bf16"]:
        with Mesh(("cta",), layout=(_CROSS_SCOPE_GRID,), names=("unit",)) as _mesh:
            result = x
            for _batch in range(1):
                for _stage in range(_CROSS_SCOPE_STAGES):
                    result = stage(result, inner)
                    result = stage(result, outer)
            return result
