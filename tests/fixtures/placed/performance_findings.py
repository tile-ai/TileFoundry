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
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
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
    """TF-MESH-LEVELS: the one form of this placement that is accepted today.

    A warp-specialized kernel's placement is per-CTA *and* per-lane. This states
    the CTA half; the lane half is what the finding is about, and its two refused
    forms are absent on purpose. One CTA+thread `Mesh` is refused by
    `local_type_of`, and two nested single-level meshes are refused at decoration
    time, which would make this whole file unimportable. They join it with the
    fix, because a fixture that cannot be imported states nothing.
    """

    @func
    def kernel(x: Tensor[(B, KV, H, D), "f16"]) -> Tensor[(B, KV, H, D), "f16"]:
        with Mesh(("cta",), layout=(B, H, SPLIT), names=("seq", "head", "split")) as g:
            t = tf.reshard(x, (B @ g.seq, KV @ g.split, H @ g.head, D), "rmem")
            return tf.reshard(
                tf.square(t), (B @ g.seq, KV @ g.split, H @ g.head, D), "gmem"
            )
