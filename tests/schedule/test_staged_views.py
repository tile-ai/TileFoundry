"""Read one staged key tile as K^T for Q @ K^T and as V for P @ V, staging it once.

The scores take the tile through a transposed view, the weighted sum through a
regrouping of the same addresses. Neither allocates or copies. K_SMEM is the key tile
as it lies (keys as rows, dims contiguous); KT_ISSUES and V_ISSUES are the same
addresses grouped as Q @ K^T and P @ V read them, four k16 issues each.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, ReduceKind, T, Tensor, Topology, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- authored tile loops
from tilefoundry.inspection.python_printer import as_script
from tilefoundry.ir.types import ComposedLayout, Layout, ShardLayout, Split, Swizzle
from tilefoundry.ir.types import Mesh as ThreadMesh
from tilefoundry.schedule import finalize
from tilefoundry.schedule.candidates import candidates, render
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.verify import verify_prim_function

H, KEYS, DQ = 64, 64, 64
SW128 = Swizzle(3, 4, 3)
_COMPUTE = ThreadMesh(
    (Topology("thread", 128),),
    ComposedLayout(None, 0, Layout((4, 8, 4), (32, 4, 1))),
    ("warp", "lane8", "lane4"),
)
_HELD = (Split(2), Split(0), Split(4))
Q_SMEM = ComposedLayout(SW128, 0, Layout(((8, 8), (4, 16)), ((512, 64), (16, 1))))
K_SMEM = ComposedLayout(SW128, 0, Layout(((8, 8), 64), ((512, 64), 1)))
KT_ISSUES = ComposedLayout(SW128, 0, Layout(((4, 16), (8, 8)), ((16, 1), (512, 64))))
V_ISSUES = ComposedLayout(SW128, 0, Layout(((4, 2, 8), (1, 64)), ((1024, 512, 64), (1024, 1))))
ACC = ShardLayout(Layout((8, 2, 4, 2, 4, 8), (1, 8, 16, 64, 128, 512)), _HELD, _COMPUTE)
P_RS = ShardLayout(
    Layout((4, 8, 2, 4, 2, 4, 2), (1024, 1, 8, 16, 64, 128, 512)),
    (Split(3), Split(1), Split(5)),
    _COMPUTE,
)


@module(
    entry="step",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 128)),
)
class KeyTileViews:
    @func
    def step(
        q: Tensor[(H, DQ), "bf16"], keys: Tensor[(KEYS, DQ), "bf16"]
    ) -> Tensor[(H, DQ), "bf16", "umat"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(128,), names=("lane",)) as _threads:
                scores_atom = T.cuda.sm90.Wgmma(
                    n=KEYS,
                    form=T.cuda.sm90.Form.SS,
                    a_major=T.cuda.sm90.Major.K,
                    b_major=T.cuda.sm90.Major.K,
                )
                values_atom = T.cuda.sm90.Wgmma(n=DQ, form=T.cuda.sm90.Form.RS)
                qs = tf.schedule((q[:, 0:DQ],), op=T.copy_async(smem_layout=Q_SMEM))
                ks = tf.schedule((keys[:, 0:DQ],), op=T.copy_async(smem_layout=K_SMEM))
                acc = tf.zeros(Tensor[(H, KEYS), "f32", ACC, "rmem"])
                kt = tf.reshard(tf.transpose(ks, (1, 0), view=True), KT_ISSUES, "smem")
                acc = tf.schedule(
                    (acc, qs, kt), op=T.tiled_mma(atom=scores_atom), repeat=(1, 1, 4)
                )
                peak = tf.reduce(acc, axes=(1,), keepdim=True, kind=ReduceKind.MAX)
                weights = tf.exp2(acc - peak)
                p = tf.schedule((tf.cast(weights, dtype="bf16"),), op=T.copy(rmem_layout=P_RS))
                out = tf.zeros(Tensor[(H, DQ), "f32", ACC, "rmem"])
                vs = tf.reshard(ks, V_ISSUES, "smem")
                out = tf.schedule(
                    (out, p, vs), op=T.tiled_mma(atom=values_atom), repeat=(1, 1, 4)
                )
                return tf.cast(out, dtype="bf16")


def test_both_products_read_the_one_staged_key_tile() -> None:
    """Q and the key tile are each staged once; K^T and V are windows over the tile."""
    function = finalize(KeyTileViews)
    verify_prim_function(function)
    text = as_script(function)

    assert text.count("T.copy_async(") == 2
    assert text.count("T.ptr_of(ks[0:0 + 64, 0:0 + 64])") == 2


def test_the_views_are_not_points_an_instruction_must_fill() -> None:
    """A view moves nothing, so no copy is offered for it."""
    text = render(candidates(KeyTileViews, KeyTileViews.entry_function()))

    assert "tf.reshard" not in text
    assert "tf.transpose" not in text
