"""``Tensor[...]`` layout sugar on ``@prim_func`` params (parser.md §1.4/§1.5).

``_build_params`` (parser/base.py) is now the single HIR+TIR parameter walk,
routing both through the same ``_resolve_tensor_type`` — this locks that a
layout-sugar annotation resolves on a device ``@prim_func`` parameter exactly
as it already does on an ``@func`` parameter (see
``tests/parser/hir/test_parse_shard_sugar.py::test_int_at_single_axis_mesh_canonicalises``,
the HIR twin of this scenario).
"""
from __future__ import annotations

from tilefoundry import prim_func
from tilefoundry.dsl import Tensor
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.ir.types.shard.shard_layout import ShardLayout, Split
from tilefoundry.ir.types.storage import StorageKind

_M_CTA = Mesh(Topology("cta", 128), Layout((128,), (1,)), names=("cta",))


def test_prim_func_param_layout_sugar_canonicalises() -> None:
    """On a single-axis mesh, ``8192 @ cta`` (extent 128) canonicalises into
    ``(128, 64)`` with the mesh axis bound as a Split on the new layout axis
    — the same result the HIR parser produces for an identical annotation."""

    @prim_func(target="cuda")
    def dev(a: Tensor[(1, 8192), "f32", (1, 8192 @ _M_CTA), "smem"]):  # noqa: F821
        return

    assert dev.params[0].type == TensorType(
        shape=(1, 8192),
        dtype=DType.f32,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((1, 128, 64), (8192, 64, 1)),
            attrs=(Split(1),),
            mesh=_M_CTA,
        ),
    )
