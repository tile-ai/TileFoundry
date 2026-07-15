"""Viewer builder shard-layout rendering.

A sharded type must render its shard sugar in the compact graph row, the
canonical detail-panel type, and a ``Reshard`` layout attr — all through the
shared shard-sugar core, never the verbose ``ShardLayout(...)`` repr (which the
pre-fix renderer emitted for graph/detail/attr).
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- binds bare op names (reshard, ...)
from tilefoundry.inspection.viewer.builder import (
    ViewerBuilder,
    _pretty_attr_value,
    type_to_canonical_pretty,
    type_to_compact_pretty,
)
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import (
    B,
    Layout,
    Mesh,
    P,
    S,
    ShardLayout,
    Topology,
)

# A 3-axis mesh; ``l`` splits a layout dim and ``t`` carries a Partial value state.
_TRD = Mesh(
    Topology("thread", 4 * 2 * 16), Layout((4, 2, 16), (32, 16, 1)), names=("l", "g", "t")
)


@func
def _sharded_io(
    a: Tensor[(4, 64), "f32", ((4 @ _TRD.l, 64), {_TRD.t @ P("sum")}), "smem"],
) -> Tensor[(4, 64), "f32", ((4 @ _TRD.l, 64), {_TRD.t @ P("sum")}), "smem"]:
    return a


@func
def _reshard_demo(
    a: Tensor[(4, 64), "f32"],
) -> Tensor[(4, 64), "f32"]:
    return reshard(
        a,
        layout=ShardLayout(
            layout=Layout((4, 64), (64, 1)), attrs=(S(0), B(), B()), mesh=_TRD
        ),
    )


def _mesh_name(vb: ViewerBuilder, mesh) -> str:
    return vb.mesh_name_map[id(mesh)]


def test_compact_graph_row_renders_shard_sugar() -> None:
    """The function node's compact return-type row inlines the Split on its
    tensor axis and carries the Partial value-state suffix — not a plain
    ``f32[4, 64]`` that drops the sharding."""
    fn = _sharded_io
    vb = ViewerBuilder(fn)
    src = vb.build().source
    name = _mesh_name(vb, fn.return_type.layout.mesh)
    # Inline Split on the tensor axis and the Partial suffix (``"`` is HTML
    # escaped to ``&quot;`` inside the DOT label).
    assert f"{name}.l" in src
    assert f"{name}.t @ P(&quot;sum&quot;)" in src
    # The graph must not fall back to the verbose constructor for a named mesh.
    assert "ShardLayout(" not in src


def test_detail_panel_type_is_canonical_shard_sugar() -> None:
    """The detail panel renders the canonical 4-slot sugar for a sharded type,
    not the verbose ``ShardLayout(...)`` form."""
    fn = _sharded_io
    name = _mesh_name(ViewerBuilder(fn), fn.return_type.layout.mesh)
    canonical = type_to_canonical_pretty(
        fn.return_type, {id(fn.return_type.layout.mesh): name}
    )
    assert canonical == (
        f'Tensor[(4, 64), "f32", ((4 @ {name}.l, 64), {{{name}.t @ P("sum")}}), "smem"]'
    )
    assert "ShardLayout(" not in canonical


def test_reshard_attr_renders_in_sugar() -> None:
    """A ``Reshard`` layout attr renders through the sugar core (an inline-split
    layout tuple), never the verbose ``ShardLayout(...)`` repr."""
    fn = _reshard_demo
    vb = ViewerBuilder(fn)
    src = vb.build().source
    name = _mesh_name(vb, fn.body.target.layout.mesh)
    assert f"layout: (4 @ {name}.l, 64)" in src
    assert "ShardLayout(" not in src


def test_compact_and_canonical_partial_from_one_core() -> None:
    """Compact and canonical render the same shard layout (Split + Partial +
    default Broadcast) consistently from the shared core."""
    mesh = Mesh(
        Topology("thread", 4 * 2 * 16), Layout((4, 2, 16), (32, 16, 1)), names=("l", "g", "t")
    )
    ty = TensorType(
        shape=(4, 64),
        dtype=DType.f32,
        storage=StorageKind.SMEM,
        layout=ShardLayout(
            layout=Layout((4, 64), (64, 1)),
            attrs=(S(0), B(), P("sum")),
            mesh=mesh,
        ),
    )
    mm = {id(mesh): "m"}
    assert type_to_compact_pretty(ty, mm) == 'f32[4 @ m.l, 64] {m.t @ P("sum")} @smem'
    assert type_to_canonical_pretty(ty, mm) == (
        'Tensor[(4, 64), "f32", ((4 @ m.l, 64), {m.t @ P("sum")}), "smem"]'
    )


def test_unnamed_mesh_attr_falls_back_to_verbose() -> None:
    """A ``ShardLayout`` whose mesh has no named axes cannot use sugar, so the
    attr renderer falls back to the verbose ``ShardLayout(...)`` form."""
    mesh = Mesh(Topology("thread", 4), Layout((4,), (1,)))  # no names=
    sl = ShardLayout(layout=Layout((4, 64), (64, 1)), attrs=(S(0),), mesh=mesh)
    out = _pretty_attr_value(sl, full=True, mesh_name_map={id(mesh): "m"})
    assert out.startswith("ShardLayout(")
