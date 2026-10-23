"""TileFoundry's register-fragment invariants, independent of the golden layouts."""

from __future__ import annotations

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.pattern.utils import fixed_pattern_value, selected_pattern
from tilefoundry.ir.types import DType, Layout, ShardLayout, Split, TensorType
from tilefoundry.ir.types.int_tuple import flatten
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.typeinfer import inference_type


def _check_split_extents_match_mesh(sl: ShardLayout) -> None:
    mesh_shape = flatten(sl.mesh.layout.shape)
    assert len(sl.attrs) == len(mesh_shape), (
        f"attrs len {len(sl.attrs)} != mesh rank {len(mesh_shape)}"
    )
    for mesh_i, attr in enumerate(sl.attrs):
        assert isinstance(attr, Split), f"mma fragment shard attrs must be Split, got {attr}"
        ax = attr.axis
        assert sl.layout.shape[ax] == mesh_shape[mesh_i], (
            f"split tensor axis {ax} extent {sl.layout.shape[ax]} != "
            f"mesh axis {mesh_i} extent {mesh_shape[mesh_i]}"
        )


def _assert_reshard_typeinfer_ok(
    src_shape: tuple[int, ...], src_dtype_name: str, dst_layout: ShardLayout
) -> None:
    """Assert reshard typeinfer ok.

    Run the registered Reshard typeinfer rule against a synthesised
    Call whose source is a plain global-storage tensor. Assert the
    resulting TensorType pins the requested ShardLayout while
    preserving the logical shape.
    """
    dtype = getattr(DType, src_dtype_name)
    src_ty = TensorType(shape=src_shape, dtype=dtype, layout=None, storage=StorageKind.GMEM)
    src = Var(type=src_ty, name="x")
    op = Reshard(layout=dst_layout, storage=StorageKind.RMEM)

    call = Call(type=src_ty, target=op, args=(src,))
    out_ty = inference_type(call)
    assert isinstance(out_ty, TensorType), f"expected TensorType, got {out_ty}"
    assert out_ty.layout is dst_layout, "output layout must reference the requested ShardLayout"

    assert out_ty.shape == src_shape
    assert out_ty.dtype == dtype
    assert out_ty.storage == StorageKind.RMEM


def _bound_fragment(atom, role: str) -> ShardLayout:
    pattern = selected_pattern(atom.role(role), atom.bindings).layout
    bindings = dict(atom.bindings)
    if role == "C" and "n" in bindings:
        bindings["c"] = bindings["n"] // 8
    layout = Layout(
        fixed_pattern_value(pattern.layout.shape, bindings),
        fixed_pattern_value(pattern.layout.strides, bindings),
    )
    return ShardLayout(layout, pattern.attrs, atom.required_execution_mesh)
