"""TileFoundry's register-fragment invariants, independent of the golden layouts."""

from __future__ import annotations

from tilefoundry.interfaces import typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir.hir import sharding


def _check_split_extents_match_mesh(sl: types.ShardLayout) -> None:
    mesh_shape = types.flatten(sl.mesh.layout.shape)
    assert len(sl.attrs) == len(mesh_shape), (
        f"attrs len {len(sl.attrs)} != mesh rank {len(mesh_shape)}"
    )
    for mesh_i, attr in enumerate(sl.attrs):
        assert isinstance(attr, types.Split), f"mma fragment shard attrs must be Split, got {attr}"
        ax = attr.axis
        assert sl.layout.shape[ax] == mesh_shape[mesh_i], (
            f"split tensor axis {ax} extent {sl.layout.shape[ax]} != "
            f"mesh axis {mesh_i} extent {mesh_shape[mesh_i]}"
        )


def _assert_reshard_typeinfer_ok(
    src_shape: tuple[int, ...], src_dtype_name: str, dst_layout: types.ShardLayout
) -> None:
    """Assert reshard typeinfer ok.

    Run the registered Reshard typeinfer rule against a synthesised
    Call whose source is a plain global-storage tensor. Assert the
    resulting TensorType pins the requested ShardLayout while
    preserving the logical shape.
    """
    dtype = getattr(expr.DType, src_dtype_name)
    src_ty = types.TensorType(shape=src_shape, dtype=dtype, layout=None, storage=expr.StorageKind.GMEM)
    src = expr.Var(type=src_ty, name="x")
    op = sharding.Reshard(layout=dst_layout, storage=expr.StorageKind.RMEM)

    call = expr.Call(type=src_ty, target=op, args=(src,))
    out_ty = typeinfer.inference_type(call)
    assert isinstance(out_ty, types.TensorType), f"expected TensorType, got {out_ty}"
    assert out_ty.layout is dst_layout, "output layout must reference the requested ShardLayout"

    assert out_ty.shape == src_shape
    assert out_ty.dtype == dtype
    assert out_ty.storage == expr.StorageKind.RMEM


def _bound_fragment(atom, role: str) -> types.ShardLayout:
    declared = pattern.selected(atom.role(role), atom.bindings).layout
    bindings = dict(atom.bindings)
    if role == "C" and "n" in bindings:
        bindings["c"] = bindings["n"] // 8
    layout = types.Layout(
        pattern.fixed_value(declared.layout.shape, bindings),
        pattern.fixed_value(declared.layout.strides, bindings),
    )
    return types.ShardLayout(layout, declared.attrs, atom.required_execution_mesh)
