"""Cover Reshard materialization strides and rejected destinations.

Storage changes require explicit layouts and cannot target ``umat``. The target
memory owner selects shared or per-instance strides; without a target the default
is shared. Dynamic axes have symbolic compact strides.

See [hir §1.3](docs/spec/hir.md#13-op).
"""

from __future__ import annotations

import pytest

from tests.core.ops.ir.op_case import ExpectedError, OpCase, op_case_test
from tilefoundry.dsl import storage as dsl_storage
from tilefoundry.interfaces import access
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import sharding


@pytest.mark.parametrize("storage", [expr.StorageKind.GMEM, expr.StorageKind.SMEM, expr.StorageKind.RMEM])
def test_compact_factored_layouts_alias(storage):
    compact = types.TensorType.from_shape((64,), layout=types.Layout((64,), (1,)), storage=storage)
    factored = types.TensorType.from_shape((64,), layout=types.Layout(((8, 8),), ((8, 1),)), storage=storage)
    assert aliases(compact, factored) == 0
    assert aliases(factored, compact) == 0


def test_large_gmem_factorization_aliases_source():
    compact = types.TensorType.from_shape(
        (1, 2048, 8192),
        layout=types.Layout((1, 2048, 8192), (16777216, 8192, 1)),
        storage=expr.StorageKind.GMEM,
    )
    mesh = types.Mesh((types.Topology("cta", 128),), types.Layout((128,), (1,)), ("tile",))
    sharded = types.TensorType.from_shape(
        compact.shape,
        storage=expr.StorageKind.GMEM,
        layout=types.ShardLayout(types.Layout((1, 2048, 128, 64), (16777216, 8192, 64, 1)), (types.Split(2),), mesh),
    )
    assert aliases(compact, sharded) == 0
    assert aliases(sharded, compact) == 0


def aliases(source, result):
    value = expr.Var("source", type=source)
    call = expr.Call(sharding.Reshard(layout=result.layout, storage=result.storage), (value,), type=result)
    return access.aliased_operand(call)


@pytest.mark.parametrize("storage", (expr.StorageKind.GMEM, expr.StorageKind.RMEM))
def test_broadcast_split_alias_follows_storage(storage):
    mesh = types.Mesh((types.Topology("thread", 4),), types.Layout((4,), (1,)), ("unit",))
    broadcast = types.TensorType.from_shape(
        (64,), layout=types.ShardLayout(types.Layout((4, 16), (16, 1)), (types.Broadcast(),), mesh), storage=storage
    )
    split = types.TensorType.from_shape(
        (64,),
        layout=types.ShardLayout(
            types.Layout((4, 16), (0 if storage is expr.StorageKind.RMEM else 16, 1)), (types.Split(0),), mesh
        ),
        storage=storage,
    )
    expected = None if storage is expr.StorageKind.RMEM else 0
    assert aliases(broadcast, split) == expected
    assert aliases(split, broadcast) == expected


@pytest.mark.parametrize("storage", (expr.StorageKind.GMEM, expr.StorageKind.RMEM))
def test_repartition_follows_independent_split_coordinates(storage):
    mesh = types.Mesh((types.Topology("cta", 4),), types.Layout((4,), (1,)), ("tile",))
    other = types.Mesh((types.Topology("cta", 8),), types.Layout((8,), (1,)), ("tile",))
    zero = storage is expr.StorageKind.RMEM
    source = types.TensorType.from_shape(
        (64,),
        layout=types.ShardLayout(types.Layout((4, 16), (0 if zero else 16, 1)), (types.Split(0),), mesh),
        storage=storage,
    )
    result = types.TensorType.from_shape(
        (64,),
        layout=types.ShardLayout(types.Layout((8, 8), (0 if zero else 8, 1)), (types.Split(0),), other),
        storage=storage,
    )
    assert aliases(source, result) == (None if zero else 0)


@pytest.mark.parametrize("storage", (expr.StorageKind.GMEM, expr.StorageKind.SMEM, expr.StorageKind.RMEM))
def test_reshard_transpose_requires_data_movement(storage):
    source = types.TensorType.from_shape((4, 4), layout=types.Layout((4, 4), (4, 1)), storage=storage)
    result = types.TensorType.from_shape((4, 4), layout=types.Layout((4, 4), (1, 4)), storage=storage)
    assert aliases(source, result) is None


def _lanes(layout) -> types.Mesh:
    return types.Mesh((types.Topology("thread", 64),), layout, ("a", "b"))


_LANES = _lanes(types.Layout((4, 8), (8, 1)))


@pytest.mark.parametrize("storage", (expr.StorageKind.GMEM, expr.StorageKind.SMEM, expr.StorageKind.RMEM))
@pytest.mark.parametrize(
    ("held_by", "moves"),
    (
        (_lanes(types.ComposedLayout(None, 0, types.Layout((4, 8), (8, 1)))), False),
        (_lanes(types.ComposedLayout(None, 32, types.Layout((4, 8), (8, 1)))), True),
        (_lanes(types.Layout((4, 8), (1, 4))), True),
        (types.Mesh((types.Topology("cta", 64),), types.Layout((4, 8), (8, 1)), ("a", "b")), True),
    ),
    ids=(
        "the same threads with a zero offset",
        "other threads",
        "the same threads reordered",
        "the same numbering on CTAs",
    ),
)
def test_registers_alias_only_where_each_element_keeps_its_thread(storage, held_by, moves):
    """Register storage aliases only when every element stays with its thread; shared may move."""
    private = storage is expr.StorageKind.RMEM
    layout = types.Layout((4, 8), (0, 0) if private else (8, 1))
    source = types.TensorType.from_shape(
        (32,), layout=types.ShardLayout(layout, (types.Split(0), types.Split(1)), _LANES), storage=storage
    )
    result = types.TensorType.from_shape(
        (32,), layout=types.ShardLayout(layout, (types.Split(0), types.Split(1)), held_by), storage=storage
    )
    assert aliases(source, result) == (None if private and moves else 0)


def test_registers_move_when_the_mesh_axes_trade_modes():
    """The same mesh and strides; each mesh axis now holds the mode the other one held."""
    mesh = types.Mesh((types.Topology("thread", 16),), types.Layout((4, 4), (4, 1)), ("a", "b"))
    layout = types.Layout((4, 4, 2), (8, 2, 1))
    source = types.TensorType.from_shape(
        (32,), layout=types.ShardLayout(layout, (types.Split(0), types.Split(1)), mesh), storage=expr.StorageKind.RMEM
    )
    result = types.TensorType.from_shape(
        (32,), layout=types.ShardLayout(layout, (types.Split(1), types.Split(0)), mesh), storage=expr.StorageKind.RMEM
    )
    assert aliases(source, result) is None


def test_registers_move_within_a_thread():
    """Each element stays on its thread but changes register."""
    mesh = types.Mesh((types.Topology("thread", 4),), types.Layout((4,), (1,)), ("unit",))
    source = types.TensorType.from_shape(
        (16,),
        layout=types.ShardLayout(types.Layout((4, 2, 2), (4, 2, 1)), (types.Split(0),), mesh),
        storage=expr.StorageKind.RMEM,
    )
    result = types.TensorType.from_shape(
        (16,),
        layout=types.ShardLayout(types.Layout((4, 2, 2), (4, 1, 2)), (types.Split(0),), mesh),
        storage=expr.StorageKind.RMEM,
    )
    assert aliases(source, result) is None


def test_broadcast_registers_alias_in_another_notation():
    """Every thread of the broadcast axis holds a copy; factoring the local mode keeps them."""
    source = types.TensorType.from_shape(
        (32,),
        layout=types.ShardLayout(types.Layout((8, 4), (4, 1)), (types.Broadcast(), types.Split(0)), _LANES),
        storage=expr.StorageKind.RMEM,
    )
    result = types.TensorType.from_shape(
        (32,),
        layout=types.ShardLayout(types.Layout((8, 2, 2), (4, 2, 1)), (types.Broadcast(), types.Split(0)), _LANES),
        storage=expr.StorageKind.RMEM,
    )
    assert aliases(source, result) == 0
    one = types.Mesh(
        (types.Topology("thread", 64),), types.ComposedLayout(None, 0, types.Layout((1, 8), (8, 1))), ("a", "b")
    )
    first = types.TensorType.from_shape(
        (32,),
        layout=types.ShardLayout(types.Layout((8, 4), (4, 1)), (types.Broadcast(), types.Split(0)), one),
        storage=expr.StorageKind.RMEM,
    )
    assert aliases(source, first) is None, "copies on four threads are not copies on one"


def test_private_split_factorizations_alias():
    mesh = types.Mesh((types.Topology("thread", 4),), types.Layout((4,), (1,)), ("unit",))
    source = types.TensorType.from_shape(
        (64,),
        layout=types.ShardLayout(types.Layout((4, 16), (0, 1)), (types.Split(0),), mesh),
        storage=expr.StorageKind.RMEM,
    )
    result = types.TensorType.from_shape(
        (64,),
        layout=types.ShardLayout(types.Layout((1, 4, 8, 2), (0, 0, 2, 1)), (types.Split(1),), mesh),
        storage=expr.StorageKind.RMEM,
    )
    assert aliases(source, result) == 0


def _shard_layout(shape) -> types.ShardLayout:
    return types.ShardLayout(
        layout=types.Layout(shape=shape, strides=tuple([1] * len(shape))),
        attrs=(),
        mesh=types.Mesh((types.Topology("cta", 128),), types.Layout((128,), (1,)), ("g",)),
    )


_SL_PRESERVES_SHAPE = _shard_layout((1, 8, 192))
_MESH_H2L = types.Mesh(
    topologies=(types.Topology("thread", 4),), layout=types.Layout(shape=(4,), strides=(1,)), names=("t",)
)
_SL_H2L = types.ShardLayout(
    layout=types.Layout(shape=(2, 4, 128), strides=None), attrs=(types.Split(1),), mesh=_MESH_H2L
)
_MESH_L2H = types.Mesh(
    topologies=(types.Topology("thread", 4),), layout=types.Layout(shape=(4,), strides=(1,)), names=("w",)
)
_REG_L2H_LAYOUT = types.ShardLayout(
    layout=types.Layout(shape=(4, 64), strides=(0, 1)), attrs=(types.Split(0),), mesh=_MESH_L2H
)
_SL_L2H = types.ShardLayout(layout=types.Layout(shape=(4, 64), strides=None), attrs=(types.Split(0),), mesh=_MESH_L2H)


def _materialized(shape, strides, attrs, mesh):
    return types.ShardLayout(layout=types.Layout(shape=shape, strides=strides), attrs=attrs, mesh=mesh)


_S_DYN = expr.DimVar("seq_len", 1, 3)
_MESH_DYN = types.Mesh(topologies=(types.Topology("cta", 8),), layout=types.Layout(shape=(8,), strides=(1,)))
_SL_DYN_BARE = types.ShardLayout(
    layout=types.Layout(shape=(1, _S_DYN, 32, 128), strides=None), attrs=(types.Split(axis=2),), mesh=_MESH_DYN
)
CASES = (
    OpCase(
        "destination_umat_rejected",
        sharding.Reshard(layout=_SL_PRESERVES_SHAPE, storage=expr.StorageKind.UMAT),
        (types.TensorType.from_shape((1, 1536)),),
        error=ExpectedError(match="unmaterialized"),
    ),
    OpCase(
        "storage_change_without_layout_errors",
        sharding.Reshard(storage=dsl_storage.rmem),
        (types.TensorType.from_shape((1, 1536)),),
        error=ExpectedError(match="storage change requires"),
    ),
    OpCase(
        "low_to_high_sugar_materializes_shared",
        sharding.Reshard(layout=_SL_L2H, storage=dsl_storage.gmem),
        (types.TensorType.from_shape((4, 64), storage=dsl_storage.rmem, layout=_REG_L2H_LAYOUT),),
        type=types.TensorType.from_shape(
            (4, 64), storage=dsl_storage.gmem, layout=_materialized((4, 64), (64, 1), (types.Split(0),), _MESH_L2H)
        ),
    ),
    OpCase(
        "unmaterialized_to_high_sugar_materializes_shared",
        sharding.Reshard(layout=_SL_L2H, storage=dsl_storage.gmem),
        (types.TensorType.from_shape((4, 64), storage=expr.StorageKind.UMAT),),
        type=types.TensorType.from_shape(
            (4, 64), storage=dsl_storage.gmem, layout=_materialized((4, 64), (64, 1), (types.Split(0),), _MESH_L2H)
        ),
    ),
    OpCase(
        "same_storage_dynamic_bare_axis_shared_engine",
        sharding.Reshard(layout=_SL_DYN_BARE, storage=None),
        (types.TensorType.from_shape((1, _S_DYN, 32, 128)),),
        type=types.TensorType.from_shape(
            (1, _S_DYN, 32, 128),
            storage=dsl_storage.gmem,
            layout=_materialized(
                (1, _S_DYN, 32, 128), (0, 32 * 128, 128, 1), (types.Split(2),), _MESH_DYN
            ),
        ),
    ),
)
test_op_case = op_case_test(CASES)
