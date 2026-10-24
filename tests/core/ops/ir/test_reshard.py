"""Cover Reshard materialization strides and rejected destinations.

Storage changes require explicit layouts and cannot target ``umat``. The target
memory owner selects shared or per-instance strides; without a target the default
is shared. Dynamic axes have symbolic compact strides.

See [hir §1.3](docs/spec/hir.md#13-op).
"""

from __future__ import annotations

import pytest

from tests.core.ops.ir.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.dsl.storage import gmem, rmem
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import Broadcast, Layout, Mesh, ShardLayout, Topology, make_tensor_type
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard_layout import Split
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.buffer_alias import aliased_operand


@pytest.mark.parametrize("storage", [StorageKind.GMEM, StorageKind.SMEM, StorageKind.RMEM])
def test_compact_and_factored_plain_layouts_share_locations(storage):
    compact = make_tensor_type((64,), layout=Layout((64,), (1,)), storage=storage)
    factored = make_tensor_type((64,), layout=Layout(((8, 8),), ((8, 1),)), storage=storage)
    assert aliases(compact, factored) == 0
    assert aliases(factored, compact) == 0


def test_large_gmem_factorization_reuses_the_source():
    compact = make_tensor_type(
        (1, 2048, 8192),
        layout=Layout((1, 2048, 8192), (16777216, 8192, 1)),
        storage=StorageKind.GMEM,
    )
    mesh = Mesh((Topology("cta", 128),), Layout((128,), (1,)), ("tile",))
    sharded = make_tensor_type(
        compact.shape,
        storage=StorageKind.GMEM,
        layout=ShardLayout(Layout((1, 2048, 128, 64), (16777216, 8192, 64, 1)), (Split(2),), mesh),
    )
    assert aliases(compact, sharded) == 0
    assert aliases(sharded, compact) == 0


def aliases(source, result):
    value = Var("source", type=source)
    call = Call(Reshard(layout=result.layout, storage=result.storage), (value,), type=result)
    return aliased_operand(call)


@pytest.mark.parametrize("storage", (StorageKind.GMEM, StorageKind.RMEM))
def test_broadcast_split_transitions_follow_address_spaces(storage):
    mesh = Mesh((Topology("thread", 4),), Layout((4,), (1,)), ("unit",))
    broadcast = make_tensor_type(
        (64,), layout=ShardLayout(Layout((4, 16), (16, 1)), (Broadcast(),), mesh), storage=storage
    )
    split = make_tensor_type(
        (64,),
        layout=ShardLayout(
            Layout((4, 16), (0 if storage is StorageKind.RMEM else 16, 1)), (Split(0),), mesh
        ),
        storage=storage,
    )
    expected = None if storage is StorageKind.RMEM else 0
    assert aliases(broadcast, split) == expected
    assert aliases(split, broadcast) == expected


@pytest.mark.parametrize("storage", (StorageKind.GMEM, StorageKind.RMEM))
def test_repartition_follows_independent_split_coordinates(storage):
    mesh = Mesh((Topology("cta", 4),), Layout((4,), (1,)), ("tile",))
    other = Mesh((Topology("cta", 8),), Layout((8,), (1,)), ("tile",))
    zero = storage is StorageKind.RMEM
    source = make_tensor_type(
        (64,),
        layout=ShardLayout(Layout((4, 16), (0 if zero else 16, 1)), (Split(0),), mesh),
        storage=storage,
    )
    result = make_tensor_type(
        (64,),
        layout=ShardLayout(Layout((8, 8), (0 if zero else 8, 1)), (Split(0),), other),
        storage=storage,
    )
    assert aliases(source, result) == (None if zero else 0)


@pytest.mark.parametrize("storage", (StorageKind.GMEM, StorageKind.SMEM, StorageKind.RMEM))
def test_reshard_transpose_requires_data_movement(storage):
    source = make_tensor_type((4, 4), layout=Layout((4, 4), (4, 1)), storage=storage)
    result = make_tensor_type((4, 4), layout=Layout((4, 4), (1, 4)), storage=storage)
    assert aliases(source, result) is None


def test_equivalent_private_split_factorizations_reuse_storage():
    mesh = Mesh((Topology("thread", 4),), Layout((4,), (1,)), ("unit",))
    source = make_tensor_type(
        (64,),
        layout=ShardLayout(Layout((4, 16), (0, 1)), (Split(0),), mesh),
        storage=StorageKind.RMEM,
    )
    result = make_tensor_type(
        (64,),
        layout=ShardLayout(Layout((1, 4, 8, 2), (0, 0, 2, 1)), (Split(1),), mesh),
        storage=StorageKind.RMEM,
    )
    assert aliases(source, result) == 0


def _shard_layout(shape) -> ShardLayout:
    return ShardLayout(
        layout=Layout(shape=shape, strides=tuple([1] * len(shape))),
        attrs=(),
        mesh=Mesh((Topology("cta", 128),), Layout((128,), (1,)), ("g",)),
    )


_SL_PRESERVES_SHAPE = _shard_layout((1, 8, 192))


_MESH_H2L = Mesh(
    topologies=(Topology("thread", 4),), layout=Layout(shape=(4,), strides=(1,)), names=("t",)
)


_SL_H2L = ShardLayout(
    layout=Layout(shape=(2, 4, 128), strides=None), attrs=(Split(1),), mesh=_MESH_H2L
)


_MESH_L2H = Mesh(
    topologies=(Topology("thread", 4),), layout=Layout(shape=(4,), strides=(1,)), names=("w",)
)


_REG_L2H_LAYOUT = ShardLayout(
    layout=Layout(shape=(4, 64), strides=(0, 1)), attrs=(Split(0),), mesh=_MESH_L2H
)


_SL_L2H = ShardLayout(layout=Layout(shape=(4, 64), strides=None), attrs=(Split(0),), mesh=_MESH_L2H)


def _materialized(shape, strides, attrs, mesh):
    return ShardLayout(layout=Layout(shape=shape, strides=strides), attrs=attrs, mesh=mesh)


_S_DYN = DimVar("seq_len", 1, 3)


_MESH_DYN = Mesh(topologies=(Topology("cta", 8),), layout=Layout(shape=(8,), strides=(1,)))


_SL_DYN_BARE = ShardLayout(
    layout=Layout(shape=(1, _S_DYN, 32, 128), strides=None),
    attrs=(Split(axis=2),),
    mesh=_MESH_DYN,
)


CASES = [
    TypeInferCase(
            "destination_umat_rejected",
            Reshard(layout=_SL_PRESERVES_SHAPE, storage=StorageKind.UMAT),
            (make_tensor_type((1, 1536)),),
            ExpectedError(match="unmaterialized"),
        ),
    TypeInferCase(
            "storage_change_without_layout_errors",
            Reshard(storage=rmem),
            (make_tensor_type((1, 1536)),),
            ExpectedError(match="storage change requires"),
        ),
    TypeInferCase(
            "low_to_high_sugar_materializes_shared",
            Reshard(layout=_SL_L2H, storage=gmem),
            (make_tensor_type((4, 64), storage=rmem, layout=_REG_L2H_LAYOUT),),
            make_tensor_type(
                (4, 64), storage=gmem, layout=_materialized((4, 64), (64, 1), (Split(0),), _MESH_L2H)
            ),
        ),
    TypeInferCase(
            "unmaterialized_to_high_sugar_materializes_shared",
            Reshard(layout=_SL_L2H, storage=gmem),
            (make_tensor_type((4, 64), storage=StorageKind.UMAT),),
            make_tensor_type(
                (4, 64),
                storage=gmem,
                layout=_materialized((4, 64), (64, 1), (Split(0),), _MESH_L2H),
            ),
        ),
    TypeInferCase(
            "same_storage_dynamic_bare_axis_shared_engine",
            Reshard(layout=_SL_DYN_BARE, storage=None),
            (make_tensor_type((1, _S_DYN, 32, 128)),),
            make_tensor_type(
                (1, _S_DYN, 32, 128),
                storage=gmem,
                layout=_materialized(
                    (1, _S_DYN, 32, 128), (0, 32 * 128, 128, 1), (Split(2),), _MESH_DYN
                ),
            ),
        ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reshard_typeinfer(case):
    run_typeinfer_case(case)
