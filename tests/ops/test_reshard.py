"""Cover Reshard materialization strides and rejected destinations.

Storage changes require explicit layouts and cannot target ``umat``. Direction
selects shared C-order or per-instance strides; dynamic axes are accepted only
where the chosen storage form can represent them.

See [hir §1.3](docs/spec/hir.md#13-op).
"""

from __future__ import annotations

import pytest

from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.dsl.storage import gmem, rmem
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import make_tensor_type
from tilefoundry.ir.types.dim import DimMul, DimVar, simplify_dim
from tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Topology, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split
from tilefoundry.ir.types.storage import StorageKind


def _shard_layout(shape) -> ShardLayout:
    return ShardLayout(
        layout=Layout(shape=shape, strides=tuple([1] * len(shape))),
        attrs=(),
        mesh=make_mesh((128,), topology=Topology("cta", 128)),
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


_S_DYN = DimVar("seq_len", 1, 4)
_MESH_DYN = Mesh(topologies=(Topology("cta", 8),), layout=Layout(shape=(8,), strides=(1,)))
_SL_DYN_BARE = ShardLayout(
    layout=Layout(shape=(1, _S_DYN, 32, 128), strides=None),
    attrs=(Split(axis=2),),
    mesh=_MESH_DYN,
)


_DYN_OUTER_STRIDE = simplify_dim(DimMul, (32 * 128, _S_DYN))


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
        "high_to_low_sugar_materializes_per_instance",
        Reshard(layout=_SL_H2L, storage=rmem),
        (make_tensor_type((2, 4, 128)),),
        make_tensor_type(
            (2, 4, 128),
            storage=rmem,
            layout=_materialized((2, 4, 128), (128, 0, 1), (Split(1),), _MESH_H2L),
        ),
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
                (1, _S_DYN, 32, 128), (_DYN_OUTER_STRIDE, 32 * 128, 128, 1), (Split(2),), _MESH_DYN
            ),
        ),
    ),
    TypeInferCase(
        "high_to_low_dynamic_bare_axis_materializes_per_instance",
        Reshard(layout=_SL_DYN_BARE, storage=rmem),
        (make_tensor_type((1, _S_DYN, 32, 128)),),
        make_tensor_type(
            (1, _S_DYN, 32, 128),
            storage=rmem,
            layout=_materialized(
                (1, _S_DYN, 32, 128), (0, 4 * 128, 0, 1), (Split(2),), _MESH_DYN
            ),
        ),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reshard_typeinfer(case):
    run_typeinfer_case(case)
