"""CUDA Reshard materialization for target-owned memory."""

from __future__ import annotations

from tests.core.ops.ir.op_case import OpCase, op_case_test
from tests.core.ops.ir.test_reshard import (
    _MESH_DYN,
    _MESH_H2L,
    _S_DYN,
    _SL_DYN_BARE,
    _SL_H2L,
    _materialized,
)
from tilefoundry.dsl import storage
from tilefoundry.ir import types
from tilefoundry.ir.hir import sharding
from tilefoundry_nvidia import cuda

CASES = (
    OpCase(
        "high_to_low_sugar_materializes_per_instance",
        sharding.Reshard(
            layout=types.ShardLayout.from_modes(
                _SL_H2L.shape,
                _SL_H2L.mesh,
                _SL_H2L.attrs,
                storage=storage.rmem,
                owners=cuda.Target("nvidia.h200_sxm").storage_owners,
            ),
            storage=storage.rmem,
        ),
        (types.TensorType.from_shape((2, 4, 128)),),
        type=types.TensorType.from_shape(
            (2, 4, 128),
            storage=storage.rmem,
            layout=_materialized((2, 4, 128), (128, 0, 1), (types.Split(1),), _MESH_H2L),
        ),
        target=cuda.Target("nvidia.h200_sxm"),
    ),
    OpCase(
        "high_to_low_dynamic_bare_axis_materializes_per_instance",
        sharding.Reshard(
            layout=types.ShardLayout.from_modes(
                _SL_DYN_BARE.shape,
                _SL_DYN_BARE.mesh,
                _SL_DYN_BARE.attrs,
                storage=storage.rmem,
                owners=cuda.Target("nvidia.h200_sxm").storage_owners,
            ),
            storage=storage.rmem,
        ),
        (types.TensorType.from_shape((1, _S_DYN, 32, 128)),),
        type=types.TensorType.from_shape(
            (1, _S_DYN, 32, 128),
            storage=storage.rmem,
            layout=_materialized((1, _S_DYN, 32, 128), (0, 4 * 128, 0, 1), (types.Split(2),), _MESH_DYN),
        ),
        target=cuda.Target("nvidia.h200_sxm"),
    ),
)
test_op_case = op_case_test(CASES)
