"""Cover Reshard materialization strides and rejected destinations."""

from __future__ import annotations

import pytest

from tests.core.ops.ir.test_reshard import (
    _MESH_DYN,
    _MESH_H2L,
    _S_DYN,
    _SL_DYN_BARE,
    _SL_H2L,
    _materialized,
)
from tests.core.ops.ir.typeinfer_utils import (
    TypeInferCase,
    run_typeinfer_case,
)
from tilefoundry.dsl.storage import rmem
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import ShardLayout, make_tensor_type
from tilefoundry.ir.types.shard_layout import Split
from tilefoundry_nvidia import CudaTarget

CASES = [
    TypeInferCase(
            "high_to_low_sugar_materializes_per_instance",
            Reshard(
                layout=ShardLayout.from_modes(
                    _SL_H2L.shape,
                    _SL_H2L.mesh,
                    _SL_H2L.attrs,
                    storage=rmem,
                    owners=CudaTarget("nvidia.h200_sxm").storage_owners,
                ),
                storage=rmem,
            ),
            (make_tensor_type((2, 4, 128)),),
            make_tensor_type(
                (2, 4, 128),
                storage=rmem,
                layout=_materialized((2, 4, 128), (128, 0, 1), (Split(1),), _MESH_H2L),
            ),
            target=CudaTarget("nvidia.h200_sxm"),
        ),
    TypeInferCase(
            "high_to_low_dynamic_bare_axis_materializes_per_instance",
            Reshard(
                layout=ShardLayout.from_modes(
                    _SL_DYN_BARE.shape,
                    _SL_DYN_BARE.mesh,
                    _SL_DYN_BARE.attrs,
                    storage=rmem,
                    owners=CudaTarget("nvidia.h200_sxm").storage_owners,
                ),
                storage=rmem,
            ),
            (make_tensor_type((1, _S_DYN, 32, 128)),),
            make_tensor_type(
                (1, _S_DYN, 32, 128),
                storage=rmem,
                layout=_materialized((1, _S_DYN, 32, 128), (0, 4 * 128, 0, 1), (Split(2),), _MESH_DYN),
            ),
            target=CudaTarget("nvidia.h200_sxm"),
        ),
]

@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_reshard_typeinfer(case):
    run_typeinfer_case(case)
