"""Reshard admits Partial from a varying copy or the same contribution.

The error cells use the row and column ordering of STATES from the original
conversion table.
"""

from itertools import product

import pytest

from tests.core.ops.ir.op_case import infer_call
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import (
    B,
    DType,
    Layout,
    Mesh,
    P,
    S,
    ShardLayout,
    StorageKind,
    TensorType,
    Topology,
    V,
)
from tilefoundry_nvidia import CudaTarget

STATES = (B(), V(), S(0), P("sum"), P("max"))
ERRORS = {
    "gpu": {(0, 3), (0, 4), (2, 3), (2, 4), (3, 4), (4, 3)},
    "cta": {(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (4, 3)},
}


@pytest.mark.parametrize("level", ("gpu", "cta"))
@pytest.mark.parametrize("before,after", tuple(product(STATES, repeat=2)))
def test_reshard_partial_admission(level, before, after):
    target = CudaTarget("nvidia.h200_sxm")
    mesh = Mesh((Topology(level, 4),), Layout((4,), (1,)))
    address = Layout((8,), (1,))
    source = TensorType((8,), DType.f32, ShardLayout(address, (before,), mesh), StorageKind.GMEM)
    operation = Reshard(layout=ShardLayout(address, (after,), mesh))
    allowed = (STATES.index(before), STATES.index(after)) not in ERRORS[level]
    if allowed:
        result = infer_call(operation, source, target=target)
        assert result.layout == operation.layout
    else:
        with pytest.raises(VerifyError, match="cannot become"):
            infer_call(operation, source, target=target)
