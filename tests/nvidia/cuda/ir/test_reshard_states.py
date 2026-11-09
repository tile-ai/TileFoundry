"""Reshard admits Partial from a varying copy or the same contribution.

The error cells use the row and column ordering of STATES from the original
conversion table.
"""

from itertools import product

import pytest

from tests.core.ops.ir.op_case import infer_call
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import sharding
from tilefoundry_nvidia import cuda

STATES = (types.B(), types.V(), types.S(0), types.P("sum"), types.P("max"))
ERRORS = {
    "gpu": {(0, 3), (0, 4), (2, 3), (2, 4), (3, 4), (4, 3)},
    "cta": {(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (4, 3)},
}


@pytest.mark.parametrize("level", ("gpu", "cta"))
@pytest.mark.parametrize("before,after", tuple(product(STATES, repeat=2)))
def test_reshard_partial_admission(level, before, after):
    target = cuda.Target("nvidia.h200_sxm")
    mesh = types.Mesh((types.Topology(level, 4),), types.Layout((4,), (1,)))
    address = types.Layout((8,), (1,))
    source = types.TensorType(
        (8,), expr.DType.f32, types.ShardLayout(address, (before,), mesh), expr.StorageKind.GMEM
    )
    operation = sharding.Reshard(layout=types.ShardLayout(address, (after,), mesh))
    allowed = (STATES.index(before), STATES.index(after)) not in ERRORS[level]
    if allowed:
        result = infer_call(operation, source, target=target)
        assert result.layout == operation.layout
    else:
        with pytest.raises(expr.VerifyError, match="cannot become"):
            infer_call(operation, source, target=target)
