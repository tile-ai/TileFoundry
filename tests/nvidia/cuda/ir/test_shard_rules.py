"""Owner-aware Partial rejection and its type-inference diagnostics."""

import pytest

from tilefoundry.ir import expr, types, utils
from tilefoundry_nvidia import cuda


@pytest.mark.parametrize("levels", (("thread", "gpu"), ("gpu", "thread")))
def test_plain_multilinear_operand_reads_every_frame(levels):
    target = cuda.Target("nvidia.h200_sxm")
    address = types.Layout((8,), (1,))
    operands = tuple(
        types.TensorType(
            (8,),
            expr.DType.f32,
            types.ShardLayout(
                address,
                (types.P("max"),),
                types.Mesh((types.Topology(level, 2),), types.Layout((2,), (1,))),
            ),
            expr.StorageKind.SMEM,
        )
        for level in levels
    )
    plain = types.TensorType((8,), expr.DType.f32, address, expr.StorageKind.SMEM)
    with pytest.raises(ValueError, match="carries Partial.*operand plain:.*Varying"):
        utils.check_multilinear_partials(
            (("a", operands[0]), ("b", operands[1]), ("plain", plain)),
            target,
            kind=expr.BinaryKind.ADD,
        )
