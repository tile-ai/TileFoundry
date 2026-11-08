"""Owner-aware Partial rejection and its type-inference diagnostics."""

import pytest

from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    P,
    ShardLayout,
    StorageKind,
    TensorType,
    Topology,
)
from tilefoundry.ir.types.utils import check_multilinear_partials
from tilefoundry_nvidia import CudaTarget


@pytest.mark.parametrize("levels", (("thread", "gpu"), ("gpu", "thread")))
def test_plain_multilinear_operand_reads_every_frame(levels):
    target = CudaTarget("nvidia.h200_sxm")
    address = Layout((8,), (1,))
    types = tuple(
        TensorType(
            (8,),
            DType.f32,
            ShardLayout(address, (P("max"),), Mesh((Topology(level, 2),), Layout((2,), (1,)))),
            StorageKind.SMEM,
        )
        for level in levels
    )
    plain = TensorType((8,), DType.f32, address, StorageKind.SMEM)
    with pytest.raises(ValueError, match="carries Partial.*operand plain:.*Varying"):
        check_multilinear_partials(
            (("a", types[0]), ("b", types[1]), ("plain", plain)),
            target,
            kind=BinaryKind.ADD,
        )
