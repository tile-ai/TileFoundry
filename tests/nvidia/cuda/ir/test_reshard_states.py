"""Reshard admits Partial from the same contribution; Bitcast states Varying copies equal or summed.

The error cells use the row and column ordering of STATES from the original
conversion table. On gmem the gpu axis separates copies and the cta axis does
not. Where a mesh axis spans several copies, a Reshard refuses Varying to
Broadcast or Partial: the Bitcast form that keeps the layout states it.
"""

from itertools import product

import pytest

from tests.core.ops.ir.op_case import infer_call
from tilefoundry.ir import expr, types, utils
from tilefoundry.ir.hir import sharding, tensor
from tilefoundry_nvidia import cuda

STATES = (types.B(), types.V(), types.S(0), types.P("sum"), types.P("max"))
ERRORS = {
    "gpu": {(0, 3), (0, 4), (1, 0), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (4, 3)},
    "cta": {(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (4, 3)},
}
ONE_UNIT = {(0, 3), (0, 4), (2, 3), (2, 4), (3, 4), (4, 3)}
ADDRESS = types.Layout((8,), (1,))


def held(attr, mesh):
    return types.TensorType(
        (8,), expr.DType.f32, types.ShardLayout(ADDRESS, (attr,), mesh), expr.StorageKind.GMEM
    )


@pytest.mark.parametrize(("level", "units"), (("gpu", 4), ("cta", 4), ("gpu", 1)))
@pytest.mark.parametrize("before,after", tuple(product(STATES, repeat=2)))
def test_reshard_partial_admission(level, units, before, after):
    mesh = types.Mesh((types.Topology(level, units),), types.Layout((units,), (1,)))
    operation = sharding.Reshard(layout=types.ShardLayout(ADDRESS, (after,), mesh))
    errors = ERRORS[level] if units > 1 else ONE_UNIT
    allowed = (STATES.index(before), STATES.index(after)) not in errors
    if allowed:
        result = infer_call(operation, held(before, mesh), target=cuda.Target("nvidia.h200_sxm"))
        assert result.layout == utils.canonical_layout(operation.layout)
    else:
        with pytest.raises(expr.VerifyError, match="cannot become"):
            infer_call(operation, held(before, mesh), target=cuda.Target("nvidia.h200_sxm"))


@pytest.mark.parametrize("before,after", tuple(product(STATES, repeat=2)))
def test_bitcast_keeping_the_layout_only_declares_varying_copies(before, after):
    """On gmem, a Bitcast keeping layout and mesh admits V to B or P, and B to V."""
    mesh = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)))
    operation = tensor.Bitcast(layout=types.ShardLayout(ADDRESS, (after,), mesh))
    if (
        before == after
        or (before == types.V() and isinstance(after, (types.Broadcast, types.Partial)))
        or (before == types.B() and after == types.V())
    ):
        result = infer_call(operation, held(before, mesh), target=cuda.Target("nvidia.h200_sxm"))
        assert result.layout == utils.canonical_layout(operation.layout)
    else:
        with pytest.raises(expr.VerifyError):
            infer_call(operation, held(before, mesh), target=cuda.Target("nvidia.h200_sxm"))


_GPUS = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)))


@pytest.mark.parametrize(
    ("source", "layout", "allowed"),
    (
        (types.TensorType((8,), expr.DType.f32, ADDRESS, expr.StorageKind.GMEM), types.ShardLayout(ADDRESS, (types.V(),), _GPUS), True),
        (held(types.V(), _GPUS), ADDRESS, True),
        (types.TensorType((8,), expr.DType.f32, types.Layout((8,), (2,)), expr.StorageKind.GMEM), types.ShardLayout(ADDRESS, (types.V(),), _GPUS), False),
        (held(types.V(), _GPUS[0:2]), types.ShardLayout(ADDRESS, (types.B(),), _GPUS), False),
    ),
    ids=("plain_source_to_varying", "varying_source_to_plain", "other_address_layout", "other_mesh"),
)
def test_bitcast_keeping_the_layout_reads_a_plain_end_as_broadcast(source, layout, allowed):
    operation = tensor.Bitcast(layout=layout)
    if allowed:
        assert infer_call(operation, source, target=cuda.Target("nvidia.h200_sxm")).layout == layout
    else:
        with pytest.raises(expr.VerifyError):
            infer_call(operation, source, target=cuda.Target("nvidia.h200_sxm"))


@pytest.mark.parametrize("kept", (True, False), ids=("offset_stated_again", "offset_dropped"))
def test_bitcast_keeping_the_layout_keeps_a_source_offset(kept):
    """Form (ii) keeps every address: an offset around the source shard stays around the result."""
    mesh = types.Mesh((types.Topology("gpu", 4),), types.Layout((4,), (1,)))
    source = types.TensorType(
        (8,),
        expr.DType.f32,
        types.make_composed_layout(None, 4, types.ShardLayout(ADDRESS, (types.V(),), mesh)),
        expr.StorageKind.GMEM,
    )
    stated = types.ShardLayout(ADDRESS, (types.B(),), mesh)
    layout = types.make_composed_layout(None, 4, stated) if kept else stated
    operation = tensor.Bitcast(layout=layout)
    if kept:
        assert infer_call(operation, source, target=cuda.Target("nvidia.h200_sxm")).layout == utils.canonical_layout(layout)
    else:
        with pytest.raises(expr.VerifyError):
            infer_call(operation, source, target=cuda.Target("nvidia.h200_sxm"))
