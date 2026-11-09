"""A change in mode factorization preserves row-major logical addresses."""

from types import SimpleNamespace

import pytest
import torch

from tilefoundry import core
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import expr, types
from tilefoundry.ir.hir import tensor


@pytest.mark.parametrize("source_layout,result_layout", [
    (types.Layout((64,), (1,)), types.Layout(((8, 8),), ((8, 1),))),
    (types.Layout(((8, 8),), ((8, 1),)), types.Layout((64,), (1,))),
])
def test_compact_and_factored_bitcasts_preserve_logical_elements(source_layout, result_layout):
    assert tensor.address_mapping(source_layout, result_layout, 64, (64,), (64,)) == tuple(range(64))
    assert tensor.address_relation(source_layout, result_layout, (64,), (64,)).is_identity()
    data = torch.arange(64, dtype=torch.float32)
    source = types.TensorType((64,), expr.DType.f32, source_layout, expr.StorageKind.RMEM)
    result = types.TensorType((64,), expr.DType.f32, result_layout, expr.StorageKind.RMEM)
    actual = tensor.eval_bitcast(SimpleNamespace(args=(evaluator.TensorValue(core.ShardTensor(source, data)),), result_type=result))
    torch.testing.assert_close(actual.data, data, rtol=0, atol=0)
