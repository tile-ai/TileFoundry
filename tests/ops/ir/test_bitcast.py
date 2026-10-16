"""A change in mode factorization preserves row-major logical addresses."""

from types import SimpleNamespace

import pytest
import torch

from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.hir.tensor.bitcast import _address_mapping, _address_relation, _eval_bitcast
from tilefoundry.ir.types import DType, Layout, StorageKind, TensorType


@pytest.mark.parametrize("source_layout,result_layout", [
    (Layout((64,), (1,)), Layout(((8, 8),), ((8, 1),))),
    (Layout(((8, 8),), ((8, 1),)), Layout((64,), (1,))),
])
def test_compact_and_factored_bitcasts_preserve_logical_elements(source_layout, result_layout):
    assert _address_mapping(source_layout, result_layout, 64, (64,), (64,)) == tuple(range(64))
    assert _address_relation(source_layout, result_layout, (64,), (64,)).is_identity()
    data = torch.arange(64, dtype=torch.float32)
    source = TensorType((64,), DType.f32, source_layout, StorageKind.RMEM)
    result = TensorType((64,), DType.f32, result_layout, StorageKind.RMEM)
    actual = _eval_bitcast(SimpleNamespace(args=(TensorValue(data=data, type=source),), result_type=result))
    torch.testing.assert_close(actual.data, data, rtol=0, atol=0)
