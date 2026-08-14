"""Arange's symbolic half-open coordinate contract."""

from __future__ import annotations

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.cost_utils import CostCase, run_cost_case
from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.specialize import residual_dims, specialize_concretely
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import ceildiv
from tilefoundry.ir.types.dim_isl import normalize_dim
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.contexts import TrafficBytes, TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_N = DimVar("arange_n", 1, 17)


def _infer(op: Arange) -> TensorType:
    call = Call(type=TensorType.umat_scalar(), target=op, args=())
    return TypeInferVisitor(TypeInferContext()).visit(call)


@func
def _symbolic_arange(x: Tensor[(_N,), "f32"]):
    return tf.arange(_N, start=1, step=2, dtype="i32")


def test_arange_static_type_evaluation_and_cost():
    op = Arange(end=8, start=2, step=2, dtype=DType.i64)

    assert _infer(op) == TensorType(
        shape=(3,), dtype=DType.i64, layout=None, storage=StorageKind.UMAT
    )
    run_eval_case(
        EvalCase(
            "half_open",
            op,
            (),
            torch.tensor([2, 4, 6], dtype=torch.int64),
        )
    )
    run_cost_case(CostCase("arange", op, (), traffic=(TrafficBytes(),)))


def test_arange_symbolic_extent_resolves_from_runtime_shape():
    call = _symbolic_arange.body
    assert isinstance(call, Call) and isinstance(call.target, Arange)
    assert call.type.shape == (normalize_dim(ceildiv(_N - 1, 2)),)
    assert call.type.dtype == DType.i32

    actual = evaluate(_symbolic_arange, torch.zeros(8), device="cpu")
    torch.testing.assert_close(actual, torch.tensor([1, 3, 5, 7], dtype=torch.int32))

    concrete = specialize_concretely(_symbolic_arange, {"arange_n": 8})
    assert residual_dims(concrete) == ()
    assert isinstance(concrete.body.target, Arange)
    assert concrete.body.target.end == 8


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (Arange(end=True), "end must be a static or symbolic shape dimension"),
        (Arange(end=8, step=0), "step must be a positive static integer"),
        (Arange(end=2, start=3), "requires end >= start"),
        (Arange(end=8, dtype=DType.f32), "dtype must be i32 or i64"),
    ],
)
def test_arange_rejects_unsupported_attributes(op, message):
    with pytest.raises(VerifyError, match=message):
        _infer(op)
