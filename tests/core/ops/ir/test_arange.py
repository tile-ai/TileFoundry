"""Arange's typed coordinate contract."""

from __future__ import annotations

import pytest
import torch

from tests.core.ops.ir.op_case import Cost, OpCase, check_op_case
from tilefoundry.dsl import *
from tilefoundry.evaluator.context import EvaluateContext
from tilefoundry.evaluator.registry import eval_registry
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.types.layout import Layout
from tilefoundry.ir.types.mesh import make_mesh
from tilefoundry.visitor_registry.contexts import TrafficBytes, TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor

_COORD_MESH = Mesh((Topology("thread", 4),), Layout((4,), (1,)), ("t",))
_COORD_INDEX = TensorType.scalar(DType.i64, storage=StorageKind.RMEM)


def _coord(mesh: Mesh = _COORD_MESH) -> Call:
    axis = Constant(value=0, type=_COORD_INDEX)
    return Call(type=_COORD_INDEX, target=MeshCoord(mesh=mesh), args=(axis,))


def _infer(op: Arange) -> TensorType:
    call = Call(type=TensorType.umat_scalar(), target=op, args=())
    return TypeInferVisitor().visit(call, TypeInferContext())


def test_arange_infers_evaluates_and_costs():
    type_ = TensorType(shape=(3,), dtype=DType.i64, layout=Layout((3,)), storage=StorageKind.GMEM)
    op = Arange(type=type_, start=2, step=2)
    assert _infer(op) is type_
    check_op_case(OpCase("half_open", op, (), value=torch.tensor([2, 4, 6], dtype=torch.int64)))
    check_op_case(OpCase("arange", op, (), cost=Cost(traffic=(TrafficBytes(),))))


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (
            Arange(
                type=TensorType(shape=(), dtype=DType.i64, layout=None, storage=StorageKind.GMEM)
            ),
            "type must have rank 1",
        ),
        (
            Arange(
                type=TensorType(shape=(8,), dtype=DType.i64, layout=None, storage=StorageKind.GMEM),
                step=0,
            ),
            "step must be a positive static integer",
        ),
        (
            Arange(
                type=TensorType(shape=(8,), dtype=DType.f32, layout=None, storage=StorageKind.GMEM)
            ),
            "dtype must be i32 or i64",
        ),
    ],
)
def test_arange_rejects_unsupported_attributes(op, message):
    with pytest.raises(VerifyError, match=message):
        _infer(op)


def test_bound_mesh_coord_is_scalar() -> None:
    inferred = TypeInferVisitor().visit(_coord(), TypeInferContext(current_mesh=_COORD_MESH))
    assert inferred.shape == ()
    assert inferred.dtype == DType.i64
    assert inferred.layout == Layout(())
    assert inferred.storage is StorageKind.RMEM


def test_unbound_mesh_coord_rejected() -> None:
    with pytest.raises(ValueError, match="must be bound by the current mesh scope"):
        TypeInferVisitor().visit(_coord(), TypeInferContext())


def test_inner_mesh_coord_binds_multilevel() -> None:
    cta = Mesh((Topology("cta", 2),), Layout((2,), (1,)), ("c",))
    current = make_mesh(cta, _COORD_MESH)
    assert (
        TypeInferVisitor().visit(_coord(), TypeInferContext(current_mesh=current)) == _COORD_INDEX
    )


def test_mesh_coord_evaluation_requires_execution_scope() -> None:
    handler = eval_registry.lookup(MeshCoord)
    with pytest.raises(EvalError, match="static mesh execution scope"):
        handler(EvaluateContext())
