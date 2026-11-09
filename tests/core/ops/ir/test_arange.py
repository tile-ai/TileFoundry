"""Arange's typed coordinate contract."""

from __future__ import annotations

import pytest
import torch

from tests.core.ops.ir.op_case import Cost, OpCase, check_op_case
from tilefoundry import core
from tilefoundry.dsl import *
from tilefoundry.interfaces import cost, evaluator, typeinfer
from tilefoundry.ir import dim, expr, types
from tilefoundry.ir.hir import tensor

_COORD_MESH = types.Mesh((Topology("thread", 4),), types.Layout((4,), (1,)), ("t",))
_COORD_INDEX = TensorType.from_scalar(DType.i64, storage=StorageKind.RMEM)


def _coord(mesh: types.Mesh = _COORD_MESH) -> expr.Call:
    axis = expr.Constant(value=0, type=_COORD_INDEX)
    return expr.Call(type=_COORD_INDEX, target=dim.MeshCoord(mesh=mesh), args=(axis,))


def _infer(op: tensor.Arange) -> TensorType:
    call = expr.Call(type=TensorType.from_umat_scalar(), target=op, args=())
    return typeinfer.TypeInferVisitor().visit(call, typeinfer.TypeInferContext())


def test_arange_infers_evaluates_and_costs():
    type_ = TensorType(shape=(3,), dtype=DType.i64, layout=types.Layout((3,)), storage=StorageKind.GMEM)
    op = tensor.Arange(type=type_, start=2, step=2)
    assert _infer(op) is type_
    check_op_case(OpCase("half_open", op, (), value=torch.tensor([2, 4, 6], dtype=torch.int64)))
    check_op_case(OpCase("arange", op, (), cost=Cost(traffic=(cost.TrafficBytes(),))))


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (
            tensor.Arange(
                type=TensorType(shape=(), dtype=DType.i64, layout=None, storage=StorageKind.GMEM)
            ),
            "type must have rank 1",
        ),
        (
            tensor.Arange(
                type=TensorType(shape=(8,), dtype=DType.i64, layout=None, storage=StorageKind.GMEM),
                step=0,
            ),
            "step must be a positive static integer",
        ),
        (
            tensor.Arange(
                type=TensorType(shape=(8,), dtype=DType.f32, layout=None, storage=StorageKind.GMEM)
            ),
            "dtype must be i32 or i64",
        ),
    ],
)
def test_arange_rejects_unsupported_attributes(op, message):
    with pytest.raises(expr.VerifyError, match=message):
        _infer(op)


def test_bound_mesh_coord_is_scalar() -> None:
    inferred = typeinfer.TypeInferVisitor().visit(_coord(), typeinfer.TypeInferContext(current_mesh=_COORD_MESH))
    assert inferred.shape == ()
    assert inferred.dtype == DType.i64
    assert inferred.layout == types.Layout(())
    assert inferred.storage is StorageKind.RMEM


def test_unbound_mesh_coord_rejected() -> None:
    with pytest.raises(ValueError, match="must be bound by the current mesh scope"):
        typeinfer.TypeInferVisitor().visit(_coord(), typeinfer.TypeInferContext())


def test_inner_mesh_coord_binds_multilevel() -> None:
    cta = types.Mesh((Topology("cta", 2),), types.Layout((2,), (1,)), ("c",))
    current = types.Mesh.from_meshes(cta, _COORD_MESH)
    assert (
        typeinfer.TypeInferVisitor().visit(_coord(), typeinfer.TypeInferContext(current_mesh=current)) == _COORD_INDEX
    )


def test_mesh_coord_evaluates_one_value_per_unit() -> None:
    handler = evaluator.registry.get(dim.MeshCoord, None)
    axis = evaluator.TensorValue(core.ShardTensor(TensorType.from_scalar(DType.i64), torch.tensor(0)))
    result = handler(evaluator.EvaluateContext(op=dim.MeshCoord(mesh=_COORD_MESH), args=(axis,), result_type=_COORD_INDEX))
    assert [int(piece) for _, piece in result.tensor.tensor] == [0, 1, 2, 3]
