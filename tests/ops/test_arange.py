"""Arange's typed coordinate contract."""

from __future__ import annotations

import pytest
import torch

from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.ops.cost_utils import CostCase, run_cost_case
from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.registry import eval_registry
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.specialize import residual_dims, specialize_concretely
from tilefoundry.ir.hir.tensor.arange import Arange
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import ceildiv
from tilefoundry.ir.types.dim_isl import normalize_dim
from tilefoundry.ir.types.shard import Layout, Mesh, Topology, composed
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.contexts import TrafficBytes, TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

_N = DimVar("arange_n", 1, 17)

_COORD_MESH = Mesh(
    (Topology("thread", 4),), Layout((4,), (1,)), ("t",)
)
_COORD_INDEX = TensorType(
    shape=(), dtype=DType.i64, layout=None, storage=StorageKind.RMEM
)


def _coord(mesh: Mesh = _COORD_MESH) -> Call:
    axis = Constant(value=0, type=_COORD_INDEX)
    return Call(type=_COORD_INDEX, target=MeshCoord(mesh=mesh), args=(axis,))


def _infer(op: Arange) -> TensorType:
    call = Call(type=TensorType.umat_scalar(), target=op, args=())
    return TypeInferVisitor().visit(call, TypeInferContext())


@func
def _symbolic_arange(x: Tensor[(_N,), "f32"]):
    return tf.arange(Tensor[(ceildiv(_N - 1, 2),), "i32"], start=1, step=2)


def test_arange_static_type_evaluation_and_cost():
    type_ = TensorType(shape=(3,), dtype=DType.i64, layout=None, storage=StorageKind.GMEM)
    op = Arange(type=type_, start=2, step=2)

    assert _infer(op) is type_
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

    actual = evaluate(_symbolic_arange, torch.zeros(8))
    torch.testing.assert_close(actual, torch.tensor([1, 3, 5, 7], dtype=torch.int32))

    concrete = specialize_concretely(_symbolic_arange, {"arange_n": 8})
    assert residual_dims(concrete) == ()
    assert isinstance(concrete.body.target, Arange)
    assert concrete.body.target.type.shape == (4,)


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
                type=TensorType(
                    shape=(8,),
                    dtype=DType.i64,
                    layout=None,
                    storage=StorageKind.GMEM,
                ),
                step=0,
            ),
            "step must be a positive static integer",
        ),
        (
            Arange(
                type=TensorType(
                    shape=(8,),
                    dtype=DType.f32,
                    layout=None,
                    storage=StorageKind.GMEM,
                )
            ),
            "dtype must be i32 or i64",
        ),
    ],
)
def test_arange_rejects_unsupported_attributes(op, message):
    with pytest.raises(VerifyError, match=message):
        _infer(op)


def test_a_bound_mesh_coordinate_is_one_number_without_placement() -> None:
    inferred = TypeInferVisitor().visit(
        _coord(), TypeInferContext(current_mesh=_COORD_MESH)
    )
    assert inferred.shape == ()
    assert inferred.dtype == DType.i64
    assert inferred.layout is None
    assert inferred.storage is StorageKind.RMEM


def test_an_unbound_mesh_coordinate_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be bound by the current mesh scope"):
        TypeInferVisitor().visit(_coord(), TypeInferContext())


def test_an_inner_mesh_coordinate_is_bound_by_a_multilevel_scope() -> None:
    cta = Mesh((Topology("cta", 2),), Layout((2,), (1,)), ("c",))
    current = composed((cta, _COORD_MESH))
    assert TypeInferVisitor().visit(
        _coord(), TypeInferContext(current_mesh=current)
    ) == _COORD_INDEX


def test_mesh_coordinate_evaluation_is_explicitly_unmodelled() -> None:
    handler = eval_registry.lookup(MeshCoord)
    with pytest.raises(EvalError, match="one mesh participant"):
        handler(None)
