"""Asking which unit this is: one node, and what every walk makes of it."""

from __future__ import annotations

import pytest

from tilefoundry.evaluator.registry import eval_registry
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Constant
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.mesh import Mesh, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.contexts import CostContext, TypeInferContext
from tilefoundry.visitor_registry.visitors import CostEvaluator, TypeInferVisitor

_MESH = Mesh((Topology("thread", 4),), Layout((4,), (1,)), ("t",))
_INDEX = TensorType(shape=(), dtype=DType.i64, layout=None, storage=StorageKind.RMEM)


def _coord(mesh: Mesh = _MESH) -> Call:
    axis = Constant(value=0, type=_INDEX)
    return Call(type=_INDEX, target=MeshCoord(mesh=mesh), args=(axis,))


def test_a_coordinate_is_one_number_and_carries_no_placement() -> None:
    """It says which unit this is, which is not a piece of anybody's data.

    A ShardLayout would claim the answer is spread over the participants, when
    each of them holds the whole of its own.
    """
    inferred = TypeInferVisitor().visit(_coord(), TypeInferContext())

    assert inferred.shape == ()
    assert inferred.dtype == DType.i64
    assert inferred.layout is None
    assert inferred.storage is StorageKind.RMEM


def test_asking_which_unit_this_is_costs_nothing() -> None:
    """The machine already knows; reporting it is not work anyone pays for."""
    cost = CostEvaluator().visit_Call(_coord(), CostContext())

    assert cost.flops == {}
    assert all(item.read == 0 and item.write == 0 for item in cost.traffic)


def test_the_interpreter_says_it_runs_one_participant_rather_than_guessing() -> None:
    """Evaluation runs a single participant, so there is no unit to report.

    Answering zero would silently pick one, and a program that branches on the
    coordinate would then be compared against a run that never took the branch.
    """
    handler = eval_registry.lookup(MeshCoord)
    with pytest.raises(EvalError, match="one mesh participant"):
        handler(None)
