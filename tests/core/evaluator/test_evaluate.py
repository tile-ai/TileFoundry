"""Core scalar, call, and loop evaluation semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.core._corpus import Case, case, check_case, special
from tilefoundry.dsl import *
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.hir.tensor.shape_of import ShapeOf
from tilefoundry.ir.types import make_shard_tensor_type

CORE = Path(__file__).parents[1] / "fixtures" / "evaluator" / "core_semantics.py"


class CoreEvalCase(Case):
    """A core-semantics program evaluated against its torch reference."""


@case
class PythonScalarConstantOperand(CoreEvalCase):
    path: Path = CORE
    name: str | None = "AddScalar"

    def guard(self, state, problems):
        add_scalar = state.entry_function()
        x = torch.randn(4)
        assert torch.allclose(evaluate(add_scalar, x), x + 2.0)


@case
class CallBindsCalleeParams(CoreEvalCase):
    path: Path = CORE
    name: str | None = "CallerCallee"

    def guard(self, state, problems):
        caller = state.entry_function()
        a, b = torch.randn(4), torch.randn(4)
        assert torch.allclose(evaluate(caller, a, b), (a + b) * b)


def test_equal_params_keep_distinct_bindings():
    tensor_type = TensorType((1,), DType.f32, layout=None, storage="gmem")
    first = Var(name="same", type=tensor_type)
    second = Var(name="same", type=tensor_type)
    function = Function.build(
        name="same_named_params",
        params=(first, second),
        body=first,
        return_type=tensor_type,
    )

    result = evaluate(function, torch.tensor([1.0]), torch.tensor([2.0]))

    assert torch.equal(result, torch.tensor([1.0]))


@case
class SingleCarryAccumulator(CoreEvalCase):
    """Carry init comes from the IR's args (the param ``a``), looped 3×."""

    path: Path = CORE
    name: str | None = "CarrySum"

    def guard(self, state, problems):
        carry_sum = state.entry_function()
        a, b = torch.randn(4), torch.randn(4)
        assert torch.allclose(evaluate(carry_sum, a, b), a + 3 * b)


@case
class MultiCarryAccumulator(CoreEvalCase):
    """Two carries projected through TupleGetItem (a TupleValue)."""

    path: Path = CORE
    name: str | None = "CarryTwo"

    def guard(self, state, problems):
        carry_two = state.entry_function()
        a, b = torch.randn(4), torch.randn(4)
        out = evaluate(carry_two, a, b)
        assert torch.allclose(out, (a + 2 * b) + (b + 2 * a))


CASES = special(CoreEvalCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_core_eval_case(case: CoreEvalCase) -> None:
    check_case(case)


def _mesh_program(body, params, mesh):
    region = MeshRegion(mesh, body, params, params, type=body.type)
    return Function.build(name="mesh_reference", params=params, body=region, return_type=body.type)


def _coordinate(mesh, axis=0):
    held = TensorType.scalar(DType.i64, storage=StorageKind.RMEM)
    return Call(MeshCoord(mesh=mesh), (Constant(axis, type=held),), type=held)


@pytest.mark.parametrize("participants", (1, 2, 4))
def test_mesh_coordinates_survive_repeated_calls_and_loop_carries(participants):
    mesh = Mesh((Topology("gpu", participants),), Layout((participants,), (1,)), ("dp",))
    held = make_shard_tensor_type((8,), mesh=mesh, attrs=(Split(0),), dtype=DType.f32)
    x, p = Var("x", type=held), Var("p", type=held)
    rank = _coordinate(mesh)
    hundred = Constant(100.0, type=TensorType.scalar(DType.f32))
    offset = Call(Binary(kind=BinaryKind.MUL), (rank, hundred), type=hundred.type)
    added = Call(Binary(kind=BinaryKind.ADD), (p, offset), type=held)
    callee = Function.build(name="rank_add", params=(p,), body=added, return_type=held)
    first = Call(callee, (x,), type=held)
    second = Call(callee, (first,), type=held)
    phi = Var("carried", type=held)
    iv = Var("i", type=TensorType.scalar(DType.i64))
    updated = Call(callee, (phi,), type=held)
    loop = LoopRegion(iv, (phi,), (second,), updated, (updated,), 3, 1, type=held)
    actual = evaluate(_mesh_program(loop, (x,), mesh), torch.arange(8, dtype=torch.float32))
    expected = torch.cat([
        chunk + 500 * rank
        for rank, chunk in enumerate(torch.arange(8, dtype=torch.float32).chunk(participants))
    ])
    assert torch.equal(actual, expected)


def test_local_exposes_local_shape_without_a_second_slice():
    mesh = Mesh((Topology("gpu", 2),), Layout((2,), (1,)), ("dp",))
    held = make_shard_tensor_type((8, 4), mesh=mesh, attrs=(Split(1),), dtype=DType.f32)
    x = Var("x", type=held)
    local = Call(Local(), (x,), type=TensorType((8, 2), DType.f32, Layout((8, 2)), StorageKind.GMEM))
    shape = Call(ShapeOf(), (local,), type=TensorType((2,), DType.i64, Layout((2,)), StorageKind.GMEM))
    assert torch.equal(
        evaluate(_mesh_program(shape, (x,), mesh), torch.arange(32).reshape(8, 4)),
        torch.tensor([8, 2]),
    )


def test_plain_output_does_not_silently_choose_one_participant():
    mesh = Mesh((Topology("gpu", 2),), Layout((2,), (1,)), ("dp",))
    with pytest.raises(EvalError, match="disagree on a Broadcast/plain output"):
        evaluate(_mesh_program(_coordinate(mesh), (), mesh))
