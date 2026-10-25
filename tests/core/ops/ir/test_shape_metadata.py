"""HIR shape metadata reads concrete runtime extents without device traffic."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import isl
import pytest
import torch

from tests.core._corpus import Case, case, check_case, special
from tests.core.ops.ir.op_case import Cost, OpCase, check_op_case
from tilefoundry.dsl import *
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.schedule import operand_relations
from tilefoundry.ir.hir.tensor.full_like import FullLike
from tilefoundry.ir.hir.tensor.rank import Rank
from tilefoundry.ir.hir.tensor.shape_of import ShapeOf
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.types.layout import Layout
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.contexts import TrafficBytes, TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor

_S = DimVar("runtime_shape", 1, 8)
SHAPE_METADATA = Path(__file__).parents[2] / "fixtures" / "ops" / "shape_metadata.py"


def _idle(arity: int) -> tuple[TrafficBytes, ...]:
    return tuple((TrafficBytes() for _ in range(arity + 1)))


COST_CASES = [
    OpCase("rank", Rank(), (make_tensor_type((_S, 4)),), cost=Cost(traffic=_idle(1))),
    OpCase("shape_of", ShapeOf(), (make_tensor_type((_S, 4)),), cost=Cost(traffic=_idle(1))),
    OpCase(
        "full_like",
        FullLike(value=0.0),
        (make_tensor_type((4, 8), DType.f32),),
        cost=Cost(traffic=(TrafficBytes(), TrafficBytes(write=4 * 8 * 4))),
    ),
]


@pytest.mark.parametrize("case", COST_CASES, ids=lambda case: case.name)
def test_shape_metadata_cost(case):
    """Metadata reads no element and moves nothing; a template's elements are not read.

    Each boundary is a relation at the rank of the value it describes: the
    result's own rank, not the operand's, on the output side. Every input is
    reached nowhere; a metadata result is reached nowhere, and a filled one at
    every coordinate it has, once.
    """
    check_op_case(case)
    args = tuple((Var(type=type_, name=f"x{i}") for i, type_ in enumerate(case.inputs)))
    call = Call(type=case.inputs[0], target=case.op, args=args)
    call = replace(call, type=TypeInferVisitor().visit(call, TypeInferContext()))
    relations = operand_relations(case.op, tuple(case.inputs))
    values = (*case.inputs, call.type)
    for boundary, value in zip(relations, values, strict=True):
        assert boundary.relation.dim(isl.dim_type.OUT) == len(value.shape)
    *inputs, written = (boundary.relation for boundary in relations)
    assert all((relation.is_empty() for relation in inputs))
    if isinstance(case.op, FullLike):
        assert written.is_equal(shape_to_isl_set(tuple(call.type.shape), {}).identity())
    else:
        assert written.is_empty()


class ShapeMetadataCase(Case):
    """A shape-metadata program evaluated on a concrete runtime tensor."""


@case
class RuntimeShapeAndHostTypes(ShapeMetadataCase):
    path: Path = SHAPE_METADATA
    name: str | None = "ShapeMetadata"

    def guard(self, state, problems):
        _shape_metadata = state.entry_function()
        actual_shape, actual_rank = evaluate(_shape_metadata, torch.zeros(3, 4))
        torch.testing.assert_close(actual_shape, torch.tensor([3, 4], dtype=torch.int64))
        torch.testing.assert_close(actual_rank, torch.tensor(2, dtype=torch.int64))
        calls = [expr for expr in collect_exprs(_shape_metadata.body) if isinstance(expr, Call)]
        metadata_calls = [call for call in calls if isinstance(call.target, (Rank, ShapeOf))]
        assert metadata_calls
        assert all((call.type.storage is StorageKind.UMAT for call in metadata_calls))
        assert all((call.type.layout == Layout(call.type.shape) for call in metadata_calls))


@case
class ShapeElementPathHasTheCanonicalType(ShapeMetadataCase):
    path: Path = SHAPE_METADATA
    name: str | None = "ShapeElementPaths"

    def guard(self, state, problems):
        _shape_element_paths = state.entry_function()
        assert _shape_element_paths.body.type == TensorType.umat_scalar()
        actual = evaluate(_shape_element_paths, torch.zeros(3, 4))
        torch.testing.assert_close(actual, torch.tensor(4, dtype=torch.int64))


CASES = special(ShapeMetadataCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_shape_metadata_case(case: ShapeMetadataCase) -> None:
    check_case(case)
