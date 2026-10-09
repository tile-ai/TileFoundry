"""HIR shape metadata reads concrete runtime extents without device traffic."""

from __future__ import annotations

from dataclasses import replace

import isl
import pytest
import torch

from tests.ops.ir.cost_utils import CostCase, run_cost_case
from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.tensor.full_like import FullLike
from tilefoundry.ir.hir.tensor.rank import Rank
from tilefoundry.ir.hir.tensor.shape_of import ShapeOf
from tilefoundry.ir.isl_interop import shape_to_isl_set
from tilefoundry.ir.types import DType, TensorType, make_tensor_type
from tilefoundry.ir.types.layout import EMPTY_LAYOUT
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.access_relation import relations_of
from tilefoundry.visitor_registry.contexts import TrafficBytes, TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor

_S = DimVar("runtime_shape", 1, 8)


@func
def _shape_metadata(x: Tensor[(_S, 4), "f32"]):
    return tf.shape_of(x), tf.rank(x)


def _idle(arity: int) -> tuple[TrafficBytes, ...]:
    return tuple(TrafficBytes() for _ in range(arity + 1))


COST_CASES = [
    CostCase("rank", Rank(), (make_tensor_type((_S, 4)),), traffic=_idle(1)),
    CostCase("shape_of", ShapeOf(), (make_tensor_type((_S, 4)),), traffic=_idle(1)),
    CostCase(
        "full_like",
        FullLike(value=0.0),
        (make_tensor_type((4, 8), DType.f32),),
        traffic=(TrafficBytes(), TrafficBytes(write=4 * 8 * 4)),
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
    run_cost_case(case)
    args = tuple(Var(type=type_, name=f"x{i}") for i, type_ in enumerate(case.inputs))
    call = Call(type=case.inputs[0], target=case.op, args=args)
    call = replace(call, type=TypeInferVisitor().visit(call, TypeInferContext()))
    relations = relations_of(call, TypeInferContext(memo={id(a): (a, a.type) for a in args}))
    values = (*case.inputs, call.type)
    for boundary, value in zip(relations, values, strict=True):
        assert boundary.relation.dim(isl.dim_type.OUT) == len(value.shape)
    *inputs, written = (boundary.relation for boundary in relations)
    assert all(relation.is_empty() for relation in inputs)
    if isinstance(case.op, FullLike):
        assert written.is_equal(shape_to_isl_set(tuple(call.type.shape), {}).identity())
    else:
        assert written.is_empty()


def test_shape_metadata_uses_runtime_shape_and_host_types() -> None:
    actual_shape, actual_rank = evaluate(_shape_metadata, torch.zeros(3, 4))

    torch.testing.assert_close(actual_shape, torch.tensor([3, 4], dtype=torch.int64))
    torch.testing.assert_close(actual_rank, torch.tensor(2, dtype=torch.int64))
    calls = [expr for expr in collect_exprs(_shape_metadata.body) if isinstance(expr, Call)]
    metadata_calls = [call for call in calls if isinstance(call.target, (Rank, ShapeOf))]
    assert metadata_calls
    assert all(call.type.storage is StorageKind.UMAT for call in metadata_calls)
    assert all(call.type.layout == EMPTY_LAYOUT for call in metadata_calls)


@func
def _shape_element_paths(x: Tensor[(_S, 4), "f32"]):
    return tf.shape_of(x)[1]


def test_shape_element_path_has_the_canonical_type() -> None:
    assert _shape_element_paths.body.type == TensorType.umat_scalar()
    actual = evaluate(_shape_element_paths, torch.zeros(3, 4))
    torch.testing.assert_close(actual, torch.tensor(4, dtype=torch.int64))
