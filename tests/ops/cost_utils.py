"""Shared cost-evaluator entry for ops tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Topology
from tilefoundry.visitor_registry.contexts import CostContext, TrafficBytes, TypeInferContext
from tilefoundry.visitor_registry.visitors import CostEvaluator, TypeInferVisitor


@dataclass(frozen=True)
class CostCase:
    """One declarative cost case: ``op`` over ``inputs`` with expected work."""

    name: str
    op: object
    inputs: tuple[TensorType, ...]
    flops: Mapping[DType, int] = field(default_factory=dict)
    service: Mapping[str, int] = field(default_factory=dict)
    traffic: tuple[TrafficBytes, ...] = ()
    level: str | None = None
    topologies: tuple[Topology, ...] = ()


def run_cost_case(case: CostCase) -> None:
    """Infer and evaluate one Call's cost, including every traffic operand."""
    args = tuple(Var(type=type_, name=f"x{i}") for i, type_ in enumerate(case.inputs))
    placeholder = case.inputs[0] if case.inputs else TensorType.umat_scalar()
    call = Call(type=placeholder, target=case.op, args=args)
    result_type = TypeInferVisitor().visit(call, TypeInferContext())
    call = replace(call, type=result_type)
    selected_types = {id(arg): type_ for arg, type_ in zip(args, case.inputs)}
    ctx = CostContext(
        selected_types=selected_types,
        selected_output_type=result_type,
        level=case.level,
        topologies=case.topologies,
    )

    cost = CostEvaluator().visit_Call(call, ctx)

    assert cost.flops == case.flops
    assert cost.service == case.service
    assert cost.traffic == case.traffic
