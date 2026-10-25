"""Local is a zero-traffic view at every requested topology level."""

from __future__ import annotations

from pathlib import Path

import isl
import pytest

from tilefoundry.analysis import ComputeCostMetadata, MemoryMetadata
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.schedule import operand_relations
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.contexts import TrafficBytes

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_N = DimVar("n", 1, 128)


@pytest.mark.parametrize(("name", "shape", "dims"), (("main", (2,), None), ("symbolic", (_N,), {"n": 8})))
def test_local_analyzes_as_a_zero_traffic_topology_view(name, shape, dims) -> None:
    _LocalProgram = load_authored_ir(f"{OPS / 'local_view.py'}:_LocalProgram")
    entry = next(fn for fn in _LocalProgram.functions if fn.name == name)
    local = next(
        expr
        for expr in collect_exprs(entry.body)
        if isinstance(expr, Call) and isinstance(expr.target, Local)
    )
    assert local.type.shape == shape
    assert tuple(local.type.layout.shape) == shape
    assert isinstance(local.type.layout, Layout)
    relations = operand_relations(local.target, tuple(arg.type for arg in local.args))
    values = (local.args[0].type, local.type)
    for boundary, value in zip(relations, values, strict=True):
        relation = boundary.relation
        assert relation.dim(isl.dim_type.OUT) == len(value.shape)
        assert relation.is_empty(), "a view of the same value reads and writes no element"

    for topology_level in ("cta", "thread"):
        result = analyze(
            *check_program(_LocalProgram, entry, topology_level=topology_level, dims=dims),
            analysis=("compute-cost", "memory"),
            topology_level=topology_level,
        )
        analysed_local = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, Local)
        )
        record = get_metadata(analysed_local, ComputeCostMetadata)
        moved = get_metadata(analysed_local, MemoryMetadata)
        assert result.topology_level == topology_level
        assert record is not None
        assert record.flops.kinds == ()
        assert moved.traffic.storage.kinds == ()
        assert moved.operands == (TrafficBytes(), TrafficBytes())
