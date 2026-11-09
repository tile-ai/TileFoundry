"""Local is a zero-traffic view at every requested topology level."""

from __future__ import annotations

from pathlib import Path

import isl
import pytest

from tilefoundry import analysis, cli
from tilefoundry.dsl import *
from tilefoundry.interfaces import cost, schedule
from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import sharding

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_N = DimVar("n", 1, 128)


@pytest.mark.parametrize(("name", "shape", "dims"), (("main", (2,), None), ("symbolic", (_N,), {"n": 8})))
def test_local_view_zero_traffic(name, shape, dims) -> None:
    _LocalProgram = cli.load_authored_ir(f"{OPS / 'local_view.py'}:_LocalProgram")
    entry = next(fn for fn in _LocalProgram.functions if fn.name == name)
    local = next(
        node
        for node in core.collect_exprs(entry.body)
        if isinstance(node, expr.Call) and isinstance(node.target, sharding.Local)
    )
    assert local.type.shape == shape
    assert tuple(local.type.layout.shape) == shape
    assert isinstance(local.type.layout, Layout)
    relations = schedule.operand_relations(local.target, tuple(arg.type for arg in local.args))
    values = (local.args[0].type, local.type)
    for boundary, value in zip(relations, values, strict=True):
        relation = boundary.relation
        assert relation.dim(isl.dim_type.OUT) == len(value.shape)
        assert relation.is_empty(), "a view of the same value reads and writes no element"

    for topology_level in ("cta", "thread"):
        result = analysis.analyze(
            *analysis.check_program(_LocalProgram, entry, topology_level=topology_level, dims=dims),
            analysis=("compute-cost", "memory"),
            topology_level=topology_level,
        )
        analysed_local = next(
            node
            for node in core.collect_exprs(result.function.body)
            if isinstance(node, expr.Call) and isinstance(node.target, sharding.Local)
        )
        record = expr.get_metadata(analysed_local, analysis.ComputeCostMetadata)
        moved = expr.get_metadata(analysed_local, analysis.MemoryMetadata)
        assert result.topology_level == topology_level
        assert record is not None
        assert record.flops.kinds == ()
        assert moved.traffic.storage.kinds == ()
        assert moved.operands == (cost.TrafficBytes(), cost.TrafficBytes())
