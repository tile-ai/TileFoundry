"""CUDA CacheUpdate analysis costs at program and CTA scope."""

from __future__ import annotations

from pathlib import Path

from tests.core.ops.ir.test_cache_update import (
    _GLOBAL_TRAFFIC,
    _WINDOW_BYTES,
)
from tilefoundry import analysis, cli
from tilefoundry.dsl import *
from tilefoundry.interfaces import cost
from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import tensor

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_CTA = Topology("cta", 2)
_CTA_MESH = Mesh((_CTA,), Layout((2,), (1,)), ("g",))


def test_cache_update_cost_per_level() -> None:
    """One crossing is the first legal window, in both windows of the question.

    How many rows this writes is a runtime value, so one occurrence is the first
    legal binding of it: one row. How many such crossings the program performs
    is the footprint family's question, and raising a single crossing to cover
    them would be answering it here.
    """
    _KVCacheAppend = cli.load_authored_ir(f"{OPS / 'kv_cache_append.py'}:_KVCacheAppend")
    entry = _KVCacheAppend.entry_function()
    update = next(
        node
        for node in core.collect_exprs(entry.body)
        if isinstance(node, expr.Call) and isinstance(node.target, tensor.CacheUpdate)
    )
    selected_types = {id(arg): arg.type for arg in update.args}
    program_ctx = cost.Context(selected_types=selected_types, selected_output_type=update.type)
    cta_ctx = cost.Context(
        selected_types=selected_types,
        selected_output_type=update.type,
        topology_level="cta",
        topologies=(_CTA,),
    )
    assert cta_ctx.local_type_of(update.args[3]).shape == (1, 4, 4, 8)
    assert cost.Visitor().visit_Call(update, program_ctx).traffic == _GLOBAL_TRAFFIC
    assert cost.Visitor().visit_Call(update, cta_ctx).traffic == (
        cost.TrafficBytes(),
        cost.TrafficBytes(read=4),
        cost.TrafficBytes(read=4),
        cost.TrafficBytes(read=_WINDOW_BYTES // 2),
        cost.TrafficBytes(write=_WINDOW_BYTES // 2),
    )

    result = analysis.analyze(
        *analysis.check_program(_KVCacheAppend, entry, topology_level="cta"),
        analysis=("compute-cost", "memory"),
        topology_level="cta",
    )
    analysed_update = next(
        node
        for node in core.collect_exprs(result.function.body)
        if isinstance(node, expr.Call) and isinstance(node.target, tensor.CacheUpdate)
    )
    record = expr.get_metadata(analysed_update, analysis.ComputeCostMetadata)
    moved = expr.get_metadata(analysed_update, analysis.MemoryMetadata)
    assert result.topology_level == "cta"
    assert record is not None
    assert record.flops.kinds == ()
    row_bytes = _WINDOW_BYTES // 4
    assert moved.operands == (
        cost.TrafficBytes(),
        cost.TrafficBytes(read=4),
        cost.TrafficBytes(read=4),
        cost.TrafficBytes(read=row_bytes),
        cost.TrafficBytes(write=row_bytes),
    )
