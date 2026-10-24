"""CacheUpdate's value oracle, runtime bounds and Partial cache contract."""

from __future__ import annotations

from tests.core.ops.ir.test_cache_update import (
    _GLOBAL_TRAFFIC,
    _NEW_SHAPE,
    _WINDOW_BYTES,
)
from tilefoundry.analysis import ComputeCostMetadata, MemoryMetadata
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.visitor_registry.contexts import CostContext, TrafficBytes
from tilefoundry.visitor_registry.visitors import CostEvaluator
from tilefoundry_nvidia import CudaTarget

_CACHE_SHAPE = (2, 64, 4, 8)


_CTA = Topology("cta", 2)
_CTA_MESH = Mesh((_CTA,), Layout((2,), (1,)), ("g",))


@module(
    entry="append",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(_CTA,),
)
class _KVCacheAppend:
    @func
    def append(
        cache: Tensor[_CACHE_SHAPE, "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        s: Tensor[(1,), "i32"],
        new: Tensor[_NEW_SHAPE, "bf16"],
    ):
        with Mesh(("cta",), layout=(2,), names=("tile",)) as cta:
            local_cache = tf.reshard(cache, (2 @ cta.tile, 64, 4, 8), "gmem")
            local_new = tf.reshard(new, (2 @ cta.tile, 4, 4, 8), "gmem")
            return tf.cache_update(local_cache, cur_pos, s, local_new)


def test_cache_update_function_analyzes_program_and_cta_cost() -> None:
    """One crossing is the first legal window, in both windows of the question.

    How many rows this writes is a runtime value, so one occurrence is the first
    legal binding of it: one row. How many such crossings the program performs
    is the footprint family's question, and raising a single crossing to cover
    them would be answering it here.
    """
    entry = _KVCacheAppend.entry_function()
    update = next(
        expr
        for expr in collect_exprs(entry.body)
        if isinstance(expr, Call) and isinstance(expr.target, CacheUpdate)
    )
    selected_types = {id(arg): arg.type for arg in update.args}
    program_ctx = CostContext(selected_types=selected_types, selected_output_type=update.type)
    cta_ctx = CostContext(
        selected_types=selected_types,
        selected_output_type=update.type,
        topology_level="cta",
        topologies=(_CTA,),
    )
    assert cta_ctx.local_type_of(update.args[3]).shape == (1, 4, 4, 8)
    assert CostEvaluator().visit_Call(update, program_ctx).traffic == _GLOBAL_TRAFFIC
    assert CostEvaluator().visit_Call(update, cta_ctx).traffic == (
        TrafficBytes(),
        TrafficBytes(read=4),
        TrafficBytes(read=4),
        TrafficBytes(read=_WINDOW_BYTES // 2),
        TrafficBytes(write=_WINDOW_BYTES // 2),
    )

    result = analyze(
        *check_program(_KVCacheAppend, entry, topology_level="cta"),
        analysis=("compute-cost", "memory"),
        topology_level="cta",
    )
    analysed_update = next(
        expr
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, CacheUpdate)
    )
    record = get_metadata(analysed_update, ComputeCostMetadata)
    moved = get_metadata(analysed_update, MemoryMetadata)
    assert result.topology_level == "cta"
    assert record is not None
    assert record.flops.kinds == ()
    row_bytes = _WINDOW_BYTES // 4
    assert moved.operands == (
        TrafficBytes(),
        TrafficBytes(read=4),
        TrafficBytes(read=4),
        TrafficBytes(read=row_bytes),
        TrafficBytes(write=row_bytes),
    )
