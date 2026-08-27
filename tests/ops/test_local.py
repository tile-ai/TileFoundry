"""Local is a zero-traffic view at every requested topology level."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.analysis import ComputeCostMetadata, TrafficMetadata
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.walk import collect_exprs
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.sharding.local import Local
from tilefoundry.ir.types.shard import Layout
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.contexts import TrafficBytes


@module(
    entry="main",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2), Topology("thread", 4)),
)
class _LocalProgram:
    @func
    def main(source: Tensor[(8,), "f32"]):
        with Mesh(("thread",), layout=(4,), names=("lane",)) as thread:
            sharded = tf.reshard(source, (8 @ thread.lane,), "rmem")
            return tf.local(sharded)


def test_local_analyzes_as_a_zero_traffic_topology_view() -> None:
    entry = _LocalProgram.entry_function()
    local = next(
        expr
        for expr in collect_exprs(entry.body)
        if isinstance(expr, Call) and isinstance(expr.target, Local)
    )
    assert local.type.shape == (2,)
    assert isinstance(local.type.layout, Layout)

    for level in ("cta", "thread"):
        result = analyze(_LocalProgram, entry, analysis=("compute-cost", "memory"), level=level)
        analysed_local = next(
            expr
            for expr in collect_exprs(result.function.body)
            if isinstance(expr, Call) and isinstance(expr.target, Local)
        )
        record = get_metadata(analysed_local, ComputeCostMetadata)
        moved = get_metadata(analysed_local, TrafficMetadata)
        assert result.level == level
        assert record is not None
        assert record.flops == record.flops_per_unit == ()
        assert moved.whole == ()
        assert moved.operands == (TrafficBytes(), TrafficBytes())
