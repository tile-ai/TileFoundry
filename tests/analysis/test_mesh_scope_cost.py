"""Execution regions turn one unit's work into whole-scope cost."""

from __future__ import annotations

from tests.fixtures.placed.region_boundaries import RegionBoundaries
from tilefoundry import func, module
from tilefoundry.analysis import ComputeCostMetadata, analyze
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget

_TARGET = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 1), Topology("thread", 4))


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class NoScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        return x + x


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class WithScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        with Mesh(("cta",), (1,), ("tile",)) as _cta:
            with Mesh(("thread",), (4,), ("t",)) as thread:
                local = tf.reshard(x, (8, 16 @ thread.t), "rmem")
                return tf.reshard(local + local, (8, 16), "gmem")


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class UnshardedInScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        with Mesh(("cta",), (1,), ("tile",)) as _cta:
            with Mesh(("thread",), (4,), ("t",)) as _thread:
                local = tf.zeros(Tensor[(8, 16), "f32", "rmem"])
                return tf.reshard(local + local, (8, 16), "gmem")


def _cost(owner) -> tuple[int, int]:
    result = analyze(
        owner,
        owner.entry_function(),
        analysis="compute-cost",
        level="thread",
    )
    record = get_metadata(result.function, ComputeCostMetadata)
    assert record is not None
    return dict(record.flops)["f32"], dict(record.flops_per_unit)["f32"]


def test_scope_positions_turn_per_unit_cost_into_total_cost() -> None:
    assert _cost(NoScope) == (128, 128)
    assert _cost(WithScope) == (128, 32)
    assert _cost(UnshardedInScope) == (512, 128)


def test_region_boundaries_price_calls_per_position_and_values_once() -> None:
    """Inline preserves helper repetition while counting the escaped value once."""
    helper = analyze(
        RegionBoundaries,
        RegionBoundaries.lookup("helper"),
        analysis="compute-cost",
        level="thread",
    )
    helper_record = get_metadata(helper.function, ComputeCostMetadata)
    assert helper_record is not None
    helper_flops = dict(helper_record.flops)["f32"]
    assert helper_flops == 8

    result = analyze(
        RegionBoundaries,
        RegionBoundaries.entry_function(),
        analysis="compute-cost",
        level="thread",
    )
    record = get_metadata(result.function, ComputeCostMetadata)
    assert record is not None
    assert dict(record.flops)["f32"] == 40
    assert dict(record.flops_per_unit)["f32"] == 6
    binaries = [
        get_metadata(expr, ComputeCostMetadata)
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, Binary)
    ]
    binary_flops = sorted(dict(item.flops)["f32"] for item in binaries)
    assert binary_flops == [8, helper_flops * 2, 16]
