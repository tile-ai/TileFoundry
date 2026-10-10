"""Execution regions turn one unit's work into whole-scope cost."""

from __future__ import annotations

import pytest

from tests.fixtures.placed.region_boundaries import RegionBoundaries
from tilefoundry.analysis import (
    ComputeCostMetadata,
    RegionMemoryMetadata,
    RooflineMetadata,
    analyze,
)
from tilefoundry.analysis.metadata import shares
from tilefoundry.dsl import *
from tilefoundry.ir.core import Call, Var, VerifyError, get_metadata
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.contexts import TypeInferContext
from tilefoundry.visitor_registry.typeinfer import TypeInferVisitor

_TARGET = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 2), Topology("thread", 4))


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class NoScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        return x + x


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class WithScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        with Mesh(("cta",), (2,), ("tile",)) as _cta:
            with Mesh(("thread",), (4,), ("t",)) as thread:
                local = tf.reshard(x, (8, 16 @ thread.t), "rmem")
                return tf.reshard(local + local, ((8, 16), (16, 1), {}), "gmem")


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class UnshardedInScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        with Mesh(("cta",), (2,), ("tile",)) as _cta:
            with Mesh(("thread",), (4,), ("t",)) as _thread:
                local = tf.zeros(Tensor[(8, 16), "f32", "rmem"])
                return tf.reshard(local + local, ((8, 16), (16, 1), {}), "gmem")


def _cost(owner) -> tuple[int, int, int, int]:
    result = analyze(
        owner,
        owner.entry_function(),
        analysis=("compute-cost", "memory", "roofline"),
        topology_level="thread",
    )
    record = get_metadata(result.function, ComputeCostMetadata)
    assert record is not None
    memory = get_metadata(result.function, RegionMemoryMetadata)
    footprint_total = sum(
        held.of("gmem").total for _buffer, held in memory.footprint.buffers
    )
    assert get_metadata(result.function, RooflineMetadata).memory_ns == -(
        -(footprint_total * 1_000_000_000) // 4_800_000_000_000
    )
    return (
        shares(record.flops, record.topologies)["f32"],
        shares(record.flops, record.topologies, "thread")["f32"],
        memory.traffic.storage.of("gmem").total.total_bytes,
        footprint_total,
    )


def test_scope_positions_turn_per_unit_cost_into_total_cost() -> None:
    assert _cost(NoScope) == (128, 128, 1536, 1024)
    assert _cost(WithScope) == (256, 32, 2048, 2048)
    assert _cost(UnshardedInScope) == (1024, 128, 4096, 1024)


def test_region_boundaries_price_calls_per_position_and_values_once() -> None:
    """Inline preserves helper repetition while counting the escaped value once."""
    helper = analyze(
        RegionBoundaries,
        RegionBoundaries.lookup("helper"),
        analysis="compute-cost",
        topology_level="thread",
    )
    helper_record = get_metadata(helper.function, ComputeCostMetadata)
    assert helper_record is not None
    helper_flops = shares(helper_record.flops, helper_record.topologies)["f32"]
    assert helper_flops == 8

    result = analyze(
        RegionBoundaries,
        RegionBoundaries.entry_function(),
        analysis=("compute-cost", "memory"),
        topology_level="thread",
    )
    record = get_metadata(result.function, ComputeCostMetadata)
    assert record is not None
    assert shares(record.flops, record.topologies)["f32"] == 40
    assert shares(record.flops, record.topologies, "thread")["f32"] == 6
    moved = get_metadata(result.function, RegionMemoryMetadata)
    assert {
        level: spread.total.total_bytes for level, spread in moved.traffic.storage.kinds
    } == {"gmem": 384, "smem": 64, "rmem": 720}
    binaries = [
        get_metadata(expr, ComputeCostMetadata)
        for expr in collect_exprs(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, Binary)
    ]
    binary_flops = sorted(shares(item.flops, item.topologies)["f32"] for item in binaries)
    assert binary_flops == [8, helper_flops * 2, 16]


@pytest.mark.parametrize("kind", (MeshRegion, LoopRegion))
def test_analysis_rejects_a_region_body_that_embeds_its_raw_argument(kind) -> None:
    """The isolation invariant is checked on hand-built HIR as an analysis consumer."""
    value_type = TensorType(shape=(8,), dtype=DType.f32, layout=None, storage=StorageKind.GMEM)
    value = Var(name="value", type=value_type)
    fields = (
        {"mesh": Mesh((Topology("cta", 1),), Layout((1,), (1,)), ("cta",))}
        if kind is MeshRegion
        else {
            "induction_var": Var(name="i", type=TensorType.scalar(DType.i64)),
            "yield_values": (),
            "extent": 2,
            "step": 1,
        }
    )
    scope = kind(
        **fields,
        params=(Var(name="param", type=value_type),),
        args=(value,),
        body=value,
        type=value_type,
    )

    with pytest.raises(VerifyError, match="region is not isolated"):
        TypeInferVisitor().visit(scope, TypeInferContext())
