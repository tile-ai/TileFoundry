"""Analyze a root that reaches a child Module as one kernel.

A call into a child is device work in the current invocation, so its cost lands
in the caller's totals and repeating the site or looping over it multiplies that
work rather than adding an invocation. What the fixed-dimension query has to
establish first is that the two ends share one execution context.

No GPU, no codegen, no runtime.
"""

from __future__ import annotations

import pytest

from tests.fixtures.logical.hir_composition import REFERENCE_PROGRAMS, CrossModule, Expert
from tests.fixtures.placed.moe_mega_kernel import MoEMegaKernel
from tilefoundry import func, module
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.errors import AnalysisError
from tilefoundry.analysis.metadata import ComputeCostMetadata, TrafficMetadata
from tilefoundry.analysis.walk import postorder, reachable_functions, tensor_types
from tilefoundry.dsl import ConstTensor, DimVar, Tensor, Topology, tf
from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.types.shard.layout import ComposedLayout
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.target import CudaTarget

_H200 = "nvidia.h200_sxm"
_CTA = (Topology("cta", 132),)


def _matmul_records(result) -> tuple[tuple[ComputeCostMetadata, TrafficMetadata], ...]:
    """Both halves of the record on each inlined MatMul occurrence."""
    records = []
    for expr in postorder(result.function.body):
        if not isinstance(expr, Call) or not isinstance(expr.target, MatMul):
            continue
        record = get_metadata(expr, ComputeCostMetadata)
        moved = get_metadata(expr, TrafficMetadata)
        assert record is not None
        records.append((record, moved))
    return tuple(records)


def _flops(records) -> dict[str, int]:
    total: dict[str, int] = {}
    for item in records:
        record = item[0] if isinstance(item, tuple) else item
        for name, value in record.flops:
            total[name] = total.get(name, 0) + value
    return total


def _traffic(records) -> dict[str, int]:
    total: dict[str, int] = {}
    for _record, moved in records:
        assert moved is not None, "traffic was asked of a run that did not measure it"
        for name, bytes_ in moved.whole:
            total[name] = total.get(name, 0) + bytes_.total_bytes
    return total


def test_a_repeated_call_site_counts_its_work_again() -> None:
    @module(entry="twice", target=CudaTarget(_H200), topologies=_CTA)
    class _Twice:
        mm = Expert

        @func
        def twice(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return tf.add(mm(x), mm(x))  # noqa: F821

    one_result = analyze(CrossModule, CrossModule.entry_function(), analysis="compute-cost")
    two_result = analyze(_Twice, _Twice.entry_function(), analysis="compute-cost")
    one_records = _matmul_records(one_result)
    two_records = _matmul_records(two_result)
    one = _flops(one_records)
    two = _flops(two_records)

    assert len(one_records) == 1 and two_records == one_records * 2
    assert one and two == {name: 2 * value for name, value in one.items()}


def test_a_child_call_a_loop_varies_is_counted_once_per_trip() -> None:
    @module(entry="looped", target=CudaTarget(_H200), topologies=_CTA)
    class _Looped:
        mm = Expert

        @func
        def looped(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            acc = x
            for _ in range(3):
                acc = mm(acc)  # noqa: F821
            return acc

    one_result = analyze(CrossModule, CrossModule.entry_function(), analysis="compute-cost")
    looped_result = analyze(
        _Looped, _Looped.entry_function(), analysis="compute-cost"
    )
    one_occurrence = _matmul_records(one_result)
    looped_occurrence = _matmul_records(looped_result)
    one_root = get_metadata(one_result.function, ComputeCostMetadata)
    looped_root = get_metadata(looped_result.function, ComputeCostMetadata)

    assert len(one_occurrence) == 1 and looped_occurrence == one_occurrence
    assert one_root is not None and looped_root is not None
    assert _flops((looped_root,)) == {
        name: 3 * value for name, value in _flops((one_root,)).items()
    }


def test_the_weight_traffic_of_a_fused_root_is_what_its_callees_read() -> None:
    @module(entry="direct", target=CudaTarget(_H200), topologies=_CTA)
    class _Direct:
        @func
        def direct(
            x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 8), "f32"]
        ) -> Tensor[(4, 8), "f32"]:
            return tf.matmul(x, w)

    direct_result = analyze(_Direct, _Direct.entry_function(), analysis=("compute-cost", "memory"))
    fused_result = analyze(CrossModule, CrossModule.entry_function(), analysis=("compute-cost", "memory"))
    direct = _traffic(_matmul_records(direct_result))
    fused = _traffic(_matmul_records(fused_result))

    assert direct and fused == direct


def test_a_reached_child_resolving_another_hierarchy_is_invalid() -> None:
    @module(entry="run", topologies=(Topology("warp", 4),))
    class _Warped:
        @func
        def run(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return tf.add(x, x)

    @module(entry="fused", target=CudaTarget(_H200), topologies=_CTA)
    class _Mismatch:
        warped = _Warped

        @func
        def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return warped(x)  # noqa: F821

    with pytest.raises(AnalysisError, match="different topology hierarchy") as caught:
        analyze(_Mismatch, _Mismatch.entry_function(), analysis="compute-cost")
    assert "launch" not in str(caught.value)


def test_a_child_declaring_the_caller_hierarchy_is_accepted() -> None:
    extent = DimVar("child_topology_extent", 1, 133)
    hierarchy = (Topology("cta", extent),)

    @module(entry="run", topologies=hierarchy)
    class _Declared:
        @func
        def run(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return tf.add(x, x)

    @module(entry="fused", target=CudaTarget(_H200), topologies=hierarchy)
    class _Agreeing:
        declared = _Declared

        @func
        def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return declared(x)  # noqa: F821

    result = analyze(
        _Agreeing,
        _Agreeing.entry_function(),
        analysis=("compute-cost", "memory"),
        dims={"child_topology_extent": 132},
    )
    assert set(result.metadata_types) >= {ComputeCostMetadata, TrafficMetadata}


@pytest.mark.parametrize("name,root", REFERENCE_PROGRAMS, ids=[n for n, _ in REFERENCE_PROGRAMS])
def test_each_reference_program_is_one_analysable_kernel(name, root) -> None:
    """The shared programs measure as one root each; what a call means is not restated."""
    result = analyze(root, root.entry_function(), analysis=("compute-cost", "memory"))

    assert set(result.metadata_types) >= {ComputeCostMetadata, TrafficMetadata}


def _placed_primitives(fn) -> list[tuple[str, object]]:
    """Each costed primitive result in *fn* that a sliced Mesh reached."""
    found: list[tuple[str, object]] = []
    for expr in postorder(fn.body):
        if not isinstance(expr, Call) or isinstance(expr.target, Function):
            continue
        for leaf in tensor_types(expr.type):
            if isinstance(leaf.layout, ShardLayout) and isinstance(
                leaf.layout.mesh.layout, ComposedLayout
            ):
                found.append((type(expr.target).__name__, leaf.layout.mesh.layout))
    return found


def test_each_placed_branch_keeps_its_slice_on_its_primitive_results() -> None:
    """A lexical Mesh scope places nothing; the reshard into it is what does.

    So the branch is read from what its results retained: the sub-box the slice
    selected, as its own extent and origin rather than an extent alone.
    """
    expected = {"routed_expert": ((120,), 0), "shared_expert": ((12,), 120)}
    reached = {
        fn.name: _placed_primitives(fn)
        for fn in reachable_functions(MoEMegaKernel.entry_function())
    }

    for branch, (shape, offset) in expected.items():
        placed = reached[branch]
        assert placed, branch
        assert {op for op, _ in placed} - {"Reshard"}, branch
        for _op, layout in placed:
            assert layout.outer.shape == shape
            assert layout.offset == offset


def test_the_branches_are_placed_alike_where_their_results_meet() -> None:
    """The join is only meaningful because each branch reshards back first.

    A slice is what one branch runs on, not what its result is handed over as,
    so the two values the join consumes carry the whole topology and carry it
    identically.
    """
    joined = MoEMegaKernel.entry_function().body
    branches = [
        arg for arg in joined.args if isinstance(arg, Call) and isinstance(arg.target, Function)
    ]

    assert len(branches) == 2
    meshes = []
    for branch in branches:
        (leaf,) = tensor_types(branch.type)
        assert isinstance(leaf.layout, ShardLayout)
        assert not isinstance(leaf.layout.mesh.layout, ComposedLayout)
        assert leaf.layout.mesh.layout.shape == (132,)
        meshes.append(leaf.layout.mesh)
    assert meshes[0] == meshes[1]
