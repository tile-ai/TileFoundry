"""Compare hard decoder access relations and dependences to hand-written maps.

Real-model analysis proves coverage, not correctness. These tests pin row
reductions, floor-divided expansion, multi-output calls, and data-dependent
gathers at readable dimensions. Expected maps use semantic forms rather than
implementation output, so a formula round-trip cannot validate itself.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from dataclasses import MISSING, dataclass, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import get_args, get_origin

import isl
import pytest
import torch

import tilefoundry.analysis.api as analysis_api
import tilefoundry.analysis.metadata as analysis_records
import tilefoundry.cli.analyze as cli_analyze
import tilefoundry.visitor_registry.access_relation as access_relation
from tests.analysis.test_analysis_families import (
    CORPUS,
    _NoParallelLevel,
    _oversized_working_set,
)
from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.fixtures.logical.authored_constraint import AuthoredConstraint
from tests.fixtures.logical.gqa_static import static_online_attend
from tests.fixtures.placed.flash_split_k_decode import FlashSplitKDecode
from tests.fixtures.placed.gqa_decode import GqaOnline
from tests.fixtures.placed.moe_mega_kernel import MoEMegaKernel
from tests.fixtures.placed.rmsnorm import RmsnormModule
from tests.fixtures.placed.square_cuda import Model as SquareCuda
from tests.fixtures.shapes.matmul_programs import gemm_rms_norm
from tests.fixtures.shapes.scaled_modules import PairedScaledParent
from tilefoundry import func, module
from tilefoundry.analysis import (
    AnalysisError,
    LoopFootprintMetadata,
    OccurrenceProvenance,
    TileGraph,
    analyze,
    check_program,
    extract,
)
from tilefoundry.analysis.buffer_plan import BufferPlan, PlannedBuffer, build_buffer_plan
from tilefoundry.analysis.check import _call_placements
from tilefoundry.analysis.facts import ThroughputFacts
from tilefoundry.analysis.metadata import (
    BufferAllocationMetadata,
    BufferRef,
    ComputeCostMetadata,
    MemoryMetadata,
    TrafficMetadata,
)
from tilefoundry.analysis.movement import (
    _bytes_for,
    _moved_bytes,
    _stated_movement,
    call_traffic,
)
from tilefoundry.analysis.performance import analyze_performance
from tilefoundry.analysis.preflight import validate_authored
from tilefoundry.analysis.roofline import _cost_bound, analyze_roofline
from tilefoundry.analysis.walk import (
    attach,
    describe,
    detach,
    enclosing_trips,
    loop_scopes,
    postorder,
    tensor_types,
    values_of,
)
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- op names resolved dynamically
from tilefoundry.inspection.analysis_report import render_analysis
from tilefoundry.inspection.values import (
    ENTRIES,
    ENTRY,
    FIELD,
    FIELDS,
    PAIR,
    PER_UNIT,
    TRIPS,
    AdvisorySummary,
    Prose,
    comment_of,
    declared_records,
    expr_fields,
    family_of,
    render_comment,
    render_record,
)
from tilefoundry.ir.core import (
    BindingMetadata,
    Call,
    Constant,
    Op,
    SourceSpanMetadata,
    TotalAndPerUnit,
    TripInterval,
    Tuple,
    TypeInferContext,
    Var,
    binding_name,
    get_metadata,
)
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.kinds import BinaryKind, ReduceKind, UnaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.cuda.nn.mma import Mma_SM80_16x8x16, Wgmma_SM90_64x128x16
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.nn.conv2d import Conv2D
from tilefoundry.ir.hir.nn.layer_norm import LayerNorm
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.nn.relu import ReLU
from tilefoundry.ir.hir.nn.rope import RoPE
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.hir.tensor.index_add import IndexAdd
from tilefoundry.ir.hir.tensor.index_copy import IndexCopy
from tilefoundry.ir.hir.tensor.index_select import IndexSelect
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.ir.hir.tensor.reduce import Reduce
from tilefoundry.ir.hir.tensor.repeat_interleave import RepeatInterleave
from tilefoundry.ir.hir.tensor.reshape import Reshape, is_induction_var_singleton_reshape
from tilefoundry.ir.hir.tensor.slice import Slice as SliceOp
from tilefoundry.ir.hir.tensor.split import Split
from tilefoundry.ir.hir.tensor.split import Split as SplitOp
from tilefoundry.ir.hir.tensor.stack import Stack
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import (
    DType,
    TensorType,
    TupleType,
    local_type_of,
    make_shard_tensor_type,
    make_tensor_type,
    tensor_bytes,
)
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.dim_isl import dim_range
from tilefoundry.ir.types.shape_helpers import i64_const
from tilefoundry.ir.types.shard import Layout, ShardLayout, Topology, make_mesh
from tilefoundry.ir.types.shard.layout_algebra import try_c_order_strides
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial
from tilefoundry.ir.types.shard.shard_layout import Split as ShardSplit
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.schedule import ScheduleError, schedule
from tilefoundry.schedule.partition import PartitionProgramError, build_partition_program
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    access_relation_registry,
    broadcast_access,
    control_leaves,
    coordinates_of,
    identity_access,
    index_set,
    leaves_of,
    linearized_view,
    reached_elements,
    reached_leaves,
    register_access_relation,
    relation_of,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import Cost, CostContext, FunctionScope, TrafficBytes
from tilefoundry.visitor_registry.visitors import CostEvaluator, TypeInferVisitor


@module(
    entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),)
)
class _RenamesTwice:
    """A window of a window, so a displacement has to compose with one."""

    @func
    def main(source: Tensor[(32,), "f32"]):
        first = source[8:24]
        second = first[4:12]
        return add(second, second)  # noqa: F405


@module(
    entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),)
)
class _TakesAField:
    """One field of several, so a displacement has to pick the right buffer."""

    @func
    def main(source: Tensor[(32,), "f32"]):
        parts = split(source, axis=0, num_splits=2)  # noqa: F405
        second = parts[1]
        return add(second, second)  # noqa: F405


@module(
    entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),)
)
class _RenamesWhatItCannotPlace:
    """A window nobody can place, windowed again by one that could be placed."""

    @func
    def main(source: Tensor[(8, 4), "f32"], start: Tensor[(), "i64"]):
        first = source[start : start + 4, :]
        second = first[1:3, :]
        return add(second, second)  # noqa: F405


@module(
    entry="main", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 4),)
)
class _RenamesAnUnplacedWindowAtNoDistance:
    """A window nobody can place, re-indexed without moving off its front."""

    @func
    def main(source: Tensor[(8, 4), "f32"], start: Tensor[(), "i64"]):
        first = source[start : start + 4, :]
        flat = reshape(first, (16,))  # noqa: F405
        return add(flat, flat)  # noqa: F405


REPEATS = 4
B, S, H, D = 1, 5, 2, 3


HQ, HKV, HEAD_DIM, MAX_POS = 16, 8, 128, 8


def test_check_program_is_the_reusable_gate_for_public_consumers(
    monkeypatch,
) -> None:
    entry = RmsnormModule.entry_function()
    invalid = Module(
        "invalid-topology",
        (entry,),
        entry.name,
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("warp", 4),),
    )
    metadata_before = {id(expr): expr.metadata for expr in values_of(entry)}

    with pytest.raises(
        AnalysisError,
        match=r"level 'warp' with extent 4 is invalid: .*unsupported topology level 'warp'",
    ):
        check_program(invalid, entry, level="warp")
    assert {id(expr): expr.metadata for expr in values_of(entry)} == metadata_before

    monkeypatch.setattr(cli_analyze, "load_authored_ir", lambda _source: invalid)
    with pytest.raises(AnalysisError, match="unsupported topology level 'warp'"):
        cli_analyze.run_authored_analysis("unused.py:invalid", ())

    with pytest.raises(AnalysisError, match="unsupported topology level 'warp'"):
        analysis_api.analyze(invalid, entry, analysis="compute-cost")
    with pytest.raises(ScheduleError, match="unsupported topology level 'warp'"):
        schedule(invalid, entry, topology="warp")


def test_analyze_applies_authored_readiness_after_the_shared_program_check() -> None:
    module = AuthoredConstraint
    function = module.entry_function()
    metadata_before = {id(expr): expr.metadata for expr in values_of(function)}

    expanded = check_program(module, function, level="cta")
    assert expanded is not function
    assert not any(
        isinstance(expr, Call) and isinstance(expr.target, Function)
        for expr in postorder(expanded.body)
    )
    with pytest.raises(
        AnalysisError,
        match=r"authored analysis does not accept where\(\.\.\.\)",
    ):
        analysis_api.analyze(module, function, analysis="compute-cost")
    assert {id(expr): expr.metadata for expr in values_of(function)} == metadata_before


def test_check_program_keeps_loops_and_names_inlined_occurrences() -> None:
    module = static_online_attend
    authored = module.entry_function()
    authored_body = authored.body
    source_metadata = {id(expr): expr.metadata for expr in postorder(authored.body)}

    first = check_program(module, authored)
    second = check_program(module, authored)

    first_values = postorder(first.body)
    second_values = postorder(second.body)
    first_calls = [expr for expr in first_values if isinstance(expr, Call)]
    second_calls = [expr for expr in second_values if isinstance(expr, Call)]
    authored_calls = [
        expr for expr in postorder(authored.body) if isinstance(expr, Call)
    ]
    (loop,) = (expr for expr in first_values if isinstance(expr, GridRegionExpr))

    assert authored.body is authored_body
    assert {id(expr): expr.metadata for expr in postorder(authored.body)} == source_metadata
    assert loop.extent == 4096
    assert loop.induction_var.name == "i"
    assert tuple(item.name for item in loop.carried_args) == ("l", "o", "m")
    assert len(loop.yield_values) == 3
    assert len(first_calls) < 100
    source_keys = [
        (
            get_metadata(expr, SourceSpanMetadata),
            id(expr.target),
            tuple(id(arg) for arg in expr.args),
        )
        for expr in authored_calls
    ]
    assert len(source_keys) == len(set(source_keys))
    assert [binding_name(expr) for expr in first_calls] == [
        binding_name(expr) for expr in second_calls
    ]
    assert len({binding_name(expr) for expr in first_calls}) == len(first_calls)
    assert [get_metadata(expr, OccurrenceProvenance) for expr in first_calls] == [
        OccurrenceProvenance(source_call=id(source), call_path=(authored.name,))
        for source in authored_calls
    ]


def test_check_program_reuses_promoted_child_resources_and_enforces_its_budget() -> None:
    from tests.fixtures.logical.hir_composition import CrossModule  # noqa: PLC0415

    authored = CrossModule.entry_function()
    expanded = check_program(CrossModule, authored)

    assert [(param.name, param.is_const) for param in expanded.params] == [
        ("x", False),
        ("expert.w", True),
    ]
    (resource,) = (param for param in expanded.params if param.is_const)
    calls = [expr for expr in postorder(expanded.body) if isinstance(expr, Call)]
    assert all(any(arg is resource for arg in expr.args) for expr in calls)
    assert {
        get_metadata(expr, OccurrenceProvenance).call_path for expr in calls
    } == {("root", "run", "0")}
    with pytest.raises(
        AnalysisError,
        match=r"produces \d+ body nodes, exceeding the node budget 0",
    ):
        check_program(CrossModule, authored, budget=0)


def test_check_program_keeps_resources_of_two_attachments_distinct() -> None:
    expanded = check_program(PairedScaledParent, PairedScaledParent.entry_function())
    resources = {param.name: param for param in expanded.params if param.is_const}
    calls = [expr for expr in postorder(expanded.body) if isinstance(expr, Call)]

    assert expanded.params[0].name == "w" and not expanded.params[0].is_const
    assert tuple(resources) == ("left.w", "right.w")
    assert any(resources["left.w"] in call.args for call in calls)
    assert any(resources["right.w"] in call.args for call in calls)
    assert resources["left.w"] is not resources["right.w"]
def test_only_unmaterialized_loop_indices_get_the_singleton_reshape_exemption() -> None:
    """A concrete scalar index remains data-dependent under the HIR contract."""

    def make_function(storage: StorageKind) -> Function:
        x = Var(type=make_tensor_type((8,), DType.f32), name="x")
        induction_var = Var(
            type=TensorType.scalar(DType.i32, storage=storage), name="i"
        )
        index = Call(
            type=TensorType(shape=(1,), dtype=DType.i32, layout=None, storage=storage),
            target=Reshape(new_shape=(1,)),
            args=(induction_var,),
        )
        selected = Call(
            type=make_tensor_type((1,), DType.f32),
            target=IndexSelect(dim=0),
            args=(x, index),
        )
        loop = GridRegionExpr(
            type=selected.type,
            induction_var=induction_var,
            carried_args=(),
            init_args=(),
            body=selected,
            yield_values=(selected,),
            extent=1,
            step=1,
        )
        return Function.build(
            name=f"singleton_{storage.name.lower()}",
            params=(x,),
            body=loop,
            return_type=selected.type,
        )

    umat_function = make_function(StorageKind.UMAT)
    assert isinstance(umat_function.body, GridRegionExpr)
    assert isinstance(umat_function.body.body, Call)
    umat_index = umat_function.body.body.args[1]
    assert is_induction_var_singleton_reshape(umat_index)
    validate_authored((umat_function,))
    build_partition_program(
        Module("umat_singleton_partition", (umat_function,), entry=umat_function.name),
        umat_function,
    )

    rmem_function = make_function(StorageKind.RMEM)
    assert isinstance(rmem_function.body, GridRegionExpr)
    assert isinstance(rmem_function.body.body, Call)
    rmem_index = rmem_function.body.body.args[1]
    assert not is_induction_var_singleton_reshape(rmem_index)
    with pytest.raises(AnalysisError, match="unresolved layout"):
        validate_authored((rmem_function,))
    with pytest.raises(PartitionProgramError, match="storage.*RMEM"):
        build_partition_program(
            Module("rmem_singleton_partition", (rmem_function,), entry=rmem_function.name),
            rmem_function,
        )


@func
def gqa_expand(x: Tensor[(B, S, H, D), "f32"]) -> Tensor[(B, S, H * REPEATS, D), "f32"]:
    y = repeat_interleave(x, repeats=REPEATS, axis=2)  # noqa: F405
    return y


@func
def rope_gqa(
    q: Tensor[(1, 4, HQ, HEAD_DIM), "f32"],
    k: Tensor[(1, 4, HKV, HEAD_DIM), "f32"],
    cos_cache: Tensor[(MAX_POS, HEAD_DIM), "f32"],
    sin_cache: Tensor[(MAX_POS, HEAD_DIM), "f32"],
    pos_ids: Tensor[(4,), "i32"],
):
    q_rope, k_rope = rope(q, k, cos_cache, sin_cache, pos_ids)  # noqa: F405
    return q_rope, k_rope


def test_the_dependences_are_exactly_what_the_access_relations_imply() -> None:
    """A matmul feeding a normalisation: two dependences, nothing more.

    Both are derived by isl from the read/write maps alone, and both are written
    out here from what the two ops do. The matmul accumulates along k, so
    iteration k needs what k-1 wrote; the normalisation reads the whole row the
    matmul's last k-step completed, which is why the source is the k=3 slice and
    why every (i, j) writer of that row is an edge. `is_equal` and not
    `is_subset`: a relation that invented a third dependence would order work
    that need not be ordered, and only an exact comparison says so.
    """
    tg = extract(gemm_rms_norm)
    assert isinstance(tg, TileGraph)
    assert {u.name: type(u.op.target).__name__ for u in tg.units} == {
        "MM": "MatMul",
        "RN": "RMSNorm",
    }

    k_carry = isl.map("{ MM[i,j,k] -> MM[i,j,k+1] : 0<=i<2 and 0<=j<2 and 0<=k<3 }")
    mm_to_rn = isl.map("{ MM[i,j,3] -> RN[i] : 0<=i<2 and 0<=j<2 }")
    assert tg.deps.is_equal(isl.union_map("{}").union(k_carry).union(mm_to_rn))


def test_only_an_accumulated_dimension_is_serial() -> None:
    """`parallel_dims` is read off the dependences, so it says the same thing.

    A matmul is serial in its own k and parallel everywhere else. A normalisation
    reduces and is still parallel in every dimension it has -- the axis it
    reduces is not one of them -- which is the half of the rule a matmul alone
    cannot show, and the half a scheduler gets wrong by assuming that reducing
    and being serial are the same property.
    """
    tg = extract(gemm_rms_norm)

    assert tg.parallel_dims["MM"] == (True, True, False)
    assert tg.parallel_dims["RN"] == (True,)


def test_an_expanded_axis_reads_through_a_floor_division() -> None:
    """GQA's kv-head expansion, which aliases.

    GQA's kv-head expansion, which aliases: `repeats` consecutive output
    positions read one input element.

    The domain is the *output* space, so the read map has to divide axis 2 by
    `repeats` and be identity elsewhere. Written as the inequality
    `repeats*o2 <= d2 <= repeats*o2 + repeats-1`, which is what
    `o2 = floor(d2/repeats)` means, rather than as the `floor(a/b)` text the
    relation itself emits.
    """
    tg = extract(gqa_expand)

    bounds = f"0<=d0<{B} and 0<=d1<{S} and 0<=d2<{H * REPEATS} and 0<=d3<{D}"
    assert tg.domain.is_equal(isl.union_set(f"{{ RepeatInterleave[d0,d1,d2,d3] : {bounds} }}"))
    assert tg.reads.is_equal(
        isl.union_map(
            f"{{ RepeatInterleave[d0,d1,d2,d3] -> x[d0,d1,o2,d3] : {bounds} "
            f"and {REPEATS}*o2<=d2<={REPEATS}*o2+{REPEATS - 1} }}"
        )
    )
    assert tg.writes.is_equal(
        isl.union_map(f"{{ RepeatInterleave[d0,d1,d2,d3] -> y[d0,d1,d2,d3] : {bounds} }}")
    )

    assert tg.deps.is_equal(isl.union_map("{}"))


def test_one_call_producing_two_tensors_becomes_two_statements() -> None:
    """A rotation returns q and k, and under GQA they are different widths.

    The two outputs answer on different parts of the one space the call is asked
    by, so it lifts into two statements, each over its own head count, each
    writing its own output buffer (the `_{index}` suffix any multi-output op's
    outputs take, which is the name the statement takes too). The two are
    independent, so no dependence may appear between them -- an edge here would
    serialise two rotations that have nothing to say to each other.
    """
    tg = extract(rope_gqa)

    assert {u.name: type(u.op.target).__name__ for u in tg.units} == {
        "RoPE_0": "RoPE",
        "RoPE_1": "RoPE",
    }
    expected = isl.union_set("{}")
    for name, heads in (("RoPE_0", HQ), ("RoPE_1", HKV)):
        expected = expected.union(
            isl.set(
                f"{{ {name}[d0,d1,d2,d3] : 0<=d0<1 and 0<=d1<4 "
                f"and 0<=d2<{heads} and 0<=d3<{HEAD_DIM} }}"
            )
        )
    assert tg.domain.is_equal(expected)

    writes = isl.union_map("{}")
    for name, heads, buffer in (("RoPE_0", HQ, "rope_0"), ("RoPE_1", HKV, "rope_1")):
        writes = writes.union(
            isl.map(
                f"{{ {name}[d0,d1,d2,d3] -> {buffer}[d0,d1,d2,d3] : "
                f"0<=d0<1 and 0<=d1<4 and 0<=d2<{heads} and 0<=d3<{HEAD_DIM} }}"
            )
        )
    assert tg.writes.is_equal(writes)
    assert tg.deps.is_equal(isl.union_map("{}"))


def test_each_rotation_reads_its_own_value_and_not_the_other() -> None:
    """Exactly which buffers each statement reads, not merely which it includes.

    Rotating Q and rotating K are separate work sharing only the tables and the
    positions. A statement that also read the value it does not rotate would
    claim a dependence on bytes it never touches, and every subset assertion
    would still pass while it did. So this asks for the whole set: each side has
    its own value, neither has the other's, and both have what they share.
    """
    tg = extract(rope_gqa)

    read: dict[str, set[str]] = {}
    tg.reads.foreach_map(
        lambda access: read.setdefault(
            access.get_tuple_name(isl.dim_type.IN), set()
        ).add(access.get_tuple_name(isl.dim_type.OUT))
    )
    shared = {"cos_cache", "sin_cache", "pos_ids"}
    assert read == {"RoPE_0": shared | {"q"}, "RoPE_1": shared | {"k"}}


def test_a_rotation_reads_its_tables_at_the_position_and_not_at_random() -> None:
    """V1 decodes at `pos_ids == arange(seq)`, so the table selection is affine.

    That assumption is what lets a rotation be modelled at all: `cos[pos[s]]`
    degenerates to `cos[s]`, broadcast over batch and heads. Asserted on both
    branches, because the formula is the same and only the surrounding head
    extent differs -- a relation that read the tables per head, or per element of
    the wrong axis, would still produce a report.
    """
    tg = extract(rope_gqa)

    for name, heads in (("RoPE_0", HQ), ("RoPE_1", HKV)):
        bounds = f"0<=d0<1 and 0<=d1<4 and 0<=d2<{heads} and 0<=d3<{HEAD_DIM}"
        for table in ("cos_cache", "sin_cache"):
            read = isl.map(f"{{ {name}[d0,d1,d2,d3] -> {table}[d1,d3] : {bounds} }}")
            assert read.is_subset(tg.reads), f"{name}: {table} is not a seq identity"
        positions = isl.map(f"{{ {name}[d0,d1,d2,d3] -> pos_ids[d1] : {bounds} }}")
        assert positions.is_subset(tg.reads), f"{name}: pos_ids is not a seq identity"


@func
def elementwise_pair(x: Tensor[(64, 64), "f32"]) -> Tensor[(64, 64), "f32"]:
    y = sigmoid(x)  # noqa: F405
    z = exp(y)  # noqa: F405
    return z


@func
def softmax_row(x: Tensor[(2, 64), "f32"]) -> Tensor[(2, 64), "f32"]:
    y = softmax(x, axis=-1)  # noqa: F405
    return y


def test_an_op_with_no_registered_relation_has_no_fallback() -> None:
    """Which is why every op a decoder reaches has to carry one.

    ``extract`` has no fallback when the relation registry has no entry. Elementwise
    and fused row reductions therefore must register relations. ``SoftMax`` is
    pinned exactly: one statement owns a row, with the reduced axis existential
    in access maps and absent from the domain.
    """
    elementwise = extract(elementwise_pair)
    assert [type(u.op.target).__name__ for u in elementwise.units] == [
        "Sigmoid",
        "Unary",
    ]

    rows = extract(softmax_row)
    assert [u.name for u in rows.units] == ["SoftMax"]
    assert rows.domain.is_equal(isl.union_set("{ SoftMax[i] : 0 <= i < 2 }"))
    assert rows.reads.is_equal(
        isl.union_map("{ SoftMax[i] -> x[i, j] : 0 <= i < 2 and 0 <= j < 64 }")
    )
    assert rows.writes.is_equal(
        isl.union_map("{ SoftMax[i] -> y[i, j] : 0 <= i < 2 and 0 <= j < 64 }")
    )

    assert "-> y[" not in str(rows.reads)






def test_loop_scopes_choose_the_containing_loop_instead_of_the_smaller_work_set() -> None:
    scalar = TensorType.scalar(DType.i32)
    outer_iv = Var(type=scalar, name="outer")
    middle_iv = Var(type=scalar, name="middle")
    inner_iv = Var(type=scalar, name="inner")

    outer_value = Call(type=scalar, target=ReLU(), args=(outer_iv,))
    middle_values = [Call(type=scalar, target=ReLU(), args=(middle_iv,))]
    for _ in range(4):
        middle_values.append(
            Call(type=scalar, target=ReLU(), args=(middle_values[-1],))
        )

    inner = GridRegionExpr(
        type=scalar,
        induction_var=inner_iv,
        carried_args=(),
        init_args=(outer_value, middle_values[-1]),
        body=middle_values[-1],
        yield_values=(),
        extent=2,
        step=1,
    )
    middle = GridRegionExpr(
        type=scalar,
        induction_var=middle_iv,
        carried_args=(),
        init_args=(outer_value,),
        body=inner,
        yield_values=(),
        extent=2,
        step=1,
    )
    outer = GridRegionExpr(
        type=scalar,
        induction_var=outer_iv,
        carried_args=(),
        init_args=(),
        body=middle,
        yield_values=(),
        extent=2,
        step=1,
    )
    function = Function.build(
        name="nested_variance",
        params=(),
        body=outer,
        return_type=scalar,
    )

    parent, scope_of = loop_scopes(function)

    assert parent == {
        id(inner): id(middle),
        id(middle): id(outer),
        id(outer): None,
    }
    assert scope_of[id(middle_values[-1])] == id(middle)


def _bounded(rank: int, extent: int = 4) -> "AffineAccess":
    """A bounded identity, which is the shape a real handler states.

    An Op walks a space it can be counted over, so a test standing in for one
    says how far its coordinates go rather than leaving them open.
    """
    dims = ", ".join(f"d{index}" for index in range(rank)) or ""
    guards = " and ".join(f"0 <= d{index} < {extent}" for index in range(rank))
    where = f" : {guards}" if guards else ""
    return AffineAccess(isl.map(f"{{ [{dims}] -> [{dims}]{where} }}"))


def _relations(target, shape, *args) -> AccessRelations:
    """The registered relation of one op, at the black-box (whole-call) level."""
    operands = (Var(type=make_tensor_type(shape, DType.bf16), name="x"), *args)
    call = Call(type=make_tensor_type(shape, DType.bf16), target=target, args=operands)
    return relations_of(call, TypeInferContext())


_UNTYPED = "the result Type this Call is about to be given"


@dataclass
class _NothingKnowsTheResult(TypeInferContext):
    """A context that refuses to say what one Call returns.

    Type inference asks for the coordinates in order to derive that Type, so a
    relation stated in terms of it cannot be the one type inference asks: the
    question would be its own answer. Refusing here is the only way to tell a
    handler that derived the space from its inputs from one that asked for the
    result and got it because the context was willing to infer it again.
    """

    about: object = None

    def type_of(self, expr):
        if expr is self.about:
            raise AssertionError(
                "the canonical relation asked for the result Type it is being "
                "asked in order to derive"
            )
        return super().type_of(expr)

    def local_type_of(self, expr):
        if expr is self.about:
            raise AssertionError(
                "the canonical relation asked for the result's local Type, which "
                "is a projection of an answer that does not exist yet"
            )
        return super().local_type_of(expr)


def _pre_type(target, *args) -> AccessRelations:
    """One Op's relations, asked before anything knows what it returns."""
    handler = access_relation_registry.lookup(type(target))
    assert handler is not None
    call = Call(type=_UNTYPED, target=target, args=args)
    return handler(call, _NothingKnowsTheResult(about=call))


def _walked_by(relations: AccessRelations) -> "isl.set":
    """The space every boundary of one Op is asked by, compared as a set.

    Two spellings of one space are one space, so this asks isl whether they are
    equal rather than whether they read the same.
    """
    walked = None
    for boundary in (*relations.inputs, *relations.outputs):
        own = relation_of(boundary.pattern).domain()
        if walked is None:
            walked = own
            continue
        assert walked.is_equal(own), (
            f"one Op states one space: {walked} against {own}"
        )
    assert walked is not None
    return walked


def test_what_one_op_states_about_its_own_space_is_checked_before_its_type() -> None:
    """The two ways a space is wrong, and the three ways it is legitimately partial.

    Type inference has to be handed something already worth believing, and what
    is checkable without the Type is the space itself: every boundary asked by
    the same coordinates, and each one's own part of it bounded once parameters
    stand for values. Neither refuses a boundary for answering on part of the
    space, which is how real Ops are written.
    """

    def asked(name, inputs, outputs):
        target = type(name, (Op,), {})
        register_typeinfer(target)(lambda call, ctx: make_tensor_type((4,), DType.f32))
        register_access_relation(target)(
            lambda call, ctx: AccessRelations(
                inputs=tuple(
                    BoundaryRelation(pattern) for pattern in inputs
                ),
                outputs=tuple(BoundaryRelation(pattern) for pattern in outputs),
            )
        )
        return Call(
            type=make_tensor_type((4,), DType.f32),
            target=target(),
            args=(Var(type=make_tensor_type((4,), DType.f32), name="x"),),
        )

    two_ranks = asked(
        "_AsksTwoRanks",
        (AffineAccess(isl.map("{ [d0, d1] -> [d0] : 0 <= d0 < 4 and 0 <= d1 < 4 }")),),
        (_bounded(1),),
    )
    with pytest.raises(ValueError, match="one Op walks one space"):
        coordinates_of(two_ranks, TypeInferContext())

    open_ended = asked(
        "_LeavesItOpen",
        (AffineAccess(isl.map("{ [d0] -> [d0] : 0 <= d0 }")),),
        (_bounded(1),),
    )
    with pytest.raises(ValueError, match="no parameter binding makes bounded"):
        coordinates_of(open_ended, TypeInferContext())

    empty_beside_full = asked(
        "_AddressesWithoutReading",
        (AffineAccess(isl.map("{ [d0] -> [d0] : false }")),),
        (_bounded(1),),
    )
    coordinates_of(empty_beside_full, TypeInferContext())

    parametric = asked(
        "_WaitsForAValue",
        (AffineAccess(
            isl.map("[n] -> { [d0] -> [d0] : 0 <= d0 < n and 0 < n <= 4 }"),
            (("n", DimVar("n", 1, 5)),),
        ),),
        (_bounded(1),),
    )
    coordinates_of(parametric, TypeInferContext())


def test_the_partial_spaces_real_ops_state_are_accepted() -> None:
    """Three shapes that answer on part of a space, each for its own reason.

    A slice reads the numbers that place its window and none of the window, so
    that boundary answers at one point. An insert writes a window and keeps
    everything else, so those two do not meet. A grouped rotation walks one
    coordinate saying which value it is rotating, and each side answers at one
    value of it. All three are what the checks have to admit while still
    refusing an unwalkable space.
    """
    bounds = Tuple(
        type=TupleType(fields=(make_tensor_type((), DType.i64),)),
        elements=(Constant(type=make_tensor_type((), DType.i64), value=2),),
    )
    window = Call(
        type=make_tensor_type((4,), DType.f32),
        target=SliceOp(sizes=(4,), strides=(1,)),
        args=(Var(type=make_tensor_type((10,), DType.f32), name="x"), bounds),
    )
    addressed = coordinates_of(window, TypeInferContext())
    assert reached_elements(addressed.inputs[1].pattern) == 1, (
        "a slice reads the one number that places its window, not the window"
    )
    assert not relation_of(addressed.inputs[0].pattern).is_empty()

    dst = make_tensor_type((10,), DType.f32)
    inserted = Call(
        type=dst,
        target=InsertSlice(),
        args=(
            Var(type=dst, name="dst"),
            Var(type=make_tensor_type((4,), DType.f32), name="update"),
            Constant(type=make_tensor_type((), DType.i64), value=2),
        ),
    )
    placed = coordinates_of(inserted, TypeInferContext())
    kept = relation_of(placed.inputs[0].pattern).domain()
    written = relation_of(placed.outputs[0].pattern).domain()
    assert not kept.is_equal(written), "what it keeps is what the window is not"
    assert kept.intersect(written).is_empty()

    rotated = extract(rope_gqa)
    walked = sorted(str(unit.name) for unit in rotated.units)
    assert walked == ["RoPE_0", "RoPE_1"], (
        "and a rotation's two instances answer at one value of that coordinate each"
    )


def test_one_relation_answers_before_its_result_type_exists() -> None:
    """The registry that answers analysis is the one type inference has to ask.

    A shape and a shard layout are derived from where an Op reads and writes, so
    the coordinates cannot be stated in terms of what the Op returns -- that is
    the thing being derived. Every rank here is built from the inputs and the
    attributes: a product with and without batch axes, an elementwise op at
    three ranks, a broadcast, a permutation, a reduction, and the two fixed
    tiles whose space is the instruction's own count.
    """
    f32 = DType.f32
    cases = (
        ("matmul", MatMul(), (make_tensor_type((8, 4), f32), make_tensor_type((4, 2), f32))),
        (
            "batched matmul",
            MatMul(),
            (make_tensor_type((3, 8, 4), f32), make_tensor_type((3, 4, 2), f32)),
        ),
        ("unary rank 1", Unary(kind=UnaryKind.NEG), (make_tensor_type((5,), f32),)),
        ("unary rank 3", Unary(kind=UnaryKind.NEG), (make_tensor_type((2, 3, 4), f32),)),
        (
            "broadcast",
            Binary(kind=BinaryKind.ADD),
            (make_tensor_type((4, 8), f32), make_tensor_type((8,), f32)),
        ),
        ("permutation", Transpose(perm=(1, 0)), (make_tensor_type((3, 5), f32),)),
        (
            "reduction",
            Reduce(axes=(1,), keepdim=False, kind=ReduceKind.SUM),
            (make_tensor_type((2, 3, 4), f32),),
        ),
        (
            "m16n8k16 tile",
            Mma_SM80_16x8x16(dtype_a=DType.f16, dtype_b=DType.f16, dtype_acc=DType.f32),
            (
                make_tensor_type((16, 16), DType.f16),
                make_tensor_type((16, 8), DType.f16),
            ),
        ),
        (
            "m64n128k16 tile",
            Wgmma_SM90_64x128x16(
                dtype_a=DType.f16, dtype_b=DType.f16, dtype_acc=DType.f32
            ),
            (
                make_tensor_type((64, 16), DType.f16),
                make_tensor_type((16, 128), DType.f16),
            ),
        ),
    )
    for label, target, shapes in cases:
        args = tuple(
            Var(type=shape, name=f"operand{index}") for index, shape in enumerate(shapes)
        )
        relations = _pre_type(target, *args)
        assert len(relations.inputs) == len(args), label
        assert relations.outputs, label
        _walked_by(relations)


def test_the_shape_type_inference_answers_is_the_one_the_relation_reaches() -> None:
    """Two answers that must be one: what a Call's Type says, and where it writes.

    Type inference derives the result's extents from the output boundary, so a
    relation that reached other coordinates would infer another shape. Asserted
    where the two could disagree: a contraction drops the axis it sums, a
    reduction drops the axes it reduces, and a permutation reorders them.
    """
    f32 = DType.f32
    for target, shapes, expected in (
        (MatMul(), ((8, 4), (4, 2)), (8, 2)),
        (MatMul(), ((3, 8, 4), (3, 4, 2)), (3, 8, 2)),
        (Transpose(perm=(1, 0)), ((3, 5),), (5, 3)),
        (Reduce(axes=(1,), keepdim=False, kind=ReduceKind.SUM), ((2, 3, 4),), (2, 4)),
    ):
        args = tuple(
            Var(type=make_tensor_type(shape, f32), name=f"operand{index}")
            for index, shape in enumerate(shapes)
        )
        relations = _pre_type(target, *args)
        reached = relation_of(relations.outputs[0].pattern).range()
        extents = tuple(
            int(str(reached.dim_max_val(axis))) + 1
            for axis in range(reached.dim(isl.dim_type.SET))
        )
        assert extents == expected, f"{type(target).__name__} reaches {extents}"
        inferred = TypeInferVisitor(TypeInferContext()).visit(
            Call(type=_UNTYPED, target=target, args=args)
        )
        assert tuple(inferred.shape) == expected == extents, (
            f"{type(target).__name__}: inference and the relation disagree"
        )


def test_one_wrong_relation_is_wrong_for_both_of_its_readers() -> None:
    """There is one relation, so breaking it has to break everything reading it.

    A registry consulted by only one of two readers can be wrong for a whole
    release without anything failing. This replaces the product's relation with
    one that contracts the wrong axis and asks both readers: the shape type
    inference derives, and the amount the movement consumer counts. If either
    still answers as before, they are not reading the same thing.
    """
    lhs = Var(type=make_tensor_type((8, 4), DType.f32), name="lhs")
    rhs = Var(type=make_tensor_type((4, 2), DType.f32), name="rhs")
    call = Call(type=_UNTYPED, target=MatMul(), args=(lhs, rhs))

    honest = access_relation_registry.lookup(MatMul)

    def contracts_the_kept_axis(one, ctx) -> AccessRelations:
        """A product that sums the axis it should have kept."""
        relations = honest(one, ctx)
        swapped = relation_of(relations.outputs[0].pattern)
        return AccessRelations(
            inputs=relations.inputs,
            outputs=(
                dataclasses.replace(
                    relations.outputs[0],
                    pattern=AffineAccess(
                        swapped.project_out(isl.dim_type.OUT, 0, 1), ()
                    ),
                ),
            ),
        )

    shape, moved = inferred_and_moved(call)
    access_relation_registry._map[MatMul] = contracts_the_kept_axis
    try:
        broken_shape, broken_moved = inferred_and_moved(call)
    finally:
        access_relation_registry._map[MatMul] = honest
    assert broken_moved != moved, "the movement consumer did not read the relation"
    assert broken_shape != shape, "type inference did not read the same relation"
    assert inferred_and_moved(call) == (shape, moved)


def inferred_and_moved(call: Call) -> tuple:
    """What both readers say about one Call: its inferred shape, and its movement."""
    inferred = TypeInferVisitor(TypeInferContext()).visit(call)
    relations = access_relation_registry.lookup(type(call.target))(
        call, TypeInferContext()
    )
    return (
        tuple(getattr(inferred, "shape", ())),
        tuple(
            reached_elements(boundary.pattern) for boundary in relations.outputs
        ),
    )


def test_one_view_relation_answers_dependence_and_footprint() -> None:
    """Folding a view is one relation's job, so breaking it breaks both readers.

    A window's stride and start used to be rebuilt from the Op in each consumer.
    This replaces the registered Slice relation with one reading a stride of two
    and asks the polyhedral model what depends on what and the footprint family
    what a loop touches: a reader still rebuilding the arithmetic would not move.
    """
    case = next(item for item in CORPUS if item.id == "access_footprint.qkv")
    selected = case.analyze[0]
    honest = access_relation_registry.lookup(SliceOp)

    def reads_every_other_row(call, ctx) -> AccessRelations:
        """A window that takes every other coordinate of what it names."""
        relations = honest(call, ctx)
        reads = relation_of(relations.inputs[0].pattern)
        stretched = reads.apply_range(
            isl.map(
                "{ [c0, c1] -> [o0, o1] : o0 = 2c0 and o1 = c1 }"
                if reads.dim(isl.dim_type.OUT) == 2
                else "{ [c0] -> [o0] : o0 = 2c0 }"
            )
        )
        return AccessRelations(
            inputs=(
                BoundaryRelation(
                    AffineAccess(
                        stretched,
                        tuple(relations.inputs[0].pattern.parameters),
                    )
                ),
                *relations.inputs[1:],
            ),
            outputs=relations.outputs,
        )

    def measured():
        owner, entry = case.resolve(case.build(), selected.selector)
        result = analyze(
            owner, entry, analysis=("compute-cost", "memory"), dims=selected.dims
        )
        footprints = tuple(
            tuple(sorted((item.buffer, item.bytes) for item in record.footprints))
            for expr in postorder(result.function.body)
            if (record := get_metadata(expr, LoopFootprintMetadata)) is not None
        )
        owner, entry = case.resolve(case.build(), selected.selector)
        return (str(extract(entry).reads), footprints)

    before = measured()
    access_relation_registry._map[SliceOp] = reads_every_other_row
    try:
        after = measured()
    finally:
        access_relation_registry._map[SliceOp] = honest
    assert measured() == before, "the honest relation was not put back"

    assert after[0] != before[0], "the dependence did not read the view's relation"
    assert after[1] != before[1], "and neither did the loop footprint"


def test_one_relation_answers_the_whole_program_and_one_participant() -> None:
    """Both windows read the same relation, so breaking it has to break both.

    A quantity taken from the relation in one window and from a legacy evaluator
    in the other is two answers again, and the whole-program one is what a
    headline reports. This replaces a product's read with one reaching a
    quarter as far, and asks for the whole reading, the per-unit reading and the
    per-operand record.
    """
    cta = Topology("cta", 2)
    mesh = make_mesh((2,), ("c",), topology=cta)
    lhs = make_shard_tensor_type(
        (8, 4), mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f32
    )
    rhs = make_tensor_type((4, 2), DType.f32)
    call = Call(
        type=make_tensor_type((8, 2), DType.f32),
        target=MatMul(),
        args=(Var(type=lhs, name="lhs"), Var(type=rhs, name="rhs")),
    )
    honest = access_relation_registry.lookup(MatMul)

    def reads_a_quarter(one, ctx) -> AccessRelations:
        """A product that reads a quarter of the operand it was handed."""
        relations = honest(one, ctx)
        held = relation_of(relations.inputs[0].pattern)
        return AccessRelations(
            inputs=(
                BoundaryRelation(
                    AffineAccess(held.intersect_range(isl.set("{ [c0, c1] : c1 < 1 }")))
                ),
                *relations.inputs[1:],
            ),
            outputs=relations.outputs,
        )

    def measured():
        whole = CostEvaluator(CostContext())
        unit = CostEvaluator(CostContext(level="cta", topologies=(cta,)))
        moved = call_traffic(call, whole, unit)
        return (moved.whole, moved.per_unit, moved.operands)

    before = measured()
    access_relation_registry._map[MatMul] = reads_a_quarter
    try:
        after = measured()
    finally:
        access_relation_registry._map[MatMul] = honest
    assert measured() == before, "the honest relation was not put back"

    assert after[0] != before[0], "the whole-program reading did not read the relation"
    assert after[1] != before[1], "the per-participant reading did not read it either"
    assert after[2] != before[2], "and neither did the per-operand record"


def test_a_scan_depends_on_every_element_of_the_axis_it_scans() -> None:
    """One result of a scan depends on the whole axis, and the relations say so.

    `topk` and `argmax` each read every element of the axis they scan to produce
    one output. Both boundaries are functions of the coordinate the Op walks --
    the scan coordinate is one of them -- so the property is in the dependence
    between them: composing the write backwards with the read is one-to-many.
    A consumer that concluded otherwise would tile an axis a scan cannot be
    tiled along.
    """

    def depends(relations, output: int = 0):
        write = relation_of(relations.outputs[output].pattern)
        read = relation_of(relations.inputs[0].pattern)
        return write.reverse().apply_range(read)

    logits = _relations(TopK(k=8), (1, 128))
    assert len(logits.inputs) == 1
    assert len(logits.outputs) == 2
    assert not depends(logits).is_single_valued()
    assert not depends(logits, 1).is_single_valued()

    picked = _relations(ArgMax(), (1, 151936))
    assert len(picked.outputs) == 1
    assert not depends(picked).is_single_valued()


def test_a_relation_says_how_a_data_dependent_operand_is_read() -> None:
    """A table read through positions is a lookup, not an unknown.

    At this level a rotation's tables are indexed by data, so which entry the
    read lands on is not known here. Telling a lookup from an identity is the
    whole safety property: a relation that returned a function for a lookup
    would be believed, so this one is not a function, and instead reaches every
    entry the table could name. That keeps it countable rather than opaque, and
    an over-count for a table larger than the positions asking for it.
    """
    heads = make_tensor_type((1, 4, HEAD_DIM), DType.bf16)
    tables = make_tensor_type((4096, HEAD_DIM), DType.bf16)
    positions = make_tensor_type((1,), DType.i32)
    relation = _relations(
        RoPE(),
        (1, 32, HEAD_DIM),
        Var(type=heads, name="k"),
        Var(type=tables, name="cos"),
        Var(type=tables, name="sin"),
        Var(type=positions, name="pos"),
    )
    rotated = isl.set(
        f"{{ [d0, d1, d2, b] : 0 <= d0 < 1 and 0 <= d1 < 32 "
        f"and 0 <= d2 < {HEAD_DIM} and 0 <= b < 2 }}"
    )

    assert len(relation.inputs) == 5
    assert relation_of(relation.inputs[0].pattern).is_single_valued()
    assert relation_of(relation.inputs[1].pattern).is_single_valued()
    for lookup in (relation.inputs[2], relation.inputs[3]):
        read = relation_of(lookup.pattern)
        assert not read.is_single_valued(), (
            "a lookup cannot promise the entry it lands on"
        )
        assert int(str(rotated.apply(read).count_val())) == 4096 * HEAD_DIM, (
            "so it states every entry the table has, which is an answer"
        )
    asked = rotated.apply(relation_of(relation.inputs[4].pattern))
    assert int(str(asked.count_val())) == 1, "the one position that decided all of it"
    assert len(relation.outputs) == 2
    assert all(
        relation_of(item.pattern).is_single_valued() for item in relation.outputs
    )


def test_every_boundary_states_the_movement_its_op_performs() -> None:
    """Hand-counted, boundary by boundary, against what each relation reaches.

    A relation says where a value came from, and how much crossed is how many
    distinct coordinates it reaches. Every output of a scan depends on the whole
    input the scan reads once, and a matrix product's result domain says nothing
    about how big its operands were -- so each Op walks its own space, and every
    count below is worked out by hand from that.
    """

    def counted(relations) -> tuple[list[int], list[int]]:
        return (
            [reached_elements(item.pattern) for item in relations.inputs],
            [reached_elements(item.pattern) for item in relations.outputs],
        )

    scanned = _relations(TopK(k=8), (1, 128))
    assert counted(scanned) == ([128], [8, 8])

    picked = _relations(ArgMax(), (1, 128))
    assert counted(picked) == ([128], [1])

    product = _relations(
        MatMul(),
        (2, 3),
        Var(type=make_tensor_type((3, 4), DType.bf16), name="rhs"),
    )
    assert counted(product) == ([6, 12], [8])

    rotated = _relations(
        RoPE(),
        (1, 5, 8),
        Var(type=make_tensor_type((1, 5, 8), DType.bf16), name="k"),
        Var(type=make_tensor_type((4096, 4), DType.bf16), name="cos"),
        Var(type=make_tensor_type((4096, 4), DType.bf16), name="sin"),
        Var(type=make_tensor_type((5,), DType.i32), name="pos"),
    )
    assert counted(rotated) == ([40, 40, 16384, 16384, 5], [40, 40]), (
        "a table row is an element of pos_ids, which no boundary holds, so both "
        "tables are reached at every row they could name"
    )

    ctx = TypeInferContext()
    joined = Call(
        type=make_tensor_type((7, 4), DType.bf16),
        target=Concat(axis=0),
        args=(
            Var(type=make_tensor_type((3, 4), DType.bf16), name="head"),
            Var(type=make_tensor_type((4, 4), DType.bf16), name="tail"),
        ),
    )
    assert counted(relations_of(joined, ctx)) == (
        [12, 16],
        [28],
    )

    gathered = Call(
        type=make_tensor_type((3, 8), DType.bf16),
        target=IndexSelect(dim=0),
        args=(
            Var(type=make_tensor_type((64, 8), DType.bf16), name="table"),
            Var(type=make_tensor_type((3,), DType.i32), name="index"),
        ),
    )
    assert counted(relations_of(gathered, ctx)) == (
        [512, 3],
        [24],
    ), "a gather reaches every row it could have named, no boundary holding which"

    written = Call(
        type=make_tensor_type((4, 6), DType.bf16),
        target=InsertSlice(),
        args=(
            Var(type=make_tensor_type((4, 6), DType.bf16), name="dst"),
            Var(type=make_tensor_type((2, 6), DType.bf16), name="update"),
            Tuple(
                type=make_tensor_type((), DType.i64),
                elements=(
                    Constant(type=make_tensor_type((), DType.i64), value=1),
                    Constant(type=make_tensor_type((), DType.i64), value=0),
                ),
            ),
        ),
    )
    assert counted(relations_of(written, ctx)) == (
        [12, 12, 2],
        [12],
    )

    cached = Call(
        type=make_tensor_type((2, 16, 4, 8), DType.bf16),
        target=CacheUpdate(),
        args=(
            Var(type=make_tensor_type((2, 16, 4, 8), DType.bf16), name="cache"),
            Constant(type=make_tensor_type((), DType.i32), value=0),
            Constant(type=make_tensor_type((), DType.i32), value=3),
            Var(type=make_tensor_type((2, 5, 4, 8), DType.bf16), name="new"),
        ),
    )
    assert counted(relations_of(cached, ctx)) == (
        [832, 1, 1, 192],
        [192],
    )


def _as_map(pattern) -> "isl.map":
    """One comparable carrier, whichever affine form a boundary stated."""
    return relation_of(pattern)


@dataclass(frozen=True)
class _WholeWeight:
    """A context answering with the program's weight, whatever is asked."""

    held: object

    args: tuple = (None, None, None)

    def type_of(self, _arg) -> object:
        return self.held


def _counted(relations) -> tuple[list[int], list[int]]:
    """Every boundary's amount, as its own relation reaches it: inputs, outputs."""
    return (
        [reached_elements(item.pattern) for item in relations.inputs],
        [reached_elements(item.pattern) for item in relations.outputs],
    )


def _asked(op, result, *operands, ctx=None):
    """One Call's relations, built the way a consumer builds them."""
    call = Call(
        type=result,
        target=op,
        args=tuple(
            Var(type=type_, name=f"a{index}") for index, type_ in enumerate(operands)
        ),
    )
    return relations_of(call, ctx or TypeInferContext())


def test_a_reduction_reads_the_extent_its_own_participant_holds() -> None:
    """A reduced axis can itself be split, and then the extent is not the program's.

    The result carries `Broadcast` where the reduction happened and each unit
    contributes over the piece it holds. Multiplying by the program's extent
    charges every one of them the whole contraction: on the 6x32 mesh below that
    is 64 against the 2 a unit really reads.
    """
    whole = _asked(
        Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM),
        make_tensor_type((12, 1), DType.bf16),
        make_tensor_type((12, 32), DType.bf16),
    )
    assert _counted(whole) == ([384], [12])

    unit = _asked(
        Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM),
        make_tensor_type((2, 1), DType.bf16),
        make_tensor_type((2, 1), DType.bf16),
    )
    assert _counted(unit) == ([2], [2])

    tall = _asked(
        Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM),
        make_tensor_type((132, 1), DType.f32),
        make_tensor_type((132, 128), DType.f32),
    )
    assert _counted(tall) == ([16_896], [132])


def test_a_reduction_maps_the_axes_its_layout_factored() -> None:
    """A logical axis can become several layout positions, and often does.

    Splitting an axis of 12 over a mesh of 6 makes the projected shape
    `(1, 2, 1)`: two positions for logical axis 0 and one for axis 1. Reducing
    "axis 1" by position would collapse the mesh factor of axis 0 and carry its
    residual through untouched -- and the amount can come out right while that
    happens, which is why the maps are what this asserts. A reduction walks what
    it reads, so the collapse is on the way out.
    """
    mesh = make_mesh((6, 32), ("w", "t"), topology=Topology("thread", 6 * 32))
    source = make_shard_tensor_type(
        (12, 32), mesh=mesh, attrs=(ShardSplit(0), ShardSplit(1)), dtype=DType.bf16
    )
    result = make_shard_tensor_type(
        (12, 1), mesh=mesh, attrs=(ShardSplit(0), Broadcast()), dtype=DType.bf16
    )
    call = Call(
        type=result,
        target=Reduce(axes=(1,), keepdim=True, kind=ReduceKind.SUM),
        args=(Var(type=source, name="x"),),
    )
    relations = relations_of(
        call, CostContext(level="thread", topologies=(Topology("thread", 6 * 32),))
    )
    reads = relation_of(relations.inputs[0].pattern)
    assert reads.dim(isl.dim_type.OUT) == 3, "one image entry per position it made"
    assert reads.is_equal(
        isl.map("{ [d0, d1] -> [0, d0, 0] : 0 <= d0 <= 1 and d1 = 0 }")
    ), "every source coordinate this participant holds, read once"
    written = relation_of(relations.outputs[0].pattern)
    assert written.is_equal(
        isl.map("{ [d0, d1] -> [0, d0, 0] : 0 <= d0 <= 1 and d1 = 0 }")
    ), "and the surviving axis written at the positions holding it, not by number"


def _reduced(type_, axes, keepdim, level=None, topologies=()):
    """One Reduce's relations, with the result its own type inference derives."""
    held = Var(type=type_, name="x")
    target = Reduce(axes=axes, keepdim=keepdim, kind=ReduceKind.SUM)
    inferred = TypeInferVisitor(TypeInferContext()).visit(
        Call(type=type_, target=target, args=(held,))
    )
    call = Call(type=inferred, target=target, args=(held,))
    return relation_of(
        relations_of(
            call, CostContext(level=level, topologies=topologies)
        ).outputs[0].pattern
    )


def test_a_reduction_without_keepdim_names_the_axis_that_survived() -> None:
    """A result coordinate names the logical axis it kept, not a position.

    Dropping the reduced axes shifts the survivors down, so a result coordinate
    and a source axis at the same number are different axes. Once a layout
    factors things the two are not even the same count: reducing axis 1 of a
    `(4,8,5)` whose axis 1 is split leaves the last logical axis at position 3,
    and writing it at position 1 takes the mesh factor of a different axis.
    """
    walked = "0 <= d0 <= 1 and 0 <= d1 <= 2 and 0 <= d2 <= 3"
    assert _reduced(make_tensor_type((2, 3, 4), DType.f32), (1,), False).is_equal(
        isl.map(f"{{ [d0, d1, d2] -> [d0, d2] : {walked} }}")
    )
    assert _reduced(make_tensor_type((2, 3, 4), DType.f32), (0, 2), False).is_equal(
        isl.map(f"{{ [d0, d1, d2] -> [d1] : {walked} }}")
    )
    assert _reduced(make_tensor_type((2, 3, 4), DType.f32), (1, 2), False).is_equal(
        isl.map(f"{{ [d0, d1, d2] -> [d0] : {walked} }}")
    )

    cta = Topology("cta", 2)
    split = make_shard_tensor_type(
        (4, 8, 5),
        mesh=make_mesh((2,), ("c",), topology=cta),
        attrs=(ShardSplit(1),),
        dtype=DType.f32,
    )
    held = "0 <= d0 <= 3 and 0 <= d1 <= 3 and 0 <= d2 <= 4"
    for keepdim, kept in ((False, "d0, 0, 0, d2"), (True, "d0, 0, 0, 0, d2")):
        assert _reduced(split, (1,), keepdim, "cta", (cta,)).is_equal(
            isl.map(f"{{ [d0, d1, d2] -> [{kept}] : {held} }}")
        ), "the last logical axis is written at the position holding it"


def test_layer_norm_reads_its_parameters_across_the_whole_suffix() -> None:
    """The parameters match `x.shape[axis:]`, which can be more than one axis.

    A per-axis reading of a rank-2 suffix would have said 3 or 4 where the
    answer is 12, and the verifier refuses a split at or beyond the normalized
    axis, so that number is the same in every view a program can produce.
    """
    suffix = make_tensor_type((3, 4), DType.f32)
    whole = _asked(
        LayerNorm(axis=1, eps=1e-5),
        make_tensor_type((132, 3, 4), DType.f32),
        make_tensor_type((132, 3, 4), DType.f32),
        suffix,
        suffix,
    )
    assert _counted(whole) == ([1584, 12, 12], [1584])

    unit = _asked(
        LayerNorm(axis=1, eps=1e-5),
        make_tensor_type((1, 3, 4), DType.f32),
        make_tensor_type((1, 3, 4), DType.f32),
        suffix,
        suffix,
    )
    assert _counted(unit) == ([12, 12, 12], [12])


def test_a_convolution_reads_the_source_once_however_often_it_depends_on_it() -> None:
    """Overlapping windows read one element several times; that is one element.

    Sixteen is how many source elements exist to be read. Thirty-six is the
    dependence count of a 2x2 output over a 3x3 kernel, and counting those would
    charge a convolution for arithmetic rather than for movement. A coordinate
    outside the source is a guarded load that fetches nothing, so it is clipped
    rather than charged.
    """
    relations = _asked(
        Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        make_tensor_type((1, 1, 2, 2), DType.f16),
        make_tensor_type((1, 1, 4, 4), DType.f16),
        make_tensor_type((1, 1, 3, 3), DType.f16),
        make_tensor_type((1,), DType.f16),
    )
    assert _counted(relations) == ([16, 9, 1], [4])

    padded = _asked(
        Conv2D(stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        make_tensor_type((1, 32, 32, 32), DType.f16),
        make_tensor_type((1, 16, 32, 32), DType.f16),
        make_tensor_type((32, 16, 3, 3), DType.f16),
        make_tensor_type((32,), DType.f16),
    )
    assert _counted(padded) == ([16_384, 4608, 32], [32_768])


_CONV_CTA = Topology("cta", 2)
_CONV_MESH = make_mesh((2,), ("c",), topology=_CONV_CTA)


def _sharded(shape, attrs, dtype=DType.f16):
    return make_shard_tensor_type(shape, mesh=_CONV_MESH, attrs=attrs, dtype=dtype)


def _projected(op, result, *operands):
    """Both views of one Call, through the context an analysis really uses."""
    call = Call(
        type=result,
        target=op,
        args=tuple(Var(type=t, name=f"a{i}") for i, t in enumerate(operands)),
    )
    views = []
    for level in (None, "cta"):
        relations = relations_of(
            call, CostContext(level=level, topologies=(_CONV_CTA,))
        )
        views.append((_counted(relations), relations))
    return views


def test_a_convolution_projects_onto_the_channels_this_participant_computes() -> None:
    """A replicated weight still looks whole; only its own channels are read.

    Every amount comes off the output's extents and the contraction's, never off
    an operand's shape, because a weight nobody sharded projects to all 4608 of
    itself while the participant reads the 2304 belonging to the sixteen output
    channels it has. The split is driven by the bias, whose only axis is the
    output channel, which is the shape this IR actually builds.
    """
    (whole, _), (unit, _) = _projected(
        Conv2D(stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
        _sharded((1, 32, 32, 32), (ShardSplit(1),)),
        make_tensor_type((1, 16, 32, 32), DType.f16),
        make_tensor_type((32, 16, 3, 3), DType.f16),
        _sharded((32,), (ShardSplit(0),)),
    )
    assert whole == ([16_384, 4608, 32], [32_768])
    assert unit == ([16_384, 2304, 16], [16_384])


def test_a_grouped_convolution_reads_only_the_groups_it_computes() -> None:
    """One group's output channels read one group's input channels.

    The contraction extent is a group's worth, so a participant computing one
    group of two reads eight channels rather than sixteen: 512, where the
    program's channel count would bill 1024. The map says which eight, through
    the group offset the Op's own access relation already states.
    """
    (whole, spanning), (unit, _) = _projected(
        Conv2D(stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=2),
        _sharded((1, 32, 8, 8), (ShardSplit(1),)),
        make_tensor_type((1, 16, 8, 8), DType.f16),
        make_tensor_type((32, 8, 3, 3), DType.f16),
        _sharded((32,), (ShardSplit(0),)),
    )
    assert whole == ([1024, 2304, 32], [2048])
    assert unit == ([512, 1152, 16], [1024])
    walked = (
        "0 <= n < 1 and 0 <= co < 32 and 0 <= oh < 8 and 0 <= ow < 8 "
        "and 0 <= ci < 8 and 0 <= kh < 3 and 0 <= kw < 3"
    )
    assert relation_of(spanning.inputs[0].pattern).is_equal(
        isl.map(
            "{ [n, co, oh, ow, ci, kh, kw] -> "
            "[0, floor(co/16) * 8 + ci, oh + kh - 1, ow + kw - 1] : "
            f"0 <= oh + kh - 1 < 8 and 0 <= ow + kw - 1 < 8 and {walked} }}"
        )
    )


def test_a_grouped_split_that_straddles_a_group_is_refused() -> None:
    """A shard crossing a group boundary reads two groups; its neighbour reads one.

    Twelve output channels in three groups over four shards gives every shard
    three channels, and shards one and two straddle a boundary: the real
    footprints are 1, 2, 2 and 1 groups. A projection here knows a shard's
    extent and not its offset, so it would report the first participant's
    answer for all four. Refused where the author can align the split.
    """
    cta = Topology("cta", 4)
    mesh = make_mesh((4,), ("c",), topology=cta)
    call = Call(
        type=make_tensor_type((1, 12, 8, 8), DType.f16),
        target=Conv2D(stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=3),
        args=(
            Var(type=make_tensor_type((1, 12, 8, 8), DType.f16), name="x"),
            Var(type=make_tensor_type((12, 4, 3, 3), DType.f16), name="w"),
            Var(
                type=make_shard_tensor_type(
                    (12,), mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f16
                ),
                name="b",
            ),
        ),
    )
    with pytest.raises(VerifyError, match="can straddle a group boundary"):
        TypeInferVisitor(TypeInferContext()).visit(call)


def test_a_split_contraction_reads_only_its_share_of_the_channels() -> None:
    """Each participant sums over its own input channels into a partial result.

    The output does not shrink -- every participant computes a partial of the
    whole, which is what the `Partial(sum)` says -- while the weight and the
    input halve with the contraction. Reading the program's channel count gives
    256 and 144, exactly double what moves.
    """
    (whole, _), (unit, _) = _projected(
        Conv2D(stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
        _sharded((1, 4, 6, 6), (Partial("sum"),)),
        _sharded((1, 4, 8, 8), (ShardSplit(1),)),
        _sharded((4, 4, 3, 3), (ShardSplit(1),)),
        _sharded((4,), (Partial("sum"),)),
    )
    assert whole == ([256, 144, 4], [144])
    assert unit == ([128, 72, 4], [144])


def test_an_interleave_reads_each_source_element_once() -> None:
    """Three result positions depend on one element; the element crossed once.

    Four, not twelve. The Op refuses a sharded layout outright, so whole is the
    only view a program can ask about.
    """
    relations = _asked(
        RepeatInterleave(repeats=3, axis=0),
        make_tensor_type((12,), DType.f32),
        make_tensor_type((4,), DType.f32),
    )
    assert _counted(relations) == ([4], [12])


def test_a_tile_instruction_does_not_divide_under_projection() -> None:
    """One instruction is one instruction, however many participants issue it.

    The shape is the instruction's rather than its operands', which is what
    separates these from a MatMul, and a projection that divided them would
    describe a smaller instruction nobody has.
    """
    for op, (m, n, k), expected in (
        (
            Mma_SM80_16x8x16(dtype_a=DType.f16, dtype_b=DType.f16, dtype_acc=DType.f32),
            (16, 8, 16),
            ([256, 128], [128]),
        ),
        (
            Wgmma_SM90_64x128x16(
                dtype_a=DType.f16, dtype_b=DType.f16, dtype_acc=DType.f32
            ),
            (64, 128, 16),
            ([1024, 2048], [8192]),
        ),
    ):
        relations = _asked(
            op,
            make_tensor_type((m, n), DType.f32),
            make_tensor_type((m, k), DType.f16),
            make_tensor_type((k, n), DType.f16),
        )
        assert _counted(relations) == expected


def test_an_indexed_update_keeps_its_container_and_writes_the_rows_it_names() -> None:
    """Which rows are written and where the container lives are two questions.

    The index answers the first, so those boundaries reach every row the axis
    could legally name rather than claiming one. It answers nothing about the
    second: which rows get overwritten does not change which buffer they are in.
    A repeated index writes one row twice and an out-of-order one writes no
    window at all, and neither changes what any boundary reaches.
    """
    shapes = (
        make_tensor_type((4, 8), DType.f32),
        make_tensor_type((2,), DType.i64),
        make_tensor_type((2, 8), DType.f32),
    )
    container = isl.set("{ [d0, d1] : 0 <= d0 < 4 and 0 <= d1 < 8 }")

    def reaches(pattern) -> int:
        return int(str(container.apply(pattern.relation).count_val()))

    copied = _asked(IndexCopy(dim=0), shapes[0], *shapes)
    assert _counted(copied) == ([32, 2, 16], [32])
    assert reaches(copied.inputs[2].pattern) == 16, (
        "the payload rows land wherever the index says, so all of them are read"
    )

    added = _asked(IndexAdd(dim=0), shapes[0], *shapes)
    assert _counted(added) == ([32, 2, 16], [32])
    assert reaches(added.inputs[0].pattern) == 32, (
        "and the rows added to are any of the container's, for the same reason"
    )

    for relations in (copied, added):
        written = relations.outputs[0].pattern
        assert not written.relation.is_single_valued(), (
            "a scatter cannot promise the row it lands on"
        )
        assert reaches(written) == 32


def test_a_view_reads_its_source_at_the_positions_that_source_has() -> None:
    """A renaming reads the coordinate holding the value it renames, and no other.

    A view stated in logical axes is composed with the layout its source ended
    up with, so a participant holding one row of a split source reads position
    zero of it and not the row number the program wrote. The two cases below are
    the shapes that were claimed fixed a round before they were.
    """
    cta = Topology("cta", 2)
    mesh = make_mesh((2,), ("c",), topology=cta)

    def split(shape):
        return make_shard_tensor_type(
            shape, mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f32
        )

    def reads(op, source):
        held = Var(type=source, name="x")
        inferred = TypeInferVisitor(TypeInferContext()).visit(
            Call(type=source, target=op, args=(held,))
        )
        call = Call(type=inferred, target=op, args=(held,))
        relations = relations_of(call, CostContext(level="cta", topologies=(cta,)))
        return relations.inputs[0].pattern

    held = split((8,))
    same = reads(Reshard(layout=held.layout, storage=held.storage), held)
    assert _as_map(same).is_equal(isl.map("{ [d0] -> [0, d0] : 0 <= d0 <= 3 }"))

    turned = reads(Transpose(perm=(0, 1)), split((8, 4)))
    assert _as_map(turned).is_equal(
        isl.map("{ [d0, d1] -> [0, d0, d1] : 0 <= d0 <= 3 and 0 <= d1 <= 3 }")
    )


def _split_call(source):
    """A two-way split of `source` along its last axis."""
    held = Var(type=source, name="x")
    op = SplitOp(axis=1, num_splits=2)
    inferred = TypeInferVisitor(TypeInferContext()).visit(
        Call(type=source, target=op, args=(held,))
    )
    return Call(type=inferred, target=op, args=(held,))


def test_a_divided_axis_the_layout_also_divides_is_refused() -> None:
    """Which part a participant holds needs an offset a projection has not got.

    Splitting an axis two ways over a source already handed out two ways gives
    each participant one whole part -- but saying which one means knowing where
    that participant starts, and a projection carries extents alone. Refused at
    the contract the author wrote against, so it is a program this compiler
    declines rather than one an analysis cannot answer. Dividing an axis the
    layout leaves whole is expressible and stays legal.
    """
    cta = Topology("cta", 2)
    source = make_shard_tensor_type(
        (8, 4),
        mesh=make_mesh((2,), ("c",), topology=cta),
        attrs=(ShardSplit(0),),
        dtype=DType.f32,
    )
    held = Var(type=source, name="x")
    with pytest.raises(VerifyError, match="already Split across participants"):
        TypeInferVisitor(TypeInferContext()).visit(
            Call(type=source, target=SplitOp(axis=0, num_splits=2), args=(held,))
        )

    TypeInferVisitor(TypeInferContext()).visit(
        Call(type=source, target=SplitOp(axis=1, num_splits=2), args=(held,))
    )


def test_a_cache_whose_rows_are_split_is_refused() -> None:
    """`cur_pos` is stated against the whole row axis, so a slice of it cannot say.

    A batch split is fine and stays fine: each participant writes its own rows
    at the same position. Splitting the rows is the case that needs an offset
    nothing here carries, and it is refused from either side -- a split `new`
    against a whole cache asks the same question from the other end. So does a
    batch split only one side agrees to.
    """
    cta = Topology("cta", 2)
    mesh = make_mesh((2,), ("c",), topology=cta)

    def split(shape, axis):
        return make_shard_tensor_type(
            shape, mesh=mesh, attrs=(ShardSplit(axis),), dtype=DType.bf16
        )

    i32 = make_tensor_type((), DType.i32)
    controls = (Constant(type=i32, value=0), Constant(type=i32, value=4))

    batched = (
        Var(type=split((4, 16, 2, 8), 0), name="cache"),
        *controls,
        Var(type=split((4, 4, 2, 8), 0), name="new"),
    )
    TypeInferVisitor(TypeInferContext()).visit(
        Call(type=batched[0].type, target=CacheUpdate(), args=batched)
    )

    for cache, new in (
        (split((1, 16, 2, 8), 1), split((1, 4, 2, 8), 1)),
        (make_tensor_type((1, 16, 2, 8), DType.bf16), split((1, 4, 2, 8), 1)),
    ):
        rowwise = (
            Var(type=cache, name="cache"),
            *controls,
            Var(type=new, name="new"),
        )
        with pytest.raises(VerifyError, match="row axis is Split across participants"):
            TypeInferVisitor(TypeInferContext()).visit(
                Call(type=rowwise[0].type, target=CacheUpdate(), args=rowwise)
            )

    disagreeing = (
        Var(type=split((4, 16, 2, 8), 0), name="cache"),
        *controls,
        Var(type=make_tensor_type((4, 4, 2, 8), DType.bf16), name="new"),
    )
    with pytest.raises(VerifyError, match="Split on one side and not the other"):
        TypeInferVisitor(TypeInferContext()).visit(
            Call(type=disagreeing[0].type, target=CacheUpdate(), args=disagreeing)
        )




def test_a_boundary_reads_the_coordinates_its_op_actually_touches() -> None:
    """How much crossed is derived; from where is what the relation states.

    A count can be right while the coordinates are wrong, and a dependence
    reads the coordinates. Three shapes here have no identity to fall back on:
    a result with an axis its inputs do not have, a source read in pieces by
    several outputs, and an operand broadcast against a larger result. Reusing
    another boundary's map in any of them describes a program nobody wrote.
    """
    ctx = TypeInferContext()

    stacked = Call(
        type=make_tensor_type((2, 4, 8), DType.f32),
        target=Stack(axis=0),
        args=(
            Var(type=make_tensor_type((4, 8), DType.f32), name="lower"),
            Var(type=make_tensor_type((4, 8), DType.f32), name="upper"),
        ),
    )
    relations = relations_of(stacked, ctx)
    stacks = "0 <= d0 <= 1 and 0 <= d1 <= 3 and 0 <= d2 <= 7"
    assert _as_map(relations.inputs[0].pattern).is_equal(
        isl.map(f"{{ [d0, d1, d2] -> [d1, d2] : d0 = 0 and {stacks} }}")
    )
    assert _as_map(relations.inputs[1].pattern).is_equal(
        isl.map(f"{{ [d0, d1, d2] -> [d1, d2] : d0 = 1 and {stacks} }}")
    )
    assert _as_map(relations.outputs[0].pattern).is_equal(
        isl.map(f"{{ [d0, d1, d2] -> [d0, d1, d2] : {stacks} }}")
    )

    parted = Call(
        type=TupleType(
            fields=(
                make_tensor_type((3, 5), DType.f32),
                make_tensor_type((3, 5), DType.f32),
            )
        ),
        target=Split(axis=0, num_splits=2),
        args=(Var(type=make_tensor_type((6, 5), DType.f32), name="whole"),),
    )
    relations = relations_of(parted, ctx)
    parts = "0 <= d0 <= 5 and 0 <= d1 <= 4"
    assert _as_map(relations.inputs[0].pattern).is_equal(
        isl.map(f"{{ [d0, d1] -> [d0, d1] : {parts} }}")
    )
    for field, offset in enumerate((0, 3)):
        writes = "d0" if not offset else f"d0 - {offset}"
        assert _as_map(relations.outputs[field].pattern).is_equal(
            isl.map(
                f"{{ [d0, d1] -> [{writes}, d1] : "
                f"{offset} <= d0 <= {offset + 2} and 0 <= d1 <= 4 }}"
            )
        ), "each part is written on its own run of the axis it was cut on"

    added = Call(
        type=make_tensor_type((4, 8), DType.f32),
        target=Binary(kind=BinaryKind.ADD),
        args=(
            Var(type=make_tensor_type((4, 8), DType.f32), name="whole"),
            Var(type=make_tensor_type((8,), DType.f32), name="row"),
        ),
    )
    relations = relations_of(added, ctx)
    rows = "0 <= d0 <= 3 and 0 <= d1 <= 7"
    assert _as_map(relations.inputs[0].pattern).is_equal(
        isl.map(f"{{ [d0, d1] -> [d0, d1] : {rows} }}")
    )
    assert _as_map(relations.inputs[1].pattern).is_equal(
        isl.map(f"{{ [d0, d1] -> [d1] : {rows} }}")
    )

    held = Call(
        type=make_tensor_type((4, 8), DType.f32),
        target=Binary(kind=BinaryKind.MUL),
        args=(
            Var(type=make_tensor_type((4, 8), DType.f32), name="whole"),
            Var(type=make_tensor_type((4, 1), DType.f32), name="column"),
        ),
    )
    relations = relations_of(held, ctx)
    assert _as_map(relations.inputs[1].pattern).is_equal(
        isl.map("{ [d0, d1] -> [d0, 0] : 0 <= d0 <= 3 and 0 <= d1 <= 7 }")
    )


def test_a_boundary_states_one_carrier_and_refuses_the_others() -> None:
    """One carrier, so a reader never asks which kind it was handed.

    A bare isl map or function says where it reaches and nothing about the
    values its parameters stand for, and whoever restricts it then guesses. The
    carrier that says both is the only one a boundary takes, and every helper
    that builds one hands it over already wrapped.
    """
    for raw in (isl.map("{ [d0] -> [d0] }"), isl.multi_aff("{ [d0] -> [d0] }")):
        with pytest.raises(ValueError, match="through an AffineAccess"):
            BoundaryRelation(raw)
    with pytest.raises(ValueError, match="through an AffineAccess"):
        BoundaryRelation("{ [d0] -> [d0] }")

    held = BoundaryRelation(AffineAccess(isl.map("{ [d0] -> [d0] }")))
    assert isinstance(held.pattern, AffineAccess)
    assert isinstance(identity_access(2), AffineAccess), (
        "a helper hands over the carrier, not the map inside it"
    )
    assert isinstance(broadcast_access((4, 8), (8,)), AffineAccess)
    assert isinstance(linearized_view((2, 3), (6,)), AffineAccess)
    assert AffineAccess(isl.multi_aff("{ [d0] -> [d0] }")).relation.is_equal(
        isl.map("{ [d0] -> [d0] }")
    ), "a function is the relation it is, and is kept as one"

    for op, operands in (
        (MatMul(), ((4, 8), (8, 2))),
        (Reduce(axes=(1,), keepdim=False, kind=ReduceKind.SUM), ((4, 8),)),
        (Concat(axis=0), ((3, 5), (3, 5))),
    ):
        relations = _asked(
            op,
            make_tensor_type((4, 2), DType.f32),
            *(make_tensor_type(shape, DType.f32) for shape in operands),
        )
        for boundary in (*relations.inputs, *relations.outputs):
            assert isinstance(boundary.pattern, AffineAccess)


def test_a_partial_boundary_is_narrowed_and_not_cut_away() -> None:
    """A participant's share cuts every boundary, and a partial one survives it.

    An insert states two things about its destination: the window it writes and
    the rest it keeps. Splitting that destination halves both -- the participant
    keeps its own half of the complement -- and reading the window boundary's
    silence outside its own coordinates as a restriction would cut the
    complement to nothing instead.
    """
    cta = Topology("cta", 2)
    mesh = make_mesh((2,), ("c",), topology=cta)
    destination = make_shard_tensor_type(
        (8,), mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f32
    )
    update = make_shard_tensor_type(
        (4,), mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f32
    )
    call = Call(
        type=destination,
        target=InsertSlice(),
        args=(
            Var(type=destination, name="dst"),
            Var(type=update, name="update"),
            Constant(type=make_tensor_type((), DType.i64), value=2),
        ),
    )

    whole = relations_of(call, TypeInferContext())
    assert _counted(whole) == ([4, 4, 1], [4])

    unit = relations_of(call, CostContext(level="cta", topologies=(cta,)))
    assert _counted(unit) == ([2, 2, 1], [2]), (
        "each participant keeps half the complement and writes half the window"
    )
    assert not relation_of(unit.inputs[0].pattern).is_empty(), (
        "the rest of the container is still there to keep"
    )


def test_a_boundary_reaching_past_its_operand_is_held_to_what_it_was_handed()  -> None:
    """A relation may be written past its value; a projected one never reaches there.

    An insert reads its update at the coordinate the window shifted back to, and
    for the coordinates before the window that is a negative one. Every
    projected boundary is held to the positions this participant was given, so
    what it reaches is inside them and which iterations are its own follows from
    that rather than from a read nobody could perform.
    """
    cta = Topology("cta", 2)
    mesh = make_mesh((2,), ("c",), topology=cta)
    destination = make_shard_tensor_type(
        (8,), mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f32
    )
    update = make_shard_tensor_type(
        (4,), mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f32
    )
    call = Call(
        type=destination,
        target=InsertSlice(),
        args=(
            Var(type=destination, name="dst"),
            Var(type=update, name="update"),
            Constant(type=make_tensor_type((), DType.i64), value=2),
        ),
    )
    ctx = CostContext(level="cta", topologies=(cta,))

    stated = coordinates_of(call, ctx)
    reads = relation_of(stated.inputs[1].pattern)
    assert not reads.intersect_range(
        isl.set("{ [c0] : c0 < 0 }")
    ).is_empty(), "the window's own read runs before its operand begins"

    relations = relations_of(call, ctx)
    held = (
        *(ctx.local_type_of(arg) for arg in call.args),
        ctx.local_type_of(call),
    )
    for boundary, view in zip((*relations.inputs, *relations.outputs), held, strict=True):
        reached = relation_of(boundary.pattern).range()
        if not isinstance(view, TensorType) or reached.is_empty():
            continue
        box = index_set(tuple(view.shape))
        assert box is not None and reached.is_subset(box), (
            f"a boundary reached {reached} outside the {tuple(view.shape)} it was given"
        )
    assert relation_of(relations.inputs[1].pattern).domain().is_equal(
        isl.set("{ [d0] : 2 <= d0 <= 3 }")
    ), "so the iterations left are the ones whose read this participant holds"


def test_a_windows_amount_does_not_move_with_where_it_lands() -> None:
    """Where a window sits is a runtime fact; how much it covers is not.

    A literal offset, an offset that arrives as a value, and an offset a loop
    counts out all place the same window, so the update and the container around
    it move the same bytes each time -- including at the ends, where a window
    flush against a boundary is still its own size. Two numbers place it either
    way, one per axis. What the offset does change is the pattern, which is
    where a reader looks to find out.
    """

    def written(offset) -> tuple[list[int], list[int], object]:
        ctx = TypeInferContext()
        call = Call(
            type=make_tensor_type((4, 6), DType.bf16),
            target=InsertSlice(),
            args=(
                Var(type=make_tensor_type((4, 6), DType.bf16), name="dst"),
                Var(type=make_tensor_type((2, 6), DType.bf16), name="update"),
                offset,
            ),
        )
        relations = relations_of(call, ctx)
        return (
            [reached_elements(item.pattern) for item in relations.inputs],
            [reached_elements(item.pattern) for item in relations.outputs],
            relations.inputs[0].pattern,
        )

    def axes(*values) -> Tuple:
        return Tuple(
            type=make_tensor_type((), DType.i64),
            elements=tuple(
                value
                if isinstance(value, Var)
                else Constant(type=make_tensor_type((), DType.i64), value=value)
                for value in values
            ),
        )

    top, middle, bottom = written(axes(0, 0)), written(axes(1, 0)), written(axes(2, 0))
    assert top[:2] == middle[:2] == bottom[:2] == ([12, 12, 2], [12])
    assert top[2].relation != bottom[2].relation, (
        "where the window lands is what the relation is for"
    )
    assert isl.set("{ [d0, d1] : 0 <= d0 < 2 and 0 <= d1 < 6 }").apply(
        top[2].relation
    ).is_empty(), "a window at the top leaves the rows below it"
    assert isl.set("{ [d0, d1] : 2 <= d0 < 4 and 0 <= d1 < 6 }").apply(
        bottom[2].relation
    ).is_empty(), "and one at the bottom leaves the rows above"

    row = Var(type=make_tensor_type((), DType.i64), name="row")
    runtime = written(axes(row, 0))
    assert runtime[:2] == top[:2]
    (name, bound_to), = runtime[2].parameters
    assert name == "o0" and bound_to is row, (
        "an offset only known later is the value it is, not its spelling"
    )

    induction = Var(type=make_tensor_type((), DType.i64), name="row")
    body = Call(
        type=make_tensor_type((4, 6), DType.bf16),
        target=InsertSlice(),
        args=(
            Var(type=make_tensor_type((4, 6), DType.bf16), name="dst"),
            Var(type=make_tensor_type((2, 6), DType.bf16), name="update"),
            axes(induction, 0),
        ),
    )
    loop = GridRegionExpr(
        type=body.type,
        induction_var=induction,
        carried_args=(),
        init_args=(),
        body=body,
        yield_values=(body,),
        extent=3,
        step=1,
    )
    counted = written(loop.body.args[2])
    assert counted[:2] == top[:2]
    assert [name for name, _value in counted[2].parameters] == ["o0"]


def test_an_unbound_row_count_names_the_operands_that_decide_it() -> None:
    """An `s` nobody bound is a parameter of the relation, bound to that operand.

    Where the rows land and how many there are are two runtime numbers, so the
    relation names them and says which operand each is: a reader restricts it
    rather than guessing. One crossing is the first legal binding of them, and
    the complement is what that leaves -- charging the whole cache would be
    neither, and how many crossings a loop performs is a footprint's question.
    """
    ctx = TypeInferContext()
    call = Call(
        type=make_tensor_type((2, 16, 4, 8), DType.bf16),
        target=CacheUpdate(),
        args=(
            Var(type=make_tensor_type((2, 16, 4, 8), DType.bf16), name="cache"),
            Var(type=make_tensor_type((), DType.i32), name="cur_pos"),
            Var(type=make_tensor_type((), DType.i32), name="s"),
            Var(type=make_tensor_type((2, 5, 4, 8), DType.bf16), name="new"),
        ),
    )
    relations = relations_of(call, ctx)
    per_row, held = 2 * 4 * 8, 2 * 16 * 4 * 8

    update = relations.inputs[3]
    assert reached_elements(update.pattern) == per_row, (
        "one crossing is the first legal window, which is one row"
    )
    waited = dict(update.pattern.parameters)
    assert waited["o1"] is call.args[1], "where the rows land is that operand"
    assert waited["e1"] is call.args[2], "and how many there are is that one"

    kept = relations.inputs[0]
    assert reached_elements(kept.pattern) == held - per_row, (
        "and the complement of that window is what it leaves"
    )
    assert not kept.pattern.relation.is_empty(), (
        "replacing some rows leaves the others"
    )
    left_alone = dict(kept.pattern.parameters)
    assert left_alone["o1"] is call.args[1], "what is left alone is cut by the same start"
    assert left_alone["e1"] is call.args[2], "and by the same row count"
    assert reached_elements(relations.outputs[0].pattern) == per_row


def test_a_lookup_reaches_every_row_it_could_have_named() -> None:
    """The same index shape reaches the same coordinates, whatever it points at.

    The three index vectors here really are different -- run them and the
    results differ -- and nothing in the relation tells them apart: no boundary
    holds the deciding element. So the table read is every row the axis could
    legally name, which is fail-closed and the same for all three, while the
    index itself is read at its own length.
    """
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    ordered, repeated, backwards = (
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([4, 4, 4], dtype=torch.int32),
        torch.tensor([5, 0, 3], dtype=torch.int32),
    )
    produced = []
    for index in (ordered, repeated, backwards):
        run_eval_case(
            EvalCase("", IndexSelect(dim=0), (table, index), torch.index_select(table, 0, index))
        )
        produced.append(torch.index_select(table, 0, index))
    assert not torch.equal(produced[0], produced[1])
    assert not torch.equal(produced[0], produced[2])

    def declared(length: int) -> tuple[int, ...]:
        ctx = TypeInferContext()
        call = Call(
            type=make_tensor_type((length, 4), DType.f32),
            target=IndexSelect(dim=0),
            args=(
                Var(type=make_tensor_type((6, 4), DType.f32), name="table"),
                Var(type=make_tensor_type((length,), DType.i32), name="index"),
            ),
        )
        relations = relations_of(call, ctx)
        return tuple(reached_elements(item.pattern) for item in relations.inputs)

    assert declared(3) == (24, 3), "three rows nobody named is any of the six"
    assert declared(6) == (24, 6)








def test_an_empty_shape_relabels_to_an_empty_relation() -> None:
    """A view of nothing reads nowhere, and says so rather than dividing by it.

    Linearizing a reshape divides by an axis extent, and an axis of length zero
    would make that a division by zero -- or worse, an isl expression that
    parses and means something else. The relation is empty instead, which is
    what a shape holding no elements actually says about where they came from.
    """
    assert _as_map(linearized_view((0, 3), (0, 3))).is_empty()
    assert _as_map(linearized_view((2, 0), (0, 2))).is_empty()
    assert not _as_map(linearized_view((2, 3), (6,))).is_empty()

    with pytest.raises(ValueError, match="a view relabels a shape it can count"):
        linearized_view((2, "ctx"), (6,))










def test_a_relation_is_held_to_the_call_it_was_asked_about() -> None:
    """One entry per operand and one per output, checked against that Call.

    A handler describes an Op in general and is asked about one Call in
    particular, so the two are compared where they still can be. A description
    that skips an operand or invents an output would otherwise be found by
    whichever consumer indexed past the end of it, which is a long way from
    where the mistake is.
    """
    holds = make_tensor_type((4,), DType.f32)
    pair = TupleType(fields=(holds, holds))

    def described(name: str, inputs: int, outputs: int, result):
        target = type(name, (Op,), {})
        register_typeinfer(target)(lambda call, ctx, _r=result: _r)

        @register_access_relation(target)
        def _handler(call, ctx, _in=inputs, _out=outputs) -> AccessRelations:
            return AccessRelations(
                inputs=tuple(BoundaryRelation(_bounded(1)) for _ in range(_in)),
                outputs=tuple(BoundaryRelation(_bounded(1)) for _ in range(_out)),
            )

        return target

    def asked(target, operands: int) -> AccessRelations:
        call = Call(
            type=holds,
            target=target(),
            args=tuple(Var(type=holds, name=f"a{index}") for index in range(operands)),
        )
        return relations_of(call, TypeInferContext())

    for name, inputs, outputs, result, operands, complaint in (
        ("_SkipsAnOperand", 1, 1, holds, 2, "1 input boundary of a call with 2"),
        ("_InventsAnOutput", 1, 2, holds, 1, "2 output boundaries of a call with 1"),
        ("_SkipsAField", 1, 1, pair, 1, "1 output boundary of a call with 2"),
        ("_InventsAField", 1, 3, pair, 1, "3 output boundaries of a call with 2"),
    ):
        with pytest.raises(ValueError, match=re.escape(complaint)):
            asked(described(name, inputs, outputs, result), operands)

    fits = asked(described("_Fits", 2, 1, holds), 2)
    assert (len(fits.inputs), len(fits.outputs)) == (2, 1)
    tupled = asked(described("_FitsATuple", 1, 2, pair), 1)
    assert (len(tupled.inputs), len(tupled.outputs)) == (1, 2)


def test_a_service_count_is_a_number_of_results_and_not_a_truth() -> None:
    """A count says how many; `True` says whether, and Python confuses the two.

    `bool` is an `int` in Python, so a handler that reports a flag where a count
    belongs passes an integer check and prices one result. The reader has no way
    to tell that apart from an operation that really asked for one, so the field
    refuses it at the point it is stated.
    """
    moved = (TrafficBytes(read=4), TrafficBytes(write=4))
    assert Cost({}, moved, {"predicate": 0}).service == {"predicate": 0}

    for refused in ({"predicate": True}, {"predicate": -1}, {"predicate": 1.0}):
        with pytest.raises(ValueError, match="non-negative integers"):
            Cost({}, moved, refused)




def test_a_quantised_scale_is_written_once_per_group() -> None:
    """One scale per group of the quantised axis, which is many-to-one.

    So the scale's own map is an `isl.map` carrying the group size, not an
    identity: `128` elements of the last axis share one entry. A relation that
    made the scale an identity would claim a scale per element and price the
    quantisation as no saving at all.
    """
    relation = _relations(Quant(group=128), (1, 2048))

    assert len(relation.inputs) == 1
    assert relation_of(relation.inputs[0].pattern).is_single_valued()

    assert len(relation.outputs) == 2
    scale = relation_of(relation.outputs[1].pattern)
    assert not scale.is_injective(), "a group of elements shares one entry"
    assert "128" in str(scale)
    assert reached_elements(relation.outputs[1].pattern) == 2048 // 128, (
        "one entry per group, which is the whole saving"
    )


_KEYED = re.compile(rf'(?P<key>[\w-]+){FIELD}(?P<value>"(?:[^"\\]|\\.)*"|\S+)')


def _emitted(text: str, family: str) -> dict[str, str]:
    """One comment taken apart by the record and field layers of the ladder.

    The record layer splits outside a quoted value, which is the whole reason a
    value that holds a separator has to bracket itself. That the pairs rejoin to
    exactly what was read is how the split is held to being one.
    """
    if text.startswith(f"{family}{FIELD}"):
        return {family: text[len(family) + len(FIELD) :]}
    head, _, rest = text.partition(FIELDS)
    assert head == family, text
    keyed = list(_KEYED.finditer(rest))
    assert FIELDS.join(match.group(0) for match in keyed) == rest, rest
    return {match["key"]: match["value"] for match in keyed}


def _assert_shape(text: str, declared: object) -> None:
    """One value keeps the shape its declared type renders in.

    Which is also how the separator ladder is checked: an inner value holding an
    outer separator would not match the shape its own type renders in.
    """
    if declared is int:
        assert re.fullmatch(r"-?\d+", text), (text, declared)
    elif declared is Prose:
        assert re.fullmatch(r'"(?:[^"\\]|\\.)*"', text), (text, declared)
    elif declared is str:
        assert FIELD not in text and FIELDS not in text, (text, declared)
    elif declared is TrafficBytes:
        assert re.fullmatch(rf"r-?\d+{re.escape(PAIR)}w-?\d+", text), text
    elif declared is TripInterval:
        assert re.fullmatch(rf"\[[^{ENTRIES}]+{ENTRIES}[^{ENTRIES}]+\)(\{TRIPS}\d+)?", text), text
    elif get_origin(declared) is TotalAndPerUnit:
        (inner,) = get_args(declared)
        total, separator, per_unit = text.partition(PER_UNIT)
        assert separator == PER_UNIT, text
        _assert_shape(total, inner)
        _assert_shape(per_unit, inner)
    elif get_origin(declared) is dict:
        key_type, value_type = get_args(declared)
        for entry in text.split(ENTRIES):
            key, separator, value = entry.partition(ENTRY)
            assert separator == ENTRY, entry
            _assert_shape(key, key_type)
            _assert_shape(value, value_type)
    elif get_origin(declared) is tuple:
        (item_type,) = (arg for arg in get_args(declared) if arg is not Ellipsis)
        for item in text.split(ENTRIES):
            _assert_shape(item, item_type)
    else:
        raise AssertionError(f"{declared} has no rendering: {text}")


_PROSE_PROBE = 'l2 holds 2 B, spills; says "no" over \\'


def _analysed_mega() -> tuple[dict[type, list[tuple[object, object]]], tuple[object, ...]]:
    """What one real analysis leaves on the IR, and the views it reports."""
    result = analyze(
        MoEMegaKernel,
        MoEMegaKernel.entry_function(),
        analysis=("compute-cost", "memory", "roofline", "performance"),
    )
    rendered = render_analysis(result)
    function = result.function
    found: dict[type, list[tuple[object, object]]] = {}
    for expr in (function, *postorder(function.body)):
        for value in expr.metadata:
            found.setdefault(type(value), []).append((value, expr))
    loop_result = analyze(
        FlashSplitKDecode,
        FlashSplitKDecode.entry_function(),
        analysis="memory",
        dims={"ctx": 4096},
    )
    for expr in postorder(loop_result.function.body):
        record = get_metadata(expr, LoopFootprintMetadata)
        if record is not None:
            found.setdefault(type(record), []).append((record, expr))
    return found, rendered.summary


def _every_stated_record() -> dict[type, list[object]]:
    """Every declared record, from real analyses, plus one prose probe.

    The mega kernel fits its caches, so an advisory needs the second program;
    the probe then carries every separator the ladder uses, which is what a
    quoted value has to survive.
    """
    found, views = _analysed_mega()
    stated: dict[type, list[object]] = {
        record_type: [record for record, _ in records]
        for record_type, records in found.items()
    }
    for view in views:
        stated.setdefault(type(view), []).append(view)
    advisory = render_analysis(
        analyze(_oversized_working_set, _oversized_working_set.entry_function(), analysis="memory")
    )
    stated.setdefault(AdvisorySummary, []).extend(
        view for view in advisory.summary if isinstance(view, AdvisorySummary)
    )
    stated[AdvisorySummary].append(AdvisorySummary(Prose(_PROSE_PROBE)))

    assert set(declared_records()) <= set(stated)
    assert {BindingMetadata, OccurrenceProvenance} <= set(stated)
    assert len(stated[AdvisorySummary]) > 1
    return stated


def test_a_record_comment_states_only_what_it_declared() -> None:
    """Every key on a comment maps back to a field or a declared projection.

    A key spelled by hand drifts from the field it reports -- five did -- and a
    projection nobody declared can grow a sixth. So the walk is held to the
    declarations: every key is one of them, every value keeps the shape its
    declared type renders in, and a record that measured nothing states its
    family name and stops. Metadata that is not a report states nothing at all,
    and a summary line is held to the same rules as an annotated equation.
    """
    for record_type, records in _every_stated_record().items():
        declared = comment_of(record_type)
        if declared is None:
            assert all(render_comment(record) is None for record in records)
            continue
        keys = {emission.key.replace("_", "-"): emission for emission in declared.emissions}
        for record in records:
            emitted = _emitted(render_comment(record), declared.family)
            for key, value in emitted.items():
                emission = keys.get(key) or (
                    declared.emissions[0] if key == declared.family else None
                )
                assert emission is not None, (key, declared)
                assert not emission.opt_in, (key, "asked for nothing")
                _assert_shape(value, emission.type)
                if emission.type is Prose:
                    assert json.loads(value) == str(emission.of(record))

    for record_type in declared_records():
        if any(field.default is MISSING for field in dataclass_fields(record_type)):
            continue
        declared = comment_of(record_type)
        nothing = _emitted(render_comment(record_type()), declared.family)
        assert set(nothing) <= {declared.family, "waves"}, nothing
        assert nothing.get("waves", "1") == "1"


def test_a_reported_record_keys_every_field_by_its_own_name() -> None:
    """JSON is the record's own field names, and the comment cannot crop it.

    A handwritten projection could rename a key and no output assertion would
    notice, and a comment leaving a key out must not take it out of what programs
    read: a default, a zero, and a null are facts a consumer branches on. So the
    report states every field a record has, under the field's own name, and the
    only key it may leave out is one read from the expression, which is where a
    Function Call has no operand split to state.
    """
    found, _ = _analysed_mega()
    for record_type, records in found.items():
        if comment_of(record_type) is None:
            continue
        names = {field.name for field in dataclass_fields(record_type)}
        for record, expr in records:
            reported = render_record(record, expr)

            assert set(reported) <= names
            assert names - set(reported) <= expr_fields(record_type)
            for key, value in _emitted(
                render_comment(record), family_of(record_type)
            ).items():
                stated = reported.get(key.replace("-", "_"))
                if isinstance(stated, int | str):
                    assert value == str(stated), (key, value, stated)


def _planned() -> tuple[object, object]:
    """The decode program's buffers, whole and as one participant sees them.

    Eight CTAs share one allocation. Query heads are the interesting axis: 32 of
    them, factored `(4, 8)` and split on the 8, so a CTA holds four heads that
    are eight apart rather than four in a row.
    """
    aimed = replace(
        GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),)
    )
    result = analyze(
        aimed, aimed.entry_function(), analysis="memory", level="cta", dims={"ctx_len": 1024}
    )
    plan = build_buffer_plan(result.function, "cta")
    return plan, result.function


def test_a_participant_owns_the_positions_its_layout_gave_it() -> None:
    """What a shard holds is a run of layout positions, not a run of an axis.

    Thirty-two heads over eight CTAs is four heads each, and which four depends
    on which position the mesh divides. Here the axis is factored `(4, 8)` and
    the 8 is divided, so CTA 3 holds heads 3, 11, 19 and 27. Stated as a logical
    range that would be heads 12 through 15: the right count of the wrong data,
    which is why the domain is written in positions.
    """
    plan, _ = _planned()
    heads = next(item for item in plan.buffers if item.extents == (1, 1, 4, 8, 128))
    assert heads.ref.shape == (1, 1, 32, 128)

    mine = next(
        item
        for item in plan.project(3).buffers
        if (item.binding, item.field) == (heads.binding, heads.field)
    )
    assert mine.origin == (0, 0, 0, 3, 0)
    assert mine.extents == (1, 1, 4, 1, 128)
    assert mine.domain.is_equal(
        isl.set("{ [0, 0, i2, 3, i4] : 0 <= i2 < 4 and 0 <= i4 < 128 }")
    )


def test_a_shard_is_a_view_of_one_buffer_and_not_a_buffer_of_its_own() -> None:
    """Projection narrows what a participant sees, never where the bytes are.

    Every participant names the same buffer at the same address, so a reader
    that adds up per-participant traffic against buffer numbers is adding up
    one allocation. Together they cover it exactly, because a shard that no
    one holds is a byte no one wrote and one two hold is one counted twice.
    """
    plan, _ = _planned()
    whole = {(item.binding, item.field): item for item in plan.buffers}
    covered: dict[tuple[str, int], object] = {}
    for participant in range(8):
        for item in plan.project(participant).buffers:
            key = (item.binding, item.field)
            assert item.ref == whole[key].ref
            seen = covered.get(key)
            covered[key] = item.domain if seen is None else seen.union(item.domain)
    assert set(covered) == set(whole)
    for key, union in covered.items():
        assert union.is_equal(whole[key].domain)


def test_a_loop_carries_one_factorization_round_its_own_buffer() -> None:
    """What a loop is entered with, names each trip, and yields is one buffer.

    Three derivations reach the same value -- the init's own Op, the loop's phi,
    and whatever the body computed -- and a layout that factors its axes one way
    on the way in and another on the way out is two buffers under one name. The
    plan and the type system then project it differently, which is a wrong
    address rather than a wrong number.
    """
    _plan, function = _planned()
    topologies = (Topology("cta", 8),)

    def factored(expr) -> tuple:
        held = local_type_of(expr.type, level="cta", topologies=topologies)
        return tuple(tuple(leaf.shape) for leaf in tensor_types(held))

    loops = [
        expr for expr in postorder(function.body) if isinstance(expr, GridRegionExpr)
    ]
    assert loops, "this program was expected to carry a buffer round a loop"
    for loop in loops:
        for init, carried, yielded in zip(
            loop.init_args, loop.carried_args, loop.yield_values, strict=True
        ):
            assert factored(init) == factored(carried) == factored(yielded), (
                "one buffer round a loop is one factorization"
            )
        assert factored(loop) == tuple(
            item for value in loop.yield_values for item in factored(value)
        ), "and the loop's own result is what its last trip yielded"


def test_a_participants_share_of_a_buffer_is_the_type_it_was_given() -> None:
    """The plan and the type system project the same program the same way.

    They are separate derivations -- one from the mesh coordinate a position is
    divided by, one from the local shape the type carries -- and a difference
    between them would mean an access relation written against one and priced
    against the other.
    """
    plan, function = _planned()
    topologies = (Topology("cta", 8),)
    stated: dict[int, list[tuple[int, ...]]] = {}
    for expr in (*function.params, *postorder(function.body)):
        if get_metadata(expr, BufferAllocationMetadata) is None:
            continue
        local = local_type_of(expr.type, level="cta", topologies=topologies)
        stated[id(expr)] = [tuple(leaf.shape) for leaf in tensor_types(local)]
    assert stated
    checked = 0
    for participant in range(8):
        for item in plan.project(participant).buffers:
            held = stated[item.expr_id]
            if item.field < len(held):
                assert item.extents == held[item.field]
                checked += 1
    assert checked


def test_a_buffer_no_participant_holds_is_not_in_that_participants_plan() -> None:
    """A mesh that names some participants leaves the rest nothing to see."""
    mesh = make_mesh((2,), ("c",), topology=Topology("cta", 4))
    held = make_shard_tensor_type((8,), mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f32)
    item = PlannedBuffer(
        expr_id=1,
        binding="x",
        field=0,
        ref=BufferRef(0, "gmem", 0, 32, (8,), held.layout),
        origin=(0, 0),
        extents=(2, 4),
    )
    plan = BufferPlan(level="cta", buffers=(item,))
    assert plan.project(1).buffers[0].origin == (1, 0)
    assert plan.project(3).buffers == ()


def test_the_fields_of_one_value_tile_the_allocation_it_was_given() -> None:
    """An address means something only if the bytes behind it are the value's.

    Leaves that share a buffer are laid out consecutively inside the allocation
    the value's lifetime was sized by, so the last one ends exactly where that
    allocation does. A gap would mean bytes charged to a value no field of it
    occupies, and an overrun would mean two values sharing bytes neither one
    placed. Leaves in different buffers are a value naming other values' bytes
    rather than holding any, and tile nothing; neither does a value only known
    to be somewhere in a buffer, which has no offset to tile from.
    """
    plan, function = _planned()
    record = get_metadata(function, MemoryMetadata)
    sizes = {(item.level, item.binding): item.bytes for item in record.lifetimes}
    owned: dict[tuple[int, str], list[PlannedBuffer]] = {}
    for item in plan.buffers:
        owned.setdefault((item.expr_id, item.ref.level), []).append(item)
    assert owned
    checked = 0
    for (_value, level), held in owned.items():
        refs = [item.ref for item in sorted(held, key=lambda item: item.field)]
        if len({ref.buffer_id for ref in refs}) != 1:
            continue
        if any(ref.offset is None for ref in refs):
            continue
        cursor = refs[0].offset
        for ref in refs:
            assert ref.offset == cursor, (held[0].binding, level)
            cursor += ref.size
        stated = sizes.get((level, held[0].binding))
        if stated is None:
            continue
        assert cursor - refs[0].offset == stated, (held[0].binding, level)
        checked += 1
    assert checked


def _traffic_of(module, level, dims):
    """Every primitive occurrence's traffic as the analysis attached it, by op."""
    result = analyze(
        module, module.entry_function(), analysis=("compute-cost", "memory"),
        level=level, dims=dims,
    )
    found = {}
    for expr in postorder(result.function.body):
        if not isinstance(expr, Call) or isinstance(expr.target, Function):
            continue
        record = get_metadata(expr, TrafficMetadata)
        if record is not None:
            found.setdefault(type(expr.target).__name__, []).append(record)
    return found


def test_re_indexing_moves_nothing_and_computing_moves_something() -> None:
    """Renaming the axes over elements asks the run for none of them.

    Which boundaries move at all is the Op's own evaluator: a reshape reports no
    direction on any of them, so whatever its relations reach comes to nothing,
    while an add reports reading its operands and writing its result and is
    charged what those relations reach.
    """
    found = _traffic_of(
        replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),)),
        "cta",
        {"ctx_len": 1024},
    )
    renames = found["Reshape"]
    assert renames
    assert any(record.whole == () for record in renames)

    for record in found["Binary"]:
        assert record.whole, "an operation that reads its operands moved bytes"
        for level, moved in record.whole:
            assert level == "gmem" and moved.read and moved.write


def test_a_unit_moves_the_share_of_a_buffer_it_was_given() -> None:
    """One participant's bytes are one participant's, not the program's.

    The unit reading is the relation asked of a unit, already the projection
    onto one participant; the whole window is never intersected with anything to
    get it. Eight CTAs divide 32 query heads, so a reduction reading 8192 bf16
    elements and writing 64 reads 1024 and writes 8 per unit. A value none of
    them divides is read whole by each, which is why the share is the layout's
    answer and not a division by the number of participants.
    """
    found = _traffic_of(
        replace(GqaOnline, target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 8),)),
        "cta",
        {"ctx_len": 1024},
    )
    divided = [
        record
        for record in found["Reduce"]
        if record.whole == (("gmem", TrafficBytes(16384, 128)),)
    ]
    assert divided, "no reduction read 8192 elements and wrote 64"
    for record in divided:
        assert record.per_unit == (("gmem", TrafficBytes(2048, 16)),)

    replicated = [
        record
        for record in found["Cast"]
        if record.whole == (("gmem", TrafficBytes(1024, 2048)),)
    ]
    assert replicated, "no cast read a value every participant holds whole"
    for record in replicated:
        assert record.per_unit == record.whole


def test_an_occurrence_nobody_placed_is_charged_for_the_copy_nobody_ruled_out() -> None:
    """Addresses prove a move came to nothing; without them nothing is proven.

    This program's machine places nothing it keeps, so no two values can be
    shown to be the same bytes. What follows is that every transfer is a copy,
    not that the program moves nothing: an unproven move is a move, and the
    occurrence still says how much it moved.
    """
    result = analyze(
        _NoParallelLevel, _NoParallelLevel.entry_function(), analysis=("compute-cost", "memory")
    )
    function = result.function
    assert get_metadata(function, MemoryMetadata).allocation is None
    assert build_buffer_plan(function, "cta").buffers == ()

    stated = _occurrences(function)
    assert stated
    for expr in stated:
        moved = get_metadata(expr, TrafficMetadata)
        assert moved is not None and moved.whole

    total = get_metadata(function, TrafficMetadata)
    assert total is not None
    assert dict(total.whole) == {"gmem": TrafficBytes(2 * 256 * 4, 256 * 4)}


def _matmul_relations(lhs_shape, rhs_shape, out_shape):
    """One contraction's relations, at the shapes it was written with."""
    call = Call(
        type=make_tensor_type(out_shape, DType.f32),
        target=MatMul(),
        args=(
            Var(type=make_tensor_type(lhs_shape, DType.f32), name="a"),
            Var(type=make_tensor_type(rhs_shape, DType.f32), name="b"),
        ),
    )
    return relations_of(call, CostContext())


def test_a_contraction_reads_each_operand_at_its_own_extents() -> None:
    """A contraction reaches along the axis it sums, not along the one it keeps.

    Reading each operand at the result's coordinates is right only while every
    extent agrees. For `(128,64) x (64,32)` it claims the left operand has 32
    columns and the right 128 rows, which is the wrong half of both: the shapes
    the map names have to be the shapes the operands have. The axis being summed
    is one of the coordinates the Op is asked by, so it appears in the domain
    rather than as something existential inside an image.
    """
    relations = _matmul_relations((128, 64), (64, 32), (128, 32))
    walked = "0 <= d0 < 128 and 0 <= d1 < 32 and 0 <= k < 64"
    assert relation_of(relations.inputs[0].pattern).is_equal(
        isl.map(f"{{ [d0, d1, k] -> [d0, k] : {walked} }}")
    )
    assert relation_of(relations.inputs[1].pattern).is_equal(
        isl.map(f"{{ [d0, d1, k] -> [k, d1] : {walked} }}")
    )
    assert relation_of(relations.outputs[0].pattern).is_equal(
        isl.map(f"{{ [d0, d1, k] -> [d0, d1] : {walked} }}")
    ), "and the result is accumulated over it, which is why the write repeats"
    assert relation_of(relations.inputs[0].pattern).range().is_equal(
        isl.set("{ [i0, i1] : 0 <= i0 < 128 and 0 <= i1 < 64 }")
    )
    assert relation_of(relations.inputs[1].pattern).range().is_equal(
        isl.set("{ [i0, i1] : 0 <= i0 < 64 and 0 <= i1 < 32 }")
    )
    assert reached_elements(relations.outputs[0].pattern) == 4096
    assert [
        reached_elements(boundary.pattern) for boundary in relations.inputs
    ] == [8192, 2048], "walking the summed axis moves no operand element twice"


def test_a_contraction_reads_a_batch_it_broadcasts_at_one_coordinate() -> None:
    """A batch an operand has one of is read there however many the result has."""
    relations = _matmul_relations((1, 128, 64), (4, 64, 32), (4, 128, 32))
    walked = "0 <= d0 < 4 and 0 <= d1 < 128 and 0 <= d2 < 32 and 0 <= k < 64"
    assert relation_of(relations.inputs[0].pattern).is_equal(
        isl.map(f"{{ [d0, d1, d2, k] -> [0, d1, k] : {walked} }}")
    )
    assert relation_of(relations.inputs[1].pattern).is_equal(
        isl.map(f"{{ [d0, d1, d2, k] -> [d0, k, d2] : {walked} }}")
    )
    assert reached_elements(relations.inputs[0].pattern) == 128 * 64, (
        "read once, not once per batch the result has"
    )


def _occurrences(function):
    """Every primitive occurrence of *function* that states its accesses."""
    return [
        expr
        for expr in postorder(function.body)
        if isinstance(expr, Call)
        and not isinstance(expr.target, Function)
        and access_relation_registry.lookup(type(expr.target)) is not None
    ]


def test_the_two_families_answer_about_different_halves_of_one_call() -> None:
    """Work and movement are one declaration read twice, not two declarations.

    An Op states its flops, its typed service and its movement once. Which
    family reports which half is a question about ownership, and the halves do
    not overlap: a record of work states no bytes, and a record of movement
    states no flops. Both are asked of the same occurrence, so every priced
    Call carries one of each.
    """
    placed = replace(
        GqaOnline,
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 8),),
    )
    for owner, dims in ((MoEMegaKernel, None), (placed, {"ctx_len": 1024})):
        asked = {} if dims is None else {"dims": dims}
        result = analyze(
            owner, owner.entry_function(), analysis=("compute-cost", "memory"), **asked
        )
        checked = 0
        for expr in (result.function, *postorder(result.function.body)):
            cost = get_metadata(expr, ComputeCostMetadata)
            moved = get_metadata(expr, TrafficMetadata)
            if cost is None and moved is None:
                continue
            assert cost is not None and moved is not None, (
                "one half of an occurrence was recorded without the other"
            )
            assert not hasattr(cost, "traffic"), "the work record still states bytes"
            assert not hasattr(moved, "flops"), "the movement record still states work"
            checked += 1
        assert checked > 1, "this program stated no cost to carry"

def test_a_bound_divides_only_what_the_target_published_a_rate_for() -> None:
    """A nanosecond is owed by what could have been priced, not by everything.

    Bytes at a level no vendor rates are stated and left untimed, so a bound
    that answers one nanosecond for them reports a floor the machine never
    promised. A dtype whose rate is simply absent is the opposite case: the work
    is of a kind this prices, the number is missing, and the bound says so.
    """
    facts = CudaTarget("nvidia.h200_sxm").get_facts(ThroughputFacts)
    assert facts.bandwidth_level == "gmem"

    for level in ("rmem", "smem"):
        local = TrafficMetadata(whole=((level, TrafficBytes(read=8, write=8)),))
        bound = _cost_bound(ComputeCostMetadata(), local, facts)
        assert (bound.ideal_ns, bound.bound_by) == (0, "none"), (
            f"{level} bytes nobody rated were given a nanosecond"
        )

    crossed = TrafficMetadata(whole=(("gmem", TrafficBytes(read=4096)),))
    assert _cost_bound(ComputeCostMetadata(), crossed, facts).bound_by == "memory"
    computed = ComputeCostMetadata(flops=(("f32", 4096),))
    assert _cost_bound(computed, TrafficMetadata(), facts).bound_by == "compute"

    unrated = ComputeCostMetadata(flops=(("f4e2m1", 4096),))
    priced = _cost_bound(unrated, TrafficMetadata(), facts)
    assert (priced.ideal_ns, priced.bound_by) == (1, "unrated"), (
        "work of a kind this prices lost its floor along with the rate"
    )


def test_a_bound_is_refused_where_half_of_what_it_divides_is_missing() -> None:
    """A dependency that was declared is one whose answer may be relied on.

    Both readers of the two records name memory among their dependencies, so an
    occurrence reaching them without the bytes it moves is a broken run rather
    than an occurrence that moved none. Treating the absence as zero reports a
    program faster than the one that was measured, which is the one direction a
    lower bound may not be wrong in.
    """
    for run in (analyze_roofline, analyze_performance):
        result = analyze(
            SquareCuda, SquareCuda.entry_function(), analysis=("compute-cost", "memory")
        )
        robbed = next(
            expr for expr in postorder(result.function.body) if isinstance(expr, Call)
        )
        detach(robbed, TrafficMetadata)
        with pytest.raises(AnalysisError, match="traffic record"):
            run(SquareCuda, result.function, SquareCuda.resolve_target(), "cta", None)

    result = analyze(
        SquareCuda, SquareCuda.entry_function(), analysis=("compute-cost", "memory")
    )
    assert all(
        get_metadata(expr, TrafficMetadata) is not None
        for expr in postorder(result.function.body)
        if isinstance(expr, Call)
    ), "this case is about the total, so every occurrence must still state its own"
    detach(result.function, TrafficMetadata)
    with pytest.raises(AnalysisError, match="traffic root record"):
        analyze_roofline(
            SquareCuda, result.function, SquareCuda.resolve_target(), "cta", None
        )




def test_a_derived_answer_does_not_survive_into_a_program_that_cannot_give_it() -> None:
    """An answer belongs to the analysis that reached it, not to the program.

    A record left on an authored value travels into every view built from it
    unless it is known to be derived. It is put here on a value no analysis
    attaches one to, so nothing this round writes can cover it up: whatever
    reaches the view came from the round before, and a reader would take it for
    this one's.
    """
    authored = MoEMegaKernel.entry_function()
    carried = next(
        expr
        for expr in postorder(authored.body)
        if not isinstance(expr, Call) and tensor_types(expr.type)
    )
    attach(carried, TrafficMetadata(whole=(("marker", TrafficBytes(1, 1)),)))
    try:
        result = analyze(
            MoEMegaKernel,
            MoEMegaKernel.entry_function(),
            analysis=("compute-cost", "memory"),
        )
        seen = 0
        for expr in postorder(result.function.body):
            record = get_metadata(expr, TrafficMetadata)
            assert record is None or "marker" not in dict(record.whole), describe(expr)
            seen += record is not None
        assert seen, "this round attached nothing, so nothing covered anything up"
    finally:
        detach(carried, TrafficMetadata)


def test_a_function_moves_its_occurrences_as_often_as_its_loops_repeat_them() -> None:
    """A total is the bytes counted again for every trip that moves them.

    One occurrence inside a loop of 24 moves what it moves 24 times, and the
    rule for saying so lives here rather than in each reader: two readers with
    two copies of it drift. The boundaries are not repeated with it -- which
    operand moved what belongs to the occurrence, not to the total. Three
    occurrences in it place a window by three eight-byte numbers each, two of
    them twenty-four times over and one of them twice, which is what the
    register total counts.
    """
    case = next(item for item in CORPUS if item.id == "access_footprint.grouped_moe")
    selected = case.analyze[0]
    owner, entry = case.resolve(case.build(), selected.selector)
    result = analyze(owner, entry, analysis=("compute-cost", "memory"), dims=selected.dims)
    function = result.function

    record = get_metadata(function, TrafficMetadata)
    assert record is not None

    trips = enclosing_trips(function.body)
    counted: dict[str, list[int]] = {}
    repeated = 0
    for expr in _occurrences(function):
        moved = get_metadata(expr, TrafficMetadata)
        assert moved is not None
        times = trips.get(id(expr), 1)
        repeated = times if times > repeated else repeated
        for level, bytes_ in moved.whole:
            running = counted.setdefault(level, [0, 0])
            running[0] += bytes_.read * times
            running[1] += bytes_.write * times
    assert repeated > 1, "no occurrence of this program is repeated by a loop"
    assert dict(record.whole) == {
        level: TrafficBytes(*moved) for level, moved in counted.items()
    }
    assert dict(record.whole)["rmem"] == TrafficBytes(3 * 8 * (24 + 2 + 24), 0)


def test_a_total_counts_work_a_unit_does_not_do_only_where_it_happened() -> None:
    """The program moved it; this unit did not. The total says both.

    Each expert here runs on its own slice of the machine, so an occurrence of
    one is work the other's units never do. It belongs in what the program
    moves and in none of what a unit moves, and a total that confused the two
    would either lose the work or charge it to everybody.
    """
    result = analyze(
        MoEMegaKernel, MoEMegaKernel.entry_function(), analysis=("compute-cost", "memory")
    )
    function = result.function
    record = get_metadata(function, TrafficMetadata)
    assert record is not None

    places = _call_placements(MoEMegaKernel, function, "cta")
    elsewhere, here = {}, {}
    for expr in _occurrences(function):
        moved = get_metadata(expr, TrafficMetadata)
        into = here if 0 in places.get(id(expr), {0}) else elsewhere
        for level, bytes_ in moved.whole:
            running = into.setdefault(level, [0, 0])
            running[0] += bytes_.read
            running[1] += bytes_.write
    assert elsewhere, "every occurrence of this program runs on the first participant"

    for level, moved in record.whole:
        counted = [
            here.get(level, [0, 0])[side] + elsewhere.get(level, [0, 0])[side]
            for side in (0, 1)
        ]
        assert (moved.read, moved.write) == tuple(counted)
    for level, moved in record.per_unit:
        assert moved.read <= here.get(level, [0, 0])[0]
        assert moved.write <= here.get(level, [0, 0])[1]


def _totalled(target, cost=None):
    """One occurrence of *target* alone in a function, and what totalling it gave."""
    held = make_tensor_type((4,), DType.f32)
    source = Var(type=held, name="held")
    body = Call(type=held, target=target, args=(source,))
    if cost is not None:
        attach(body, cost)
    function = Function(
        type=held, name="main", params=(source,), body=body, return_type=held
    )
    _record_traffic(function, FunctionScope(_NoParallelLevel, function), None, ())
    return body, function


def test_a_window_is_given_bytes_of_its_own_and_still_moves_none() -> None:
    """A window nobody planned is allocated, and allocating it is not moving it.

    No plan has put a window at its source's addresses, so each is given bytes
    of its own. What it moves is a separate question and the answer is nothing:
    naming the bytes it is a window of asks the run for none of them.
    """
    result = analyze(
        _RenamesTwice, _RenamesTwice.entry_function(), analysis=("compute-cost", "memory")
    )
    windows = [
        expr
        for expr in postorder(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, SliceOp)
    ]
    assert len(windows) == 2
    outer, inner = (get_metadata(expr, BufferAllocationMetadata) for expr in windows)
    assert outer.fields[0].buffer_id != inner.fields[0].buffer_id, (
        "no plan shares these addresses, so each window owns its own"
    )

    for expr in windows:
        moved = dict(get_metadata(expr, TrafficMetadata).whole)
        assert moved["gmem"] == TrafficBytes(0, 0), "a static window was charged one"


def test_a_window_whose_start_is_only_known_at_run_time_is_still_a_window() -> None:
    """Not knowing where a window lands is not knowing that it moved.

    Its start arrives as a value, so what the occurrence reads is that value
    rather than the bytes it names. Whoever really reads the window is who moves
    anything; being given bytes of its own is not reading them.
    """
    case = next(item for item in CORPUS if item.id == "deepseek_v4_flash")
    selected = next(item for item in case.analyze if item.selector == "mla_kv_update")
    owner, entry = case.resolve(case.build(), selected.selector)
    result = analyze(owner, entry, analysis=("compute-cost", "memory"), dims=selected.dims)

    windows = [
        expr
        for expr in postorder(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, SliceOp)
    ]
    assert windows, "this program was expected to window something"
    assert all(
        get_metadata(expr, BufferAllocationMetadata) is not None for expr in windows
    ), "every window of this program was given somewhere to be"
    for expr in windows:
        moved = dict(get_metadata(expr, TrafficMetadata).whole)
        assert moved.get("gmem", TrafficBytes()).write == 0, (
            "a window was charged for writing bytes it only named"
        )
        assert moved.get("gmem", TrafficBytes()).read < tensor_bytes(expr.type), (
            "a window was charged for reading the bytes it only named"
        )


def _slice_relation(source_shape, sizes, strides, starts, *, split_over=None):
    """The relation one Slice states, built through the registered handler.

    `Slice.sizes` is how many elements the window has, not how far it reaches, so
    the result's shape is those sizes whatever the strides are.

    *split_over* gives the source a layout that factors its first axis over that
    many participants, so the relation has to spread the logical coordinate it
    walks over the positions the layout made.
    """
    out_shape = tuple(sizes)
    source = make_tensor_type(source_shape, storage=StorageKind.GMEM)
    result = make_tensor_type(out_shape, storage=StorageKind.GMEM)
    held = source
    if split_over is not None:
        first = source_shape[0] // split_over
        held = make_tensor_type(
            (first, *source_shape[1:]),
            layout=ShardLayout(
                Layout(
                    (first, *source_shape[1:]),
                    try_c_order_strides((first, *source_shape[1:])),
                ),
                (ShardSplit(0),),
                make_mesh(
                    (split_over,), ("block",), Topology("cta", split_over)
                ),
            ),
            storage=StorageKind.GMEM,
        )
    operand = Var(type=source, name="x")
    elements = tuple(
        i64_const(start)
        if isinstance(start, int)
        else Var(type=make_tensor_type((), storage=StorageKind.GMEM), name=start)
        for start in starts
    )
    offsets = Tuple(
        type=TupleType(tuple(item.type for item in elements)), elements=elements
    )
    call = Call(
        type=result,
        target=SliceOp(sizes=tuple(sizes), strides=tuple(strides)),
        args=(operand, offsets),
    )
    handler = access_relation_registry.lookup(SliceOp)
    relations = handler(
        call,
        _TypesOnly(
            {id(operand): source, id(call): result},
            local={id(operand): held, id(call): result},
        ),
    )
    return relations.inputs[0].pattern


class _TypesOnly:
    """The little a relation handler asks of a context: the types it is given."""

    def __init__(self, types: dict, local: dict | None = None) -> None:
        self._types = types
        self._local = types if local is None else local

    def type_of(self, expr):
        found = self._types.get(id(expr))
        return expr.type if found is None else found

    def local_type_of(self, expr):
        found = self._local.get(id(expr))
        return expr.type if found is None else found


def _reached(pattern, extent: int):
    """The coordinates a window reaches, walking every element it has."""
    domain = isl.set(f"{{ [d0] : 0 <= d0 < {extent} }}")
    return domain.apply(pattern.relation)


def test_what_a_boundary_moves_is_what_its_relation_reaches() -> None:
    """A quantity is not a second field to keep in step with a relation.

    Reaching one element from many coordinates moves it once, so a broadcast
    costs its source and not its dependences. Reaching past the coordinates an
    operand has is not reaching, so a relation stated over a bigger container
    still answers for the smaller one it was given.
    """
    walked = "0 <= d0 < 8 and 0 <= d1 < 4"
    replicated = AffineAccess(isl.map(f"{{ [d0, d1] -> [0, d1] : {walked} }}"))
    assert reached_elements(replicated, index_set((1, 4))) == 4, (
        "one row read by eight is one row moved"
    )
    beyond = AffineAccess(isl.map(f"{{ [d0, d1] -> [d0, d1] : {walked} }}"))
    assert reached_elements(beyond, index_set((2, 4))) == 8, (
        "and coordinates the operand does not have are not reached at all"
    )


def test_an_unbound_parameter_settles_at_the_smallest_window_it_allows() -> None:
    """A runtime scalar leaves one number to pick, and the Op says which.

    The window `s` rows wide is any of them, so a single answer has to come from
    somewhere; the smallest the relation itself permits is the one a reader can
    check, and it is the same choice on every boundary of the call -- what was
    written and what was left alone still add up to the cache.
    """
    ctx = TypeInferContext()
    call = Call(
        type=make_tensor_type((2, 16, 4, 8), DType.bf16),
        target=CacheUpdate(),
        args=(
            Var(type=make_tensor_type((2, 16, 4, 8), DType.bf16), name="cache"),
            Var(type=make_tensor_type((), DType.i32), name="cur_pos"),
            Var(type=make_tensor_type((), DType.i32), name="s"),
            Var(type=make_tensor_type((2, 5, 4, 8), DType.bf16), name="new"),
        ),
    )
    relations = relations_of(call, ctx)
    cache = index_set((2, 16, 4, 8))
    per_row, held = 2 * 4 * 8, 2 * 16 * 4 * 8

    written = reached_elements(relations.outputs[0].pattern, cache)
    assert written == per_row, "one row is the smallest window this Op describes"
    kept = reached_elements(relations.inputs[0].pattern, cache)
    assert kept == held - per_row, "and the rest of the cache is what it left"
    assert reached_elements(relations.inputs[2].pattern, index_set(())) == 1, (
        "the row count itself is one number, read once"
    )


def test_a_window_states_its_coefficients_and_binds_what_it_cannot_know() -> None:
    """Every number a window is built from is in the relation, one way or another.

    A start or a stride written down is a coefficient. One that is not is a
    parameter bound to the operand element it is, so whoever restricts the
    relation resolves it rather than guessing, and constrained by what the Op
    guarantees: the last element a window touches is `start + (size - 1) * stride`
    and that has to be a position the axis has.
    """
    static = _slice_relation((10,), (4,), (1,), (2,))
    assert static.parameters == ()
    assert str(_reached(static, 4)) == "{ [i0] : 2 <= i0 <= 5 }"

    strided = _slice_relation((10,), (4,), (2,), (1,))
    assert strided.parameters == ()
    walked = _reached(strided, 4)
    assert int(str(walked.count_val())) == 4, "sizes are elements, not reach"
    assert str(walked.dim_max_val(0)) == "7", "the last of four at stride two from one"

    runtime = _slice_relation((10,), (4,), (2,), ("begin",))
    (name, bound_to), = runtime.parameters
    assert name == "s0"
    assert isinstance(bound_to, Var) and bound_to.name == "begin"
    assert "s0 <= 3" in str(runtime.relation), (
        "extent 10, size 4, stride 2 leaves room for a start of 3"
    )

    factored = _slice_relation((16, 4), (8, 4), (1, 1), (2, 0), split_over=4)
    assert factored.parameters == ()
    assert factored.relation.dim(isl.dim_type.OUT) == 2, (
        "the image names one coordinate per position the layout gave the source"
    )


def test_an_open_extent_constrains_the_start_against_that_extent() -> None:
    """A range is not its own largest value, so the start is bound to it.

    A dimension left open is chosen later, and a start legal at the top of its
    range is not legal further down. So the extent is a parameter too, bound to
    the dimension it is, and the start is constrained against that parameter
    rather than against the one number the range happens to stop at.
    """
    opened = _slice_relation((DimVar("open_n", 1, 33),), (4,), (1,), ("begin",))

    assert [name for name, _value in opened.parameters] == ["n0", "s0"]
    extent, start = (value for _name, value in opened.parameters)
    assert isinstance(extent, DimVar) and extent.name == "open_n"
    assert isinstance(start, Var) and start.name == "begin"

    at_five = opened.relation.intersect_params(isl.set("[n0, s0] -> { : n0 = 5 }"))
    assert "s0 <= 1" in str(at_five), (
        "five positions with a window of four leave room for a start of one"
    )
    at_top = opened.relation.intersect_params(isl.set("[n0, s0] -> { : n0 = 32 }"))
    assert "s0 <= 28" in str(at_top)


def test_an_extent_over_several_dimensions_takes_its_real_interval() -> None:
    """An expression's range is not every dimension at once at one end.

    `n - m + 10` is smallest where `n` is smallest and `m` is largest, so reading
    both at their low end and then both at their high end says the extent is
    exactly ten -- which would fix a parameter that is nothing of the kind. The
    bounds come from the shared dimension arithmetic, which already answers this.
    """
    n, m = DimVar("n", 1, 10), DimVar("m", 1, 10)
    assert dim_range(n - m + 10) == (2, 19)

    mixed = _slice_relation((n - m + 10,), (4,), (1,), ("begin",))

    relation = str(mixed.relation)
    assert "2 <= n0 <= 18" in relation, "the interval the dimensions really allow"
    assert "s0 <= -4 + n0" in relation, "and the start bound against it"


def test_a_window_cannot_begin_before_what_it_reads() -> None:
    """A start below zero is not a smaller window, it is no window.

    Left unconstrained, a negative start describes coordinates the operand does
    not have, and bounding the image afterwards would quietly turn that into a
    smaller legal-looking access instead of refusing it. Both the start written
    down and the one arriving later are held to it.
    """
    written = _slice_relation((10,), (4,), (1,), (-1,))
    assert written.relation.is_empty()

    arriving = _slice_relation((10,), (4,), (1,), ("begin",))
    at_minus_one = arriving.relation.intersect_params(
        isl.set("[s0] -> { : s0 = -1 }")
    )
    assert at_minus_one.is_empty()
    at_zero = arriving.relation.intersect_params(isl.set("[s0] -> { : s0 = 0 }"))
    assert not at_zero.is_empty()


def test_an_axis_may_have_no_positions_at_all() -> None:
    """An extent of zero is an extent, and the window is what fails to fit.

    A dimension declaring that it may be zero is entitled to be zero, so the
    relation must not rule that out to keep itself tidy. What rules it out for a
    window of four is the window: no start puts four elements inside none.
    """
    opened = _slice_relation((DimVar("open_n", 0, 9),), (4,), (1,), ("begin",))

    assert "0 <= n0" in str(opened.relation), "a zero extent is not excluded"
    empty = opened.relation.intersect_params(isl.set("[n0, s0] -> { : n0 = 0 }"))
    assert empty.is_empty(), "four elements do not fit in none"
    fits = opened.relation.intersect_params(isl.set("[n0, s0] -> { : n0 = 8 }"))
    assert not fits.is_empty()


def test_a_window_that_fits_nowhere_reaches_nothing() -> None:
    """A window too big for its axis has no legal start, and says so.

    Clamping the start to zero would answer that it begins at the beginning,
    which is a fact about no program: eight elements do not fit in four however
    they are placed. An empty relation is what reaching nothing looks like.
    """
    for starts in ((0,), ("begin",)):
        oversize = _slice_relation((4,), (8,), (1,), starts)
        assert oversize.relation.is_empty(), (
            "a window that cannot fit was given a start anyway"
        )

    fits = _slice_relation((4,), (4,), (1,), (0,))
    assert not fits.relation.is_empty()


def test_a_field_is_placed_in_the_buffer_that_field_is_in() -> None:
    """Taking the second of two is not taking the first of them.

    Sixteen floats into a thirty-two float value, the second half begins where
    the first one ends. Taking one of them is given bytes of its own, because no
    plan has put it in the tuple's, and it is still the size of the field it
    took rather than of the field beside it.
    """
    result = analyze(
        _TakesAField, _TakesAField.entry_function(), analysis=("compute-cost", "memory")
    )
    taken = next(
        expr
        for expr in postorder(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, TupleGetItem)
    )
    parts = next(
        expr
        for expr in postorder(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, SplitOp)
    )
    fields = get_metadata(parts, BufferAllocationMetadata).fields
    mine = get_metadata(taken, BufferAllocationMetadata).fields

    assert [ref.size for ref in fields] == [16 * 4, 16 * 4]
    assert fields[1].offset == fields[0].offset + 16 * 4
    assert [ref.size for ref in mine] == [fields[1].size], (
        "the field it took, not the one beside it"
    )
    assert mine[0].buffer_id not in {ref.buffer_id for ref in fields}, (
        "and bytes of its own, because no plan has put it in the tuple's"
    )
    assert get_metadata(taken, TrafficMetadata).whole == ()


def _insert_slice_controls(dst_shape, update_shape, offsets):
    """One insert's boundary quantities, at the shapes it was written with."""
    call = Call(
        type=make_tensor_type(dst_shape, DType.f32),
        target=InsertSlice(),
        args=(
            Var(type=make_tensor_type(dst_shape, DType.f32), name="dst"),
            Var(type=make_tensor_type(update_shape, DType.f32), name="upd"),
            offsets,
        ),
    )
    relations = relations_of(call, CostContext())
    return [reached_elements(boundary.pattern) for boundary in relations.inputs]


def test_an_insert_reads_one_number_for_every_axis_it_places_its_window_on() -> None:
    """A window is placed by one number per axis, and N of them is N reads.

    The offsets arrive as one operand whether there is one of them or three, and
    counting the operand rather than the numbers in it charged a rank-3 insert
    for a third of what it read. A bare scalar start is the one case where the
    operand and the number are the same thing.
    """
    scalar = Constant(type=make_tensor_type((), DType.i64), value=2)

    def tuple_of(count):
        held = make_tensor_type((), DType.i64)
        return Tuple(
            type=TupleType(fields=tuple(held for _ in range(count))),
            elements=tuple(Constant(type=held, value=0) for _ in range(count)),
        )

    assert _insert_slice_controls((10,), (4,), scalar) == [10 - 4, 4, 1]
    assert _insert_slice_controls((8, 8), (2, 2), tuple_of(2)) == [64 - 4, 4, 2]
    assert _insert_slice_controls((8, 8, 8), (2, 2, 2), tuple_of(3)) == [512 - 8, 8, 3]


def test_naming_bytes_is_not_reading_them() -> None:
    """A window's bounds address it; a write's offsets are read to place it.

    The two arrive the same way -- scalars beside a tensor -- and mean opposite
    things. A slice's bounds say which bytes the result is, which is addressing
    and moves nothing. An insert is handed an address and reads it to know where
    to put its window. A reader comparing one against the other sees traffic
    appear or vanish where nothing changed.
    """
    held = make_tensor_type((10,), DType.f32)
    start = Constant(type=make_tensor_type((), DType.i64), value=2)
    bounds = Tuple(
        type=TupleType(fields=(make_tensor_type((), DType.i64),)), elements=(start,)
    )
    window = Call(
        type=make_tensor_type((4,), DType.f32),
        target=SliceOp(sizes=(4,), strides=(1,)),
        args=(Var(type=held, name="x"), bounds),
    )
    naming = relations_of(window, CostContext())
    assert [reached_elements(item.pattern) for item in naming.inputs] == [4, 1], (
        "the window it names, and the one number placing it"
    )

    assert _insert_slice_controls((10,), (4,), start) == [10 - 4, 4, 1]


def test_re_indexing_something_unplaced_moves_none_of_it() -> None:
    """Renaming a window nobody placed is still renaming.

    Reading it under other extents covers the same bytes and asks the run for
    nothing, so the re-indexing moves nothing at any level. A reshape is the one
    renaming that is not given bytes of its own, because renaming the axes over
    the same elements in the same order is all it does.
    """
    result = analyze(
        _RenamesAnUnplacedWindowAtNoDistance,
        _RenamesAnUnplacedWindowAtNoDistance.entry_function(),
        analysis=("compute-cost", "memory"),
    )
    windows = [
        expr
        for expr in postorder(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, (SliceOp, Reshape))
    ]
    assert [type(expr.target).__name__ for expr in windows] == ["Slice", "Reshape"]
    opened, flat = (
        get_metadata(expr, BufferAllocationMetadata).fields[0] for expr in windows
    )
    assert flat.buffer_id == opened.buffer_id, "a reshape is where what it renames is"

    assert dict(get_metadata(windows[0], TrafficMetadata).whole)["gmem"] == TrafficBytes(8, 0)
    assert get_metadata(windows[1], TrafficMetadata).whole == (), (
        "re-indexing a value onto its own bytes was charged for moving them"
    )
    reader = [e for e in postorder(result.function.body) if isinstance(e, Call)][-1]
    assert dict(get_metadata(reader, TrafficMetadata).whole)["gmem"] == TrafficBytes(128, 64)


def test_a_value_nobody_can_place_does_not_place_the_one_that_renames_it() -> None:
    """A window whose start is a value reads the start, not the window.

    Where it lands is a number the run supplies, so the occurrence reads that
    number -- one eight-byte scalar -- and names the sixty-four bytes it does
    not move. Charging it the window would say a program that looks at part of
    a tensor copied that part.
    """
    result = analyze(
        _RenamesWhatItCannotPlace,
        _RenamesWhatItCannotPlace.entry_function(),
        analysis=("compute-cost", "memory"),
    )
    windows = [
        expr
        for expr in postorder(result.function.body)
        if isinstance(expr, Call) and isinstance(expr.target, SliceOp)
    ]
    assert len(windows) == 2
    held = [get_metadata(expr, BufferAllocationMetadata) for expr in windows]
    assert held[0].fields[0].buffer_id != held[1].fields[0].buffer_id, (
        "no plan shares these addresses, so each window owns its own"
    )

    opened, into = (dict(get_metadata(expr, TrafficMetadata).whole) for expr in windows)
    assert opened["gmem"] == TrafficBytes(8, 0), "a run-time window was charged its window"
    assert into["gmem"] == TrafficBytes(0, 0), "a window of a window was charged one"

    reader = [e for e in postorder(result.function.body) if isinstance(e, Call)][-1]
    assert dict(get_metadata(reader, TrafficMetadata).whole)["gmem"] == TrafficBytes(64, 32)


def _spec_records(page="analysis.md"):
    """Every dataclass one spec page declares, as name to field list."""
    text = (Path(__file__).resolve().parents[2] / "docs" / "spec" / page).read_text()
    found = {}
    for block in re.findall(r"```python\n(.*?)```", text, re.S):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            fields = [
                (item.target.id, ast.unparse(item.annotation))
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            ]
            if fields:
                found[node.name] = fields
    return found


def test_a_record_the_spec_declares_has_the_fields_it_declares() -> None:
    """The spec states these shapes to be read by people who cannot run them.

    A field the implementation widened and the spec did not says a value is
    invalid that the implementation emits, and a reader writing against the spec
    builds something that breaks on the first program that hits it. A record the
    spec still describes in a shape nobody builds any more is the same thing
    read the other way. Comparing them is what keeps a written shape a shape.
    """
    checked = 0
    pages = {
        "analysis.md": (analysis_records,),
        "visitor-registry.md": (access_relation,),
    }
    for page, modules in pages.items():
        checked += _held_to_the_code(_spec_records(page), modules)
    assert checked > 15, "the spec declared almost nothing to compare against"


def _bare(annotation: str) -> str:
    """One annotation with the quoting and spacing a forward reference adds."""
    return annotation.replace(" ", "").replace("'", "").replace('"', "")


def _held_to_the_code(declared_records, modules) -> int:
    """How many of *declared_records* matched a dataclass one module builds."""
    checked = 0
    for name, declared in sorted(declared_records.items()):
        real = next(
            (found for found in (getattr(item, name, None) for item in modules) if found),
            None,
        )
        if real is None or not dataclass_fields(real):
            continue
        checked += 1
        actual = [
            (item.name, item.type if isinstance(item.type, str) else str(item.type))
            for item in dataclass_fields(real)
        ]
        assert [name for name, _ in declared] == [name for name, _ in actual], name
        for (stated, written), (_held, real_type) in zip(declared, actual):
            assert _bare(written) == _bare(real_type), f"{name}.{stated}"
    return checked






def test_a_program_with_no_addresses_moves_bytes_and_takes_no_stated_time() -> None:
    """What a program moves and how long it takes are different questions.

    Bytes are what the operations say they move, and saying it needs no address;
    a time is a claim about a machine running a placed program, and a program
    whose buffers nobody placed has none to report. Reading the second answer as
    the first is what would make a program that moves megabytes look free.
    """
    result = analyze(
        _NoParallelLevel, _NoParallelLevel.entry_function(), analysis=("compute-cost", "memory")
    )
    function = result.function
    assert get_metadata(function, MemoryMetadata).allocation is None
    moved = get_metadata(function, TrafficMetadata)
    assert moved is not None and moved.whole

    with pytest.raises(AnalysisError):
        analyze(
            _NoParallelLevel, _NoParallelLevel.entry_function(), analysis="performance"
        )


def test_a_control_operand_is_charged_the_leaves_it_reaches() -> None:
    """Which leaf a boundary reaches decides the bytes, because widths differ.

    A tuple of numbers is indexed by one coordinate, so a boundary that lands on
    the second of them owes the second one's width. Charging the first, or the
    whole tuple, is a wrong number at a plausible size -- and a nested tuple is
    the same question with the leaves counted flat.
    """
    narrow = make_tensor_type((), DType.i32)
    wide = make_tensor_type((), DType.i64)
    mixed = TupleType(fields=(narrow, wide))
    nested = TupleType(fields=(narrow, TupleType(fields=(wide, narrow))))

    assert control_leaves(mixed) == 2
    assert control_leaves(nested) == 3
    assert [leaf.dtype for leaf in leaves_of(nested)] == [
        DType.i32,
        DType.i64,
        DType.i32,
    ]

    def reaching(where: str, count: int) -> AffineAccess:
        return AffineAccess(isl.map(f"{{ [d0] -> [l] : 0 <= d0 < 4 and {where} }}"))

    assert reached_leaves(reaching("l = 1", 2), 2) == frozenset({1})
    assert reached_leaves(reaching("0 <= l < 2", 2), 2) == frozenset({0, 1})
    assert reached_leaves(reaching("l = 2", 3), 3) == frozenset({2})

    assert _moved_bytes(mixed, reaching("l = 0", 2)) == 4, "the narrow leaf it took"
    assert _moved_bytes(mixed, reaching("l = 1", 2)) == 8, "and the wide one"
    assert _moved_bytes(mixed, reaching("0 <= l < 2", 2)) == 12, "or both of them"
    assert _moved_bytes(nested, reaching("l = 1", 3)) == 8, (
        "a nested field is one flat leaf, not one field of the top level"
    )


def test_an_evaluator_that_reports_the_wrong_operand_count_is_refused() -> None:
    """Too few and too many are the same mistake, and both have to be caught.

    Rebuilding a cost from its relations pairs each boundary with the direction
    the evaluator reported, and pairing stops at the shorter of the two -- so an
    evaluator reporting one too many would have its extra silently dropped and
    come out the right length.
    """
    call = Call(
        type=make_tensor_type((4,), DType.f32),
        target=Binary(kind=BinaryKind.ADD),
        args=(
            Var(type=make_tensor_type((4,), DType.f32), name="a"),
            Var(type=make_tensor_type((4,), DType.f32), name="b"),
        ),
    )
    for count in (2, 4):
        cost = Cost({}, tuple(TrafficBytes(read=1) for _ in range(count)))
        with pytest.raises(AnalysisError, match=f"cost reports {count} operands"):
            _stated_movement(call, cost, CostContext())


def test_a_value_of_bits_is_charged_the_bytes_it_takes_up() -> None:
    """A boolean is a bit, and part of a byte is still a byte to fetch.

    Counting a packed value as a share of the whole rounds down, so one bool
    would cost nothing and nine of them one byte instead of the two they are
    read in. The width of an element is what says this, rounded up, and a value
    whose width nobody states is refused rather than guessed at.
    """
    mask = make_tensor_type((512,), DType.bool)
    assert tensor_bytes(mask) == 64

    assert [_bytes_for(mask, count) for count in (0, 1, 8, 9, 512)] == [0, 1, 1, 2, 64]

    packed = make_tensor_type((512,), DType.f4e2m1)
    assert [_bytes_for(packed, count) for count in (0, 1, 2, 3, 512)] == [0, 1, 1, 2, 256]

    held = make_tensor_type((512,), DType.f32)
    assert _bytes_for(held, 512) == tensor_bytes(held)
    assert _bytes_for(held, 1) == 4
    assert _bytes_for(held, None) is None
