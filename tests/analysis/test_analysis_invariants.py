"""Compare hard decoder access relations and dependences to hand-written maps.

Real-model analysis proves coverage, not correctness. These tests pin row
reductions, floor-divided expansion, multi-output calls, and data-dependent
gathers at readable dimensions. Expected maps use semantic forms rather than
implementation output, so a formula round-trip cannot validate itself.
"""

from __future__ import annotations

import ast
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
import tilefoundry.analysis.traffic as analysis_traffic
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
from tilefoundry.analysis.compute_cost import _prove_storage, _Storage
from tilefoundry.analysis.footprint import _local_type as footprint_local_type
from tilefoundry.analysis.memory import _record_traffic
from tilefoundry.analysis.metadata import (
    BufferAllocationMetadata,
    BufferRef,
    ComputeCostMetadata,
    MemoryMetadata,
)
from tilefoundry.analysis.preflight import validate_authored
from tilefoundry.analysis.traffic import (
    TrafficMetadata,
    _moved_bytes,
    lower_traffic,
)
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
from tilefoundry.ir.core.kinds import BinaryKind, ReduceKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.cuda.nn.mma import Mma_SM80_16x8x16, Wgmma_SM90_64x128x16
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
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
from tilefoundry.ir.types.shard import Topology, make_mesh
from tilefoundry.ir.types.shard.shard_layout import Broadcast, Partial
from tilefoundry.ir.types.shard.shard_layout import Split as ShardSplit
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.schedule import ScheduleError, schedule
from tilefoundry.schedule.partition import PartitionProgramError, build_partition_program
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessMode,
    AccessQuantity,
    AccessRelations,
    BoundaryAccess,
    IndexedAccess,
    OperandValue,
    OutputStorage,
    StorageEffectClaim,
    StorageEffectKind,
    StorageLink,
    StorageSpan,
    access_relation_registry,
    build_relation,
    linearized_view,
    moves,
    register_access_relation,
    transfers,
    writes,
)
from tilefoundry.visitor_registry.contexts import Cost, CostContext, FunctionScope, TrafficBytes
from tilefoundry.visitor_registry.relation_build import identity_access
from tilefoundry.visitor_registry.visitors import TypeInferVisitor


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

    One iteration domain cannot cover both, so the call lifts into two
    statements, each over its own head count, each writing its own output buffer
    (the `_{index}` suffix any multi-output op's outputs take). The two are
    independent, so no dependence may appear between them -- an edge here would
    serialise two rotations that have nothing to say to each other.
    """
    tg = extract(rope_gqa)

    assert {u.name: type(u.op.target).__name__ for u in tg.units} == {
        "RoPE_q": "RoPE",
        "RoPE_k": "RoPE",
    }
    expected = isl.union_set("{}")
    for name, heads in (("RoPE_q", HQ), ("RoPE_k", HKV)):
        expected = expected.union(
            isl.set(
                f"{{ {name}[d0,d1,d2,d3] : 0<=d0<1 and 0<=d1<4 "
                f"and 0<=d2<{heads} and 0<=d3<{HEAD_DIM} }}"
            )
        )
    assert tg.domain.is_equal(expected)

    writes = isl.union_map("{}")
    for name, heads, buffer in (("RoPE_q", HQ, "rope_0"), ("RoPE_k", HKV, "rope_1")):
        writes = writes.union(
            isl.map(
                f"{{ {name}[d0,d1,d2,d3] -> {buffer}[d0,d1,d2,d3] : "
                f"0<=d0<1 and 0<=d1<4 and 0<=d2<{heads} and 0<=d3<{HEAD_DIM} }}"
            )
        )
    assert tg.writes.is_equal(writes)
    assert tg.deps.is_equal(isl.union_map("{}"))


def test_a_rotation_reads_its_tables_at_the_position_and_not_at_random() -> None:
    """V1 decodes at `pos_ids == arange(seq)`, so the table selection is affine.

    That assumption is what lets a rotation be modelled at all: `cos[pos[s]]`
    degenerates to `cos[s]`, broadcast over batch and heads. Asserted on both
    branches, because the formula is the same and only the surrounding head
    extent differs -- a relation that read the tables per head, or per element of
    the wrong axis, would still produce a report.
    """
    tg = extract(rope_gqa)

    for name, heads in (("RoPE_q", HQ), ("RoPE_k", HKV)):
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


def test_reshard_relation_rejects_redistribution_but_accepts_storage_moves() -> None:
    function = check_program(SquareCuda, SquareCuda.entry_function())
    reshards = [
        expr
        for expr in postorder(function.body)
        if isinstance(expr, Call) and isinstance(expr.target, Reshard)
    ]
    assert len(reshards) == 2

    redistribution, storage_move = reshards
    redistribution_inputs = tuple(
        footprint_local_type(arg.type) for arg in redistribution.args
    )
    with pytest.raises(
        NotImplementedError,
        match=r"cross-position redistribution changes the local shape from \(168,\) to \(1,\)",
    ):
        build_relation(redistribution, redistribution_inputs, TypeInferContext())

    storage_inputs = tuple(footprint_local_type(arg.type) for arg in storage_move.args)
    assert storage_inputs[0].shape == (1,)
    relation = build_relation(storage_move, storage_inputs, TypeInferContext())
    assert relation is not None
    assert relation.domain.is_equal(isl.set("{ [d0 = 0] }"))
    assert len(relation.maps) == 2
    assert all(item.is_equal(isl.map("{ [d0] -> [d0] }")) for item in relation.maps)


def test_forward_relations_distinguish_layer_norm_from_elementwise() -> None:
    x_type = make_tensor_type((2, 3, 4), DType.f32)
    affine_type = make_tensor_type((3, 4), DType.f32)
    x = Var(type=x_type, name="x")
    weight = Var(type=affine_type, name="weight")
    bias = Var(type=affine_type, name="bias")
    layer_norm = Call(
        type=x_type,
        target=LayerNorm(axis=1, eps=1e-5),
        args=(x, weight, bias),
    )

    relation = build_relation(
        layer_norm, (x_type, affine_type, affine_type), TypeInferContext()
    )
    assert relation is not None
    assert relation.domain.is_equal(isl.set("{ [d0] : 0 <= d0 < 2 }"))
    row = isl.map("{ [d0] -> [d0, j0, j1] : 0 <= j0 < 3 and 0 <= j1 < 4 }")
    affine = isl.map(
        "{ [d0] -> [j0, j1] : 0 <= j0 < 3 and 0 <= j1 < 4 }"
    )
    assert len(relation.maps) == 4
    assert relation.maps[0].is_equal(row)
    assert relation.maps[1].is_equal(affine)
    assert relation.maps[2].is_equal(affine)
    assert relation.maps[3].is_equal(row)

    relu = Call(type=x_type, target=ReLU(), args=(x,))
    elementwise = build_relation(relu, (x_type,), TypeInferContext())
    assert elementwise is not None
    identity = isl.map("{ [d0, d1, d2] -> [d0, d1, d2] }")
    assert elementwise.domain.is_equal(
        isl.set(
            "{ [d0, d1, d2] : 0 <= d0 < 2 and 0 <= d1 < 3 and 0 <= d2 < 4 }"
        )
    )
    assert len(elementwise.maps) == 2
    assert all(item.is_equal(identity) for item in elementwise.maps)


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


def _relations(target, shape, *args) -> AccessRelations:
    """The registered relation of one op, at the black-box (whole-call) level."""
    operands = (Var(type=make_tensor_type(shape, DType.bf16), name="x"), *args)
    call = Call(type=make_tensor_type(shape, DType.bf16), target=target, args=operands)
    handler = access_relation_registry.lookup(type(target))
    assert handler is not None
    return handler(call, TypeInferContext())


def test_a_data_dependent_read_is_a_relation_and_not_a_function() -> None:
    """A scan over an axis cannot be an `isl.multi_aff`, and must not claim to be.

    `topk` and `argmax` each read every element of the axis they scan to produce
    one output, so the read is one-to-many: an `isl.map`. The distinction is the
    whole safety property -- a `multi_aff` is a function, and a consumer that
    took one would conclude each output element depends on a single input
    element, then happily tile the axis a scan cannot be tiled along.
    """
    logits = _relations(TopK(k=8), (1, 128))
    assert len(logits.inputs) == 1
    assert isinstance(logits.inputs[0].pattern, isl.map)
    assert len(logits.outputs) == 2
    assert all(isinstance(item.pattern, isl.multi_aff) for item in logits.outputs)

    picked = _relations(ArgMax(), (1, 151936))
    assert isinstance(picked.inputs[0].pattern, isl.map)
    assert len(picked.outputs) == 1


def test_a_relation_says_how_a_data_dependent_operand_is_read() -> None:
    """A table read through positions is a lookup, not an unknown.

    At this level a rotation's tables are indexed by data, so the relation names
    the operand those indices come from and the operand being indexed. Which
    entry it lands on is not known here; telling a lookup from an identity is
    the safety property, because a relation that returned an identity for a
    lookup would be believed.
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

    assert len(relation.inputs) == 5
    assert isinstance(relation.inputs[0].pattern, isl.multi_aff)
    assert isinstance(relation.inputs[1].pattern, isl.multi_aff)
    assert relation.inputs[2].pattern == IndexedAccess(
        index_operand=4, axis=0
    )
    assert relation.inputs[3].pattern == IndexedAccess(
        index_operand=4, axis=0
    )
    assert isinstance(relation.inputs[4].pattern, isl.multi_aff)
    assert len(relation.outputs) == 2
    assert all(isinstance(item.pattern, isl.multi_aff) for item in relation.outputs)


def test_every_boundary_states_the_movement_its_op_performs() -> None:
    """Hand-counted, boundary by boundary, because nothing derives this.

    A pattern says where a value came from and a quantity says how much crossed;
    the second does not follow from the first. Every output of a scan depends on
    the whole input the scan reads once, and a matrix product's result domain
    says nothing about how big its operands were. So each Op states its own, and
    each is checked here against a number worked out by hand.
    """

    def counted(relations) -> tuple[list[int], list[int]]:
        return (
            [item.quantity.upper for item in relations.inputs],
            [item.quantity.upper for item in relations.outputs],
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
    assert counted(rotated) == ([40, 40, 20, 20, 5], [40, 40])

    ctx = TypeInferContext()
    joined = Call(
        type=make_tensor_type((7, 4), DType.bf16),
        target=Concat(axis=0),
        args=(
            Var(type=make_tensor_type((3, 4), DType.bf16), name="head"),
            Var(type=make_tensor_type((4, 4), DType.bf16), name="tail"),
        ),
    )
    assert counted(access_relation_registry.lookup(Concat)(joined, ctx)) == (
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
    assert counted(access_relation_registry.lookup(IndexSelect)(gathered, ctx)) == (
        [24, 3],
        [24],
    )

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
    assert counted(access_relation_registry.lookup(InsertSlice)(written, ctx)) == (
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
    assert counted(access_relation_registry.lookup(CacheUpdate)(cached, ctx)) == (
        [832, 1, 1, 192],
        [192],
    )


def _as_map(pattern) -> "isl.map":
    """One comparable carrier, whichever of the two affine forms was stated."""
    return pattern if isinstance(pattern, isl.map) else isl.map.from_multi_aff(pattern)


@dataclass(frozen=True)
class _WholeWeight:
    """A context answering with the program's weight, whatever is asked."""

    held: object

    args: tuple = (None, None, None)

    def type_of(self, _arg) -> object:
        return self.held


def _counted(relations) -> tuple[list[int], list[int]]:
    """Every boundary's stated amount, inputs then outputs."""
    return (
        [item.quantity.upper for item in relations.inputs],
        [item.quantity.upper for item in relations.outputs],
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
    return access_relation_registry.lookup(type(op))(call, ctx or TypeInferContext())


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
    "axis 1" by position would reduce the mesh factor of axis 0 and carry its
    residual through untouched -- and the amount can come out right while that
    happens, which is why the map is what this asserts.
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
    relations = access_relation_registry.lookup(Reduce)(
        call, CostContext(level="thread", topologies=(Topology("thread", 6 * 32),))
    )
    reads = relations.inputs[0].pattern
    assert reads.dim(isl.dim_type.OUT) == 3
    assert reads.is_equal(isl.map("{ [d0, d1, d2] -> [0, d1, 0] }"))


def _reduced(type_, axes, keepdim, level=None, topologies=()):
    """One Reduce's relations, with the result its own type inference derives."""
    held = Var(type=type_, name="x")
    target = Reduce(axes=axes, keepdim=keepdim, kind=ReduceKind.SUM)
    inferred = TypeInferVisitor(TypeInferContext()).visit(
        Call(type=type_, target=target, args=(held,))
    )
    call = Call(type=inferred, target=target, args=(held,))
    return access_relation_registry.lookup(Reduce)(
        call, CostContext(level=level, topologies=topologies)
    ).inputs[0].pattern


def test_a_reduction_without_keepdim_names_the_axis_that_survived() -> None:
    """A result coordinate names the logical axis it kept, not a position.

    Dropping the reduced axes shifts the survivors down, so a result coordinate
    and a source axis at the same number are different axes. Once a layout
    factors things the two are not even the same count: reducing axis 1 of a
    `(4,8,5)` whose axis 1 is split leaves the last logical axis at position 3,
    and reading it from position 1 takes the mesh factor of a different axis.
    """
    assert _reduced(make_tensor_type((2, 3, 4), DType.f32), (1,), False).is_equal(
        isl.map("{ [d0, d1] -> [d0, r1, d1] : 0 <= r1 < 3 }")
    )
    assert _reduced(make_tensor_type((2, 3, 4), DType.f32), (0, 2), False).is_equal(
        isl.map("{ [d0] -> [r0, d0, r2] : 0 <= r0 < 2 and 0 <= r2 < 4 }")
    )
    assert _reduced(make_tensor_type((2, 3, 4), DType.f32), (1, 2), False).is_equal(
        isl.map("{ [d0] -> [d0, r1, r2] : 0 <= r1 < 3 and 0 <= r2 < 4 }")
    )

    cta = Topology("cta", 2)
    split = make_shard_tensor_type(
        (4, 8, 5),
        mesh=make_mesh((2,), ("c",), topology=cta),
        attrs=(ShardSplit(1),),
        dtype=DType.f32,
    )
    for keepdim in (False, True):
        assert _reduced(split, (1,), keepdim, "cta", (cta,)).is_equal(
            isl.map("{ [d0, d1, d2, d3] -> [d0, 0, r1, d3] : 0 <= r1 < 4 }")
        )


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
        relations = access_relation_registry.lookup(type(op))(
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
    the group offset the type relation already uses.
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
    assert spanning.inputs[0].pattern.is_equal(
        isl.map(
            "{ [n, co, oh, ow, ci, kh, kw] -> "
            "[0, floor(co/16) * 8 + ci, oh + kh - 1, ow + kw - 1] : "
            "0 <= oh + kh - 1 < 8 and 0 <= ow + kw - 1 < 8 }"
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

    The index answers the first, so that side is a lookup. It answers nothing
    about the second: the whole destination is preserved through one affine
    identity link, because these are the same bytes whichever rows get
    overwritten. A repeated index writes one row twice and an out-of-order one
    writes no window at all, and neither changes a quantity -- there is no
    subtraction left to go negative.
    """
    shapes = (
        make_tensor_type((4, 8), DType.f32),
        make_tensor_type((2,), DType.i64),
        make_tensor_type((2, 8), DType.f32),
    )
    copied = _asked(IndexCopy(dim=0), shapes[0], *shapes)
    assert _counted(copied) == ([32, 2, 16], [16])
    assert copied.inputs[0].mode is AccessMode.TRANSFER
    assert isinstance(copied.inputs[2].pattern, isl.multi_aff)

    added = _asked(IndexAdd(dim=0), shapes[0], *shapes)
    assert _counted(added) == ([16, 2, 16], [16])
    assert added.inputs[0].mode is AccessMode.READ
    assert added.inputs[0].pattern == IndexedAccess(index_operand=1, axis=0)

    for relations in (copied, added):
        assert relations.outputs[0].pattern == IndexedAccess(index_operand=1, axis=0)
        (link,) = relations.outputs[0].storage.links
        assert link.kind == "preserve" and link.quantity == AccessQuantity(32, 32)
        assert isinstance(link.where, isl.multi_aff)


def test_a_view_and_the_value_it_renames_state_the_same_coordinates() -> None:
    """The repros I claimed and did not check: a link that must compare equal.

    A forward link is honoured when it reads a coordinate and answers with the
    one holding it, so a view whose link says otherwise is a copy however
    plainly it is a renaming. Each of these was said to be fixed a round before it was, because
    the tests covered a different shape than the claim did.
    """
    cta = Topology("cta", 2)
    mesh = make_mesh((2,), ("c",), topology=cta)

    def split(shape):
        return make_shard_tensor_type(
            shape, mesh=mesh, attrs=(ShardSplit(0),), dtype=DType.f32
        )

    def linked(op, source):
        held = Var(type=source, name="x")
        inferred = TypeInferVisitor(TypeInferContext()).visit(
            Call(type=source, target=op, args=(held,))
        )
        call = Call(type=inferred, target=op, args=(held,))
        relations = access_relation_registry.lookup(type(op))(
            call, CostContext(level="cta", topologies=(cta,))
        )
        return relations.outputs[0].storage.links[0]

    held = split((8,))
    same = linked(Reshard(layout=held.layout, storage=held.storage), held)
    assert _as_map(same.where).is_equal(isl.map("{ [d0, d1] -> [0, d1] }"))

    turned = linked(Transpose(perm=(0, 1)), split((8, 4)))
    assert _as_map(turned.where).is_equal(isl.map("{ [d0, d1, d2] -> [0, d1, d2] }"))

    parted = access_relation_registry.lookup(SplitOp)(
        _split_call(split((8, 4))), CostContext(level="cta", topologies=(cta,))
    )
    for field, offset in enumerate((0, 2)):
        (link,) = parted.outputs[field].storage.links
        assert _as_map(link.where).is_equal(
            isl.map(f"{{ [d0, d1, d2] -> [0, d1, d2 + {offset}] }}")
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


def test_a_window_is_stated_in_the_positions_its_layout_made() -> None:
    """A window is per logical axis and a projected Type is per position.

    The two lengths disagree the moment anything is split, and nothing was
    checking it: a window of two offsets rode along beside an image of three
    coordinates, and a cache reported the row count its layout happened to put
    at position one. The row axis keeps its own extent while a batch split
    shrinks the rest, which is what tells the two apart.
    """
    cta = Topology("cta", 2)
    mesh = make_mesh((2,), ("c",), topology=cta)

    def split(shape, dtype=DType.f32):
        return make_shard_tensor_type(
            shape, mesh=mesh, attrs=(ShardSplit(0),), dtype=dtype
        )

    i64 = make_tensor_type((), DType.i64)
    offsets = Tuple(
        type=i64,
        elements=(Constant(type=i64, value=0), Constant(type=i64, value=3)),
    )
    written = InsertSlice()
    held = tuple(
        Var(type=type_, name=name)
        for type_, name in ((split((4, 8)), "dst"), (split((4, 2)), "upd"))
    )
    inferred = TypeInferVisitor(TypeInferContext()).visit(
        Call(type=held[0].type, target=written, args=(*held, offsets))
    )
    call = Call(type=inferred, target=written, args=(*held, offsets))
    whole, unit = (
        access_relation_registry.lookup(InsertSlice)(
            call, CostContext(level=level, topologies=(cta,))
        )
        for level in (None, "cta")
    )
    assert (whole.inputs[0].pattern.offsets, whole.inputs[0].pattern.extents) == (
        (0, 3),
        (4, 2),
    )
    assert (unit.inputs[0].pattern.offsets, unit.inputs[0].pattern.extents) == (
        (0, 0, 3),
        (1, 2, 2),
    )

    cache = CacheUpdate()
    i32 = make_tensor_type((), DType.i32)
    rows = (
        Var(type=split((4, 16, 2, 8), DType.bf16), name="cache"),
        Constant(type=i32, value=0),
        Constant(type=i32, value=4),
        Var(type=split((4, 4, 2, 8), DType.bf16), name="new"),
    )
    inferred = TypeInferVisitor(TypeInferContext()).visit(
        Call(type=rows[0].type, target=cache, args=rows)
    )
    call = Call(type=inferred, target=cache, args=rows)
    whole, unit = (
        access_relation_registry.lookup(CacheUpdate)(
            call, CostContext(level=level, topologies=(cta,))
        )
        for level in (None, "cta")
    )
    assert whole.inputs[0].pattern.extents == (4, 4, 2, 8)
    assert whole.inputs[3].quantity.upper == 256
    assert unit.inputs[0].pattern.extents == (1, 2, 4, 2, 8)
    assert unit.inputs[3].quantity.upper == 128


def test_a_boundary_reads_the_coordinates_its_op_actually_touches() -> None:
    """A quantity says how much crossed; the pattern says from where.

    The amount can be right while the coordinates are wrong, and a dependence
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
    relations = access_relation_registry.lookup(Stack)(stacked, ctx)
    assert _as_map(relations.inputs[0].pattern).is_equal(
        isl.map("{ [d0, d1, d2] -> [d1, d2] : d0 = 0 }")
    )
    assert _as_map(relations.inputs[1].pattern).is_equal(
        isl.map("{ [d0, d1, d2] -> [d1, d2] : d0 = 1 }")
    )
    assert _as_map(relations.outputs[0].pattern).is_equal(
        isl.map("{ [d0, d1, d2] -> [d0, d1, d2] }")
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
    relations = access_relation_registry.lookup(Split)(parted, ctx)
    assert _as_map(relations.inputs[0].pattern).is_equal(
        isl.map("{ [d0, d1] -> [d0, d1] }")
    )
    for field, offset in enumerate((0, 3)):
        assert _as_map(relations.outputs[field].pattern).is_equal(
            isl.map("{ [d0, d1] -> [d0, d1] }")
        )
        (link,) = relations.outputs[field].storage.links
        assert link.input == 0 and link.input_field is None
        assert _as_map(link.where).is_equal(
            isl.map(f"{{ [d0, d1] -> [d0 + {offset}, d1] }}")
        )
        assert link.quantity == AccessQuantity(15, 15)

    added = Call(
        type=make_tensor_type((4, 8), DType.f32),
        target=Binary(kind=BinaryKind.ADD),
        args=(
            Var(type=make_tensor_type((4, 8), DType.f32), name="whole"),
            Var(type=make_tensor_type((8,), DType.f32), name="row"),
        ),
    )
    relations = access_relation_registry.lookup(Binary)(added, ctx)
    assert _as_map(relations.inputs[0].pattern).is_equal(
        isl.map("{ [d0, d1] -> [d0, d1] }")
    )
    assert _as_map(relations.inputs[1].pattern).is_equal(
        isl.map("{ [d0, d1] -> [d1] }")
    )

    held = Call(
        type=make_tensor_type((4, 8), DType.f32),
        target=Binary(kind=BinaryKind.MUL),
        args=(
            Var(type=make_tensor_type((4, 8), DType.f32), name="whole"),
            Var(type=make_tensor_type((4, 1), DType.f32), name="column"),
        ),
    )
    relations = access_relation_registry.lookup(Binary)(held, ctx)
    assert _as_map(relations.inputs[1].pattern).is_equal(
        isl.map("{ [d0, d1] -> [d0, 0] }")
    )


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
        relations = access_relation_registry.lookup(InsertSlice)(call, ctx)
        return (
            [item.quantity.upper for item in relations.inputs],
            [item.quantity.upper for item in relations.outputs],
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
    assert top[2].offsets == (0, 0)
    assert bottom[2].offsets == (2, 0)

    runtime = written(axes(Var(type=make_tensor_type((), DType.i64), name="row"), 0))
    assert runtime[:2] == top[:2]
    assert runtime[2].offsets == (OperandValue(operand=2, element=0), 0)

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
    assert counted[2].offsets == (OperandValue(operand=2, element=0), 0)


def test_an_unbound_row_count_states_its_range_and_where_it_came_from() -> None:
    """An `s` nobody bound is still bounded, and says by what.

    The Op's own contract is that it writes at least one row and no more than
    the fewer of what `new` brought and what the cache holds. That is a range a
    reader can check, and the complement is the same range read from the other
    side; charging the whole cache would be neither. The result says the same
    range: how big the cache is and how much of it this occurrence wrote are
    different numbers, and a boundary is asked for the second.
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
    relations = access_relation_registry.lookup(CacheUpdate)(call, ctx)
    per_row, held = 2 * 4 * 8, 2 * 16 * 4 * 8

    update = relations.inputs[3]
    assert update.quantity.lower == per_row
    assert update.quantity.upper == 5 * per_row
    assert "new supplies" in update.quantity.provenance
    assert update.pattern.extents[1] == OperandValue(operand=2, bound=(1, 5))

    kept = relations.inputs[0]
    assert (kept.quantity.lower, kept.quantity.upper) == (
        held - 5 * per_row,
        held - per_row,
    )
    assert kept.pattern.complement
    assert kept.pattern.offsets[1] == OperandValue(operand=1)
    assert relations.outputs[0].quantity == update.quantity


def test_a_lookups_amount_does_not_move_with_the_values_it_looks_up() -> None:
    """The same index shape reads the same amount, whatever it points at.

    The three index vectors here really are different -- run them and the
    results differ -- and a longer one really does read more. That is the claim
    a lookup makes: which rows it lands on is a runtime fact, how many rows it
    reads is not.
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
        relations = access_relation_registry.lookup(IndexSelect)(call, ctx)
        return tuple(item.quantity.upper for item in relations.inputs)

    assert declared(3) == (12, 3)
    assert declared(6) == (24, 6)


def test_a_field_of_a_tuple_is_named_by_the_link_that_forwards_it() -> None:
    """Which field, not just which operand: the two fields are different sizes.

    A `TopK` produces values and their indices, four bytes an element against
    eight. A link naming only the operand pointed at whichever field came
    first, and the width check that would have caught it was skipped, because a
    tuple has no width of its own. The naming is what is watched here: a
    handler that really states two widths is refused elsewhere, since a
    `TupleGetItem`'s result is inferred from its field and always agrees.
    """
    ctx = TypeInferContext()
    values = make_tensor_type((4,), DType.f32)
    indices = make_tensor_type((4,), DType.i64)
    pair = Var(type=TupleType(fields=(values, indices)), name="picked")

    for field, held in ((0, values), (1, indices)):
        taken = Call(type=held, target=TupleGetItem(index=field), args=(pair,))
        relations = access_relation_registry.lookup(TupleGetItem)(taken, ctx)
        (link,) = relations.outputs[0].storage.links
        assert (link.input, link.input_field) == (0, field)
        assert link.quantity == AccessQuantity(4, 4)




def test_a_claim_no_link_supports_is_refused_when_the_handler_answers() -> None:
    """The older shape of one fact is held to the newer one, on every call.

    A whole-Call claim and per-boundary links say the same thing twice, and two
    hand-written statements drift. Sampling a few models would only catch the
    drift those models happen to reach, so the check is where every handler
    passes: the registration wrapper, once per call.

    The converse is allowed. A link is a candidate -- these bytes may be shared
    -- and a claim is a conclusion the handler reaches only when placement and
    size already agree, so a reshard across levels states its link and no claim.
    """

    class _Overclaims(Op):
        pass

    register_typeinfer(_Overclaims)(lambda call, ctx: make_tensor_type((4,), DType.f32))

    @register_access_relation(_Overclaims)
    def _handler(call, ctx) -> AccessRelations:
        return AccessRelations(
            inputs=(moves(identity_access(1), 4),),
            outputs=(writes(identity_access(1), 4),),
            storage_effect=StorageEffectClaim(StorageEffectKind.FORWARD, (0,)),
        )

    call = Call(
        type=make_tensor_type((4,), DType.f32),
        target=_Overclaims(),
        args=(Var(type=make_tensor_type((4,), DType.f32), name="held"),),
    )
    with pytest.raises(ValueError, match="no link of its output says so"):
        access_relation_registry.lookup(_Overclaims)(call, TypeInferContext())


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


def test_a_relation_that_cannot_be_true_is_refused_where_it_is_written() -> None:
    """An impossible description fails at its author, not at its reader.

    A boundary that says it writes on the input side is a broken handler. A
    consumer that quietly reads it as a read hides the break behind a number
    that looks like every other number. Everything answerable without the Call
    is answered here, naming the side and the index so the handler is findable.
    """
    reads = identity_access(1)
    held = AccessQuantity(4, 4)

    def refused(complaint: str, **kwargs):
        with pytest.raises(ValueError, match=re.escape(complaint)):
            AccessRelations(**kwargs)

    refused(
        "input 0 says it does write",
        inputs=(BoundaryAccess(reads, held, AccessMode.WRITE),),
        outputs=(writes(reads, 4),),
    )
    refused(
        "output 0 says it does read",
        inputs=(moves(reads, 4),),
        outputs=(BoundaryAccess(reads, held, AccessMode.READ),),
    )
    refused(
        "input 0 states where bytes live",
        inputs=(BoundaryAccess(reads, held, AccessMode.READ, OutputStorage()),),
        outputs=(writes(reads, 4),),
    )
    refused(
        "output 0 transfers but names no source",
        inputs=(moves(reads, 4),),
        outputs=(BoundaryAccess(reads, held, AccessMode.TRANSFER, OutputStorage()),),
    )

    beyond = StorageLink("forward", 3, reads, held)
    refused(
        "output 0 links to operand 3, and this call has 1",
        inputs=(moves(reads, 4),),
        outputs=(transfers(reads, held, beyond),),
    )

    with pytest.raises(ValueError, match="a link either forwards a value"):
        StorageLink("borrow", 0, reads, held)


def test_a_link_is_held_to_the_operand_it_names() -> None:
    """Sharing bytes with an operand of another element width cannot be meant.

    One side's coordinate would land inside the other's element. Compared in
    bits, because a bool is one and a packed float four: reading those as "no
    whole number of bytes, so never mind" let a bool share a coordinate with an
    f32. The record cannot see this -- it has no Call and no types -- so the
    registration wrapper, which has both, is where it is caught.
    """

    class _Narrows(Op):
        pass

    register_typeinfer(_Narrows)(
        lambda call, ctx: make_tensor_type((4,), DType.f32)
    )

    @register_access_relation(_Narrows)
    def _handler(call, ctx) -> AccessRelations:
        held = AccessQuantity(4, 4)
        return AccessRelations(
            inputs=(BoundaryAccess(identity_access(1), held, AccessMode.TRANSFER),),
            outputs=(
                transfers(
                    identity_access(1),
                    held,
                    StorageLink("forward", 0, identity_access(1), held),
                ),
            ),
        )

    call = Call(
        type=make_tensor_type((4,), DType.f32),
        target=_Narrows(),
        args=(Var(type=make_tensor_type((4,), DType.i64), name="wider"),),
    )
    with pytest.raises(ValueError, match="whose elements are 64 bits against 32"):
        access_relation_registry.lookup(_Narrows)(call, TypeInferContext())


def test_a_lookup_is_held_to_the_operand_and_the_axis_it_names() -> None:
    """Two non-negative numbers say nothing checkable until there is a Call.

    A lookup states which operand's values choose its coordinates and along
    which axis, and neither can be verified against a record that has no
    operands and no Types. Both are verified where they can be, and a handler
    naming an operand nobody passed or an axis the boundary does not have is
    refused rather than left for whichever consumer indexes past the end.
    """

    def registered(name, pattern_for):
        target = type(name, (Op,), {})
        register_typeinfer(target)(
            lambda call, ctx: make_tensor_type((4,), DType.f32)
        )

        @register_access_relation(target)
        def _handler(call, ctx, _pattern=pattern_for) -> AccessRelations:
            return AccessRelations(
                inputs=(
                    BoundaryAccess(_pattern(), AccessQuantity(4, 4), AccessMode.READ),
                ),
                outputs=(writes(identity_access(1), 4),),
            )

        return target

    def asked(target):
        call = Call(
            type=make_tensor_type((4,), DType.f32),
            target=target(),
            args=(Var(type=make_tensor_type((4,), DType.f32), name="held"),),
        )
        return access_relation_registry.lookup(target)(call, TypeInferContext())

    absent = registered("_LooksUpNobody", lambda: IndexedAccess(index_operand=3, axis=0))
    with pytest.raises(ValueError, match="through operand 3, and this call has 1"):
        asked(absent)

    beyond = registered("_LooksUpTooFar", lambda: IndexedAccess(index_operand=0, axis=2))
    with pytest.raises(ValueError, match="along axis 2, and it has 1"):
        asked(beyond)

    with pytest.raises(ValueError, match="names its index operand by position"):
        IndexedAccess(index_operand=-1, axis=0)
    with pytest.raises(ValueError, match="names its axis by position"):
        IndexedAccess(index_operand=0, axis=-1)


def test_a_link_source_that_names_a_field_is_checked_against_that_field() -> None:
    """A tuple has no rank; the field a link names does, and that is what holds.

    Passing the whole tuple to the axis check made every rank zero, so a lookup
    into a perfectly good tensor field was refused for having an axis. The field
    is resolved the same way the element-width check resolves it.
    """

    def registered(name, axis):
        target = type(name, (Op,), {})
        register_typeinfer(target)(
            lambda call, ctx: make_tensor_type((4,), DType.f32)
        )

        @register_access_relation(target)
        def _handler(call, ctx, _axis=axis) -> AccessRelations:
            held = AccessQuantity(4, 4)
            return AccessRelations(
                inputs=(
                    BoundaryAccess(identity_access(1), held, AccessMode.TRANSFER),
                    moves(identity_access(1), 4),
                ),
                outputs=(
                    transfers(
                        identity_access(1),
                        held,
                        StorageLink(
                            "forward",
                            0,
                            IndexedAccess(index_operand=1, axis=_axis),
                            held,
                            input_field=0,
                        ),
                    ),
                ),
            )

        return target

    def asked(target):
        rows = make_tensor_type((4,), DType.f32)
        call = Call(
            type=rows,
            target=target(),
            args=(
                Var(type=TupleType(fields=(rows, rows)), name="pair"),
                Var(type=make_tensor_type((4,), DType.i64), name="index"),
            ),
        )
        return access_relation_registry.lookup(target)(call, TypeInferContext())

    relations = asked(registered("_LooksIntoAField", 0))
    (link,) = relations.outputs[0].storage.links
    assert link.input_field == 0

    with pytest.raises(ValueError, match="along axis 2, and it has 1"):
        asked(registered("_LooksPastAField", 2))


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
                inputs=tuple(moves(identity_access(1), 4) for _ in range(_in)),
                outputs=tuple(writes(identity_access(1), 4) for _ in range(_out)),
            )

        return target

    def asked(target, operands: int) -> AccessRelations:
        call = Call(
            type=holds,
            target=target(),
            args=tuple(Var(type=holds, name=f"a{index}") for index in range(operands)),
        )
        return access_relation_registry.lookup(target)(call, TypeInferContext())

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


def test_a_storage_claim_covers_every_operand_it_names() -> None:
    """Addressing one operand does not conclude anything about a second.

    A conclusion is read as "the result lives in these operands", and its reader
    retires the movement of each. A handler that names two while proving one
    would have the reader retire bytes nothing was shown about, so the claim is
    refused rather than trimmed to the part that held. An Op with no boundary
    relation still states this: the two are one registration, not one fact.
    """

    class _ReachesBoth(Op):
        pass

    class _ReachesOne(Op):
        pass

    holds = make_tensor_type((4,), DType.f32)
    size = tensor_bytes(holds)
    left = Var(type=holds, name="left")
    right = Var(type=holds, name="right")

    def _both(call: Call, ctx) -> StorageEffectClaim:
        return StorageEffectClaim(StorageEffectKind.FORWARD, (0,), (StorageSpan(0, 0, size),))

    def _one(call: Call, ctx) -> StorageEffectClaim:
        return StorageEffectClaim(StorageEffectKind.FORWARD, (0, 1), (StorageSpan(0, 0, size),))

    def _forwarding(claim, *operands):
        """A handler whose links say what its legacy claim says."""

        def _handler(call, ctx) -> AccessRelations:
            held = AccessQuantity(4, 4)
            reads = identity_access(1)
            return AccessRelations(
                inputs=tuple(
                    BoundaryAccess(reads, held, AccessMode.TRANSFER)
                    for _ in call.args
                ),
                outputs=(
                    transfers(
                        reads,
                        held,
                        *(
                            StorageLink("forward", operand, reads, held)
                            for operand in operands
                        ),
                    ),
                ),
                storage_effect=claim(call, ctx),
            )

        return _handler

    register_access_relation(_ReachesBoth)(_forwarding(_both, 0))
    register_access_relation(_ReachesOne)(_forwarding(_one, 0, 1))

    walk = _Storage(
        type_of=lambda expr: expr.type, users={}, positions={}, caller_owned=frozenset()
    )
    covered = Call(type=holds, target=_ReachesBoth(), args=(left, right))
    assert _prove_storage(covered, walk) == (StorageEffectKind.FORWARD, (0,))

    partial = Call(type=holds, target=_ReachesOne(), args=(left, right))
    assert _prove_storage(partial, walk) is None
    assert id(partial) not in walk.bases


def test_a_quantised_scale_is_written_once_per_group() -> None:
    """One scale per group of the quantised axis, which is many-to-one.

    So the scale's own map is an `isl.map` carrying the group size, not an
    identity: `128` elements of the last axis share one entry. A relation that
    made the scale an identity would claim a scale per element and price the
    quantisation as no saving at all.
    """
    relation = _relations(Quant(group=128), (1, 2048))

    assert len(relation.inputs) == 1
    assert isinstance(relation.inputs[0].pattern, isl.multi_aff)

    assert len(relation.outputs) == 2
    scale = relation.outputs[1].pattern
    assert isinstance(scale, isl.map)
    assert "128" in str(scale)


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


def test_a_rename_that_lands_on_the_same_bytes_moves_none_of_them() -> None:
    """Two names for one address are one value, and copying it costs nothing.

    The proof is the addresses: a link's two sides are composed with the layouts
    of the buffers they name, over the iteration the occurrence runs, and only
    an equality there earns the zero. A reader that took the op's name for it
    would score a reshape between different buffers as free.
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
        assert sum(item.read for item in record.boundaries) == 16384
        assert sum(item.write for item in record.boundaries) == 128

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
    return access_relation_registry.lookup(MatMul)(call, CostContext())


def test_a_contraction_reads_each_operand_at_its_own_extents() -> None:
    """A contraction reaches along the axis it sums, not along the one it keeps.

    Reading each operand at the result's coordinates is right only while every
    extent agrees. For `(128,64) x (64,32)` it claims the left operand has 32
    columns and the right 128 rows, which is the wrong half of both: the shapes
    the map names have to be the shapes the operands have.
    """
    relations = _matmul_relations((128, 64), (64, 32), (128, 32))
    assert relations.inputs[0].pattern.is_equal(
        isl.map("{ [d0, d1] -> [d0, k] : 0 <= k < 64 }")
    )
    assert relations.inputs[1].pattern.is_equal(
        isl.map("{ [d0, d1] -> [k, d1] : 0 <= k < 64 }")
    )
    result = isl.set("{ [d0, d1] : 0 <= d0 < 128 and 0 <= d1 < 32 }")
    assert relations.inputs[0].pattern.intersect_domain(result).range().is_equal(
        isl.set("{ [i0, i1] : 0 <= i0 < 128 and 0 <= i1 < 64 }")
    )
    assert relations.inputs[1].pattern.intersect_domain(result).range().is_equal(
        isl.set("{ [i0, i1] : 0 <= i0 < 64 and 0 <= i1 < 32 }")
    )
    assert [boundary.quantity.upper for boundary in relations.inputs] == [8192, 2048]
    assert relations.outputs[0].quantity.upper == 4096


def test_a_contraction_reads_a_batch_it_broadcasts_at_one_coordinate() -> None:
    """A batch an operand has one of is read there however many the result has."""
    relations = _matmul_relations((1, 128, 64), (4, 64, 32), (4, 128, 32))
    assert relations.inputs[0].pattern.is_equal(
        isl.map("{ [d0, d1, d2] -> [0, d1, k] : 0 <= k < 64 }")
    )
    assert relations.inputs[1].pattern.is_equal(
        isl.map("{ [d0, d1, d2] -> [d0, k, d2] : 0 <= k < 64 }")
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


def test_a_unit_that_holds_none_of_a_value_is_not_a_unit_that_moves_none() -> None:
    """Not running an occurrence and reading what a neighbour holds differ.

    This program gives each expert its own slice of the machine, so a unit
    running one of them does none of the other's work: that is a share of
    nothing, and the occurrence still moves everything it moves. A unit that did
    run it while holding no part of what it reads would be reading between
    units, which nothing here answers, and is refused rather than counted as the
    same nothing.
    """
    result = analyze(
        MoEMegaKernel, MoEMegaKernel.entry_function(), analysis=("compute-cost", "memory")
    )
    function = result.function
    places = _call_placements(MoEMegaKernel, function, "cta")
    elsewhere = [
        expr for expr in _occurrences(function) if 0 not in places.get(id(expr), {0})
    ]
    assert elsewhere, "every occurrence of this program runs on the first participant"
    moved = 0
    for expr in elsewhere:
        record = get_metadata(expr, TrafficMetadata)
        assert record is not None and record.per_unit == ()
        moved += bool(record.whole)
    assert moved, "none of the work this unit skips moved anything at all"

    plan = build_buffer_plan(function, "cta")
    scope = FunctionScope(MoEMegaKernel, function)
    whole = CostContext(scope=scope)
    unit = CostContext(
        scope=scope, level="cta", topologies=MoEMegaKernel.effective_topologies()
    )
    reading = next(
        expr for expr in elsewhere if get_metadata(expr, TrafficMetadata).whole
    )
    handler = access_relation_registry.lookup(type(reading.target))
    with pytest.raises(AnalysisError, match="holds no part of"):
        lower_traffic(
            reading, handler(reading, whole), handler(reading, unit),
            plan, whole, unit, participant=0, runs=True, umat_level="gmem",
        )


def test_an_occurrence_this_cannot_state_carries_no_record_rather_than_an_empty_one() -> None:
    """Saying nothing and saying zero are different, and both get said.

    An Op that states no relations at this shape leaves its occurrence with
    nothing said about it, and an empty record there would claim it moved
    nothing. A program that can be told states a row for every boundary that
    moved and none for those that did not, so its rows and its totals agree.
    """

    class _DeclinesHere(Op):
        pass

    register_typeinfer(_DeclinesHere)(
        lambda call, ctx: make_tensor_type((4,), DType.f32)
    )

    @register_access_relation(_DeclinesHere)
    def _declines(call, ctx) -> AccessRelations:
        raise NotImplementedError("this Op does not state relations at this shape")

    silent, function = _totalled(_DeclinesHere())
    assert get_metadata(silent, TrafficMetadata) is None
    assert get_metadata(function, TrafficMetadata) is None

    result = analyze(
        MoEMegaKernel, MoEMegaKernel.entry_function(), analysis=("compute-cost", "memory")
    )
    stated = _occurrences(result.function)
    assert stated
    moving = 0
    for expr in stated:
        record = get_metadata(expr, TrafficMetadata)
        assert record is not None
        assert all(item.read or item.write for item in record.boundaries)
        assert bool(record.whole) == bool(record.boundaries)
        moving += bool(record.boundaries)
    assert moving, "every occurrence this program states moved nothing"
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


def test_a_handler_that_breaks_its_own_contract_is_not_an_op_nothing_can_be_said_of() -> None:
    """Two silences that mean opposite things are told apart at the boundary.

    An Op whose relation says it cannot state a case is a case this does not
    cover, and the occurrence goes without a record. A handler raising because
    it built something invalid is a bug in the handler, and swallowing it would
    file that bug under the same heading and leave it there.
    """

    class _Breaks(Op):
        pass

    register_typeinfer(_Breaks)(lambda call, ctx: make_tensor_type((4,), DType.f32))

    @register_access_relation(_Breaks)
    def _broken(call, ctx) -> AccessRelations:
        raise ValueError("this handler built something it should not have")

    class _Declines(Op):
        pass

    register_typeinfer(_Declines)(lambda call, ctx: make_tensor_type((4,), DType.f32))

    @register_access_relation(_Declines)
    def _declines(call, ctx) -> AccessRelations:
        raise NotImplementedError("this Op does not state relations at this shape")

    held = make_tensor_type((4,), DType.f32)
    source = Var(type=held, name="held")
    broken = Call(type=held, target=_Breaks(), args=(source,))
    declines = Call(type=held, target=_Declines(), args=(source,))

    for body, expected in ((broken, ValueError), (declines, None)):
        function = Function(
            type=held, name="main", params=(source,), body=body, return_type=held
        )
        scope = FunctionScope(_NoParallelLevel, function)
        if expected is None:
            _record_traffic(function, scope, None, ())
            assert get_metadata(body, TrafficMetadata) is None
            continue
        with pytest.raises(expected, match="should not have"):
            _record_traffic(function, scope, None, ())


def test_a_function_moves_its_occurrences_as_often_as_its_loops_repeat_them() -> None:
    """A total is the bytes counted again for every trip that moves them.

    One occurrence inside a loop of 24 moves what it moves 24 times, and the
    rule for saying so lives here rather than in each reader: two readers with
    two copies of it drift. The boundaries are not repeated with it -- which
    operand moved what belongs to the occurrence, not to the total. The insert
    in that loop reads three eight-byte numbers to place its window, so the
    register total is those three, counted twenty-four times.
    """
    case = next(item for item in CORPUS if item.id == "access_footprint.grouped_moe")
    selected = case.analyze[0]
    owner, entry = case.resolve(case.build(), selected.selector)
    result = analyze(owner, entry, analysis=("compute-cost", "memory"), dims=selected.dims)
    function = result.function

    record = get_metadata(function, TrafficMetadata)
    assert record is not None and record.boundaries == ()

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
    assert dict(record.whole)["rmem"] == TrafficBytes(3 * 8 * 24, 0)


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


def test_a_total_missing_one_of_its_parts_is_not_stated_at_all() -> None:
    """A sum that left something out reads as a smaller program.

    An occurrence nobody can answer for makes the function's total an unknown
    rather than a smaller number, so it is not stated and a reader asking for it
    is told nothing instead of told wrong. An Op that states no accesses at all
    counts the same way when it was going to move something: silence about an
    occurrence that moves bytes is not the same as an occurrence that does not.
    """

    class _Declines(Op):
        pass

    register_typeinfer(_Declines)(lambda call, ctx: make_tensor_type((4,), DType.f32))

    @register_access_relation(_Declines)
    def _declines(call, ctx) -> AccessRelations:
        raise NotImplementedError("this Op does not state relations at this shape")

    class _MovesWithoutSaying(Op):
        pass

    register_typeinfer(_MovesWithoutSaying)(
        lambda call, ctx: make_tensor_type((4,), DType.f32)
    )

    refused, function = _totalled(_Declines())
    assert get_metadata(refused, TrafficMetadata) is None
    assert get_metadata(function, TrafficMetadata) is None

    moving = ComputeCostMetadata(traffic=(("gmem", TrafficBytes(16, 0)),))
    silent, function = _totalled(_MovesWithoutSaying(), moving)
    assert get_metadata(silent, TrafficMetadata) is None
    assert get_metadata(function, TrafficMetadata) is None

    still, function = _totalled(_MovesWithoutSaying(), ComputeCostMetadata())
    assert get_metadata(still, TrafficMetadata) is None
    assert get_metadata(function, TrafficMetadata) == TrafficMetadata()


def test_a_window_is_placed_where_its_own_link_says_it_begins() -> None:
    """A view of a view is somewhere inside the value neither of them copied.

    Both windows were placed in the buffer they name rather than given one of
    their own, which is the whole of what says they moved nothing. Where inside
    it they begin is a question the placement left open, and leaving it open is
    not the same as answering it with the front of the buffer.
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
    assert outer.fields[0].buffer_id == inner.fields[0].buffer_id
    assert [ref.offset for ref in (*outer.fields, *inner.fields)] == [None, None]

    for expr in windows:
        assert get_metadata(expr, TrafficMetadata).whole == ()


def test_a_window_whose_start_is_only_known_at_run_time_is_still_a_window() -> None:
    """Not knowing where a window lands is not knowing that it moved.

    Its start arrives as a value, so nothing here states an address for it. It
    was still placed in the buffer it names rather than given one of its own,
    and a value living in another's bytes did not copy them to get there --
    whoever really reads it is who moves anything.
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
    open_ = [
        expr
        for expr in windows
        if get_metadata(expr, BufferAllocationMetadata).fields[0].offset is None
    ]
    assert open_, "no window of this program had a start only the run knows"
    for expr in windows:
        assert get_metadata(expr, TrafficMetadata).whole == (), (
            "a window was charged for naming bytes it was placed in"
        )


def test_a_field_is_placed_in_the_buffer_that_field_is_in() -> None:
    """Taking the second of two is not taking the first of them.

    Which field a value took decides which bytes it is, and the link that
    forwards it names that field. Sixteen floats into a thirty-two float value,
    the second half begins where the first one ends, and a reader given the
    first would be reading somebody else's numbers at the right size.
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
    assert [(ref.buffer_id, ref.size) for ref in mine] == [
        (fields[1].buffer_id, fields[1].size)
    ]
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
    relations = access_relation_registry.lookup(InsertSlice)(call, CostContext())
    return [boundary.quantity.upper for boundary in relations.inputs]


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
    naming = access_relation_registry.lookup(SliceOp)(window, CostContext())
    assert [item.quantity.upper for item in naming.inputs] == [4, 0]

    assert _insert_slice_controls((10,), (4,), start) == [10 - 4, 4, 1]


def test_re_indexing_something_unplaced_moves_none_of_it() -> None:
    """Renaming a window nobody placed is still renaming.

    Reading it under other extents covers the same bytes, and it was placed in
    the same buffer the window itself was placed in. Neither states an address,
    and neither had to: what says they moved nothing is that neither was given
    a buffer of its own.
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

    assert (opened.offset, flat.offset) == (None, None), (
        "re-indexing an unplaced window invented an address for it"
    )
    assert flat.buffer_id == opened.buffer_id
    for expr in windows:
        assert get_metadata(expr, TrafficMetadata).whole == ()
    reader = [e for e in postorder(result.function.body) if isinstance(e, Call)][-1]
    assert dict(get_metadata(reader, TrafficMetadata).whole)["gmem"].read == 128, (
        "the occurrence that really reads the window was let off with it"
    )


def test_a_value_nobody_can_place_does_not_place_the_one_that_renames_it() -> None:
    """A distance from an unknown address is another unknown address.

    A window whose start arrives as a value is somewhere in the buffer it names
    and nowhere in particular, so measuring the next window from the front of
    that buffer answers with a number wrong by wherever the first really began.
    Neither window states an address, both were placed in what they name, and
    the occurrence that really reads them is the one charged for the bytes.
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
    assert all(item is not None for item in held)
    assert [ref.offset for item in held for ref in item.fields] == [None, None]

    opened, into = (item.fields[0] for item in held)
    assert opened.buffer_id == into.buffer_id, (
        "a window measured from another was given an allocation of its own"
    )
    for expr in windows:
        assert get_metadata(expr, TrafficMetadata).whole == ()
    reader = [e for e in postorder(result.function.body) if isinstance(e, Call)][-1]
    assert dict(get_metadata(reader, TrafficMetadata).whole)["gmem"].read == 64, (
        "the occurrence that really reads the window was let off with it"
    )


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
        "analysis.md": (analysis_records, analysis_traffic),
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


def test_a_link_is_asked_by_as_many_coordinates_as_its_output_has() -> None:
    """A link answers for one coordinate of its output, so it is asked by all of them.

    A map indexed by fewer is a link to some other value, and it composes with
    this occurrence without complaining: the reader finds out only when the
    spaces refuse to meet, by which point the answer has been a conservative
    charge rather than a refused handler for however long.
    """

    class _AsksTooFew(Op):
        pass

    register_typeinfer(_AsksTooFew)(lambda call, ctx: make_tensor_type((2, 2), DType.f32))

    @register_access_relation(_AsksTooFew)
    def _handler(call, ctx) -> AccessRelations:
        held = AccessQuantity(4, 4)
        return AccessRelations(
            inputs=(
                BoundaryAccess(identity_access(1), held, AccessMode.TRANSFER),
            ),
            outputs=(
                transfers(
                    identity_access(2),
                    held,
                    StorageLink("forward", 0, identity_access(1), held),
                ),
            ),
        )

    call = Call(
        type=make_tensor_type((2, 2), DType.f32),
        target=_AsksTooFew(),
        args=(Var(type=make_tensor_type((4,), DType.f32), name="x"),),
    )
    with pytest.raises(ValueError, match="from 1 coordinates, and it has 2"):
        access_relation_registry.lookup(_AsksTooFew)(call, CostContext())


def test_a_link_that_states_both_ranks_is_taken() -> None:
    """The shapes a real op states are the ones the check is meant to admit.

    A concat's link answers for a coordinate of the whole and names one of the
    piece it came from; a split's answers for a coordinate of a piece and names
    one of the whole. Both are indexed by their own output and land in their own
    input, at the same rank, which is what the check asks and no more.
    """
    held = make_tensor_type((6, 5), DType.f32)
    piece = make_tensor_type((3, 5), DType.f32)

    joined = Call(
        type=held,
        target=Concat(axis=0),
        args=(Var(type=piece, name="a"), Var(type=piece, name="b")),
    )
    relations = access_relation_registry.lookup(Concat)(joined, TypeInferContext())
    for index, offset in enumerate((0, 3)):
        (link,) = [
            item
            for item in relations.outputs[0].storage.links
            if item.input == index
        ]
        assert _as_map(link.where).is_equal(
            isl.map(f"{{ [d0, d1] -> [d0 - {offset}, d1] }}")
        )
        assert link.where.dim(isl.dim_type.IN) == 2
        assert link.where.dim(isl.dim_type.OUT) == 2

    parted = access_relation_registry.lookup(SplitOp)(
        _split_call(make_tensor_type((6, 4), DType.f32)), TypeInferContext()
    )
    for field, offset in enumerate((0, 2)):
        (link,) = parted.outputs[field].storage.links
        assert _as_map(link.where).is_equal(
            isl.map(f"{{ [d0, d1] -> [d0, d1 + {offset}] }}")
        )
        assert link.where.dim(isl.dim_type.IN) == 2


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


def test_a_value_of_bits_is_charged_the_bytes_it_takes_up() -> None:
    """A boolean is a bit, and a bit is not a fraction of a byte to charge.

    Charging one element at a time asks what a bit costs in bytes, which has no
    answer, and refusing there loses the traffic of every mask a program
    computes. A leaf is addressed on its own, so the count is converted whole
    and rounded up the way the type system already sizes it.
    """
    mask = make_tensor_type((512,), DType.bool)
    assert tensor_bytes(mask) == 64

    assert _moved_bytes(512, mask) == 64
    assert _moved_bytes(1, mask) == 1
    assert _moved_bytes(0, mask) == 0
    assert _moved_bytes(9, mask) == 2

    held = make_tensor_type((512,), DType.f32)
    assert _moved_bytes(512, held) == tensor_bytes(held)
