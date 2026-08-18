"""Compare hard decoder access relations and dependences to hand-written maps.

Real-model analysis proves coverage, not correctness. These tests pin row
reductions, floor-divided expansion, multi-output calls, and data-dependent
gathers at readable dimensions. Expected maps use semantic forms rather than
implementation output, so a formula round-trip cannot validate itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import MISSING
from dataclasses import fields as dataclass_fields
from typing import get_args, get_origin

import isl
import pytest
import torch

import tilefoundry.analysis.api as analysis_api
import tilefoundry.cli.analyze as cli_analyze
from tests.analysis.test_analysis_families import _oversized_working_set
from tests.evaluator.eval_utils import EvalCase, run_eval_case
from tests.fixtures.logical.authored_constraint import AuthoredConstraint
from tests.fixtures.logical.gqa_static import static_online_attend
from tests.fixtures.placed.flash_split_k_decode import FlashSplitKDecode
from tests.fixtures.placed.moe_mega_kernel import MoEMegaKernel
from tests.fixtures.placed.rmsnorm import RmsnormModule
from tests.fixtures.placed.square_cuda import Model as SquareCuda
from tests.fixtures.shapes.matmul_programs import gemm_rms_norm
from tests.fixtures.shapes.scaled_modules import PairedScaledParent
from tilefoundry import func
from tilefoundry.analysis import (
    AnalysisError,
    LoopFootprintMetadata,
    OccurrenceProvenance,
    TileGraph,
    analyze,
    check_program,
    extract,
)
from tilefoundry.analysis.compute_cost import _prove_storage, _Storage
from tilefoundry.analysis.footprint import _local_type as footprint_local_type
from tilefoundry.analysis.preflight import validate_authored
from tilefoundry.analysis.walk import loop_scopes, postorder, values_of
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
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.nn.layer_norm import LayerNorm
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.nn.relu import ReLU
from tilefoundry.ir.hir.nn.rope import RoPE
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.hir.tensor.cache_update import CacheUpdate
from tilefoundry.ir.hir.tensor.concat import Concat
from tilefoundry.ir.hir.tensor.index_select import IndexSelect
from tilefoundry.ir.hir.tensor.insert_slice import InsertSlice
from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.ir.hir.tensor.reshape import Reshape, is_induction_var_singleton_reshape
from tilefoundry.ir.hir.tensor.split import Split
from tilefoundry.ir.hir.tensor.stack import Stack
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.types import (
    DType,
    TensorType,
    TupleType,
    make_tensor_type,
    tensor_bytes,
)
from tilefoundry.ir.types.shard import Topology
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
    identity_relations,
    linearized_view,
    moves,
    register_access_relation,
    storage_effect_of,
    transfers,
    writes,
)
from tilefoundry.visitor_registry.contexts import Cost, CostContext, FunctionScope, TrafficBytes
from tilefoundry.visitor_registry.relation_build import identity_access

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
        source_operand=2, index_operand=4, source_axis=0
    )
    assert relation.inputs[3].pattern == IndexedAccess(
        source_operand=3, index_operand=4, source_axis=0
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
        [12, 12, 1],
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
        assert _as_map(link.source).is_equal(
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
    flush against a boundary is still its own size. What the offset does change
    is the pattern, which is where a reader looks to find out.
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
    assert top[:2] == middle[:2] == bottom[:2] == ([12, 12, 1], [12])
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


def test_the_legacy_claim_says_what_the_links_say() -> None:
    """Two shapes of one fact, and the older one is derived rather than written.

    The whole-Call claim predates per-boundary links and still has consumers.
    Keeping both by hand is how they drift, so a claim is held to the links: it
    may not name an operand, or a kind, that no link of that output states.

    The converse does not hold. A link is a candidate -- these bytes may be
    shared -- while the claim is a conclusion the handler reaches only when
    placement and size already agree. A reshard across levels states its link
    and no claim, correctly. The allocation is what retires the difference.
    """
    checked = set()
    for owner in (MoEMegaKernel, FlashSplitKDecode, SquareCuda):
        for function in owner.functions:
            ctx = CostContext(scope=FunctionScope(owner, function))
            for expr in postorder(function.body):
                if not isinstance(expr, Call):
                    continue
                handler = access_relation_registry.lookup(type(expr.target))
                if handler is None:
                    continue
                relations = handler(expr, ctx)
                stated = relations.storage_effect
                if stated is None:
                    continue
                derived = storage_effect_of(relations)
                name = type(expr.target).__name__
                checked.add(name)
                assert derived is not None, (
                    f"{name} claims {stated.kind.value} storage in operands "
                    f"{stated.operands}, and no link of its output says so"
                )
                assert (stated.kind, stated.operands) == (
                    derived.kind,
                    derived.operands,
                ), f"{name}: {stated!r} against {derived!r}"
    assert {"Reshard", "Slice"} <= checked


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

    beyond = StorageLink("forward", 3, reads, reads, held)
    refused(
        "output 0 links to operand 3, and this call has 1",
        inputs=(moves(reads, 4),),
        outputs=(transfers(reads, held, beyond),),
    )

    crossed = StorageLink("forward", 0, identity_access(2), reads, held)
    refused(
        "output 0 links two patterns over different domains",
        inputs=(moves(reads, 4),),
        outputs=(transfers(reads, held, crossed),),
    )

    with pytest.raises(ValueError, match="a link either forwards a value"):
        StorageLink("borrow", 0, reads, reads, held)


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
                    StorageLink(
                        "forward", 0, identity_access(1), identity_access(1), held
                    ),
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

    register_access_relation(_ReachesBoth)(identity_relations(2, _both))
    register_access_relation(_ReachesOne)(identity_relations(2, _one))

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
