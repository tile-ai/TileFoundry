"""Compare hard decoder access relations and dependences to hand-written maps.

Real-model analysis proves coverage, not correctness. These tests pin row
reductions, floor-divided expansion, multi-output calls, and data-dependent
gathers at readable dimensions. Expected maps use semantic forms rather than
implementation output, so a formula round-trip cannot validate itself.
"""

from __future__ import annotations

import isl

from tilefoundry import func
from tilefoundry.analysis import TileGraph, extract
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- op names resolved dynamically
from tilefoundry.ir.core import Call, TypeInferContext, Var
from tilefoundry.ir.hir.nn.rope import RoPE
from tilefoundry.ir.hir.tensor.argmax import ArgMax
from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.ir.hir.tensor.topk import TopK
from tilefoundry.ir.types import DType, make_tensor_type
from tilefoundry.visitor_registry.access_relation import (
    OPAQUE,
    AccessRelations,
    access_relation_registry,
)

REPEATS = 4
B, S, H, D = 1, 5, 2, 3


HQ, HKV, HEAD_DIM, MAX_POS = 16, 8, 128, 8


@func
def gemm_rmsnorm(
    x: Tensor[(2, 4), "f32"],
    w: Tensor[(4, 2), "f32"],
    weight: Tensor[(2,), "f32"],
) -> Tensor[(2, 2), "f32"]:
    h = matmul(x, w)  # noqa: F405
    y = rms_norm(h, weight)  # noqa: F405
    return y


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
    tg = extract(gemm_rmsnorm)
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
    tg = extract(gemm_rmsnorm)

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
    assert isinstance(logits.inputs[0], isl.map)

    assert len(logits.outputs) == 2
    assert all(isinstance(item, isl.multi_aff) for item in logits.outputs)

    picked = _relations(ArgMax(), (1, 151936))
    assert isinstance(picked.inputs[0], isl.map)
    assert len(picked.outputs) == 1


def test_a_relation_says_which_operands_it_cannot_describe() -> None:
    """OPAQUE is a statement, not a gap.

    At this level a rotation's tables are indexed by data, so the relation
    reports them as opaque rather than inventing an affine access for them; q and
    k it does describe, and it describes both of its outputs. Everything
    downstream depends on being able to tell "unknown" from "identity", and a
    relation that returned an identity for an unknown access would be believed.
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
    assert isinstance(relation.inputs[0], isl.multi_aff)
    assert isinstance(relation.inputs[1], isl.multi_aff)
    assert relation.inputs[2:] == (OPAQUE, OPAQUE, OPAQUE)
    assert len(relation.outputs) == 2
    assert all(isinstance(item, isl.multi_aff) for item in relation.outputs)


def test_a_quantised_scale_is_written_once_per_group() -> None:
    """One scale per group of the quantised axis, which is many-to-one.

    So the scale's own map is an `isl.map` carrying the group size, not an
    identity: `128` elements of the last axis share one entry. A relation that
    made the scale an identity would claim a scale per element and price the
    quantisation as no saving at all.
    """
    relation = _relations(Quant(group=128), (1, 2048))

    assert len(relation.inputs) == 1
    assert isinstance(relation.inputs[0], isl.multi_aff)

    assert len(relation.outputs) == 2
    scale = relation.outputs[1]
    assert isinstance(scale, isl.map)
    assert "128" in str(scale)
