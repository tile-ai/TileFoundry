"""Qwen3-1.7B, as the installation ships it, asked through the commands.

The arithmetic checks at the end are this model's alone: its dimensions are public
and its decoder layer is a sum anybody can do on paper, so the numbers the analyses
report are compared against numbers derived from the checkpoint's own config rather
than against a recorded run.
"""

from __future__ import annotations

import json

import contract
import pytest
import torch

from tests.models.qwen3_1_7b import reference
from tests.models.qwen3_1_7b.model import published

MODEL = "qwen3_1_7b"
CASES = contract.model_cases(MODEL)

ANALYSED = [
    pytest.param(case, selected, id=selected.id) for case in CASES for selected in case.analyze
]
PLANNED = [
    pytest.param(case, planned, id=planned.id) for case in CASES for planned in case.schedule
]

FIRST_PLAN = [pytest.param(case, case.schedule[0], id=case.id) for case in CASES]
SIZED = [pytest.param(case, sized, id=sized.id) for case in CASES for sized in case.sized]


ZERO_SIZED = frozenset(("k_cache", "v_cache"))


@pytest.mark.parametrize(("case", "selected"), ANALYSED)
def test_every_selected_function_analyses(tf, shipped_source, case, selected) -> None:
    contract.analysed_every_family(
        tf, shipped_source(MODEL), case, selected.selector, selected.dims
    )


@pytest.mark.parametrize(("case", "planned"), PLANNED)
def test_every_selected_function_plans(tf, shipped_source, case, planned) -> None:
    contract.scheduled(tf, shipped_source(MODEL), case, planned)


@pytest.mark.parametrize(("case", "planned"), FIRST_PLAN)
def test_the_plan_reaches_the_level_below_the_one_it_partitions(
    tf, shipped_source, case, planned
) -> None:
    """The source declares two levels, and this model can be asked for both.

    Nothing here names an extent for either: the copied ``model.py`` carries the
    levels, and a root that stopped declaring ``thread`` leaves this command with
    no level to be given and no test to inject one. Only this model and
    ``qwen2_5_1_5b`` can witness that -- the other roots declare ``thread`` too, but
    their IR reaches ops with no registered access relation, so the command fails for
    a reason that has nothing to do with the level.
    """
    contract.scheduled(tf, shipped_source(MODEL), case, planned, topology="thread")


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_every_analysis_answers_at_the_largest_context(tf, shipped_source, case, sized) -> None:
    """At the ceiling the case states, not at a sample of it."""
    contract.analysed_every_family(tf, shipped_source(MODEL), case, sized.selector, sized.ceiling)


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_the_ceiling_is_reasoned_about_at_its_stated_length(
    tf, shipped_source, case, sized
) -> None:
    """What the analysis reports has to grow with the context.

    Growth rather than an absolute number: an analysis quietly working at a length
    it could afford instead of the one it was asked about would report the same
    footprint at both, and a number nobody compares would not show it.
    """
    source = shipped_source(MODEL)
    short = contract.traffic_read(tf, source, case, sized.selector, sized.dims)
    full = contract.traffic_read(tf, source, case, sized.selector, sized.ceiling)

    assert full > short, (
        f"analysing at {dict(sized.ceiling)} reports no more traffic than at "
        f"{dict(sized.dims)}, so the stated length changed nothing"
    )


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_the_open_dimensions_are_analysed_at_zero(tf, shipped_source, case, sized) -> None:
    """A binding whose whole cost is the context has to cost nothing without one."""
    source = shipped_source(MODEL)
    zero = contract.lifetimes(tf, source, case, sized.selector, {name: 0 for name in sized.dims})
    nonzero = contract.lifetimes(tf, source, case, sized.selector, sized.dims)

    assert ZERO_SIZED <= zero.keys()
    assert all(zero[binding] == 0 for binding in ZERO_SIZED)
    assert all(nonzero[binding] > 0 for binding in ZERO_SIZED)


@pytest.mark.parametrize(("case", "planned"), FIRST_PLAN)
def test_the_command_reports_a_real_model_as_json(tf, shipped_source, case, planned) -> None:
    done = contract.analysed(
        tf,
        shipped_source(MODEL),
        case,
        planned.selector,
        "compute-cost",
        planned.dims,
        json_output=True,
    )

    assert json.loads(done.stdout)


@pytest.mark.parametrize("ctx_len", [0, 24])
def test_the_decode_step_and_the_cache_entry_it_hands_back(
    tf, shipped_source, tmp_path, ctx_len
) -> None:
    """One decode step of one layer, and the state the step hands back.

    Attention through ``o_proj`` and the remaining layer are compared separately
    with ``Qwen3DecoderLayer.forward`` to localize failures. Returned cache entries
    are checked against oracle state rebuilt one token longer, so unchanged inputs
    fail. A zero context covers the first token. All three attention outputs are
    judged.
    """
    drawn = reference.decode_step_inputs(ctx_len=ctx_len, device="cpu")
    source, case = shipped_source(MODEL), CASES[0]
    want_k, want_v = reference.appended_cache_oracle(drawn)
    entry_k, entry_v = want_k[:, ctx_len:], want_v[:, ctx_len:]
    entry = (contract.one_rounding(entry_k), contract.one_rounding(entry_v))

    want_attention = reference.attention_reference(drawn.layer, drawn.hidden_ctx, drawn.hidden_new)
    contract.compared(
        tf,
        tmp_path,
        source,
        case,
        "self_attention",
        activations=drawn.args,
        weights=drawn.loaded.constants,
        expected=(want_attention, entry_k, entry_v),
        held=(contract.one_rounding(want_attention), *entry),
        dims={"ctx_len": ctx_len},
    )

    want_out = reference.decode_step_oracle(drawn)
    contract.compared(
        tf,
        tmp_path,
        source,
        case,
        "decoder_layer",
        activations=drawn.args,
        weights=drawn.loaded.constants,
        expected=(want_out, entry_k, entry_v),
        held=(contract.one_rounding(want_out), *entry),
        dims={"ctx_len": ctx_len},
    )

    assert drawn.k_cache.shape[1] == ctx_len
    assert entry_k.shape[1] == 1 and want_k.shape[1] == ctx_len + 1


def test_the_placed_mlp_matches_the_reference(tf, shipped_source, tmp_path) -> None:
    """The performance witness keeps the shipped MLP's numerical boundary."""
    drawn = reference.decode_step_inputs(ctx_len=0, device="cpu")
    want = reference.mlp_reference(drawn.layer, drawn.hidden_new)

    contract.compared(
        tf,
        tmp_path,
        shipped_source(MODEL),
        CASES[0],
        "placed_mlp",
        activations=(drawn.hidden_new,),
        weights=drawn.loaded.constants,
        expected=(want,),
        held=(contract.one_rounding(want),),
    )


CONFIG = published()


DT = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}[
    str(CONFIG.dtype).removeprefix("torch.")
]
BYTES_PER = torch.finfo(CONFIG.dtype).bits // 8


Q_PROJ = CONFIG.num_attention_heads * CONFIG.head_dim
KV_PROJ = CONFIG.num_key_value_heads * CONFIG.head_dim


CTX = 1024


FLOPS_PER_MAC = 2


ELEMENTWISE_SHARE = 0.01

_ARITHMETIC = ("compute-cost", "memory", "roofline")


def _reported(tf, source, selector, dims):
    """What three families say about one function of the shipped source."""
    return contract.reported(tf, source, CASES[0], selector, _ARITHMETIC, dims)


def _mlp_matmul_flops() -> int:
    """Gate, up and down, each one token against a `hidden x intermediate`."""
    return 3 * FLOPS_PER_MAC * CONFIG.hidden_size * CONFIG.intermediate_size


def _projection_flops() -> int:
    """q, k, v and o for one token. GQA makes k and v narrower than q."""
    widths = (Q_PROJ, KV_PROJ, KV_PROJ, Q_PROJ)
    return sum(FLOPS_PER_MAC * CONFIG.hidden_size * width for width in widths)


def _attention_flops(ctx: int) -> int:
    """Scores against the context, then the same again gathering values.

    One query per head against `ctx` keys of `head_dim`, and the weighted sum over
    `ctx` values of `head_dim`. The context includes no self term because the
    decoded token's own key is written into the cache by the step and attended as
    part of it.
    """
    per_head = FLOPS_PER_MAC * ctx * CONFIG.head_dim
    return 2 * CONFIG.num_attention_heads * per_head


def _holds(reported: int, derived: int, label: str) -> None:
    """*reported* is *derived* plus only elementwise work."""
    assert reported >= derived, (
        f"{label}: analysis reports {reported} flops, fewer than the {derived} "
        f"its matmuls alone require -- a matmul is being missed or counted small"
    )
    excess = reported - derived
    assert excess <= ELEMENTWISE_SHARE * derived, (
        f"{label}: analysis reports {reported} flops against {derived} of matmul, "
        f"an excess of {excess} that elementwise work over vectors of "
        f"{CONFIG.hidden_size} and {CONFIG.intermediate_size} cannot account for"
    )


def test_the_mlp_costs_its_three_matrices(tf, shipped_source) -> None:
    """Both MLP views include 75,497,472 flops of three matmuls."""
    source = shipped_source(MODEL)
    authored = _reported(tf, source, "mlp", None)
    placed = _reported(tf, source, "placed_mlp", None)
    authored_flops = authored["totals"]["flops"][DT]
    placed_flops = placed["totals"]["flops"][DT]

    assert placed_flops == authored_flops
    _holds(authored_flops, _mlp_matmul_flops(), "mlp")


def test_the_attention_costs_its_projections_and_its_context(tf, shipped_source) -> None:
    """The four projections plus two passes over a 1024-token context."""
    data = _reported(tf, shipped_source(MODEL), "self_attention", {"ctx_len": CTX})

    _holds(
        data["totals"]["flops"][DT],
        _projection_flops() + _attention_flops(CTX),
        "self_attention",
    )


def test_the_layer_costs_its_two_halves_and_nothing_else(tf, shipped_source) -> None:
    """A decoder layer is the attention and the MLP.

    A decoder layer is the attention and the MLP. Asserted against the sum of
    the two derivations rather than against the layer's own report, so a layer
    that counted a half twice fails even though each half is right.
    """
    data = _reported(tf, shipped_source(MODEL), "decoder_layer", {"ctx_len": CTX})

    _holds(
        data["totals"]["flops"][DT],
        _mlp_matmul_flops() + _projection_flops() + _attention_flops(CTX),
        "decoder_layer",
    )


def test_the_layer_reads_at_least_its_weights_and_its_cache(tf, shipped_source) -> None:
    """Traffic is what a decode step is actually about, so it is checked too.

    A lower bound, because an intermediate is charged in full at every consumer
    that reads it, so the reported total sits above the truth by an amount this
    file cannot derive. What is exactly derivable is the floor: the step cannot
    read *less* than its weights, the cache it attends, and the rotary rows its
    position lands on. A step that read a weight matrix per output element, or
    forgot the cache, fails this.
    """
    data = _reported(tf, shipped_source(MODEL), "decoder_layer", {"ctx_len": CTX})

    weights = BYTES_PER * (
        3 * CONFIG.hidden_size * CONFIG.intermediate_size
        + 2 * CONFIG.hidden_size * Q_PROJ
        + 2 * CONFIG.hidden_size * KV_PROJ
    )
    cache = BYTES_PER * 2 * CTX * CONFIG.num_key_value_heads * CONFIG.head_dim

    rope_rows = 2 * 2 * CONFIG.head_dim * BYTES_PER
    read = data["totals"]["traffic"]["gmem"]["read"]

    assert read >= weights + cache + rope_rows, (
        f"the step reads {read} B, less than the {weights} B of weights, "
        f"{cache} B of cache and {rope_rows} B of rotary rows it must read"
    )


def test_a_decode_step_is_bound_by_memory(tf, shipped_source) -> None:
    """The conclusion the whole analysis exists to reach, and the one that was wrong.

    One token through one layer performs roughly 100 million flops while moving
    roughly 300 million bytes, making it decisively memory-bound on supported
    accelerators. The test pins the reported verdict because that is what callers
    consume.
    """
    data = _reported(tf, shipped_source(MODEL), "decoder_layer", {"ctx_len": CTX})

    bound = data["function_records"]["roofline"]
    assert bound["bound_by"] == "memory", (
        f"a one-token decode step is reported as {bound['bound_by']}-bound; it "
        f"moves {data['totals']['traffic']['gmem']['read']} B to do "
        f"{data['totals']['flops']['f32']} flops"
    )
    assert bound["ideal_ns"] > 0
