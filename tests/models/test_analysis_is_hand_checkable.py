"""What the analyses say about a real decoder layer, checked against arithmetic.

Every other test of these families asks whether they run. That is a different
question from whether they are right, and the difference is not academic: the work
of one authored call was being multiplied by every thread on the machine, so a
decode step's cost came back sixteen thousand times too large and the roofline
called a memory-bound step compute-bound. Nothing failed, because nothing compared
a number to a number.

So this compares numbers. The subject is Qwen3-1.7B because its dimensions are
public and its decoder layer is arithmetic anybody can do on paper: three MLP
matrices, four attention projections, and an attention over a stated context.

The expectations are derived here from the model's own published dimensions, not
copied from a run. That is the point -- a snapshot of today's output would pass
whatever today's output is, which is exactly how the factor above survived. Where
a quantity cannot be derived without knowing an evaluator's per-element
convention, it is bounded rather than asserted: the matmuls are stated exactly,
and the elementwise remainder is held to being small, because a vector of two
thousand elements cannot account for a measurable share of seventy-five million
flops. A wrong count fails the bound in one direction or the other.
"""

from __future__ import annotations

from tests.models.fixtures import ACCEPTANCE
from tests.models.qwen3_1_7b.config import REAL
from tests.models.registry import case
from tilefoundry.analysis import analyze
from tilefoundry.inspection.analysis_report import report

#: The context the layer is asked about. Any length works; this one is stated so
#: the attention terms below can be written down.
CTX = 1024

#: A multiply and an add per multiply-accumulate.
FLOPS_PER_MAC = 2

#: How much of a total may be work this file does not derive. The elementwise
#: stages -- the two norms, the rotary embedding, SiLU, the residual adds -- run
#: over vectors of `hidden` or `intermediate` elements at a handful of flops each,
#: so together they are thousandths of a matmul over the same widths. One percent
#: leaves room for every convention an evaluator might use and no room for a
#: missing or duplicated matmul.
ELEMENTWISE_SHARE = 0.01


def _analysed(function_name: str, dims):
    """One function of the real model, analysed on the acceptance machine."""
    model = case("qwen3_1_7b")
    module = model.build_for(ACCEPTANCE())
    function = module.lookup(function_name)
    return report([
        analyze(module, function, analysis=family, dims=dims)
        for family in ("compute-cost", "memory", "roofline")
    ])


def _mlp_matmul_flops() -> int:
    """Gate, up and down, each one token against a `hidden x intermediate`."""
    return 3 * FLOPS_PER_MAC * REAL.hidden * REAL.intermediate


def _projection_flops() -> int:
    """q, k, v and o for one token. GQA makes k and v narrower than q."""
    widths = (REAL.q_proj, REAL.kv_proj, REAL.kv_proj, REAL.q_proj)
    return sum(FLOPS_PER_MAC * REAL.hidden * width for width in widths)


def _attention_flops(ctx: int) -> int:
    """Scores against the context, then the same again gathering values.

    One query per head against `ctx` keys of `head_dim`, and the weighted sum over
    `ctx` values of `head_dim`. The context includes no self term because the
    decoded token's own key is written into the cache by the step and attended as
    part of it.
    """
    per_head = FLOPS_PER_MAC * ctx * REAL.head_dim
    return 2 * REAL.n_q_heads * per_head


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
        f"{REAL.hidden} and {REAL.intermediate} cannot account for"
    )


def test_the_mlp_costs_its_three_matrices() -> None:
    """75,497,472 flops of matmul: 3 x 2 x 2048 x 6144."""
    data = _analysed("mlp", None)

    _holds(data["totals"]["flops"]["f32"], _mlp_matmul_flops(), "mlp")


def test_a_tiled_mlp_costs_the_same_matmuls_as_the_untiled_one() -> None:
    """Tiling reassociates a K reduction; it does not change the arithmetic.

    So the same lower bound and the same elementwise allowance hold, and between
    them they pin a loop nest from both sides. Measured, both sides were needed: the
    loop trip counts were not being applied at all, which reported this kernel at a
    thousandth of its cost, and a first attempt at applying them charged everything
    the body could reach -- including values computed once before the loop, and the
    whole of a preceding loop -- which reported it at thirty-four times.

    The excess over the matmuls is real work here rather than round-off: the tiled
    form accumulates into three buffers, which the untiled form does not do at all.
    It stays inside the same one percent.
    """
    data = _analysed("tiled_mlp", None)

    _holds(data["totals"]["flops"]["f32"], _mlp_matmul_flops(), "tiled_mlp")


def test_the_attention_costs_its_projections_and_its_context() -> None:
    """The four projections plus two passes over a 1024-token context."""
    data = _analysed("self_attention", {"ctx_len": CTX})

    _holds(
        data["totals"]["flops"]["f32"],
        _projection_flops() + _attention_flops(CTX),
        "self_attention",
    )


def test_the_layer_costs_its_two_halves_and_nothing_else() -> None:
    """A decoder layer is the attention and the MLP. Asserted against the sum of
    the two derivations rather than against the layer's own report, so a layer
    that counted a half twice fails even though each half is right."""
    data = _analysed("decoder_layer", {"ctx_len": CTX})

    _holds(
        data["totals"]["flops"]["f32"],
        _mlp_matmul_flops() + _projection_flops() + _attention_flops(CTX),
        "decoder_layer",
    )


def test_the_layer_reads_at_least_its_weights_and_its_cache() -> None:
    """Traffic is what a decode step is actually about, so it is checked too.

    A lower bound only, and the reason is a defect this measurement found rather
    than a limit of the arithmetic. `_call_traffic` derives its per-level bytes from
    each operand's whole type and discards the evaluator's own byte count, so a call
    that reads part of a table cannot say so: the two rotary tables are charged whole
    at each of the two calls that take them, which is 67,108,864 B for a step that
    gathers one position -- 1,024 B. Intermediates are likewise charged in full at
    every consumer. So the reported total is above the truth by an amount this file
    cannot derive, and an upper bound here would either encode the overcount as
    expected or be too loose to catch anything.

    What is still worth holding: the step cannot read *less* than its weights and the
    cache it attends, and those are exactly derivable. A step that read a weight
    matrix per output element, or forgot the cache, fails this.
    """
    data = _analysed("decoder_layer", {"ctx_len": CTX})

    bytes_per = 4  # f32
    weights = bytes_per * (
        3 * REAL.hidden * REAL.intermediate            # MLP
        + 2 * REAL.hidden * REAL.q_proj                # q, o
        + 2 * REAL.hidden * REAL.kv_proj               # k, v
    )
    cache = bytes_per * 2 * CTX * REAL.n_kv_heads * REAL.head_dim
    read = data["totals"]["traffic"]["gmem"]["read_bytes"]

    assert read >= weights + cache, (
        f"the step reads {read} B, less than the {weights} B of weights and "
        f"{cache} B of cache it must read"
    )


def test_a_decode_step_is_bound_by_memory() -> None:
    """The conclusion the whole analysis exists to reach, and the one that was
    wrong.

    One token through one layer does about a hundred million flops and moves about
    three hundred million bytes. On any accelerator whose peak flop rate is far
    above its bandwidth in bytes per second -- which is every accelerator this
    targets -- that is memory-bound, and it is memory-bound by a wide margin rather
    than marginally. Asserted as the reported verdict, because a reader acts on the
    verdict and not on the two numbers behind it.
    """
    data = _analysed("decoder_layer", {"ctx_len": CTX})

    bound = data["function_records"]["roofline"]
    assert bound["bound_by"] == "memory", (
        f"a one-token decode step is reported as {bound['bound_by']}-bound; it "
        f"moves {data['totals']['traffic']['gmem']['read_bytes']} B to do "
        f"{data['totals']['flops']['f32']} flops"
    )
    assert bound["theoretical_ns"] > 0
