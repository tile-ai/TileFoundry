"""One complete Qwen3.5-35B-A3B decoder layer, of each published type, against
Hugging Face's own.

The mixer and the MoE block are each checked against Hugging Face elsewhere; what
this adds is the layer around them -- the two residual additions, and the fact
that the MoE block reads the mixer's output rather than the layer's input. Both
are invisible to the two component tests: they would pass with the residuals
dropped, or with the MoE fed the wrong tensor.

There is deliberately no test of the 40-layer stack. What a stack observes and a
layer does not -- layer order, the residual thread running the length of the
model, the final norm -- is not measured here and is recorded as untested in
``test_provenance.py`` rather than approximated by a shorter stack that would
answer a different question.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.qwen3_5_35b_a3b import reference
from tests.models.qwen3_5_35b_a3b.decoder_layer import build_decoder_layer

DEV = reference.DEVICE

#: Looser than the component tests', for a reason that is *derived* rather than
#: fitted -- see ``test_the_layer_gap_is_the_mixers_gap_amplified_by_the_moe``,
#: which is what licenses this number. In short: the layer puts a 256-expert MoE
#: block downstream of the mixer, and that block amplifies a perturbation of its
#: input by a measured factor of 6.7 (full attention) to 25.9 (linear attention).
#: The mixer's own f32 round-off is 3.3e-06 and 4.1e-07, so an absolute
#: difference around 2e-05 is what amplification alone predicts, with the
#: *relative* difference staying at 1-2e-06.
#:
#: The other way to settle this would be a precision scan -- the same comparison
#: in f64, where a semantic gap survives and a rounding gap collapses. That is not
#: available: the HIR dtype surface has no f64 (``DType.from_name`` admits bf16,
#: f16, f32, and the narrow formats). Hence the amplification measurement, which
#: reaches the same conclusion using only f32.
ATOL = RTOL = 1e-4

#: Measured on an H200, f32, at ctx_len 25 and 41: full attention 1.96e-05 and
#: 1.14e-05 (reference magnitude 13.1 and 8.76); linear attention 1.17e-05 and
#: 1.60e-05 (11.7 and 7.67).
MEASURED_LAYER_MAX_ABS_DIFF = 4e-05

#: How much larger than the amplification prediction the end-to-end gap may be
#: before it stops being explained by it. Two, because the prediction pushes the
#: gap through Hugging Face's MoE while the measured path goes through the
#: kernels' own -- same sensitivity, different round-off, so they agree in
#: magnitude rather than exactly.
AMPLIFICATION_SLACK = 2.0

#: Both boundaries here span a whole layer, so both ask for a whole one built.
_STEP = {
    "full_attention": (reference.full_step, reference.full_layer_oracle),
    "linear_attention": (reference.linear_step, reference.linear_layer_oracle),
}


def _draw(block_type):
    draw, oracle = _STEP[block_type]
    return draw(device=DEV, whole_layer=True), oracle


@pytest.mark.parametrize("block_type", sorted(_STEP))
def test_decoder_layer_matches_hugging_face(block_type):
    """The complete layer vs `Qwen3_5MoeDecoderLayer.forward` at the decoded
    position -- mixer + residual, then post-norm + MoE + residual."""
    step, oracle = _draw(block_type)
    layer = build_decoder_layer(block_type)

    out, state = layer.forward(step.hidden_new, step.mixer_args, step.moe_args)

    want = oracle(step)
    difference = (out.float() - want.float()).abs().max().item()
    assert difference <= MEASURED_LAYER_MAX_ABS_DIFF, (block_type, difference)
    torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)
    assert len(state) == 2, (
        "each mixer hands two state tensors back through the layer boundary: a "
        "key and a value, or a convolution column and a recurrent matrix"
    )


@pytest.mark.parametrize("block_type", sorted(_STEP))
def test_the_layer_gap_is_the_mixers_gap_amplified_by_the_moe(block_type):
    """Why this file's tolerance is looser than the component tests', measured.

    A looser bound is worth nothing if it was chosen to make a test green, so
    this separates the two possibilities directly: either the layer's larger
    difference is the mixer's f32 round-off amplified by the block downstream of
    it, or it is a discrepancy in the layer's own composition.

    The measurement takes the layer's midpoint -- the mixer output plus the
    residual -- from both sides, and pushes *both* through the same Hugging Face
    MoE block. Nothing of the kernels is involved in that second step, so what
    comes out is the amplification of the midpoint gap and nothing else. If that
    reproduces the end-to-end gap, the end-to-end gap has no other cause.

    Measured on an H200: the MoE amplifies by 6.7x for full attention (midpoint
    gap 3.28e-06 -> 2.19e-05, against an observed 1.96e-05) and by 25.9x for
    linear attention (4.10e-07 -> 1.06e-05, against an observed 1.17e-05).
    """
    step, oracle = _draw(block_type)
    layer = build_decoder_layer(block_type)

    out, _state = layer.forward(step.hidden_new, step.mixer_args, step.moe_args)
    observed = (out.float() - oracle(step).float()).abs().max().item()

    mixed, *_ = layer.mixer(step.hidden_new, *step.mixer_args)
    ours = (step.hidden_new + mixed).float()
    mixer_oracle = (
        reference.full_mixer_oracle if block_type == "full_attention"
        else reference.linear_mixer_oracle
    )
    theirs = (step.hidden_new + mixer_oracle(step)).float()
    midpoint_gap = (ours - theirs).abs().max().item()

    hf_moe = step.layer.mlp
    hf_norm = step.layer.post_attention_layernorm
    with torch.no_grad():
        predicted = (
            (ours + hf_moe(hf_norm(ours))) - (theirs + hf_moe(hf_norm(theirs)))
        ).abs().max().item()

    assert midpoint_gap > 0.0
    amplification = predicted / midpoint_gap
    assert amplification > 1.0, (
        f"the MoE block does not amplify at all ({amplification}), so the "
        f"layer's larger difference has some other cause"
    )
    assert observed <= AMPLIFICATION_SLACK * predicted, (
        f"the layer differs from Hugging Face by {observed}, more than the "
        f"{predicted} that amplifying the mixer's own {midpoint_gap} through the "
        f"MoE accounts for -- that residue is not rounding, and this file's "
        f"tolerance must not be widened to cover it"
    )


@pytest.mark.parametrize("block_type", sorted(_STEP))
def test_the_layer_is_two_residual_additions(block_type):
    """Dropping either residual changes the answer, and by how much.

    Measured rather than asserted structurally: the layer's output is compared
    against its own two halves recombined, so a layer that added the residual
    once, twice, or to the wrong tensor disagrees. That the residual matters at
    all is worth measuring too -- if the blocks' outputs dominated it, this
    boundary would not distinguish a layer that omitted it.
    """
    step, _oracle = _draw(block_type)
    layer = build_decoder_layer(block_type)

    out, _state = layer.forward(step.hidden_new, step.mixer_args, step.moe_args)

    mixed, *_ = layer.mixer(step.hidden_new, *step.mixer_args)
    attended = step.hidden_new + mixed
    expected = attended + layer.moe(attended, *step.moe_args)

    torch.testing.assert_close(out.float(), expected.float(), atol=ATOL, rtol=RTOL)

    # The same composition with the first residual left out.
    without = mixed + layer.moe(mixed, *step.moe_args)
    moved = (out.float() - without.float()).abs().max().item()
    assert moved > 100 * ATOL, (
        f"omitting the attention residual moved the answer by only {moved}, so "
        f"this boundary does not distinguish a layer that drops it"
    )


@pytest.mark.parametrize("block_type", sorted(_STEP))
def test_the_moe_reads_the_mixed_state_not_the_layer_input(block_type):
    """The MoE block is downstream of the mixer, and this measures it.

    A layer that fed the MoE its own input instead would still return a tensor of
    the right shape, still be within the residual structure, and still pass every
    component test in this package.
    """
    step, _oracle = _draw(block_type)
    layer = build_decoder_layer(block_type)

    out, _state = layer.forward(step.hidden_new, step.mixer_args, step.moe_args)

    mixed, *_ = layer.mixer(step.hidden_new, *step.mixer_args)
    attended = step.hidden_new + mixed
    wrong_input = attended + layer.moe(step.hidden_new, *step.moe_args)

    moved = (out.float() - wrong_input.float()).abs().max().item()
    assert moved > 100 * ATOL, (
        f"feeding the MoE the layer input instead of the mixed state moved the "
        f"answer by only {moved}, so this boundary does not order the two blocks"
    )
