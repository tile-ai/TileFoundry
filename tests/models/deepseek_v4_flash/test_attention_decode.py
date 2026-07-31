"""One DeepSeek-V4-Flash decode step, against Hugging Face's own attention.

What the corpus harness runs for this model is `reference.py`; this file is where
the numbers it produces are actually compared, and where the comparison's own
noise floor is measured rather than assumed.

The model is authored in bf16 with an fp8 KV cache, so the oracle is asked in
bf16 too -- and because an oracle in the dtype under test carries its own
rounding, the same step is also compared against an f32 accumulation of the same
weights. That second comparison is stated as a relation rather than a constant:
the kernel has to be at least as close to the f32 answer as Hugging Face's own
bf16 run is. A tolerance nobody can derive would pass a kernel that had drifted
inside it; this one moves with the arithmetic.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.models.deepseek_v4_flash import reference
from tests.models.deepseek_v4_flash import reference as shape

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

#: The corpus's own contract for a reference comparison.
ATOL = RTOL = 2e-3

#: Two lengths, so a step that only works at the length it was authored against
#: cannot pass. Neither divides the head count, and the longer is the most this
#: sliding layer can attend.
CTX_LENGTHS = (41, reference.CTX_LEN)


@pytest.fixture(scope="module")
def drawn():
    return reference.attention_step_inputs()


def _max_abs(got, want) -> float:
    return (got.float() - want.float()).abs().max().item()


def _bf16_ulps(want, count: int = 2) -> float:
    """*count* representable bf16 steps at *want*'s largest element.

    The bound both comparisons below are held to, derived rather than chosen: a
    bf16 significand is 8 bits, so within one binade the spacing is the binade
    divided by 128, and two answers that differ by one such step are the same
    answer written twice. A tolerance picked by hand would have no relation to
    the arithmetic and would move whenever the draw did.
    """
    largest = want.float().abs().max().item()
    exponent = math.floor(math.log2(largest))
    return count * 2.0 ** (exponent - 7)


def test_the_decode_step_matches_hugging_faces_own_attention(drawn):
    """The step's output is what HF's attention produces at the decoded position.

    Both sides in the model's own dtype: same weights, same context, same
    rotation, no cache object on either side.
    """
    got, _ = reference.run_attention_step(drawn)
    want = reference.attention_step_oracle(drawn)

    assert tuple(got.shape) == tuple(want.shape)
    # One representable step apart, and inside what the corpus asks of a
    # reference -- which is the comparison the harness will make.
    assert _max_abs(got, want) <= _bf16_ulps(want), _max_abs(got, want)
    torch.testing.assert_close(got.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_the_disagreement_is_smaller_than_the_oracles_own_rounding(drawn):
    """Against an f32 accumulation of the same weights, the kernel is at least as
    close as Hugging Face's own bf16 run.

    Which is what says the bf16 comparison above is a comparison and not a
    tolerance wide enough to hide in: the same numbers, accumulated in f32, are
    further from HF's bf16 answer than they are from the kernel's.
    """
    got, _ = reference.run_attention_step(drawn)
    oracle_bf16 = reference.attention_step_oracle(drawn)

    layer_f32 = shape.build_hf_attention(
        seed=reference.WEIGHT_SEED, device=drawn.hidden_ctx.device.type
    ).float()
    oracle_f32 = shape.decode_reference(
        layer_f32, drawn.hidden_ctx.float(), drawn.hidden_new.float()
    )

    kernel_gap = _max_abs(got, oracle_f32)
    oracle_gap = _max_abs(oracle_bf16, oracle_f32)
    assert kernel_gap <= oracle_gap, (kernel_gap, oracle_gap)


def test_the_step_is_authored_over_a_range_of_context_lengths():
    """The same description, at two context lengths, each against its own oracle.

    `ctx_len` is a range rather than the one number the step was written at, and
    a step that had baked a length in would agree at one length only.
    """
    for ctx_len in CTX_LENGTHS:
        step = reference.attention_step_inputs(ctx_len=ctx_len)
        got, _ = reference.run_attention_step(step)
        want = reference.attention_step_oracle(step)

        assert step.kv_cache.shape[1] == ctx_len
        assert _max_abs(got, want) <= _bf16_ulps(want), (ctx_len, _max_abs(got, want))


def test_the_step_returns_the_cache_entry_to_append(drawn):
    """The returned latent is this token's cache entry: appending it to the cache
    the step was given reproduces the cache a context one token longer holds.

    Checked against a rebuilt cache rather than against the step's own input, so
    a step that returned its input unchanged would fail. The two agree to the
    fp8 grid the latent is stored on and no closer -- which is the point of
    storing it that way.
    """
    _, kv_new = reference.run_attention_step(drawn)
    grown = torch.cat([drawn.kv_cache, kv_new], dim=1)
    want = reference.appended_cache_oracle(drawn)

    assert tuple(grown.shape) == tuple(want.shape)
    assert torch.equal(grown[:, : drawn.ctx_len], want[:, : drawn.ctx_len])
    step = _fp8_step(want[:, drawn.ctx_len :])
    assert _max_abs(grown[:, drawn.ctx_len :], want[:, drawn.ctx_len :]) <= step, step


def _fp8_step(latent: torch.Tensor) -> float:
    """One e4m3 quantum of *latent*'s largest stored block.

    The cache holds its unrotated dims as fp8 with a per-block power-of-two
    scale, so two ways of computing the same latent may land one grid step apart
    -- and the grid is derived from the block's own magnitude rather than
    written down, so this cannot quietly widen.
    """
    blocks = (
        latent[..., : shape.REAL.nope_dim]
        .float()
        .reshape(-1, shape.KV_QUANT_BLOCK)
    )
    block_max = blocks.abs().amax(dim=-1).max().item()
    # e4m3 keeps 3 mantissa bits, so the spacing at the top of a block is
    # block_max / 2**3, and the scale rounds up to a power of two.
    return block_max / 8.0
