"""Qwen2.5-1.5B, as the installation ships it, asked through the commands."""

from __future__ import annotations

from tests.nvidia.cuda.installed.models import contract
from tests.nvidia.cuda.models.qwen2_5_1_5b import reference

MODEL = "qwen2_5_1_5b"
CASES = contract.model_cases(MODEL)


def test_decode_step_returns_cache_entry(tf, shipped_source, comparison_cache) -> None:
    """One decode step of one layer, and the state the step hands back.

    The whole layer is compared with ``Qwen2DecoderLayer.forward``. Returned cache
    entries are checked against oracle state rebuilt one token longer, so returning
    inputs unchanged fails. The supplied cache is oracle state, making the appended
    entry the only computed part. Every returned output is judged.
    """
    drawn = reference.decode_step_inputs(device="cpu")
    source, case = shipped_source(MODEL), CASES[0]
    want_out = reference.decode_step_oracle(drawn)
    want_k, want_v = reference.appended_cache_oracle(drawn)
    entry_k, entry_v = want_k[:, drawn.ctx_len :], want_v[:, drawn.ctx_len :]

    contract.compared(
        tf,
        comparison_cache,
        source,
        case,
        "decoder_layer",
        activations=drawn.args,
        weights={name: drawn.loaded.resource.load(name) for name in drawn.loaded.module.weights},
        expected=(want_out, entry_k, entry_v),
        held=(
            contract.three_roundings(want_out),
            contract.one_rounding(entry_k),
            contract.one_rounding(entry_v),
        ),
        dims={"ctx_len": drawn.ctx_len},
    )

    assert want_k.shape[1] == drawn.ctx_len + 1
    assert entry_k.shape[1] == 1 and entry_v.shape[1] == 1


test_model_analysis, test_model_ceiling = contract.analysis_tests(MODEL)
