"""DSV4 decode IR smoke.

One compact, unsharded ``@func`` composes the accepted decode contracts:
batch-aware Gather, a sum-reduced score, ``exp``, ``TopK``, a rank-3 single-row
cache ``insert_slice``, and a literal tuple return. It is verified three ways:
it evaluates against a Torch oracle, it round-trips through
``as_script`` / ``parse_script``, and it builds in the HTML viewer.
"""
from __future__ import annotations

import torch

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings for @func bodies
from tilefoundry.evaluator import evaluate
from tilefoundry.inspection import as_script
from tilefoundry.inspection.viewer.builder import ViewerBuilder
from tilefoundry.parser.hir_parser import parse_script

_B, _T, _D, _S, _K = 2, 6, 4, 5, 3
_POS = 2
_DEV = "cpu"


@func
def _dsv4_decode_step(
    k_cache: Tensor[(_B, _T, _D), "f32"],
    sel: Tensor[(_B, _S), "i32"],
    cache: Tensor[(_B, _T, _D), "f32"],
    new_row: Tensor[(_B, 1, _D), "f32"],
):
    kv = gather(k_cache, sel, axis=1, batch_dims=1)  # noqa: F405
    scores = reduce(kv, axes=(2,), keepdim=False, kind="sum")  # noqa: F405
    p = exp(scores)  # noqa: F405
    vals, idx = topk(p, k=_K, axis=-1, largest=True, sorted=True)  # noqa: F405
    cache_out = insert_slice(cache, new_row, (0, _POS, 0))  # noqa: F405
    return (vals, cache_out)


def _inputs(seed: int):
    torch.manual_seed(seed)
    return (
        torch.randn(_B, _T, _D),
        torch.randint(0, _T, (_B, _S), dtype=torch.int32),
        torch.randn(_B, _T, _D),
        torch.randn(_B, 1, _D),
    )


def _torch_oracle(k_cache, sel, cache, new_row):
    kv = k_cache[torch.arange(_B).unsqueeze(1), sel.long()]  # batched gather [B, S, D]
    p = kv.sum(dim=2).exp()  # sum-score then exp [B, S]
    vals = torch.topk(p, _K, dim=-1, largest=True, sorted=True).values  # [B, K]
    cache_out = cache.clone()
    cache_out[:, _POS : _POS + 1, :] = new_row  # single-row cache write
    return vals, cache_out


def test_dsv4_decode_smoke_evaluates_against_torch() -> None:
    k_cache, sel, cache, new_row = _inputs(0)
    out = evaluate(_dsv4_decode_step, k_cache, sel, cache, new_row, device=_DEV)
    assert isinstance(out, tuple) and len(out) == 2
    exp_vals, exp_cache = _torch_oracle(k_cache, sel, cache, new_row)
    torch.testing.assert_close(out[0], exp_vals)
    torch.testing.assert_close(out[1], exp_cache)


def test_dsv4_decode_smoke_roundtrips() -> None:
    script = as_script(_dsv4_decode_step)
    reparsed = parse_script(script)
    assert as_script(reparsed) == script
    k_cache, sel, cache, new_row = _inputs(1)
    out = evaluate(reparsed, k_cache, sel, cache, new_row, device=_DEV)
    exp_vals, exp_cache = _torch_oracle(k_cache, sel, cache, new_row)
    torch.testing.assert_close(out[0], exp_vals)
    torch.testing.assert_close(out[1], exp_cache)


def test_dsv4_decode_smoke_renders_in_viewer() -> None:
    src = ViewerBuilder(_dsv4_decode_step).build().source
    assert src
    lowered = src.lower()
    for name in ("gather", "reduce", "exp", "topk", "insert"):
        assert name in lowered, f"expected {name!r} node in viewer source"
