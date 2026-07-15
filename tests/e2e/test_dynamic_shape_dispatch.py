"""GPU end-to-end for entry-level dynamic-shape dispatch.

A ``pass``-bodied ``main`` prototype declares a single envelope
``DimVar('S', 1, 8)`` on a 1-D input tensor; two ``main.specialize``
variants partition it (half-open ranges ``[lo, hi)``). The runtime tensor
shape selects which variant runs:

- ``S in [1, 4)`` (1..3) → ``mul(x, x)`` (element-wise square).
- ``S in [4, 8)`` (4..7) → ``add(x, x)`` (element-wise double).

``tilefoundry.compile(...)`` lowers the prototype into a single ``main``
host wrapper that holds the dispatch if-chain and forwards to each
variant's mangled host wrapper. The kernels iterate at runtime extent
(``tilefoundry::ops::binary(... , x_shape_0, ...)``) and copy out via
``tilefoundry::ops::copy_n``, so the same compiled binary handles any shape
inside the envelope.
"""
from __future__ import annotations

import torch

import tilefoundry
from tilefoundry import func, module
from tilefoundry.dsl import DimVar, DimVarRangePat, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — binds bare ``mul`` / ``add``

_S = DimVar("S", 1, 8)   # half-open [1, 8) = 1..7


@module(entry="main")
class Dispatch:
    @func
    def main(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
        pass

    @main.specialize(DimVarRangePat("S", 1, 4))
    def _(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
        return mul(x, x)  # noqa: F821  (bound via ``from tilefoundry.dsl.tf import *``)

    @main.specialize(DimVarRangePat("S", 4, 8))
    def _(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
        return add(x, x)  # noqa: F821


def _build_runtime_module():
    """Compile the dispatch prototype to a fully-loaded RuntimeModule."""
    return tilefoundry.compile(Dispatch, target="cuda")


def test_entry_dispatch_variant_a_square_matches_torch() -> None:
    rm = _build_runtime_module()
    x = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    rm(x, out)
    torch.cuda.synchronize()
    expected = x * x
    assert torch.allclose(out, expected, rtol=0, atol=0)


def test_entry_dispatch_variant_b_double_matches_torch() -> None:
    rm = _build_runtime_module()
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    rm(x, out)
    torch.cuda.synchronize()
    expected = x + x
    assert torch.allclose(out, expected, rtol=0, atol=0)


def test_entry_dispatch_both_variants_in_one_session() -> None:
    """Both dispatch arms run through the same compiled binary in
    sequence. Verifies that the dispatch entry routes each call to its
    matching variant based on the current tensor's runtime shape."""
    rm = _build_runtime_module()

    x_a = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")  # S=3 → variant A
    out_a = torch.empty_like(x_a)
    rm(x_a, out_a)

    x_b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                       dtype=torch.float32, device="cuda")  # S=7 → variant B
    out_b = torch.empty_like(x_b)
    rm(x_b, out_b)
    torch.cuda.synchronize()

    assert torch.allclose(out_a, x_a * x_a, rtol=0, atol=0)
    assert torch.allclose(out_b, x_b + x_b, rtol=0, atol=0)
