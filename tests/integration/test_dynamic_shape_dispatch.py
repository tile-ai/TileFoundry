"""GPU end-to-end for dynamic-shape dispatch.

A ``pass`` prototype partitions ``DimVar('S', 1, 8)`` into half-open ranges:
``[1, 4)`` squares and ``[4, 8)`` doubles. Compilation emits one host dispatch
wrapper forwarding to specialized kernels. Runtime extents drive their loops
and copies, so one binary handles every shape in the envelope.
"""

from __future__ import annotations

import torch

import tilefoundry
from tilefoundry import func, module
from tilefoundry.dsl import DimVar, DimVarRangePat, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — binds bare ``mul`` / ``add``
from tilefoundry.target import CudaTarget

_S = DimVar("S", 1, 8)


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
    return tilefoundry.compile(Dispatch, target=CudaTarget("nvidia.h200_sxm"))


def test_entry_dispatch_both_variants_in_one_session() -> None:
    """Both dispatch arms run through the same compiled binary in sequence.

    Both dispatch arms run through the same compiled binary in
    sequence. Verifies that the dispatch entry routes each call to its
    matching variant based on the current tensor's runtime shape.
    """
    rm = _build_runtime_module()

    x_a = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32, device="cuda")
    out_a = torch.empty_like(x_a)
    rm(x_a, out_a)

    x_b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float32, device="cuda")
    out_b = torch.empty_like(x_b)
    rm(x_b, out_b)
    torch.cuda.synchronize()

    assert torch.allclose(out_a, x_a * x_a, rtol=0, atol=0)
    assert torch.allclose(out_b, x_b + x_b, rtol=0, atol=0)
