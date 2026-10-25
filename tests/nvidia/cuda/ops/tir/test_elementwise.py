"""``ops::elementwise`` over the three ways a mesh can name what holds a tile.

A stride-0 operand is the whole of broadcast. Beyond it, a tile can be cut by
two levels at once -- a CTA takes a block of it and a thread takes a row of
that -- or by two axes of one level, which is the same cut written as a grid.

See [runtime §2.6](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
import tilefoundry_nvidia.cuda.codegen  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import *
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.tir.arith import Binary
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Return, Sequential
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_nvidia import CudaTarget

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_CUDA = CudaTarget("nvidia.h200_sxm")

_CTAS, _THREADS = 4, 8
_ROWS = _CTAS * _THREADS


def test_binary_refuses_an_operand_outside_register_memory() -> None:
    types = (
        make_tensor_type((8,), storage="smem"),
        make_tensor_type((8,), storage="rmem"),
        make_tensor_type((8,), storage="rmem"),
    )
    args = tuple(Var(type=type_, name=f"a{index}") for index, type_ in enumerate(types))
    mesh = Mesh((Topology("thread", 32),), Layout((32,), (1,)))
    fn = PrimFunction(
        target=CudaTarget("nvidia.h200_sxm"),
        name="fn",
        params=args,
        body=Sequential(
            body=(
                MeshScope(
                    mesh,
                    Var(name="thread", type=mesh),
                    Sequential(body=(Evaluate(callable=Binary(kind=BinaryKind.ADD), args=args),)),
                ),
                Return(),
            )
        ),
    )
    with pytest.raises(VerifyError, match=r"lhs: smem does not match rmem"):
        verify_prim_function(fn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_mesh_names_what_holds_a_tile_three_ways() -> None:
    """Broadcast, two levels over one tile, and two axes over one tensor axis.

    The last two cut the same tile the same way and must agree: a CTA taking a
    block of rows while a thread takes one of them is the grid one level writes
    as two axes.
    """
    Elementwise = load_authored_ir(f"{OPS / 'elementwise_mesh_cuts.py'}:Elementwise")
    rm = tilefoundry.compile(Elementwise, target=_CUDA)
    torch.manual_seed(0)
    a = torch.randn(32, dtype=torch.float32, device="cuda")
    r = torch.randn(4, dtype=torch.float32, device="cuda")
    out = torch.zeros(32, dtype=torch.float32, device="cuda")
    out_sub = torch.zeros_like(out)
    out_scalar = torch.zeros_like(out)
    rows = torch.randn(_ROWS, dtype=torch.float32, device="cuda")
    level_out = torch.zeros(_ROWS, dtype=torch.float32, device="cuda")
    axis_out = torch.zeros(_ROWS, dtype=torch.float32, device="cuda")

    rm(a, r, out, out_sub, out_scalar, rows, level_out, rows, axis_out)
    torch.cuda.synchronize()

    assert torch.allclose(out, a * r.repeat_interleave(8), rtol=0, atol=0)
    assert torch.allclose(out_sub, r.repeat_interleave(8) - a, rtol=0, atol=0)
    assert torch.allclose(out_scalar, a + 0.5, rtol=0, atol=0)
    assert torch.allclose(level_out, rows * rows, rtol=0, atol=0)
    assert torch.equal(level_out, axis_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_cast_window_and_a_filled_parameter_emit_and_execute() -> None:
    """Effect-form Cast, Fill on a kernel parameter, and a mesh-coordinate window."""
    CastWindowHost = load_authored_ir(f"{OPS / 'cast_window.py'}:CastWindowHost")
    rm = tilefoundry.compile(CastWindowHost, target=_CUDA)
    torch.manual_seed(0)
    a = torch.randn(_ROWS, dtype=torch.float32, device="cuda")
    out = torch.zeros(_ROWS, dtype=torch.float16, device="cuda")
    filled = torch.zeros(_ROWS, dtype=torch.float32, device="cuda")

    rm(a, out, filled)
    torch.cuda.synchronize()

    assert torch.equal(out, a.half())
    assert torch.equal(filled, torch.full_like(filled, 0.5))
