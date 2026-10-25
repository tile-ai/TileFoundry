"""Cover the CUDA staging-copy definition: direction, dtype, shape, and the call.

See [tir §2.3](docs/spec/tir.md#23-tir-ops).
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
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Return, Sequential
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_nvidia import CudaTarget
from tilefoundry_nvidia.cuda.ir.tir.memory.copy_async_bulk import CopyAsyncBulk

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_BAR = make_tensor_type((1,), DType.from_name("i64"), storage="smem")


def _pf(src, dst, bar=_BAR) -> PrimFunction:
    args = tuple(Var(type=t, name=n) for t, n in ((src, "src"), (dst, "dst"), (bar, "bar")))
    mesh = Mesh((Topology("thread", 32),), Layout((32,), (1,)))
    return PrimFunction(
        target=CudaTarget("nvidia.h200_sxm"),
        name="fn",
        params=args,
        body=Sequential(
            body=(
                MeshScope(
                    mesh=mesh,
                    binding=Var(name="thread", type=mesh),
                    body=Sequential(body=(Evaluate(callable=CopyAsyncBulk(), args=args),)),
                ),
                Return(),
            )
        ),
    )


def _ty(n, dtype=DType.from_name("f32"), storage="gmem"):
    return make_tensor_type((n,), dtype, storage=storage)


def test_accepts_a_whole_grain_gmem_to_smem_run() -> None:
    """The shape this op is built for: a gmem run into a shared tile."""
    verify_prim_function(_pf(_ty(8), _ty(8, storage="smem")))


def test_refuses_the_wrong_direction() -> None:
    """Only gmem into smem is this instruction."""
    with pytest.raises(VerifyError, match="src: smem does not match gmem"):
        verify_prim_function(_pf(_ty(8, storage="smem"), _ty(8, storage="smem")))
    with pytest.raises(VerifyError, match="dst: gmem does not match smem"):
        verify_prim_function(_pf(_ty(8), _ty(8, storage="gmem")))


def test_refuses_a_barrier_outside_shared_memory() -> None:
    with pytest.raises(VerifyError, match="barrier: gmem does not match smem"):
        verify_prim_function(
            _pf(
                _ty(8),
                _ty(8, storage="smem"),
                make_tensor_type((1,), DType.from_name("i64"), storage="gmem"),
            )
        )


def test_refuses_a_dtype_change() -> None:
    """A staging copy moves bytes; it does not convert them."""
    with pytest.raises(VerifyError, match="reads src.dtype = dst.dtype"):
        verify_prim_function(_pf(_ty(8), _ty(8, DType.from_name("bf16"), storage="smem")))


def test_refuses_a_shape_change() -> None:
    with pytest.raises(VerifyError, match="reads src.shape = dst.shape"):
        verify_prim_function(_pf(_ty(8), _ty(4, storage="smem")))


@pytest.mark.parametrize(
    ("n", "dtype"),
    [(5, DType.from_name("f32")), (1, DType.from_name("f32")), (7, DType.from_name("bf16"))],
)
def test_admits_a_transfer_off_the_sixteen_byte_grain(n, dtype) -> None:
    """The grain belongs to one instruction, and the op does not name one.

    See [runtime §2.6](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#26-cudaops).
    """
    verify_prim_function(_pf(_ty(n, dtype), _ty(n, dtype, storage="smem")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_both_tiers_stage_the_input_unchanged() -> None:
    """One compile behind two assertions, each naming the tier that failed."""
    TmaTiers = load_authored_ir(f"{OPS / 'tma_tiers.py'}:TmaTiers")
    rm = tilefoundry.compile(TmaTiers, target=CudaTarget("nvidia.h200_sxm"))
    torch.manual_seed(0)
    bulk_a = torch.randn(256, dtype=torch.float32, device="cuda")
    bulk_b = torch.zeros(256, dtype=torch.float32, device="cuda")
    odd_a = torch.randn(5, dtype=torch.float32, device="cuda")
    odd_b = torch.zeros(5, dtype=torch.float32, device="cuda")
    rm(bulk_a, bulk_b, odd_a, odd_b)
    torch.cuda.synchronize()
    assert torch.equal(bulk_b, bulk_a), (
        "bulk tier (two contiguous runs of the same element type): one_run_v is "
        "cosize == size on the projected view, and a broadcast operand's "
        "projection is the allocation's own layout, so bulk_eligible_v is true "
        "at compile time. 256 floats is 1024 bytes, a whole number of grains, so "
        "the elected lane issues the instruction and the block waits on the phase"
    )
    assert torch.equal(odd_b, odd_a), (
        "strided tier (a byte count the bulk instruction refuses): five floats "
        "is 20 bytes and bytes & 15 is not zero, which is the check Bulk makes "
        "before issuing and now the only route into Strided by construction, "
        "the static tier having been retired -- every instance of the "
        "destination mesh strides the run, then one arrival says the tile is "
        "readable. Same entry, same barrier, same answer"
    )
