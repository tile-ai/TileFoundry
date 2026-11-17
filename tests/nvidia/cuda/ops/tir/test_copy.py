"""``ops::copy`` across the storage boundaries and both vector widths, one kernel.

See [runtime §2.6](targets/nvidia/tilefoundry_nvidia/cuda/docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tests.nvidia.cuda.ops.tir.targets import CUDA_TARGETS, retarget
from tilefoundry import cli
from tilefoundry_nvidia import cuda

OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_CUDA = cuda.Target("nvidia.h200_sxm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_crossings_round_trip_exactly() -> None:
    """One compile behind five assertions, each naming the pair that failed."""
    CopyStorage = cli.load_authored_ir(f"{OPS / 'copy_storage.py'}:CopyStorage")
    rm = tilefoundry.compile(CopyStorage, target=_CUDA)
    torch.manual_seed(0)
    a_smem = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    b_smem = torch.zeros_like(a_smem)
    a_rmem = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    b_rmem = torch.zeros_like(a_rmem)
    a_bcast = torch.arange(128, dtype=torch.float32, device="cuda").view(32, 4)
    b_bcast = torch.full((32, 4), -1.0, dtype=torch.float32, device="cuda")
    a_wide = torch.randn(128, 4, dtype=torch.float32, device="cuda")
    b_wide = torch.zeros_like(a_wide)
    a_narrow = torch.randn(128, 2, dtype=torch.float32, device="cuda")
    b_narrow = torch.zeros_like(a_narrow)
    rm(
        a_smem,
        b_smem,
        a_rmem,
        b_rmem,
        a_bcast,
        b_bcast,
        a_wide,
        b_wide,
        a_narrow,
        b_narrow,
    )
    torch.cuda.synchronize()
    assert torch.equal(b_smem, a_smem), (
        "gmem->smem, smem->gmem: shared memory belongs to the CTA, so the "
        "allocation is sized to the whole (128, 4) tile while local() offsets "
        "each thread into its own row -- the pair of facts that a buffer sized "
        "to one instance's share gets wrong for every instance but the first"
    )
    assert torch.equal(b_rmem, a_rmem), (
        "smem->rmem, rmem->gmem: a thread's registers are its own, so local() "
        "applies no offset to the fragment; the shared side does carry one, "
        "which is what makes this crossing different from the global one"
    )
    assert torch.equal(b_bcast, a_bcast), (
        "broadcast register tile: local() hands a register engine back by "
        "reference for a broadcast ShardLayout exactly as for a split one, so a "
        "value written into the tile has to be there to read back -- "
        f"{int((b_bcast == -1.0).sum())} of 128 cells never written is a lost "
        "write rather than a wrong offset, which would move data, not delete it"
    )
    assert torch.equal(b_wide, a_wide), (
        "gmem->rmem at 128 bits: a static contiguous run of four floats is what "
        "selects the vector load, and it must copy through the fragment "
        "unchanged"
    )
    assert torch.equal(b_narrow, a_narrow), (
        "gmem->rmem at 64 bits: a sub-128-bit fragment falls back to the element "
        "loop, which must answer identically to the vector path"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_plain_window_round_trips_fragments() -> None:
    """A copy with one plain and one sharded operand moves the instance's slice.

    ``local_tensor`` projects each operand through its own shard layout, so a
    plain operand used to stay whole for every instance: the gmem->rmem leg
    read the window's first 40 elements into every fragment, and the rmem->gmem
    leg tiled the fragment across the whole window (or, vectorized, read it out
    of bounds -- the tutorial's ``Invalid __local__ read of size 2``). The
    plain side is the slice the shard side's layout hands this instance.
    """
    RegisterRoundTrip = cli.load_authored_ir(f"{OPS / 'register_round_trip.py'}:RegisterRoundTrip")
    rm = tilefoundry.compile(RegisterRoundTrip, target=_CUDA)
    torch.manual_seed(0)
    a = torch.randn(2, 5120, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(a)
    rm(a, out)
    torch.cuda.synchronize()
    assert torch.equal(out, a), (
        "plain->shard->plain round trip: each thread holds the 40 elements its "
        "(x, y, t) coordinate names, so the window must reassemble exactly -- "
        "every thread copying the window's head, or tiling its fragment across "
        "the whole window, is the boundary projected wrong"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_plain_window_enters_row_major() -> None:
    """The projection's row-major steps must not come out transposed.

    CuTe decodes an integer index first-mode-fastest, so composing the
    instance's row-major window and origin into the plain tensor's rank-2
    layout unreversed hands every lane a permutation of its slice: lane 1 of
    a (2, 4) tile reads element (1, 0) where it owns (0, 2). A round trip
    hides it -- write and read share the permutation -- so this test copies
    in once and compares against the input.
    """
    OneWay = cli.load_authored_ir(f"{OPS / 'one_way_column_pairs.py'}:OneWay")
    rm = tilefoundry.compile(OneWay, target=_CUDA)
    a = torch.arange(4 * 64, dtype=torch.float32, device="cuda").view(4, 64)
    out = torch.zeros_like(a)
    rm(a, out)
    torch.cuda.synchronize()
    assert torch.equal(out, a), (
        "plain->shard with arange input: lane t must hold columns 2t and "
        "2t+1 of each row, so a transposed or otherwise permuted slice shows "
        "up as out != a exactly where the projection misread the index"
    )


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.parametrize(
    ("name", "by_columns"), [("ViewIntoParameter", False), ("ColumnsIntoRows", True)]
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_parameter_copies_by_coordinate(name: str, by_columns: bool, target: cuda.Target) -> None:
    """A copy into a parameter lands every element at its own coordinate.

    CUDA holds a parameter as one flat mode in memory order, while CuTe numbers
    a (2, 4) view first mode fastest. Pairing the two by CuTe's number put
    ``a[1, 0]`` where ``a[0, 1]`` belongs, and two parameters stored in
    different orders were copied as raw memory.
    """
    module = cli.load_authored_ir(f"{OPS / 'parameter_copies.py'}:{name}")
    rm = tilefoundry.compile(retarget(module, target), target=target)
    a = torch.arange(8, dtype=torch.float32, device="cuda").view(2, 4)
    if by_columns:
        a = a.t().contiguous().t()
    out = torch.zeros(2, 4, dtype=torch.float32, device="cuda")
    rm(a, out)
    torch.cuda.synchronize()
    assert torch.equal(out, a)
