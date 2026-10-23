"""The staged-ring schedule fixtures, compiled and run on the card.

``staged_dynamic_inner`` and ``staged_nested`` reach full CUDA emission
through the tuple-choice pointer plumbing: two smem stages held as one
tuple of tensor_views, subscripted by a runtime index. Their rings are
re-authored with what execution needs and the corpus does not state --
the dynamic-smem carve-out, the cp.async commit/wait before read-back,
and trip counts uniform across the warp.

See [runtime §2.6.2](docs/spec/runtime.md#262-opscopycuh).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry_cuda.codegen  # noqa: F401 -- trigger emitter autodiscovery
from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget
from tilefoundry_cuda import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


@module(
    entry="counter_ring_host",
    target=_CUDA,
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class CounterRing:
    """``staged_dynamic_inner``'s ring, driven by its rmem stage counter.

    The choice is as authored: two tensor_views over raw smem offsets held in
    one tuple, subscripted by the counter modulo two, so the selected stage's
    base pointer is decided on the card each iteration.
    """

    @prim_func(target=_CUDA)
    def counter_ring_device(x: Tensor[(1, 8), "bf16"], out: Tensor[(1, 8), "bf16"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:  # noqa: F841
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                ring_stages = (
                    T.tensor_view(0, dtype="bf16", storage="smem", layout=(1, 8), shape=(1, 8)),
                    T.tensor_view(16, dtype="bf16", storage="smem", layout=(1, 8), shape=(1, 8)),
                )
                stage_counter = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(stage_counter, 0)
                stage_one = T.alloc_tensor(tensor_type=Tensor[(), "i64", "rmem"])
                T.fill(stage_one, 1)
                for i in range(0, 2, 1):
                    value = T.tensor_view(T.ptr_of(x), layout=(1, 8), shape=(1, 8))
                    stage = T.tensor_view(
                        T.ptr_of(ring_stages[stage_counter % 2]), layout=(1, 8), shape=(1, 8)
                    )
                    T.copy_async(value, stage)
                    T.cp_async_commit()
                    T.cp_async_wait(n=0)
                    T.sync(scope)
                    out_view = T.tensor_view(T.ptr_of(out), layout=(1, 8), shape=(1, 8))
                    T.copy(stage, out_view)
                    T.binary(stage_counter, stage_one, stage_counter, kind=BinaryKind.ADD)

    @prim_func(target=CpuTarget())
    def counter_ring_host(x: Tensor[(1, 8), "bf16"], out: Tensor[(1, 8), "bf16"]):
        launch(  # noqa: F821
            counter_ring_device,  # noqa: F821
            x,
            out,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            dynamic_smem=32,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_counter_driven_ring_stages_the_tile_through_both_stages() -> None:
    """The runtime index picks a stage per iteration; the tile arrives intact.

    A choice rendered wrong -- an index that does not read the counter, or
    stage bases spelled so the subscript cannot compile -- either fails
    emission here outright or lands the tile nowhere near ``out``.
    """
    rm = tilefoundry.compile(CounterRing, target=_CUDA)
    x = torch.arange(8, dtype=torch.bfloat16, device="cuda").view(1, 8)
    out = torch.full((1, 8), -1.0, dtype=torch.bfloat16, device="cuda")
    rm(x, out)
    torch.cuda.synchronize()
    assert torch.equal(out, x), (
        "counter-driven ring: iteration k copies x through stage k % 2, so "
        "out must hold x exactly -- a stale or uninitialized stage read "
        "breaks the equality"
    )


@module(
    entry="row_ring_host",
    target=_CUDA,
    topologies=(Topology("cta", 1), Topology("thread", 32)),
)
class RowRing:
    """``staged_nested``'s six rows through the same two-stage ring.

    Kept as authored: row ``3i + j`` stages through slot ``(3i + j) % 2``,
    the row windows are slice-based views with a stride-0 first mode, and
    the ring index is dim arithmetic over the two loop counters.
    """

    @prim_func(target=_CUDA)
    def row_ring_device(x: Tensor[(6, 8), "bf16"], out: Tensor[(6, 8), "bf16"]):
        with Mesh((Topology("cta", 1),), layout=(1,), names=("d0",)) as cta:  # noqa: F841
            with Mesh((Topology("thread", 32),), layout=(32,), names=("d0",)) as scope:
                ring_stages = (
                    T.tensor_view(0, dtype="bf16", storage="smem", layout=(1, 8), shape=(1, 8)),
                    T.tensor_view(16, dtype="bf16", storage="smem", layout=(1, 8), shape=(1, 8)),
                )
                for i in range(0, 2, 1):
                    for j in range(0, 3, 1):
                        value = T.tensor_view(
                            T.ptr_of(x[(3 * i) + j:(3 * i) + j + 1, 0:0 + 1]),
                            layout=((1, 8), (0, 1)),
                            shape=(1, 8),
                        )
                        stage = T.tensor_view(
                            T.ptr_of(ring_stages[((i * 3) + j) % 2]),
                            layout=(1, 8),
                            shape=(1, 8),
                        )
                        T.copy_async(value, stage)
                        T.cp_async_commit()
                        T.cp_async_wait(n=0)
                        T.sync(scope)
                        out_row = T.tensor_view(
                            T.ptr_of(out[(3 * i) + j:(3 * i) + j + 1, 0:0 + 1]),
                            layout=((1, 8), (0, 1)),
                            shape=(1, 8),
                        )
                        T.copy(stage, out_row)

    @prim_func(target=CpuTarget())
    def row_ring_host(x: Tensor[(6, 8), "bf16"], out: Tensor[(6, 8), "bf16"]):
        launch(  # noqa: F821
            row_ring_device,  # noqa: F821
            x,
            out,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            dynamic_smem=32,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_every_row_stages_through_the_slot_its_index_arithmetic_picks() -> None:
    """Dim arithmetic over both loop counters drives the ring; rows arrive.

    The index is ``((i * 3) + j) % 2`` rendered for the card, over windows
    whose bases come out of the slice path, so a row landed anywhere but its
    slot's stage breaks the row-for-row equality.
    """
    rm = tilefoundry.compile(RowRing, target=_CUDA)
    x = torch.arange(48, dtype=torch.bfloat16, device="cuda").view(6, 8)
    out = torch.full((6, 8), -1.0, dtype=torch.bfloat16, device="cuda")
    rm(x, out)
    torch.cuda.synchronize()
    assert torch.equal(out, x), (
        "row ring: row 3i + j passes through stage (3i + j) % 2 into out's "
        "row 3i + j, so out must equal x row for row"
    )
