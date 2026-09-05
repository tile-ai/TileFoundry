"""GPU e2e for the warp, mbarrier and TMA runtime ops, against torch.

Each op runs through its single public ``tilefoundry::ops`` entry from a
hand-written kernel -- the way a model's runtime twin calls it -- and the result
is compared to the torch expression that states the same thing. These groups
have TIR definitions but no codegen emit, so they cannot be reached the way
``test_mma_tir_handwritten.py`` reaches ``ops::mma``. What ties the TIR
definition to the implementation exercised here is therefore review, not this
test: this pins the implementation, and the emit would pin the pair.
"""

from __future__ import annotations

import pytest
import torch

from tests.integration.runtime_ops import device_ops

_THREADS = 256
_WARP = 32


@pytest.fixture(scope="module")
def ops():
    return device_ops()


def _lanes(n_threads: int) -> torch.Tensor:
    """Deterministic input with both signs and no ties across a warp."""
    g = torch.Generator(device="cpu").manual_seed(20260905)
    return torch.randn(n_threads, generator=g).cuda()


def test_warp_reduce_sum_matches_torch_sum(ops) -> None:
    """Every lane holds its warp's total, so the reference broadcasts back."""
    x = _lanes(4 * _THREADS)
    got = ops.warp_reduce_sum(x)
    want = x.reshape(-1, _WARP).sum(-1, keepdim=True).expand(-1, _WARP).reshape(-1)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)


def test_warp_reduce_max_matches_torch_amax(ops) -> None:
    x = _lanes(4 * _THREADS)
    got = ops.warp_reduce_max(x)
    want = x.reshape(-1, _WARP).amax(-1, keepdim=True).expand(-1, _WARP).reshape(-1)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("lane_mask", [1, 2, 4, 8, 16])
def test_shuffle_xor_exchanges_with_the_masked_lane(ops, lane_mask: int) -> None:
    """Lane ``l`` must receive lane ``l ^ lane_mask``'s value."""
    x = _lanes(2 * _THREADS)
    got = ops.shuffle_xor(x, lane_mask)
    lane = torch.arange(_WARP, device=x.device)
    partner = lane ^ lane_mask
    want = x.reshape(-1, _WARP).index_select(1, partner).reshape(-1)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("width", [8, 32, 256])
def test_shuffle_elect_admits_exactly_one_thread_per_block(ops, width: int) -> None:
    """An mbarrier armed for one arrival is correct only if this is exactly 1.

    a per-warp elect would make it 8 in a 256-thread block.
    """
    counts = ops.elect_count(width, 5)
    assert counts.tolist() == [1] * 5


def test_mbarrier_pipeline_flips_parity_across_rounds(ops) -> None:
    """A consumer that reads the wrong phase is caught by the addend.

    The producer adds the round index, so a tile read from the wrong phase shows
    up as the wrong addend rather than as stale data.
    """
    tile, rounds = 512, 6
    x = _lanes(tile * rounds)
    got = ops.mbarrier_pipeline(x, tile)
    offsets = torch.arange(rounds, device=x.device, dtype=x.dtype)
    want = x.reshape(rounds, tile) + offsets[:, None]
    torch.testing.assert_close(got, want.reshape(-1), rtol=0, atol=0)


@pytest.mark.parametrize("stages", [1, 3])
def test_tma_copy_stages_every_tile(ops, stages: int) -> None:
    """Every tile arrives, over a ring that does not divide the tile count.

    7 tiles over a 3-deep ring is two full turns plus one, so the parity is
    exercised in both states and the tail is not a whole number of turns.
    """
    tile, ntile = 1024, 7
    x = _lanes(tile * ntile)
    got = ops.tma_stage(x, tile, stages, 1)
    torch.testing.assert_close(got, x * 2.0, rtol=0, atol=0)


def test_tma_copy_takes_the_element_tier_for_a_strided_source(ops) -> None:
    """A source layout that is not one run of bytes cannot be a bulk copy.

    The kernel ``static_assert``s that this call picked the other tier, so a
    trait that quietly admitted it would fail to compile rather than run
    slowly; this pins that the values still land.
    """
    tile, ntile = 1024, 7
    x = _lanes(tile * ntile * 2)
    got = ops.tma_stage(x, tile, 3, 2)
    torch.testing.assert_close(got, x[::2] * 2.0, rtol=0, atol=0)


def test_tma_copy_falls_back_off_the_16_byte_grain(ops) -> None:
    """6 floats is contiguous but 24 bytes, so the instruction cannot carry it.

    The extent is what the shard leaves behind, not a property of the layout
    type, so this is a run-time hand-off inside the same entry rather than a
    different tier -- and the caller still just waits on the phase.
    """
    tile, ntile = 6, 7
    x = _lanes(_THREADS)[: tile * ntile]
    got = ops.tma_stage(x, tile, 3, 1)
    torch.testing.assert_close(got, x * 2.0, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("n", "src_stride", "threads", "width"),
    [(1024, 1, 1, 8), (1024, 2, 1, 1), (1024, 1, 256, 4), (512, 1, 256, 2)],
)
def test_copy_moves_the_tile_at_the_width_the_layouts_admit(
    ops, n: int, src_stride: int, threads: int, width: int
) -> None:
    """The width is the layouts' answer, and the kernel ``static_assert``s it.

    A contiguous bf16 run gives 8 elements per move; a strided one gives 1; and
    splitting the run over 256 threads leaves each of them 4 or 2, which caps
    the move at what it owns. This pins the values; the width is pinned at
    compile time, so a wrong choice never reaches here.
    """
    x = _lanes(n * src_stride)
    got = ops.copy_tile(x, src_stride, threads)
    want = x[::src_stride].to(torch.bfloat16).to(torch.float32)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert width in (1, 2, 4, 8)


def test_copy_widens_the_element_on_the_way_out(ops) -> None:
    """bf16 into f32: the wide load still holds, the conversion is on the store.

    4 elements is 8 bytes in and 16 out, which is the cap the wider side sets.
    """
    x = _lanes(_THREADS)
    got = ops.copy_widen(x)
    torch.testing.assert_close(got, x.to(torch.bfloat16).to(torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("b_k_major", [True, False])
def test_mma_loops_the_atom_over_a_whole_tile(ops, b_k_major: bool) -> None:
    """16x128x128 through one ``ops::mma`` call, against torch's matmul.

    ``b`` is logically ``(N, K)`` either way, so both cases compute the same
    product and the reference is one expression; ``b_k_major`` changes only
    which buffer it names, and therefore only its strides. That the same call
    reads both is the reason the entry has no transpose flag.
    """
    g = torch.Generator(device="cpu").manual_seed(20260905)
    a = torch.randn(16, 128, generator=g).cuda().to(torch.bfloat16)
    nk = torch.randn(128, 128, generator=g).cuda().to(torch.bfloat16)
    b = nk if b_k_major else nk.t().contiguous()
    got = ops.mma_tile(a, b, b_k_major)
    torch.testing.assert_close(got, a.float() @ nk.float().t(), rtol=2e-2, atol=2e-2)
