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
def test_tma_bulk_copy_stages_every_tile(ops, stages: int) -> None:
    """Every tile arrives, over a ring that does not divide the tile count.

    7 tiles over a 3-deep ring is two full turns plus one, so the parity is
    exercised in both states and the tail is not a whole number of turns.
    """
    tile, ntile = 1024, 7
    x = _lanes(tile * ntile)
    got = ops.tma_stage(x, tile, stages)
    torch.testing.assert_close(got, x * 2.0, rtol=0, atol=0)


def test_tma_bulk_copy_rejects_an_unaligned_tile(ops) -> None:
    """An off-grain tile is refused, not transferred.

    The instruction has no defined behaviour off the 16-byte grain, so the entry
    refuses rather than moving something else.
    """
    x = _lanes(_THREADS)
    with pytest.raises(RuntimeError, match="16-byte multiple"):
        ops.tma_stage(x, 3, 1)
