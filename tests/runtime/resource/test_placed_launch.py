"""What a launch is told about which card it is, checked on two of them.

The host places a card and the kernel projects its own rows out of the shard
layout it was given. Which rows come back written is the only thing that says
the id survived the whole way: through the entry's internal argument, the
launch shim, the block's copy of it, and ``program_id`` inside a tensor view.
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tests.fixtures.placed.gpu_placed_rows import (
    COLS,
    GPUS,
    ROWS,
    ROWS_PER_CARD,
    GpuPlacedRows,
)
from tilefoundry.ir.types.shard import Placement
from tilefoundry.runtime.resource import DictResource


def test_a_placed_kernel_writes_the_rows_its_card_holds() -> None:
    """Each card writes its own rows and leaves every other card's alone."""
    if torch.cuda.device_count() < GPUS:
        pytest.fail(
            "a placed launch is a claim about two cards; with one there is "
            "nothing to place, so a skip here would report a pass it never earned"
        )

    compiled = tilefoundry.compile(GpuPlacedRows)
    assert compiled.type.places == ("gpu",)

    standing = {"gpu": 0}
    compiled.load(
        DictResource({}),
        placement=Placement(
            program_ids_getter=lambda levels: tuple(
                standing["gpu"] if level.name == "gpu" else None for level in levels
            )
        ),
    )

    selected = torch.cuda.current_device()
    try:
        for card in range(GPUS):
            torch.cuda.set_device(card)
            standing["gpu"] = card
            source = torch.arange(
                ROWS * COLS, dtype=torch.float32, device=card
            ).reshape(ROWS, COLS)
            written = torch.zeros_like(source)

            compiled(source, written)
            torch.cuda.synchronize()

            mine = slice(card * ROWS_PER_CARD, (card + 1) * ROWS_PER_CARD)
            assert torch.equal(written[mine], source[mine])
            elsewhere = torch.ones(ROWS, dtype=torch.bool, device=card)
            elsewhere[mine] = False
            assert not written[elsewhere].any(), "wrote rows another card holds"
    finally:
        torch.cuda.set_device(selected)
