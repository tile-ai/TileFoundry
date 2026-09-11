"""What a launch is told about which card it is, checked on two of them.

The host places a card and the kernel projects its own rows out of the shard
layout it was given. Which rows come back written is the only thing that says
the id survived the whole way: through the entry's internal argument, the
launch shim, the block's copy of it, and ``program_id`` inside a tensor view.
One thread and one module per card, so nothing about a module changes while it
holds data narrowed for the card it was loaded on.
"""

from __future__ import annotations

from threading import Thread

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
    """Each card writes its own rows and leaves every other card's alone.

    One thread and one loaded module per card, which is the shape every
    mainstream runtime places in: the current device is thread-local and set
    once, and each module knows which program it is because it was told when it
    was loaded rather than because it asks the process.
    """
    if torch.cuda.device_count() < GPUS:
        pytest.fail(
            "a placed launch is a claim about two cards; with one there is "
            "nothing to place, so a skip here would report a pass it never earned"
        )

    entry = tilefoundry.compile(GpuPlacedRows).type
    assert tuple(p.name for p in entry.leading) == ("gpu",)
    failures: list[BaseException] = []

    def run(card: int) -> None:
        try:
            torch.cuda.set_device(card)
            compiled = tilefoundry.compile(GpuPlacedRows)
            compiled.load(DictResource({}), placement=Placement({"gpu": card}))

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
        except BaseException as error:  # noqa: BLE001 -- reported on the main thread
            failures.append(error)

    threads = [Thread(target=run, args=(card,)) for card in range(GPUS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, failures[0]
