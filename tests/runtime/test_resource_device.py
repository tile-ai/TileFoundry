"""Which device a checkpoint's tensors land on when the caller says ``"cuda"``.

`safetensors` reads the device string itself, and it takes a bare ``"cuda"`` as
index 0. Every torch spelling of the same word -- ``.cuda()``, ``device="cuda"``,
a ``torch.device`` context -- means whichever device the process has selected. On
a single-card machine the two agree and nothing here can go wrong; on a machine
with several they disagree, and a weight loaded from the checkpoint cannot be
compared against anything computed alongside it.

That is what these check: not that loading works, but that the word means one
thing. The comparison is against `torch.cuda.current_device()` rather than against
a literal index, because a literal would pass on the very machine the defect
cannot occur on.
"""

from __future__ import annotations

import torch

from tilefoundry.runtime.resource import _resolved_device


def test_a_bare_cuda_resolves_to_the_device_the_process_selected() -> None:
    if not torch.cuda.is_available():
        assert _resolved_device("cuda") == "cuda"
        return
    assert _resolved_device("cuda") == f"cuda:{torch.cuda.current_device()}"


def test_an_index_the_caller_wrote_is_left_alone() -> None:
    """A caller naming a device is naming it; nothing here second-guesses that."""
    assert _resolved_device("cuda:3") == "cuda:3"
    assert _resolved_device("cpu") == "cpu"


def test_it_follows_the_selection_rather_than_reading_it_once() -> None:
    """The resolved device tracks a later `set_device`, so a worker assigned a card
    after import still loads onto that card."""
    if torch.cuda.device_count() < 2:
        return
    original = torch.cuda.current_device()
    try:
        for index in (0, 1):
            torch.cuda.set_device(index)
            assert _resolved_device("cuda") == f"cuda:{index}"
    finally:
        torch.cuda.set_device(original)
