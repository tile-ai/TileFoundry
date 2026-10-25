"""CUDA resource drawing and process-selected device handling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.runtime.resource import DrawnResource, _resolved_device


def cpu_gen() -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(0)


def test_drawn_resource_never_redraws():
    leaf_weights = Path(__file__).parents[2] / "fixtures/placed/leaf_weights.py"
    Small = load_authored_ir(f"{leaf_weights}:Small")
    resource = DrawnResource(Small, cpu_gen(), "cpu")
    assert resource.load("w") is resource.load("w")


def test_bare_cuda_resolves_current_device() -> None:
    if not torch.cuda.is_available():
        assert _resolved_device("cuda") == "cuda"
        return
    assert _resolved_device("cuda") == f"cuda:{torch.cuda.current_device()}"


def test_resolved_device_tracks_set_device() -> None:
    """The resolved device tracks a later `set_device`.

    The resolved device tracks a later `set_device`, so a worker assigned a card
    after import still loads onto that card.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("tracking a later set_device needs a second card to move to")
    original = torch.cuda.current_device()
    try:
        for index in (0, 1):
            torch.cuda.set_device(index)
            assert _resolved_device("cuda") == f"cuda:{index}"
    finally:
        torch.cuda.set_device(original)
