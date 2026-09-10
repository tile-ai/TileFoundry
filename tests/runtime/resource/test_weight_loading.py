"""Two cards, one process, one checkpoint: does each get the shard it is owed.

The judgement is in the twin's own function bodies
(``tests/fixtures/placed/tiny_tp_decoder.py``), where the runtime hands one
program its weights. What is left here is the staging a distributed read needs
and the two properties a body cannot see: that the checkpoint was read once per
card rather than once per call, and that a repeated dispatch is handed back the
tensor already held.
"""

from __future__ import annotations

import pytest
import torch

from tests.fixtures.placed.tiny_tp_decoder import (
    DECODE_FULL,
    PROJECT_FULL,
    C,
    R,
    TinyTPDecoderLM,
    TinyTPDecoderLMTwin,
)
from tilefoundry.ir.types.shard import Placement
from tilefoundry.runtime.resource import DictResource, SafetensorsResource


class _Counted:
    """A resource that records every raw name read through it, prefix and all."""

    def __init__(self, inner, counts: "dict[str, int] | None" = None, prefix: str = "") -> None:
        self.inner, self.prefix = inner, prefix
        self.counts = {} if counts is None else counts

    def load(self, name: str) -> torch.Tensor:
        key = f"{self.prefix}{name}"
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.inner.load(name)

    def load_group(self, name: str):
        return self.inner.load_group(name)

    def subtree(self, seg: str) -> "_Counted":
        return _Counted(self.inner.subtree(seg), self.counts, f"{self.prefix}{seg}.")


def test_distributed_weight_loading(tmp_path) -> None:
    """One load, two cards, and each body sees only what its program id selects."""
    if torch.cuda.device_count() < 2:
        pytest.fail(
            "distributed weight loading is a claim about two cards; with one there is "
            "nothing to get wrong, so a skip here would report a pass it never earned"
        )

    TinyTPDecoderLM.prepare(
        DictResource(
            {"layer.project_weight": PROJECT_FULL, "layer.decode_weight": DECODE_FULL}
        ),
        str(tmp_path),
    )
    reads = _Counted(SafetensorsResource(str(tmp_path)))
    twin = TinyTPDecoderLMTwin()
    twin.load(
        reads,
        placement=Placement(
            program_ids_getter=lambda topologies: (torch.cuda.current_device(), None)
        ),
    )

    selected = torch.cuda.current_device()
    try:
        for gpu in (0, 1):
            torch.cuda.set_device(gpu)
            x, row = torch.zeros(R, C, device=gpu), torch.zeros(R, device=gpu)
            first, replicated = twin.forward(x, row, 2)
            again, _ = twin.forward(x, row, 2)

            assert first.tensor.device.index == gpu
            assert replicated.tensor.device.index == gpu
            assert again.tensor.data_ptr() == first.tensor.data_ptr()
    finally:
        torch.cuda.set_device(selected)

    assert reads.counts == {"layer.project_weight": 2, "layer.decode_weight": 2}
