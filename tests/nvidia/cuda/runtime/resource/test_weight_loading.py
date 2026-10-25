"""Two cards, one process, one checkpoint: does each get the shard it is owed.

The judgement is in the twin's own function bodies
(``tests/nvidia/cuda/fixtures/placed/tiny_tp_decoder.py``), where the runtime hands one
program its weights. What is left here is the staging a distributed read needs
and the two properties a body cannot see: that the checkpoint was read once per
card rather than once per call, and that a repeated dispatch is handed back the
tensor already held.
"""

from __future__ import annotations

from pathlib import Path
from threading import Thread

import pytest
import torch

from tilefoundry.cli.source import load_authored_ir, load_namespace
from tilefoundry.ir.types import Placement
from tilefoundry.runtime.resource import DictResource, SafetensorsResource

PLACED = Path(__file__).parents[2] / "fixtures" / "placed"


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


def test_weight_shard_per_card(tmp_path) -> None:
    """One thread and one module per card; each body sees only its own shard.

    Two modules rather than one moving between cards: a module is told which
    program it is when it is loaded, so it cannot answer differently later
    while holding weights narrowed for the card it was loaded on.
    """
    if torch.cuda.device_count() < 2:
        pytest.fail(
            "distributed weight loading is a claim about two cards; with one there is "
            "nothing to get wrong, so a skip here would report a pass it never earned"
        )

    tiny_tp_decoder = load_namespace(str(PLACED / "tiny_tp_decoder.py"))[0]
    PROJECT_FULL, DECODE_FULL = tiny_tp_decoder["PROJECT_FULL"], tiny_tp_decoder["DECODE_FULL"]
    R, C = tiny_tp_decoder["R"], tiny_tp_decoder["C"]
    TinyTPDecoderLMTwin = tiny_tp_decoder["TinyTPDecoderLMTwin"]
    TinyTPDecoderLM = load_authored_ir(f"{PLACED / 'tiny_tp_decoder.py'}:TinyTPDecoderLM")
    TinyTPDecoderLM.prepare(
        DictResource(
            {"layer.project_weight": PROJECT_FULL, "layer.decode_weight": DECODE_FULL}
        ),
        str(tmp_path),
    )
    reads = _Counted(SafetensorsResource(str(tmp_path), device="cuda"))
    failures: list[BaseException] = []

    def run(gpu: int) -> None:
        try:
            torch.cuda.set_device(gpu)
            twin = TinyTPDecoderLMTwin()
            twin.load(reads, placement=Placement({"gpu": gpu}))

            x, row = torch.zeros(R, C, device=gpu), torch.zeros(R, device=gpu)
            first, replicated = twin.forward(x, row, 2)
            again, _ = twin.forward(x, row, 2)

            assert first.tensor.device.index == gpu
            assert replicated.tensor.device.index == gpu
            assert again.tensor.data_ptr() == first.tensor.data_ptr(), (
                "the second call read the weight again instead of the one it held"
            )
        except BaseException as error:  # noqa: BLE001 -- reported on the main thread
            failures.append(error)

    threads = [Thread(target=run, args=(gpu,)) for gpu in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, failures[0]

    assert reads.counts == {"layer.project_weight": 2, "layer.decode_weight": 2}, (
        "each card reads the checkpoint once; neither read it twice"
    )


@pytest.mark.parametrize("case", ("cyclic", "mesh_order"))
def test_weight_load_refuses_noncanonical_axes(tmp_path, case):
    """Reject two unsupported views instead of silently loading the wrong elements."""
    root, size = {"cyclic": ("CyclicSplitWeight", 8), "mesh_order": ("MeshOrderWeight", 32)}[case]
    program = load_authored_ir(f"{PLACED / 'cyclic_split_weight.py'}:{root}")
    twin_type = load_namespace(str(PLACED / "cyclic_split_weight.py"))[0][f"{root}Twin"]
    program.prepare(DictResource({"weight": torch.arange(size, dtype=torch.float32)}), str(tmp_path))
    twin = twin_type()
    twin.load(SafetensorsResource(str(tmp_path), device="cpu"), placement=Placement({"gpu": 1}))
    with pytest.raises(ValueError, match="ShardTensor: the tensor is laid out .* while its type says"):
        twin.read()
