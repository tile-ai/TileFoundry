"""Pin checkpoint loading to the process-selected CUDA device.

Safetensors interprets bare ``"cuda"`` as index 0, while torch uses the current
device. They diverge only on multi-GPU hosts. These tests require all spellings
to resolve through ``torch.cuda.current_device()`` so loaded weights and adjacent
computations share a device.
"""

from __future__ import annotations

import gc

import pytest
import torch
from safetensors.torch import save_file

from tilefoundry.runtime import Preprocessed
from tilefoundry.runtime.resource import DrawnResource, SafetensorsResource, _resolved_device


def cpu_gen() -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(0)


def _unsharded(directory, tensors) -> str:
    """A checkpoint directory of one ``model.safetensors`` and no index file."""
    save_file(tensors, str(directory / "model.safetensors"))
    return str(directory)


def test_drawn_resource_never_redraws():
    from tests.fixtures.placed.leaf_weights import Small  # noqa: PLC0415 -- test fixture import

    resource = DrawnResource(Small, cpu_gen(), "cpu")
    assert resource.load("w") is resource.load("w")


def test_safetensors_cache_hits_while_held_and_drops_after(tmp_path):
    ckpt = _unsharded(tmp_path, {"w": torch.ones(4)})
    resource = SafetensorsResource(ckpt, device="cpu")
    held = resource.load("w")
    assert resource.load("w") is held
    del held
    gc.collect()
    assert not resource._read_tensors


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


def test_a_directory_with_one_shard_and_no_index_is_read(tmp_path) -> None:
    """A published checkpoint that was never sharded is read as it is."""
    ckpt = _unsharded(tmp_path, {"model.norm.weight": torch.ones(4)})
    store = SafetensorsResource(ckpt, device="cpu", alias={"gamma": "model.norm.weight"})

    assert torch.equal(store.load("gamma"), torch.ones(4))
    assert store.subtree("model").load("norm.weight").shape == (4,)


def test_a_directory_with_neither_says_so(tmp_path) -> None:
    """The report names what is missing, not just the index it looked for first."""
    with pytest.raises(FileNotFoundError, match="neither"):
        SafetensorsResource(str(tmp_path), device="cpu").load("anything")


def test_the_stored_dtype_is_what_comes_back(tmp_path) -> None:
    """Every read preserves the checkpoint's dtype, including a subtree view."""
    ckpt = _unsharded(
        tmp_path,
        {
            "w": torch.ones(4, dtype=torch.bfloat16),
            "nested.w": torch.ones(4, dtype=torch.bfloat16),
        },
    )

    assert SafetensorsResource(ckpt, device="cpu").load("w").dtype is torch.bfloat16
    read = SafetensorsResource(ckpt, device="cpu")
    assert read.load("w").dtype is torch.bfloat16
    assert read.subtree("nested").load("w").dtype is torch.bfloat16


def test_preprocessed_rejects_a_group_at_construction() -> None:
    """No existing M1 workflow constructs the public invalid one-to-many form."""
    with pytest.raises(TypeError, match="converter"):
        Preprocessed(("a", "b"), lambda value: value)


def test_preprocessed_cannot_name_a_subtree_segment() -> None:
    """M2 validates leaf preprocessing, so this public invalid segment needs coverage here."""
    resource = SafetensorsResource(
        "unused",
        alias={
            "layer": Preprocessed("model.layers.0", lambda value: value),
        },
    )

    with pytest.raises(TypeError, match="path, not a tensor"):
        resource.subtree("layer")
