"""Loop-carried extent-one splits in CUDA shard propagation."""

from __future__ import annotations

from pathlib import Path

from tests.core._corpus import load_program
from tilefoundry.ir.types.shard_layout import Split


def test_two_extent_one_splits_survive_a_loop_carry() -> None:
    module = load_program(Path(__file__).parents[1] / "fixtures/logical/unit_axes_carry.py")
    assert module.entry_function().return_type.layout.attrs == (Split(0), Split(1), Split(2))
