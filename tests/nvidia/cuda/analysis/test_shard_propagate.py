"""Loop-carried extent-one splits in CUDA shard propagation."""

from __future__ import annotations

from pathlib import Path

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.types.shard_layout import Split


def test_unit_splits_survive_loop_carry() -> None:
    module = load_authored_ir(
        str(Path(__file__).parents[1] / "fixtures/logical/unit_axes_carry.py")
    )
    assert module.entry_function().return_type.layout.attrs == (Split(0), Split(1), Split(2))
