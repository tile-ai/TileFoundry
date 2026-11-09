"""Loop-carried extent-one splits in CUDA shard propagation."""

from __future__ import annotations

from pathlib import Path

from tilefoundry import cli
from tilefoundry.ir import types


def test_unit_splits_survive_loop_carry() -> None:
    module = cli.load_authored_ir(
        str(Path(__file__).parents[1] / "fixtures/logical/unit_axes_carry.py")
    )
    assert module.entry_function().return_type.layout.attrs == (types.Split(0), types.Split(1), types.Split(2))
