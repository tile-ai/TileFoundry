"""Spec 003 shard primitives — a layout that cannot be inverted must say so."""

from __future__ import annotations

import pytest

from tilefoundry.ir.types.shard import Layout
from tilefoundry.ir.types.shard import layout_algebra as la


@pytest.mark.parametrize(
    "outer",
    [
        Layout(shape=(2, 2), strides=(1, 1)),
        Layout(shape=(4,), strides=(0,)),
        Layout(shape=(2, 2), strides=(2, 2)),
    ],
    ids=lambda l: f"{l.shape}:{l.strides}",
)
def test_non_injective_outer_rejected(outer):

    assert not la.is_inverse_projectable(outer)
