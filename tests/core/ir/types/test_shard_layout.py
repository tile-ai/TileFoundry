"""Spec 003 shard primitives — a layout that cannot be inverted must say so."""

from __future__ import annotations

import pytest

from tilefoundry.ir import types
from tilefoundry.ir.types import layout_algebra


@pytest.mark.parametrize(
    "outer",
    [
        types.Layout(shape=(2, 2), strides=(1, 1)),
        types.Layout(shape=(4,), strides=(0,)),
        types.Layout(shape=(2, 2), strides=(2, 2)),
    ],
    ids=lambda l: f"{l.shape}:{l.strides}",
)
def test_non_injective_outer_rejected(outer):

    assert not layout_algebra.is_inverse_projectable(outer)
