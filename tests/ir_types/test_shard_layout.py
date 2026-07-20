"""Spec 003 shard primitives smoke coverage."""

from __future__ import annotations

import pytest

from tilefoundry.ir.types.shard import (
    B,
    ComposedLayout,
    Layout,
    LayoutBase,
    ShardLayout,
    Topology,
    make_mesh,
)
from tilefoundry.ir.types.shard import layout_algebra as la


def test_layout_base_contract_preserves_nested_shard_domain():
    base = Layout(shape=((None, 4), 8), strides=None)
    mesh = make_mesh((2,), topology=Topology("thread", 2))
    prior_stage = ShardLayout(layout=base, attrs=(B(),), mesh=mesh)
    composed = ComposedLayout(inner=None, offset=3, outer=prior_stage)

    assert isinstance(base, LayoutBase)
    assert isinstance(prior_stage, LayoutBase)
    assert isinstance(composed, LayoutBase)
    assert prior_stage.shape == base.shape
    assert composed.shape == base.shape
    assert base.domain_rank == prior_stage.domain_rank == composed.domain_rank == 3


def test_composed_layout_none_components_are_identity():
    base = Layout(shape=(4,), strides=(2,))
    inner_identity = ComposedLayout(inner=None, offset=3, outer=base)
    outer_identity = ComposedLayout(inner=base, offset=0, outer=None)

    assert la._apply_any(inner_identity, 1) == 5
    assert la._apply_any(outer_identity, 1) == 2
    assert inner_identity.shape == outer_identity.shape == base.shape
    assert inner_identity.domain_rank == outer_identity.domain_rank == 1


@pytest.mark.parametrize(
    "outer",
    [
        Layout(shape=(2, 2), strides=(1, 1)),  # c0 and c1 collide in the image
        Layout(shape=(4,), strides=(0,)),      # broadcast: all coords -> 0
        Layout(shape=(2, 2), strides=(2, 2)),  # overlapping strides collide
    ],
    ids=lambda l: f"{l.shape}:{l.strides}",
)
def test_non_injective_outer_rejected(outer):
    # A non-projectable outer must fail closed, not pick a colliding representative.
    assert not la.is_inverse_projectable(outer)
