from __future__ import annotations

import ast

from tilefoundry.ir.types.shard import Partial, ShardLayout, Split, make_mesh
from tilefoundry.parser.sugar import parse_sugar


def test_named_mesh_axis_sugar_carries_a_layout_index() -> None:
    mesh = make_mesh((8,), names=("cta",), topology="cta")
    node = ast.parse("(8 @ cta.cta,)", mode="eval").body

    layout = parse_sugar(
        node,
        ShardLayout,
        mesh_resolver=lambda name: mesh if name == "cta" else None,
    )
    partial_node = ast.parse('((8,), {cta.cta @ P("sum")})', mode="eval").body
    partial_layout = parse_sugar(
        partial_node,
        ShardLayout,
        mesh_resolver=lambda name: mesh if name == "cta" else None,
    )

    assert layout.attrs == (Split(0),)
    assert partial_layout.attrs == (Partial("sum"),)
