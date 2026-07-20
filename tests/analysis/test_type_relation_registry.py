"""Forward type-relation registry + AccessRelationResult carrier — mechanics."""
from __future__ import annotations

import isl

from tilefoundry.ir.core import Op
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    build_relation,
    register_type_relation,
    type_relation_registry,
)


def _result() -> AccessRelationResult:
    return AccessRelationResult(
        domain=isl.set("{ [m, k, n] : 0 <= m < 8 and 0 <= k < 4 and 0 <= n < 2 }"),
        maps=(
            isl.map("{ [m, k, n] -> [m, k] }"),
            isl.map("{ [m, k, n] -> [k, n] }"),
            isl.map("{ [m, k, n] -> [m, n] }"),
        ),
    )


def test_register_and_build_relation():
    class _DummyOp(Op):
        pass

    @register_type_relation(_DummyOp)
    def _(call, input_types, ctx):
        return _result()

    class _Call:
        target = _DummyOp()

    out = build_relation(_Call(), (), None)
    assert isinstance(out, AccessRelationResult)
    assert len(out.maps) == 3


def test_build_relation_returns_none_for_unregistered():
    class _UnregOp(Op):
        pass

    class _Call:
        target = _UnregOp()

    assert build_relation(_Call(), (), None) is None
    assert type_relation_registry.lookup(_UnregOp) is None
