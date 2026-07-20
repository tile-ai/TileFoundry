"""Access relation registry — basic mechanics.

"""
from __future__ import annotations

import pytest

from tilefoundry.ir.core import Op
from tilefoundry.ir.hir.tensor.quant import Quant
from tilefoundry.visitor_registry.access_relation import (
    OPAQUE,
    AccessRelations,
    OpaqueRelation,
    access_relation_registry,
    register_access_relation,
)


def test_opaque_is_singleton():
    assert OpaqueRelation() is OPAQUE
    assert OpaqueRelation() is OpaqueRelation()


def test_double_register_raises():
    class _DummyOp(Op):
        pass

    @register_access_relation(_DummyOp)
    def _(call, ctx):
        return AccessRelations(inputs=(OPAQUE,), outputs=(OPAQUE,))

    with pytest.raises(RuntimeError, match="access_relation: _DummyOp already registered"):
        register_access_relation(_DummyOp)(lambda call, ctx: None)


def test_lookup_miss_returns_none():
    class _UnregOp(Op):
        pass

    assert access_relation_registry.lookup(_UnregOp) is None
    assert access_relation_registry.has(_UnregOp) is False


def test_quant_handler_registered():
    """Smoke: Quant import side-effect registers its handler."""

    assert access_relation_registry.has(Quant)
