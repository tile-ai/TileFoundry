"""Spec 001 core-ir — Op singleton cache."""

from __future__ import annotations

from tilefoundry.ir.core import Op


def test_op_attribute_singleton_cache() -> None:
    """No-attribute Ops are cached — ``Foo() is Foo()`` (spec 001)."""
    class _OpB(Op):
        pass

    assert _OpB() is _OpB()
