"""Tests for ``tilefoundry.dsl`` surface package — ``__getattr__`` resolution."""

from __future__ import annotations

import pytest

from tilefoundry import inspection
from tilefoundry.dsl import *
from tilefoundry.interfaces import valuation
from tilefoundry.ir import expr, pattern
from tilefoundry.ir.expr import op_registry

IrDimVar = DimVar


@pytest.fixture(autouse=True)
def _clean_schema_registry():
    snapshot = dict(op_registry.registry.entries)
    yield
    op_registry.registry.entries.clear()
    op_registry.registry.entries.update(snapshot)


def test_dialect_resolves_own_ops() -> None:
    """``tf`` and ``T`` are two dialects behind one ``__getattr__``.

    ``tf`` and ``T`` are two dialects behind one ``__getattr__``. A name
    registered in one dialect resolves there to a callable that builds the Op,
    is listed by ``dir`` for that dialect only, and stays an ``AttributeError``
    in the other — an unknown name must not quietly resolve to something else's
    op.
    """

    @expr.register_op(dialect="tf", category="math", name="my_add",
                 aliases={"my_sum": {}})
    class _MyAdd:
        a = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
        b = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())

        def __init__(self, **kw):
            self.kw = kw

    obj = tf.my_add(a="X", b="Y")
    assert isinstance(obj, _MyAdd)
    assert obj.kw == {"a": "X", "b": "Y"}
    assert expr.get_registered_name(_MyAdd) == "tf.my_add"
    assert expr.get_registered_name(obj) == "tf.my_sum"
    assert isinstance(tf.my_sum(a="X", b="Y"), _MyAdd)

    assert "my_add" in dir(tf)
    assert "my_add" not in dir(T)

    with pytest.raises(AttributeError, match="no op named"):
        _ = T.my_add
    with pytest.raises(AttributeError, match="no op named"):
        _ = tf.this_op_does_not_exist


_S = DimVar("S", 1, 6)


@func
def sub(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
    pass


@sub.specialize(pattern.Range("S", 1, 2))
def narrow_s(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
    return x


@sub.specialize(pattern.Range("S", 4, 6))
def wide_s(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
    return x


def test_func_specializations_parse_to_variants() -> None:

    assert sub.body is None
    variants = sub.variants
    assert len(variants) == 2
    assert [v.name for v in variants] == ["sub", "sub"]
    assert variants[0].specializations == (pattern.Range("S", 1, 2),)
    assert variants[1].specializations == (pattern.Range("S", 4, 6),)

    assert valuation.display_name(variants[0]) == "narrow_s"
    assert valuation.display_name(variants[1]) == "wide_s"

    printed = inspection.as_script(sub)
    assert "def wide_s(" in printed
    assert "def narrow_s(" in printed

    for v in variants:
        (param,) = v.params
        (dim,) = param.type.shape
        assert isinstance(dim, IrDimVar)
        assert dim.name == "S"
