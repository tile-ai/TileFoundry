"""Tests for ``tilefoundry.dsl`` surface package — ``__getattr__`` resolution."""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import DimVar, DimVarRangePat, T, Tensor, tf
from tilefoundry.inspection import as_script
from tilefoundry.ir.core.op_registry import _schemas_by_dialect_name
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor as TensorPat
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.specialize import display_name
from tilefoundry.ir.types.dim import DimVar as IrDimVar


@pytest.fixture(autouse=True)
def _clean_schema_registry():
    snapshot = {k: list(v) for k, v in _schemas_by_dialect_name.items()}
    yield
    _schemas_by_dialect_name.clear()
    _schemas_by_dialect_name.update(snapshot)


def test_a_dialect_namespace_resolves_only_its_own_ops() -> None:
    """``tf`` and ``T`` are two dialects behind one ``__getattr__``.

    ``tf`` and ``T`` are two dialects behind one ``__getattr__``. A name
    registered in one dialect resolves there to a callable that builds the Op,
    is listed by ``dir`` for that dialect only, and stays an ``AttributeError``
    in the other — an unknown name must not quietly resolve to something else's
    op.
    """

    @register_op(dialect="tf", category="math", name="my_add")
    class _MyAdd:
        a = ParamDef(kind="input", pattern=TensorPat)
        b = ParamDef(kind="input", pattern=TensorPat)

        def __init__(self, **kw):
            self.kw = kw

    obj = tf.my_add(a="X", b="Y")
    assert isinstance(obj, _MyAdd)
    assert obj.kw == {"a": "X", "b": "Y"}
    assert "my_add" in dir(tf)
    assert "my_add" not in dir(T)

    with pytest.raises(AttributeError, match="no op named"):
        _ = T.my_add
    with pytest.raises(AttributeError, match="no op named"):
        _ = tf.this_op_does_not_exist


_S = DimVar("S", 1, 7)


@func
def sub(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
    pass


@sub.specialize(DimVarRangePat("S", 1, 3))
def narrow_s(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
    return x


@sub.specialize(DimVarRangePat("S", 4, 7))
def wide_s(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
    return x


def test_func_specializations_parse_to_variants() -> None:

    assert sub.body is None
    variants = sub.variants
    assert len(variants) == 2
    assert [v.name for v in variants] == ["sub", "sub"]
    assert variants[0].specializations == (DimVarRangePat("S", 1, 3),)
    assert variants[1].specializations == (DimVarRangePat("S", 4, 7),)

    assert display_name(variants[0]) == "narrow_s"
    assert display_name(variants[1]) == "wide_s"

    printed = as_script(sub)
    assert "def wide_s(" in printed
    assert "def narrow_s(" in printed

    for v in variants:
        (param,) = v.params
        (dim,) = param.type.shape
        assert isinstance(dim, IrDimVar)
        assert dim.name == "S"
