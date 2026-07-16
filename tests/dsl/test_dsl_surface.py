"""Tests for ``tilefoundry.dsl`` surface package — ``__getattr__`` resolution."""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import DimVar, DimVarRangePat, T, Tensor, tf
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.op_registry import _schemas_by_dialect_name
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor as TensorPat
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types.dim import DimVar as IrDimVar


@pytest.fixture(autouse=True)
def _clean_schema_registry():
    snapshot = {k: list(v) for k, v in _schemas_by_dialect_name.items()}
    yield
    _schemas_by_dialect_name.clear()
    _schemas_by_dialect_name.update(snapshot)


def test_tf_unknown_name_raises_attribute_error() -> None:

    with pytest.raises(AttributeError, match="no op named"):
        _ = tf.this_op_does_not_exist


def test_t_unknown_name_raises_attribute_error() -> None:

    with pytest.raises(AttributeError, match="no op named"):
        _ = T.does_not_exist


def test_tf_resolves_registered_op_to_callable() -> None:
    @register_op(dialect="tf", category="math", name="my_add")
    class _MyAdd:
        a = ParamDef(kind="input", pattern=TensorPat)
        b = ParamDef(kind="input", pattern=TensorPat)

        def __init__(self, **kw):
            self.kw = kw


    fn = tf.my_add
    assert callable(fn)
    obj = fn(a="X", b="Y")
    assert isinstance(obj, _MyAdd)
    assert obj.kw == {"a": "X", "b": "Y"}


def test_dir_lists_registered_dialect_names() -> None:
    @register_op(dialect="tf", category="math", name="alpha")
    class _A:
        x = ParamDef(kind="input")

        def __init__(self, **kw): ...

    @register_op(dialect="T", category="memory", name="beta")
    class _B:
        x = ParamDef(kind="input")

        def __init__(self, **kw): ...


    assert "alpha" in dir(tf)
    assert "alpha" not in dir(T)
    assert "beta" in dir(T)
    assert "beta" not in dir(tf)


# ── `.specialize` parse positive -------------------------------------------
#
# A ``pass``-bodied ``@func`` base declares a dispatch prototype; each
# ``@base.specialize(DimVarRangePat(...))`` registers a variant in source
# order on ``base.variants``. This is the public DSL surface contract for
# dynamic-shape specialization; compile / codegen / execute lives in
# ``tests/e2e/``.

_S = DimVar("S", 1, 7)


@func
def sub(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
    pass


@sub.specialize(DimVarRangePat("S", 1, 3))
def _(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
    return x


@sub.specialize(DimVarRangePat("S", 4, 7))
def _(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
    return x


def test_func_specializations_parse_to_variants() -> None:
    # The base is a dispatch prototype (no body) carrying both variants.
    assert sub.body is None
    variants = sub.variants
    assert len(variants) == 2
    assert variants[0].name == "sub"
    assert variants[1].name == "sub"
    assert variants[0].specializations == (DimVarRangePat("S", 1, 3),)
    assert variants[1].specializations == (DimVarRangePat("S", 4, 7),)

    # Param shape on each variant carries the DimVar from the annotation.
    for v in variants:
        (param,) = v.params
        (dim,) = param.type.shape
        assert isinstance(dim, IrDimVar)
        assert dim.name == "S"

    # The prototype is a single Module entry; attribute access returns it.
    ir_mod = Module(name="m", functions=(sub,), entry="sub")
    assert ir_mod.lookup("sub") is sub
    assert ir_mod.function_named("missing") == ()
