"""``@register_op`` decorator + ``_build_schema`` contract."""

from __future__ import annotations

import pytest

from tilefoundry.ir.core.op_registry import (
    _register_schema,
    _schemas_by_dialect_name,
    get_schemas,
    iter_schema_names,
)
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import (
    _build_schema,
    _derive_dialect_and_category,
    register_op,
)


@pytest.fixture(autouse=True)
def _clean_schema_registry():
    snapshot = {k: list(v) for k, v in _schemas_by_dialect_name.items()}
    yield
    _schemas_by_dialect_name.clear()
    _schemas_by_dialect_name.update(snapshot)


class _DummyBase:
    pass


def test_derive_from_module_path() -> None:
    """HIR / TIR module paths derive dialect+category; others return None."""
    assert _derive_dialect_and_category("tilefoundry.ir.hir.nn.relu") == ("tf", "nn")
    assert _derive_dialect_and_category("tilefoundry.ir.tir.memory.copy") == ("T", "memory")
    assert _derive_dialect_and_category("mylib.my_ops") == (None, None)


def test_build_schema_explicit_args_and_validation() -> None:
    """Explicit args → schema; missing dialect/category outside builtin path raises."""
    class Add(_DummyBase):
        a = ParamDef(kind="input", pattern=Tensor)
        b = ParamDef(kind="input", pattern=Tensor)

    schema = _build_schema(Add, dialect="tf", category="math", name="add")
    assert schema.name == "add"
    assert schema.dialect == "tf"
    assert len(schema.signature) == 2

    # default name = cls.__name__.lower() (simple, NOT snake_case)
    class ScaledGemm(_DummyBase):
        x = ParamDef(kind="input", pattern=Tensor)

    assert _build_schema(ScaledGemm, dialect="tf", category="nn").name == "scaledgemm"

    with pytest.raises(ValueError, match="dialect"):
        _build_schema(Add, dialect="zzz", category="nn")
    with pytest.raises(ValueError, match="category"):
        _build_schema(Add, dialect="tf")


def test_register_op_with_name_override() -> None:
    @register_op(dialect="tf", category="math", name="my_add")
    class AddVariant(_DummyBase):
        x = ParamDef(kind="input", pattern=Tensor)

    schemas = get_schemas("tf", "my_add")
    assert len(schemas) == 1
    assert schemas[0].op_class is AddVariant
    assert AddVariant._op_schema is schemas[0]


def test_register_op_overload_and_iter_dedupe() -> None:
    """Multi-schema overloads append in registration order; iter dedupes names."""
    @register_op(dialect="T", category="nn", name="testdup_relu")
    class _A(_DummyBase):
        x = ParamDef(kind="input")

    @register_op(dialect="T", category="nn", name="testdup_relu")
    class _B(_DummyBase):
        src = ParamDef(kind="input", pattern=Tensor)
        dst = ParamDef(kind="input", pattern=Tensor)

    bucket = get_schemas("T", "testdup_relu")
    assert [s.op_class for s in bucket] == [_A, _B]
    names = list(iter_schema_names("T"))
    assert names.count("testdup_relu") == 1


def test_register_op_idempotent_on_same_schema() -> None:
    @register_op(dialect="tf", category="math")
    class MulOnce(_DummyBase):
        x = ParamDef(kind="input")


    _register_schema(MulOnce._op_schema)
    assert len(get_schemas("tf", "mulonce")) == 1
