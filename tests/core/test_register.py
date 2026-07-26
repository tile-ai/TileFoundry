"""``@register_op`` decorator + ``_build_schema`` contract."""

from __future__ import annotations

import pytest

from tilefoundry.ir.core.op_registry import (
    _schemas_by_dialect_name,
    get_schemas,
    iter_schema_names,
)
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op


@pytest.fixture(autouse=True)
def _clean_schema_registry():
    snapshot = {k: list(v) for k, v in _schemas_by_dialect_name.items()}
    yield
    _schemas_by_dialect_name.clear()
    _schemas_by_dialect_name.update(snapshot)


class _DummyBase:
    pass


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
