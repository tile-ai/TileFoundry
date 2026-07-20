"""User-registered custom Op via ``@register_op`` — registry/dispatch.

Verifies the registration-side surface of a custom op:

- ``@register_op`` lands the op in the OpSchema registry under the
  expected ``(dialect, name)`` key and surfaces through the
  class-keyed view helpers.
- ``parser.dispatch.resolve_op`` finds the custom op by bare name
  (so ``@func`` bodies can call it).

The ``@func``-usage integration form (a real ``@func`` that calls the
registered custom op and resolves it as the ``Call`` target) lives in
``tests/parser/hir/test_parse_custom_op.py``.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.op_registry import (
    _schemas_by_dialect_name,
    get_op_by_name,
    get_schemas,
)
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.parser.dispatch import resolve_op


@pytest.fixture
def _isolated_registry():
    """Snapshot/restore the schema registry around each test."""
    snapshot = {k: list(v) for k, v in _schemas_by_dialect_name.items()}
    try:
        yield
    finally:
        _schemas_by_dialect_name.clear()
        _schemas_by_dialect_name.update(snapshot)


def test_register_custom_hir_op(_isolated_registry) -> None:
    """``@register_op`` lands an op in the schema registry + lookup helpers."""

    @register_op(dialect="tf", category="custom", name="custom_addsq")
    class CustomAddSq(Op):
        """Custom op: lhs + rhs, then squared. (Test-only fixture.)"""
        lhs = ParamDef(kind="input", pattern=Tensor)
        rhs = ParamDef(kind="input", pattern=Tensor)

    schemas = get_schemas("tf", "custom_addsq")
    assert len(schemas) == 1
    s = schemas[0]
    assert s.op_class is CustomAddSq
    assert s.dialect == "tf"
    assert s.category == "custom"
    assert s.name == "custom_addsq"

    # Class-keyed view reflects the new schema.
    assert get_op_by_name("custom_addsq") is CustomAddSq


def test_parser_can_resolve_custom_op(_isolated_registry) -> None:
    """``parser.dispatch.resolve_op`` finds the custom op by bare name."""

    @register_op(dialect="tf", category="custom", name="custom_dispatch_op")
    class CustomDispatchOp(Op):
        x = ParamDef(kind="input", pattern=Tensor)

    cls = resolve_op("custom_dispatch_op")
    assert cls is CustomDispatchOp
