"""The op registry as a table: what a name resolves to.

The op registry as a table: what a name resolves to, and how many things it
may resolve to.

Kinded sugar names (``add`` / ``sub`` / ...) resolve to a single alias schema;
there are no per-name legacy Op classes. Each kinded sugar name has *exactly
one* schema in the registry — the alias — while a genuinely overloaded name may
hold several. These are the boundaries a parse error cannot tell apart from a
missing registration, so they are asserted on the table directly.
"""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import add as _tf_add  # noqa: F401  -- closure capture
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.op_registry import (
    _first_schema,
    _schemas_by_dialect_name,
    get_op_by_name,
    get_schemas,
    iter_schema_names,
)
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor as TensorPat
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.types import DType


@pytest.fixture
def clean_schema_registry():
    """Registering into the process-wide table must not leak into other tests."""
    snapshot = {k: list(v) for k, v in _schemas_by_dialect_name.items()}
    yield
    _schemas_by_dialect_name.clear()
    _schemas_by_dialect_name.update(snapshot)


def test_kinded_alias_registers_one_schema_no_legacy_op() -> None:
    """A kinded sugar name resolves to exactly one schema.

    A kinded sugar name resolves to exactly one schema — the alias — with
    ``op_class=None``, a callable builder, and no legacy Op class. The builder
    constructs the kinded op and reuses the static ParamDef references for its
    signature; a binary and a unary name stand for their whole families.
    """
    schemas = get_schemas("tf", "add")
    assert len(schemas) == 1
    assert schemas[0].op_class is None
    assert get_op_by_name("add") is None

    binary = _first_schema("tf", "add")
    assert binary is not None and binary.op_class is None
    assert callable(binary.builder)
    assert binary.signature == (Binary.lhs, Binary.rhs)
    add_inst = binary.builder()
    assert isinstance(add_inst, Binary)
    assert add_inst.kind is BinaryKind.ADD

    unary = _first_schema("tf", "neg")
    assert unary is not None and unary.op_class is None
    assert unary.signature == (Unary.x,)
    neg_inst = unary.builder()
    assert isinstance(neg_inst, Unary)
    assert neg_inst.kind is UnaryKind.NEG


def test_register_op_overload_and_iter_dedupe(clean_schema_registry) -> None:
    """Multi-schema overloads append in registration order; iter dedupes names."""

    class _DummyBase:
        pass

    @register_op(dialect="T", category="nn", name="testdup_relu")
    class _A(_DummyBase):
        x = ParamDef(kind="input")

    @register_op(dialect="T", category="nn", name="testdup_relu")
    class _B(_DummyBase):
        src = ParamDef(kind="input", pattern=TensorPat)
        dst = ParamDef(kind="input", pattern=TensorPat)

    bucket = get_schemas("T", "testdup_relu")
    assert [s.op_class for s in bucket] == [_A, _B]
    names = list(iter_schema_names("T"))
    assert names.count("testdup_relu") == 1


@func
def _alias_call(
    a: Tensor[(8,), DType.f32],
    b: Tensor[(8,), DType.f32],
) -> Tensor[(8,), DType.f32]:

    return _tf_add(a, b)


def test_bare_add_routes_through_alias() -> None:
    body = _alias_call.body
    assert isinstance(body, Call)
    assert isinstance(body.target, Binary)
    assert body.target.kind is BinaryKind.ADD
