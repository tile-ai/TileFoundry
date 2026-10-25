"""Assert the unique alias schema and the duplicate-name rejection contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Call
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.op_registry import (
    _REGISTRY,
    _first_schema,
    children,
    get_op_by_name,
    lookup,
)
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.pattern import is_ranked_tensor

ALIAS_CALL = Path(__file__).parents[2] / "fixtures" / "ir" / "alias_call.py"


@pytest.fixture
def clean_schema_registry():
    """Registering into the process-wide table must not leak into other tests."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def test_kinded_alias_registers_one_schema() -> None:
    """A kinded sugar name resolves to exactly one schema.

    A kinded sugar name resolves to exactly one schema — the alias — with
    ``op_class=None``, a callable builder, and no legacy Op class. The builder
    constructs the kinded op and reuses the static ParamDef references for its
    signature; a binary and a unary name stand for their whole families.
    """
    schema = lookup("tf", (), "add")
    assert schema.op_class is None
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


def test_register_op_rejects_duplicate_name(clean_schema_registry) -> None:
    """Each path has one owner; a duplicate identifies the existing owner."""

    @register_op(dialect="T", category="nn", name="testdup_relu")
    class _A:
        x = ParamDef(kind="input")

    with pytest.raises(ValueError, match="already owned by .*_A"):
        @register_op(dialect="T", category="nn", name="testdup_relu")
        class _B:
            src = ParamDef(kind="input", pattern=is_ranked_tensor())
            dst = ParamDef(kind="input", pattern=is_ranked_tensor())

    assert lookup("T", (), "testdup_relu").op_class is _A
    assert children("T", ()).count("testdup_relu") == 1


def test_bare_add_routes_through_alias() -> None:
    body = load_authored_ir(str(ALIAS_CALL)).entry_function().body
    assert isinstance(body, Call)
    assert isinstance(body.target, Binary)
    assert body.target.kind is BinaryKind.ADD
