"""Custom Op parse integration — a real ``@func`` calls a registered op.

The registration/dispatch mechanics for ``@register_op`` (registry +
schema landing, bare-name ``resolve_op`` lookup) live in
``tests/core/test_custom_op_register.py``. This file exercises the
parser end of that contract: a custom op registered via
``@register_op`` is callable from a real ``@func`` body and resolves to
the custom ``Op`` class as the parsed ``Call`` target on
````.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor as TensorPattern
from tilefoundry.ir.core.register import register_op
from tilefoundry.visitor_registry import register_typeinfer


@register_op(dialect="tf", category="custom", name="custom_parse_addsq")
class CustomParseAddSq(Op):
    """Custom op: lhs + rhs, then squared. (Test-only fixture.)"""
    lhs = ParamDef(kind="input", pattern=TensorPattern)
    rhs = ParamDef(kind="input", pattern=TensorPattern)


@register_typeinfer(CustomParseAddSq)
def _(call, ctx):
    return ctx.type_of(call.args[0])


# Bind the registered Op class under its bare name so the ``@func`` body
# below can call it; the parser resolves bare callees through the
# function's closure (mirrors ``from tilefoundry.dsl.tf import *``).
custom_parse_addsq = CustomParseAddSq


@func
def _use_custom_op(
    a: Tensor[(8,), "f32"], b: Tensor[(8,), "f32"],
) -> Tensor[(8,), "f32"]:
    return custom_parse_addsq(a, b)


def test_parse_custom_op_resolves_to_custom_op_call_target() -> None:
    """A ``@func`` calling the registered custom op parses to a
    ``Call`` whose target is the custom ``Op`` instance."""
    body = _use_custom_op.body
    assert isinstance(body, Call)
    assert isinstance(body.target, CustomParseAddSq)
    assert len(body.args) == 2
