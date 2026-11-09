"""ParamDef descriptor — the definitions it must refuse.

A well-formed signature is exercised by every op in the corpus; what a model run
cannot report is a *definition* that should never have been accepted, or an
``optional`` flag read as permission to omit the argument.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir import expr


def test_paramdef_rejects_an_unknown_kind_and_keeps_required_independent() -> None:
    """``kind`` is a closed set: an output is a result, never a parameter.

    ``kind`` is a closed set: an output is a result, never a parameter. And
    ``optional`` (a nullable type) is independent of ``default`` (an omittable
    argument) — only the latter makes a parameter non-required at the call site.
    """
    with pytest.raises(ValueError):
        expr.ParamDef(kind="output")

    required = expr.ParamDef(kind="input")
    assert required.is_required and not required.has_default

    optional_required = expr.ParamDef(kind="input", optional=True)
    assert optional_required.is_required

    omittable = expr.ParamDef(kind="attribute", default=0)
    assert not omittable.is_required and omittable.has_default


def test_paramdef_effect_is_an_explicit_input_only_flag() -> None:
    read_write = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ | expr.MemoryEffect.WRITE)
    assert read_write.effect == expr.MemoryEffect.READ | expr.MemoryEffect.WRITE
    assert expr.ParamDef(kind="input").effect is None

    with pytest.raises(TypeError, match="MemoryEffect"):
        expr.ParamDef(kind="input", effect="read")
    with pytest.raises(ValueError, match="attributes"):
        expr.ParamDef(kind="attribute", effect=expr.MemoryEffect.READ)
    with pytest.raises(ValueError, match="READ, WRITE"):
        expr.ParamDef(kind="input", effect=expr.MemoryEffect(0))
