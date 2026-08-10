"""ParamDef descriptor — the definitions it must refuse.

A well-formed signature is exercised by every op in the corpus; what a model run
cannot report is a *definition* that should never have been accepted, or an
``optional`` flag read as permission to omit the argument.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core.param_def import ParamDef


def test_paramdef_rejects_an_unknown_kind_and_keeps_required_independent() -> None:
    """``kind`` is a closed set: an output is a result, never a parameter.

    ``kind`` is a closed set: an output is a result, never a parameter. And
    ``optional`` (a nullable type) is independent of ``default`` (an omittable
    argument) — only the latter makes a parameter non-required at the call site.
    """
    with pytest.raises(ValueError):
        ParamDef(kind="output")  # type: ignore[arg-type]

    required = ParamDef(kind="input")
    assert required.is_required and not required.has_default

    optional_required = ParamDef(kind="input", optional=True)
    assert optional_required.is_required

    omittable = ParamDef(kind="attribute", default=0)
    assert not omittable.is_required and omittable.has_default
