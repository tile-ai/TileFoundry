"""Python printer renders a dispatch prototype as ``pass`` base + ``.specialize``.

Per [inspection §2.7](docs/spec/inspection.md#27-round-trip-contract), a dispatch
prototype's rendering is display-only, so these check emitted text and syntax
validity rather than structural round-trip validation.
"""

from __future__ import annotations

from tilefoundry import inspection
from tilefoundry.dsl import *
from tilefoundry.ir import core, expr


def _s_type():
    return TensorType.from_shape((DimVar(name="S", lo=1, hi=6),))


def _fn(*, body_is_self: bool, lo: int = 0, hi: int = 0) -> core.Function:
    ty = _s_type()
    x = expr.Var(type=ty, name="x")
    return core.Function.from_parts(
        name="main",
        params=(x,),
        body=x if body_is_self else None,
        return_type=ty,
        specializations=(pattern.Range("S", lo, hi),) if lo else (),
    )


def _prototype() -> core.Function:
    base = _fn(body_is_self=False)
    base.add_variant(_fn(body_is_self=True, lo=1, hi=2))
    base.add_variant(_fn(body_is_self=True, lo=4, hi=6))
    return base


def test_prototype_prints_pass_base_and_specialize_blocks() -> None:
    """Test prototype prints pass base and specialize blocks.

    The base is a pass-bodied prototype and each variant a ``.specialize`` block
    over a generated binding. The ``@module``-wrapped form must emit the same
    ``pattern.Range`` import as the standalone form — module and standalone output
    share one header emitter, so a construct requiring an extra import in one mode
    requires it in both.
    """
    standalone = inspection.as_script(_prototype())
    in_module = inspection.as_script(_prototype(), module="M")

    for src in (standalone, in_module):
        assert "    pass" in src
        assert "def variant_S_1_2(" in src
        assert "def variant_S_4_6(" in src
        assert '@main.specialize(pattern.Range("S", 1, 2))' in src
        assert '@main.specialize(pattern.Range("S", 4, 6))' in src

        assert "from tilefoundry.dsl import *" in src
        compile(src, "<test>", "exec")

    assert "@func\ndef main(" in standalone
    assert "@module" in in_module and "class M:" in in_module


def test_normal_function_omits_specialize() -> None:
    src = inspection.as_script(_fn(body_is_self=True))
    assert ".specialize(" not in src
    assert "pattern.Range" not in src
    assert "    pass" not in src
