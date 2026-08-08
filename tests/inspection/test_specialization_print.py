"""Python printer renders a dispatch prototype as ``pass`` base + ``.specialize``.

Per [inspection §2.7](docs/spec/inspection.md#27-round-trip-contract) a dispatch prototype's rendering is display-only and must not be used as
a structural round-trip validation artifact, so these check emitted text plus
syntax validity instead.
"""

from __future__ import annotations

from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Var
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.types import make_tensor_type
from tilefoundry.ir.types.dim import DimVar


def _s_type():
    return make_tensor_type((DimVar(name="S", lo=1, hi=7),))


def _fn(*, body_is_self: bool, lo: int = 0, hi: int = 0) -> HirFunction:
    ty = _s_type()
    x = Var(type=ty, name="x")
    return HirFunction.build(
        name="main", params=(x,), body=x if body_is_self else None, return_type=ty,
        specializations=(DimVarRangePat("S", lo, hi),) if lo else (),
    )


def _prototype() -> HirFunction:
    base = _fn(body_is_self=False)
    base.add_variant(_fn(body_is_self=True, lo=1, hi=3))
    base.add_variant(_fn(body_is_self=True, lo=4, hi=7))
    return base


def test_prototype_prints_pass_base_and_specialize_blocks() -> None:
    """The base is a pass-bodied prototype and each variant a ``.specialize`` block
    over a throwaway ``def _``. The ``@module``-wrapped form must emit the same
    ``DimVarRangePat`` import as the standalone form — module and standalone output
    share one header emitter, so a construct requiring an extra import in one mode
    requires it in both."""
    standalone = as_script(_prototype())
    in_module = as_script(_prototype(), module="M")

    for src in (standalone, in_module):
        assert "    pass" in src
        assert "def _(" in src
        assert '@main.specialize(DimVarRangePat("S", 1, 3))' in src
        assert '@main.specialize(DimVarRangePat("S", 4, 7))' in src
        # The DimVarRangePat constructor is importable in the emitted source.
        assert "from tilefoundry.ir.core.pattern import DimVarRangePat" in src
        compile(src, "<test>", "exec")

    assert "@func\ndef main(" in standalone
    assert "@module" in in_module and "class M:" in in_module


def test_normal_function_omits_specialize() -> None:
    src = as_script(_fn(body_is_self=True))
    assert ".specialize(" not in src
    assert "DimVarRangePat" not in src
    assert "    pass" not in src
