"""Python printer renders a dispatch prototype as ``pass`` base + ``.specialize``."""

from __future__ import annotations

from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Var
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.types import make_tensor_type
from tilefoundry.ir.types.dim import DimVar


def _s_type():
    return make_tensor_type((DimVar(name="S", lo=1, hi=7),))


def _variant(lo: int, hi: int) -> HirFunction:
    ty = _s_type()
    x = Var(type=ty, name="x")
    return HirFunction.build(
        name="main", params=(x,), body=x, return_type=ty,
        specializations=(DimVarRangePat("S", lo, hi),),
    )


def _prototype() -> HirFunction:
    ty = _s_type()
    x = Var(type=ty, name="x")
    base = HirFunction.build(name="main", params=(x,), body=None, return_type=ty)
    base.add_variant(_variant(1, 3))
    base.add_variant(_variant(4, 7))
    return base


def _normal() -> HirFunction:
    ty = _s_type()
    x = Var(type=ty, name="x")
    return HirFunction.build(name="main", params=(x,), body=x, return_type=ty)


def test_prototype_prints_pass_base_and_specialize_blocks() -> None:
    src = as_script(_prototype())
    # Base is a pass-bodied prototype.
    assert "@func\ndef main(" in src
    assert "    pass" in src
    # Each variant is a `.specialize` block over a throwaway `def _`.
    assert '@main.specialize(DimVarRangePat("S", 1, 3))' in src
    assert '@main.specialize(DimVarRangePat("S", 4, 7))' in src
    assert "def _(" in src
    # The DimVarRangePat constructor is importable in the emitted source.
    assert "from tilefoundry.ir.core.pattern import DimVarRangePat" in src


def test_normal_function_omits_specialize() -> None:
    src = as_script(_normal())
    assert ".specialize(" not in src
    assert "DimVarRangePat" not in src
    assert "    pass" not in src


def test_module_mode_prototype_shares_header_with_standalone() -> None:
    """The ``@module``-wrapped form of a dispatch prototype must emit the same
    ``DimVarRangePat`` import as the standalone form (previous test) — module
    and standalone output share one header emitter, so a construct requiring
    an extra import in one mode requires it in both.

    Per §2.7 a dispatch prototype's rendering is display-only and must not be
    used as a ``parse_script`` validation artifact, so — like the standalone
    ``test_prototype_prints_pass_base_and_specialize_blocks`` above — this
    checks emitted text + syntax validity rather than a parse round-trip.
    """
    src = as_script(_prototype(), module="M")
    assert "@module" in src and "class M:" in src
    assert "from tilefoundry.ir.core.pattern import DimVarRangePat" in src
    assert "    pass" in src
    assert '@main.specialize(DimVarRangePat("S", 1, 3))' in src
    assert '@main.specialize(DimVarRangePat("S", 4, 7))' in src
    compile(src, "<test>", "exec")
