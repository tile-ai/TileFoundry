"""Canonical demo fixture builds + printer round-trip preserves topologies."""

from __future__ import annotations

from tests.fixtures.demo_canonical import build_demo_canonical
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import TensorType
from tilefoundry.parser.hir_parser import parse_script


def _structural_equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, Function):
        return (
            a.name == b.name
            and len(a.params) == len(b.params)
            and all(_structural_equal(pa, pb) for pa, pb in zip(a.params, b.params))
            and _structural_equal(a.body, b.body)
            and _structural_equal(a.return_type, b.return_type)
        )
    if isinstance(a, Var):
        return a.name == b.name and _structural_equal(a.type, b.type)
    if isinstance(a, Constant):
        return a.value == b.value
    if isinstance(a, Call):
        if type(a.target) is not type(b.target) or len(a.args) != len(b.args):
            return False
        if not all(_structural_equal(aa, bb) for aa, bb in zip(a.args, b.args)):
            return False
        for pi in type(a.target).params():
            if pi.kind == "attribute":
                if getattr(a.target, pi.name, None) != getattr(b.target, pi.name, None):
                    # Best-effort tuple compare for layouts / nested attrs.
                    if not _attr_equal(
                        getattr(a.target, pi.name, None),
                        getattr(b.target, pi.name, None),
                    ):
                        return False
        return _structural_equal(a.type, b.type)
    if isinstance(a, TensorType):
        return (
            a.shape == b.shape and a.dtype == b.dtype and a.storage == b.storage
            and _attr_equal(a.layout, b.layout)
        )
    return a == b


def _attr_equal(a, b) -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_attr_equal(aa, bb) for aa, bb in zip(a, b))
    if hasattr(a, "__dataclass_fields__"):
        return all(
            _attr_equal(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__
        )
    return a == b


def test_canonical_fixture_builds_with_topologies_and_ssa_chain() -> None:
    """Canonical fixture has 2 topologies + reshard→relu→reshard SSA chain."""

    fn = build_demo_canonical()
    assert isinstance(fn, Function)
    assert {(t.name, t.size) for t in fn.topologies} == {("cta", 128), ("thread", 256)}

    def count_reshard(expr):
        if isinstance(expr, Call):
            c = 1 if isinstance(expr.target, Reshard) else 0
            for arg in expr.args:
                c += count_reshard(arg)
            return c
        return 0

    assert count_reshard(fn.body) == 3  # shared, reg, global


def test_canonical_roundtrip_preserves_topologies_and_compiles() -> None:
    """Round-trip is structurally equal + printed text is valid Python +
    the @func(topologies=...) declaration survives."""

    fn1 = build_demo_canonical()
    src = as_script(fn1)
    compile(src, "<test>", "exec")
    assert "@func(topologies=(" in src
    assert 'Topology("cta", 128)' in src

    fn2 = parse_script(src)
    assert _structural_equal(fn1, fn2)


def test_canonical_module_form_roundtrip() -> None:
    """``@module`` form preserves topologies in the class-body @func decorator."""

    fn1 = build_demo_canonical()
    src = as_script(fn1, module="M")
    assert "@module" in src and "class M:" in src and "@func(topologies=(" in src
    fn2 = parse_script(src)
    assert len(fn2.topologies) == 2
