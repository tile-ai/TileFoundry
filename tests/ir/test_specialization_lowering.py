"""Cover HIR-to-TIR dispatch lowering for specialized prototypes.

Cases distinguish static functions, entry and nested dispatch, mangled variant
symbols, and an empty reachable variant set.

See [passes §7.1](docs/spec/passes.md#71-hirtotirpass).
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Call, Var, VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.tir.verify import verify_module, verify_prim_function
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import StmtVisitor, walk_prim_function
from tilefoundry.passes.transforms import HirToTirPass


def _tensor(shape) -> TensorType:
    return TensorType(shape=shape, dtype=DType.f32, layout=None, storage="gmem")



def _find_function(mod: Module, name: str) -> PrimFunction:
    matches = [fn for fn in mod.functions if fn.name == name]
    assert len(matches) == 1, f"expected one function named {name!r}"
    return matches[0]

def _callees(pf: PrimFunction) -> set[str]:
    names: set[str] = set()
    class _V(StmtVisitor):
        def visit_Evaluate(self, stmt):
            if isinstance(stmt.callable, SymbolRef):
                names.add(stmt.callable.name)
            return self.generic_visit(stmt)
    walk_prim_function(_V(), pf)
    return names


def _S(env=(1, 7)) -> DimVar:
    return DimVar(name="S", lo=env[0], hi=env[1])


def _variant(
    name: str, lo: int, hi: int, *, calls: HirFunction | None = None, env=(1, 7)
) -> HirFunction:
    """A specialization variant over ``DimVar('S', *env)``.

    A specialization variant over ``DimVar('S', *env)``. With ``calls``, its
    body is a sub-call to that callee (a prototype); otherwise it is identity.
    """
    ty = _tensor((_S(env),))
    x = Var(type=ty, name="x")
    body = Call(type=ty, target=calls, args=(x,)) if calls is not None else x
    return HirFunction.build(
        name=name,
        params=(x,),
        body=body,
        return_type=ty,
        specializations=(DimVarRangePat("S", lo, hi),),
    )


def _prototype(name: str, variants: tuple[HirFunction, ...], env=(1, 7)) -> HirFunction:
    """A dispatch prototype (``body=None``) carrying ``variants``."""
    ty = _tensor((_S(env),))
    x = Var(type=ty, name="x")
    base = HirFunction.build(name=name, params=(x,), body=None, return_type=ty)
    for v in variants:
        base.add_variant(v)
    return base


def test_static_function_lowers_without_variants() -> None:
    ty = _tensor((8,))
    x = Var(type=ty, name="x")
    fn = HirFunction.build(name="static_fn", params=(x,), body=x, return_type=ty)
    mod = Module(name="m", functions=(fn,), entry="static_fn")
    out = HirToTirPass().run(mod)

    assert len(out.functions) == 1
    pf = out.functions[0]
    assert isinstance(pf, PrimFunction)
    assert pf.name == "static_fn"
    assert not pf.variants
    verify_module(list(out.functions))


def test_entry_dispatch_two_arms() -> None:
    proto = _prototype(
        "main",
        (_variant("main", 1, 3), _variant("main", 4, 7)),
    )
    mod = Module(name="m", functions=(proto,), entry="main")
    out = HirToTirPass().run(mod)

    names = sorted(fn.name for fn in out.functions)
    assert names == ["main"]

    entry = _find_function(out, "main")

    assert entry.params[-1].name == "x_shape_0"
    assert entry.params[-1].type == TensorType.scalar(
        dtype=DType.i32, storage=StorageKind.RMEM
    )

    assert len(entry.variants) == 2
    assert tuple(v.specializations for v in entry.variants) == ((DimVarRangePat("S", 1, 3),), (DimVarRangePat("S", 4, 7),))
    assert tuple(v.name for v in entry.variants) == ("main$S$1_3", "main$S$4_7")

    verify_module(list(out.functions))


def test_sub_call_group_lowers_to_variants() -> None:
    inner = _prototype(
        "inner",
        (_variant("inner", 1, 3), _variant("inner", 4, 7)),
    )
    ty = _tensor((_S(),))
    xm = Var(type=ty, name="x")
    main = HirFunction.build(
        name="main",
        params=(xm,),
        body=Call(type=ty, target=inner, args=(xm,)),
        return_type=ty,
    )
    mod = Module(name="m", functions=(inner, main), entry="main")
    out = HirToTirPass().run(mod)

    names = sorted(fn.name for fn in out.functions)
    assert names == ["inner", "main"]

    inner_pf = _find_function(out, "inner")
    assert len(inner_pf.variants) == 2
    assert {v.name for v in inner_pf.variants} == {"inner$S$1_3", "inner$S$4_7"}
    caller = _find_function(out, "main")
    assert _callees(caller) >= {"inner$S$1_3", "inner$S$4_7"}
    verify_module(list(out.functions))


def test_nested_dispatch_chain_three_levels() -> None:
    """3-level chain ``main -> inner -> leaf``, each a 2-arm dispatch group.

    ``verify_module`` checks that every symbol call forwards the correct
    parameters at each level.
    """
    leaf = _prototype(
        "leaf",
        (_variant("leaf", 1, 3), _variant("leaf", 4, 7)),
    )

    inner = _prototype(
        "inner",
        (_variant("inner", 1, 3, calls=leaf), _variant("inner", 4, 7, calls=leaf)),
    )

    main = _prototype(
        "main",
        (_variant("main", 1, 3, calls=inner), _variant("main", 4, 7, calls=inner)),
    )

    mod = Module(
        name="m",
        functions=(leaf, inner, main),
        entry="main",
    )
    out = HirToTirPass().run(mod)

    inner_proto = _find_function(out, "inner")
    for inner_pf in inner_proto.variants:
        assert _callees(inner_pf) >= {"leaf$S$1_3", "leaf$S$4_7"}

    main_entry = _find_function(out, "main")
    assert len(main_entry.variants) == 2
    assert all(_callees(v) >= {"inner$S$1_3", "inner$S$4_7"} for v in main_entry.variants)

    verify_module(list(out.functions))


def test_variant_requires_single_specialization() -> None:
    x = Var(type=_tensor((_S(),)), name="x")
    variant = PrimFunction(
        name="f$S$1_3", params=(x,), body=Sequential(body=()),
        specializations=(DimVarRangePat("S", 1, 3), DimVarRangePat("S", 3, 5)),
    )
    fn = PrimFunction(name="f", params=(x,), body=Sequential(body=()), variants=(variant,))
    with pytest.raises(VerifyError, match="one DimVarRangePat"):
        verify_prim_function(fn)


def test_variant_subject_must_be_in_parameters() -> None:
    x = Var(type=_tensor((8,)), name="x")
    variant = PrimFunction(
        name="f$S$1_3", params=(x,), body=Sequential(body=()),
        specializations=(DimVarRangePat("S", 1, 3),),
    )
    fn = PrimFunction(name="f", params=(x,), body=Sequential(body=()), variants=(variant,))
    with pytest.raises(VerifyError, match="cannot be derived"):
        verify_prim_function(fn)


def test_empty_reachable_set_raises() -> None:
    inner = _prototype(
        "inner",
        (_variant("inner", 1, 3), _variant("inner", 4, 7)),
    )
    callee_ty = _tensor((_S(),))

    T = DimVar(name="T", lo=100, hi=200)
    xm = Var(type=_tensor((T,)), name="x")
    main = HirFunction.build(
        name="main",
        params=(xm,),
        body=Call(type=callee_ty, target=inner, args=(xm,)),
        return_type=callee_ty,
    )
    mod = Module(name="m", functions=(inner, main), entry="main")
    with pytest.raises(TypeError, match="empty reachable"):
        HirToTirPass().run(mod)
