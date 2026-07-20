"""HIR→TIR lowering produces ``tir.DispatchCall`` for a dispatch prototype.

Covers:

1. Static-shape path: a normal Function lowers to a single PrimFunction
   with no DispatchCall in the body.
2. Entry dispatch: a prototype with two ``DimVarRangePat`` variants lowers
   to two mangled PrimFunctions + one unmangled entry PrimFunction whose
   body is a ``DispatchCall`` over both variants.
3. Sub-call dispatch: a caller's body invokes a prototype callee; the
   caller body lowers to a ``DispatchCall`` selecting between the mangled
   variant callees.
4. Empty reachable set: a caller whose argument-side range does not
   intersect any variant range raises a compile-time error.
"""
from __future__ import annotations

import pytest

from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import (
    For,
    If,
    LetStmt,
    MeshScope,
    Sequential,
    While,
)
from tilefoundry.ir.tir.verify import verify_module
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.passes.transforms import HirToTirPass


def _tensor(shape) -> TensorType:
    return TensorType(shape=shape, dtype=DType.f32, layout=None, storage="gmem")


def _has_dispatch_call(body: Sequential) -> bool:
    for s in body.body:
        if isinstance(s, DispatchCall):
            return True
    return False


def _find_function(mod: Module, name: str) -> PrimFunction:
    matches = [fn for fn in mod.functions if fn.name == name]
    assert len(matches) == 1, f"expected one function named {name!r}"
    return matches[0]


def _S(env=(1, 7)) -> DimVar:
    return DimVar(name="S", lo=env[0], hi=env[1])


def _variant(name: str, lo: int, hi: int, *, calls: HirFunction | None = None,
             env=(1, 7)) -> HirFunction:
    """A specialization variant over ``DimVar('S', *env)``. With ``calls``, its
    body is a sub-call to that callee (a prototype); otherwise it is identity."""
    ty = _tensor((_S(env),))
    x = Var(type=ty, name="x")
    body = Call(type=ty, target=calls, args=(x,)) if calls is not None else x
    return HirFunction.build(
        name=name, params=(x,), body=body, return_type=ty,
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


def test_static_function_lowers_without_dispatch_call() -> None:
    ty = _tensor((8,))
    x = Var(type=ty, name="x")
    fn = HirFunction.build(name="static_fn", params=(x,), body=x, return_type=ty)
    mod = Module(name="m", functions=(fn,), entry="static_fn")
    out = HirToTirPass().run(mod)

    assert len(out.functions) == 1
    pf = out.functions[0]
    assert isinstance(pf, PrimFunction)
    assert pf.name == "static_fn"
    assert not _has_dispatch_call(pf.body)
    verify_module(list(out.functions))


def test_entry_dispatch_two_arms() -> None:
    proto = _prototype(
        "main", (_variant("main", 1, 3), _variant("main", 4, 7)),
    )
    mod = Module(name="m", functions=(proto,), entry="main")
    out = HirToTirPass().run(mod)

    names = sorted(fn.name for fn in out.functions)
    assert names == ["main", "main$S$1_3", "main$S$4_7"]

    entry = _find_function(out, "main")
    # Entry must have an extra <param>_shape_<axis>: i32 kernel param.
    assert entry.params[-1].name == "x_shape_0"
    assert entry.params[-1].type == TensorType.scalar(dtype=DType.i32)
    # Body is a single DispatchCall + Return.
    body_stmts = entry.body.body
    assert isinstance(body_stmts[0], DispatchCall)
    dc = body_stmts[0]
    assert dc.callee_name == "main"
    assert isinstance(dc.subjects[0], ShapeOf)
    assert dc.subjects[0].param is entry.params[0]
    assert dc.subjects[0].axis == 0
    assert dc.case_patterns == (
        (DimVarRangePat("S", 1, 3),),
        (DimVarRangePat("S", 4, 7),),
    )
    assert dc.case_calls[0].callable.name == "main$S$1_3"
    assert dc.case_calls[1].callable.name == "main$S$4_7"

    verify_module(list(out.functions))


def test_sub_call_dispatch_emits_dispatch_call() -> None:
    inner = _prototype(
        "inner", (_variant("inner", 1, 3), _variant("inner", 4, 7)),
    )
    ty = _tensor((_S(),))
    xm = Var(type=ty, name="x")
    main = HirFunction.build(
        name="main", params=(xm,),
        body=Call(type=ty, target=inner, args=(xm,)), return_type=ty,
    )
    mod = Module(name="m", functions=(inner, main), entry="main")
    out = HirToTirPass().run(mod)

    # Mangled inner callees + entries, plus the (static) main holding the
    # sub-call dispatch.
    names = sorted(fn.name for fn in out.functions)
    assert names == [
        "inner",
        "inner$S$1_3",
        "inner$S$4_7",
        "main",
    ]

    # main (a normal function) holds the sub-call DispatchCall into inner.
    caller_pf = _find_function(out, "main")
    dispatches: list[DispatchCall] = []

    def walk(stmt) -> None:
        if isinstance(stmt, Sequential):
            for s in stmt.body:
                walk(s)
        elif isinstance(stmt, DispatchCall):
            dispatches.append(stmt)
        elif isinstance(stmt, LetStmt):
            walk(stmt.body)
        elif isinstance(stmt, (For, While, MeshScope)):
            walk(stmt.body)
        elif isinstance(stmt, If):
            walk(stmt.then_body)
            walk(stmt.else_body)

    walk(caller_pf.body)
    assert len(dispatches) == 1
    dc = dispatches[0]
    assert dc.callee_name == "inner"
    assert {c.callable.name for c in dc.case_calls} == {
        "inner$S$1_3", "inner$S$4_7",
    }
    verify_module(list(out.functions))


def test_nested_dispatch_chain_three_levels() -> None:
    """3-level chain ``main -> inner -> leaf``, each a 2-arm dispatch group.

    Each dispatch-level sub-call must forward the trailing
    ``<param>_shape_<axis>`` kernel scalars its callee declared, so the
    full chain verifies and each ``Evaluate(SymbolRef, args)``'s ``args``
    match the callee ``PrimFunction.params`` at every level.
    """
    # leaf: 2 variants (identity)
    leaf = _prototype(
        "leaf", (_variant("leaf", 1, 3), _variant("leaf", 4, 7)),
    )
    # inner: 2 variants, each sub-calls the leaf prototype
    inner = _prototype(
        "inner",
        (_variant("inner", 1, 3, calls=leaf),
         _variant("inner", 4, 7, calls=leaf)),
    )
    # main: 2 variants, each sub-calls the inner prototype
    main = _prototype(
        "main",
        (_variant("main", 1, 3, calls=inner),
         _variant("main", 4, 7, calls=inner)),
    )

    mod = Module(
        name="m", functions=(leaf, inner, main), entry="main",
    )
    out = HirToTirPass().run(mod)

    # Locate inner mangled PrimFunctions and check each sub-call refs leaf$...
    for inner_name in ("inner$S$1_3", "inner$S$4_7"):
        inner_pf = _find_function(out, inner_name)
        dispatches: list[DispatchCall] = []

        def walk(stmt) -> None:
            if isinstance(stmt, Sequential):
                for s in stmt.body:
                    walk(s)
            elif isinstance(stmt, DispatchCall):
                dispatches.append(stmt)
            elif isinstance(stmt, LetStmt):
                walk(stmt.body)
            elif isinstance(stmt, (For, While, MeshScope)):
                walk(stmt.body)
            elif isinstance(stmt, If):
                walk(stmt.then_body)
                walk(stmt.else_body)

        walk(inner_pf.body)
        assert len(dispatches) == 1
        dc = dispatches[0]
        for cc in dc.case_calls:
            assert cc.callable.name.startswith("leaf$")
            assert len(cc.args) == len(cc.callable.type.parameters)

    # main entry must dispatch into mangled main$... variants.
    main_entry = _find_function(out, "main")
    dc = main_entry.body.body[0]
    assert isinstance(dc, DispatchCall)
    for cc in dc.case_calls:
        assert cc.callable.name.startswith("main$")
        assert len(cc.args) == len(cc.callable.type.parameters)

    # mangled main variants must each carry a sub-call DispatchCall into inner$...
    for main_name in ("main$S$1_3", "main$S$4_7"):
        main_pf = _find_function(out, main_name)
        dispatches = []

        def walk(stmt) -> None:
            if isinstance(stmt, Sequential):
                for s in stmt.body:
                    walk(s)
            elif isinstance(stmt, DispatchCall):
                dispatches.append(stmt)
            elif isinstance(stmt, LetStmt):
                walk(stmt.body)
            elif isinstance(stmt, (For, While, MeshScope)):
                walk(stmt.body)
            elif isinstance(stmt, If):
                walk(stmt.then_body)
                walk(stmt.else_body)

        walk(main_pf.body)
        assert len(dispatches) == 1
        dc = dispatches[0]
        for cc in dc.case_calls:
            assert cc.callable.name.startswith("inner$")
            assert len(cc.args) == len(cc.callable.type.parameters)

    verify_module(list(out.functions))


def test_empty_reachable_set_raises() -> None:
    inner = _prototype(
        "inner", (_variant("inner", 1, 3), _variant("inner", 4, 7)),
    )
    callee_ty = _tensor((_S(),))
    # Caller's arg-side DimVar carries [100, 200] — disjoint from both callee
    # ranges. A distinct name keeps the module-wide bound rule satisfied.
    T = DimVar(name="T", lo=100, hi=200)
    xm = Var(type=_tensor((T,)), name="x")
    main = HirFunction.build(
        name="main", params=(xm,),
        body=Call(type=callee_ty, target=inner, args=(xm,)),
        return_type=callee_ty,
    )
    mod = Module(name="m", functions=(inner, main), entry="main")
    with pytest.raises(TypeError, match="empty reachable"):
        HirToTirPass().run(mod)
