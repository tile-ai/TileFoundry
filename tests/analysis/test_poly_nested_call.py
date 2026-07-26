"""``extract`` coverage for penetrating a nested ``@func`` call -- M1's
replacement for the deleted ``inline_calls`` pre-flattening pass. Instead
of rejecting ``Call(target=Function)``, the walker binds the callee's
params to the caller's own argument expressions and recurses straight
into its body, prefixing every statement/buffer it contributes with the
callee name plus a per-call-site index (``poly._walk_calls``).

Self-recursion and an arity mismatch cannot be authored through the
normal ``@func`` surface: ``tilefoundry.script._definition_namespace``
only resolves a callee bound *before* its caller (no forward references),
and ``hir.function.elaborate`` rejects an arity mismatch at parse time.
Those two cases construct HIR directly instead, mirroring
``tests/ir/test_function_call_typeinfer.py``'s own hand-built
``Function``/``Call`` pattern. A dispatch prototype call *is* authorable
normally (a ``pass`` body typechecks fine as a callee -- only extract, not
elaboration, cannot resolve it statically).
"""
from __future__ import annotations

from pathlib import Path

import isl
import pytest

from tests.models.qwen3_1_7b import decoder_layer as qwen3
from tilefoundry import func
from tilefoundry.analysis import ExtractError, TileGraph, extract
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul resolved dynamically
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType, make_tensor_type


def _write_bufs(tg: TileGraph, stmt_name: str) -> list[str]:
    maps: list["isl.map"] = []
    tg.writes.foreach_map(maps.append)
    return [
        m.get_tuple_name(isl.dim_type.OUT)
        for m in maps
        if m.get_tuple_name(isl.dim_type.IN) == stmt_name
    ]


def test_self_attention_penetrates_input_rms_norm_with_prefixed_names():
    """AC-1-1: ``self_attention`` calls the ``input_rms_norm`` helper --
    extraction walks straight through it (no ``ExtractError``), the
    helper's own RMSNorm op shows up as a statement, and both its
    statement name and its output buffer carry the ``input_rms_norm0_``
    call-site prefix so a hole is traceable to the helper it came from."""
    tg = extract(qwen3.self_attention)
    assert isinstance(tg, TileGraph)

    stmt_names = [u.name for u in tg.units]
    penetrated = [n for n in stmt_names if n.startswith("input_rms_norm0_")]
    assert penetrated, f"no penetrated statement in {stmt_names}"

    penetrated_units = [u for u in tg.units if u.name in penetrated]
    assert {type(u.op.target).__name__ for u in penetrated_units} == {"RMSNorm"}

    # input_rms_norm's own (penetrated) RMSNorm + self_attention's own
    # per-head q_norm/k_norm: three RMSNorm statements total.
    op_names = [type(u.op.target).__name__ for u in tg.units]
    assert op_names.count("RMSNorm") == 3

    bufs = _write_bufs(tg, penetrated_units[0].name)
    assert bufs and all(b.startswith("input_rms_norm0_") for b in bufs)


def test_no_inline_calls_anywhere_in_the_repo():
    """AC-1-1's other half: the deleted ``inline_calls`` pass has no
    remaining caller anywhere under ``src/``."""
    src_root = Path(__file__).resolve().parents[2] / "src"
    hits = [p for p in src_root.rglob("*.py") if "inline_calls" in p.read_text(encoding="utf-8")]
    assert hits == []


_DYN_ROWS = DimVar("dyn_rows", 1, 128)


@func
def _scale_rows_helper(
    x: Tensor[(_DYN_ROWS, 4), "bf16"], w: Tensor[(4, 2), "bf16"],
) -> Tensor[(_DYN_ROWS, 2), "bf16"]:
    return matmul(x, w)


@func
def _call_scale_rows_helper(
    x: Tensor[(_DYN_ROWS, 4), "bf16"], w: Tensor[(4, 2), "bf16"],
) -> Tensor[(_DYN_ROWS, 2), "bf16"]:
    return _scale_rows_helper(x, w)


def test_penetrated_helper_binds_dim_var_from_caller_shape():
    """AC-1-2: the helper's ``DimVar``-carrying parameter (``dyn_rows``)
    binds from the caller's actual argument shape, extraction succeeds,
    and the parameter's name appears in the returned ``TileGraph.params``.
    """
    tg = extract(_call_scale_rows_helper)
    assert isinstance(tg, TileGraph)
    assert "dyn_rows" in tg.params

    stmt_names = [u.name for u in tg.units]
    assert any(n.startswith("_scale_rows_helper0_") for n in stmt_names)


def test_self_recursive_call_raises_naming_the_callee():
    """AC-1-3: a hand-built ``Function`` whose body calls itself."""
    ty = make_tensor_type((2, 2), DType.f32)
    x = Var(type=ty, name="x")
    stub = Function.build(name="loopy", params=(x,), body=x, return_type=ty)
    object.__setattr__(stub, "body", Call(type=ty, target=stub, args=(x,)))

    with pytest.raises(ExtractError, match="loopy"):
        extract(stub)


@func
def _dispatch_prototype_helper(x: Tensor[(4, 4), "f32"]) -> Tensor[(4, 4), "f32"]:
    pass


@func
def _call_dispatch_prototype(x: Tensor[(4, 4), "f32"]) -> Tensor[(4, 4), "f32"]:
    return _dispatch_prototype_helper(x)


def test_dispatch_prototype_call_raises_naming_the_callee():
    """AC-1-3: a callee with no body (dispatch by variant, unresolved
    without a concrete runtime shape) cannot be penetrated statically."""
    with pytest.raises(ExtractError, match="_dispatch_prototype_helper"):
        extract(_call_dispatch_prototype)


def test_arity_mismatch_call_raises_naming_the_callee():
    """AC-1-3: a hand-built call passing fewer args than the callee
    declares."""
    ty = make_tensor_type((2, 2), DType.f32)
    x = Var(type=ty, name="x")
    y = Var(type=ty, name="y")
    callee = Function.build(name="needs_two", params=(x, y), body=x, return_type=ty)
    only_arg = Var(type=ty, name="only_one")
    bad_call = Call(type=ty, target=callee, args=(only_arg,))
    caller = Function.build(name="caller", params=(only_arg,), body=bad_call, return_type=ty)

    with pytest.raises(ExtractError, match="needs_two"):
        extract(caller)
