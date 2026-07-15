"""Nested ``@func`` call support.

Locks the DSL surface for nested ``@func`` → ``@func`` calls, the
``@register_typeinfer(Function)`` arg-contract handler, and the
viewer's inline-expansion of ``Call(target=hir.Function)``:

1. **Parser** produces ``Call(target=hir.Function, args=...)`` for a
   nested ``@func`` call site (no fixture-only inline fallback, no
   placeholder Op).
2. **TypeInfer** validates arg count + each arg's type against the
   callee's parameter types, and returns the callee's
   ``return_type``.
3. **Viewer** inline-expands the callee subgraph into the caller graph.

No GPU, no codegen, no runtime.
"""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl.tf import (  # noqa: F401 — binds bare ``add``, ``mul``
    add,
    mul,
)
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.expr import Call
from tilefoundry.ir.hir.function import Function as HirFunction
from tilefoundry.ir.hir.function import elaborate
from tilefoundry.ir.types.shard.shard_layout import Split
from tests.ops.typeinfer_utils import mesh, sharded

# ---------------------------------------------------------------------------
# Fixtures — two ``@func``s where the outer one calls the inner one.
# ---------------------------------------------------------------------------


N = DimVar("N", 1, 64)


@func
def _inner_double(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
    return add(x, x)  # noqa: F821 — bound via ``from tilefoundry.dsl.tf import *``


@func
def _outer_call_inner(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
    return _inner_double(x)


# ---------------------------------------------------------------------------
# Parser produces ``Call(target=hir.Function)``.
# ---------------------------------------------------------------------------


def test_parser_emits_call_with_hir_function_target() -> None:
    outer_ir = _outer_call_inner
    body = outer_ir.body
    # The outer body is the call expression directly (single-return form).
    assert isinstance(body, Call)
    assert isinstance(body.target, HirFunction), (
        f"expected Call.target to be hir.Function, got "
        f"{type(body.target).__name__}"
    )
    # The callee is the same canonical Function instance the inner
    # ``@func`` produced (no clone, no surrogate).
    inner_ir = _inner_double
    assert body.target is inner_ir


# ---------------------------------------------------------------------------
# TypeInfer threads callee return_type and enforces arg contract.
# ---------------------------------------------------------------------------


def test_call_type_matches_callee_return_type() -> None:
    outer_ir = _outer_call_inner
    inner_ir = _inner_double
    assert outer_ir.body.type == inner_ir.return_type


def test_arity_mismatch_rejected_at_parse_time() -> None:
    # The parser enforces the arity hard so we don't even reach
    # typeinfer with a malformed Call.
    with pytest.raises(VerifyError, match="arity mismatch"):

        @func
        def _bad_arity(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
            return _inner_double(x, x)  # type: ignore[call-arg]  # noqa: F841


def test_wildcard_chain_reelaborates_nested_call_target() -> None:
    # 3-level wildcard chain outer -> mid -> leaf, elaborated for a Split
    # arg: Call.target must be the fresh Split instance at every level (a
    # viewer/printer reads call.target.body), not the parse-time unsharded
    # sibling Function that ``@func`` originally produced.
    @func
    def leaf(x: Tensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        return add(x, x)  # noqa: F821

    @func
    def mid(x: Tensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        return leaf(x)

    @func
    def outer_fn(x: Tensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        return mid(x)

    x_split = sharded((8, 64), (Split(0),), mesh((4,)))
    new_outer = elaborate(outer_fn, (x_split,))
    tgt = new_outer.body.target
    assert tgt is not mid
    assert tgt.params[0].type == x_split
    assert tgt.body.type == x_split


def test_two_parse_time_call_sites_share_target_instance() -> None:
    # Two call sites of the same template with identical arg types must
    # reference the same Function instance (identity), not just an equal one.
    @func
    def leaf2(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        return add(x, x)  # noqa: F821

    @func
    def outer_two_calls(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        p = leaf2(x)
        q = leaf2(x)
        return add(p, q)  # noqa: F821

    body = outer_two_calls.body
    assert body.args[0].target is body.args[1].target


def test_reelaboration_same_args_share_target_instance() -> None:
    # Same property under re-elaboration: the elaboration cache scoped to
    # one elaborate() tree walk must dedup identical-typed call sites too.
    @func
    def leaf3(x: Tensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        return add(x, x)  # noqa: F821

    @func
    def outer_two_calls_split(x: Tensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        p = leaf3(x)
        q = leaf3(x)
        return add(p, q)  # noqa: F821

    x_split = sharded((8, 64), (Split(0),), mesh((4,)))
    inst = elaborate(outer_two_calls_split, (x_split,))
    body = inst.body
    assert body.args[0].target is body.args[1].target


def test_arg_type_mismatch_rejected_at_typeinfer() -> None:
    # Callee declares ``Tensor[(N,), "f32"]`` but caller passes
    # ``Tensor[(N,), "bf16"]`` — typeinfer must surface the
    # parameter-type mismatch.
    with pytest.raises(VerifyError, match="type mismatch"):

        @func
        def _bad_dtype(x: Tensor[(N,), "bf16"]) -> Tensor[(N,), "f32"]:
            return _inner_double(x)  # noqa: F841
