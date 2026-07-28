"""What ``extract`` refuses when it walks into a nested ``@func`` call.

Penetrating a call is the ordinary path -- the walker binds the callee's params to
the caller's own argument expressions and recurses into its body, prefixing every
statement and buffer it contributes with the callee name and a per-call-site
index (``poly._walk_calls``). A real decoder layer is nothing but nested calls, so
that path is exercised wholesale by the corpus Analyze witness. What is left here
are the three shapes it cannot walk, each of which has to name the callee it
stopped at: without the name, a failure in a module of many small helpers says
only that something somewhere could not be extracted.

Self-recursion and an arity mismatch cannot be authored through the normal
``@func`` surface: ``tilefoundry.script._definition_namespace`` only resolves a
callee bound *before* its caller (no forward references), and
``hir.function.elaborate`` rejects an arity mismatch at parse time. Those two
construct HIR directly instead. A dispatch prototype call *is* authorable
normally (a ``pass`` body typechecks fine as a callee -- only extract, not
elaboration, cannot resolve it statically).
"""
from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.analysis import ExtractError, extract
from tilefoundry.dsl import Tensor
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType, make_tensor_type


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
