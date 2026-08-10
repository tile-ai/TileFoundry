"""Pin the nested ``@func`` call shapes that ``extract`` refuses.

Normal calls bind arguments, recurse, and prefix contributed statements and
buffers by callee and call-site index. Failures must name the stopped callee.
Self-recursion and arity mismatches require hand-built HIR because the authoring
surface rejects forward references and bad arity. A dispatch prototype is
authorable normally, but extraction cannot resolve its ``pass`` body statically.
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
    """A hand-built self-recursive function reports its callee name."""
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
    """A callee with no body cannot be penetrated statically.

    A dispatch prototype is unresolved
    without a concrete runtime shape) cannot be penetrated statically.
    """
    with pytest.raises(ExtractError, match="_dispatch_prototype_helper"):
        extract(_call_dispatch_prototype)


def test_arity_mismatch_call_raises_naming_the_callee():
    """A call passing fewer arguments than declared reports its callee."""
    ty = make_tensor_type((2, 2), DType.f32)
    x = Var(type=ty, name="x")
    y = Var(type=ty, name="y")
    callee = Function.build(name="needs_two", params=(x, y), body=x, return_type=ty)
    only_arg = Var(type=ty, name="only_one")
    bad_call = Call(type=ty, target=callee, args=(only_arg,))
    caller = Function.build(name="caller", params=(only_arg,), body=bad_call, return_type=ty)

    with pytest.raises(ExtractError, match="needs_two"):
        extract(caller)
