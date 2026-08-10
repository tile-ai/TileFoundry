"""Nested ``@func`` call boundary.

A nested ``@func`` → ``@func`` call parses to ``Call(target=hir.Function)`` and
``@register_typeinfer(Function)`` checks the arg contract against the callee's
parameters. The real-model corpus exercises the positive path (a decoder layer
calling its submodules, printed, re-imported and evaluated); what it does not
exercise is a malformed call site, and re-elaboration of a call chain for a
sharded argument.

No GPU, no codegen, no runtime.
"""

from __future__ import annotations

import pytest

from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl.tf import add  # noqa: F401 — binds bare ``add``
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.hir.function import elaborate
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.shard_layout import Split

N = DimVar("N", 1, 64)


@func
def _inner_double(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
    return add(x, x)  # noqa: F821 — bound via ``from tilefoundry.dsl.tf import add``


def test_arity_mismatch_rejected_at_parse_time() -> None:

    with pytest.raises(VerifyError, match="arity mismatch"):

        @func
        def _bad_arity(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
            return _inner_double(x, x)  # type: ignore[call-arg]  # noqa: F841


def test_arg_type_mismatch_rejected_at_typeinfer() -> None:

    with pytest.raises(VerifyError, match="type mismatch"):

        @func
        def _bad_dtype(x: Tensor[(N,), "bf16"]) -> Tensor[(N,), "f32"]:
            return _inner_double(x)  # noqa: F841


def test_wildcard_chain_reelaborates_nested_call_target() -> None:

    @func
    def leaf(x: Tensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        return add(x, x)  # noqa: F821

    @func
    def mid(x: Tensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        return leaf(x)

    @func
    def outer_fn(x: Tensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
        return mid(x)

    x_split = make_shard_tensor_type((8, 64), mesh=make_mesh((4,)), attrs=(Split(0),))
    new_outer = elaborate(outer_fn, (x_split,))
    tgt = new_outer.body.target
    assert tgt is not mid
    assert tgt.params[0].type == x_split
    assert tgt.body.type == x_split
