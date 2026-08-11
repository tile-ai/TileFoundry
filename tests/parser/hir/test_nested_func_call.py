"""Nested ``@func`` call boundary, including a call to a child Module.

A nested ``@func`` → ``@func`` call parses to ``Call(target=hir.Function)`` and
``@register_typeinfer(Function)`` checks the arg contract against the callee's
parameters. A bare class-body binding to a Module is the same kind of call, on
that Module's entry. The real-model corpus exercises the positive same-Module
path; what it does not exercise is a malformed call site, re-elaboration of a
call chain for a sharded argument, or a child-Module callee at all.

No GPU, no codegen, no runtime.
"""

from __future__ import annotations

import pytest

from tilefoundry import func, module, prim_func
from tilefoundry.dsl import ConstTensor, DimVar, DimVarRangePat, Mesh, T, Tensor, tf
from tilefoundry.dsl.tf import add  # noqa: F401 — binds bare ``add``
from tilefoundry.ir.core import Constant, VerifyError, get_metadata
from tilefoundry.ir.hir.function import Function, elaborate
from tilefoundry.ir.hir.specialize import origin_of, specialize_function
from tilefoundry.ir.types import make_shard_tensor_type
from tilefoundry.ir.types.shard import Layout, Topology, make_mesh
from tilefoundry.ir.types.shard import Mesh as TirMesh
from tilefoundry.ir.types.shard.shard_layout import Split
from tilefoundry.module import _DECLARING
from tilefoundry.parser.base import _ModuleCallee
from tilefoundry.target import CudaTarget

N = DimVar("N", 1, 64)


@module(entry="entry")
class _Callee:
    @func
    def helper(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return tf.mul(x, x)

    @func
    def entry(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return helper(x)  # noqa: F821 — sibling binding in the class body


@module(entry="scale")
class _Scaled:
    @func
    def scale(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
        return tf.mul(x, x)


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


def test_a_same_module_helper_stays_callable_by_its_bare_binding() -> None:
    assert _Callee.entry_function().body.target is _Callee.lookup("helper")


def test_a_bare_child_binding_calls_the_attached_child_entry() -> None:
    @module(entry="outer", target=CudaTarget("nvidia.h200_sxm"))
    class _Outer:
        leaf = _Callee

        @func
        def outer(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return leaf(x)  # noqa: F821 — the class-body child binding

    (child,) = _Outer.modules
    assert child.name == "leaf"
    call = _Outer.entry_function().body
    assert isinstance(call.target, Function)
    assert call.target is child.entry_function()
    assert call.target is not _Callee.entry_function()
    assert get_metadata(call, _ModuleCallee) is None


def test_two_bindings_of_one_module_call_their_own_attached_child() -> None:
    @module(entry="both", target=CudaTarget("nvidia.h200_sxm"))
    class _TwoAliases:
        first = _Callee
        second = _Callee

        @func
        def both(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return tf.add(first(x), second(x))  # noqa: F821

    one, two = _TwoAliases.modules
    assert (one.name, two.name) == ("first", "second")
    left, right = _TwoAliases.entry_function().body.args
    assert left.target is one.entry_function()
    assert right.target is two.entry_function()


def test_two_bindings_rebuilt_at_a_call_site_keep_their_own_origin() -> None:
    @module(
        entry="both",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 4),),
    )
    class _TwoSharded:
        first = _Callee
        second = _Callee

        @func
        def both(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
                local = tf.reshard(x, (8 @ cta.tile,), "rmem")
                return tf.add(first(local), second(local))  # noqa: F821

    one, two = _TwoSharded.modules
    left, right = _TwoSharded.entry_function().body.args
    assert left.target is not right.target
    assert origin_of(left.target) is one.entry_function()
    assert origin_of(right.target) is two.entry_function()


def test_a_specialization_variant_calls_the_attached_child() -> None:
    @module(entry="dispatch", target=CudaTarget("nvidia.h200_sxm"))
    class _VariantCaller:
        leaf = _Scaled

        @func
        def dispatch(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
            pass

        @dispatch.specialize(DimVarRangePat("N", 1, 64))
        def _(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
            return leaf(x)  # noqa: F821

    (child,) = _VariantCaller.modules
    (variant,) = _VariantCaller.entry_function().variants
    assert variant.body.target is child.entry_function()
    assert variant.body.target is not _Scaled.entry_function()
    assert get_metadata(variant.body, _ModuleCallee) is None


def test_a_weight_converter_calls_the_attached_child_too() -> None:
    @module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
    class _WithConverter:
        leaf = _Callee

        @func
        def run(x: Tensor[(8,), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return tf.add(x, w)

        @run.converter("w")
        def _convert_w(w: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return leaf(w)  # noqa: F821

    (child,) = _WithConverter.modules
    ((weight, converter),) = _WithConverter.lookup("run").converters
    assert weight == "w"
    assert converter.body.target is child.entry_function()
    assert converter.body.target is not _Callee.entry_function()
    assert get_metadata(converter.body, _ModuleCallee) is None


def test_a_weight_converter_naming_no_direct_binding_is_refused() -> None:
    with pytest.raises(ValueError, match="no class-body binding attaches"):

        @module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
        class _ConverterUnattached:
            @func
            def run(x: Tensor[(8,), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return tf.add(x, w)

            @run.converter("w")
            def _convert_w(w: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return _Callee(w)


def test_a_multiply_rebuilt_child_target_is_owned_only_by_that_child() -> None:
    @module(entry="root", target=CudaTarget("nvidia.h200_sxm"))
    class _Root:
        left = _Scaled
        right = _Scaled

        @func
        def root(x: Tensor[(N,), "f32"]) -> Tensor[(N,), "f32"]:
            return tf.add(left(x), right(x))  # noqa: F821

    one, two = _Root.modules
    sized = specialize_function(one.entry_function(), {"N": 8})
    resharded = elaborate(
        sized, (make_shard_tensor_type((8,), mesh=make_mesh((4,)), attrs=(Split(0),)),)
    )
    assert resharded is not sized is not one.entry_function()
    assert one.owns(resharded, derived=True)
    assert not two.owns(resharded, derived=True)


def test_reaching_into_a_module_for_a_function_is_refused() -> None:
    with pytest.raises(VerifyError, match="called through its bare binding"):

        @module(entry="reach")
        class _ReachEntry:
            leaf = _Callee

            @func
            def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return leaf.entry(x)  # noqa: F821

    with pytest.raises(VerifyError, match="called through its bare binding"):

        @module(entry="reach")
        class _ReachHelper:
            leaf = _Callee

            @func
            def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return leaf.helper(x)  # noqa: F821

    with pytest.raises(VerifyError, match="called through its bare binding"):

        @func
        def _reach_by_class(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return _Callee.entry(x)


def test_a_module_with_no_callable_entry_is_refused_with_its_reason() -> None:
    @module
    class _NoEntry:
        @func
        def only(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return tf.add(x, x)

    with pytest.raises(VerifyError, match="calling a Module calls its entry"):

        @module(entry="reach")
        class _CallsEntryless:
            leaf = _NoEntry

            @func
            def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return leaf(x)  # noqa: F821

    @module(entry="device")
    class _PrimEntry:
        @prim_func(target=CudaTarget("nvidia.h200_sxm"))
        def device(x: Tensor[(8,), "f32"]) -> None:  # noqa: ARG001
            with TirMesh((Topology("thread", 8),), Layout(shape=(8,), strides=(1,))) as m:
                T.sync(m)

    with pytest.raises(VerifyError, match="rather than an hir Function"):

        @module(entry="reach")
        class _CallsPrim:
            leaf = _PrimEntry

            @func
            def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return leaf(x)  # noqa: F821


def test_a_module_callee_outside_a_module_class_body_is_refused() -> None:
    with pytest.raises(VerifyError, match="authored in a @module class body"):

        @func
        def _standalone(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            return _Callee(x)


def test_a_not_yet_called_bare_decorator_gives_its_body_no_child_to_reach() -> None:
    with pytest.raises(VerifyError, match="authored in a @module class body"):

        @module
        class _BareDecorated:
            leaf = _Callee

            @func
            def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return leaf(x)  # noqa: F821


def test_a_module_call_naming_no_direct_binding_is_refused() -> None:
    with pytest.raises(ValueError, match="no class-body binding attaches"):

        @module(entry="reach")
        class _Unattached:
            @func
            def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return _Callee(x)

    with pytest.raises(ValueError, match="no class-body binding attaches"):

        @module(entry="reach")
        class _ListAttached:
            kids = [_Callee]

            @func
            def reach(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return _Callee(x)


def test_a_declaration_left_open_by_a_failed_class_body_resolves_nothing() -> None:
    open_declarations = len(_DECLARING)
    try:
        with pytest.raises(RuntimeError, match="boom"):

            @module(entry="never")
            class _Boom:
                raise RuntimeError("boom")

        with pytest.raises(VerifyError, match="authored in a @module class body"):

            @func
            def _after(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
                return _Callee(x)
    finally:
        del _DECLARING[open_declarations:]


@module(entry="run")
class _Weighted:
    @func
    def run(x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return tf.matmul(x, w)


def test_a_child_call_carries_activations_and_leaves_the_constants_declared() -> None:
    @module(entry="fused", target=CudaTarget("nvidia.h200_sxm"))
    class _Fused:
        mlp = _Weighted

        @func
        def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
            return mlp(x)  # noqa: F821

    (child,) = _Fused.modules
    call = _Fused.entry_function().body
    assert len(call.args) == 1
    assert [(p.name, p.is_const) for p in call.target.params] == [("x", False), ("w", True)]
    assert call.target.params[1].type == child.weights["w"]
    assert all(not isinstance(arg, Constant) for arg in call.args)
    assert dict(_Fused.weights) == {}
    assert set(child.weights) == {"w"}


def test_a_direct_function_call_keeps_its_declared_arity() -> None:
    with pytest.raises(VerifyError, match="nested @func call arity mismatch"):

        @module(entry="root", target=CudaTarget("nvidia.h200_sxm"))
        class _Direct:
            @func
            def leaf(
                x: Tensor[(4, 8), "f32"], w: ConstTensor[(8, 8), "f32"]
            ) -> Tensor[(4, 8), "f32"]:
                return tf.matmul(x, w)

            @func
            def root(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
                return leaf(x)  # noqa: F821


def test_a_child_call_of_the_wrong_width_is_refused_in_activations() -> None:
    with pytest.raises(VerifyError, match="takes 1 activation"):

        @module(entry="fused", target=CudaTarget("nvidia.h200_sxm"))
        class _TooMany:
            mlp = _Weighted

            @func
            def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
                return mlp(x, x)  # noqa: F821


@module(entry="run")
class _Grand:
    @func
    def run(x: Tensor[(8,), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return tf.mul(x, w)


@module(entry="mid")
class _Mid:
    grand = _Grand

    @func
    def mid(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return grand(x)  # noqa: F821


def test_a_deeper_child_call_survives_a_layout_rebuild_at_the_root() -> None:
    """A grandchild call is still a child call once the root reshards its input.

    Rebuilding the middle body for a caller's layout re-elaborates the call it
    makes in turn, and by then the binding record that classified it has been
    consumed by the middle Module's own collection.
    """

    @module(
        entry="root",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 4),),
    )
    class _Root:
        mid = _Mid

        @func
        def root(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
            with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
                local = tf.reshard(x, (8 @ cta.tile,), "gmem")
                return mid(local)  # noqa: F821

    (middle,) = _Root.modules
    (grandchild,) = middle.modules
    call = _Root.entry_function().body
    assert len(call.args) == 1
    inner = call.target.body
    assert len(inner.args) == 1
    assert [p.is_const for p in inner.target.params] == [False, True]
    assert origin_of(inner.target) is grandchild.entry_function()
