"""``@module(entry=...)`` decorator: collect class-body DSL functions into a Module.

The decorator scans the class for ``@func`` / ``@prim_func`` members (which are
``hir.Function`` / ``tir.PrimFunction`` values), builds a ``Module`` in
definition order, and binds the decorated name to it. The class body is a pure
function container: a non-dunder, non-DSL member is rejected; ``entry`` is an
explicit, required name that must resolve to a collected function. A composed
member may call siblings defined above it (the call lowers to a ``Call``
targeting the sibling); forward references stay unresolved and fail loudly.
"""

from __future__ import annotations

import dataclasses

import pytest

from tilefoundry import func, module, prim_func
from tilefoundry.dsl import T, Tensor, tf  # noqa: F401 — tf/T used by bodies
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.target import CpuTarget, CudaTarget
from tilefoundry.utils.spec_ref import spec_ref_render


@module(entry="composed")
class _Demo:
    @func
    def leaf(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)

    @func
    def composed(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        y = leaf(x, g)
        return tf.add(y, y)


def test_module_collects_functions_in_order_and_resolves_them_by_name():
    """Test module collects functions in order and resolves them by name.

    Members land on the Module in definition order and each collected name
    resolves through ``lookup`` (the IR-node path — bare attribute access
    instead returns a runnable callable, see ``ir.core.module.Module``); a name
    the class never declared raises ``AttributeError``.
    """
    assert _Demo.name == "_Demo"
    assert [fn.name for fn in _Demo.functions] == ["leaf", "composed"]
    assert _Demo.lookup("leaf").name == "leaf"
    assert _Demo.lookup("composed").name == "composed"
    with pytest.raises(AttributeError):
        _Demo.not_a_function


def test_attribute_access_ambiguous_name_and_real_fields():
    """Test attribute access ambiguous name and real fields.

    A duplicated function name is ambiguous under attribute access (raises),
    real Module fields are never intercepted, and ``function_named`` returns all
    matches — the core-ir [parser §2](docs/spec/parser.md#2-syntax-and-rules) ambiguity rule.
    """
    base = _Demo.lookup("leaf")
    dup_a = dataclasses.replace(base, name="dup")
    dup_b = dataclasses.replace(base, name="dup")
    mod = Module(name="Dup", functions=(dup_a, dup_b), entry="dup")
    assert mod.name == "Dup"
    assert len(mod.functions) == 2
    with pytest.raises(AttributeError):
        mod.dup
    assert len(mod.function_named("dup")) == 2


def test_collects_orchestration_method_and_rejects_other_members():
    """A plain Python method is the third member kind.

    A plain Python method is the third member kind — an orchestration method,
    bound on the resulting Module. A member that is none of the three kinds
    (DSL function / child Module / plain function) is still rejected.
    """

    @module(entry="only")
    class _WithMethod:
        @func
        def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
            return tf.rms_norm(x, g)

        def describe(self):  # noqa: ANN001 — orchestration method, bound on the Module
            return f"{self.name}/{self.entry}"

    assert _WithMethod.describe() == "_WithMethod/only"

    with pytest.raises(TypeError, match="only these three member kinds"):

        @module(entry="only")
        class _BadMember:
            @func
            def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.rms_norm(x, g)

            budget = 3


def test_what_a_class_body_must_declare_to_be_a_module():
    """Only an empty body is refused: one function, one child, or one plain method is enough.

    Only an empty body is refused: one function, one child, or one plain method
    is enough, so a body of methods alone — refused before for owning no function —
    is valid. A supplied ``entry`` is still checked, and a class-body ``__call__``
    is refused rather than dropped.
    """
    with pytest.raises(TypeError, match="empty class body"):

        @module
        class _Empty:
            pass

    @module
    class _MethodsOnly:
        def walk(self, x):
            return x

    assert _MethodsOnly.entry is None
    assert _MethodsOnly.walk(3) == 3

    with pytest.raises(ValueError, match=r"entry 'absent' names no collected function"):

        @module(entry="absent")
        class _WrongEntry:
            @func
            def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.rms_norm(x, g)

    with pytest.raises(TypeError, match=r"__call__ has no effect"):

        @module(entry="only")
        class _OwnCall:
            @func
            def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.rms_norm(x, g)

            def __call__(self, x):
                raise AssertionError("never reached")


def test_a_module_without_a_default_step_says_so_rather_than_blaming_entry():
    """A bare call is answered by naming what to call.

    A bare call is answered by naming what to call; asking for the entry is
    answered by saying there is no default step. Neither may read as ``entry``
    being wrong, which is what ``None`` reaching the entry lookup produces.
    """

    @module
    class _NoStep:
        @func
        def helper(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
            return tf.rms_norm(x, g)

    with pytest.raises(TypeError, match=r"no forward method and no entry.*helper"):
        _NoStep()

    with pytest.raises(ValueError, match=r"declares no entry, so it has no default step"):
        _NoStep.entry_function()

    assert _NoStep.lookup("helper").name == "helper"


def test_the_runner_on_an_authored_module_takes_the_weights_too():
    """A Module's callable takes every declared param, a ``ConstTensor`` one included.

    A Module's callable takes every declared param, a ``ConstTensor`` one
    included; a ``LoadedModule``'s takes activations alone. Each wrong argument
    list is refused naming the runner it wanted, and neither is sent to the
    other's.
    """
    import torch  # noqa: PLC0415 — only this test needs a real tensor

    from tilefoundry.dsl import ConstTensor  # noqa: PLC0415
    from tilefoundry.runtime.resource import DictResource  # noqa: PLC0415

    @module(entry="scale")
    class _Weighted:
        @func
        def scale(x: Tensor[(2,), "f32"], w: ConstTensor[(2,), "f32"]):
            return x * w

    ones = torch.ones(2, dtype=torch.float32)
    weight = torch.full((2,), 3.0)

    assert _Weighted.scale(ones, weight).float().cpu().tolist() == [3.0, 3.0]

    with pytest.raises(TypeError, match=r"declares 2 parameters but got 1.*load\(resource\)"):
        _Weighted.scale(ones)

    loaded = _Weighted.load(DictResource({"w": weight}))
    with pytest.raises(TypeError, match=r"takes 1 activation") as excinfo:
        loaded.scale(ones, weight)
    assert "load(resource)" not in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        _Weighted.load(DictResource({}))
    refused = str(excinfo.value)
    assert "missing declared weight 'w'" in refused
    assert "prepare produces it" in refused
    assert (
        spec_ref_render(
            "[runtime §1.1.2](docs/spec/runtime.md#112-weight-converter-and-prepare--forward)"
        )
        in refused
    )


def test_one_shared_child_binds_once_per_owner():
    """Two owners over one child IR read their own subtrees rather than the last one loaded winning.

    Two owners over one child IR read their own subtrees rather than the last
    one loaded winning.
    """
    import copy  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from tilefoundry.dsl import ConstTensor  # noqa: PLC0415
    from tilefoundry.runtime.resource import DictResource  # noqa: PLC0415

    @module(entry="scale")
    class _Leaf:
        @func
        def scale(x: Tensor[(2,), "f32"], w: ConstTensor[(2,), "f32"]):
            return x * w

    @module(entry="twice")
    class _Owner:
        leaf = _Leaf

        @func
        def twice(x: Tensor[(2,), "f32"]):
            return x + x

    def aliasing(name):

        node = copy.copy(_Owner)
        object.__setattr__(node, "name", name)
        return node

    left, right = aliasing("left"), aliasing("right")
    assert left.modules[0] is right.modules[0]

    ones = torch.ones(2, dtype=torch.float32)
    loaded_left = left.load(DictResource({"leaf.w": torch.full((2,), 3.0)}))
    loaded_right = right.load(DictResource({"leaf.w": torch.full((2,), 10.0)}))

    assert loaded_left.leaf.module is loaded_right.leaf.module
    assert loaded_left.leaf.scale(ones).float().cpu().tolist() == [3.0, 3.0]
    assert loaded_right.leaf.scale(ones).float().cpu().tolist() == [10.0, 10.0]


def test_forward_reference_sibling_fails_loudly():
    """Test forward reference sibling fails loudly.

    A method that calls a sibling defined *below* it cannot resolve the
    sibling (only callee-before-caller is supported) and raises rather than
    silently mis-parsing the call.
    """
    with pytest.raises(VerifyError):

        @module(entry="caller")
        class _Forward:
            @func
            def caller(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                y = callee(x, g)  # noqa: F821 — defined below, unresolved on purpose
                return tf.add(y, y)

            @func
            def callee(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
                return tf.rms_norm(x, g)


def test_prim_func_host_resolves_sibling_device_in_class_body():
    """Test prim func host resolves sibling device in class body.

    A ``@prim_func`` cpu host can ``launch`` a sibling cuda device kernel
    defined above it in the same ``@module`` class body — class-local sibling
    resolution works for prim_func, not only @func.
    """

    @module(entry="host")
    class _Launch:
        @prim_func(target=CudaTarget("nvidia.h200_sxm"))
        def dev(a: Tensor[(8,), "f32"]):  # noqa: ARG001
            with Mesh((Topology("thread", 8),), Layout(shape=(8,), strides=(1,))) as m:
                T.sync(m)

        @prim_func(target=CpuTarget())
        def host(a: Tensor[(8,), "f32"]):
            launch(dev, a, grid=(1, 1, 1), block=(8, 1, 1))  # noqa: F821

    assert isinstance(_Launch, Module)
    assert [fn.name for fn in _Launch.functions] == ["dev", "host"]
    assert _Launch.entry == "host"
