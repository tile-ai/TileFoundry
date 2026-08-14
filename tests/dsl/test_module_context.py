"""Module-owned execution context: Target and the ordered Topology hierarchy.

A Module is the execution domain of the functions it owns. The outermost
Module declares the Target and every Module below it inherits that one
declaration. Topologies are declared per Module: omitting them inherits the
owner's hierarchy, while an explicit empty tuple declares a topology-free
domain. Resolution is lexical over the Module tree, so a selected Module
answers for its own effective context without any Function carrying it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests._source import import_dsl
from tests.fixtures.logical import module_context as context_fixture
from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf  # noqa: F401 -- used by bodies
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CpuTarget, CudaTarget

_CTA = context_fixture.CONTEXT_CTA
_WARP = context_fixture.CONTEXT_WARP
_THREAD = context_fixture.CONTEXT_THREAD
_Root = context_fixture.ContextTree


def test_the_root_declaration_is_the_only_target_anywhere_below() -> None:
    """Test the root declaration is the only target anywhere below.

    One Target is declared at the outermost Module and every Module below it
    resolves to that one; a child that declares its own is rejected rather than
    shadowing its owner, and a Module with none says which path it searched.
    """
    assert _Root.resolve_target() == CudaTarget("nvidia.h200_sxm")
    assert _Root.inherits.resolve_target() == CudaTarget("nvidia.h200_sxm")
    assert _Root.topology_free.resolve_target() == CudaTarget("nvidia.h200_sxm")
    assert _Root.replaces.resolve_target() == CudaTarget("nvidia.h200_sxm")

    forward = _Root.lookup("forward")
    child = Module("child", (forward,), "forward", target=CpuTarget())
    with pytest.raises(ValueError, match="child module 'child' declares its own target"):
        Module("root", (), "forward", modules=(child,), target=CudaTarget("nvidia.h200_sxm"))

    with pytest.raises(ValueError, match="Module 'bare'.*no target is declared"):
        Module("bare", (forward,), "forward").resolve_target()


def test_a_module_owns_its_children_so_a_copy_cannot_retarget_the_original() -> None:
    """Placing the same child value under a second owner.

    Placing the same child value under a second owner -- as a re-export
    that declares its own Target does -- must not change what the first
    owner's child resolves against.
    """
    child = Module("child", (_Root.lookup("forward"),), "forward")
    on_cuda = Module("root", (), "forward", modules=(child,), target=CudaTarget("nvidia.h200_sxm"))
    on_cpu = replace(on_cuda, name="copy", target=CpuTarget())

    assert on_cuda.modules[0].resolve_target() == CudaTarget("nvidia.h200_sxm")
    assert on_cpu.modules[0].resolve_target() == CpuTarget()
    assert on_cuda.modules == on_cpu.modules


def test_a_topology_tuple_is_undeclared_empty_or_a_whole_replacement() -> None:
    """The three declaration forms are three different domains.

    The three declaration forms are three different domains: omitting the
    tuple inherits the owner's hierarchy, an explicit empty tuple declares a
    topology-free domain (it does not fall back to the owner), and an explicit
    tuple replaces the inherited hierarchy whole rather than extending it.
    """
    assert _Root.inherits.topologies is None
    assert _Root.inherits.effective_topologies() == (_CTA, _WARP)
    assert _Root.inherits.resolve_topology("warp") is _WARP

    assert _Root.topology_free.topologies == ()
    assert _Root.topology_free.effective_topologies() == ()
    with pytest.raises(ValueError, match="no topology named 'cta'"):
        _Root.topology_free.resolve_topology("cta")

    assert _Root.replaces.effective_topologies() == (_THREAD,)
    assert _Root.replaces.resolve_topology("thread") is _THREAD
    with pytest.raises(ValueError, match="no topology named 'cta'"):
        _Root.replaces.resolve_topology("cta")


@pytest.mark.parametrize("invalid", [[_CTA], (_CTA, object())])
def test_module_rejects_a_non_topology_tuple(invalid) -> None:
    with pytest.raises(TypeError, match="topologies must be a tuple of Topology"):

        @module(topologies=invalid)
        class Invalid:
            pass


def test_a_raising_class_body_is_not_visible_to_a_later_standalone_func() -> None:
    try:

        @module(topologies=(Topology("leaked", 1),))
        class Failed:
            raise RuntimeError("class body failed")
    except RuntimeError:
        pass

    with pytest.raises(VerifyError, match="topology 'leaked' not declared"):

        @func
        def probe(x: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:
            with Mesh(("leaked",), (1,), ("lane",)) as _mesh:
                return tf.relu(x)


def test_a_failed_body_is_not_inherited_by_a_deeper_module() -> None:
    try:

        @module(topologies=(Topology("leaked", 1),))
        class Failed:
            raise RuntimeError("class body failed")
    except RuntimeError:
        pass

    def define_one_level_deeper():
        @module(entry="probe")
        class Later:
            @func
            def probe(x: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:
                with Mesh(("leaked",), (1,), ("lane",)) as _mesh:
                    return tf.relu(x)

        return Later

    with pytest.raises(VerifyError, match="topology 'leaked' not declared"):
        define_one_level_deeper()


def test_topology_resolution_failures_name_what_the_domain_holds() -> None:
    """An unresolved level lists the levels in scope.

    An unresolved level lists the levels in scope, and a repeated level is
    rejected at construction -- a duplicate name would make lexical resolution
    answer with whichever declaration happened to come first.
    """
    with pytest.raises(ValueError, match="effective topology levels are cta, warp"):
        _Root.resolve_topology("block")

    with pytest.raises(ValueError, match="duplicate topology name"):
        Module(
            "dupe",
            (_Root.lookup("forward"),),
            "forward",
            topologies=(_CTA, Topology("cta", 8)),
        )


def test_declaring_context_on_a_function_yields_its_own_module() -> None:
    """A Function never carries execution context of its own.

    A Function never carries execution context of its own: ``@func`` with a
    Target or a topology declaration (including an explicit empty one) yields the
    Module that owns it, and a plain ``@func`` stays a Function.
    """

    @func(target=CudaTarget("nvidia.h200_sxm"), topologies=(_CTA,))
    def standalone(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    @func(topologies=())
    def empty(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    @func
    def plain(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    assert isinstance(standalone, Module) and isinstance(empty, Module)
    assert (standalone.name, standalone.entry) == ("standalone", "standalone")
    assert standalone.resolve_target() == CudaTarget("nvidia.h200_sxm")
    assert standalone.effective_topologies() == (_CTA,)
    assert empty.topologies == ()
    assert isinstance(plain, Function)
    assert not hasattr(plain, "target") and not hasattr(plain, "topologies")


_MEMBER_CONTEXT_SOURCE = """
import tilefoundry
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

@tilefoundry.module(
    entry="attention",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2),),
)
class Layer:
    @tilefoundry.func
    def attention(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        with Mesh(("cta",), (2,), ("block",)) as _cta:
            return tf.relu(x)

    @tilefoundry.func(topologies=(Topology("thread", 4),))
    def softmax(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        with Mesh(("thread",), (4,), ("lane",)) as _thread:
            return tf.square(x)
"""


def test_member_context_builds_its_own_runtime_domain() -> None:
    imported = import_dsl(_MEMBER_CONTEXT_SOURCE, "Layer")

    assert [function.name for function in imported.functions] == ["attention"]
    assert [child.name for child in imported.modules] == ["softmax"]
    assert imported.modules[0].topologies == (Topology("thread", 4),)
    assert imported.modules[0].resolve_target() == CudaTarget("nvidia.h200_sxm")
