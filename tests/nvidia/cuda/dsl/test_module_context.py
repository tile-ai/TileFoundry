"""Module-owned execution context: Target and the ordered Topology hierarchy.

A Module is the execution domain of the functions it owns. The outermost
Module declares its Target; a child declaration defines its own domain, while
an undeclared child inherits during a visitor walk. Topologies are declared per
Module: omitting them inherits the owner's hierarchy, while an explicit empty
tuple declares a topology-free
domain. A visitor frame supplies inherited context without any
Function carrying it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import load_program
from tests.nvidia.cuda.fixtures.logical import module_context as context_fixture
from tilefoundry.dsl import *
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.module_utils import list_call_paths
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CTA = context_fixture.CONTEXT_CTA
_WARP = context_fixture.CONTEXT_WARP
_THREAD = context_fixture.CONTEXT_THREAD
_Root = context_fixture.ContextTree


def test_a_target_declaration_defines_its_own_execution_domain() -> None:
    """A child's target can differ; an undeclared standalone root has no target."""
    target = _Root.resolve_target()
    assert target == CudaTarget("nvidia.h200_sxm")
    assert all(frame.target is target for frame in list_call_paths(_Root).values())
    forward = _Root.lookup("forward")
    child = Module("child", (forward,), "forward", target=CpuTarget())
    root = Module("root", (forward,), "forward", children={"child": ModuleList(child)}, target=target)
    assert root.resolve_target() is target
    assert child.resolve_target() is child.target

    with pytest.raises(ValueError, match="Module 'bare'.*no target is declared"):
        Module("bare", (forward,), "forward").resolve_target()


def test_a_topology_tuple_is_undeclared_empty_or_a_whole_replacement() -> None:
    """The three declaration forms are three different domains.

    The three declaration forms are three different domains: omitting the
    tuple inherits the owner's hierarchy, an explicit empty tuple declares a
    topology-free domain (it does not fall back to the owner), and an explicit
    tuple replaces the inherited hierarchy whole rather than extending it.
    """
    assert _Root.inherits.topologies is None
    inherits = _Root.inherits
    frame = next(frame for frame in list_call_paths(_Root).values() if frame.module is inherits)
    assert frame.topologies == (_CTA, _WARP)
    assert frame.topologies[1] is _WARP

    assert _Root.topology_free.topologies == ()
    topology_free = _Root.topology_free
    assert topology_free.effective_topologies() == ()
    with pytest.raises(ValueError, match="no topology named 'cta'"):
        topology_free.resolve_topology("cta")

    replaces = _Root.replaces
    assert replaces.effective_topologies() == (_THREAD,)
    assert replaces.resolve_topology("thread") is _THREAD
    with pytest.raises(ValueError, match="no topology named 'cta'"):
        replaces.resolve_topology("cta")


@pytest.mark.parametrize("invalid", [[_CTA], (_CTA, object())])
def test_module_rejects_a_non_topology_tuple(invalid) -> None:
    with pytest.raises(TypeError, match="topologies must be a tuple of Topology"):

        @module(topologies=invalid)
        class Invalid:
            pass


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


def test_member_context_builds_its_own_runtime_domain() -> None:
    imported = load_program(Path(__file__).parents[1] / "fixtures/logical/member_context.py")

    assert [function.name for function in imported.functions] == ["attention"]
    assert list(imported.children) == ["softmax"]
    assert imported.children["softmax"].elements[0].topologies == (Topology("thread", 4),)
    assert list_call_paths(imported)["softmax.softmax"].target == CudaTarget("nvidia.h200_sxm")
