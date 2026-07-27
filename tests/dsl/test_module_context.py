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

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 -- tf used by bodies
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import Topology
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.target import CpuTarget, CudaTarget

_CTA = Topology("cta", 132)
_WARP = Topology("warp", 4)
_THREAD = Topology("thread", 32)


@module(entry="forward", target=CudaTarget())
class _Root:
    topologies = (_CTA, _WARP)

    @func
    def forward(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    @func
    def other(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.square(x)

    @module(entry="step")
    class inherits:
        @func
        def step(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.relu(x)

    @module(entry="step")
    class topology_free:
        topologies = ()

        @func
        def step(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.relu(x)

    @module(entry="step")
    class replaces:
        topologies = (_THREAD,)

        @func
        def step(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
            return tf.relu(x)


def test_the_root_declaration_is_the_effective_target_everywhere_below() -> None:
    assert _Root.resolve_target() == CudaTarget()
    assert _Root.inherits.resolve_target() == CudaTarget()
    assert _Root.topology_free.resolve_target() == CudaTarget()
    assert _Root.replaces.resolve_target() == CudaTarget()


def test_an_undeclared_topology_tuple_inherits_the_owner_hierarchy() -> None:
    assert _Root.inherits.topologies is None
    assert _Root.inherits.effective_topologies() == (_CTA, _WARP)
    assert _Root.inherits.resolve_topology("warp") is _WARP


def test_an_explicit_empty_tuple_declares_a_topology_free_domain() -> None:
    assert _Root.topology_free.topologies == ()
    assert _Root.topology_free.effective_topologies() == ()
    with pytest.raises(ValueError, match="no topology named 'cta'"):
        _Root.topology_free.resolve_topology("cta")


def test_an_explicit_tuple_replaces_the_inherited_hierarchy_whole() -> None:
    assert _Root.replaces.effective_topologies() == (_THREAD,)
    assert _Root.replaces.resolve_topology("thread") is _THREAD
    with pytest.raises(ValueError, match="no topology named 'cta'"):
        _Root.replaces.resolve_topology("cta")


def test_an_undeclared_target_reports_the_module_path_it_searched() -> None:
    bare = Module("bare", (_Root.lookup("forward"),), "forward")
    with pytest.raises(ValueError, match="Module 'bare'.*no target is declared"):
        bare.resolve_target()


def test_an_unresolved_topology_name_lists_the_available_levels() -> None:
    with pytest.raises(ValueError, match="effective topology levels are cta, warp"):
        _Root.resolve_topology("block")


def test_a_repeated_topology_level_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="duplicate topology name"):
        Module(
            "dupe", (_Root.lookup("forward"),), "forward",
            topologies=(_CTA, Topology("cta", 8)),
        )


def test_a_child_may_not_override_the_inherited_target() -> None:
    child = Module(
        "child", (_Root.lookup("forward"),), "forward", target=CpuTarget(),
    )
    with pytest.raises(ValueError, match="child module 'child' declares its own target"):
        Module("root", (), "forward", modules=(child,), target=CudaTarget())


def test_a_module_owns_its_children_so_a_copy_cannot_retarget_the_original() -> None:
    """Placing the same child value under a second owner -- as a re-export
    that declares its own Target does -- must not change what the first
    owner's child resolves against."""
    child = Module("child", (_Root.lookup("forward"),), "forward")
    on_cuda = Module("root", (), "forward", modules=(child,), target=CudaTarget())
    on_cpu = replace(on_cuda, name="copy", target=CpuTarget())

    assert on_cuda.modules[0].resolve_target() == CudaTarget()
    assert on_cpu.modules[0].resolve_target() == CpuTarget()
    assert on_cuda.modules == on_cpu.modules


def test_every_owned_function_stays_independently_selectable() -> None:
    assert [fn.name for fn in _Root.functions] == ["forward", "other"]
    assert _Root.entry_function() is _Root.lookup("forward")
    assert _Root.lookup("other").name == "other"
    assert _Root.inherits.entry_function().name == "step"


def test_a_function_carries_no_execution_context_of_its_own() -> None:
    forward = _Root.lookup("forward")
    assert isinstance(forward, Function)
    assert not hasattr(forward, "target")
    assert not hasattr(forward, "topologies")


def test_declaring_context_on_a_function_yields_its_own_module() -> None:
    @func(target=CudaTarget(), topologies=(_CTA,))
    def standalone(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    assert isinstance(standalone, Module)
    assert standalone.name == "standalone"
    assert standalone.entry == "standalone"
    assert standalone.resolve_target() == CudaTarget()
    assert standalone.effective_topologies() == (_CTA,)


def test_a_plain_function_decorator_still_yields_a_function() -> None:
    @func
    def plain(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    assert isinstance(plain, Function)


def test_an_explicit_empty_topology_tuple_on_a_function_declares_its_module() -> None:
    @func(topologies=())
    def empty(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    assert isinstance(empty, Module)
    assert empty.topologies == ()


_SOURCE_PRELUDE = """
import tilefoundry
from tilefoundry.ir.types.shard.mesh import Topology
from tilefoundry.dsl import Tensor, tf

@tilefoundry.module(entry="k")
class M:
    topologies = {declaration}

    @tilefoundry.func
    def k(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)
"""


@pytest.mark.parametrize("declaration", ["1", '"cta"', "(1, 2)", "(Topology,)"])
def test_source_rejects_a_malformed_topology_declaration(declaration: str) -> None:
    """A declaration the parser cannot read is an error, not an empty domain:
    silently yielding a topology-free Module would strip the hierarchy every
    body below it names."""
    with pytest.raises(VerifyError, match="topologies"):
        parse_script(_SOURCE_PRELUDE.format(declaration=declaration))


def test_source_keeps_a_deferred_topology_extent() -> None:
    """``Topology(name, None)`` is the dynamic-launch extent, so it must parse
    as a declaration rather than be dropped as unreadable."""
    parsed = parse_script(_SOURCE_PRELUDE.format(declaration='(Topology("cta", None),)'))

    assert parsed.topologies == (Topology("cta", None),)


def test_source_rejects_a_prim_func_member() -> None:
    """Parsing a Module from source text reads HIR only. A TIR member is
    rejected rather than skipped, so the source never yields a Module that
    silently lost one of the functions it declares."""
    source = """
import tilefoundry
from tilefoundry.dsl import Tensor, T, tf

@tilefoundry.module(entry="k")
class M:
    @tilefoundry.func
    def k(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.relu(x)

    @tilefoundry.prim_func
    def lowered(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return x
"""
    with pytest.raises(VerifyError, match="prim_func"):
        parse_script(source)


def test_source_rejects_a_bare_prim_func_naming_the_hir_boundary() -> None:
    """A top-level ``@prim_func`` source is refused for the same reason as a
    ``@prim_func`` member, and says so, rather than reporting only that no
    ``@func`` was found."""
    source = """
import tilefoundry
from tilefoundry.dsl import Tensor, T

@tilefoundry.prim_func
def lowered(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
    return x
"""
    with pytest.raises(VerifyError, match="prim_func"):
        parse_script(source)
