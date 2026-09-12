"""What a module with no host entry gets, and what it keeps.

Both device shapes take one route: the callee's own shape decides whether the
call reads as a launch or as a dispatch, and no function's target is rewritten
on the way.
"""

from __future__ import annotations

import pytest

from tests.fixtures.tir.square import TirSquare
from tilefoundry import module, prim_func
from tilefoundry.dsl import DimVar, T, Tensor
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.stmts import Evaluate
from tilefoundry.ir.types.shard import Layout, Mesh, S, ShardLayout, Topology
from tilefoundry.passes.transforms import insert_default_host_entry
from tilefoundry.target import CpuTarget, CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")
_S = DimVar("S", 1, 256)


def _rows(threads: int) -> ShardLayout:
    """One element per thread of a *threads*-wide block."""
    return ShardLayout(
        layout=Layout(shape=(threads,), strides=(1,)),
        attrs=(S(0),),
        mesh=Mesh(topologies=(Topology("thread", threads),), layout=Layout((threads,), (1,))),
    )


@module(entry="copy_one", topologies=(Topology("thread", 128),))
class _OneKernel:
    """A single kernel, and no host entry to call it."""

    @prim_func(target=_CUDA)
    def copy_one(x: Tensor[(128,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(x, layout=_rows(128))
            T.copy(view, view)
            T.sync(thread)


@module(entry="square", topologies=(Topology("thread", 128),))
class _Prototype:
    """A prototype with no body of its own, and two variants that have one."""

    @prim_func(target=_CUDA)
    def square(x: Tensor[(_S,), "f32"]):
        pass

    @square.specialize(DimVarRangePat("S", 1, 127))
    def small(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(x, layout=_rows(128))
            T.copy(view, view)
            T.sync(thread)

    @square.specialize(DimVarRangePat("S", 128, 255))
    def large(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(x, layout=_rows(128))
            T.copy(view, view)
            T.sync(thread)


@module(entry="square", topologies=(Topology("thread", 128),))
class _DisagreeingVariants:
    """The same shape, except the two variants mesh over different thread counts."""

    @prim_func(target=_CUDA)
    def square(x: Tensor[(_S,), "f32"]):
        pass

    @square.specialize(DimVarRangePat("S", 1, 127))
    def small(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 64),), Layout((64,), (1,))) as thread:
            view = T.tensor_view(x, layout=_rows(64))
            T.copy(view, view)
            T.sync(thread)

    @square.specialize(DimVarRangePat("S", 128, 255))
    def large(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(x, layout=_rows(128))
            T.copy(view, view)
            T.sync(thread)


def _launch_of(mod):
    """The one ``Launch`` the synthesized entry's body is made of."""
    (statement,) = mod.entry_function().body.body
    assert isinstance(statement, Evaluate)
    assert isinstance(statement.callable, Launch)
    return statement


def _geometry(evaluate) -> tuple[tuple[int, ...], tuple[int, ...]]:
    extents = tuple(int(arg.value) for arg in evaluate.args[1:7])
    return extents[:3], extents[3:]


@pytest.mark.parametrize("authored", [_OneKernel, _Prototype])
def test_a_device_entry_gains_a_host_entry_that_launches_it(authored) -> None:
    lowered = insert_default_host_entry(authored)
    entry = lowered.entry_function()
    assert lowered.entry == "main"
    assert isinstance(entry.target, CpuTarget)
    assert [p.name for p in entry.params] == ["x"]
    assert _launch_of(lowered).args[0].name == authored.entry


@pytest.mark.parametrize("authored", [_OneKernel, _Prototype])
def test_the_pass_leaves_every_recorded_target_alone(authored) -> None:
    lowered = insert_default_host_entry(authored)
    assert isinstance(lowered.lookup(authored.entry).target, CudaTarget)


def test_a_prototype_reads_its_geometry_off_the_variants_that_have_a_body() -> None:
    """The prototype's own body is empty, so a launch of it would run one thread."""
    assert _geometry(_launch_of(insert_default_host_entry(_Prototype))) == ((1, 1, 1), (128, 1, 1))


def test_a_lone_kernel_reads_its_geometry_off_its_own_body() -> None:
    assert _geometry(_launch_of(insert_default_host_entry(_OneKernel))) == ((1, 1, 1), (128, 1, 1))


def test_variants_that_want_different_geometries_are_refused() -> None:
    with pytest.raises(ValueError, match="different geometries"):
        insert_default_host_entry(_DisagreeingVariants)


def test_a_module_that_already_has_a_host_entry_is_left_alone() -> None:
    assert insert_default_host_entry(TirSquare) is TirSquare
