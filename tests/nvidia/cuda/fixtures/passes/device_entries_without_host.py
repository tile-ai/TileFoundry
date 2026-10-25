"""Device entries with no host entry: a single kernel and a dispatch prototype."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")
_S = DimVar("S", 1, 255)


def _rows(threads: int) -> ShardLayout:
    """One element per thread of a *threads*-wide block."""
    return ShardLayout(
        layout=Layout(shape=(threads,), strides=(1,)),
        attrs=(S(0),),
        mesh=Mesh(topologies=(Topology("thread", threads),), layout=Layout((threads,), (1,))),
    )


@module(entry="copy_one", target=_CUDA, topologies=(Topology("thread", 128),))
class _OneKernel:
    """A single kernel, and no host entry to call it."""

    @prim_func(target=_CUDA)
    def copy_one(x: Tensor[(128,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(T.ptr_of(x), layout=_rows(128))
            T.copy(view, view)
            T.sync(thread)


@module(entry="square", target=_CUDA, topologies=(Topology("thread", 128),))
class _Prototype:
    """A prototype with no body of its own, and two variants that have one."""

    @prim_func(target=_CUDA)
    def square(x: Tensor[(_S,), "f32"]):
        pass

    @square.specialize(RangePattern("S", 1, 127))
    def small(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(T.ptr_of(x), layout=_rows(128), shape=(128,))
            T.copy(view, view)
            T.sync(thread)

    @square.specialize(RangePattern("S", 128, 255))
    def large(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(T.ptr_of(x), layout=_rows(128), shape=(128,))
            T.copy(view, view)
            T.sync(thread)
