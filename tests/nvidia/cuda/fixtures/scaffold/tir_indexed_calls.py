"""Named TIR calls through indexed child slots, by constant and run-time index."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

CUDA = cuda.Target("nvidia.h200_sxm")


@module(entry="run", topologies=(Topology("thread", 32),))
class Leaf:
    @prim_func(target=CUDA)
    def run(x: Tensor[(8,), "f32"]):
        pass


@module(entry="run", topologies=(Topology("thread", 32),))
class Other:
    @prim_func(target=CUDA)
    def run(x: Tensor[(8,), "f32"]):
        pass


@module(entry="run", topologies=(Topology("thread", 32),))
class Stack:
    layers = ModuleList(Leaf, 3)
    mixed = ModuleList((Leaf, Other))

    @prim_func(target=CUDA)
    def run(x: Tensor[(8,), "f32"]):
        pass


@module(entry="main", target=CUDA, topologies=(Topology("thread", 32),))
class Root:
    a = Stack

    @prim_func(target=CUDA)
    def main(x: Tensor[(8,), "f32"], i: Tensor[(), "i32"]):
        a.layers[2].run(x)
        a.layers[i].run(x)
        a.mixed[1].run(x)
