"""Two authored launches spanning separate device topology domains."""

import torch

from tilefoundry import runtime
from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

S = DimVar("S", 1, 16)
CUDA = cuda.Target("nvidia.h200_sxm")


@module(entry="run", topologies=(Topology("thread", 64),))
class Child:
    @prim_func(target=CUDA, output_count=0)
    def run(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @run.specialize(pattern.Range("S", 1, 8))
    def small(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @run.specialize(pattern.Range("S", 9, 16))
    def large(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass


@module(entry="host", target=CUDA, topologies=(Topology("thread", 32),))
class Root:
    child = Child

    @prim_func(target=CUDA, output_count=0)
    def first(x: Tensor[(S, 8), "f32"], w_root: ConstTensor[(8,), "f32"]):
        pass

    @prim_func(target=cpu.Target(), output_count=0)
    def host(x: Tensor[(S, 8), "f32"], w_root: ConstTensor[(8,), "f32"],
             w_child: ConstTensor[(8,), "f32"]):
        launch(first, x, w_root, grid=(1, 1, 1), block=(32, 1, 1))
        launch(child.run, x, w_child, grid=(1, 1, 1), block=(64, 1, 1))


class Runtime(runtime.Module):
    def __init__(self, library_path):
        super().__init__("Root", entry="host")
        self.library = runtime.CompiledModule.from_library(library_path)
        self.root_weight = torch.randn(8, device="cuda")
        self.child_weight = torch.randn(8, device="cuda")
        self.host_fn = self.library.host
        self.library.load(runtime.DictResource({"w_root": self.root_weight, "w_child": self.child_weight}))

    def forward(self, x):
        self.host_fn(x)
        return x
