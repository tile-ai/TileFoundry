"""Two authored launches spanning separate device topology domains."""

import torch
import tvm_ffi

from tilefoundry.dsl import *
from tilefoundry.runtime.module import RuntimeModule
from tilefoundry.target import CpuTarget, CudaTarget

S = DimVar("S", 1, 16)
CUDA = CudaTarget("nvidia.h200_sxm")


@module(entry="run", topologies=(Topology("thread", 64),))
class Child:
    @prim_func(target=CUDA)
    def run(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @run.specialize(RangePattern("S", 1, 8))
    def small(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @run.specialize(RangePattern("S", 9, 16))
    def large(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass


@module(entry="host", target=CUDA, topologies=(Topology("thread", 32),))
class Root:
    child = Child

    @prim_func(target=CUDA)
    def first(x: Tensor[(S, 8), "f32"], w_root: ConstTensor[(8,), "f32"]):
        pass

    @prim_func(target=CpuTarget())
    def host(x: Tensor[(S, 8), "f32"], w_root: ConstTensor[(8,), "f32"],
             w_child: ConstTensor[(8,), "f32"]):
        launch(first, x, w_root, grid=(1, 1, 1), block=(32, 1, 1))  # noqa: F821
        launch(child.run, x, w_child, grid=(1, 1, 1), block=(64, 1, 1))


class Runtime(RuntimeModule):
    def __init__(self, library_path):
        super().__init__("Root", entry="host")
        self.library = tvm_ffi.load_module(str(library_path))
        self.root_weight = torch.randn(8, device="cuda")
        self.child_weight = torch.randn(8, device="cuda")
        self.host_fn = tvm_ffi.get_global_func("Root.host")

    def forward(self, x):
        self.host_fn(x, self.root_weight, self.child_weight)
        return x
