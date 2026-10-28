"""A device prototype with three ranges and an authored CPU entry."""

import torch
import tvm_ffi

from tilefoundry.dsl import *
from tilefoundry.runtime.module import RuntimeModule
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

S = DimVar("S", 1, 16)
CUDA = CudaTarget("nvidia.h200_sxm")


@module(entry="main", target=CUDA, topologies=(Topology("thread", 32),))
class Root:
    @prim_func(target=CUDA)
    def run(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @run.specialize(RangePattern("S", 1, 4))
    def small(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @run.specialize(RangePattern("S", 5, 10))
    def medium(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @run.specialize(RangePattern("S", 11, 16))
    def large(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @prim_func(target=CpuTarget())
    def main(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        launch(run, x, w, grid=(1, 1, 1), block=(32, 1, 1))


class Runtime(RuntimeModule):
    def __init__(self, library_path):
        super().__init__("Root", entry="main")
        self.library = tvm_ffi.load_module(str(library_path))
        self.weight = torch.randn(8, device="cuda")
        self.host_fn = tvm_ffi.get_global_func("Root.main")

    def forward(self, x):
        self.host_fn(x, self.weight)
        return x
