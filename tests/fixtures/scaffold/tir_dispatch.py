"""A single device prototype with three ranges and a synthesized CPU entry."""

import torch
import tvm_ffi

from tilefoundry.dsl import *
from tilefoundry.runtime.module import RuntimeModule
from tilefoundry_cuda import CudaTarget

S = DimVar("S", 1, 16)
CUDA = CudaTarget("nvidia.h200_sxm")


@module(entry="run", target=CUDA, topologies=(Topology("thread", 32),))
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


class Runtime(RuntimeModule):
    def __init__(self, library_path):
        super().__init__("Root", entry="main")
        self.library = tvm_ffi.load_module(str(library_path))
        self.weight = torch.randn(8, device="cuda")
        self.host_fn = tvm_ffi.get_global_func("Root.main")

    def forward(self, x):
        self.host_fn(x, self.weight)
        return x
