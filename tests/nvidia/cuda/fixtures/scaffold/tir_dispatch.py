"""A device prototype with three ranges and an authored CPU entry."""

import torch
import tvm_ffi

from tilefoundry import runtime
from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

S = DimVar("S", 1, 16)
CUDA = cuda.Target("nvidia.h200_sxm")


@module(entry="main", target=CUDA, topologies=(Topology("thread", 32),))
class Root:
    @prim_func(target=CUDA)
    def run(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @run.specialize(pattern.Range("S", 1, 4))
    def small(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @run.specialize(pattern.Range("S", 5, 10))
    def medium(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @run.specialize(pattern.Range("S", 11, 16))
    def large(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        pass

    @prim_func(target=cpu.Target())
    def main(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]):
        launch(run, x, w, grid=(1, 1, 1), block=(32, 1, 1))


class Runtime(runtime.Module):
    def __init__(self, library_path):
        super().__init__("Root", entry="main")
        self.library = tvm_ffi.load_module(str(library_path))
        self.weight = torch.randn(8, device="cuda")
        self.host_fn = tvm_ffi.get_global_func("Root.main")

    def forward(self, x):
        self.host_fn(x, self.weight)
        return x
