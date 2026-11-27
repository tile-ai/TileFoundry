"""HIR dispatch plus a child whose weights enter only through orchestration."""

import torch
import tvm_ffi

from tilefoundry import runtime
from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

S = DimVar("S", 1, 16)


@module(entry="run")
class Router:
    @func
    def run(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        return tf.mul(x, w)


@module(entry="model", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("thread", 32),))
class Root:
    router = Router

    @func
    def model(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        pass

    @model.specialize(pattern.Range("S", 1, 8))
    def small(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        return tf.mul(x, w)

    @model.specialize(pattern.Range("S", 9, 16))
    def large(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        return tf.mul(x, w)

    def forward(self, x):
        return self.router.run(x)


class RouterRuntime(runtime.Module):
    def __init__(self, library):
        super().__init__("router", entry="run")
        self.library = library
        self.weight = torch.randn(8, device="cuda")
        self.run_fn = tvm_ffi.get_global_func("Root.router.run")

    def forward(self, x):
        out = torch.empty_like(x)
        self.run_fn(x, self.weight, out)
        return out


class Runtime(runtime.Module):
    def __init__(self, library_path):
        self.library = tvm_ffi.load_module(str(library_path))
        self.router = RouterRuntime(self.library)
        super().__init__("Root", entry="forward", children={"router": self.router})
        self.weight = torch.randn(8, device="cuda")
        self.run_fn = tvm_ffi.get_global_func("Root.model")

    def forward(self, x):
        out = torch.empty_like(x)
        self.run_fn(x, self.weight, out)
        return self.router(out)
