"""Two child levels, equal explicit hierarchy and same-owner weight forwarding."""

import torch
import tvm_ffi

from tilefoundry.dsl import *
from tilefoundry.runtime.module import RuntimeModule
from tilefoundry.target import CudaTarget

S = DimVar("S", 1, 16)
HIERARCHY = (Topology("thread", 32),)


@module(entry="run", topologies=HIERARCHY)
class Leaf:
    @func
    def run(x: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        return tf.mul(x, w)


@module
class Layers:
    leaf = Leaf


@module(entry="model", target=CudaTarget("nvidia.h200_sxm"), topologies=HIERARCHY)
class Root:
    layers = Layers

    @func
    def run(out: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        return tf.mul(out, w)

    @func
    def model(out: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        pass

    @model.specialize(RangePattern("S", 1, 8))
    def small(out: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        return run(out, w)  # noqa: F821

    @model.specialize(RangePattern("S", 9, 16))
    def large(out: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        return run(out, w)  # noqa: F821

    def forward(self, x):
        return self.layers.leaf.run(x)


class Runtime(RuntimeModule):
    def __init__(self, library_path):
        super().__init__("Root", entry="forward")
        self.library = tvm_ffi.load_module(str(library_path))
        self.weight = torch.randn(8, device="cuda")
        self.leaf_weight = torch.randn(8, device="cuda")
        self.helper_fn = tvm_ffi.get_global_func("Root.run")
        self.run_fn = tvm_ffi.get_global_func("Root.model")
        self.leaf_fn = tvm_ffi.get_global_func("Root.layers.leaf.run")

    def forward(self, x):
        helper_out, run_out, leaf_out = (torch.empty_like(x) for _ in range(3))
        self.helper_fn(x, self.weight, helper_out)
        self.run_fn(helper_out, self.weight, run_out)
        self.leaf_fn(run_out, self.leaf_weight, leaf_out)
        return leaf_out
