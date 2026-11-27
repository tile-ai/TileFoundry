"""Two child levels, equal explicit hierarchy and same-owner weight forwarding."""

import torch
import tvm_ffi

from tilefoundry import runtime
from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

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


@module(entry="model", target=cuda.Target("nvidia.h200_sxm"), topologies=HIERARCHY)
class Root:
    layers = Layers

    @func
    def run(out: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        return tf.mul(out, w)

    @func
    def model(out: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        pass

    @model.specialize(pattern.Range("S", 1, 8))
    def small(out: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        return run(out, w)

    @model.specialize(pattern.Range("S", 9, 16))
    def large(out: Tensor[(S, 8), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(S, 8), "f32"]:
        return run(out, w)

    def forward(self, x):
        return self.layers.leaf.run(x)


class Runtime(runtime.Module):
    def __init__(self, library_path):
        super().__init__("Root", entry="forward")
        self.library = tvm_ffi.load_module(str(library_path))
        self.weight = torch.randn(8, device="cuda")
        self.leaf_weight = torch.randn(8, device="cuda")
        self.run_fn = tvm_ffi.get_global_func("Root.model")
        self.leaf_fn = tvm_ffi.get_global_func("Root.layers.leaf.run")

    def forward(self, x):
        run_out, leaf_out = (torch.empty_like(x) for _ in range(2))
        self.run_fn(x, self.weight, run_out)
        self.leaf_fn(run_out, self.leaf_weight, leaf_out)
        return leaf_out
