"""Two child levels, equal explicit hierarchy and same-owner weight forwarding."""

from dataclasses import replace

import torch

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


LeafRoot = replace(Leaf, target=Root.target)


class Runtime(runtime.Module):
    def __init__(self, library_path, child_library_path):
        super().__init__("Root", entry="forward")
        self.library = runtime.CompiledModule.from_library(library_path)
        self.weight = torch.randn(8, device="cuda")
        self.leaf_weight = torch.randn(8, device="cuda")
        self.run_fn = self.library.model
        self.library.load(runtime.DictResource({"w": self.weight}))
        self.leaf = runtime.CompiledModule.from_library(child_library_path)
        self.leaf.load(runtime.DictResource({"w": self.leaf_weight}))
        self.leaf_fn = self.leaf.run

    def forward(self, x):
        run_out, leaf_out = (torch.empty_like(x) for _ in range(2))
        self.run_fn(x, run_out)
        self.leaf_fn(run_out, leaf_out)
        return leaf_out
