"""An unplaced two-output orchestration and its runtime reference."""

from tilefoundry import runtime
from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="add_pair", target=cpu.Target(), topologies=(Topology("cta", 168),))
class Orchestrated:
    @func
    def add_pair(
        x: Tensor[(168,), "f32"], a: Tensor[(168,), "f32"], b: Tensor[(168,), "f32"]
    ) -> Tensor[(168,), "f32"]:
        return x + a + b

    @func
    def affine_pair(
        x: Tensor[(168,), "f32"], scale: Tensor[(168,), "f32"], bias: Tensor[(168,), "f32"]
    ) -> Tensor[(168,), "f32"]:
        return x * scale + bias

    def forward(self, x, pair):
        a, b, scale, bias = pair
        return self.add_pair(x, a, b), self.affine_pair(x, scale, bias)


@runtime.module(Orchestrated)
class OrchestratedTwin:
    @runtime.func
    def add_pair(self, x, a, b):
        return x + a + b

    @runtime.func
    def affine_pair(self, x, scale, bias):
        return x * scale + bias

    def forward(self, x, pair):
        a, b, scale, bias = pair
        return self.add_pair(x, a, b), self.affine_pair(x, scale, bias)
