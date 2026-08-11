"""An unplaced two-output orchestration and its runtime reference."""

from tilefoundry import module
from tilefoundry.dsl import Tensor, Topology, func
from tilefoundry.runtime import runtime_func, runtime_module
from tilefoundry.target import CpuTarget


@module(entry="add_pair", target=CpuTarget(), topologies=(Topology("cta", 168),))
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


@runtime_module(Orchestrated)
class OrchestratedTwin:
    @runtime_func
    def add_pair(self, x, a, b):
        return x + a + b

    @runtime_func
    def affine_pair(self, x, scale, bias):
        return x * scale + bias

    def forward(self, x, pair):
        a, b, scale, bias = pair
        return self.add_pair(x, a, b), self.affine_pair(x, scale, bias)
