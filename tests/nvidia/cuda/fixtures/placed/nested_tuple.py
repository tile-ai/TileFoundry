"""Tuples inside tuples, read one level at a time, beside the same programs with flat tuples.

NestedParam takes its tensors as Tuple[Tensor, Tuple[Tensor]] and picks the
inner tuple before its tensor. NestedReturn calls a function that returns
(out, (saved,)), the shape differentiate gives a forward with calls. NestedDeep
picks a tuple whose second field is a tuple again, so the picked value has more
leaves than fields. The Flat twins hold the same tensors in one level.
"""

from typing import Tuple

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm")


@module(entry="forward", target=target, topologies=(Topology("cta", 1),))
class NestedParam:
    @func
    def forward(saved: Tuple[Tensor[(4, 8), "f32"], Tuple[Tensor[(4, 8), "f32"]]]):
        inner = saved[1]
        return saved[0] * inner[0]


@module(entry="forward", target=target, topologies=(Topology("cta", 1),))
class FlatParam:
    @func
    def forward(saved: Tuple[Tensor[(4, 8), "f32"], Tensor[(4, 8), "f32"]]):
        return saved[0] * saved[1]


@module(entry="forward", target=target, topologies=(Topology("cta", 1),))
class NestedReturn:
    @func
    def step(x: Tensor[(4, 8), "f32"]):
        out = x * x
        return out, (x,)

    @func
    def forward(x: Tensor[(4, 8), "f32"]):
        held = step(x)
        inner = held[1]
        return held[0] + inner[0]


@module(entry="forward", target=target, topologies=(Topology("cta", 1),))
class FlatReturn:
    @func
    def step(x: Tensor[(4, 8), "f32"]):
        out = x * x
        return out, x

    @func
    def forward(x: Tensor[(4, 8), "f32"]):
        held = step(x)
        return held[0] + held[1]


@module(entry="forward", target=target, topologies=(Topology("cta", 1),))
class NestedDeep:
    @func
    def forward(saved: Tuple[Tensor[(4, 8), "f32"], Tuple[Tensor[(4, 8), "f32"], Tuple[Tensor[(4, 8), "f32"], Tensor[(4, 8), "f32"]]]]):
        inner = saved[1]
        deeper = inner[1]
        return saved[0] * inner[0] + deeper[0] * deeper[1]


@module(entry="forward", target=target, topologies=(Topology("cta", 1),))
class FlatDeep:
    @func
    def forward(saved: Tuple[Tensor[(4, 8), "f32"], Tensor[(4, 8), "f32"], Tensor[(4, 8), "f32"], Tensor[(4, 8), "f32"]]):
        return saved[0] * saved[1] + saved[2] * saved[3]
