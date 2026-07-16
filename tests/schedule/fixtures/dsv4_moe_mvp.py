from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf


@func
def route_func(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    return tf.add(x, 1.0)


@func
def routed_func(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    return tf.mul(x, 2.0)


@func
def shared_func(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    return tf.add(x, 3.0)


@func
def combine_func(
    routed: Tensor[(8,), "f32"], shared: Tensor[(8,), "f32"]
) -> Tensor[(8,), "f32"]:
    return tf.add(routed, shared)


def moe_entry(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    routed_value: where(storage="rmem") = route_func(x)
    routed_output = routed_func(routed_value)
    shared_output = shared_func(x)
    return combine_func(routed_output, shared_output)


__all__ = [
    "combine_func",
    "moe_entry",
    "route_func",
    "routed_func",
    "shared_func",
]
