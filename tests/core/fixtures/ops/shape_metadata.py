"""Shape metadata read from a runtime-shaped tensor: the whole shape, its rank, one element."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget

_S = DimVar("runtime_shape", 1, 8)


@module(entry="shape_metadata", target=CpuTarget())
class ShapeMetadata:
    @func
    def shape_metadata(x: Tensor[(_S, 4), "f32"]):
        return (tf.shape_of(x), tf.rank(x))


@module(entry="shape_element_paths", target=CpuTarget())
class ShapeElementPaths:
    @func
    def shape_element_paths(x: Tensor[(_S, 4), "f32"]):
        return tf.shape_of(x)[1]
