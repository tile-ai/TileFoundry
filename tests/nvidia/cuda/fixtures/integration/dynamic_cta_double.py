"""A dynamic-CTA kernel that doubles each row with no host of its own, so the entry is implicit."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.dsl import storage
from tilefoundry.ir import types
from tilefoundry_nvidia import cuda

_TILE = 12
_NT = DimVar("Ntile", 1, 63)


@module(
    entry="dyn_double",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", _NT),),
)
class DynamicCtaDouble:
    @func
    def dyn_double(a: Tensor[(_NT, _TILE), "f32"]) -> Tensor[(_NT, _TILE), "f32"]:
        with Mesh(("cta",), layout=types.Layout(shape=(_NT,), strides=(1,))) as cta:
            reg = tf.reshard(
                a,
                layout=ShardLayout(
                    layout=types.Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                    attrs=(S(0),),
                    mesh=cta,
                ),
                storage=storage.rmem,
            )
            out = tf.mul(reg, reg)
            return tf.reshard(
                out,
                layout=ShardLayout(
                    layout=types.Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                    attrs=(S(0),),
                    mesh=cta,
                ),
                storage=storage.gmem,
            )
