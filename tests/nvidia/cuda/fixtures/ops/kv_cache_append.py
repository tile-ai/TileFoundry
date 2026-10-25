"""A KV-cache append split over two CTAs."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_CACHE_SHAPE = (2, 64, 4, 8)
_NEW_SHAPE = (2, 4, 4, 8)
_CTA = Topology("cta", 2)


@module(
    entry="append",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(_CTA,),
)
class _KVCacheAppend:
    @func
    def append(
        cache: Tensor[_CACHE_SHAPE, "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        s: Tensor[(1,), "i32"],
        new: Tensor[_NEW_SHAPE, "bf16"],
    ):
        with Mesh(("cta",), layout=(2,), names=("tile",)) as cta:
            local_cache = tf.reshard(cache, (2 @ cta.tile, 64, 4, 8), "gmem")
            local_new = tf.reshard(new, (2 @ cta.tile, 4, 4, 8), "gmem")
            return tf.cache_update(local_cache, cur_pos, s, local_new)
