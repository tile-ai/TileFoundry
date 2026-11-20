"""One Reshard per way the units holding a value can change, on four GPUs."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=4)
gpu = Topology("gpu", 4)
SHAPE = (4, 8)


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class SameUnits:
    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[(4, 8), "f32", (4, 8), "gmem"]):
        return tf.reshard(x, (4, 8), "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class SplitToBroadcast:
    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[(4, 8), "f32", (4 @ mesh.g, 8), "gmem"]):
        return tf.reshard(x, (4, 8), "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class OneToAll:
    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[(4, 8), "f32", (4, 8) @ mesh[0:1], "gmem"]):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as _m:
            return tf.reshard(x, (4, 8), "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class AllToOne:
    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[(4, 8), "f32", (4, 8), "gmem"]):
        return tf.reshard(x, (4, 8) @ mesh[0:1], "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class ShiftedPair:
    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[(4, 8), "f32", (4, 8) @ mesh[0:2], "gmem"]):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as units:
            return tf.reshard(x, (4, 8) @ units[1:3], "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class SplitToDisjointPair:
    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[(4, 8), "f32", (4 @ mesh[0:2].g, 8), "gmem"]):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as units:
            return tf.reshard(x, (4 @ units[2:4].g, 8), "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class StageColumns:
    """Data-parallel GPUs {0, 2} hand a stage to {1, 3}: units that are not contiguous."""

    @func(mesh=Mesh(("gpu",), (2, 2), names=("dp", "pp")))
    def run(x: Tensor[(4, 8), "f32", (4, 8) @ mesh[:, 0], "gmem"]):
        with Mesh(("gpu",), layout=(2, 2), names=("dp", "pp")) as grid:
            return tf.reshard(x, (4, 8) @ grid[:, 1], "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class ScaledHandoff:
    """GPU 0 scales by a plain parameter, so the product varies there and keeps GPU 0's mesh."""

    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[SHAPE, "f32", (4, 8) @ mesh[0:1], "gmem"], s: Tensor[SHAPE, "f32", "gmem"]):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as units:
            with Mesh(units[0:1], layout=(1,), names=("s",)) as _first:
                scaled = x * s
            moved = tf.reshard(scaled, (4, 8) @ units[1:2], "gmem")
            return moved
