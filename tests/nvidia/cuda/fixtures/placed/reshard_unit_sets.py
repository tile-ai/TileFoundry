"""One Reshard per way the units holding a value can change, on four GPUs."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=4)
gpu = Topology("gpu", 4)
mesh = Mesh((gpu,), Layout((4,)), ("g",))
SHAPE = (4, 8)


def on(units, attr=Broadcast()):
    return TensorType(SHAPE, DType.f32, ShardLayout.from_logical(
        SHAPE, units, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


FIRST, SECOND, MIDDLE = on(mesh[0:1]), on(mesh[1:2]), on(mesh[1:3])
LAST_SPLIT = on(mesh[2:4], Split(0))
grid = Mesh((gpu,), Layout((2, 2)), ("dp", "pp"))
SECOND_STAGE = on(grid[:, 1])


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class SameUnits:
    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[(4, 8), "f32", (4, 8) @ {B()}, "gmem"]):
        return tf.reshard(x, (4, 8) @ {B()}, "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class SplitToBroadcast:
    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[(4, 8), "f32", (4 @ mesh.g, 8), "gmem"]):
        return tf.reshard(x, (4, 8) @ {B()}, "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class OneToAll:
    @func
    def run(x: on(mesh[0:1])):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as _m:
            return tf.reshard(x, (4, 8) @ {B()}, "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class AllToOne:
    @func(mesh=Mesh(("gpu",), (4,), names=("g",)))
    def run(x: Tensor[(4, 8), "f32", (4, 8) @ {B()}, "gmem"]):
        return tf.reshard(x, FIRST.layout, "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class ShiftedPair:
    @func
    def run(x: on(mesh[0:2])):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as _m:
            return tf.reshard(x, MIDDLE.layout, "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class SplitToDisjointPair:
    @func
    def run(x: on(mesh[0:2], Split(0))):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as _m:
            return tf.reshard(x, LAST_SPLIT.layout, "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class StageColumns:
    """Data-parallel GPUs {0, 2} hand a stage to {1, 3}: units that are not contiguous."""

    @func
    def run(x: on(grid[:, 0])):
        with Mesh(("gpu",), layout=(2, 2), names=("dp", "pp")) as _m:
            return tf.reshard(x, SECOND_STAGE.layout, "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class ScaledHandoff:
    """GPU 0 scales by a plain parameter, so the product varies there and keeps GPU 0's mesh."""

    @func
    def run(x: on(mesh[0:1]), s: Tensor[SHAPE, "f32", "gmem"]):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as units:
            with Mesh(units[0:1], layout=(1,), names=("s",)) as _first:
                scaled = x * s
            moved = tf.reshard(scaled, SECOND.layout, "gmem")
            return moved
