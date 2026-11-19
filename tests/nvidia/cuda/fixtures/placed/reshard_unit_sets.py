"""One Reshard per way the units holding a value can change, on four GPUs."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=4)
gpu = Topology("gpu", 4)
mesh = Mesh((gpu,), Layout((4,)), ("g",))
SHAPE = (4, 8)


def on(units, attr=Broadcast()):
    return TensorType(SHAPE, DType.f32, ShardLayout.from_logical(
        SHAPE, units, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


ALL, FIRST, MIDDLE = on(mesh), on(mesh[0:1]), on(mesh[1:3])
LAST_SPLIT = on(mesh[2:4], Split(0))
grid = Mesh((gpu,), Layout((2, 2)), ("dp", "pp"))
SECOND_STAGE = on(grid[:, 1])


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class SameUnits:
    @func
    def run(x: on(mesh)):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as _m:
            return tf.reshard(x, ALL.layout, "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class SplitToBroadcast:
    @func
    def run(x: on(mesh, Split(0))):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as _m:
            return tf.reshard(x, ALL.layout, "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class OneToAll:
    @func
    def run(x: on(mesh[0:1])):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as _m:
            return tf.reshard(x, ALL.layout, "gmem")


@module(entry="run", target=target, topologies=(gpu, Topology("cta", 1)))
class AllToOne:
    @func
    def run(x: on(mesh)):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as _m:
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
