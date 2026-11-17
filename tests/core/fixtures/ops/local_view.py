"""Unit-local QKV slicing and reshape across broadcast and split layouts."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


def program(units):
    target = cuda.Target("nvidia.h200_sxm", device_count=units)
    gpu = Topology("gpu", units)
    mesh = Mesh((gpu,), Layout((units,)), ("g",))
    attr = Broadcast() if units == 1 else Split(0)
    layout = ShardLayout.from_logical(
        (256, 8, 4, 128), mesh, (attr,), storage="gmem", owners=target.storage_owners
    )

    @module(entry="f", target=target, topologies=(gpu,))
    class LocalView:
        @func
        def f(x: Tensor[(256, 8, 4, 128), "f32", layout]):
            q = x[:, :, 0:2, :]
            q = tf.reshape(q, new_shape=(256, 16, 128))
            return tf.reshape(q, new_shape=(1, 256, 16, 128))

    return LocalView


BroadcastView = program(1)
SplitView = program(2)
