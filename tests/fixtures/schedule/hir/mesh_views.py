"""Materialize captured slice, sharded bitcast and transpose in their consumer."""
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(entry="gemm", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1), Topology("thread", 32)))
class MeshViews:
    @func
    def gemm(x: Tensor[(4, 8), "f32"]) -> Tensor[(8, 4), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(4, 8), names=("row", "col")) as producer:
                held = tf.schedule(
                    (x, tf.empty(Tensor[(4, 8), "f32", ((4 @ producer.row, 8 @ producer.col), (8, 1)), "rmem"])),
                    op=T.copy(),
                )
                sliced = held[:, :]
                recast = tf.bitcast(sliced, ((4 @ producer.row, 8 @ producer.col), (8, 1)))
                transposed = tf.transpose(recast, perm=(1, 0))
            with Mesh(("thread",), layout=(4, 8), names=("row", "col")) as _reader:
                mixed = sliced + recast
                moved = tf.transpose(mixed, perm=(1, 0))
            with Mesh(("thread",), layout=(8, 4), names=("row", "col")) as _consumer:
                output = tf.zeros(Tensor[(8, 4), "f32"])
                result = transposed + moved
                return tf.insert_slice(output, result, (0, 0))
