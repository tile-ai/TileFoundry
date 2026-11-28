"""An affine task over a feature map larger than L2, staged sixteen rows at a time: x + 1."""
from tilefoundry.dsl import *
from tilefoundry_apple import ane


@module(entry="unary_tiled", target=ane.Target("apple.m2_pro.ane"), topologies=(Topology("engine", 1),))
class Program:
    @func
    def unary_tiled(x: Tensor[(1, 64, 128, 128), "f16", ((1, 64, 128, 128), (1048576, 16384, 128, 1)), "gmem"]):
        with Mesh(("engine",), layout=(1,), names=("engine",)) as _engine:
            return tf.schedule((tf.empty(Tensor[(1, 64, 128, 128), "f16", ((1, 64, 128, 128), (1048576, 16384, 128, 1)), "gmem"]), x), op=T.ane.unary(bias=1.0, tile_height=16))
