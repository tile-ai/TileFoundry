"""A callee two calls down reads one element of a tuple it builds from a reshaped argument.

``top -> mid -> inner``: ``mid`` passes ``inner`` a reshape of its parameter, and
``inner`` returns ``t[0] * 2.0`` for ``t = (x, x)``. Its accesses land on the root of
``flat`` and must keep the reshape's view when they are carried up a second call.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="top", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("cta", 132),))
class TupleElementTwoCalls:
    @func
    def inner(x: Tensor[(4, 3), "f32"]):
        t = (x, x)
        return t[0] * 2.0

    @func
    def mid(flat: Tensor[(12,), "f32"]):
        return inner(tf.reshape(flat, new_shape=(4, 3)))

    @func
    def top(flat: Tensor[(12,), "f32"]):
        return mid(flat)
