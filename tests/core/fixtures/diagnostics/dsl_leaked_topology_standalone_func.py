"""A standalone function after a raising Module class body, using that body's topology.

The function stays standalone: whether a failed declaration leaks into a later
standalone definition is the question, so it is not wrapped in a Module.
"""

from __future__ import annotations

from tilefoundry.dsl import *

try:

    @module(topologies=(Topology("leaked", 1),))
    class Failed:
        raise RuntimeError("class body failed")
except RuntimeError:
    pass


@func
def probe(x: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:
    with Mesh(("leaked",), (1,), ("lane",)) as _mesh:
        return tf.relu(x)
