"""Selections of a 32-thread mesh given new layouts: two that start at thread 0, one that does not.

The first half and the run from thread 16 hold part of the level, so each stays a slice of it;
the whole mesh in a new arrangement holds every thread and needs none.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="run", target=cpu.Target(), topologies=(Topology("thread", 32),))
class RelabelledSelections:
    @func
    def run(x: Tensor[(16,), "f32"]):
        with Mesh(("thread",), (32,), names=("t",)) as threads:
            with Mesh(threads[0:16], layout=(16,), names=("head",)) as _head:
                first = tf.add(x, x)
            with Mesh(threads[16:32], layout=(16,), names=("tail",)) as _tail:
                second = tf.add(first, x)
            with Mesh(threads[0:32], layout=(4, 8), names=("row", "column")) as _grid:
                return tf.add(second, x)
