"""Generic shard propagation engine over a matmul-shaped relation.

Domain dims: 0=M, 1=N, 2=K. lhs[M,K], rhs[K,N], out[M,N] (K reduced).

What is kept here is one case per rule the engine has to decide, plus every case
where the answer is "refuse": a Split the engine cannot carry into the output has
to fail rather than silently degrade to Broadcast, because a value quietly
declared replicated when each point holds a shard of it is read whole by everyone
downstream.
"""

from __future__ import annotations

from tests.core._source import import_dsl
from tilefoundry.ir.types.shard_layout import Split


def test_two_extent_one_splits_survive_a_loop_carry() -> None:
    module = import_dsl('''from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

@module(entry="accumulate", target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 2),))
class UnitAxesCarry:
    @func
    def accumulate(x: Tensor[(1, 1, 64), "f32"]):
        with Mesh(("cta",), layout=(1, 1, 2), names=("a", "b", "c")) as cta:
            total = tf.zeros(Tensor[(1 @ cta.a, 1 @ cta.b, 64 @ cta.c), "f32", "rmem"])
            for i in range(2):
                part = tf.reshard(x, (1 @ cta.a, 1 @ cta.b, 64 @ cta.c), "rmem")
                total = total + part
            return tf.reshard(total, (1 @ cta.a, 1 @ cta.b, 64 @ cta.c), "gmem")
''')
    assert module.entry_function().return_type.layout.attrs == (Split(0), Split(1), Split(2))
