"""Expert parallel with each GPU's own index: pick rows, then add them back.

Each GPU holds its own slots, so `tf.local` reads them as a Varying index. The
picked rows and the rows added back are then Varying too: every GPU computes
its own expert's part, and nothing moves between GPUs.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

P, E, C, K = 2, 8, 8, 4


@module(
    entry="forward",
    target=cuda.Target("nvidia.h200_sxm", device_count=P),
    topologies=(Topology("gpu", P), Topology("cta", 1)),
)
class ExpertLocalIndex:
    mesh = Mesh(("gpu",), (P,), names=("ep",))

    @func
    def forward(x: Tensor[(E, C), "f32", "gmem"], slot: Tensor[(P @ mesh.ep, K), "i32", "gmem"]):
        with mesh:
            idx = tf.reshape(tf.local(slot), (K,))
            rows = tf.zeros(Tensor[(E, C), "f32"])
            picked = tf.index_select(x, idx, dim=0)
            back = tf.index_add(rows, idx, picked, dim=0)
            return back * back
