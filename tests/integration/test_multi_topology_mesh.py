"""A Mesh spanning several topologies addresses the whole topology product.

``Mesh`` models an ordered topology sequence ([shard §5](docs/spec/shard.md#5-mesh)):
``topology`` is the primary level and ``topologies`` the full tuple. A mesh over
``cta`` x ``thread`` therefore has a domain of ``n_cta * n_thread``, and its
layout subdivides that product.

The emitted C++ used to carry only the primary topology, so the mesh coordinate
came from ``program_id<cta>()`` alone and every thread of a CTA resolved to the
same coordinate: each CTA wrote one thread's worth of elements and left the rest
untouched. The kernel still compiled and ran, so only a full-output assertion
catches it — hence the exact all-elements comparison below rather than a spot
check.
"""

from __future__ import annotations

import torch

import tilefoundry
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.dsl.storage import gmem, rmem
from tilefoundry.ir.types.shard import Layout, Mesh, Topology

_CTAS = 16
_THREADS = 256
_PER_THREAD = 8
# One row per CTA; every thread of the CTA owns a contiguous run of the row.
_COLS = _THREADS * _PER_THREAD


@func(topologies=(Topology("cta", _CTAS), Topology("thread", _THREADS)))
def square_grid_of_threads(
    a: Tensor[(_CTAS, _COLS), "f32"],
) -> Tensor[(_CTAS, _COLS), "f32"]:
    # One mesh over both levels: the layout's leading axis spans the ctas and
    # the trailing one the threads, so the mesh domain is 16 * 256 = 4096.
    with Mesh(
        topology=[Topology("cta", _CTAS), Topology("thread", _THREADS)],
        layout=Layout(shape=(_CTAS, _THREADS), strides=(_THREADS, 1)),
        names=("c", "t"),
    ) as m:
        reg = tf.reshard(a, layout=(_CTAS @ m.c, _COLS @ m.t), storage=rmem)
        out = tf.mul(reg, reg)
        return tf.reshard(out, layout=(_CTAS @ m.c, _COLS @ m.t), storage=gmem)


def test_a_cta_by_thread_mesh_writes_every_element() -> None:
    rm = tilefoundry.compile(square_grid_of_threads, target="cuda")
    torch.manual_seed(0)
    # Non-uniform data so a coordinate collision cannot pass by coincidence,
    # and no zeros so an untouched output element cannot look correct.
    x = torch.randn(_CTAS, _COLS, dtype=torch.float32, device="cuda") + 2.0
    out = torch.full_like(x, float("nan"))
    rm(x, out)
    torch.cuda.synchronize()

    expected = x * x
    assert torch.allclose(out, expected, rtol=0, atol=0), (
        f"{int((out != expected).sum())} of {out.numel()} elements wrong "
        f"(every thread of a CTA resolving to one mesh coordinate leaves "
        f"most of each row untouched)"
    )
