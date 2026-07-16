"""Builds the demo hir.Function via ``@func`` DSL (parser path).

⚠️ **Legacy fixture** — Uses closure-captured ``shared_layout`` /
``reg_layout`` / ``cta_mesh`` / ``thread_mesh`` at module level. This
pattern is NOT canonical. Prefer ``tests/models/demo/demo_canonical.py``
which uses ``@func(topologies=...)`` + ``with Mesh(topology="cta", ...)``.

Also provides the legacy ``build_demo()`` API returning ``(Function, cta_mesh,
thread_mesh)`` for backwards compatibility with existing tests.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403  -- binds bare op names (reshard, relu, ...)
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import (
    Layout,
    Mesh,
    S,
    ShardLayout,
    Topology,
)

# ── Mesh / Layout definitions ────────────────────────────────────────

cta_topo = Topology("cta", 128)
thread_topo = Topology("thread", 8 * 32)

cta_mesh = Mesh(topology=cta_topo, layout=Layout(shape=(128,), strides=(1,)))
thread_mesh = Mesh(topology=thread_topo, layout=Layout(shape=(8, 32), strides=(32, 1)))

LOGICAL_SHAPE = (1, 1536)

shared_layout = ShardLayout(
    layout=Layout(shape=LOGICAL_SHAPE, strides=(1536, 1)),
    attrs=(),
    mesh=cta_mesh,
)

reg_layout = ShardLayout(
    layout=Layout(shape=(1, 8, 192), strides=(1536, 192, 1)),
    attrs=(S(1), S(2)),
    mesh=thread_mesh,
)


# ── @func DSL definition ─────────────────────────────────────────────


@func
def demo(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:
    b = reshard(a, layout=shared_layout, storage="smem")
    c = reshard(b, layout=reg_layout, storage="rmem")
    d = relu(c)
    e = reshard(d, layout=reg_layout, storage="gmem")
    return e


# ── Legacy API for test compatibility ────────────────────────────────


def build_demo():
    """Return ``(hir.Function, cta_mesh, thread_mesh)`` for existing tests."""
    ir = demo
    if not isinstance(ir, Function):
        raise TypeError(f"expected Function, got {type(ir)}")
    return ir, cta_mesh, thread_mesh
