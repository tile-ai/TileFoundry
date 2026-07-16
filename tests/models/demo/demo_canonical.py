"""Canonical topology-aware demo fixture.

Uses ``@func(topologies=...)`` with nested
``with Mesh(topology="cta"/"thread", ...)`` scopes and inline
``ShardLayout(...)`` constructor kwarg values.

The old ``build_demo()`` (closure-captured ``shared_layout`` /
``reg_layout`` / ``cta_mesh`` / ``thread_mesh``) remains in
``tests/models/demo/demo_ir.py`` as temporary backward-compat for
existing lower/codegen tests.
"""

from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403  -- bind bare op names
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import (
    Layout,
    Mesh,
    S,
    ShardLayout,
    Topology,
)


@func(topologies=(Topology("cta", 128), Topology("thread", 8 * 32)))
def demo_canonical(
    a: Tensor[(1, 1536), "f32"],
) -> Tensor[(1, 1536), "f32"]:
    with Mesh(
        topology="cta", layout=Layout(shape=(128,), strides=(1,))
    ) as cta_mesh:
        b = reshard(
            a,
            layout=ShardLayout(
                layout=Layout(shape=(1, 1536), strides=(1536, 1)),
                attrs=(),
                mesh=cta_mesh,
            ),
            storage="smem",
        )
        with Mesh(
            topology="thread",
            layout=Layout(shape=(8, 32), strides=(32, 1)),
        ) as thread_mesh:
            c = reshard(
                b,
                layout=ShardLayout(
                    layout=Layout(shape=(1, 8, 192), strides=(1536, 192, 1)),
                    attrs=(S(1), S(2)),
                    mesh=thread_mesh,
                ),
                storage="rmem",
            )
            d = relu(c)
            e = reshard(
                d,
                layout=ShardLayout(
                    layout=Layout(shape=(1, 8, 192), strides=(1536, 192, 1)),
                    attrs=(S(1), S(2)),
                    mesh=thread_mesh,
                ),
                storage="gmem",
            )
            return e
    raise RuntimeError("unreachable")


def build_demo_canonical() -> Function:
    """Return the canonical ``hir.Function`` from the topology-aware fixture."""
    ir = demo_canonical
    if not isinstance(ir, Function):
        raise TypeError(f"expected Function, got {type(ir)}")
    return ir
