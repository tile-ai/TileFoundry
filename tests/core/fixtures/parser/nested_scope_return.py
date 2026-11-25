"""A return inside nested mesh scopes, after the inner scope rebinds a parameter."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 4),))
class Rebound:
    mesh = Mesh(("gpu",), (4,), names=("d",))

    @func
    def forward(h: Tensor[(4 @ mesh.d,), "f32", "gmem"]):
        with mesh:
            with Mesh(mesh, layout=(4,), names=("e",)) as _inner:
                h = h * 5.0
                return h * 3.0


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 4),))
class Looped:
    mesh = Mesh(("gpu",), (4,), names=("d",))

    @func
    def forward(h: Tensor[(4 @ mesh.d,), "f32", "gmem"]):
        with mesh:
            with Mesh(mesh, layout=(4,), names=("e",)) as _inner:
                for _ in range(2):
                    h = h + 1.0
                return h * 3.0
