"""A child Module with no Target takes a Split parameter from a two-GPU H200 root.

``Child`` parses its annotation with no storage owners; the root's argument carries
the H200 owner strides. ``RootWithTarget`` calls ``ChildWithTarget``, the same child
stating the root's Target, and is the reference the inherited call must match.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

_H200 = cuda.Target("nvidia.h200_sxm", device_count=2)
_GPU = (Topology("gpu", 2),)


@module(entry="forward", topologies=_GPU)
class Child:
    mesh = Mesh(("gpu",), (2,), names=("cp",))

    @func
    def forward(x: Tensor[(8 @ mesh.cp,), "f32", "gmem"]):
        with mesh:
            return x + x


@module(entry="forward", target=_H200, topologies=_GPU)
class ChildWithTarget:
    mesh = Mesh(("gpu",), (2,), names=("cp",))

    @func
    def forward(x: Tensor[(8 @ mesh.cp,), "f32", "gmem"]):
        with mesh:
            return x + x


@module(entry="forward", target=_H200, topologies=_GPU)
class Root:
    mesh = Mesh(("gpu",), (2,), names=("cp",))
    child = Child

    @func
    def forward(x: Tensor[(8 @ mesh.cp,), "f32", "gmem"]):
        with mesh:
            return child(x)


@module(entry="forward", target=_H200, topologies=_GPU)
class RootWithTarget:
    mesh = Mesh(("gpu",), (2,), names=("cp",))
    child = ChildWithTarget

    @func
    def forward(x: Tensor[(8 @ mesh.cp,), "f32", "gmem"]):
        with mesh:
            return child(x)
