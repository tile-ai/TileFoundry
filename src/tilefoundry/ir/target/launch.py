"""Launch configuration types for the host ``launch`` operation.

These describe a device kernel launch *as authored in the IR*: ``grid`` and
``block`` are 3-tuples of scalar expressions (an ``int`` constant or a
dim-arithmetic ``Expr`` such as ``ceildiv(N, tile)``), so the host can compute
a runtime grid. This is distinct from
:class:`tilefoundry.runtime.module.LaunchConfig`, which is the post-codegen,
fully-resolved ``dim3`` metadata of the generated launcher — alias-import one
of them when both are used in the same module.

``cluster`` / ``stream`` / ``attrs`` are representable but a target's lowering
rejects any value it does not support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from tilefoundry.ir.types.shape_dim import ShapeDim


class CudaLaunchAttr(IntEnum):
    """CUDA launch attribute selector (subset of ``cudaLaunchAttributeID``).

    CUDA-prefixed because the value space is backend-specific; another
    backend would define its own attribute enum.
    """

    COOPERATIVE = 1
    PROGRAMMATIC_STREAM_SERIALIZATION = 2
    CLUSTER_DIMENSION = 3


class CudaClusterDim(IntEnum):
    """Axis index for a CUDA thread-block-cluster dimension."""

    X = 0
    Y = 1
    Z = 2


@dataclass(frozen=True)
class LaunchAttrs:
    """Neutral container of target-interpreted launch attributes.

    ``entries`` pairs an attribute selector with its value; CUDA lowering
    interprets them. v1 is empty by default.
    """

    entries: tuple[tuple[CudaLaunchAttr, object], ...] = ()


@dataclass(frozen=True)
class LaunchConfig:
    """Launch configuration authored in the IR (a ``.launch``-style grid/block shape).

    ``grid`` / ``block`` (and ``cluster`` when present) are 3-tuples of
    compile-time launch extents — ``ShapeDim`` (an ``int``, a ``DimVar``, or a
    dim-arithmetic expression such as ``ceildiv(S, tile)``) — so the host can
    size them from a runtime shape; ``dynamic_smem`` is a ``ShapeDim`` or
    ``int``. These are launch configuration, not runtime Exprs. ``stream`` /
    ``attrs`` are representable; unsupported values error in the target's
    lowering.
    """

    grid: "tuple[ShapeDim, ShapeDim, ShapeDim]"
    block: "tuple[ShapeDim, ShapeDim, ShapeDim]"
    cluster: "Optional[tuple[ShapeDim, ShapeDim, ShapeDim]]" = None
    dynamic_smem: "Union[ShapeDim, int]" = 0
    stream: object = None
    attrs: LaunchAttrs = field(default_factory=LaunchAttrs)


__all__ = [
    "CudaClusterDim",
    "CudaLaunchAttr",
    "LaunchAttrs",
    "LaunchConfig",
]
