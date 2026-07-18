"""``tir.Launch`` — host-side device-kernel launch effect Op.

The authored-IR launch descriptors (``LaunchAttrs`` / ``LaunchConfig`` /
``CudaLaunchAttr`` / ``CudaClusterDim``) live beside the Op that consumes
them. ``LaunchConfig`` describes a launch *as authored in the IR*: ``grid``
and ``block`` are 3-tuples of scalar expressions (an ``int`` constant or a
dim-arithmetic ``Expr`` such as ``ceildiv(N, tile)``), so the host can
compute a runtime grid. This is distinct from
:class:`tilefoundry.runtime.module.LaunchConfig`, which is the post-codegen,
fully-resolved ``dim3`` metadata of the generated launcher — alias-import one
of them when both are used in the same module. ``cluster`` / ``stream`` /
``attrs`` are representable but a target's lowering rejects any value it
does not support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Union

from tilefoundry.ir.core.op import Op
from tilefoundry.ir.core.param_def import ParamDef

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
    """Launch configuration authored in the IR (CuTeDSL ``.launch`` shape).

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


class Launch(Op):
    """Host launch of a device kernel — an effect Op producing no value."""

    cluster = ParamDef(kind="attribute", default=None)
    dynamic_smem = ParamDef(kind="attribute", default=0)
    stream = ParamDef(kind="attribute", default=None)
    attrs = ParamDef(kind="attribute", default=LaunchAttrs())


def launch_call(
    callee,
    forwarded_args,
    grid,
    block,
    *,
    cluster=None,
    dynamic_smem=0,
    stream=None,
    attrs=None,
):
    """Build ``Evaluate(Launch(...), args)`` for a host launch of *callee*.

    *grid* / *block* are 3-tuples of launch extents (``int`` / ``Constant`` /
    ``DimVar`` / dim-arithmetic ``Call``); each is canonicalised to an Expr:
    an integer extent becomes a rank-0 i64 ``Constant``, a bare ``DimVar``
    becomes a ``ShapeOf`` of the forwarded tensor argument whose callee tensor
    parameter carries that exact ``DimVar`` identity at that axis, and a
    dim-arithmetic ``Call`` keeps its op with recursively canonicalised
    operands. A ``DimVar`` not found on any forwarded tensor input, or bound to
    more than one forwarded ``(tensor, axis)`` source, is rejected — the host
    cannot pick a runtime extent source silently.
    """
    from dataclasses import replace  # noqa: PLC0415

    from tilefoundry.ir.core import Call, Constant  # noqa: PLC0415
    from tilefoundry.ir.tir.shape import ShapeOf  # noqa: PLC0415
    from tilefoundry.ir.tir.stmts import Evaluate  # noqa: PLC0415
    from tilefoundry.ir.tir.symbol_ref import SymbolRef  # noqa: PLC0415
    from tilefoundry.ir.types import (  # noqa: PLC0415
        CallableType,
        DType,
        TensorType,
        callable_type_for_prim_function,
    )
    from tilefoundry.ir.types.dim import (  # noqa: PLC0415
        DimAdd,
        DimFloorDiv,
        DimMax,
        DimMin,
        DimMod,
        DimMul,
        DimSub,
        DimVar,
    )

    forwarded_args = tuple(forwarded_args)
    _DIM_OPS = (DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax)
    i64 = TensorType.scalar(DType.i64)
    i32 = TensorType.scalar(DType.i32)

    # DimVar identity -> (forwarded tensor arg, axis), from each callee tensor
    # parameter zipped positionally with its forwarded argument. Bare-variable
    # axes only; a variable bound to two different sources is rejected.
    dimvar_src: dict[int, tuple] = {}
    for param, arg in zip(callee.params, forwarded_args):
        pty = getattr(param, "type", None)
        if not isinstance(pty, TensorType):
            continue
        for axis, dim in enumerate(pty.shape):
            if not isinstance(dim, DimVar):
                continue
            src = (arg, axis)
            prev = dimvar_src.get(id(dim))
            if prev is not None and prev != src:
                raise ValueError(
                    f"launch_call: dimension variable {dim.name!r} is bound to "
                    f"more than one forwarded tensor source; a host launch "
                    f"extent cannot choose one without a runtime shape check"
                )
            dimvar_src.setdefault(id(dim), src)

    def _canon(dim):
        if isinstance(dim, bool):
            raise ValueError(f"launch_call: bool is not a launch extent: {dim!r}")
        if isinstance(dim, int):
            return Constant(type=i64, value=dim)
        if isinstance(dim, Constant):
            return dim
        if isinstance(dim, DimVar):
            src = dimvar_src.get(id(dim))
            if src is None:
                raise ValueError(
                    f"launch_call: launch extent references dimension variable "
                    f"{dim.name!r}, which is not a bare axis of any forwarded "
                    f"tensor argument; its runtime extent cannot be resolved"
                )
            arg, axis = src
            return ShapeOf(type=i32, param=arg, axis=axis)
        if isinstance(dim, Call) and isinstance(dim.target, _DIM_OPS):
            return replace(dim, args=tuple(_canon(a) for a in dim.args))
        raise ValueError(
            f"launch_call: unsupported launch extent {type(dim).__name__}"
        )

    grid_e = tuple(_canon(d) for d in grid)
    block_e = tuple(_canon(d) for d in block)

    callee_type = getattr(callee, "type", None)
    if not isinstance(callee_type, CallableType):
        callee_type = callable_type_for_prim_function(callee)
    ref = SymbolRef(name=callee.name, type=callee_type)
    op = Launch(
        cluster=cluster,
        dynamic_smem=dynamic_smem,
        stream=stream,
        attrs=attrs if attrs is not None else LaunchAttrs(),
    )
    return Evaluate(callable=op, args=(ref, *grid_e, *block_e, *forwarded_args))


__all__ = [
    "Launch",
    "launch_call",
    "CudaClusterDim",
    "CudaLaunchAttr",
    "LaunchAttrs",
    "LaunchConfig",
]
