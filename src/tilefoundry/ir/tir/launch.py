"""``tir.Launch`` — host-side device-kernel launch effect Op."""
from __future__ import annotations

from tilefoundry.ir.core.op import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.target.launch import LaunchAttrs


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


__all__ = ["Launch", "launch_call"]
