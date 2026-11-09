"""``tir.Launch`` — host-side device-kernel launch effect Op.

``LaunchAttrs`` carries target-owned selector/value pairs consumed by
``Launch``. Target lowering interprets these attributes; launch geometry
is emitted into the generated host entry and is not carried past codegen.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from tilefoundry.ir import core, expr, types


@dataclass(unsafe_hash=True)
class LaunchAttrs:
    """Authored launch attribute selector/value pairs.

    Target lowering interprets ``entries``; unsupported values are
    rejected there.
    """

    entries: tuple[tuple[Enum, object], ...] = ()


class Launch(expr.Op):
    """Host launch of a device kernel — an effect Op producing no value."""

    cluster = expr.ParamDef(kind="attribute", default=None)
    dynamic_smem = expr.ParamDef(kind="attribute", default=0)
    stream = expr.ParamDef(kind="attribute", default=None)
    attrs = expr.ParamDef(kind="attribute", default=LaunchAttrs())


DIM_OPS = (
    expr.DimAdd,
    expr.DimSub,
    expr.DimMul,
    expr.DimFloorDiv,
    expr.DimMod,
    expr.DimMin,
    expr.DimMax,
)


class LaunchExtentMutator(core.ExprCloner):
    """Rewrite a launch extent so each bare ``DimVar`` reads a forwarded tensor axis."""

    def __init__(self, dimvar_src, dim_ops, i32, shape_of) -> None:
        super().__init__()
        self.dimvar_src = dimvar_src
        self.dim_ops = dim_ops
        self.i32 = i32
        self.shape_of = shape_of

    def visit_Constant(self, dim, ctx=None):
        return dim

    def visit_DimVar(self, dim, ctx=None):
        src = self.dimvar_src.get(id(dim))
        if src is None:
            raise ValueError(
                f"launch_call: launch extent references dimension variable "
                f"{dim.name!r}, which is not a bare axis of any forwarded "
                f"tensor argument; its runtime extent cannot be resolved"
            )
        arg, axis = src
        return self.shape_of(type=self.i32, param=arg, axis=axis)

    def visit_Call(self, dim, ctx=None):
        if not isinstance(dim.target, self.dim_ops):
            raise ValueError(
                f"launch_call: unsupported launch extent {type(dim).__name__}"
            )
        return replace(dim, args=tuple(self.visit(arg, ctx) for arg in dim.args))

    def default_visit(self, dim, ctx=None):
        raise ValueError(
            f"launch_call: unsupported launch extent {type(dim).__name__}"
        )


def launch_call(
    callee,
    forwarded_args,
    grid,
    block,
    *,
    symbol=None,
    cluster=None,
    dynamic_smem=0,
    stream=None,
    attrs=None,
):
    """Build a host ``Evaluate(Launch(...), args)`` for *callee*.

    Launch extents canonicalize to scalar expressions. Bare ``DimVar`` values
    become ``ShapeOf`` calls only when exactly one forwarded tensor axis
    provides their identity; missing or ambiguous runtime sources are rejected.

    See [tir §2.3](docs/spec/tir.md#23-tir-ops).
    """
    forwarded_args = tuple(forwarded_args)
    i64 = types.TensorType.from_scalar(expr.DType.i64, storage=expr.StorageKind.RMEM)
    i32 = types.TensorType.from_scalar(expr.DType.i32, storage=expr.StorageKind.RMEM)

    dimvar_src: dict[int, tuple] = {}
    for param, arg in zip(callee.params, forwarded_args):
        pty = getattr(param, "type", None)
        if not isinstance(pty, types.TensorType):
            continue
        for axis, dim in enumerate(pty.shape):
            if not isinstance(dim, expr.DimVar):
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

    def canon(dim):
        if isinstance(dim, bool):
            raise ValueError(f"launch_call: bool is not a launch extent: {dim!r}")
        if isinstance(dim, int):
            return expr.Constant(type=i64, value=dim)
        if isinstance(dim, (expr.Constant, expr.DimVar, expr.Call)):
            mutator = LaunchExtentMutator(dimvar_src, DIM_OPS, i32, core.ShapeOf)
            return mutator.visit(dim)
        raise ValueError(f"launch_call: unsupported launch extent {type(dim).__name__}")

    grid_e = tuple(canon(d) for d in grid)
    block_e = tuple(canon(d) for d in block)

    ref = core.symbol_ref(callee, symbol)
    op = Launch(
        cluster=cluster,
        dynamic_smem=dynamic_smem,
        stream=stream,
        attrs=attrs if attrs is not None else LaunchAttrs(),
    )
    return core.Evaluate(callable=op, args=(ref, *grid_e, *block_e, *forwarded_args))


__all__ = [
    "Launch",
    "launch_call",
    "LaunchAttrs",
]
