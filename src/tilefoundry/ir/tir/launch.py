"""``tir.Launch`` — host-side device-kernel launch effect Op.

``LaunchAttrs`` and ``CudaLaunchAttr`` are authored-IR metadata consumed by
``Launch``. CUDA lowering interprets their selector/value pairs; launch geometry
is emitted into the generated host entry and is not carried past codegen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from tilefoundry.ir.core.op import Op
from tilefoundry.ir.core.param_def import ParamDef


class CudaLaunchAttr(IntEnum):
    """Authored selector for a CUDA launch attribute.

    The values are a subset of ``cudaLaunchAttributeID`` and are interpreted
    by CUDA target lowering when carried in ``LaunchAttrs``.
    """

    COOPERATIVE = 1
    PROGRAMMATIC_STREAM_SERIALIZATION = 2
    CLUSTER_DIMENSION = 3


@dataclass(frozen=True)
class LaunchAttrs:
    """Authored launch attribute selector/value pairs.

    CUDA target lowering interprets ``entries``; unsupported values are
    rejected there.
    """

    entries: tuple[tuple[CudaLaunchAttr, object], ...] = ()


class Launch(Op):
    """Host launch of a device kernel — an effect Op producing no value."""

    cluster = ParamDef(kind="attribute", default=None)
    dynamic_smem = ParamDef(kind="attribute", default=0)
    stream = ParamDef(kind="attribute", default=None)
    attrs = ParamDef(kind="attribute", default=LaunchAttrs())


_LAUNCH_EXTENT_MUTATOR_TYPE = None


def _launch_extent_mutator_type():
    global _LAUNCH_EXTENT_MUTATOR_TYPE
    if _LAUNCH_EXTENT_MUTATOR_TYPE is None:
        from tilefoundry.ir.visitor import ExprMutator  # noqa: PLC0415

        class _LaunchExtentMutator(ExprMutator):
            def __init__(self, dimvar_src, dim_ops, i32, shape_of) -> None:
                super().__init__()
                self.dimvar_src = dimvar_src
                self.dim_ops = dim_ops
                self.i32 = i32
                self.shape_of = shape_of

            def visit_Constant(self, dim):
                return dim

            def visit_DimVar(self, dim):
                src = self.dimvar_src.get(id(dim))
                if src is None:
                    raise ValueError(
                        f"launch_call: launch extent references dimension variable "
                        f"{dim.name!r}, which is not a bare axis of any forwarded "
                        f"tensor argument; its runtime extent cannot be resolved"
                    )
                arg, axis = src
                return self.shape_of(type=self.i32, param=arg, axis=axis)

            def visit_Call(self, dim):
                if not isinstance(dim.target, self.dim_ops):
                    raise ValueError(
                        f"launch_call: unsupported launch extent {type(dim).__name__}"
                    )
                from dataclasses import replace  # noqa: PLC0415

                return replace(dim, args=tuple(self.visit(arg) for arg in dim.args))

            def default_visit(self, dim):
                raise ValueError(
                    f"launch_call: unsupported launch extent {type(dim).__name__}"
                )

        _LAUNCH_EXTENT_MUTATOR_TYPE = _LaunchExtentMutator
    return _LAUNCH_EXTENT_MUTATOR_TYPE


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
    """Build a host ``Evaluate(Launch(...), args)`` for *callee*.

    Launch extents canonicalize to scalar expressions. Bare ``DimVar`` values
    become ``ShapeOf`` calls only when exactly one forwarded tensor axis
    provides their identity; missing or ambiguous runtime sources are rejected.

    See [tir §2.3](docs/spec/tir.md#23-tir-ops).
    """
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
    from tilefoundry.ir.types.storage import StorageKind  # noqa: PLC0415

    forwarded_args = tuple(forwarded_args)
    _DIM_OPS = (DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax)
    i64 = TensorType.scalar(DType.i64, storage=StorageKind.RMEM)
    i32 = TensorType.scalar(DType.i32, storage=StorageKind.RMEM)

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
        if isinstance(dim, (Constant, DimVar, Call)):
            mutator = _launch_extent_mutator_type()(
                dimvar_src, _DIM_OPS, i32, ShapeOf
            )
            return mutator.visit(dim)
        raise ValueError(f"launch_call: unsupported launch extent {type(dim).__name__}")

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
    "CudaLaunchAttr",
    "LaunchAttrs",
]
