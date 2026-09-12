"""The host ``.cpp`` translation unit: one entry, and the launches it makes.

The entry takes runtime tensors, checks each is on the device its parameter's
storage names, and calls a launch shim per ``Launch`` in its body. Which
arguments that call takes is the callee's answer, reached through the symbol
table, so nothing here spells a device type or a device ABI. Output uses only
TVM FFI, DLPack and standard C++.
"""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.codegen.cpu.context import CpuCodegenContext
from tilefoundry.codegen.cpu.templates import render
from tilefoundry.codegen.linkable import LinkableFunction, LinkableModule
from tilefoundry.codegen.registry import CodeGenerator
from tilefoundry.codegen.signature import (
    CallableSignature,
    LaunchSignature,
    tensor_signature_of,
)
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
)
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.target import Target

_STORAGE_DEVICE_TYPE = {
    StorageKind.GMEM: "kDLCUDA",
    StorageKind.HOST: "kDLCPU",
}

_DIM_BINOP_CXX = {
    DimAdd: "+",
    DimSub: "-",
    DimMul: "*",
    DimFloorDiv: "/",
    DimMod: "%",
}


class _HostIntExprVisitor(ExprVisitor[str]):
    def visit_Constant(self, expr: Constant, ctx=None) -> str:
        value = static_dim_value(expr)
        if value is None:
            raise ValueError(
                f"emit_host_module: unsupported launch-extent node {type(expr).__name__}"
            )
        return str(value)

    def visit_ShapeOf(self, expr: ShapeOf, ctx=None) -> str:
        return f"{expr.param.name}.shape()[{expr.axis}]"

    def visit_Call(self, expr: Call, ctx=None) -> str:
        target = expr.target
        sym = next((s for op, s in _DIM_BINOP_CXX.items() if isinstance(target, op)), None)
        if sym is not None:
            a, b = expr.args
            return f"({self.visit(a, ctx)} {sym} {self.visit(b, ctx)})"
        if isinstance(target, (DimMin, DimMax)):
            a, b = expr.args
            ca = self.visit(a, ctx)
            cb = self.visit(b, ctx)
            cmp = "<" if isinstance(target, DimMin) else ">"
            return f"(({ca}) {cmp} ({cb}) ? ({ca}) : ({cb}))"
        raise ValueError(f"emit_host_module: unsupported launch-extent op {type(target).__name__}")

    def default_visit(self, expr, ctx=None) -> str:
        raise ValueError(f"emit_host_module: unsupported launch-extent node {type(expr).__name__}")


def _extent(expr) -> str:
    """One grid or block extent as a host C++ int expression.

    An ``int`` or integer ``Constant`` is written as a literal, a ``ShapeOf``
    reads the forwarded tensor's shape, and dim arithmetic over those composes.
    Anything else raises -- there is no silent zero.
    """
    value = static_dim_value(expr)
    if value is not None:
        return str(value)
    if not isinstance(expr, (ShapeOf, Call, Constant)):
        raise ValueError(f"emit_host_module: unsupported launch-extent node {type(expr).__name__}")
    return f"static_cast<int>({_HostIntExprVisitor().visit(expr)})"


def _static_smem(value) -> int:
    """The dynamic shared-memory size, which the host states as a constant.

    Host index-expr codegen covers grid and block extents only; a shared-memory
    size that is decided at run time has nowhere to be decided.
    """
    static = static_dim_value(value)
    if static is None:
        raise ValueError(
            "emit_host_module: dynamic_smem must be a static int/Constant; a "
            f"dynamic shared-memory size expression is not supported, got "
            f"{type(value).__name__}"
        )
    return static


def _reject_unsupported_config(cfg: Launch) -> None:
    if cfg.cluster is not None:
        raise NotImplementedError("emit_host_module: launch `cluster` is not supported yet")
    if cfg.stream is not None:
        raise NotImplementedError("emit_host_module: launch `stream` is not supported yet")
    if cfg.attrs.entries:
        raise NotImplementedError("emit_host_module: launch `attrs` are not supported yet")


def _placement_line(name: str, storage) -> str:
    """Refuse a tensor that is not where the parameter's storage says it is."""
    device_type = _STORAGE_DEVICE_TYPE.get(storage)
    if device_type is None:
        raise ValueError(
            f"emit_host_module: parameter {name!r} storage {storage!r} cannot "
            f"be a host ABI tensor argument (kernel-internal storage or unset)"
        )
    return (
        f"if ({name}.device().device_type != {device_type}) "
        f'throw std::runtime_error("tilefoundry: argument {name!r} must be a '
        f'{device_type} tensor");'
    )


def _launch_statements(entry: PrimFunction) -> tuple[Evaluate, ...]:
    """The launches *entry* makes, which is the whole of what a host entry does."""
    body = entry.body
    statements = body.body if isinstance(body, Sequential) else ()
    if not statements or not all(
        isinstance(stmt, Evaluate) and isinstance(stmt.callable, Launch) for stmt in statements
    ):
        raise ValueError(
            f"emit_host_module: entry {entry.name!r} body must be one or more Launch statements"
        )
    return tuple(statements)


def _as_this_scope_names_it(
    shim: CallableSignature, forwarded: tuple[Var, ...], launch: Evaluate
) -> CallableSignature:
    """*shim*'s parameters rewritten as the values this launch passes for them.

    Order and kind are the callee's and do not change; every name does, because
    a caller writes what its own scope calls the thing. The ids the target
    states keep their names: the entry was told them under exactly those.
    """
    geometry = (
        *(_extent(arg) for arg in launch.args[1:7]),
        str(_static_smem(launch.callable.dynamic_smem)),
        "nullptr",
    )
    return replace(
        shim,
        params=tuple(tensor_signature_of(var) for var in forwarded),
        trailing=tuple(
            LaunchSignature(name=value, ctype=signature.ctype)
            for signature, value in zip(shim.trailing, geometry)
        ),
    )


def _forwarded(entry: PrimFunction, launch: Evaluate, callee: PrimFunction) -> tuple[Var, ...]:
    """The entry parameters this launch hands the callee, in the callee's order."""
    args = launch.args[7:]
    if len(args) != len(callee.params):
        raise ValueError(
            f"emit_host_module: launch passes {len(args)} args but device function "
            f"{callee.name!r} has {len(callee.params)} parameters"
        )
    held = {param.name: param for param in entry.params}
    for arg in args:
        if not isinstance(arg, Var):
            raise ValueError(
                "emit_host_module: launch args must be host entry parameters (Var); "
                "expressions are not accepted"
            )
        if arg.name not in held:
            raise ValueError(
                f"emit_host_module: launch arg {arg.name!r} is not a parameter "
                f"of entry {entry.name!r}"
            )
    return tuple(held[arg.name] for arg in args)


def _one_launch(
    entry: PrimFunction, launch: Evaluate, callee: PrimFunction, ctx: CpuCodegenContext
) -> list[str]:
    """The lines one ``Launch`` becomes: the checks it owes, then the call."""
    _reject_unsupported_config(launch.callable)
    forwarded = _forwarded(entry, launch, callee)
    shim = ctx.signature_of(callee)
    bound = _as_this_scope_names_it(shim, forwarded, launch)
    lines = [
        _placement_line(var.name, param.type.storage)
        for var, param in zip(forwarded, callee.params)
    ]
    lines.append(f"{shim.name}({ctx.arguments(bound, callee.target)});")
    return lines


def _declare_callee(callee: PrimFunction, ctx: CpuCodegenContext) -> str:
    """The forward declaration of a symbol another unit defines.

    Written by the callee's own target, so this unit and the one that defines
    it cannot come to disagree about what the symbol takes.
    """
    shim = ctx.signature_of(callee)
    return (
        f'extern "C" void {shim.name}'
        f"({ctx.parameters(shim, callee.target, exported=True)});"
    )


def _host_entry(
    entry: PrimFunction,
    signature: CallableSignature,
    body_lines: list[str],
    ctx: CpuCodegenContext,
) -> LinkableFunction:
    """*entry* in both of its positions, written from the one signature it is called by."""
    return LinkableFunction(
        name=entry.name,
        declaration=f"void {signature.name}({ctx.parameters(signature)});",
        definition=render(
            "cpu_entry.cpp.j2",
            host_symbol=signature.name,
            params=ctx.parameters(signature),
            body_lines=body_lines,
            entry_name=entry.name,
        ),
    )


def emit_host_module(
    module: Module,
    functions: tuple[PrimFunction, ...],
    target: Target,
    ctx: CpuCodegenContext,
) -> LinkableModule:
    """Emit the host ``.cpp`` linkable module for a CPU *entry*.

    *module* is the enclosing ``Module``, which resolves each launch's callee;
    the FFI export names the symbol it exports, so it is written with the
    definition and not in the preamble every declaration follows.
    """
    if len(functions) != 1:
        raise ValueError("emit_host_module: expected exactly one CPU host entry")
    entry = functions[0]
    declarations: dict[str, str] = {}
    body_lines: list[str] = []
    for launch in _launch_statements(entry):
        callee = module.lookup(launch.args[0].name)
        declarations[ctx.signature_of(callee).name] = _declare_callee(callee, ctx)
        body_lines += _one_launch(entry, launch, callee, ctx)
    return LinkableModule(
        target="cpu",
        language="cpp",
        preamble=render("cpu_preamble.cpp.j2", shim_decls=list(declarations.values())),
        functions=(_host_entry(entry, ctx.signature_of(entry), body_lines, ctx),),
    )


CPU_CODE_GENERATOR = CodeGenerator(emit_host_module)


__all__ = ["emit_host_module"]
