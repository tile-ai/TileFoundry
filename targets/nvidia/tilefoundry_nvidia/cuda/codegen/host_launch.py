"""CUDA launch declarations, placement checks, and calls in CPU host entries."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from tilefoundry import codegen
from tilefoundry.codegen import cpu as codegen_cpu
from tilefoundry.ir import core, expr, tir
from tilefoundry_nvidia.cuda import target

STORAGE_DEVICE_TYPE = MappingProxyType({
    expr.StorageKind.GMEM: "kDLCUDA",
    expr.StorageKind.HOST: "kDLCPU",
})


class HostIntExprVisitor(core.ExprVisitor[str]):
    def visit_Constant(self, node: expr.Constant, ctx=None) -> str:
        value = expr.static_dim_value(node)
        if value is None:
            raise ValueError(
                f"emit_host_module: unsupported launch-extent node {type(node).__name__}"
            )
        return str(value)

    def visit_ShapeOf(self, node: core.ShapeOf, ctx=None) -> str:
        return f"{node.param.name}.shape()[{node.axis}]"

    def visit_Call(self, node: expr.Call, ctx=None) -> str:
        rendered = codegen.render_dim_call(node, lambda arg: self.visit(arg, ctx))
        if rendered is None:
            raise ValueError(f"emit_host_module: unsupported launch-extent op {type(node.target).__name__}")
        return rendered

    def default_visit(self, node, ctx=None) -> str:
        raise ValueError(f"emit_host_module: unsupported launch-extent node {type(node).__name__}")


def extent(node) -> str:
    """One grid or block extent as a host C++ int expression.

    An ``int`` or integer ``Constant`` is written as a literal, a ``ShapeOf``
    reads the forwarded tensor's shape, and dim arithmetic over those composes.
    Anything else raises -- there is no silent zero.
    """
    value = expr.static_dim_value(node)
    if value is not None:
        return str(value)
    if not isinstance(node, (core.ShapeOf, expr.Call, expr.Constant)):
        raise ValueError(f"emit_host_module: unsupported launch-extent node {type(node).__name__}")
    return f"static_cast<int>({HostIntExprVisitor().visit(node)})"


def static_smem(value) -> int:
    """The dynamic shared-memory size, which the host states as a constant.

    Host index-expr codegen covers grid and block extents only; a shared-memory
    size that is decided at run time has nowhere to be decided.
    """
    static = expr.static_dim_value(value)
    if static is None:
        raise ValueError(
            "emit_host_module: dynamic_smem must be a static int/Constant; a "
            f"dynamic shared-memory size expression is not supported, got "
            f"{type(value).__name__}"
        )
    return static


def reject_unsupported_config(cfg: tir.launch.Launch) -> None:
    if cfg.cluster is not None:
        raise NotImplementedError("emit_host_module: launch `cluster` is not supported yet")
    if cfg.stream is not None:
        raise NotImplementedError("emit_host_module: launch `stream` is not supported yet")
    if cfg.attrs.entries:
        raise NotImplementedError("emit_host_module: launch `attrs` are not supported yet")


def placement_line(name: str, storage) -> str:
    """Refuse a tensor that is not where the parameter's storage says it is."""
    device_type = STORAGE_DEVICE_TYPE.get(storage)
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


def as_this_scope_names_it(
    launcher: codegen.CallableSignature, forwarded: tuple[expr.Var, ...], launch: core.Evaluate
) -> codegen.CallableSignature:
    """*launcher*'s parameters rewritten as the values this launch passes for them.

    Order and kind are the callee's and do not change; every name does, because
    a caller writes what its own scope calls the thing. The ids the target
    states keep their names: the entry was told them under exactly those.
    """
    geometry = (
        extent(launch.args[1]),
        extent(launch.args[4]),
        str(static_smem(launch.callable.dynamic_smem)),
    )
    return replace(
        launcher,
        params=tuple(codegen.tensor_signature_of(var) for var in forwarded),
        trailing=tuple(
            codegen.LaunchSignature(name=value, ctype=signature.ctype)
            for signature, value in zip(launcher.trailing, geometry)
        ),
    )


def forwarded_params(entry: core.PrimFunction, launch: core.Evaluate, signature: codegen.CallableSignature) -> tuple[expr.Var, ...]:
    """The entry parameters this launch hands the callee, in the callee's order."""
    args = launch.args[7:]
    if len(args) != len(signature.params):
        raise ValueError(
            f"emit_host_module: launch passes {len(args)} args but device function "
            f"{signature.name!r} has {len(signature.params)} parameters"
        )
    held = {param.name: param for param in entry.params}
    for arg in args:
        if not isinstance(arg, expr.Var):
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


def one_launch(
    entry: core.PrimFunction, launch: core.Evaluate, callee: core.PrimFunction, ctx: codegen_cpu.Context
) -> list[str]:
    """The lines one ``Launch`` becomes: the checks it owes, then the call.

    The launcher returns the CUDA error text of a launch the device refused, and the call raises it.
    """
    reject_unsupported_config(launch.callable)
    launcher = ctx.signature_of(callee)
    forwarded = forwarded_params(entry, launch, launcher)
    bound = as_this_scope_names_it(launcher, forwarded, launch)
    lines = [
        placement_line(var.name, param.type.storage)
        for var, param in zip(forwarded, launcher.params)
    ]
    call = f"{launcher.name}({arguments(bound, ctx)})"
    lines.append(f"if (const char* error = {call}) {{")
    lines.append(
        f'  TVM_FFI_THROW(RuntimeError) << "tilefoundry: launch of {callee.name} failed: "'
        " << error;"
    )
    lines.append("}")
    return lines


def declare_callee(callee: core.PrimFunction, ctx: codegen_cpu.Context) -> str:
    """The forward declaration of a symbol another unit defines.

    Written by the callee's own target, so this unit and the one that defines
    it cannot come to disagree about what the symbol takes.
    """
    launcher = ctx.signature_of(callee)
    return f'extern "C" const char* {launcher.name}({ctx.parameters(launcher, exported=True, by=target.Target)});'


def arguments(signature: codegen.CallableSignature, ctx: codegen_cpu.Context) -> str:
    return ", ".join(
        token for param in signature.all_params
        for token in codegen.registry.require((target.Target, codegen.Role.CALLER, type(param)))(param, ctx)
    )


@codegen.register(target.Target, codegen.Role.LAUNCH, tir.launch.Launch)
def emit_host_launch(entry, stmt, ctx):
    callee = ctx.callee_of(stmt)
    return declare_callee(callee, ctx), one_launch(entry, stmt, callee, ctx)


__all__: list[str] = []
