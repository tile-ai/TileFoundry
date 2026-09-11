"""Emit the host module for the split code-generation pipeline.

Launch entries validate and bind runtime tensors before calling a device shim.
Dispatch entries use a first-match shape predicate and throw on fallback. Output
uses only TVM FFI, DLPack, and standard C++; CUDA syntax and types remain in the
device module and shims.
"""

from __future__ import annotations

from tilefoundry.codegen import names
from tilefoundry.codegen.cpu.templates import render
from tilefoundry.codegen.linkable import LinkableFunction, LinkableModule
from tilefoundry.codegen.registry import CodeGenerator
from tilefoundry.codegen.signature import (
    LAUNCH_ABI,
    CallableSignature,
    declare,
    declare_types,
    tensor_signature_of,
)
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.pattern import DimVarRangePat, locate_dim_var
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import (
    ShapeOf,
)
from tilefoundry.ir.tir.shape import (
    is_hidden_shape_scalar as _is_hidden_shape_scalar,
)
from tilefoundry.ir.tir.shape import (
    parse_shape_var_name as _parse_shape_param_name,
)
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.ir.tir.symbol_ref import symbol_call
from tilefoundry.ir.types import DType, TensorType
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


def _static_smem(value) -> int:
    """Resolve ``dynamic_smem`` to a static int.

    Resolve ``dynamic_smem`` to a static int. Host index-expr codegen
    covers grid/block extents only; a dynamic shared-memory size expression
    is not supported.
    """
    sv = static_dim_value(value)
    if sv is not None:
        return sv
    raise ValueError(
        "emit_host_module: dynamic_smem must be a static int/Constant; a "
        f"dynamic shared-memory size expression is not supported, got "
        f"{type(value).__name__}"
    )


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


def _emit_host_int_expr(expr) -> str:
    """Lower a launch-extent Expr to a host C++ integer expression.

    Accepts an ``int`` / integer ``Constant`` (emitted as a literal), a
    ``ShapeOf`` (the forwarded tensor's ``shape()`` access), or a
    dim-arithmetic ``Call`` over those. Any unsupported node raises — there
    is no silent zero/default.
    """
    sv = static_dim_value(expr)
    if sv is not None:
        return str(sv)
    if isinstance(expr, (ShapeOf, Call, Constant)):
        return _HostIntExprVisitor().visit(expr)
    raise ValueError(f"emit_host_module: unsupported launch-extent node {type(expr).__name__}")


def _hidden_names(params) -> set:
    return {p.name for p in params if _is_hidden_shape_scalar(p, params)}


def _is_user_scalar(p, hidden: set) -> bool:
    return p.name not in hidden and isinstance(p.type, TensorType) and not p.type.shape


def _is_tensor(p, hidden: set) -> bool:
    return p.name not in hidden and not _is_user_scalar(p, hidden)


def _extent(c):
    value = static_dim_value(c)
    return str(value) if value is not None else f"static_cast<int>({_emit_host_int_expr(c)})"


def _call_arg(p, host_names, hidden):
    name = host_names[p.name]
    if _is_tensor(p, hidden):
        return f"{name}.data_ptr()"
    if p.name in hidden:
        return name
    return f"static_cast<long long>({name})"


def _placement_line(name: str, storage) -> str:
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


def _shim_decl(fn: PrimFunction) -> str:
    """A types-only ``extern "C"`` forward declaration of *fn*'s launch shim."""
    hidden = _hidden_names(fn.params)
    shim = CallableSignature(
        name=names.launch_shim(fn.name),
        params=tuple(tensor_signature_of(p) for p in fn.params),
        trailing=LAUNCH_ABI,
    )
    tokens = declare_types(
        shim.all_params, lambda p: "void*" if _is_tensor(p, hidden) else "long long"
    )
    return f'extern "C" void {shim.name}({tokens});'


def emit_host_module(
    module: Module, functions: tuple[PrimFunction, ...], target: Target
) -> LinkableModule:
    """Emit the host ``.cpp`` linkable module for a CPU *entry*.

    *module* is the enclosing ``Module``; the dispatch path resolves each
    case's ``SymbolRef`` callee through ``module.lookup`` to read the variant's
    parameters.
    """
    if len(functions) != 1:
        raise ValueError("emit_host_module: expected exactly one CPU host entry")
    entry = functions[0]
    body = entry.body
    if (
        isinstance(body, Sequential)
        and body.body
        and all(
            isinstance(stmt, Evaluate) and isinstance(stmt.callable, Launch) for stmt in body.body
        )
    ):
        if len(body.body) > 1:
            shim_decls, body_lines, sig = _lower_launches(entry, body.body, module)
        else:
            shim_decls, body_lines, sig = _lower_launch(entry, body.body[0], module)
    elif entry.variants:
        pat = entry.variants[0].specializations[0]
        loc_idx = locate_dim_var(entry.params, pat.dim_var)
        loc = (entry.params[loc_idx[0]], loc_idx[1]) if loc_idx is not None else None
        if loc is None:
            raise ValueError("emit_host_module: cannot derive specialization subject")
        p, axis = loc
        subject = ShapeOf(type=TensorType.scalar(DType.i32), param=p, axis=axis)

        def _variant_call(v):
            args = []
            for vp in v.params:
                parsed = _parse_shape_param_name(vp.name)
                if parsed is None:
                    args.append(next(p for p in entry.params if p.name == vp.name))
                else:
                    base, ax = parsed
                    ep = next(p for p in entry.params if p.name == base)
                    args.append(ShapeOf(type=vp.type, param=ep, axis=ax))
            return symbol_call(v, tuple(args))

        calls = tuple(_variant_call(v) for v in entry.variants)
        shim_decls, body_lines, sig = _lower_dispatch(
            entry,
            module,
            callee_name=entry.name,
            subject=subject,
            variants=entry.variants,
            calls=calls,
        )
    else:
        raise ValueError(f"emit_host_module: entry {entry.name!r} body must be a single Launch")
    source = render(
        "cpu_module.cpp.j2",
        shim_decls=shim_decls,
        internal_host_symbol=names.host_entry(entry.name),
        wrapper_params_sig=sig,
        body_lines=body_lines,
        entry_name=entry.name,
    )
    return LinkableModule(
        target="cpu",
        language="cpp",
        source=source,
        functions=(LinkableFunction(name=entry.name, source=source),),
    )


CPU_CODE_GENERATOR = CodeGenerator(emit_host_module)


def _lower_launch(entry: PrimFunction, evaluate, module):
    device_fn = module.lookup(evaluate.args[0].name)
    if device_fn.variants:
        dim_name = device_fn.variants[0].specializations[0].dim_var
        loc_idx = locate_dim_var(device_fn.params, dim_name)
        loc = (device_fn.params[loc_idx[0]], loc_idx[1]) if loc_idx is not None else None
        if loc is None:
            raise ValueError(f"cannot derive specialization subject {dim_name!r}")
        p, axis = loc
        subject = ShapeOf(type=TensorType.scalar(DType.i32), param=p, axis=axis)
        calls = tuple(symbol_call(v, tuple(evaluate.args[7:])) for v in device_fn.variants)
        return _lower_dispatch(
            entry,
            module,
            callee_name=device_fn.name,
            subject=subject,
            variants=device_fn.variants,
            calls=calls,
        )
    return _lower_launches(entry, (evaluate,), module)


def _lower_launches(entry: PrimFunction, evaluates, module):
    """Lower one or more ``Launch`` statements of a host entry."""
    bindings = []
    kinds = {}
    used = set()
    for evaluate in evaluates:
        launch_op = evaluate.callable
        device_fn = module.lookup(evaluate.args[0].name)
        if device_fn.variants:
            raise NotImplementedError(
                "emit_host_module: multi-launch entries do not support variants"
            )
        dev_params = device_fn.params
        _reject_unsupported_config(launch_op)
        hidden = _hidden_names(dev_params)
        visible_dev = [p for p in dev_params if p.name not in hidden]
        args = evaluate.args[7:]
        if len(args) != len(visible_dev):
            raise ValueError(
                f"emit_host_module: launch passes {len(args)} args but device "
                f"function {device_fn.name!r} has {len(visible_dev)} host-visible "
                "parameters (hidden shape scalars are derived from tensor shapes)"
            )
        if not all(isinstance(a, Var) for a in args):
            raise ValueError(
                "emit_host_module: launch args must be host entry parameters (Var); "
                "expressions are not accepted"
            )
        bound = []
        for arg in args:
            ep = next((p for p in entry.params if p.name == arg.name), None)
            if ep is None:
                raise ValueError(
                    f"emit_host_module: launch arg {arg.name!r} is not a parameter "
                    f"of entry {entry.name!r}"
                )
            bound.append(ep)
        local_seen = set()
        host_name_of = {}
        for vp, ep in zip(visible_dev, bound):
            if id(ep) in local_seen:
                raise ValueError(
                    f"emit_host_module: entry parameter {ep.name!r} is bound more "
                    "than once in one launch"
                )
            local_seen.add(id(ep))
            used.add(id(ep))
            kind = _is_tensor(vp, hidden)
            if id(ep) in kinds and kinds[id(ep)] != kind:
                raise ValueError(
                    f"emit_host_module: entry parameter {ep.name!r} is bound to "
                    "incompatible device parameter types across launches"
                )
            kinds[id(ep)] = kind
            host_name_of[vp.name] = ep.name
        bindings.append((evaluate, device_fn, hidden, host_name_of))
    for ep in entry.params:
        if id(ep) not in used:
            raise ValueError(
                f"emit_host_module: entry parameter {ep.name!r} is not used by any "
                f"launch in {entry.name!r} -- an unused parameter has no device "
                "type to give the wrapper's signature"
            )
    body_lines = []
    shim_decls = []

    for index, (evaluate, device_fn, hidden, dev_to_host) in enumerate(bindings):
        dev_params = device_fn.params
        launch_op = evaluate.callable
        host_names = {
            **dev_to_host,
            **{p.name: f"l{index}__{p.name}" for p in dev_params if p.name in hidden},
        }
        for p in dev_params:
            if _is_tensor(p, hidden):
                body_lines.append(_placement_line(host_names[p.name], p.type.storage))
        for p in dev_params:
            if p.name in hidden:
                base, axis = _parse_shape_param_name(p.name)
                host_base = dev_to_host.get(base)
                if host_base is None:
                    raise ValueError(
                        f"emit_host_module: hidden shape scalar {p.name!r} "
                        f"references unknown base parameter {base!r}"
                    )
                body_lines.append(
                    f"long long {host_names[p.name]} = static_cast<long long>("
                    f"{host_base}.shape()[{axis}]);"
                )

        grid = tuple(_extent(c) for c in evaluate.args[1:4])
        block = tuple(_extent(c) for c in evaluate.args[4:7])
        call_args = [_call_arg(p, host_names, hidden) for p in dev_params] + [
            *grid,
            *block,
            str(_static_smem(launch_op.dynamic_smem)),
            "nullptr",
        ]
        body_lines.append(f"{names.launch_shim(device_fn.name)}({', '.join(call_args)});")
        shim_decls.append(_shim_decl(device_fn))
    wrapper = CallableSignature(
        name=names.host_entry(entry.name),
        params=tuple(tensor_signature_of(ep) for ep in entry.params),
    )
    host_ctype = {ep.name: "tvm::ffi::Tensor" if kinds[id(ep)] else "int" for ep in entry.params}
    return shim_decls, body_lines, declare(wrapper.all_params, lambda p: host_ctype[p.name])


def _lower_dispatch(entry: PrimFunction, module, *, callee_name, subject, variants, calls):
    if not isinstance(subject, ShapeOf):
        raise NotImplementedError(
            "emit_host_module: dispatch v1 expects exactly one ShapeOf subject"
        )
    for variant in variants:
        if len(variant.specializations) != 1 or not isinstance(
            variant.specializations[0], DimVarRangePat
        ):
            raise NotImplementedError(
                "emit_host_module: dispatch v1 expects exactly one DimVarRangePat per case"
            )

    entry_params = entry.params
    entry_names = {p.name for p in entry_params}
    hidden = _hidden_names(entry_params)

    def _host_name(ref) -> str:
        nm = ref.name if isinstance(ref, Var) else ref.param.name
        if nm not in entry_names:
            raise ValueError(
                f"emit_host_module: dispatch arg {nm!r} is not a parameter of entry {entry.name!r}"
            )
        return nm

    visible = tuple(p for p in entry_params if p.name not in hidden)
    wrapper = CallableSignature(
        name=names.host_entry(entry.name),
        params=tuple(tensor_signature_of(p) for p in visible),
    )
    body_lines = []
    for p in visible:
        if not _is_user_scalar(p, hidden):
            body_lines.append(_placement_line(p.name, p.type.storage))

    subj = subject
    s = "__tf_dispatch_subject"
    body_lines.append(
        f"long long {s} = static_cast<long long>({_host_name(subj)}.shape()[{subj.axis}]);"
    )

    for variant, call in zip(variants, calls):
        if len(call.args) != len(variant.params):
            raise ValueError(
                f"emit_host_module: dispatch call to {variant.name!r} passes "
                f"{len(call.args)} args for {len(variant.params)} parameters"
            )
    shim_decls: dict[str, str] = {}
    for idx, (variant, call) in enumerate(zip(variants, calls)):
        pat = variant.specializations[0]
        variant_symbol = variant.name
        shim_decls[names.launch_shim(variant_symbol)] = _shim_decl(variant)
        if len(call.args) != len(variant.params):
            raise ValueError(
                f"emit_host_module: dispatch call to {variant.name!r} passes "
                f"{len(call.args)} args for {len(variant.params)} parameters"
            )
        v_hidden = _hidden_names(variant.params)
        shim_args = []
        for vp, arg in zip(variant.params, call.args):
            if _is_tensor(vp, v_hidden):
                shim_args.append(f"{_host_name(arg)}.data_ptr()")
            elif vp.name in v_hidden:
                if not isinstance(arg, ShapeOf):
                    raise NotImplementedError(
                        f"emit_host_module: hidden shape param {vp.name!r} expects a ShapeOf arg"
                    )
                shim_args.append(f"static_cast<long long>({_host_name(arg)}.shape()[{arg.axis}])")
            else:
                shim_args.append(f"static_cast<long long>({_host_name(arg)})")
        from tilefoundry.codegen.cuda.emit import (  # noqa: PLC0415
            _derive_launch_config,
        )

        grid, block = _derive_launch_config(variant.body)
        if grid[0] is None:
            raise ValueError(
                f"emit_host_module: dispatch variant {variant.name!r} has a "
                f"launch-provided (dynamic) CTA extent; the dispatch host path "
                f"requires a static grid"
            )
        shim_args += [str(d) for d in (*grid, *block, 0)]
        shim_args.append("nullptr")
        pred = f"(({pat.lo} <= {s}) && ({s} <= {pat.hi}))"
        prefix = "if" if idx == 0 else "} else if"
        body_lines.append(f"{prefix} ({pred}) {{")
        body_lines.append(f"  {names.launch_shim(variant_symbol)}({', '.join(shim_args)});")
    body_lines.append("} else {")
    body_lines.append(
        f'  throw std::runtime_error("tilefoundry: no matching dispatch variant for {entry.name}");'
    )
    body_lines.append("}")
    wrapper_sig = declare(
        wrapper.all_params,
        lambda p: "int" if _is_user_scalar(p, hidden) else "tvm::ffi::Tensor",
    )
    return list(shim_decls.values()), body_lines, wrapper_sig


def _reject_unsupported_config(cfg) -> None:
    if cfg.cluster is not None:
        raise NotImplementedError("emit_host_module: launch `cluster` is not supported yet")
    if cfg.stream is not None:
        raise NotImplementedError("emit_host_module: launch `stream` is not supported yet")
    if cfg.attrs.entries:
        raise NotImplementedError("emit_host_module: launch `attrs` are not supported yet")


__all__ = ["emit_host_module"]
