"""Emit the host module for the split code-generation pipeline.

Launch entries validate and bind runtime tensors before calling a device shim.
Dispatch entries use a first-match shape predicate and throw on fallback. Output
uses only TVM FFI, DLPack, and standard C++; CUDA syntax and types remain in the
device module and shims.
"""
from __future__ import annotations

from tilefoundry.codegen.cpu.templates import render
from tilefoundry.codegen.cuda.module import shim_symbol
from tilefoundry.codegen.cuda.tir.prim_function import (
    _internal_wrapper_symbol,
    _is_hidden_shape_scalar,
    _parse_shape_param_name,
)
from tilefoundry.codegen.linkable import LinkableFunction, LinkableModule
from tilefoundry.codegen.registry import CodeGenerator
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.pattern import DimVarRangePat, locate_dim_var
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
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


_LAUNCH_ABI_DECL = ["int", "int", "int", "int", "int", "int", "int", "void*"]


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
                "emit_host_module: unsupported launch-extent node "
                f"{type(expr).__name__}"
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
        raise ValueError(
            f"emit_host_module: unsupported launch-extent op "
            f"{type(target).__name__}"
        )

    def default_visit(self, expr, ctx=None) -> str:
        raise ValueError(
            f"emit_host_module: unsupported launch-extent node "
            f"{type(expr).__name__}"
        )


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
    raise ValueError(
        f"emit_host_module: unsupported launch-extent node "
        f"{type(expr).__name__}"
    )


def _hidden_names(params) -> set:
    return {p.name for p in params if _is_hidden_shape_scalar(p, params)}


def _is_user_scalar(p, hidden: set) -> bool:
    return (
        p.name not in hidden
        and isinstance(p.type, TensorType)
        and not p.type.shape
    )


def _is_tensor(p, hidden: set) -> bool:
    return p.name not in hidden and not _is_user_scalar(p, hidden)


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
    tokens = ["void*" if _is_tensor(p, hidden) else "long long" for p in fn.params]
    tokens += _LAUNCH_ABI_DECL
    return f'extern "C" void {shim_symbol(fn.name)}({", ".join(tokens)});'


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
        and len(body.body) == 1
        and isinstance(body.body[0], Evaluate)
        and isinstance(body.body[0].callable, Launch)
    ):
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
            entry, module, callee_name=entry.name, subject=subject,
            variants=entry.variants, calls=calls)
    else:
        raise ValueError(
            f"emit_host_module: entry {entry.name!r} body must be a single "
            f"Launch"
        )
    source = render(
        "cpu_module.cpp.j2",
        shim_decls=shim_decls,
        internal_host_symbol=_internal_wrapper_symbol(entry.name),
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
    launch_op = evaluate.callable
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
        return _lower_dispatch(entry, module, callee_name=device_fn.name, subject=subject, variants=device_fn.variants, calls=calls)
    dev_params = device_fn.params
    _reject_unsupported_config(launch_op)





    grid_exprs = evaluate.args[1:4]
    block_exprs = evaluate.args[4:7]
    hidden = _hidden_names(dev_params)
    visible_dev = [p for p in dev_params if p.name not in hidden]

    args = evaluate.args[7:]
    if len(args) != len(visible_dev):
        raise ValueError(
            f"emit_host_module: launch passes {len(args)} args but device "
            f"function {device_fn.name!r} has {len(visible_dev)} host-visible "
            f"parameters (hidden shape scalars are derived from tensor shapes)"
        )
    if not all(isinstance(a, Var) for a in args):
        raise ValueError(
            "emit_host_module: launch args must be host entry parameters (Var)"
        )
    entry_by_id = {id(p): p for p in entry.params}
    entry_by_name = {p.name: p for p in entry.params}

    def _resolve(a: Var) -> Var:
        if id(a) in entry_by_id:
            return entry_by_id[id(a)]
        ep = entry_by_name.get(a.name)
        if ep is None:
            raise ValueError(
                f"emit_host_module: launch arg {a.name!r} is not a parameter "
                f"of entry {entry.name!r}"
            )
        return ep



    bound = [_resolve(a) for a in args]
    host_name_of: dict[str, str] = {}
    dev_index_of_entry: dict[int, int] = {}
    for k, (vp, ep) in enumerate(zip(visible_dev, bound)):
        host_name_of[vp.name] = ep.name
        if id(ep) in dev_index_of_entry:
            raise ValueError(
                f"emit_host_module: entry parameter {ep.name!r} is launched "
                f"more than once"
            )
        dev_index_of_entry[id(ep)] = k
    for p in dev_params:
        if p.name in hidden:
            host_name_of[p.name] = p.name
    host_names = [host_name_of[p.name] for p in dev_params]
    dev_to_host = host_name_of


    wrapper_tokens = []
    for ep in entry.params:
        k = dev_index_of_entry.get(id(ep))
        if k is None:
            raise ValueError(
                f"emit_host_module: entry parameter {ep.name!r} is not used "
                f"by the launch"
            )
        vp = visible_dev[k]
        wrapper_tokens.append(
            f"int {ep.name}" if _is_user_scalar(vp, hidden)
            else f"tvm::ffi::Tensor {ep.name}"
        )

    body_lines = []
    for i, p in enumerate(dev_params):
        if _is_tensor(p, hidden):
            body_lines.append(_placement_line(host_names[i], p.type.storage))
    for i, p in enumerate(dev_params):
        if p.name not in hidden:
            continue
        base, axis = _parse_shape_param_name(p.name)
        host_base = dev_to_host.get(base)
        if host_base is None:
            raise ValueError(
                f"emit_host_module: hidden shape scalar {p.name!r} references "
                f"unknown base parameter {base!r}"
            )
        body_lines.append(
            f"long long {host_names[i]} = "
            f"static_cast<long long>({host_base}.shape()[{axis}]);"
        )

    def _call_arg(i, p) -> str:
        hn = host_names[i]
        if _is_tensor(p, hidden):
            return f"{hn}.data_ptr()"
        if p.name in hidden:
            return hn
        return f"static_cast<long long>({hn})"






    def _extent(c) -> str:
        cv = static_dim_value(c)
        if cv is not None:
            return str(cv)
        return f"static_cast<int>({_emit_host_int_expr(c)})"

    grid = tuple(_extent(c) for c in grid_exprs)
    block = tuple(_extent(c) for c in block_exprs)
    dynamic_smem = _static_smem(launch_op.dynamic_smem)
    call_args = [_call_arg(i, p) for i, p in enumerate(dev_params)]
    call_args += [*grid, *block, str(dynamic_smem)]
    call_args.append("nullptr")
    body_lines.append(f"{shim_symbol(device_fn.name)}({', '.join(call_args)});")
    return [_shim_decl(device_fn)], body_lines, ", ".join(wrapper_tokens)


def _lower_dispatch(entry: PrimFunction, module, *, callee_name, subject, variants, calls):
    if not isinstance(subject, ShapeOf):
        raise NotImplementedError(
            "emit_host_module: dispatch v1 expects exactly one ShapeOf subject"
        )
    for variant in variants:
        if len(variant.specializations) != 1 or not isinstance(variant.specializations[0], DimVarRangePat):
            raise NotImplementedError(
                "emit_host_module: dispatch v1 expects exactly one "
                "DimVarRangePat per case"
            )

    entry_params = entry.params
    entry_names = {p.name for p in entry_params}
    hidden = _hidden_names(entry_params)

    def _host_name(ref) -> str:
        nm = ref.name if isinstance(ref, Var) else ref.param.name
        if nm not in entry_names:
            raise ValueError(
                f"emit_host_module: dispatch arg {nm!r} is not a parameter of "
                f"entry {entry.name!r}"
            )
        return nm


    wrapper_tokens, body_lines = [], []
    for p in entry_params:
        if p.name in hidden:
            continue
        if _is_user_scalar(p, hidden):
            wrapper_tokens.append(f"int {p.name}")
        else:
            wrapper_tokens.append(f"tvm::ffi::Tensor {p.name}")
            body_lines.append(_placement_line(p.name, p.type.storage))

    subj = subject
    s = "__tf_dispatch_subject"
    body_lines.append(
        f"long long {s} = "
        f"static_cast<long long>({_host_name(subj)}.shape()[{subj.axis}]);"
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
        shim_decls[shim_symbol(variant_symbol)] = _shim_decl(variant)
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
                        f"emit_host_module: hidden shape param {vp.name!r} "
                        f"expects a ShapeOf arg"
                    )
                shim_args.append(
                    f"static_cast<long long>({_host_name(arg)}.shape()[{arg.axis}])"
                )
            else:
                shim_args.append(f"static_cast<long long>({_host_name(arg)})")
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
        body_lines.append(
            f"  {shim_symbol(variant_symbol)}({', '.join(shim_args)});"
        )
    body_lines.append("} else {")
    body_lines.append(
        '  throw std::runtime_error("tilefoundry: no matching dispatch variant for '
        f'{entry.name}");'
    )
    body_lines.append("}")
    return list(shim_decls.values()), body_lines, ", ".join(wrapper_tokens)

def _reject_unsupported_config(cfg) -> None:
    if cfg.cluster is not None:
        raise NotImplementedError("emit_host_module: launch `cluster` is not supported yet")
    if cfg.stream is not None:
        raise NotImplementedError("emit_host_module: launch `stream` is not supported yet")
    if cfg.attrs.entries:
        raise NotImplementedError("emit_host_module: launch `attrs` are not supported yet")


def _derive_launch_config(body):
    # noqa lazy: avoid an import cycle with codegen.cuda.emit at module load.
    from tilefoundry.codegen.cuda.emit import _derive_launch_config as _d  # noqa: PLC0415
    return _d(body)


__all__ = ["emit_host_module"]
