"""Host linkable-module emitter (split pipeline).

Emits the host ``.cpp`` module for a CPU-target entry. Two entry-body shapes
are supported:

- ``Sequential((Launch,))`` — a tvm-ffi entry that binds runtime tensors to a
  device kernel's parameters, checks device placement, and calls that kernel's
  launch shim by symbol.
- ``Sequential((DispatchCall, [Return]))`` — the dispatch entry lowers to a
  host-side first-match ``if/else`` over a shape predicate; each case calls the
  matching variant's launch shim, and the fallback throws.

The host module compiles with a plain host compiler — it includes only
tvm-ffi / DLPack / std headers and never references CUDA, cutlass, ``<<<>>>``,
``dim3`` or ``cudaStream_t``; those live exclusively in the device module and
its shims.
"""
from __future__ import annotations

from tilefoundry.codegen.cpu.templates import render
from tilefoundry.codegen.cuda.module import shim_symbol
from tilefoundry.codegen.cuda.tir.prim_function import (
    _internal_wrapper_symbol,
    _is_dispatch_entry_shape,
    _is_hidden_shape_scalar,
    _parse_shape_param_name,
)
from tilefoundry.codegen.linkable import LinkableFunction, LinkableModule
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.pattern import DimVarRangePat
from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.tir.dispatch import DispatchCall
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.shape import ShapeOf
from tilefoundry.ir.tir.stmts import Abort, Evaluate, Sequential
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
)

# Memory-space → required DLPack device type for a host ABI tensor argument.
_STORAGE_DEVICE_TYPE = {
    StorageKind.GMEM: "kDLCUDA",
    StorageKind.HOST: "kDLCPU",
}

# Launch-config ABI tail shared by every shim signature / forward declaration.
_LAUNCH_ABI_DECL = ["int", "int", "int", "int", "int", "int", "int", "void*"]


def _is_static_int(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, Constant) and isinstance(value.value, int) and not isinstance(
        value.value, bool
    )


def _static_value(value) -> int:
    return value if isinstance(value, int) else int(value.value)


def _static_smem(value) -> int:
    """Resolve ``dynamic_smem`` to a static int. Host index-expr codegen
    covers grid/block extents only; a dynamic shared-memory size expression
    is not supported."""
    if _is_static_int(value):
        return _static_value(value)
    raise ValueError(
        "emit_host_module: dynamic_smem must be a static int/Constant; a "
        f"dynamic shared-memory size expression is not supported, got "
        f"{type(value).__name__}"
    )


# Dim-arithmetic op -> infix C++ operator. Integer shape extents are
# non-negative, so C++ ``/`` matches floor division.
_DIM_BINOP_CXX = {
    DimAdd: "+",
    DimSub: "-",
    DimMul: "*",
    DimFloorDiv: "/",
    DimMod: "%",
}


def _emit_host_int_expr(expr) -> str:
    """Lower a launch-extent Expr to a host C++ integer expression.

    Accepts an ``int`` / integer ``Constant`` (emitted as a literal), a
    ``ShapeOf`` (the forwarded tensor's ``shape()`` access), or a
    dim-arithmetic ``Call`` over those. Any unsupported node raises — there
    is no silent zero/default."""
    if _is_static_int(expr):
        return str(_static_value(expr))
    if isinstance(expr, ShapeOf):
        return f"{expr.param.name}.shape()[{expr.axis}]"
    if isinstance(expr, Call):
        target = expr.target
        sym = next(
            (s for op, s in _DIM_BINOP_CXX.items() if isinstance(target, op)), None
        )
        if sym is not None:
            a, b = expr.args
            return (
                f"({_emit_host_int_expr(a)} {sym} {_emit_host_int_expr(b)})"
            )
        if isinstance(target, (DimMin, DimMax)):
            a, b = expr.args
            ca = _emit_host_int_expr(a)
            cb = _emit_host_int_expr(b)
            cmp = "<" if isinstance(target, DimMin) else ">"
            return f"(({ca}) {cmp} ({cb}) ? ({ca}) : ({cb}))"
        raise ValueError(
            f"emit_host_module: unsupported launch-extent op "
            f"{type(target).__name__}"
        )
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


def emit_host_module(entry: PrimFunction, module) -> LinkableModule:
    """Emit the host ``.cpp`` linkable module for a CPU *entry*.

    *module* is the enclosing ``Module``; the dispatch path resolves each
    case's ``SymbolRef`` callee through ``module.lookup`` to read the variant's
    parameters.
    """
    body = entry.body
    if (
        isinstance(body, Sequential)
        and len(body.body) == 1
        and isinstance(body.body[0], Evaluate)
        and isinstance(body.body[0].callable, Launch)
    ):
        shim_decls, body_lines, sig = _lower_launch(entry, body.body[0], module)
    elif _is_dispatch_entry_shape(entry):
        shim_decls, body_lines, sig = _lower_dispatch(entry, body.body[0], module)
    else:
        raise ValueError(
            f"emit_host_module: entry {entry.name!r} body must be a single "
            f"Launch or a dispatch entry (DispatchCall)"
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


def _lower_launch(entry: PrimFunction, evaluate, module):
    launch_op = evaluate.callable  # Launch Op
    device_fn = module.lookup(evaluate.args[0].name)
    dev_params = device_fn.params
    _reject_unsupported_config(launch_op)

    # The forwarded args (after the callee + six grid/block extents) bind the
    # host-visible device params (lowered params minus hidden shape scalars).
    # Hidden scalars are appended by lowering and filled host-side from a tensor
    # arg's shape — the user never passes them.
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

    # Visible device param -> bound entry param (positional). A hidden scalar
    # keeps its own name as a host local, declared below from the runtime shape.
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

    # Wrapper signature in entry.params order; every entry param must be used.
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

    # Grid / block extents are already canonical Exprs (a static ``Constant``,
    # a ``ShapeOf`` of a forwarded tensor for a launch-provided dim, or a
    # dim-arithmetic ``Call`` over those — built when the launch was
    # constructed). A ``ShapeOf`` lowers to the forwarded tensor's runtime
    # ``shape()`` access; the wrapper parameter carries that tensor's name.
    def _extent(c) -> str:
        if _is_static_int(c):
            return str(_static_value(c))
        return f"static_cast<int>({_emit_host_int_expr(c)})"

    grid = tuple(_extent(c) for c in grid_exprs)
    block = tuple(_extent(c) for c in block_exprs)
    dynamic_smem = _static_smem(launch_op.dynamic_smem)
    call_args = [_call_arg(i, p) for i, p in enumerate(dev_params)]
    call_args += [*grid, *block, str(dynamic_smem)]
    call_args.append("nullptr")  # stream
    body_lines.append(f"{shim_symbol(device_fn.name)}({', '.join(call_args)});")
    return [_shim_decl(device_fn)], body_lines, ", ".join(wrapper_tokens)


def _lower_dispatch(entry: PrimFunction, dispatch: DispatchCall, module):
    if len(dispatch.subjects) != 1 or not isinstance(dispatch.subjects[0], ShapeOf):
        raise NotImplementedError(
            "emit_host_module: dispatch v1 expects exactly one ShapeOf subject"
        )
    for pats in dispatch.case_patterns:
        if len(pats) != 1 or not isinstance(pats[0], DimVarRangePat):
            raise NotImplementedError(
                "emit_host_module: dispatch v1 expects exactly one "
                "DimVarRangePat per case"
            )
    fb = dispatch.fallback
    if not (
        isinstance(fb, Sequential)
        and len(fb.body) == 1
        and isinstance(fb.body[0], Abort)
    ):
        raise NotImplementedError(
            "emit_host_module: dispatch v1 expects fallback Sequential((Abort,))"
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

    # Host wrapper signature + placement, from the entry's user-facing params.
    wrapper_tokens, body_lines = [], []
    for p in entry_params:
        if p.name in hidden:
            continue
        if _is_user_scalar(p, hidden):
            wrapper_tokens.append(f"int {p.name}")
        else:
            wrapper_tokens.append(f"tvm::ffi::Tensor {p.name}")
            body_lines.append(_placement_line(p.name, p.type.storage))

    subj = dispatch.subjects[0]
    s = "__tf_dispatch_subject"
    body_lines.append(
        f"long long {s} = "
        f"static_cast<long long>({_host_name(subj)}.shape()[{subj.axis}]);"
    )

    _require_uniform_case_args(dispatch.case_calls, module)
    shim_decls: dict[str, str] = {}
    for idx, (pats, call) in enumerate(
        zip(dispatch.case_patterns, dispatch.case_calls)
    ):
        pat = pats[0]
        variant = module.lookup(call.callable.name)
        shim_decls[shim_symbol(variant.name)] = _shim_decl(variant)
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
        shim_args.append("nullptr")  # dynamic_smem handled above; stream
        pred = f"(({pat.lo} <= {s}) && ({s} < {pat.hi}))"  # half-open [lo, hi)
        prefix = "if" if idx == 0 else "} else if"
        body_lines.append(f"{prefix} ({pred}) {{")
        body_lines.append(
            f"  {shim_symbol(variant.name)}({', '.join(shim_args)});"
        )
    body_lines.append("} else {")
    body_lines.append(
        '  throw std::runtime_error("tilefoundry: no matching dispatch variant for '
        f'{entry.name}");'
    )
    body_lines.append("}")
    return list(shim_decls.values()), body_lines, ", ".join(wrapper_tokens)


def _arg_descriptor(arg):
    if isinstance(arg, Var):
        return ("var", arg.name)
    if isinstance(arg, ShapeOf):
        return ("shape", arg.param.name, arg.axis)
    return ("other", type(arg).__name__)


def _param_contract(vp, v_hidden):
    """Visible ABI contract of a variant parameter: kind + dtype + (for
    tensors) static shape structure + storage."""
    if _is_tensor(vp, v_hidden):
        t = vp.type
        return ("tensor", t.dtype, repr(t.shape), t.storage)
    if vp.name in v_hidden:
        return ("hidden", vp.type.dtype)
    return ("scalar", vp.type.dtype, vp.type.storage)


def _require_uniform_case_args(case_calls, module) -> None:
    """v1: every dispatch case must expose the same visible ABI — both the
    forwarded host arguments AND each variant parameter's tensor/scalar
    contract (kind, dtype, static shape, storage). The host entry does
    placement once against the entry params, so a branch whose variant has a
    different parameter contract would be a silent ABI/placement mismatch."""
    def _key(call):
        variant = module.lookup(call.callable.name)
        if len(call.args) != len(variant.params):
            raise ValueError(
                f"emit_host_module: dispatch call to {variant.name!r} passes "
                f"{len(call.args)} args for {len(variant.params)} parameters"
            )
        v_hidden = _hidden_names(variant.params)
        return tuple(
            (_param_contract(vp, v_hidden), _arg_descriptor(arg))
            for vp, arg in zip(variant.params, call.args)
        )

    keys = {module.lookup(c.callable.name).name: _key(c) for c in case_calls}
    distinct = set(keys.values())
    if len(distinct) > 1:
        raise ValueError(
            f"emit_host_module: dispatch variants {sorted(keys)} have differing "
            f"visible parameter/argument contracts; v1 requires every variant to "
            f"expose the same tensor ABI (kind / dtype / shape / storage)"
        )


def _reject_unsupported_config(cfg) -> None:
    if cfg.cluster is not None:
        raise ValueError("emit_host_module: launch `cluster` is not supported yet")
    if cfg.stream is not None:
        raise ValueError("emit_host_module: launch `stream` is not supported yet")
    if cfg.attrs.entries:
        raise ValueError("emit_host_module: launch `attrs` are not supported yet")


def _derive_launch_config(body):
    # noqa lazy: avoid an import cycle with codegen.cuda.emit at module load.
    from tilefoundry.codegen.cuda.emit import _derive_launch_config as _d  # noqa: PLC0415
    return _d(body)


__all__ = ["emit_host_module"]
