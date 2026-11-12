"""HIR reference interpreter.

Walks a HIR ``Function`` body and returns concrete torch values.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tilefoundry import core, services, utils
from tilefoundry.ir import core as ir_core
from tilefoundry.ir import expr, pattern
from tilefoundry.ir import utils as ir_utils

from .context import EvaluateContext
from .dim import resolve_dim
from .registry import registry
from .value import EvalError, TensorValue, TupleValue, Value, tensor_type_of, to_torch_dtype

MISSING_PREPARED_WEIGHT = "[runtime §1.3](docs/spec/runtime.md#13-runtimedecoratorpy)"


def device_of(values) -> str | None:
    """The one device *values* already live on, or ``None`` when none is a tensor.

    The evaluator picks no device of its own: it computes where its inputs are.
    ``None`` leaves the choice to torch, which is what a run with no tensor input
    has to fall back on.
    """
    seen: dict[str, int] = {}
    for index, value in enumerate(values):
        held = (
            (value.tensor,) if value.is_global else tuple(piece for _, piece in value.tensor)
        ) if isinstance(value, core.ShardTensor) else (value,)
        for data in held:
            device = getattr(data, "device", None)
            if device is not None:
                seen.setdefault(str(device), index)
    if len(seen) > 1:
        spread = ", ".join(f"input {index} on {device}" for device, index in sorted(seen.items()))
        raise EvalError(
            f"evaluator: inputs are on more than one device -- {spread}. "
            "Build them on one device; evaluation moves nothing."
        )
    return next(iter(seen), None)


def devices_match(actual, expected: utils.torch.device) -> bool:
    actual = utils.torch.device(actual)
    if actual.type != expected.type:
        return False
    return expected.index is None or actual.index == expected.index


def bind_dim_vars(params, values) -> dict[str, int]:
    """Bind dim vars.

    Map each ``DimVar`` appearing directly as a parameter-shape axis to the
    concrete size of the matching argument. Conflicting bindings for the same
    name raise ``EvalError``.
    """
    binding: dict[str, int] = {}
    for p, v in zip(params, values):
        shape = getattr(p.type, "shape", None)
        if shape is None or not isinstance(v, TensorValue) or not v.tensor.is_global:
            continue
        data = v.data
        for axis, dim in enumerate(shape):
            if isinstance(dim, expr.DimVar) and axis < len(data.shape):
                size = int(data.shape[axis])
                prev = binding.get(dim.name)
                if prev is not None and prev != size:
                    raise EvalError(
                        f"evaluator: inconsistent binding for DimVar {dim.name!r}: {prev} vs {size}"
                    )
                binding[dim.name] = size
    return binding


class Visitor(ir_core.ExprVisitor):
    """``ExprVisitor[Value]`` evaluated with one memo per execution scope."""

    def __init__(self, *, memo=None) -> None:
        super().__init__(memo=memo)

    def visit_MeshRegion(self, region: ir_core.MeshRegion, ctx: EvaluateContext) -> Value:
        """Evaluate boundary arguments outside, then the isolated body."""
        args = tuple(self.visit(arg, ctx) for arg in region.args)
        if len(args) != len(region.params):
            raise EvalError(
                f"evaluator: MeshRegion expects {len(region.params)} args, got {len(args)}"
            )
        memo = {id(param): (param, value) for param, value in zip(region.params, args)}
        return Visitor(memo=memo).visit(region.body, ctx)

    def visit_leaf_Var(self, var: expr.Var, operands, ctx: EvaluateContext) -> Value:
        raise EvalError(f"evaluator: unbound variable {var.name!r}")

    def visit_leaf_Constant(self, const: expr.Constant, operands, ctx: EvaluateContext) -> TensorValue:
        data = utils.torch.as_tensor(
            const.value, dtype=to_torch_dtype(const.type.dtype), device=ctx.device
        )
        return TensorValue(core.ShardTensor(const.type, data))

    def visit_leaf_Tuple(self, tup: expr.Tuple, operands, ctx: EvaluateContext) -> TupleValue:
        return TupleValue(operands)

    def visit_leaf_Call(self, call: expr.Call, args, ctx: EvaluateContext) -> Value:
        target = call.target
        if isinstance(target, ir_core.FunctionRef):
            reading, module = ctx.loaded_module, ctx.module
            if target.path and reading is None:
                raise EvalError("evaluator: instance call requires a loaded module")
            if module is None:
                raise EvalError(
                    f"evaluator: call to {target.qualified_name!r} requires the Module that "
                    "declares the caller; pass context=EvaluateContext(module=...)"
                )
            indices = iter(target.indices(args))
            key = None
            for segment in target.path:
                if isinstance(segment, str):
                    key, reading = segment, reading.children[segment]
                    continue
                taken = segment if segment is not None else int(next(indices).data.item())
                if not 0 <= taken < len(reading):
                    raise EvalError(
                        f"evaluator: ModuleList slot {key!r}: index {taken!r} out of range for N={len(reading)}"
                    )
                reading = reading[taken]
            if target.path:
                module = reading.module
            function = module.entry_function() if target.name is None else module.lookup(target.name)
            inner = replace(ctx, module=module, loaded_module=reading)
            return call_function(function, target.arguments(args), inner, omit_weights=target.name is None)
        try:
            handler = registry.get(type(target), None)
            if handler is None:
                raise EvalError(f"no @register_eval handler for {type(target).__name__}")
            return handler(ctx.for_op(target, args, call.type))
        except Exception as error:
            raise EvalError(f"evaluator: {expr.describe_value(call)}: {error}") from error

    def visit_LoopRegion(self, region: ir_core.LoopRegion, ctx: EvaluateContext) -> Value:
        init_values = tuple(self.visit(init, ctx) for init in region.args)
        k = len(region.yield_values)
        captured = {
            id(param): (param, value)
            for param, value in zip(region.params[k:], init_values[k:], strict=True)
        }
        iv = region.induction_var
        iv_dtype = to_torch_dtype(iv.type.dtype)
        start = resolve_loop_field(region.start, "start", ctx)
        extent = resolve_loop_field(region.extent, "extent", ctx)
        step = resolve_loop_field(region.step, "step", ctx)
        if start < 0:
            raise EvalError(f"evaluator: LoopRegion start must be non-negative, got {start}")
        if extent < 0:
            raise EvalError(f"evaluator: LoopRegion extent must be non-negative, got {extent}")
        if step <= 0:
            raise EvalError(f"evaluator: LoopRegion step must be positive, got {step}")
        indices = range(start, extent, step)

        def iter_memo(i: int, carried) -> dict:
            memo = {
                **captured,
                id(iv): (
                    iv,
                    TensorValue(
                        core.ShardTensor(
                            iv.type,
                            utils.torch.as_tensor(i, dtype=iv_dtype, device=ctx.device),
                        )
                    ),
                ),
            }
            for phi, value in zip(region.params[:k], carried, strict=True):
                memo[id(phi)] = (phi, value)
            return memo

        if not k:
            last = None
            for i in indices:
                last = Visitor(memo=iter_memo(i, ())).visit(region.body, ctx)
            if last is None:
                raise EvalError("evaluator: LoopRegion has an empty iteration domain")
            return last

        carried = list(init_values[:k])
        for i in indices:
            sub = Visitor(memo=iter_memo(i, carried))
            carried = [sub.visit(y, ctx) for y in region.yield_values]
        return carried[0] if len(carried) == 1 else TupleValue(tuple(carried))


def call_function(
    callee: ir_core.Function, arg_values, ctx: EvaluateContext, *, omit_weights: bool
) -> Value:
    """Run *callee* in the Module and reading of *ctx*.

    An instance entry call omits its ``ConstTensor`` parameters, which come
    from the reading; any other call passes all of them.
    """
    supplied = [param for param in callee.params if not (omit_weights and param.is_const)]
    if len(arg_values) != len(supplied):
        kind = "activation(s)" if omit_weights else "args"
        raise EvalError(
            f"evaluator: call to {callee.name!r} expects "
            f"{len(supplied)} {kind}, got {len(arg_values)}"
        )
    given = iter(arg_values)
    args = [
        TensorValue(
            core.ShardTensor(
                param.type,
                read_weight(ctx.loaded_module, param, ctx.device, callee=callee),
            )
        )
        if omit_weights and param.is_const
        else next(given)
        for param in callee.params
    ]
    target = select_variant(callee, args) if callee.variants else callee
    for param, value in zip(target.params, args):
        if not ir_utils.types_compatible(param.annotation, value.type):
            raise EvalError(
                f"evaluator: call to {target.name!r}: argument for "
                f"{param.name!r} expects {param.annotation!r}, got {value.type!r}"
            )
    function_context = EvaluateContext(
        module=ctx.module,
        loaded_module=ctx.loaded_module,
        device=ctx.device,
        dim_bindings=bind_dim_vars(target.params, args),
    )
    memo = {id(param): (param, arg) for param, arg in zip(target.params, args)}
    return Visitor(memo=memo).visit(target.body, function_context)


def resolve_loop_field(dim, what: str, ctx: EvaluateContext) -> int:
    """Resolve loop field.

    Resolve a ``LoopRegion`` ``extent`` / ``step`` ``ShapeDim`` to a
    concrete ``int`` against the current DimVar bindings; fail closed.
    """
    if isinstance(dim, bool):
        raise EvalError(f"evaluator: LoopRegion {what} must be an integer")
    if isinstance(dim, int):
        return dim
    try:
        return resolve_dim(dim, ctx.dim_bindings)
    except ValueError as exc:
        raise EvalError(f"evaluator: LoopRegion {what}: {exc}") from None


def read_weight(loaded_module, param, device: str, *, callee: ir_core.Function | None = None):
    """Read, validate, and place-check one declared weight at first use."""
    try:
        value = loaded_module.resource.load(param.name)
    except KeyError as error:
        detail = ""
        if callee is not None:
            detail = (
                f" of {callee.name!r}; a child call takes its ConstTensor "
                "parameters from that child's own resources"
            )
        raise KeyError(
            f"Module {loaded_module.name!r}: missing declared weight {param.name!r}; "
            f"full key {f'{loaded_module.path}.{param.name}' if loaded_module.path else param.name!r}; "
            f"prepare produces it ({utils.spec_ref_render(MISSING_PREPARED_WEIGHT)}){detail}"
        ) from error
    actual = tensor_type_of(value, like=param.type)
    if not ir_utils.types_compatible(param.type, actual):
        if actual.shape == param.type.shape and actual.dtype != param.type.dtype:
            raise ValueError(
                f"Module {loaded_module.name!r}: weight {param.name!r} has dtype "
                f"{value.dtype}, declared "
                f"{param.type.dtype}; the way out is a weight converter on the model, "
                "not a flag on the read side"
            )
        raise ValueError(
            f"Module {loaded_module.name!r}: weight {param.name!r} has shape "
            f"{actual.shape}, declared {param.type.shape}"
        )
    if device is None:
        return value
    expected = utils.torch.device(device)
    if not devices_match(value.device, expected):
        raise ValueError(
            f"Module {loaded_module.name!r}: weight {param.name!r} is on {value.device}, "
            "but evaluation requested "
            f"{expected}; this runner moves nothing"
        )
    return value


def run_selected(fn: ir_core.Function, *activations, context: EvaluateContext):
    """Run one loaded function, reading constants lazily at first use.

    The context's device is where the activations already are; a weight
    somewhere else is refused at its first use rather than moved.
    """
    activation_params = tuple(param for param in fn.params if not param.is_const)
    if len(activations) != len(activation_params):
        raise EvalError(
            f"evaluator: call to {fn.name!r} expects {len(activation_params)} "
            f"activation(s), got {len(activations)}"
        )
    supplied = iter(activations)
    args = [
        read_weight(context.loaded_module, param, context.device) if param.is_const else next(supplied)
        for param in fn.params
    ]
    return run_bound(fn, args, context=context)


def unwrap(value: Value) -> Any:
    if isinstance(value, TensorValue):
        return value.data if value.tensor.is_global else value.tensor
    if isinstance(value, TupleValue):
        return tuple(unwrap(v) for v in value.elements)
    return value


def select_variant(callee: ir_core.Function, arg_values) -> ir_core.Function:
    """Pick the variant whose ``pattern.Range`` matches the runtime arg shapes.

    Errors unless exactly one matches — dispatch never falls back to the
    prototype body.
    """
    matches = []
    for v in callee.variants:
        pat = v.specializations[0]
        loc = pattern.locate_dim_var(callee.params, pat.dim_var)
        if loc is None:
            continue
        pi, axis = loc
        data = getattr(arg_values[pi], "data", None)
        if data is None or axis >= len(data.shape):
            continue
        if pat.match(int(data.shape[axis])):
            matches.append(v)
    if len(matches) != 1:
        raise EvalError(
            f"evaluator: dispatch of {callee.name!r}: runtime shapes matched "
            f"{len(matches)} variants (expected exactly one)"
        )
    return matches[0]


def bound_values(fn: ir_core.Function, args) -> list[TensorValue]:
    """*args* as evaluator values, left exactly where they already live."""
    return [bound_value(param, arg) for param, arg in zip(fn.params, args)]


def bound_value(param, arg) -> TensorValue:
    """A torch input is global logical data; a ``ShardTensor`` is taken as it is."""
    if isinstance(arg, core.ShardTensor):
        if not ir_utils.types_compatible(param.type, arg.type):
            raise EvalError(
                f"evaluator: {param.name!r} expects {param.type!r}, got {arg.type!r}"
            )
        return TensorValue(arg)
    data = utils.torch.as_tensor(arg, dtype=to_torch_dtype(param.type.dtype))
    return TensorValue(core.ShardTensor(param.type, data))


def run_bound(fn: ir_core.Function, args, *, context: EvaluateContext):
    """Evaluate *fn* over fully bound *args* in the Module and reading of *context*.

    The entry a resource reading runs through: every child call reached from
    here fills its ``ConstTensor`` parameters from the instance named by the
    call target. It is internal; the context's device is the caller's, and
    ``None`` leaves it to torch.
    """
    values = bound_values(fn, args)
    target = select_variant(fn, values) if fn.variants else fn
    memo = {id(param): (param, value) for param, value in zip(target.params, values)}
    dim_env = bind_dim_vars(target.params, values)
    return unwrap(Visitor(memo=memo).visit(target.body, replace(context, dim_bindings=dim_env)))


def evaluate(target, *inputs, context: EvaluateContext | None = None):
    """Evaluate a HIR ``Function``, a Module or a loaded module and return torch value(s).

    *target* answers which function to run and which reading, if any, supplies
    its ``ConstTensor`` parameters. A Module or a loading also gives the Module
    that resolves the calls of that function; for a ``Function``, *context*
    gives it. The run happens where *inputs* already are: the evaluator selects
    no device and no tensor engine of its own.
    """
    fn, reading = target.evaluation_target()
    context = EvaluateContext() if context is None else context
    if reading is not None:
        context = replace(context, module=reading.module, loaded_module=reading)
    elif isinstance(target, ir_core.Module):
        context = replace(context, module=target)
    context = replace(context, device=device_of(inputs))
    if reading is not None:
        return run_selected(fn, *inputs, context=context)

    if len(inputs) != len(fn.params):
        raise EvalError(
            f"evaluator: {fn.name!r} expects {len(fn.params)} inputs, got {len(inputs)}"
        )
    return run_bound(fn, inputs, context=context)


__all__ = ["Visitor", "evaluate", "run_bound"]

services.registry.register(services.evaluate, evaluate)
