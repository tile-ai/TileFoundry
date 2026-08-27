"""HIR reference interpreter.

Walks a HIR ``Function`` body and returns concrete torch values.
"""
from __future__ import annotations

from typing import Any

import torch

from tilefoundry.evaluator.context import EvaluateContext
from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.registry import eval_registry
from tilefoundry.evaluator.value import (
    EvalError,
    TensorValue,
    TupleValue,
    Value,
    to_torch_dtype,
)
from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.core.pattern import locate_dim_var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.utils import types_compatible
from tilefoundry.ir.visitor import ExprVisitor


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _bind_dim_vars(params, values) -> dict[str, int]:
    """Bind dim vars.

    Map each ``DimVar`` appearing directly as a parameter-shape axis to the
    concrete size of the matching argument. Conflicting bindings for the same
    name raise ``EvalError``.
    """
    binding: dict[str, int] = {}
    for p, v in zip(params, values):
        shape = getattr(p.type, "shape", None)
        data = getattr(v, "data", None)
        if shape is None or data is None:
            continue
        for axis, dim in enumerate(shape):
            if isinstance(dim, DimVar) and axis < len(data.shape):
                size = int(data.shape[axis])
                prev = binding.get(dim.name)
                if prev is not None and prev != size:
                    raise EvalError(
                        f"evaluator: inconsistent binding for DimVar "
                        f"{dim.name!r}: {prev} vs {size}"
                    )
                binding[dim.name] = size
    return binding


def child_module_instance(loaded_module, callee: Function):
    """The child module instance a call to *callee* runs against, else ``None``.

    ``None`` is a same-owner call, which binds every declared parameter. A
    collected call carries no binding record, so which reading supplies the
    constants is answered by which child owns the callee.
    """
    if loaded_module is None or loaded_module.module.owns(callee, derived=True):
        return None
    matches = tuple(
        child for child in loaded_module.modules if child.module.owns(callee, derived=True)
    )
    if len(matches) > 1:
        raise EvalError(
            f"evaluator: {loaded_module.name!r} holds {len(matches)} child modules owning "
            f"{callee.name!r}; one call reaches one child"
        )
    return matches[0] if matches else None


class EvaluatorVisitor(ExprVisitor):
    """``ExprVisitor[Value]`` evaluated with one memo per execution scope."""

    def __init__(self, *, memo=None) -> None:
        super().__init__(memo=memo)

    def visit_leaf_Var(self, var: Var, _operands, ctx: EvaluateContext) -> Value:
        raise EvalError(f"evaluator: unbound variable {var.name!r}")

    def visit_leaf_Constant(
        self, const: Constant, _operands, ctx: EvaluateContext
    ) -> TensorValue:
        data = torch.as_tensor(
            const.value, dtype=to_torch_dtype(const.type.dtype), device=ctx.device
        )
        return TensorValue(data=data, type=const.type)

    def visit_leaf_Tuple(self, tup: Tuple, operands, ctx: EvaluateContext) -> TupleValue:
        return TupleValue(operands)

    def visit_leaf_Call(self, call: Call, args, ctx: EvaluateContext) -> Value:
        target = call.target
        if isinstance(target, Function):
            return self._call_function(target, args, ctx)
        handler = eval_registry.lookup(type(target))
        if handler is None:
            raise EvalError(
                f"evaluator: no @register_eval handler for "
                f"{type(target).__name__}"
            )
        return handler(ctx.for_op(target, args, call.type))

    def _call_function(
        self, callee: Function, arg_values, ctx: EvaluateContext
    ) -> Value:
        child = child_module_instance(ctx.loaded_module, callee)
        supplied = [p for p in callee.params if not (child is not None and p.is_const)]
        if len(arg_values) != len(supplied):
            kind = "activation(s)" if child is not None else "args"
            raise EvalError(
                f"evaluator: call to {callee.name!r} expects "
                f"{len(supplied)} {kind}, got {len(arg_values)}"
            )
        given = iter(arg_values)
        args = [
            _child_constant(child, callee, param)
            if child is not None and param.is_const
            else next(given)
            for param in callee.params
        ]
        target = _select_variant(callee, args) if callee.variants else callee
        for param, value in zip(target.params, args):
            if not types_compatible(param.annotation, value.type):
                raise EvalError(
                    f"evaluator: call to {target.name!r}: argument for "
                    f"{param.name!r} expects {param.annotation!r}, got {value.type!r}"
                )
        function_context = EvaluateContext(
            loaded_module=child if child is not None else ctx.loaded_module,
            device=ctx.device,
            dim_bindings=_bind_dim_vars(target.params, args),
        )
        memo = {id(param): (param, arg) for param, arg in zip(target.params, args)}
        return EvaluatorVisitor(memo=memo).visit(target.body, function_context)

    def _resolve_loop_field(self, dim, what: str, ctx: EvaluateContext) -> int:
        """Resolve loop field.

        Resolve a ``GridRegionExpr`` ``extent`` / ``step`` ``ShapeDim`` to a
        concrete ``int`` against the current DimVar bindings; fail closed.
        """
        if isinstance(dim, bool):
            raise EvalError(f"evaluator: GridRegion {what} must be an integer")
        if isinstance(dim, int):
            return dim
        try:
            return resolve_dim(dim, ctx.dim_bindings)
        except ValueError as exc:
            raise EvalError(f"evaluator: GridRegion {what}: {exc}") from None

    def visit_GridRegionExpr(
        self, region: GridRegionExpr, ctx: EvaluateContext
    ) -> Value:
        init_values = tuple(self.visit(init, ctx) for init in region.init_args)
        iv = region.induction_var
        iv_dtype = to_torch_dtype(iv.type.dtype)
        start = self._resolve_loop_field(region.start, "start", ctx)
        extent = self._resolve_loop_field(region.extent, "extent", ctx)
        step = self._resolve_loop_field(region.step, "step", ctx)
        if start < 0:
            raise EvalError(
                f"evaluator: GridRegion start must be non-negative, got {start}"
            )
        if extent < 0:
            raise EvalError(
                f"evaluator: GridRegion extent must be non-negative, got {extent}"
            )
        if step <= 0:
            raise EvalError(
                f"evaluator: GridRegion step must be positive, got {step}"
            )
        indices = range(start, extent, step)

        def iter_memo(i: int, carried) -> dict:
            memo = {
                **self._memo,
                id(iv): (
                    iv,
                    TensorValue(
                    data=torch.as_tensor(i, dtype=iv_dtype, device=ctx.device),
                    type=iv.type,
                ),
                ),
            }
            for phi, value in zip(region.carried_args, carried):
                memo[id(phi)] = (phi, value)
            return memo

        if not region.carried_args:

            last = None
            for i in indices:
                last = EvaluatorVisitor(memo=iter_memo(i, ())).visit(region.body, ctx)
            if last is None:
                raise EvalError(
                    "evaluator: GridRegionExpr has an empty iteration domain"
                )
            return last

        carried = list(init_values)
        for i in indices:
            sub = EvaluatorVisitor(memo=iter_memo(i, carried))
            carried = [sub.visit(y, ctx) for y in region.yield_values]
        return carried[0] if len(carried) == 1 else TupleValue(tuple(carried))


def _child_constant(loaded_module, callee: Function, param) -> TensorValue:
    """*param*'s constant, read from the child *callee* belongs to.

    Wrapped without a device argument: placement is settled before execution
    and this must not be where a weight quietly moves.
    """
    try:
        value = loaded_module.constants[param.name]
    except KeyError:
        raise EvalError(
            f"evaluator: {loaded_module.name!r} has no binding for {param.name!r} of "
            f"{callee.name!r}; a child call takes its ConstTensor parameters "
            f"from that child's own resources"
        ) from None
    data = torch.as_tensor(value, dtype=to_torch_dtype(param.type.dtype))
    return TensorValue(data=data, type=param.type)


def _unwrap(value: Value) -> Any:
    if isinstance(value, TensorValue):
        return value.data
    if isinstance(value, TupleValue):
        return tuple(_unwrap(v) for v in value.elements)
    return value


def _select_variant(callee: Function, arg_values) -> Function:
    """Pick the variant whose ``DimVarRangePat`` matches the runtime arg shapes.

    Errors unless exactly one matches — dispatch never falls back to the
    prototype body.
    """
    matches = []
    for v in callee.variants:
        pat = v.specializations[0]
        loc = locate_dim_var(callee.params, pat.dim_var)
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


def _bound_values(fn: Function, args) -> list[TensorValue]:
    """*args* as evaluator values, left exactly where they already live."""
    return [
        TensorValue(
            data=torch.as_tensor(arg, dtype=to_torch_dtype(param.type.dtype)),
            type=param.type,
        )
        for param, arg in zip(fn.params, args)
    ]


def _dim_bindings(fn: Function, args) -> dict[str, int]:
    """The extents these argument values give *fn*'s ``DimVar`` axes."""
    return _bind_dim_vars(fn.params, _bound_values(fn, args))


def _selected_body(fn: Function, args) -> Function:
    """The Function a call on *fn* with these argument values will run."""
    if not fn.variants:
        return fn
    return _select_variant(fn, _bound_values(fn, args))


def _run_bound(fn: Function, args, *, device: str | None = None, reading=None):
    """Evaluate *fn* over fully bound *args*, with *reading* in hand.

    The entry a resource reading runs through: every child call reached from
    here fills its ``ConstTensor`` parameters from the child that owns the
    callee. It is internal; the public ``evaluate`` stays exact and
    resource-free.
    """
    device = device or _default_device()
    values = _bound_values(fn, args)
    target = _select_variant(fn, values) if fn.variants else fn
    memo = {id(param): (param, value) for param, value in zip(target.params, values)}
    dim_env = _bind_dim_vars(target.params, values)
    return _unwrap(
        EvaluatorVisitor(memo=memo).visit(
            target.body,
            EvaluateContext(
                loaded_module=reading,
                device=device,
                dim_bindings=dim_env,
            ),
        )
    )


def evaluate(fn_or_call, *inputs, backend: str = "torch", device: str | None = None):
    """Evaluate a HIR ``Function`` (or ``Call``) and return torch value(s).

    ``inputs`` bind positionally to a ``Function``'s parameters; the result is
    a ``torch.Tensor`` for a single output or a tuple for a ``TupleType``.
    """
    if backend != "torch":
        raise EvalError(f"evaluator: unsupported backend {backend!r}")
    device = device or _default_device()

    if isinstance(fn_or_call, Function):
        fn = fn_or_call
        if len(inputs) != len(fn.params):
            raise EvalError(
                f"evaluator: {fn.name!r} expects {len(fn.params)} inputs, "
                f"got {len(inputs)}"
            )
        values = [
            TensorValue(
                data=torch.as_tensor(
                    arg, dtype=to_torch_dtype(param.type.dtype), device=device
                ),
                type=param.type,
            )
            for param, arg in zip(fn.params, inputs)
        ]


        target = _select_variant(fn, values) if fn.variants else fn
        memo = {id(param): (param, value) for param, value in zip(target.params, values)}
        dim_env = _bind_dim_vars(target.params, values)
        result = EvaluatorVisitor(memo=memo).visit(
            target.body,
            EvaluateContext(device=device, dim_bindings=dim_env),
        )
    elif isinstance(fn_or_call, Call):
        if inputs:
            raise EvalError("evaluator: a Call entry takes no positional inputs")
        result = EvaluatorVisitor(memo={}).visit(
            fn_or_call, EvaluateContext(device=device)
        )
    else:
        raise EvalError(
            f"evaluator: expected a Function or Call, got {type(fn_or_call).__name__}"
        )
    return _unwrap(result)
