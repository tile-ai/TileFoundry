"""HIR reference interpreter.

Walks a HIR ``Function`` body and returns concrete torch values.
"""
from __future__ import annotations

from typing import Any

import torch

from tilefoundry.evaluator.context import EvalContext
from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.registry import eval_registry
from tilefoundry.evaluator.value import (
    EvalError,
    TensorValue,
    TupleValue,
    Value,
    to_torch_dtype,
)
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.tensor.tuple import Tuple
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.visitor import ExprVisitor


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _bind_dim_vars(params, values) -> dict[str, int]:
    """Map each ``DimVar`` appearing directly as a parameter-shape axis to the
    concrete size of the matching argument. Conflicting bindings for the same
    name raise ``EvalError``."""
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


class Evaluator(ExprVisitor):
    """``ExprVisitor[Value]`` memoized on ``id(expr)`` within one scope."""

    def __init__(
        self, env: dict[Var, Value], device: str,
        dim_env: dict[str, int] | None = None,
    ) -> None:
        self.env = env
        self.device = device
        self.dim_env = dim_env or {}
        self.memo: dict[int, Value] = {}

    def visit(self, expr) -> Value:
        key = id(expr)
        if key in self.memo:
            return self.memo[key]
        value = super().visit(expr)
        self.memo[key] = value
        return value

    def visit_Var(self, var: Var) -> Value:
        try:
            return self.env[var]
        except KeyError:
            raise EvalError(f"evaluator: unbound variable {var.name!r}") from None

    def visit_Constant(self, const: Constant) -> TensorValue:
        data = torch.as_tensor(
            const.value, dtype=to_torch_dtype(const.type.dtype), device=self.device
        )
        return TensorValue(data=data, type=const.type)

    def visit_Tuple(self, tup: Tuple) -> TupleValue:
        return TupleValue(tuple(self.visit(e) for e in tup.elements))

    def visit_Call(self, call: Call) -> Value:
        target = call.target
        if isinstance(target, Function):
            return self._call_function(target, call.args)
        args = tuple(self.visit(a) for a in call.args)
        handler = eval_registry.lookup(type(target))
        if handler is None:
            raise EvalError(
                f"evaluator: no @register_eval handler for "
                f"{type(target).__name__}"
            )
        return handler(
            EvalContext(
                op=target, args=args, result_type=call.type, device=self.device
            )
        )

    def _call_function(self, callee: Function, arg_exprs) -> Value:
        if len(arg_exprs) != len(callee.params):
            raise EvalError(
                f"evaluator: call to {callee.name!r} expects "
                f"{len(callee.params)} args, got {len(arg_exprs)}"
            )
        args = [self.visit(a) for a in arg_exprs]
        # A dispatch prototype (body is None) selects a variant by the runtime
        # argument shapes; its own None body is never evaluated.
        target = _select_variant(callee, args) if callee.variants else callee
        sub_env = {param: arg for param, arg in zip(target.params, args)}
        sub_dim_env = _bind_dim_vars(target.params, args)
        return Evaluator(sub_env, self.device, sub_dim_env).visit(target.body)

    def _resolve_loop_field(self, dim, what: str) -> int:
        """Resolve a ``GridRegionExpr`` ``extent`` / ``step`` ``ShapeDim`` to a
        concrete ``int`` against the current DimVar bindings; fail closed."""
        if isinstance(dim, bool):
            raise EvalError(f"evaluator: GridRegion {what} must be an integer")
        if isinstance(dim, int):
            return dim
        try:
            return resolve_dim(dim, self.dim_env)
        except ValueError as exc:
            raise EvalError(f"evaluator: GridRegion {what}: {exc}") from None

    def visit_GridRegionExpr(self, region: GridRegionExpr) -> Value:
        iv = region.induction_var
        iv_dtype = to_torch_dtype(iv.type.dtype)
        start = self._resolve_loop_field(region.start, "start")
        extent = self._resolve_loop_field(region.extent, "extent")
        step = self._resolve_loop_field(region.step, "step")
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

        def iter_env(i: int, carried) -> dict:
            env = {
                **self.env,
                iv: TensorValue(
                    data=torch.as_tensor(i, dtype=iv_dtype, device=self.device),
                    type=iv.type,
                ),
            }
            for phi, value in zip(region.carried_args, carried):
                env[phi] = value
            return env

        if not region.carried_args:
            # No-carry loop: the value is the final body evaluation.
            last = None
            for i in indices:
                last = Evaluator(
                    iter_env(i, ()), self.device, self.dim_env
                ).visit(region.body)
            if last is None:
                raise EvalError(
                    "evaluator: GridRegionExpr has an empty iteration domain"
                )
            return last

        carried = [self.visit(init) for init in region.init_args]
        for i in indices:
            sub = Evaluator(iter_env(i, carried), self.device, self.dim_env)
            carried = [sub.visit(y) for y in region.yield_values]
        return carried[0] if len(carried) == 1 else TupleValue(tuple(carried))


def _unwrap(value: Value) -> Any:
    if isinstance(value, TensorValue):
        return value.data
    if isinstance(value, TupleValue):
        return tuple(_unwrap(v) for v in value.elements)
    return value


def _locate_dispatch_dim(params, dim_var_name: str):
    """First ``(param_index, axis)`` where the named DimVar appears in params."""
    for i, p in enumerate(params):
        shape = getattr(p.type, "shape", None)
        if shape is None:
            continue
        for axis, dim in enumerate(shape):
            if getattr(dim, "name", None) == dim_var_name:
                return i, axis
    return None


def _select_variant(callee: Function, arg_values) -> Function:
    """Pick the variant whose ``DimVarRangePat`` matches the runtime arg shapes.

    Errors unless exactly one matches — dispatch never falls back to the
    prototype body.
    """
    matches = []
    for v in callee.variants:
        pat = v.specializations[0]
        loc = _locate_dispatch_dim(callee.params, pat.dim_var)
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
        # A dispatch prototype selects a variant by the input shapes; its own
        # None body is never evaluated.
        target = _select_variant(fn, values) if fn.variants else fn
        env: dict[Var, Value] = dict(zip(target.params, values))
        dim_env = _bind_dim_vars(target.params, values)
        result = Evaluator(env, device, dim_env).visit(target.body)
    elif isinstance(fn_or_call, Call):
        if inputs:
            raise EvalError("evaluator: a Call entry takes no positional inputs")
        result = Evaluator({}, device).visit(fn_or_call)
    else:
        raise EvalError(
            f"evaluator: expected a Function or Call, got {type(fn_or_call).__name__}"
        )
    return _unwrap(result)
