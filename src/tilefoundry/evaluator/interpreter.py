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
from tilefoundry.ir.core import Call, Constant, Tuple, Var
from tilefoundry.ir.core.pattern import locate_dim_var
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.types import TupleType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.visitor import ExprVisitor


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _maybe_torch_dtype(dtype) -> torch.dtype | None:
    """``to_torch_dtype``, but ``None`` (meaning: infer from the value,
    ``torch.as_tensor``'s default) for a declared HIR dtype with no torch
    mapping — e.g. ``f4e2m1``, real 4-bit quantization being out of scope —
    instead of raising."""
    try:
        return to_torch_dtype(dtype)
    except EvalError:
        return None


def _bind_top_level_input(arg: Any, declared_dtype, device: str) -> torch.Tensor:
    """Bind one top-level ``evaluate()`` input to a torch.Tensor on ``device``.

    An already-``torch.Tensor`` ``arg`` is only moved to ``device`` (a no-op
    if already there) — its own dtype is never overridden by the callee's
    declared HIR dtype. Per-op eval handlers key off a Call's own result
    dtype, never a param Var's declared static type (e.g. Cast's
    ``_eval_cast`` just does ``data.to(<the cast's own target dtype>)``), so
    forcing a cast here would only silently corrupt an already-prepared
    value — e.g. a ``WeightLoader.post_init``'d tensor (M1) whose physical
    dtype intentionally differs from what the HIR signature still declares
    (a real fp8e4m3 weight dequantized to bf16 once, upstream of this call).

    A non-tensor ``arg`` (a raw Python list / scalar) is constructed as a
    tensor of the declared dtype when the evaluator has a torch mapping for
    it, else left to infer its own dtype from the value (a declared dtype
    with no torch mapping at all, e.g. ``f4e2m1``).
    """
    if isinstance(arg, torch.Tensor):
        return arg.to(device=device)
    return torch.as_tensor(arg, dtype=_maybe_torch_dtype(declared_dtype), device=device)


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
        self, env: dict[int, Value], device: str,
        dim_env: dict[str, int] | None = None,
        leaves: dict[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.device = device
        self.dim_env = dim_env or {}
        self.leaves = leaves
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
            return self.env[id(var)]
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
            return self._call_function(target, call.args, call.type)
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

    def _call_function(self, callee: Function, arg_exprs, result_type: Any = None) -> Value:
        if len(arg_exprs) != len(callee.params):
            raise EvalError(
                f"evaluator: call to {callee.name!r} expects "
                f"{len(callee.params)} args, got {len(arg_exprs)}"
            )
        args = [self.visit(a) for a in arg_exprs]
        impl = self.leaves.get(callee.name) if self.leaves else None
        if impl is not None:
            return _call_leaf(impl, args, result_type)
        # A dispatch prototype (body is None) selects a variant by the runtime
        # argument shapes; its own None body is never evaluated.
        target = _select_variant(callee, args) if callee.variants else callee
        sub_env = {id(param): arg for param, arg in zip(target.params, args)}
        sub_dim_env = _bind_dim_vars(target.params, args)
        return Evaluator(sub_env, self.device, sub_dim_env, self.leaves).visit(target.body)

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
                id(iv): TensorValue(
                    data=torch.as_tensor(i, dtype=iv_dtype, device=self.device),
                    type=iv.type,
                ),
            }
            for phi, value in zip(region.carried_args, carried):
                env[id(phi)] = value
            return env

        if not region.carried_args:
            # No-carry loop: the value is the final body evaluation.
            last = None
            for i in indices:
                last = Evaluator(
                    iter_env(i, ()), self.device, self.dim_env, self.leaves
                ).visit(region.body)
            if last is None:
                raise EvalError(
                    "evaluator: GridRegionExpr has an empty iteration domain"
                )
            return last

        carried = [self.visit(init) for init in region.init_args]
        for i in indices:
            sub = Evaluator(iter_env(i, carried), self.device, self.dim_env, self.leaves)
            carried = [sub.visit(y) for y in region.yield_values]
        return carried[0] if len(carried) == 1 else TupleValue(tuple(carried))


def _unwrap(value: Value) -> Any:
    if isinstance(value, TensorValue):
        return value.data
    if isinstance(value, TupleValue):
        return tuple(_unwrap(v) for v in value.elements)
    return value


def _call_leaf(impl: Any, args: list[Value], result_type: Any) -> Value:
    """Run a registered leaf's :class:`~tilefoundry.evaluator.leaf.ImplementationPackage`
    instead of recursing into the callee's HIR body (M1): evaluate the call's
    own args as usual, then hand the unwrapped torch tensors straight to
    ``impl.fn_or_source`` and re-wrap its result against the Call's own
    (already-inferred) result type so it flows back into the surrounding HIR
    evaluation like any other value."""
    if impl.language != "torch":
        raise EvalError(
            f"evaluator: leaf language {impl.language!r} not supported "
            f"(only 'torch' is wired tonight)"
        )
    raw = impl.fn_or_source(*(_unwrap(a) for a in args))
    if isinstance(result_type, TupleType):
        return TupleValue(tuple(
            TensorValue(data=r, type=field) for r, field in zip(raw, result_type.fields)
        ))
    return TensorValue(data=raw, type=result_type)


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


def evaluate(
    fn_or_call, *inputs, backend: str = "torch", device: str | None = None,
    leaves: dict[str, Any] | None = None,
):
    """Evaluate a HIR ``Function`` (or ``Call``) and return torch value(s).

    ``inputs`` bind positionally to a ``Function``'s parameters; the result is
    a ``torch.Tensor`` for a single output or a tuple for a ``TupleType``.

    ``leaves`` (M1), when given, is a ``{fn_name: ImplementationPackage}`` map
    (see ``tilefoundry.evaluator.leaf.LeafRegistry.by_function_name``): a call
    to a Function whose name is in ``leaves`` — whether ``fn_or_call`` itself
    or any Function it calls, at any nesting depth — runs that
    ImplementationPackage instead of recursing into the callee's HIR body.
    Omitted (the default), evaluation is exactly the pre-M1 plain evaluator.
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
        impl = leaves.get(fn.name) if leaves else None
        if impl is not None:
            if impl.language != "torch":
                raise EvalError(
                    f"evaluator: leaf language {impl.language!r} not supported "
                    f"(only 'torch' is wired tonight)"
                )
            # Raw pass-through: `inputs` are already torch tensors from the
            # caller, so skip the to_torch_dtype conversion below entirely —
            # it would reject a dtype the evaluator has no torch mapping for
            # (e.g. f4e2m1) even though the leaf impl never needs one.
            result = impl.fn_or_source(*inputs)
        else:
            values = [
                TensorValue(
                    data=_bind_top_level_input(arg, param.type.dtype, device),
                    type=param.type,
                )
                for param, arg in zip(fn.params, inputs)
            ]
            # A dispatch prototype selects a variant by the input shapes; its
            # own None body is never evaluated.
            target = _select_variant(fn, values) if fn.variants else fn
            env = {id(param): value for param, value in zip(target.params, values)}
            dim_env = _bind_dim_vars(target.params, values)
            result = Evaluator(env, device, dim_env, leaves).visit(target.body)
    elif isinstance(fn_or_call, Call):
        if inputs:
            raise EvalError("evaluator: a Call entry takes no positional inputs")
        result = Evaluator({}, device, None, leaves).visit(fn_or_call)
    else:
        raise EvalError(
            f"evaluator: expected a Function or Call, got {type(fn_or_call).__name__}"
        )
    return _unwrap(result)
