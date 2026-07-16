from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any

from tilefoundry.ir.core import Call, Constant, Expr, Tuple, Var
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.op import Op
from tilefoundry.ir.hir.function import Function


def _canonical(value: Any, function_ids: dict[int, int], expr_ids: dict[int, int]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return {"enum": type(value).__name__, "value": _canonical(value.value, function_ids, expr_ids)}
    if isinstance(value, Function):
        return {"function": function_ids[id(value)]}
    if isinstance(value, Expr):
        return {"expr": expr_ids.get(id(value), _expr_key(value, function_ids, expr_ids))}
    if isinstance(value, Op):
        attrs = {}
        for info in type(value).params():
            if info.kind == "attribute":
                attrs[info.name] = _canonical(getattr(value, info.name), function_ids, expr_ids)
        return {"op": type(value).__name__, "attrs": attrs}
    if type(value).__name__ == "TensorType":
        return {
            "shape": _canonical(value.shape, function_ids, expr_ids),
            "dtype": _canonical(value.dtype, function_ids, expr_ids),
        }
    if isinstance(value, tuple):
        return [_canonical(item, function_ids, expr_ids) for item in value]
    if isinstance(value, list):
        return [_canonical(item, function_ids, expr_ids) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical(item, function_ids, expr_ids)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if dataclasses.is_dataclass(value):
        return {
            field.name: _canonical(getattr(value, field.name), function_ids, expr_ids)
            for field in dataclasses.fields(value)
            if field.name not in {"loc", "metadata"}
        }
    return repr(value)


def _expr_key(expr: Expr, function_ids: dict[int, int], expr_ids: dict[int, int]) -> Any:
    if isinstance(expr, Var):
        return {"kind": "Var", "type": _canonical(expr.type, function_ids, expr_ids)}
    if isinstance(expr, Constant):
        return {
            "kind": "Constant",
            "type": _canonical(expr.type, function_ids, expr_ids),
            "value": _canonical(expr.value, function_ids, expr_ids),
        }
    if isinstance(expr, Tuple):
        return {
            "kind": "Tuple",
            "type": _canonical(expr.type, function_ids, expr_ids),
            "elements": [_canonical(item, function_ids, expr_ids) for item in expr.elements],
        }
    if isinstance(expr, Call):
        return {
            "kind": "Call",
            "type": _canonical(expr.type, function_ids, expr_ids),
            "target": _canonical(expr.target, function_ids, expr_ids),
            "args": [_canonical(arg, function_ids, expr_ids) for arg in expr.args],
        }
    return {
        "kind": type(expr).__name__,
        "type": _canonical(expr.type, function_ids, expr_ids),
    }


def _is_reshard(expr: Expr) -> bool:
    return isinstance(expr, Call) and type(expr.target).__name__ == "Reshard"


def _reachable_functions(module: Module) -> tuple[Function, ...]:
    ordered: list[Function] = []
    seen: set[int] = set()

    def visit(fn: Function) -> None:
        if id(fn) in seen:
            return
        seen.add(id(fn))
        ordered.append(fn)
        if fn.body is None:
            return
        stack = [fn.body]
        while stack:
            expr = stack.pop()
            if isinstance(expr, Call):
                if isinstance(expr.target, Function):
                    visit(expr.target)
                stack.extend(reversed(expr.args))
            elif isinstance(expr, Tuple):
                stack.extend(reversed(expr.elements))

    entry = module.entry_function()
    if not isinstance(entry, Function):
        raise TypeError("logical fingerprint currently supports HIR Module entries only")
    visit(entry)
    return tuple(ordered)


def logical_fingerprint(module: Module) -> str:
    """Hash logical HIR semantics while ignoring names, locations, and metadata."""
    functions = _reachable_functions(module)
    function_ids = {id(fn): index for index, fn in enumerate(functions)}
    expr_ids: dict[int, int] = {}
    function_payload = []
    next_expr_id = 0
    for fn in functions:
        exprs: list[Expr] = []

        def visit_expr(expr: Expr) -> int:
            nonlocal next_expr_id
            if _is_reshard(expr):
                ref = visit_expr(expr.args[0])
                expr_ids[id(expr)] = ref
                return ref
            existing = expr_ids.get(id(expr))
            if existing is not None:
                return existing
            ref = next_expr_id
            next_expr_id += 1
            expr_ids[id(expr)] = ref
            exprs.append(expr)
            if isinstance(expr, Call):
                for arg in expr.args:
                    visit_expr(arg)
            elif isinstance(expr, Tuple):
                for element in expr.elements:
                    visit_expr(element)
            return ref

        for param in fn.params:
            visit_expr(param)
        body_ref = None if fn.body is None else visit_expr(fn.body)
        nodes = []
        for expr in exprs:
            node = _expr_key(expr, function_ids, expr_ids)
            node["id"] = expr_ids[id(expr)]
            nodes.append(node)
        function_payload.append(
            {
                "id": function_ids[id(fn)],
                "params": [expr_ids[id(param)] for param in fn.params],
                "body": body_ref,
                "return_type": _canonical(fn.return_type, function_ids, expr_ids),
                "nodes": nodes,
            }
        )
    payload = {"functions": function_payload, "entry": function_ids[id(module.entry_function())]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["logical_fingerprint"]
