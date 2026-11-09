"""Shared static-AST evaluator, parameterized over caller policy.

Shared static-AST evaluator, parameterized over caller policy (legal
node set, name lookup, ``ast.Div`` semantics). Raises ``VerifyError``;
callers needing another error contract translate at the call site.
"""

from __future__ import annotations

import ast
from types import MappingProxyType
from typing import Any, Callable, Literal

from tilefoundry import services
from tilefoundry.ir import dim, expr, types

DivMode = Literal["true", "floor"]

ALL_NODES: tuple[type, ...] = (
    ast.Constant, ast.Tuple, ast.List, ast.Name, ast.Attribute,
    ast.Subscript, ast.Call, ast.UnaryOp, ast.BinOp,
)

DIM_BINOPS = MappingProxyType({
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
})


def default_attr_resolver(owner: Any, attr: str) -> Any:
    try:
        return getattr(owner, attr)
    except AttributeError as exc:
        raise expr.VerifyError(
            f"unknown attribute {attr!r} on {type(owner).__name__}"
        ) from exc


def apply_binop(op: ast.operator, left: Any, right: Any, *, div: DivMode) -> Any:
    match op:
        case ast.Add():
            return left + right
        case ast.Sub():
            return left - right
        case ast.Mult():
            return left * right
        case ast.FloorDiv():
            return left // right
        case ast.Div():
            return left // right if div == "floor" else left / right
        case ast.Mod():
            return left % right
        case ast.Pow():
            return left ** right
    raise expr.VerifyError(f"static BinOp {type(op).__name__} not supported")


def eval_index(node: ast.AST, ev: Callable[[ast.AST], Any]) -> Any:
    """Lower a subscript index AST into a Python index value.

    ``ast.Slice`` -> ``slice``; a tuple of indices -> a tuple of lowered
    elements (slices stay slices); anything else is a scalar evaluated
    through *ev*.
    """
    if isinstance(node, ast.Slice):
        lo = None if node.lower is None else ev(node.lower)
        hi = None if node.upper is None else ev(node.upper)
        step = None if node.step is None else ev(node.step)
        return slice(lo, hi, step)
    if isinstance(node, ast.Tuple):
        return tuple(eval_index(e, ev) for e in node.elts)
    return ev(node)


def is_runtime_scalar(value: Any) -> bool:
    """Whether *value* is a runtime rank-0 integer tensor expression."""
    type_ = getattr(value, "type", None)
    return (
        isinstance(type_, types.TensorType)
        and type_.shape == ()
        and type_.dtype in (expr.DType.i32, expr.DType.i64)
    )


def eval_static(
    node: ast.AST,
    *,
    closure: dict[str, Any],
    lookup: Callable[[str], Any] | None = None,
    allowed_nodes: tuple[type, ...] = ALL_NODES,
    div: DivMode = "true",
    attr_resolver: Callable[[Any, str], Any] | None = None,
    on_closure_name: Callable[[Any, str], None] | None = None,
    allow_runtime_scalar: bool = False,
) -> Any:
    """Evaluate a restricted static-AST subset.

    *lookup* precedes closure fallback; *attr_resolver* customizes attribute
    access and *on_closure_name* observes closure resolutions. *div* selects
    true or floor division while ``FloorDiv`` always floors. Disallowed nodes,
    unresolved names, and unsupported operators raise ``VerifyError``.
    """

    def ev(n: ast.AST) -> Any:
        return eval_static(
            n,
            closure=closure,
            lookup=lookup,
            allowed_nodes=allowed_nodes,
            div=div,
            attr_resolver=attr_resolver,
            on_closure_name=on_closure_name,
            allow_runtime_scalar=allow_runtime_scalar,
        )

    match node:
        case ast.Constant(value=value) if ast.Constant in allowed_nodes:
            return value
        case ast.Tuple(elts=elts) if ast.Tuple in allowed_nodes:
            return tuple(ev(e) for e in elts)
        case ast.List(elts=elts) if ast.List in allowed_nodes:
            return [ev(e) for e in elts]
        case ast.Name(id=name) if ast.Name in allowed_nodes:
            value = None if lookup is None else lookup(name)
            from_closure = False
            if value is None:
                value = closure.get(name)
                from_closure = True
            if value is None:
                raise expr.VerifyError(f"undefined name {name!r}")
            if from_closure and on_closure_name is not None:
                on_closure_name(value, name)
            return value
        case ast.Attribute(value=value, attr=attr) if ast.Attribute in allowed_nodes:
            owner = ev(value)
            resolver = attr_resolver or default_attr_resolver
            return resolver(owner, attr)
        case ast.Subscript(value=value, slice=slice_) if ast.Subscript in allowed_nodes:
            owner = ev(value)
            return owner[eval_index(slice_, ev)]
        case ast.Call(func=func, args=args, keywords=keywords) if ast.Call in allowed_nodes:
            if any(kw.arg is None for kw in keywords):
                raise expr.VerifyError("static call does not accept **kwargs")
            fn = ev(func)
            values = tuple(ev(a) for a in args)
            kwargs = {kw.arg: ev(kw.value) for kw in keywords}
            return fn(*values, **kwargs)
        case ast.UnaryOp(op=ast.USub(), operand=operand) if ast.UnaryOp in allowed_nodes:
            return -ev(operand)
        case ast.BinOp(left=left_node, op=op, right=right_node) if ast.BinOp in allowed_nodes:
            left = ev(left_node)
            right = ev(right_node)
            numeric = isinstance(left, (int, float)) and isinstance(right, (int, float))
            dim_operands = dim.is_dimension(left) and dim.is_dimension(right)
            runtime_operands = allow_runtime_scalar and all(
                dim.is_dimension(operand) or is_runtime_scalar(operand)
                for operand in (left, right)
            )
            if not numeric and not (dim_operands or runtime_operands):
                raise expr.VerifyError(
                    f"static BinOp requires numeric or dimension operands, got "
                    f"{type(left).__name__} / {type(right).__name__}"
                )
            if not numeric:
                kind = DIM_BINOPS.get(type(op))
                if kind is None:
                    raise expr.VerifyError(
                        f"static dimension BinOp {type(op).__name__} not supported "
                        f"(use + - * // %)"
                    )
                return services.simplify_dim(kind, (left, right))
            return apply_binop(op, left, right, div=div)
    raise expr.VerifyError(f"cannot statically evaluate AST node {type(node).__name__}")


__all__ = ["eval_static", "ALL_NODES", "DivMode"]
