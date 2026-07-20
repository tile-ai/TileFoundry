"""Shared static-AST evaluator, parameterized over caller policy (legal
node set, name lookup, ``ast.Div`` semantics). Raises ``VerifyError``;
callers needing another error contract translate at the call site.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, Literal

from tilefoundry.ir.core import VerifyError

DivMode = Literal["true", "floor"]

# The full node set ``eval_static`` understands. A caller passes a subset to
# reject forms it does not support (e.g. topology sizes accept only
# Constant/BinOp/UnaryOp; decorator arguments accept no arithmetic at all).
ALL_NODES: tuple[type, ...] = (
    ast.Constant, ast.Tuple, ast.List, ast.Name, ast.Attribute,
    ast.Subscript, ast.Call, ast.UnaryOp, ast.BinOp,
)


def _default_attr_resolver(owner: Any, attr: str) -> Any:
    try:
        return getattr(owner, attr)
    except AttributeError as exc:
        raise VerifyError(
            f"unknown attribute {attr!r} on {type(owner).__name__}"
        ) from exc


def _apply_binop(op: ast.operator, left: Any, right: Any, *, div: DivMode) -> Any:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.FloorDiv):
        return left // right
    if isinstance(op, ast.Div):
        return left // right if div == "floor" else left / right
    if isinstance(op, ast.Mod):
        return left % right
    raise VerifyError(f"static BinOp {type(op).__name__} not supported")


def _eval_index(node: ast.AST, ev: Callable[[ast.AST], Any]) -> Any:
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
        return tuple(_eval_index(e, ev) for e in node.elts)
    return ev(node)


def eval_static(
    node: ast.AST,
    *,
    closure: dict[str, Any],
    lookup: Callable[[str], Any] | None = None,
    allowed_nodes: tuple[type, ...] = ALL_NODES,
    div: DivMode = "true",
    attr_resolver: Callable[[Any, str], Any] | None = None,
    on_closure_name: Callable[[Any, str], None] | None = None,
) -> Any:
    """Evaluate a restricted static-AST subset.

    ``lookup(name)`` resolves an ``ast.Name`` through parser-lexical state
    (e.g. ``LexicalEnv.lookup``) before falling back to ``closure``; omit it
    to resolve directly against ``closure`` (the decorator / topology
    entry points run before any lexical env exists). ``attr_resolver``
    customizes ``owner.attr`` access (default: ``getattr`` translating
    ``AttributeError`` to ``VerifyError``). ``on_closure_name(value, name)``
    is called whenever a ``Name`` resolves via the ``closure`` fallback
    (used to warn on closure-captured IR objects). ``div`` selects true or
    floor semantics for ``ast.Div`` (``ast.FloorDiv`` is always floor);
    this is the one load-bearing policy knob — topology sizes use
    ``"floor"``, every other static attribute position uses the default
    ``"true"``.

    A node type outside *allowed_nodes*, an unresolved name, or an
    unsupported operator all raise :class:`VerifyError`.
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
        )

    if isinstance(node, ast.Constant) and ast.Constant in allowed_nodes:
        return node.value
    if isinstance(node, ast.Tuple) and ast.Tuple in allowed_nodes:
        return tuple(ev(e) for e in node.elts)
    if isinstance(node, ast.List) and ast.List in allowed_nodes:
        return [ev(e) for e in node.elts]
    if isinstance(node, ast.Name) and ast.Name in allowed_nodes:
        value = None if lookup is None else lookup(node.id)
        from_closure = False
        if value is None:
            value = closure.get(node.id)
            from_closure = True
        if value is None:
            raise VerifyError(f"undefined name {node.id!r}")
        if from_closure and on_closure_name is not None:
            on_closure_name(value, node.id)
        return value
    if isinstance(node, ast.Attribute) and ast.Attribute in allowed_nodes:
        owner = ev(node.value)
        resolver = attr_resolver or _default_attr_resolver
        return resolver(owner, node.attr)
    if isinstance(node, ast.Subscript) and ast.Subscript in allowed_nodes:
        owner = ev(node.value)
        return owner[_eval_index(node.slice, ev)]
    if isinstance(node, ast.Call) and ast.Call in allowed_nodes:
        if any(kw.arg is None for kw in node.keywords):
            raise VerifyError("static call does not accept **kwargs")
        fn = ev(node.func)
        args = tuple(ev(a) for a in node.args)
        kwargs = {kw.arg: ev(kw.value) for kw in node.keywords}
        return fn(*args, **kwargs)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and ast.UnaryOp in allowed_nodes
    ):
        return -ev(node.operand)
    if isinstance(node, ast.BinOp) and ast.BinOp in allowed_nodes:
        left = ev(node.left)
        right = ev(node.right)
        if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
            raise VerifyError(
                f"static BinOp requires numeric operands, got "
                f"{type(left).__name__} / {type(right).__name__}"
            )
        return _apply_binop(node.op, left, right, div=div)
    raise VerifyError(f"cannot statically evaluate AST node {type(node).__name__}")


__all__ = ["eval_static", "ALL_NODES", "DivMode"]
