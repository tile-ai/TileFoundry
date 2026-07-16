from __future__ import annotations

import ast
import dataclasses
from typing import Any

from tilefoundry.ir.core import Call, Constant, Expr, Var, VerifyError
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
    simplify_dim,
)
from tilefoundry.ir.types.shard.mesh import Mesh, Topology
from tilefoundry.ir.visitor import ExprMutator
from tilefoundry.schedule.constraints import (
    AgentConstraint,
    ConstraintProvenance,
    LayoutConstraint,
    LayoutDimConstraint,
    LayoutDimKind,
    PartialConstraint,
    SourceLocation,
    merge_constraints,
)

from .base import BaseExprVisitor, _constant_from_py, _warn_if_ir_object, extract_ast
from .range_slice import RangeSlice
from .sugar import _is_tuple_sugar, parse_mesh_layout_sugar, try_parse_sugar_tensor_type
from .symtab import LexicalEnv

_DIM_OP_TYPES = (DimAdd, DimSub, DimMul, DimFloorDiv, DimMod, DimMin, DimMax)

# Python AST binary ops → dim ops, for resolving loop bounds (tile/range
# extent / step / start) that mix DimVars with ints, e.g. ``C // NUM_SPLITS``.
_AST_DIM_OPS = {
    ast.Add: DimAdd,
    ast.Sub: DimSub,
    ast.Mult: DimMul,
    ast.FloorDiv: DimFloorDiv,
    ast.Mod: DimMod,
}


def _is_dim_expr(v) -> bool:
    """A legal ``tile(...)`` extent / step: a static ``int``, a ``DimVar``, or a
    dim ``Expr`` built only from the ``dim`` ops over int / DimVar leaves.

    Rejects arbitrary ``Expr`` (e.g. a tensor-op ``Call``) so a malformed
    extent cannot reach IR."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, DimVar):
        return True
    if isinstance(v, Constant):
        return isinstance(v.value, int) and not isinstance(v.value, bool)
    if isinstance(v, Call):
        return isinstance(v.target, _DIM_OP_TYPES) and all(
            _is_dim_expr(a) for a in v.args
        )
    return False


def parse_func(fn, *, topologies=(), specializations=(), target=None, extra_closure=None) -> Function:
    """@tilefoundry.func parser entry. Parse fn's source into hir.Function.

    ``extra_closure`` adds names to the resolution namespace below ``fn``'s own
    globals/freevars; it lets an ``@func`` defined in a ``@module`` class body
    resolve sibling ``@func`` methods (which are ``hir.Function`` values) as
    nested-call targets.
    """
    if isinstance(fn, Function):
        return fn
    node = extract_ast(fn)
    closure = _collect_closure(fn, extra_closure)
    return _parse_func_node(
        node, closure, topologies=topologies,
        specializations=specializations, target=target,
        source_filename=getattr(getattr(fn, "__code__", None), "co_filename", "<string>"),
    )

def parse_func_source(src: str) -> Function:
    """Parse @func-decorated function from Python source string.

    Used for round-trip: ``parse_func_source(printer_output)`` reconstructs
    the HIR Function. Imports are executed to build a source-level prelude
    namespace (legacy compat); the canonical path relies on parser-recognised
    AST constructors (Topology, Layout, etc.) and string-name topology
    resolution rather than arbitrary Python object capture.
    """
    tree = ast.parse(src)
    # Find the FunctionDef decorated with @func or @func(...)
    func_node = None
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            for dec in stmt.decorator_list:
                if _is_func_decorator(dec):
                    func_node = stmt
                    break
        if func_node:
            break
    if func_node is None:
        raise ValueError("no @func-decorated function found in source")

    # Build closure from source-level imports and assignments
    closure: dict[str, Any] = {}
    # Collect all top-level statements that are NOT @func definitions
    prelude_lines = []
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            continue  # skip function defs
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                name = alias.asname or alias.name
                prelude_lines.append(f"import {alias.name} as {name}")
        elif isinstance(stmt, ast.ImportFrom):
            names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "")
                              for a in stmt.names)
            prelude_lines.append(f"from {stmt.module or ''} import {names}")
        elif isinstance(stmt, (ast.Assign, ast.Expr)):
            # Module-level variable assignments (e.g. sl = ShardLayout(...))
            prelude_lines.append(ast.unparse(stmt))
    exec("\n".join(prelude_lines), closure)

    # Extract topologies from decorator AST
    parsed_topologies = _extract_topologies_from_decorator(func_node)
    return _parse_func_node(func_node, closure, topologies=parsed_topologies)

def _parse_func_node(
    node: ast.FunctionDef,
    closure: dict[str, Any],
    *,
    topologies=(),
    specializations=(),
    target=None,
    source_filename: str = "<string>",
) -> Function:
    env = LexicalEnv()
    params = _build_params(node, closure)
    for p in params:
        env.define(p.name, p)
    # Build topology namespace: {name → Topology} for string-name resolution
    topo_ns: dict[str, "Topology"] = {}
    for t in topologies:
        if t.name in topo_ns:
            raise VerifyError(f"duplicate topology name {t.name!r}")
        topo_ns[t.name] = t
    visitor = _HirBodyVisitor(
        env,
        closure,
        topo_ns=topo_ns,
        source_filename=source_filename,
    )
    if _is_pass_body(node.body):
        # A `pass` body declares a dispatch prototype: signature + envelope
        # only, no implementation (hir §5). Its variants carry the bodies.
        body_expr = None
    else:
        body_expr = visitor.visit_body(node.body)
    return_type = _resolve_return_type(node, closure, body_expr)
    params, body_expr = _finalize_agent_metadata(
        params, body_expr, visitor.pending_constraints
    )
    function = Function.build(
        name=node.name,
        params=params,
        body=body_expr,
        return_type=return_type,
        topologies=tuple(topologies),
        specializations=tuple(specializations),
        target=target,
    )
    return function


class _AgentMetadataFinalizer(ExprMutator):
    """Rebuild the parsed DAG once so metadata follows shared SSA identity."""

    def __init__(self, pending, parameter_replacements):
        self.pending = pending
        self.parameter_replacements = parameter_replacements
        self.memo: dict[int, Expr] = {}

    def visit(self, expr: Expr) -> Expr:
        cached = self.memo.get(id(expr))
        if cached is not None:
            return cached
        rebuilt = super().visit(expr)
        constraints = self.pending.get(id(expr))
        if constraints:
            source_loc = constraints[0].source_loc
            rebuilt = dataclasses.replace(
                rebuilt,
                metadata=merge_constraints(
                    rebuilt.metadata, tuple(constraints), source_loc
                ),
            )
        self.memo[id(expr)] = rebuilt
        return rebuilt

    def visit_Var(self, var: Var) -> Expr:
        return self.parameter_replacements.get(id(var), var)


def _finalize_agent_metadata(params, body, pending):
    replacements = {}
    for param in params:
        if id(param) in pending:
            replacements[id(param)] = dataclasses.replace(param)
    finalizer = _AgentMetadataFinalizer(pending, replacements)
    new_params = tuple(finalizer.visit(param) for param in params)
    new_body = None if body is None else finalizer.visit(body)
    return new_params, new_body

def _extract_topologies_from_decorator(node: ast.FunctionDef) -> list["Topology"]:
    """Parse ``@func(topologies=(...))`` from the function's decorator AST.

    Returns a list of Topology objects extracted from the decorator's
    ``topologies`` keyword argument. Supports:

    - ``Topology("cta", 128)`` — inline constructor (canonical)
    - ``cta`` — variable reference resolved through closure (legacy)
    """
    result: list["Topology"] = []

    for dec in node.decorator_list:
        if isinstance(dec, ast.Call):
            if not _is_func_decorator(dec):
                continue
            for kw in dec.keywords:
                if kw.arg != "topologies":
                    continue
                # Parse topology values from the tuple/list literal
                values = kw.value
                if isinstance(values, (ast.Tuple, ast.List)):
                    for elt in values.elts:
                        topo = _parse_topology_item(elt)
                        if topo is not None:
                            result.append(topo)
                elif isinstance(values, ast.Name):
                    # Single variable reference from closure (legacy)
                    pass  # handled by the decorator's runtime kwarg passing
    return result

def _parse_topology_item(node: ast.AST) -> "Topology | None":
    """Parse a single topology declaration from an AST node.

    Returns a Topology object, or None if the node can't be parsed statically
    (e.g. a variable reference — handled at Python runtime by the decorator).

    Supports finite compile-time expressions for *size*:
    - ``Topology("cta", 128)``          (int literal)
    - ``Topology("thread", 8 * 32)``   (safe arithmetic)
    """
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "Topology":
            if len(node.args) >= 2:
                name = node.args[0]
                size = node.args[1]
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    size_val = _eval_topology_expr(size)
                    if size_val is not None:
                        return Topology(name=name.value, size=size_val)
    return None

def _eval_topology_expr(node: ast.AST) -> int | None:
    """Safely evaluate a compile-time expression for topology size.

    Handles integer literals, binary arithmetic, and unary minus.
    Returns None for expressions that cannot be statically evaluated.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int):
            return node.value
        return None
    if isinstance(node, ast.BinOp):
        left = _eval_topology_expr(node.left)
        right = _eval_topology_expr(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Div):
            if right == 0:
                return None
            # Floor div for topology sizes
            return left // right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                return None
            return left // right
        return None
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            val = _eval_topology_expr(node.operand)
            if val is not None:
                return -val
        return None
    return None


def _constraint_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return ast.unparse(node)


def _parse_partial_value(node: ast.AST) -> str:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise VerifyError("partial constraint must be P(\"reduction\")")
    if node.func.id != "P" or len(node.args) != 1 or node.keywords:
        raise VerifyError("partial constraint must be P(\"reduction\")")
    value = _constraint_value(node.args[0])
    if not isinstance(value, str) or not value:
        raise VerifyError("partial reduction must be a non-empty string")
    return value


def _parse_partial_set(node: ast.AST) -> list[tuple[str | None, str]]:
    if not isinstance(node, ast.Set):
        raise VerifyError("layout Partial sugar must be a set")
    out: list[tuple[str | None, str]] = []
    for item in node.elts:
        if not isinstance(item, ast.BinOp) or not isinstance(item.op, ast.MatMult):
            raise VerifyError("Partial sugar must use `cta @ P(\"sum\")`")
        reduction = _parse_partial_value(item.right)
        topology = _constraint_value(item.left)
        if not isinstance(topology, str):
            raise VerifyError("Partial sugar topology must be symbolic")
        out.append((topology, reduction))
    return out


def _parse_layout_constraint(
    node: ast.AST,
) -> tuple[list[LayoutDimConstraint], list[tuple[str | None, str]]]:
    if not isinstance(node, ast.Tuple):
        raise VerifyError("layout constraint must be a tuple")
    dims_node = node
    partial_node = None
    if node.elts and isinstance(node.elts[0], ast.Tuple):
        dims_node = node.elts[0]
        extras = node.elts[1:]
        if len(extras) > 1:
            raise VerifyError("layout constraint accepts at most one Partial set")
        if extras:
            partial_node = extras[0]
    if not dims_node.elts:
        raise VerifyError("layout constraint cannot be empty")
    dims: list[LayoutDimConstraint] = []
    for index, item in enumerate(dims_node.elts):
        if isinstance(item, ast.Name) and item.id == "_":
            dims.append(LayoutDimConstraint(index, LayoutDimKind.UNCONSTRAINED))
            continue
        if isinstance(item, ast.Name) and item.id == "D":
            dims.append(LayoutDimConstraint(index, LayoutDimKind.BROADCAST))
            continue
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.MatMult):
            topology = _constraint_value(item.right)
            if topology != "cta" and not str(topology).endswith(".cta"):
                raise VerifyError("layout Split constraint must target symbolic cta")
            dims.append(
                LayoutDimConstraint(
                    index=index,
                    kind=LayoutDimKind.SPLIT,
                    extent=_constraint_value(item.left),
                    topology="cta",
                )
            )
            continue
        raise VerifyError(
            "layout dimensions must use `_`, `D`, or `H @ cta`/`N @ cta`"
        )
    partials = [] if partial_node is None else _parse_partial_set(partial_node)
    return dims, partials

def _is_func_decorator(dec: ast.AST) -> bool:
    """Check if an AST decorator node is ``@func`` or ``@func(...)``."""
    # @func (bare)
    if isinstance(dec, ast.Name) and dec.id == "func":
        return True
    # module.func
    if isinstance(dec, ast.Attribute) and dec.attr == "func":
        return True
    # @func(topologies=(...)) — Call wrapping Name/Attribute
    if isinstance(dec, ast.Call):
        return _is_func_decorator(dec.func)
    return False

def _collect_closure(fn, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    closure: dict[str, Any] = {}
    if extra:
        closure.update(extra)
    if fn.__globals__ is not None:
        closure.update(fn.__globals__)
    if fn.__closure__ is not None:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                pass
    return closure

def _build_params(node: ast.FunctionDef, closure: dict[str, Any]) -> tuple[Var, ...]:
    out: list[Var] = []
    for a in node.args.args:
        if a.annotation is None:
            raise VerifyError(f"@tilefoundry.func param {a.arg!r} must be annotated")
        ann_type = _resolve_tensor_type(a.annotation, closure)
        out.append(Var(type=ann_type, name=a.arg))
    return tuple(out)

def _resolve_tensor_type(node: ast.AST, closure: dict[str, Any]) -> TensorType:
    """Resolve a tensor type annotation.

    Supports two forms:

    1. **Sugar**: ``Tensor[(M,K), bf16, ((32 @ gpu.cluster, K), {gpu.warp @ P("sum")}), "smem"]``
       — compact layout sugar, parsed directly from the AST without ``eval()``.
    2. **Verbose**: ``Tensor[(M,K), bf16, ShardLayout(...), "smem"]``
       — evaluated via ``eval()`` in *closure*.
    """
    # Try sugar path first (handles @ mesh.axis bindings)
    result = try_parse_sugar_tensor_type(node, closure)
    if result is not None:
        return result

    # Fallback: statically eval the whole annotation in the closure.
    try:
        code = compile(ast.Expression(body=node), "<ann>", "eval")
        val = eval(code, closure)  # noqa: S307 — controlled internal eval
    except Exception as exc:
        raise VerifyError(f"failed to resolve type annotation: {exc}")
    if isinstance(val, TensorType):
        return val
    raise VerifyError(f"annotation did not resolve to TensorType, got {type(val).__name__}")

def _is_pass_body(stmts: list[ast.stmt]) -> bool:
    """A dispatch-prototype body is exactly ``pass``. A ``pass`` mixed with any
    other statement is rejected (it is not a partial body form)."""
    if not any(isinstance(s, ast.Pass) for s in stmts):
        return False
    if len(stmts) != 1:
        raise VerifyError(
            "@tilefoundry.func: `pass` must be the entire body — it declares a "
            "dispatch prototype (signature only); mixing it with other "
            "statements is not allowed"
        )
    return True


def _resolve_return_type(node: ast.FunctionDef, closure, body_expr) -> TensorType:
    if node.returns is not None:
        return _resolve_tensor_type(node.returns, closure)
    if body_expr is None:
        raise VerifyError(
            "@tilefoundry.func: a `pass` prototype must annotate its return type"
        )
    # fallback: try body_expr.type (set by Op construction — coarse).
    t = getattr(body_expr, "type", None)
    if t is None:
        raise VerifyError("@tilefoundry.func: cannot determine return_type")
    return t

class _HirBodyVisitor(BaseExprVisitor):
    token = "hir"

    def __init__(
        self,
        env,
        closure,
        *,
        topo_ns=None,
        source_filename: str = "<string>",
    ):
        super().__init__(env, closure)
        self.topo_ns: dict[str, "Topology"] = topo_ns or {}
        self.source_filename = source_filename
        self.pending_constraints: dict[int, list[AgentConstraint]] = {}

    # Function body: assignment statements only update the symtab; hir is
    # SSA-as-DAG (§8.6), so variable sharing is expressed by the tail Expr
    # referencing the same Expr object via env lookup. The body returns the
    # tail `return` expression directly — no LetExpr node emitted.
    def visit_body(self, stmts: list[ast.stmt]) -> Expr:
        # A nested function definition (a `@func` helper or a plain `def`) is
        # not allowed anywhere in an @func body — including inside a
        # `with Mesh(...)` suite and after a `return`. The check is syntactic
        # (over the whole AST subtree), so it does not depend on reachability.
        for stmt in stmts:
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    raise VerifyError(
                        "hir: nested function definition not allowed in an "
                        "@tilefoundry.func body (helper functions are "
                        "module/function-level definitions)"
                    )
        return self._visit_chain(stmts, 0)

    def _visit_chain(
        self, stmts: list[ast.stmt], idx: int, require_return: bool = True
    ) -> Expr | None:
        """Fold a statement chain into a single tail ``Expr``.

        ``require_return=True`` (the function body) requires a terminal
        ``return`` and raises when the chain runs out. A ``with Mesh(...)``
        suite is folded with ``require_return=False``: a setup-only suite that
        carries no ``return`` yields ``None`` so the caller can continue folding
        the post-``with`` tail in the outer frame.
        """
        if idx >= len(stmts):
            if require_return:
                raise VerifyError("@tilefoundry.func body must end with `return`")
            return None
        node = stmts[idx]
        if isinstance(node, ast.Return):
            if node.value is None:
                raise VerifyError("§8.2: @tilefoundry.func return must carry a value")
            if isinstance(node.value, ast.Tuple):
                return self._tuple_expr_expr(node.value)
            return self.expr(node.value)
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise VerifyError("hir: only single-target assignments supported in V1")
            target = node.targets[0]
            if isinstance(target, ast.Name):
                tgt = target.id
                rhs = self.expr(node.value)
                # Auto-fill Call.loc from LHS variable name when the
                # user did not supply ``loc=`` explicitly. Applies to any
                # Call (op call, TupleGetItem from Subscript, etc.).
                rhs = self._maybe_autofill_loc(rhs, tgt)
                # Bind name → Expr directly; DAG sharing replaces LetExpr binding.
                self.env.define(tgt, rhs)
                return self._visit_chain(stmts, idx + 1, require_return)
            if isinstance(target, ast.Tuple):
                # tuple unpack: `a, b = call(...)`
                # Parser detects ast.Tuple targets, requires RHS type is TupleType,
                # and synthesizes TupleGetItem(rhs, index=i) bindings for each name.
                rhs = self.expr(node.value)
                if not isinstance(rhs.type, TupleType):
                    raise VerifyError(
                        f"hir: tuple unpack requires RHS of TupleType, "
                        f"got {type(rhs.type).__name__}"
                    )
                names: list[str] = []
                for elt in target.elts:
                    if not isinstance(elt, ast.Name):
                        raise VerifyError(
                            "hir: tuple unpack targets must all be plain names"
                        )
                    names.append(elt.id)
                if len(names) != len(rhs.type.fields):
                    raise VerifyError(
                        f"hir: tuple unpack arity mismatch — RHS has "
                        f"{len(rhs.type.fields)} fields, LHS binds {len(names)} names"
                    )
                # Parent Call default loc = DSL callable name
                # (no single LHS to fall back on); user-explicit loc is
                # preserved by call_to_op_call.
                rhs = self._maybe_autofill_loc_default(rhs)
                for i, nm in enumerate(names):
                    item = self._build_call(TupleGetItem(index=i), (rhs,))
                    item = self._maybe_autofill_loc(item, nm)
                    self.env.define(nm, item)
                return self._visit_chain(stmts, idx + 1, require_return)
            raise VerifyError("hir: only single-target Name or Tuple assignments supported in V1")
        if isinstance(node, ast.AnnAssign):
            return self._visit_annotated_assignment(node, stmts, idx, require_return)
        if isinstance(node, ast.With):
            return self._visit_with(node, stmts, idx, require_return)
        if isinstance(node, ast.Expr):
            # bare expression statement: allowed only if it produces a Call
            # whose value is not used — but hir has no ExprStmt, so this is
            # invalid unless it's the final return-like form.
            raise VerifyError("hir: bare expression statement not allowed; use assign or return")
        if isinstance(node, ast.For):
            return self._visit_loop_for(node, stmts, idx, require_return)
        raise VerifyError(f"hir: unsupported statement {type(node).__name__}")

    def _visit_annotated_assignment(
        self,
        node: ast.AnnAssign,
        stmts: list[ast.stmt],
        idx: int,
        require_return: bool,
    ) -> Expr | None:
        if not isinstance(node.target, ast.Name):
            raise VerifyError("where assignment target must be a plain Name")
        if node.value is None:
            target = self.env.lookup(node.target.id)
            if not isinstance(target, Expr):
                raise VerifyError(
                    f"where annotation target {node.target.id!r} is not an existing Expr"
                )
        else:
            target = self._maybe_autofill_loc(
                self.expr(node.value), node.target.id
            )
            self.env.define(node.target.id, target)
        self._record_annotated_assignment(node, target)
        return self._visit_chain(stmts, idx + 1, require_return)

    def _record_annotated_assignment(self, node: ast.AnnAssign, target: Expr) -> None:
        constraints = self._parse_where_annotation(node.annotation, node)
        if any(isinstance(c, LayoutConstraint) for c in constraints) and not isinstance(
            target.type, TensorType
        ):
            raise VerifyError("layout where constraint requires a tensor-valued Expr")
        self.pending_constraints.setdefault(id(target), []).extend(constraints)

    def _parse_where_annotation(
        self, annotation: ast.AST, node: ast.AnnAssign
    ) -> tuple[AgentConstraint, ...]:
        if not isinstance(annotation, ast.Call) or not isinstance(
            annotation.func, ast.Name
        ) or annotation.func.id != "where":
            raise VerifyError("annotations must use `where(...)`; require(...) is not supported")
        if annotation.args:
            raise VerifyError("where(...) accepts keyword arguments only")
        if not annotation.keywords:
            raise VerifyError("where(...) cannot be empty")
        source_loc = SourceLocation(
            filename=self.source_filename,
            line=node.lineno,
            column=node.col_offset,
            end_line=getattr(node, "end_lineno", None),
            end_column=getattr(node, "end_col_offset", None),
        )
        out: list[AgentConstraint] = []
        for keyword in annotation.keywords:
            if keyword.arg is None:
                raise VerifyError("where(...) does not accept **kwargs")
            if keyword.arg == "layout":
                dims, partials = _parse_layout_constraint(keyword.value)
                out.append(
                    LayoutConstraint(
                        dims=tuple(dims),
                        source_loc=source_loc,
                        provenance=ConstraintProvenance.AUTHOR,
                    )
                )
                out.extend(
                    PartialConstraint(
                        reduction=reduction,
                        topology=topology,
                        source_loc=source_loc,
                        provenance=ConstraintProvenance.AUTHOR,
                    )
                    for topology, reduction in partials
                )
            elif keyword.arg == "partial":
                reduction = _parse_partial_value(keyword.value)
                out.append(
                    PartialConstraint(
                        reduction=reduction,
                        source_loc=source_loc,
                        provenance=ConstraintProvenance.AUTHOR,
                    )
                )
            else:
                raise VerifyError(
                    f"where(...) has unknown field {keyword.arg!r}; "
                    "use layout=... or partial=..."
                )
        return tuple(out)

    def _resolve_loop_bound(self, node: ast.AST):
        """Resolve a ``tile`` / ``range`` bound (extent / step / start) to an
        ``int``, ``DimVar``, or dim ``Expr``.

        Unlike ``_eval_static`` (which only folds numeric constants), a
        ``BinOp`` whose operands reach a ``DimVar`` builds a dim expression via
        ``simplify_dim`` (e.g. ``C // NUM_SPLITS`` → ``DimFloorDiv(C, N)``). The
        IR / evaluator already resolve dim-expression loop bounds; this lets the
        DSL surface write them."""
        if isinstance(node, ast.BinOp):
            op = _AST_DIM_OPS.get(type(node.op))
            if op is None:
                raise VerifyError(
                    f"loop bound: unsupported operator {type(node.op).__name__} "
                    f"(use + - * // %)"
                )
            left = self._resolve_loop_bound(node.left)
            right = self._resolve_loop_bound(node.right)
            # All-numeric folds to a Python int (keeps the static path simple);
            # any DimVar operand produces a dim Expr.
            if isinstance(left, int) and not isinstance(left, bool) and \
                    isinstance(right, int) and not isinstance(right, bool):
                return self._eval_static(node)
            return simplify_dim(op, (left, right))
        return self._eval_static(node)

    def _visit_loop_for(self, node: ast.For, stmts, idx, require_return: bool = True):
        """§3.3 / §5.4: `for i in tile(...)` / `for i in range(...)` →
        GridRegionExpr, then continue the statement chain.

        ``tile`` and ``range`` share the same loop domain ``(start, extent,
        step)`` and lower to the same node; they differ only in the loop-var
        binding (``tile`` 2-arg → a RangeSlice usable as ``x[:, t]``; ``range``
        and 1-arg ``tile`` → a scalar induction var). Neither is unrolled.
        """
        grid = self._build_grid_for(node)
        if idx + 1 < len(stmts):
            return self._visit_chain(stmts, idx + 1, require_return)
        # Loop is the last statement in this chain. As a function body tail it
        # is the result value; inside a setup-only `with Mesh(...)` suite
        # (require_return=False) it is a carry-out loop that falls through to
        # the post-`with` tail in the outer frame.
        return grid if require_return else None

    def _build_grid_for(self, node: ast.For) -> Expr:
        """Build a ``GridRegionExpr`` from a ``for ... in tile/range(...)`` node,
        rebinding its carry names in the *current* frame, and return the grid.

        Carry-out lifting: any ``ast.Assign`` in the body whose single Name
        target is already bound in *outer* scope is a loop-carried rebinding —
        a fresh phi ``Var`` is bound inside the loop, the final RHS is its
        ``yield_value``, and after the loop the name rebinds to the post-loop
        value (single carry → the grid; multi-carry → TupleGetItem projections).

        Sibling statements are NOT processed here (the caller continues the
        chain), so a nested ``for`` inside a grid body composes by calling this.
        Body must not contain ``return``; v1 accepts ``=`` assigns and nested
        ``for`` loops (nested GridRegions).
        """
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name):
            raise VerifyError("hir For: iter must be a `tile(...)` or `range(...)` call")
        loop_kind = node.iter.func.id
        if loop_kind not in ("tile", "range"):
            raise VerifyError(
                f"hir For: iter must be `tile(...)` or `range(...)`, got "
                f"{loop_kind!r}"
            )
        if not isinstance(node.target, ast.Name):
            raise VerifyError("hir For: target must be a Name")
        iv = Var(type=TensorType.scalar(DType.i64), name=node.target.id)

        loop_args = node.iter.args
        iv_binding: Expr | RangeSlice
        if loop_kind == "range":
            # range(stop) | range(start, stop) | range(start, stop, step) —
            # Python semantics; the loop var is a scalar.
            if len(loop_args) == 1:
                start, extent, step = 0, self._resolve_loop_bound(loop_args[0]), 1
            elif len(loop_args) == 2:
                start = self._resolve_loop_bound(loop_args[0])
                extent = self._resolve_loop_bound(loop_args[1])
                step = 1
            elif len(loop_args) == 3:
                start = self._resolve_loop_bound(loop_args[0])
                extent = self._resolve_loop_bound(loop_args[1])
                step = self._resolve_loop_bound(loop_args[2])
            else:
                raise VerifyError(
                    f"range() takes 1-3 arguments (stop | start, stop[, step]), "
                    f"got {len(loop_args)}"
                )
            iv_binding = iv
        else:  # tile — `tile(extent)` scalar iv; `tile(extent, step)` RangeSlice
            start = 0
            if len(loop_args) == 1:
                extent = self._resolve_loop_bound(loop_args[0])
                step = 1
                iv_binding = iv
            elif len(loop_args) == 2:
                extent = self._resolve_loop_bound(loop_args[0])
                step = self._resolve_loop_bound(loop_args[1])
                iv_binding = RangeSlice(induction_var=iv, extent=extent, step=step)
            else:
                raise VerifyError(
                    f"tile() takes 1 or 2 arguments (extent[, step]), got {len(loop_args)}"
                )
        if not (_is_dim_expr(start) and _is_dim_expr(extent) and _is_dim_expr(step)):
            raise VerifyError(
                f"{loop_kind}(): start / extent / step must be a dim expression "
                f"(int / DimVar / dim-op Expr), got start={start!r}, "
                f"extent={extent!r}, step={step!r}"
            )

        # Pre-scan body for outer-scope rebindings → carry candidates, in
        # first-occurrence order. Recurse into nested ``for`` loops: a name
        # bound in outer scope but rebound only inside a nested loop is still
        # carried across THIS loop (the nested loop carries it too, chaining).
        carry_names: list[str] = []
        carry_seen: set[str] = set()

        def _scan_carries(body_stmts: list[ast.stmt]) -> None:
            for stmt in body_stmts:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                        and isinstance(stmt.targets[0], ast.Name):
                    name = stmt.targets[0].id
                    if name not in carry_seen:
                        carry_seen.add(name)
                        if isinstance(self.env.lookup(name), Expr):
                            carry_names.append(name)
                elif isinstance(stmt, ast.For):
                    _scan_carries(stmt.body)

        _scan_carries(node.body)

        # Build phi vars (typed from the outer Expr) for each carry; the outer
        # binding is that carry's initial value, stored as the loop's init_arg.
        phi_vars: list[Var] = []
        init_exprs: list[Expr] = []
        for name in carry_names:
            outer_expr = self.env.lookup(name)
            phi_vars.append(Var(type=outer_expr.type, name=name))
            init_exprs.append(outer_expr)

        # Visit body in a pushed frame with iv + phi vars defined.
        self.env.push_frame()
        try:
            self.env.define(node.target.id, iv_binding)
            for cname, phi in zip(carry_names, phi_vars):
                self.env.define(cname, phi)
            body_expr = self._visit_grid_body(node.body)
            # Snapshot final binding for each carry name — this is the
            # iteration's yield_value for that carry slot.
            yield_exprs: list[Expr] = []
            for cname in carry_names:
                v = self.env.lookup(cname)
                if not isinstance(v, Expr):
                    raise VerifyError(
                        f"tile-for: carry name {cname!r} did not resolve to "
                        f"an Expr at end of body (got {type(v).__name__})"
                    )
                yield_exprs.append(v)
        finally:
            self.env.pop_frame()

        if not carry_names:
            return GridRegionExpr(
                type=body_expr.type,
                induction_var=iv,
                carried_args=(),
                init_args=(),
                body=body_expr,
                yield_values=(),
                start=start,
                extent=extent,
                step=step,
            )

        # Carry-out path: single yield → GridRegionExpr.type = phi.type
        # (matches outer var); multi-yield → TupleType.
        if len(carry_names) == 1:
            grid_type = phi_vars[0].type
        else:
            grid_type = TupleType(fields=tuple(p.type for p in phi_vars))
        grid = GridRegionExpr(
            type=grid_type,
            induction_var=iv,
            carried_args=tuple(phi_vars),
            init_args=tuple(init_exprs),
            body=body_expr,
            yield_values=tuple(yield_exprs),
            start=start,
            extent=extent,
            step=step,
        )
        # Rebind carry names in the current frame to the post-loop value.
        if len(carry_names) == 1:
            self.env.define(carry_names[0], grid)
        else:
            # Multi-carry: project each via TupleGetItem.
            for i, cname in enumerate(carry_names):
                proj = self._build_call(TupleGetItem(index=i), (grid,))
                self.env.define(cname, proj)
        return grid

    def _visit_grid_body(self, body_stmts: list[ast.stmt]) -> Expr:
        """Process tile-for body statements and return the final body Expr.

        Body must be a sequence of Assigns (single Name or Tuple targets);
        ``return`` / bare expression statements / for / with are rejected.
        Returns the last bound RHS Expr (or first Tuple-unpack RHS if the
        last stmt is a Tuple unpack).
        """
        last_expr: Expr | None = None
        for stmt in body_stmts:
            if isinstance(stmt, ast.Return):
                raise VerifyError(
                    "hir tile-for body must not contain `return` "
                    "(use a final assignment to the carry variable instead)"
                )
            if isinstance(stmt, ast.Expr):
                raise VerifyError(
                    "hir tile-for body: bare expression statement not allowed"
                )
            if isinstance(stmt, ast.With):
                raise VerifyError(
                    "hir tile-for body: nested With not supported in v1"
                )
            if isinstance(stmt, ast.For):
                # Nested loop → nested GridRegion. Build it (rebinding its own
                # carries in the current frame); its grid value is this stmt's
                # value. The outer carry-lifting already saw any outer-scope
                # names this inner loop rebinds via the outer body pre-scan.
                last_expr = self._build_grid_for(stmt)
                continue
            if isinstance(stmt, ast.AugAssign):
                raise VerifyError(
                    "hir tile-for body: augmented assignment (+= etc.) "
                    "not supported in v1; rewrite as `x = add(x, ...)`"
                )
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1:
                    raise VerifyError(
                        "hir tile-for body: only single-target assignments "
                        "supported in v1"
                    )
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    rhs = self.expr(stmt.value)
                    rhs = self._maybe_autofill_loc(rhs, target.id)
                    self.env.define(target.id, rhs)
                    last_expr = rhs
                    continue
                if isinstance(target, ast.Tuple):
                    # Reuse the chain-style tuple unpack via a one-off helper:
                    # synthesize an inline mini-chain so the existing logic
                    # in `_visit_chain` can run for tuple targets.
                    rhs = self._visit_tuple_assign(target, stmt.value)
                    last_expr = rhs
                    continue
                raise VerifyError(
                    "hir tile-for body: assignment target must be Name or Tuple"
                )
            if isinstance(stmt, ast.AnnAssign):
                last_expr = self._record_annotated_assignment(stmt)
                continue
            raise VerifyError(
                f"hir tile-for body: unsupported statement {type(stmt).__name__}"
            )
        if last_expr is None:
            raise VerifyError(
                "hir tile-for body must contain at least one assignment"
            )
        return last_expr

    def _visit_tuple_assign(self, target: ast.Tuple, value: ast.AST) -> Expr:
        """Tuple-unpack inside tile body (mirrors _visit_chain Tuple branch)."""
        rhs = self.expr(value)
        if not isinstance(rhs.type, TupleType):
            raise VerifyError(
                f"hir: tuple unpack requires RHS of TupleType, "
                f"got {type(rhs.type).__name__}"
            )
        names: list[str] = []
        for elt in target.elts:
            if not isinstance(elt, ast.Name):
                raise VerifyError(
                    "hir: tuple unpack targets must all be plain names"
                )
            names.append(elt.id)
        if len(names) != len(rhs.type.fields):
            raise VerifyError(
                f"hir: tuple unpack arity mismatch — RHS has "
                f"{len(rhs.type.fields)} fields, LHS binds {len(names)} names"
            )
        rhs = self._maybe_autofill_loc_default(rhs)
        last_item: Expr = rhs
        for i, nm in enumerate(names):
            item = self._build_call(TupleGetItem(index=i), (rhs,))
            item = self._maybe_autofill_loc(item, nm)
            self.env.define(nm, item)
            last_item = item
        return last_item

    def _visit_with(self, node: ast.With, stmts, idx, require_return: bool = True):
        if len(node.items) != 1:
            raise VerifyError("hir: only single-item `with` supported")
        item = node.items[0]
        if item.optional_vars is None or not isinstance(item.optional_vars, ast.Name):
            raise VerifyError("hir: `with Mesh(...) as name` requires a single Name binding")

        # Resolve Mesh, supporting string topology-name resolution.
        #   with Mesh(topology="cta", layout=...) as cta_mesh:
        #       → "cta" resolved through topo_ns → Topology object
        #   with Mesh(topology=cta, layout=...) as cta_mesh:
        #       → cta resolved through env/closure → Topology object (legacy)
        mesh = self._resolve_mesh_context(item.context_expr)
        if not isinstance(mesh, Mesh):
            raise VerifyError(
                f"hir: `with` context must evaluate to a Mesh (§3.4), got {type(mesh).__name__}"
            )
        name = item.optional_vars.id
        # hir `with Mesh(...) as m` is an active mesh context (parser.md §1.6):
        # a parser-lexical alias, not a tensor-binding scope and no IR node.
        #
        # The mesh binding name `m` is suite-local: it lives in a frame pushed
        # for the suite and is dropped at the end of the block, so a reference
        # to it after the `with` is `undefined name`. Ordinary values assigned
        # in the suite keep normal function-body visibility: they are hoisted
        # out of the suite frame into the parent before the post-`with` tail is
        # folded. A `return` inside the suite is the function result (a trailing
        # unreachable Python guard is ignored); a setup-only suite falls through
        # to the tail.
        self.env.push_frame()
        try:
            self.env.define(name, mesh)
            body_result = self._visit_chain(list(node.body), 0, require_return=False)
        finally:
            suite_frame = self.env.pop_frame()
        for bound_name, bound_value in suite_frame.items():
            if bound_name != name:
                self.env.define(bound_name, bound_value)
        if body_result is not None:
            return body_result
        return self._visit_chain(stmts, idx + 1, require_return)

    def _resolve_mesh_context(self, node: ast.AST) -> Mesh:
        """Resolve a ``Mesh(...)`` call node, supporting string topology-name lookup
        and tuple layout sugar.

        Handles:
        - ``Mesh(topology="cta", layout=(128,))`` → sugar layout tuple
        - ``Mesh("cta", (128,))`` → positional sugar
        - ``Mesh(topology="cta", layout=Layout(...))`` → verbose form
        - ``Mesh(topology=<Topology obj>, layout=...)`` → legacy closure/env path
        """
        if not isinstance(node, ast.Call):
            return self._eval_static(node)

        # Helper: evaluate a single arg, detecting tuple layout sugar.
        # Sugar parsing only applies to the ``layout=`` slot (or the
        # positional layout slot, handled separately) — other tuple
        # kwargs like ``names=("x", "y")`` are plain static tuples.
        def _eval_mesh_arg(arg_node: ast.AST, *, is_layout_slot: bool = True):
            if is_layout_slot and _is_tuple_sugar(arg_node):
                return parse_mesh_layout_sugar(arg_node, closure=self.closure)
            return self._eval_static(arg_node)

        def _resolve_string_topology(name: str) -> object:
            obj = self.topo_ns.get(name)
            if obj is None:
                raise VerifyError(
                    f"topology {name!r} not declared in function/module topologies "
                    f"(available: {list(self.topo_ns.keys())})"
                )
            return obj

        def _eval_topology_node(arg_node: ast.AST) -> object:
            """Resolve a topology arg AST. Bare string constants look
            up the function-level topo_ns; tuples of string constants
            resolve to a tuple of Topology objects (multi-topology
            mesh sugar — parser owns topology resolution, not Mesh
            post_init)."""
            if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
                return _resolve_string_topology(arg_node.value)
            if isinstance(arg_node, ast.Tuple) and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str)
                for e in arg_node.elts
            ):
                return tuple(
                    _resolve_string_topology(e.value) for e in arg_node.elts
                )
            return self._eval_static(arg_node)

        # Detect topology from keyword or positional
        topo_arg = None
        topo_resolved = False
        for kw in node.keywords:
            if kw.arg == "topology":
                topo_arg = _eval_topology_node(kw.value)
                topo_resolved = True
                break

        if topo_arg is None and not topo_resolved:
            # Positional topology — first positional arg, may be string,
            # tuple-of-strings, or Topology obj.
            if node.args:
                topo_arg = _eval_topology_node(node.args[0])
                topo_resolved = True

        if topo_resolved:
            # Build remaining args/kwargs and inject the resolved topology.
            mesh_fn = self._eval_static(node.func)
            pos_args: list = [topo_arg]
            for i, a in enumerate(node.args[1:], start=1):
                pos_args.append(_eval_mesh_arg(a) if i == 1 else self._eval_static(a))
            pos_kw = {
                k.arg: _eval_mesh_arg(k.value, is_layout_slot=(k.arg == "layout"))
                for k in node.keywords
                if k.arg != "topology"
            }
            # If topology came from kwarg, drop positional[0] (which was
            # not actually present in node.args).
            if not node.args:
                pos_kw["topology"] = topo_arg
                pos_args = []
            return mesh_fn(*pos_args, **pos_kw)

        if topo_arg is None:
            # No topology specified — fall through to the generic
            # positional eval path below for legacy callers.
            mesh_fn = self._eval_static(node.func)
            pos_args = [_eval_mesh_arg(a) if i == 1 else self._eval_static(a)
                        for i, a in enumerate(node.args)]
            pos_kw = {
                k.arg: _eval_mesh_arg(k.value, is_layout_slot=(k.arg == "layout"))
                for k in node.keywords
            }
            return mesh_fn(*pos_args, **pos_kw)

        # Defensive fallthrough — should not be hit given the
        # exhaustive branches above. Evaluate everything statically
        # and let Mesh(...) raise if shape mismatches.
        mesh_fn = self._eval_static(node.func)
        pos_args = [_eval_mesh_arg(a) if i == 1 else self._eval_static(a)
                    for i, a in enumerate(node.args)]
        pos_kw = {k.arg: _eval_mesh_arg(k.value) for k in node.keywords}
        return mesh_fn(*pos_args, **pos_kw)

    def visit_Call(self, node: ast.Call) -> Expr:
        return self.call_to_op_call(node)

    def visit_Name(self, node: ast.Name) -> Expr:
        val = self.env.lookup(node.id)
        if val is None:
            val = self.closure.get(node.id)
        if val is None:
            raise VerifyError(f"undefined name {node.id!r}")
        if isinstance(val, Expr):
            return val
        if isinstance(val, (int, float, bool)):
            return _constant_from_py(val)
        _warn_if_ir_object(val, node.id)
        raise VerifyError(f"name {node.id!r} resolved to non-Expr Python value {type(val).__name__}")

def _is_module_decorator(dec: ast.AST) -> bool:
    """True for ``@module``, ``@tf.module``, or ``@module(entry=...)`` — the
    bare name/attribute form or the entry-carrying call form."""
    if isinstance(dec, ast.Call):
        dec = dec.func
    return (isinstance(dec, ast.Name) and dec.id == "module") or (
        isinstance(dec, ast.Attribute) and dec.attr == "module"
    )


def parse_script(src: str) -> Function:
    """Parse Python DSL source into a HIR Function.

    Auto-detects the source format:

    - ``@module(entry=...) class M: ... @func def fn(...): ...`` → parses with
      mesh symbol table (sugar axis resolution).
    - ``@func def fn(...): ...`` → standalone function parse.

    Args:
        src: Python DSL source string.

    Returns:
        The parsed ``hir.Function``.
    """
    tree = ast.parse(src)
    # Check for @module class
    has_module = False
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            if any(_is_module_decorator(dec) for dec in stmt.decorator_list):
                has_module = True
        if has_module:
            break
    if has_module:
        return _parse_module_source_tree(tree)
    return parse_func_source(src)

# backward-compat aliases
def parse_module_source(src: str) -> Function:
    """Backward-compat alias for parse_script with @module source."""
    return parse_script(src)

def _parse_module_source_tree(tree: ast.Module) -> Function:
    """Parse ``@module``-wrapped AST into a HIR Function."""
    module_cls = None
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            if any(_is_module_decorator(dec) for dec in stmt.decorator_list):
                module_cls = stmt
        if module_cls:
            break

    if module_cls is None:
        raise ValueError("no @module-decorated class found in source")

    # Build closure from module-level and class-level non-@func statements
    closure: dict[str, Any] = {}
    prelude_lines: list[str] = []
    func_node: ast.FunctionDef | None = None

    # Collect module-level prelude (imports) and class-level statements
    all_statements = list(tree.body) + list(module_cls.body)

    def _collect_prelude(stmts):
        nonlocal func_node
        for stmt in stmts:
            if isinstance(stmt, ast.ClassDef):
                continue  # skip the module class itself
            if isinstance(stmt, ast.FunctionDef):
                for dec in stmt.decorator_list:
                    if _is_func_decorator(dec):
                        func_node = stmt
                        return  # stop collecting after finding @func
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    name = alias.asname or alias.name
                    prelude_lines.append(f"import {alias.name} as {name}")
            elif isinstance(stmt, ast.ImportFrom):
                names = ", ".join(
                    a.name + (f" as {a.asname}" if a.asname else "")
                    for a in stmt.names
                )
                prelude_lines.append(f"from {stmt.module or ''} import {names}")
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                prelude_lines.append(ast.unparse(stmt))

    _collect_prelude(all_statements)

    if func_node is None:
        raise ValueError("no @func-decorated method found in module class")

    exec("\n".join(prelude_lines), closure)

    parsed_topologies = _extract_topologies_from_decorator(func_node)
    return _parse_func_node(func_node, closure, topologies=parsed_topologies)

__all__ = ["parse_func", "parse_func_source", "parse_module_source", "parse_script"]
