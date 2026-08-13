from __future__ import annotations

import ast
import dataclasses
from typing import Any

from tilefoundry.ir.constraints import (
    ConstraintProvenance,
    LayoutConstraint,
    MeshConstraint,
    ScheduleConstraint,
    ScheduleConstraintMetadata,
    SourceLocation,
    StorageConstraint,
)
from tilefoundry.ir.constraints.layout import _LAYOUT_WILDCARD
from tilefoundry.ir.core import BindingMetadata, Expr, Var, VerifyError, get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
    is_dim_expr,
    simplify_dim,
)
from tilefoundry.ir.types.shard import Broadcast, Layout, Mesh, Partial, Split, Topology
from tilefoundry.utils.spec_ref import spec_ref_render

from .base import (
    BaseExprVisitor,
    _build_params,
    _collect_closure,
    _resolve_tensor_type,
    extract_ast,
)
from .sugar import _is_tuple_sugar, parse_mesh_layout_sugar
from .symtab import LexicalEnv

_HIR_FUNCTION = "[hir §1.1](docs/spec/hir.md#11-function)"
_PARSER_MESH = "[parser §1.6](docs/spec/parser.md#16-with-mesh-as-m)"



_AST_DIM_OPS = {
    ast.Add: DimAdd,
    ast.Sub: DimSub,
    ast.Mult: DimMul,
    ast.FloorDiv: DimFloorDiv,
    ast.Mod: DimMod,
}


def parse_func(fn, *, topologies=(), specializations=(), extra_closure=None) -> Function:
    """@tilefoundry.func parser entry. Parse fn's source into hir.Function.

    ``topologies`` is the declared topology namespace this body may name in
    ``Mesh(("...",), ...)``; it is a parse-time namespace, not state the
    resulting Function keeps. ``extra_closure`` adds names to the resolution
    namespace below ``fn``'s own globals/freevars; it lets an ``@func`` defined
    in a ``@module`` class body resolve sibling ``@func`` methods (which are
    ``hir.Function`` values) as nested-call targets.
    """
    return _parse_func(
        fn, topologies=topologies, specializations=specializations,
        extra_closure=extra_closure,
    )


def _parse_func(
    fn, *, topologies=(), specializations=(), extra_closure=None, in_module_body=False
) -> Function:
    """`parse_func` plus whether a ``@module`` class body is being authored.

    Only the decorators can answer that, and only they may state it: it decides
    whether a name bound to a ``Module`` is a callee at all.
    """
    node = extract_ast(fn)
    closure = _collect_closure(fn, extra_closure)
    return _parse_func_node(
        node, closure, topologies=topologies,
        specializations=specializations,
        source_filename=getattr(getattr(fn, "__code__", None), "co_filename", "<string>"),
        in_module_body=in_module_body,
    )



_NOT_STATIC = object()

def _parse_func_node(
    node: ast.FunctionDef,
    closure: dict[str, Any],
    *,
    topologies=(),
    specializations=(),
    source_filename: str = "<string>",
    in_module_body: bool = False,
) -> Function:
    env = LexicalEnv()
    params = _build_params(
        node, closure, _resolve_tensor_type, decorator_name="@tilefoundry.func"
    )
    for p in params:
        env.define(p.name, p)

    topo_ns: dict[str, "Topology"] = {}
    for t in topologies:
        if t.name in topo_ns:
            raise VerifyError(f"duplicate topology name {t.name!r}")
        topo_ns[t.name] = t
    visitor = _HirBodyVisitor(
        env, closure, topo_ns=topo_ns, source_filename=source_filename,
        in_module_body=in_module_body,
    )
    if _is_pass_body(node.body):


        body_expr = None
    else:
        body_expr = visitor.visit_body(node.body)
    return_type = _resolve_return_type(node, closure, body_expr)
    return Function.build(
        name=node.name,
        params=params,
        body=body_expr,
        return_type=return_type,
        specializations=tuple(specializations),
    )


def _constraint_value(node: ast.AST):
    """Read a stage-neutral layout extent or topology reference."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node)
    raise VerifyError(
        f"where layout extent must be a literal or symbolic name, got "
        f"{type(node).__name__}"
    )


def _parse_partial_value(node: ast.AST) -> Partial:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise VerifyError('Partial binding must use P("reduction")')
    if node.func.id != "P" or len(node.args) != 1 or node.keywords:
        raise VerifyError('Partial binding must use P("reduction")')
    value = _constraint_value(node.args[0])
    if not isinstance(value, str) or not value:
        raise VerifyError("partial reduction must be a non-empty string")
    return Partial(value)


def _parse_binding_set(node: ast.AST) -> list[tuple[str, Broadcast | Partial]]:
    if not isinstance(node, ast.Set):
        raise VerifyError("layout bindings must be a set")
    out: list[tuple[str, Broadcast | Partial]] = []
    for item in node.elts:
        if not isinstance(item, ast.BinOp) or not isinstance(item.op, ast.MatMult):
            raise VerifyError('layout bindings must use `topology @ B()` or `P()`')
        topology = _constraint_value(item.left)
        if not isinstance(topology, str) or not topology:
            raise VerifyError("layout binding topology must be symbolic")
        if (
            isinstance(item.right, ast.Call)
            and isinstance(item.right.func, ast.Name)
            and item.right.func.id == "B"
            and not item.right.args
            and not item.right.keywords
        ):
            attr: Broadcast | Partial = Broadcast()
        else:
            attr = _parse_partial_value(item.right)
        out.append((topology, attr))
    return out


def _parse_layout_constraint(
    node: ast.AST,
    resolve_extent,
) -> LayoutConstraint:
    if not isinstance(node, ast.Tuple):
        raise VerifyError("layout constraint must be a tuple")
    dims_node = node
    extras: tuple[ast.AST, ...] = ()
    if node.elts and isinstance(node.elts[0], ast.Tuple):
        dims_node = node.elts[0]
        extras = tuple(node.elts[1:])
        if len(extras) > 1:
            raise VerifyError("layout constraint accepts one binding set")
    if not dims_node.elts:
        raise VerifyError("layout constraint cannot be empty")
    shape: list[object] = []
    bindings: list[tuple[str, Split | Broadcast | Partial]] = []
    for index, item in enumerate(dims_node.elts):
        if isinstance(item, ast.Name) and item.id == "_":
            shape.append(_LAYOUT_WILDCARD)
            continue
        if isinstance(item, ast.Name) and item.id == "D":
            raise VerifyError("layout broadcast must use a `{topology @ B()}` binding")
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.MatMult):
            extent = resolve_extent(item.left)
            topology = _constraint_value(item.right)
            if not isinstance(topology, str) or not topology:
                raise VerifyError("layout topology binding must be symbolic")
            shape.append(extent)
            bindings.append((topology, Split(index)))
            continue
        shape.append(resolve_extent(item))
    if extras:
        bindings.extend(_parse_binding_set(extras[0]))
    if len({topology for topology, _ in bindings}) != len(bindings):
        raise VerifyError("layout constraint cannot bind one topology more than once")
    return LayoutConstraint(layout=Layout(shape=tuple(shape)), bindings=tuple(bindings))

def _is_pass_body(stmts: list[ast.stmt]) -> bool:
    """A dispatch-prototype body is exactly ``pass``.

    A dispatch-prototype body is exactly ``pass``. A ``pass`` mixed with any
    other statement is rejected (it is not a partial body form).
    """
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

    t = getattr(body_expr, "type", None)
    if t is None:
        raise VerifyError("@tilefoundry.func: cannot determine return_type")
    return t

class _HirBodyVisitor(BaseExprVisitor):
    token = "hir"
    resolves_module_callees = True

    def __init__(
        self, env, closure, *, topo_ns=None, source_filename="<string>",
        in_module_body=False,
    ):
        super().__init__(env, closure, in_module_body=in_module_body)
        self.topo_ns: dict[str, "Topology"] = topo_ns or {}
        self.source_filename = source_filename





    def visit_body(self, stmts: list[ast.stmt]) -> Expr:
        """Fold an HIR function body into its tail expression DAG.

        Nested function definitions are rejected syntactically across the whole
        body. See [hir §1](docs/spec/hir.md#1-hir-expr-constructs) and
        [parser §5](docs/spec/parser.md#5-hir-parser).
        """
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
                raise VerifyError(
                f"{spec_ref_render(_HIR_FUNCTION)}: @tilefoundry.func return must carry a value"
            )
            if isinstance(node.value, ast.Tuple):
                return self._tuple_expr_expr(node.value)
            return self.expr(node.value)
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise VerifyError("hir: only single-target assignments supported in V1")
            target = node.targets[0]
            if isinstance(target, ast.Name):
                tgt = target.id
                bound = self._static_body_value(node.value)
                if bound is not _NOT_STATIC:


                    self.env.define(tgt, bound)
                    return self._visit_chain(stmts, idx + 1, require_return)
                rhs = self._assignment_rhs(node.value, tgt)

                self.env.define(tgt, rhs)
                return self._visit_chain(stmts, idx + 1, require_return)
            if isinstance(target, ast.Tuple):





                if not self._static_tuple_assign(target, node.value):
                    self._visit_tuple_assign(target, node.value)
                return self._visit_chain(stmts, idx + 1, require_return)
            raise VerifyError("hir: only single-target Name or Tuple assignments supported in V1")
        if isinstance(node, ast.AnnAssign):
            return self._visit_annotated_assignment(node, stmts, idx, require_return)
        if isinstance(node, ast.With):
            return self._visit_with(node, stmts, idx, require_return)
        if isinstance(node, ast.Expr):



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
            raise VerifyError(
                "where annotation target must be a bound plain Name; "
                "subscripts are not annotation lvalues"
            )
        if node.value is None:
            target = self.env.lookup(node.target.id)
            if not isinstance(target, Expr):
                raise VerifyError(
                    f"where annotation target {node.target.id!r} is unresolved "
                    f"at {self.source_filename}:{node.lineno}:{node.col_offset}"
                )
        else:
            target = self._assignment_rhs(node.value, node.target.id)
            self.env.define(node.target.id, target)
        self._record_annotated_assignment(node, target)
        return self._visit_chain(stmts, idx + 1, require_return)

    def _record_annotated_assignment(self, node: ast.AnnAssign, target: Expr) -> None:
        metadata = self._parse_where_annotation(node.annotation, node)
        if not isinstance(target.type, TensorType):
            binding = get_metadata(target, BindingMetadata)
            label = binding.name if binding is not None else self.source_filename
            raise VerifyError(
                f"where annotation requires a tensor-valued Expr at "
                f"{label}:{node.lineno}:{node.col_offset + 1}"
            )
        previous_metadata = get_metadata(target, ScheduleConstraintMetadata)
        if previous_metadata is not None:
            previous = previous_metadata.source_loc.describe()
            current = metadata.source_loc.describe()
            binding = get_metadata(target, BindingMetadata)
            label = binding.name if binding is not None else "<unnamed>"
            raise VerifyError(
                f"duplicate where annotation for Expr {label!r} "
                f"at {current}; first annotation at {previous}"
            )
        self._attach_metadata(target, metadata)

    def _assignment_rhs(self, node: ast.AST, target_name: str) -> Expr:
        """Parse an assignment RHS without turning a name alias into a node."""
        if isinstance(node, ast.Name):
            value = self.env.lookup(node.id)
            if isinstance(value, Expr):
                return value
        return self._maybe_autofill_binding(
            self.expr_with_binding(node, target_name), target_name
        )

    def _parse_where_annotation(
        self, annotation: ast.AST, node: ast.AnnAssign
    ) -> ScheduleConstraintMetadata:
        if not isinstance(annotation, ast.Call) or not isinstance(
            annotation.func, ast.Name
        ) or annotation.func.id != "where":
            raise VerifyError(
                "annotations must use `where(...)`; positional or other forms "
                "are not supported"
            )
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
        constraints: list[ScheduleConstraint] = []
        fields: set[str] = set()
        for keyword in annotation.keywords:
            if keyword.arg is None:
                raise VerifyError("where(...) does not accept **kwargs")
            if keyword.arg in fields:
                raise VerifyError(
                    f"where(...) repeats keyword {keyword.arg!r} at "
                    f"{source_loc.describe()}"
                )
            fields.add(keyword.arg)
            if keyword.arg == "layout":
                layout = _parse_layout_constraint(
                    keyword.value, self._resolve_layout_extent
                )
                constraints.append(
                    dataclasses.replace(
                        layout,
                        source_loc=source_loc,
                        provenance=ConstraintProvenance.AUTHOR,
                    )
                )
            elif keyword.arg == "mesh":
                try:
                    mesh = self._eval_static(keyword.value)
                except (TypeError, ValueError, VerifyError) as exc:
                    raise VerifyError(
                        f"where mesh constraint could not be resolved at "
                        f"{source_loc.describe()}: {exc}"
                    ) from exc
                constraints.append(
                    MeshConstraint(
                        mesh=mesh,
                        source_loc=source_loc,
                        provenance=ConstraintProvenance.AUTHOR,
                    )
                )
            elif keyword.arg == "storage":
                try:
                    storage = self._eval_static(keyword.value)
                except (TypeError, ValueError, VerifyError) as exc:
                    raise VerifyError(
                        f"where storage constraint could not be resolved at "
                        f"{source_loc.describe()}: {exc}"
                    ) from exc
                constraints.append(
                    StorageConstraint(
                        storage=storage,
                        source_loc=source_loc,
                        provenance=ConstraintProvenance.AUTHOR,
                    )
                )
            else:
                raise VerifyError(
                    f"where(...) has unknown field {keyword.arg!r}; use "
                    "layout=..., mesh=..., or storage=..."
                )
        return ScheduleConstraintMetadata(
            constraints=tuple(constraints), source_loc=source_loc
        )

    def _resolve_layout_extent(self, node: ast.AST) -> int | DimVar:
        """Resolve one ``where`` shape entry to a concrete ``int``/``DimVar``.

        Resolve one ``where(layout=...)`` shape entry to a concrete
        ``int``/``DimVar`` (the ``_`` wildcard is handled by the caller before
        this is reached). A literal resolves directly; a symbolic name
        resolves through the lexical env / authoring closure, matching
        ``_eval_static``'s ``Name`` resolution.
        """
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, int):
                raise VerifyError(
                    "layout dimensions must use `_`, an integer, or a "
                    "symbolic extent with `@ topology`"
                )
            return value
        if isinstance(node, ast.Name):
            try:
                resolved = self._eval_static(node)
            except (TypeError, ValueError, VerifyError) as exc:
                raise VerifyError(
                    f"where layout extent {node.id!r} could not be resolved: {exc}"
                ) from exc
            if isinstance(resolved, bool) or not isinstance(resolved, (int, DimVar)):
                raise VerifyError(
                    f"where layout extent {node.id!r} must resolve to an "
                    f"int or DimVar, got {type(resolved).__name__}"
                )
            return resolved
        raise VerifyError(
            "layout dimensions must use `_`, an integer, or a symbolic "
            "extent with `@ topology`"
        )

    def _resolve_loop_bound(self, node: ast.AST):
        """Resolve a ``tile`` / ``range`` bound to an ``int``, ``DimVar``, or dim ``Expr``.

        Resolve a ``tile`` / ``range`` bound (extent / step / start) to an
        ``int``, ``DimVar``, or dim ``Expr``.

        Unlike ``_eval_static`` (which only folds numeric constants), a
        ``BinOp`` whose operands reach a ``DimVar`` builds a dim expression via
        ``simplify_dim`` (e.g. ``C // NUM_SPLITS`` → ``DimFloorDiv(C, N)``). The
        IR / evaluator already resolve dim-expression loop bounds; this lets the
        DSL surface write them.
        """
        if isinstance(node, ast.BinOp):
            op = _AST_DIM_OPS.get(type(node.op))
            if op is None:
                raise VerifyError(
                    f"loop bound: unsupported operator {type(node.op).__name__} "
                    f"(use + - * // %)"
                )
            left = self._resolve_loop_bound(node.left)
            right = self._resolve_loop_bound(node.right)


            if isinstance(left, int) and not isinstance(left, bool) and \
                    isinstance(right, int) and not isinstance(right, bool):
                return self._eval_static(node)
            return simplify_dim(op, (left, right))
        return self._eval_static(node)

    def _visit_loop_for(self, node: ast.For, stmts, idx, require_return: bool = True):
        """Lower tile or range loops to a grid region and continue the chain.

        Both forms share start, extent, and step. Tile loops bind a range slice
        for indexed use; range loops bind a scalar induction variable. Neither
        form is unrolled.
        See [parser §1.7](docs/spec/parser.md#17-for-i-in-tile--for-i-in-range-hir-only).
        """
        grid = self._build_grid_for(node)
        if idx + 1 < len(stmts):
            return self._visit_chain(stmts, idx + 1, require_return)




        return grid if require_return else None

    def _build_grid_for(self, node: ast.For) -> Expr:
        """Build a grid region and rebind loop-carried names in this frame.

        Assigning an outer name creates a phi and yield; one carry maps to the
        grid and multiple carries to tuple projections after the loop. Nested
        loops compose recursively. The body accepts assignments and nested loops
        but no return; the caller processes sibling statements.
        See [parser §5.1](docs/spec/parser.md#51-gridregionexpr-carry-out-lifting).
        """
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name):
            raise VerifyError("hir For: iter must be a `tile(...)` or `range(...)` call")
        loop_kind = node.iter.func.id
        if loop_kind not in ("tile", "range"):
            raise VerifyError(
                f"hir For: iter must be `tile(...)` or `range(...)`, got "
                f"{loop_kind!r}"
            )
        if node.iter.keywords:
            raise VerifyError(
                f"{loop_kind}() does not accept keyword args "
                "(positional-only at the IR level)"
            )
        if not isinstance(node.target, ast.Name):
            raise VerifyError("hir For: target must be a Name")
        iv = Var(type=TensorType.scalar(DType.i64), name=node.target.id)

        loop_args = node.iter.args
        iv_binding: Expr | slice
        if loop_kind == "range":


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
        else:
            start = 0
            if len(loop_args) == 2:
                extent = self._resolve_loop_bound(loop_args[0])
                step = self._resolve_loop_bound(loop_args[1])
                step_expr = self._constant_expr(step) if isinstance(step, int) else step
                iv_binding = slice(
                    iv,
                    simplify_dim(DimAdd, (iv, step_expr)),
                    1,
                )
            elif len(loop_args) == 1:
                raise VerifyError(
                    "tile(extent) is not supported; use range(extent) for "
                    "scalar iteration"
                )
            else:
                raise VerifyError(
                    f"tile() takes 2 arguments (extent, step), got {len(loop_args)}"
                )
        if not (is_dim_expr(start) and is_dim_expr(extent) and is_dim_expr(step)):
            raise VerifyError(
                f"{loop_kind}(): start / extent / step must be a dim expression "
                f"(int / DimVar / dim-op Expr), got start={start!r}, "
                f"extent={extent!r}, step={step!r}"
            )





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



        phi_vars: list[Var] = []
        init_exprs: list[Expr] = []
        for name in carry_names:
            outer_expr = self.env.lookup(name)
            phi_vars.append(Var(type=outer_expr.type, name=name))
            init_exprs.append(outer_expr)


        self.env.push_frame()
        if loop_kind == "range":
            self._scalar_index_ids.add(id(iv))
        try:
            self.env.define(node.target.id, iv_binding)
            for cname, phi in zip(carry_names, phi_vars):
                self.env.define(cname, phi)
            body_expr = self._visit_grid_body(node.body)


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
            self._scalar_index_ids.discard(id(iv))
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

        if len(carry_names) == 1:
            self.env.define(carry_names[0], grid)
        else:

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
                    rhs = self._assignment_rhs(stmt.value, target.id)
                    self.env.define(target.id, rhs)
                    last_expr = rhs
                    continue
                if isinstance(target, ast.Tuple):



                    rhs = self._visit_tuple_assign(target, stmt.value)
                    last_expr = rhs
                    continue
                raise VerifyError(
                    "hir tile-for body: assignment target must be Name or Tuple"
                )
            raise VerifyError(
                f"hir tile-for body: unsupported statement {type(stmt).__name__}"
            )
        if last_expr is None:
            raise VerifyError(
                "hir tile-for body must contain at least one assignment"
            )
        return last_expr

    def _static_body_value(self, node: ast.AST):
        """A body-local name's compile-time value.

        A body-local name's compile-time value — a number or a list of Exprs —
        or ``_NOT_STATIC`` when the right-hand side belongs to the IR.
        """
        number = self._static_number(node)
        if number is not None:
            return number
        items = self._static_expr_list(node)
        return _NOT_STATIC if items is None else items

    def _static_tuple_assign(self, target: ast.Tuple, value: ast.AST) -> bool:
        """Bind ``a, b = <compile-time>, <compile-time>``, reporting whether it applied."""
        if not isinstance(value, ast.Tuple) or len(target.elts) != len(value.elts):
            return False
        if not all(isinstance(elt, ast.Name) for elt in target.elts):
            return False
        numbers = [self._static_number(el) for el in value.elts]
        if any(number is None for number in numbers):
            return False
        for elt, number in zip(target.elts, numbers):
            self.env.define(elt.id, number)
        return True

    def _visit_tuple_assign(self, target: ast.Tuple, value: ast.AST) -> Expr:
        """Tuple-unpack inside tile body (mirrors _visit_chain Tuple branch)."""
        names: list[str] = []
        for elt in target.elts:
            if not isinstance(elt, ast.Name):
                raise VerifyError(
                    "hir: tuple unpack targets must all be plain names"
                )
            names.append(elt.id)
        rhs = self.expr_with_binding(value, ", ".join(names))
        if not isinstance(rhs.type, TupleType):
            raise VerifyError(
                f"hir: tuple unpack requires RHS of TupleType, "
                f"got {type(rhs.type).__name__}"
            )
        if len(names) != len(rhs.type.fields):
            raise VerifyError(
                f"hir: tuple unpack arity mismatch — RHS has "
                f"{len(rhs.type.fields)} fields, LHS binds {len(names)} names"
            )
        rhs = self._maybe_autofill_binding_default(rhs)
        last_item: Expr = rhs
        for i, nm in enumerate(names):
            item = self._build_call(TupleGetItem(index=i), (rhs,))
            item = self._maybe_autofill_binding(item, nm)
            self.env.define(nm, item)
            last_item = item
        return last_item

    def _visit_with(self, node: ast.With, stmts, idx, require_return: bool = True):
        """Parse an active mesh context with suite-local mesh binding.

        Ordinary suite bindings escape to the function frame, while the mesh
        alias does not. See [parser §1.6](docs/spec/parser.md#16-with-mesh-as-m).
        """
        if len(node.items) != 1:
            raise VerifyError("hir: only single-item `with` supported")
        item = node.items[0]
        if item.optional_vars is None or not isinstance(item.optional_vars, ast.Name):
            raise VerifyError("hir: `with Mesh(...) as name` requires a single Name binding")


        mesh = self._resolve_mesh_context(item.context_expr)
        if not isinstance(mesh, Mesh):
            raise VerifyError(
                f"hir: `with` context must evaluate to a Mesh "
                f"({spec_ref_render(_PARSER_MESH)}), got {type(mesh).__name__}"
            )
        name = item.optional_vars.id











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
        """Resolve a ``Mesh(...)`` call with tuple topology-name sugar."""
        if not isinstance(node, ast.Call):
            return self._eval_static(node)





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

        if any(keyword.arg in {"topology", "topologies"} for keyword in node.keywords):
            raise VerifyError(
                'hir: Mesh requires a tuple of declared topology names, '
                'for example Mesh(("cta",), layout=(128,))'
            )
        if (
            not node.args
            or not isinstance(node.args[0], ast.Tuple)
            or not node.args[0].elts
            or not all(
            isinstance(entry, ast.Constant) and isinstance(entry.value, str)
            for entry in node.args[0].elts
            )
        ):
            raise VerifyError(
                'hir: Mesh requires a tuple of declared topology names, '
                'for example Mesh(("cta",), layout=(128,))'
            )

        mesh_fn = self._eval_static(node.func)
        topologies = tuple(_resolve_string_topology(entry.value) for entry in node.args[0].elts)
        pos_args = [topologies]
        for index, argument in enumerate(node.args[1:], start=1):
            pos_args.append(_eval_mesh_arg(argument, is_layout_slot=(index == 1)))
        pos_kw = {
            keyword.arg: _eval_mesh_arg(
                keyword.value, is_layout_slot=(keyword.arg == "layout")
            )
            for keyword in node.keywords
        }
        return mesh_fn(*pos_args, **pos_kw)

__all__ = ["parse_func"]
