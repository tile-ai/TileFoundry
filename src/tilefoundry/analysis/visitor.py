"""Shared structural facts collected from one normalized HIR graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Expr
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.visitor import ExprVisitor
from tilefoundry.target import Target

from .walk import children


@dataclass
class AnalyzeContext:
    """Per-call inputs and the current shared lexical scope."""

    module: Module
    target: Target
    level: str | None
    options: object | None
    root: "Scope"
    current: "Scope"
    @classmethod
    def create(
        cls,
        module: Module,
        graph: Function,
        target: Target,
        level: str | None = None,
        options: object | None = None,
    ) -> "AnalyzeContext":
        """Build a context and collect one scope/access tree for ``graph``."""
        from .scope import build_scopes  # noqa: PLC0415

        root = build_scopes(module, graph)
        return cls(module, target, level, options, root, root)

    def enter(self, child: "Scope") -> "AnalyzeContext":
        """Return a context focused on one child lexical scope."""
        return type(self)(self.module, self.target, self.level, self.options, self.root, child)


@dataclass(frozen=True, eq=False)
class ScopeMemo:
    """One lexical Function or loop scope in the normalized graph.

    Scope equality and hashing are identity-based. Parent/child links form a
    cycle, so structural equality would recurse and would not describe lexical
    ownership semantics.
    """

    owner: Function | GridRegionExpr
    parent: "ScopeMemo | None"
    children: tuple["ScopeMemo", ...]

    def is_variant(self, value: Expr) -> bool:
        """Whether ``value`` depends on this loop's induction or carry values."""
        if not isinstance(self.owner, GridRegionExpr):
            return False
        seeds = (self.owner.induction_var, *self.owner.carried_args)
        roots = (self.owner.body, *self.owner.yield_values)
        contained: set[int] = set()

        def collect(expr: Expr) -> None:
            key = id(expr)
            if key in contained:
                return
            contained.add(key)
            for operand in children(expr):
                collect(operand)

        for root in roots:
            collect(root)
        if id(value) not in contained:
            return False

        resolved: dict[int, bool] = {}

        def depends(expr: Expr) -> bool:
            key = id(expr)
            if key in resolved:
                return resolved[key]
            if any(expr is seed for seed in seeds):
                resolved[key] = True
                return True
            result = any(depends(operand) for operand in children(expr))
            resolved[key] = result
            return result

        return depends(value)

    def is_invariant(self, value: Expr) -> bool:
        """Whether ``value`` is invariant with respect to this loop scope."""
        return not self.is_variant(value)


@dataclass(frozen=True)
class ExprMemo:
    """Structural facts for one expression definition."""

    expr: Expr
    operands: tuple[Expr, ...]
    users: tuple[Expr, ...]
    definition_index: int
    parent_scope: ScopeMemo


@dataclass(frozen=True)
class StructuralMemo:
    """One identity-preserving memo of the normalized HIR structure."""

    nodes: tuple[ExprMemo, ...]
    scopes: tuple[ScopeMemo, ...]
    _by_id: dict[int, ExprMemo] = field(default_factory=dict, repr=False, compare=False)
    _scope_by_id: dict[int, ScopeMemo] = field(default_factory=dict, repr=False, compare=False)

    def node(self, expr: Expr) -> ExprMemo:
        """Return the memo for ``expr``, matching by object identity."""
        found = self._by_id.get(id(expr))
        if found is None or found.expr is not expr:
            raise KeyError(f"expression is not in this structural memo: {expr!r}")
        return found

    def scope_of(self, expr: Expr) -> ScopeMemo:
        """Return the lexical scope owning ``expr``."""
        return self.node(expr).parent_scope

    def scope(self, owner: Function | GridRegionExpr) -> ScopeMemo:
        """Return the scope introduced by ``owner``, matching by identity."""
        found = self._scope_by_id.get(id(owner))
        if found is None or found.owner is not owner:
            raise KeyError(f"scope owner is not in this structural memo: {owner!r}")
        return found

    def producers(self, expr: Expr) -> tuple[Expr, ...]:
        """Return direct operand definitions of ``expr``."""
        return self.node(expr).operands

    def users(self, expr: Expr) -> tuple[Expr, ...]:
        """Return expressions that directly consume ``expr``."""
        return self.node(expr).users

    def definition_order(self, function: Function) -> tuple[Expr, ...]:
        """Return ``function`` expressions once, operands before consumers."""
        self.scope(function)
        return tuple(item.expr for item in self.nodes)

@dataclass
class _ScopeBuilder:
    owner: Function | GridRegionExpr
    parent: "_ScopeBuilder | None"
    children: list["_ScopeBuilder"] = field(default_factory=list)


class StructuralMemoVisitor(ExprVisitor[None]):
    """Collect facts from a normalized ``check_program`` graph.

    The normalized graph contains no nested Function expression nodes; calls to
    reachable callees are represented by their call arguments and are handled
    by the enclosing graph's ownership rules.
    """

    def __init__(self) -> None:
        super().__init__()
        self._nodes: list[Expr] = []
        self._operands: dict[int, tuple[Expr, ...]] = {}
        self._owners: dict[int, _ScopeBuilder] = {}
        self._scope_stack: list[_ScopeBuilder] = []
        self._root_scope: _ScopeBuilder | None = None
        self._frozen_scopes: dict[int, ScopeMemo] = {}

    def build(self, graph: Function) -> StructuralMemo:
        """Visit ``graph`` and return its immutable structural memo."""
        self.clear()
        self._nodes.clear()
        self._operands.clear()
        self._owners.clear()
        self._scope_stack.clear()
        self._frozen_scopes.clear()
        self._root_scope = _ScopeBuilder(graph, None)
        self._scope_stack.append(self._root_scope)
        for param in graph.params:
            self.visit(param)
        if graph.body is not None:
            self.visit(graph.body)
        self._scope_stack.pop()
        root, scopes = self._freeze_scope(self._root_scope, None)
        del root
        users: dict[int, list[Expr]] = {id(expr): [] for expr in self._nodes}
        for expr in self._nodes:
            for operand in self._operands[id(expr)]:
                users.setdefault(id(operand), []).append(expr)
        memo_nodes = tuple(
            ExprMemo(
                expr=expr,
                operands=self._operands[id(expr)],
                users=tuple(users[id(expr)]),
                definition_index=index,
                parent_scope=self._frozen_scopes[id(self._owners[id(expr)].owner)],
            )
            for index, expr in enumerate(self._nodes)
        )
        return StructuralMemo(
            nodes=memo_nodes,
            scopes=scopes,
            _by_id={id(item.expr): item for item in memo_nodes},
            _scope_by_id={id(scope.owner): scope for scope in scopes},
        )

    def _record(self, expr: Expr, operands: tuple[Expr, ...]) -> None:
        key = id(expr)
        if key in self._owners:
            return
        self._owners[key] = self._scope_stack[-1]
        self._operands[key] = operands
        self._nodes.append(expr)

    def visit_Call(self, expr):
        operands = children(expr)
        for operand in operands:
            self.visit(operand)
        self._record(expr, operands)

    def visit_Tuple(self, expr):
        operands = children(expr)
        for operand in operands:
            self.visit(operand)
        self._record(expr, operands)

    def visit_GridRegionExpr(self, expr):
        outer = self._scope_stack[-1]
        init_args = tuple(expr.init_args)
        for operand in init_args:
            self.visit(operand)
        child = _ScopeBuilder(expr, outer)
        outer.children.append(child)
        self._scope_stack.append(child)
        self.visit(expr.body)
        for operand in expr.yield_values:
            self.visit(operand)
        self._scope_stack.pop()
        self._record(expr, (*init_args, expr.body, *expr.yield_values))

    def visit_Function(self, expr):
        raise TypeError("StructuralMemoVisitor expects a normalized Function graph without nested Function nodes")

    def visit_Var(self, expr):
        self._record(expr, ())

    def visit_Constant(self, expr):
        self._record(expr, ())

    def visit_SymbolRef(self, expr):
        self._record(expr, ())

    def visit_ShapeOf(self, expr):
        self._record(expr, ())

    def default_visit(self, expr):
        operands = children(expr)
        for operand in operands:
            self.visit(operand)
        self._record(expr, operands)

    def _freeze_scope(
        self, builder: _ScopeBuilder, parent: ScopeMemo | None
    ) -> tuple[ScopeMemo, tuple[ScopeMemo, ...]]:
        scope = ScopeMemo(builder.owner, parent, ())
        self._frozen_scopes[id(builder.owner)] = scope
        children_memo: list[ScopeMemo] = []
        all_scopes: list[ScopeMemo] = [scope]
        for child in builder.children:
            frozen, descendants = self._freeze_scope(child, scope)
            children_memo.append(frozen)
            all_scopes.extend(descendants)
        object.__setattr__(scope, "children", tuple(children_memo))
        return scope, tuple(all_scopes)


__all__ = [
    "AnalyzeContext",
    "ExprMemo",
    "ScopeMemo",
    "StructuralMemo",
    "StructuralMemoVisitor",
]
