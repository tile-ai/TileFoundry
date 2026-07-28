"""``tilefoundry.ir.visitor`` — Expr / Stmt visitor + mutator contract.

Every pass is written against these four guarantees, and a break in any of them
shows up in a pass's output rather than at its cause: class-name dispatch, an
untouched child stays *the same object* (so a pass rewriting one node cannot
silently deep-copy the tree), binding-site Vars are never handed to a generic
mutator, and a Stmt subclass missing from the rebuild tables is caught here
instead of dropping statements downstream.
"""

from __future__ import annotations

from tilefoundry.ir.core import Call, Constant, Expr, Op, Var
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.tir.cuda.nn.mma import Mma
from tilefoundry.ir.tir.memory import Copy, Fill
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    For,
    If,
    LetStmt,
    MeshScope,
    Return,
    Sequential,
    While,
)
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import CallableType, DType, TensorType, UnitType
from tilefoundry.ir.types.shard import make_mesh
from tilefoundry.ir.types.shard.mesh import Topology
from tilefoundry.ir.visitor import (
    ExprMutator,
    ExprVisitor,
    StmtExprMutator,
    StmtMutator,
    StmtVisitor,
    rewrite_prim_function,
    walk_prim_function,
)


def _t() -> TensorType:
    return TensorType.scalar(DType.f32)


def _i32() -> TensorType:
    return TensorType.scalar(DType.i32)


class _OpA(Op):
    pass


class _OpB(Op):
    pass


def _var(name: str, t: TensorType | None = None) -> Var:
    return Var(type=t or _t(), name=name)


def _const(v: object, t: TensorType | None = None) -> Constant:
    return Constant(type=t or _t(), value=v)


def _call(op: Op, *args: Expr) -> Call:
    return Call(type=_t(), target=op, args=args)


def _eval_call(op: Op, *args: Expr) -> Evaluate:
    return Evaluate(callable=op, args=args)


# ── ExprVisitor / ExprMutator ─────────────────────────────────────────────


def test_expr_visitor_dispatches_by_class_name_and_shares_unchanged_branches() -> None:
    """``visit_<ClassName>`` dispatch, in child order; a plain ExprMutator is the
    identity; and replacing one Var rebuilds its containing Call while sharing the
    siblings that were not touched."""
    visits = []

    class V(ExprVisitor[None]):
        def visit_Var(self, var):
            visits.append(("Var", var.name))

        def visit_Constant(self, c):
            visits.append(("Constant", c.value))

        def visit_Call(self, call):
            visits.append(("Call", type(call.target).__name__))
            self.generic_visit(call)

    tree = _call(_OpA(), _var("x"), _const(1.0))
    V().visit(tree)
    assert visits == [("Call", "_OpA"), ("Var", "x"), ("Constant", 1.0)]
    assert ExprMutator().visit(tree) is tree

    x = _var("x")
    y = _var("y")
    sub = _call(_OpA(), x, y)
    top = _call(_OpB(), sub, _const(2.0))

    class OnlyReplaceY(ExprMutator):
        def visit_Var(self, var):
            return _var("y2") if var.name == "y" else var

    out = OnlyReplaceY().visit(top)
    assert out is not top
    assert out.args[0] is not sub
    assert out.args[0].args[0] is x  # x unchanged → shared
    assert out.args[1] is top.args[1]


def test_expr_mutator_skips_grid_region_binding_vars() -> None:
    """Binding-site Vars (``induction_var`` / ``carried_args``) are not
    exposed to a generic ExprMutator (would otherwise be type-illegal)."""

    ind = _var("i", _i32())
    carried = (_var("a"), _var("b"))
    init = (_var("a0"), _var("b0"))
    region = GridRegionExpr(
        type=_t(),
        induction_var=ind,
        carried_args=carried,
        init_args=init,
        body=_var("out"),
        yield_values=(_var("y0"), _var("y1")),
        extent=1,
        step=1,
    )

    replaced: list[str] = []

    class ToConst(ExprMutator):
        def visit_Var(self, var):
            replaced.append(var.name)
            return _const(0.0)

    out = ToConst().visit(region)
    assert out.induction_var is ind and out.carried_args is carried
    assert isinstance(out.body, Constant)
    # Binding-site Vars stay untouched; init_args are value children → rewritten.
    assert "i" not in replaced and "a" not in replaced and "b" not in replaced
    assert "a0" in replaced and "b0" in replaced
    assert all(isinstance(e, Constant) for e in out.init_args)


# ── StmtVisitor / StmtMutator ─────────────────────────────────────────────


def _simple_for_body() -> For:
    i = _var("i", _i32())
    body = Sequential(body=(
        _eval_call(Copy(), _var("src"), _var("dst")),
        _eval_call(Fill(), _var("t"), _const(0.0)),
    ))
    return For(
        induction_var=i,
        start=_const(0, _i32()),
        stop=_const(16, _i32()),
        step=_const(1, _i32()),
        body=body,
    )


def test_stmt_walk_stays_in_the_stmt_tree_and_shares_unchanged_siblings() -> None:
    """``StmtVisitor`` walks child Stmts only — embedded Expr fields are NOT
    traversed (use ``StmtExprMutator`` for that) — and replacing one
    ``Evaluate(Copy)`` rebuilds the For body while sharing the untouched
    ``Evaluate(Fill)`` sibling."""
    seen: list[str] = []
    visited_vars: list[str] = []

    class V(StmtVisitor[None]):
        def visit_Evaluate(self, stmt):
            seen.append(type(stmt.callable).__name__)

        def visit_For(self, stmt):
            seen.append("For")
            self.generic_visit(stmt)

        def visit_Var(self, var):  # would only fire if Expr-walk happened
            visited_vars.append(var.name)

    V().visit(_simple_for_body())
    assert seen == ["For", "Copy", "Fill"]
    assert visited_vars == []

    s = _simple_for_body()

    class ReplaceCopy(StmtMutator):
        def visit_Evaluate(self, stmt):
            if isinstance(stmt.callable, Copy):
                return _eval_call(Copy(), _var("new_src"), stmt.args[1])
            return stmt

    out = ReplaceCopy().visit(s)
    assert out is not s
    assert out.body.body[0] is not s.body.body[0]  # Copy replaced
    assert out.body.body[1] is s.body.body[1]      # Fill shared


def test_stmt_mutator_covers_all_subclasses_with_identity_invariant() -> None:
    """Every Stmt subclass round-trips through identity ``StmtMutator``."""
    i = _var("i", _i32())
    binding = _var("mvar")

    def _seq(*items) -> Sequential:
        return Sequential(body=tuple(items))

    stmts = (
        LetStmt(var=_var("y"), value=_call(_OpA(), _var("z")), body=_seq(Return())),
        Return(),
        For(
            induction_var=i,
            start=_const(0, _i32()), stop=_const(8, _i32()), step=_const(1, _i32()),
            body=_seq(_eval_call(Copy(), _var("s"), _var("d"))),
        ),
        While(cond=_var("c"), body=_seq(_eval_call(Copy(), _var("s2"), _var("d2")))),
        If(cond=_var("c2"), then_body=_seq(), else_body=_seq()),
        MeshScope(
            mesh=make_mesh((2,), topology=Topology(name="chip", size=2)),
            binding=binding,
            body=_seq(_eval_call(Copy(), _var("s3"), _var("d3"))),
        ),
        _eval_call(Copy(), _var("s4"), _var("d4")),
        _eval_call(Fill(), _var("t"), _const(0.0)),
        _eval_call(Mma(), _var("L"), _var("R"), _var("A")),
        Sequential(body=()),
    )
    m = StmtMutator()
    for s in stmts:
        assert m.visit(s) is s, f"identity broken on {type(s).__name__}"


# ── StmtExprMutator ──────────────────────────────────────────────────────


def test_stmt_expr_mutator_rewrites_expr_fields_tuples_and_symbolref_leaf() -> None:
    """Rewrites scalar Expr fields (``For.stop``) and tuple Expr fields
    (``Evaluate.args``) with partial-share semantics, and visits an ``Evaluate``
    whose callable is a ``SymbolRef`` (an Expr leaf) without error — guarding
    against a missing SymbolRef branch in the Expr child/rebuild tables."""
    s = _simple_for_body()

    class RewriteConst(StmtExprMutator):
        def visit_Constant(self, c):
            return Constant(type=c.type, value=32) if c.value == 16 else c

    out = RewriteConst().visit_stmt(s)
    assert out.stop.value == 32
    assert out.start is s.start and out.step is s.step

    a, b, c = _var("a"), _var("b"), _var("c")
    ct = CallableType(return_type=UnitType(), parameters=(a.type, b.type, c.type))
    ref = SymbolRef(name="callee", type=ct)
    call_stmt = Evaluate(callable=ref, args=(a, b, c))

    # A no-op mutator preserves identity of the whole Evaluate, SymbolRef included.
    assert StmtExprMutator().visit_stmt(call_stmt) is call_stmt

    class ReplaceB(StmtExprMutator):
        def visit_Var(self, var):
            return _var("b2") if var.name == "b" else var

    out = ReplaceB().visit_stmt(call_stmt)
    assert out.callable is ref          # SymbolRef leaf unchanged → shared
    assert out.args[0] is a and out.args[2] is c
    assert out.args[1] is not b


# ── PrimFunction walk + rewrite ──────────────────────────────────────────


def test_prim_function_walk_and_identity_preserving_rewrite() -> None:
    """``walk_prim_function`` is read-only; ``rewrite_prim_function`` with a
    no-op mutator returns the original PrimFunction."""
    seen: list[str] = []
    s = _simple_for_body()
    pf = PrimFunction(name="foo", params=(), body=Sequential(body=(s,)))

    class V(StmtVisitor[None]):
        def visit_For(self, stmt):
            seen.append("For")

    walk_prim_function(V(), pf)
    assert seen == ["For"]
    assert rewrite_prim_function(StmtMutator(), pf) is pf
