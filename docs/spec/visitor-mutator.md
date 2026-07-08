# TileFoundry Spec — IR Visitor / Mutator

The IR traversal / rewrite framework: `ExprVisitor[T]` /
`ExprMutator` / `StmtVisitor[T]` / `StmtMutator` / `StmtExprMutator`.
A shared compiler facility, owned by neither analysis nor any
specific pass.

```mermaid
flowchart TB
    ExprVisitor["<b>ExprVisitor[T]</b><br/>read-only"]
    ExprMutator["<b>ExprMutator</b><br/>identity-preserving rewrite"]
    StmtVisitor["<b>StmtVisitor[T]</b>"]
    StmtMutator["<b>StmtMutator</b>"]
    StmtExprMutator["<b>StmtExprMutator</b><br/>Stmt + embedded Expr rewrite"]

    StmtMutator -. "MRO" .-> ExprMutator
    StmtExprMutator --> StmtMutator
    StmtExprMutator --> ExprMutator
```

## 1. Role

Visitors and mutators are the standard base classes for **recursive
IR traversal and recursive IR rewrite**. Logic that needs to "walk
the IR and collect" or "walk the IR and emit new IR" inherits from
these classes; manual `isinstance` dispatch is not the convention.

- **Visitor** — read-only traversal. Returns a user-defined `T`.
  Used for aggregation / collection / verification.
- **Mutator** — recursive rewrite. Returns the same node kind
  (`Expr → Expr`, `Stmt → Stmt`). Used for lowering / simplification
  / structural rewrite.

The base classes only define traversal scaffolding. Business logic
is injected by overriding `visit_<ClassName>` in subclasses.

## 2. Dispatch convention

`visit_<ClassName>` static dispatch (no `singledispatch`):

- `visit_Call(self, call: Call) -> T` for `Call`,
- `visit_Var(self, var: Var) -> T` for `Var`,
- `visit_For(self, stmt: For) -> Stmt` for the `For` Stmt subclass,
- `visit_Evaluate(self, stmt: Evaluate) -> Stmt` for `Evaluate`,
- and so on.

`visit(node)` looks up `visit_<type(node).__name__>` on the subclass and calls
it, falling back to `generic_visit(node)` when no such override exists.

- **Most-specific wins.** `Call` is a subclass of `Expr`, but with
  both `visit_Call` and `visit_Expr` defined, `visit_Call` wins —
  dispatch keys on the runtime class name.
- **Fallback to `generic_visit(node)`.** Default behaviour is
  recursive traversal of all child nodes.
- **No Op-level dispatch lives here.** A `Call(target=Add)` is
  caught by `visit_Call`; per-Op dispatch is the analysis registry's
  responsibility ([visitor-registry](./visitor-registry.md)).

`visit_<ClassName>` is preferred over `singledispatch` because
overrides are greppable, the inheritance chain is explicit, and it
does not depend on Python-version-specific dispatch behaviour.

## 3. `ExprVisitor[T]`

Read-only Expr-tree traversal. `T` is user-chosen (`None` for
side-effect collection, `set[Var]` for free-var analysis, etc.).

```python
class ExprVisitor(Generic[T]):                     # read-only Expr traversal; T is the user-chosen visit result type
    def visit(self, expr: Expr) -> T: ...          # dispatch to visit_<Type> else generic_visit
    def generic_visit(self, expr: Expr) -> T: ...  # recurse child Exprs; returns None by default
```

- constraints:
  - `generic_visit` recurses into all child Exprs and returns `None` by default;
    subclasses MAY override to aggregate.

`_expr_children(expr)` enumerates child Exprs of any Expr node by
fixed field order. The mapping is module-local and is the single
table that grows whenever a new Expr subclass appears:

| Node | Child Exprs |
|---|---|
| `Var` | `()` |
| `Constant` | `()` |
| `Tuple` | `fields` |
| `Call` | `args` |

Example — collect every `Var`:

```python
class VarCollector(ExprVisitor[None]):
    def __init__(self) -> None:
        self.vars: set[Var] = set()
    def visit_Var(self, var: Var) -> None:
        self.vars.add(var)
```

## 4. `ExprMutator`

Recursive Expr rewrite returning the same node kind. Core invariant:
**when no child changed, return the original node** (identity
preservation).

```python
class ExprMutator:                                    # identity-preserving Expr rewrite
    def visit(self, expr: Expr) -> Expr: ...          # dispatch to visit_<Type> else generic_visit
    def generic_visit(self, expr: Expr) -> Expr: ...  # recurse children; return original node when no child changed (identity preservation)
```

- constraints:
  - When no child changed, `generic_visit` returns the original node (identity
    preservation).

`_rebuild_expr(expr, new_children)` constructs a new Expr of the
same subclass while preserving non-child fields (`type`, `source`).

Identity preservation matters for three reasons:

1. **Structural sharing.** Untouched subtrees stay shared with the
   input IR; downstream passes avoid rebuilding equivalent state.
2. **Change detection.** A pass can decide whether to retrigger
   downstream work via `new_expr is old_expr`.
3. **Cache validity.** `typeinfer` / cost caches keyed on Expr
   identity remain valid for unchanged nodes.

A `visit_Call` override MUST itself respect identity preservation:
the unchanged branch routes through `generic_visit(call)` rather
than `return call`, so child Exprs are still recursed.

## 5. `StmtVisitor[T]` / `StmtMutator`

Same shape as the Expr family, but for the Stmt tree.

```python
class StmtVisitor(Generic[T]):                     # read-only Stmt-tree traversal
    def visit(self, stmt: Stmt) -> T: ...          # dispatch to visit_<Type> else generic_visit
    def generic_visit(self, stmt: Stmt) -> T: ...  # recurse child Stmts

class StmtMutator:                                    # identity-preserving Stmt rewrite
    def visit(self, stmt: Stmt) -> Stmt: ...          # dispatch to visit_<Type> else generic_visit
    def generic_visit(self, stmt: Stmt) -> Stmt: ...  # recurse child Stmts; invariant identical to ExprMutator
```

- constraints:
  - `StmtMutator`'s identity-preservation invariant is identical to `ExprMutator`.
  - `StmtVisitor` / `StmtMutator` do not descend into Expr fields embedded in
    Stmts; those are visited only through `StmtExprMutator` (§6).

`_stmt_children(stmt)` enumerates child Stmts only (Expr fields
come back via `StmtExprMutator`):

| Stmt | Child Stmts |
|---|---|
| `Sequential` | `body` |
| `PrimFunction` | `(body,)` |
| `For` / `While` | `(body,)` |
| `If` | `(then_body, else_body)` |
| `MeshScope` | `(body,)` |
| `LetStmt` | `(body,)` |
| `Return` | `()` |
| `Evaluate` | `()` (leaf in the Stmt tree; its Expr fields are `args`, plus `callable` when `callable` is a `SymbolRef`) |

`StmtVisitor` / `StmtMutator` do **not** descend into Expr fields
embedded in Stmts — `For.start` / `For.stop` / `For.step` /
`While.cond` / `If.cond` / `LetStmt.value` / `Evaluate.args` (and
`Evaluate.callable` when it is a `SymbolRef`) are visited only when
`StmtExprMutator` is used (§6).

## 6. `StmtExprMutator`

Composite: rewrite the Stmt tree **and** descend into the Expr
fields embedded in Stmts. This is the most common combination
(every lowering / simplification / structural rewrite needs it).

```python
class StmtExprMutator(StmtMutator, ExprMutator):      # rewrite Stmts and the Exprs embedded in their Expr-typed fields
    def visit_stmt(self, stmt: Stmt) -> Stmt: ...     # rewrite the Stmt tree via StmtMutator
    def visit_expr(self, expr: Expr) -> Expr: ...     # rewrite embedded value Exprs via ExprMutator
    def generic_visit(self, stmt: Stmt) -> Stmt: ...  # StmtMutator recurse, then rewrite each Stmt's Expr fields
```

- constraints:
  - The rewrite scope is **embedded value Exprs**, not binding `Var`s.

`_rewrite_stmt_exprs(stmt, fn)` enumerates the Expr fields of each
Stmt subclass, applies `fn` with identity preservation, and rebuilds
when needed:

| Stmt | Expr fields |
|---|---|
| `LetStmt` | `value` (`var` is a `Var` and is not rewritten) |
| `For` | `start`, `stop`, `step` (`induction_var` is a `Var` and is not rewritten) |
| `While` | `cond` |
| `If` | `cond` |
| `Return` | (none — `@prim_func` has no value return) |
| `Evaluate` | `args` (and `callable` when it is a `SymbolRef`) |
| `MeshScope` / `Sequential` | (none) |

The rewrite scope is **embedded value Exprs**, not binding `Var`s.
`α-renaming` and similar `Var`-rewriting passes use `StmtMutator`
directly to rebuild Stmts; they do not reuse `StmtExprMutator`.

## 7. Visitor entry forms for `Evaluate`

The TIR effect-form Ops (e.g. `Copy` / `Fill` / `Mma` / `ReLU` /
`RMSNorm` / `Reduce`) are `Op` subclasses, not `Stmt` subclasses; in
Stmt position they appear as `Evaluate(callable=op, args)` so the
invocation can sit in `Sequential` body position. Passes and visitors
MUST match on `Evaluate` and dispatch on `type(callable)`:

a `visit_Evaluate(self, stmt)` override branches on `type(stmt.callable)`
(e.g. `Copy`).

`StmtVisitor` / `StmtMutator` recognise `Evaluate` as a
leaf-in-stmt-tree — `_stmt_children(Evaluate)` is empty.
`StmtExprMutator` exposes `Evaluate`'s `args` (and its `callable` when
that is a `SymbolRef`) as Expr fields, so Expr-level rewrites still
reach them. A value-form `Call` to a TIR effect-form Op in Stmt
position, instead of `Evaluate(op, args)`, is malformed IR
([tir §1.4](./tir.md#14-evaluate)).

## 8. Implementation location

- File: `src/tilefoundry/ir/visitor.py`.
- Public exports: `ExprVisitor`, `ExprMutator`, `StmtVisitor`,
  `StmtMutator`, `StmtExprMutator`.
- The four child-enumeration / rebuild tables
  (`_expr_children`, `_rebuild_expr`, `_stmt_children`,
  `_rewrite_stmt_exprs`) are module-private. Adding a new Expr or
  Stmt subclass requires extending the relevant tables in this
  single file; downstream IR node files are not affected.
