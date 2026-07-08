# TileFoundry Spec — Code Organization

Implementation guide (not an architecture spec). Defines the Python
source tree layout.

## 1. Directory skeleton

> This spec describes the **stable layers** only — whether a single
> `.py` file exists is decided by the naming rules in §2 and is not
> enumerated here (so adding a new Op does not require a spec edit).
> The current file list is whatever `git ls-files src/tilefoundry`
> reports.

The top-level package `src/tilefoundry/` is divided as follows. Each
directory has an owning spec — that spec is the single source of
truth for the directory's structure and invariants.

| Directory | Owning spec | Contents |
|---|---|---|
| `ir/core/` | [core-ir](./core-ir.md) | Shared node algebra: `Module` / `Expr` / `Var` / `Constant` / `Tuple` / `Op` / `Call` / `Stmt` (base class) / `OpSchema` / `ParamDef` / `@register_op` / `@register_alias` / `op_registry` / `errors` / `registry`. |
| `ir/types/` | [types](./types.md) | Type-system root: `TensorType` / `TupleType` / `UnitType` / `CallableType` / `DType` / `Storage` / `dim.*` (with their typeinfer). |
| `ir/types/shard/` | [shard](./shard.md) | Shard / layout sublayer: `Topology` / `Mesh` / `Layout` / `ComposedLayout` / `ShardLayout` / `ShardAttr` (`Split` / `Broadcast` / `Dynamic` / `Partial`). The physical nesting reflects the spec's "sublayer" relationship. |
| `ir/visitor.py` | [visitor-mutator](./visitor-mutator.md) | `ExprVisitor` / `ExprMutator` / `StmtVisitor` / `StmtMutator` / `StmtExprMutator`. |
| `ir/hir/` | [hir](./hir.md) | HIR Op layer; one subdirectory per category (`math/` / `tensor/` / `nn/` / `shape/` / `sharding/`). One real Op per `.py` (§2 rule 1); surface-alias schemas have no per-name file and live in each category's `aliases.py` (§2 rule 5). |
| `ir/tir/` | [tir](./tir.md) | TIR layer: `stmt.py` re-exports the `Stmt` base from `ir/core/stmt.py`; `stmts.py` hosts the TIR `Stmt` subclasses (`LetStmt` / `Evaluate` / `Sequential` / `MeshScope` / …); `prim_function.py`; effect Ops and TIR-owned Expr Ops by category (`memory/` / `nn/` / …); `arith.py` / `reduce.py` for tag-dispatched `Binary` / `Unary` / `Reduce`; `intrinsic.py` for the `@intrinsic` decorator. Target-specific nodes nest under `ir/tir/<target>/<category>/` (e.g. `ir/tir/cuda/nn/mma.py`) per §2 Rule 1c. |
| `parser/` | [parser](./parser.md) | DSL → IR parsing: `base.py` (shared visitor base + dispatch), `hir_parser.py` (`@func` body), `tir_parser.py` (`@prim_func` body), layout sugar / range-slice / dispatch modules. **Not under `ir/`** — the parser is a producer of IR, not an IR sublayer. |
| `passes/` | [passes](./passes.md) | Pass framework (`pass_base.py` / `pass_manager.py`) plus concrete transforms (`transforms/<pass_name>.py`, §2 rule 6). |
| `ir/target/` | [target](./target.md) | Compilation target capability descriptors: `Target` / `CudaTarget` / `CpuTarget` / `resolve_target`. |
| `codegen/` | [codegen](./codegen.md) | Code generation: the emitter registry, the linkable / linked products and the linker, and per-target emitters under `<target>/` (mirroring `ir/tir/` file layout — `tir/<category>/<name>.py` emitter, §2 rule 2). **Not under `ir/`** — codegen is a consumer of IR; `templates/` holds boilerplate only (kernel shells / host stubs). |
| `runtime/` | [runtime](./runtime.md) | Runtime support (per-target headers, function templates, launch helpers). |
| `inspection/` | [inspection](./inspection.md) | IR visualisation: DOT, Python printer, web viewer. |
| `dsl/` | [parser](./parser.md) (authoring namespace) | User-facing import surface: `tf/` (HIR namespace) / `T/` (TIR namespace) / `_stub_gen.py` / `__main__.py`. The `tf/__init__.pyi` and `T/__init__.pyi` stubs are produced by `python -m tilefoundry.dsl regen` and are gitignored. |
| `compile.py` | [architecture](./architecture.md) | `tilefoundry.lower` / `build` / `compile` top-level public verbs. |
| `script.py` | [parser](./parser.md) | `@func` / `@prim_func` / `@module` decorator entry points. |

**Stage boundary.** The pipeline picture in
[architecture §1](./architecture.md#1-spec-relationship-map) places
`parser/` and `codegen/` outside `ir/` (front-end producer / back-end
consumer); the physical directory layout reflects that boundary
directly.

**Reading notes:**

- `ir/` holds the IR proper and its sublayers only. `ir/types/` is the
  root of the type system; `ir/types/shard/` is its shard / layout
  sublayer ([architecture §3](./architecture.md#3-type-system)). The
  physical nesting reflects the spec's conceptual "sublayer".
- The placement of `shard/` under `types/` is a filing decision, not a
  consumer restriction: `Topology` / `Mesh` / `Layout` / `ShardLayout`
  are consumed directly by `parser`, `tir`, and `codegen`. The
  hierarchy expresses "role in the type system", not "who may import
  it".
- `codegen/` and `parser/` sit outside `ir/`. By the
  [architecture §1](./architecture.md#1-spec-relationship-map)
  pipeline they are the front-end producer and back-end consumer of
  IR, not IR sublayers.
- `codegen/<target>/` consumes only TIR. The subtree mirrors
  `ir/tir/`: `prim_function` lives in `tir/`, Stmt emitters in
  `tir/stmts/`, and `memory/` / `nn/` / `arith/` / `reduce/` /
  `tensor/` each have their own subdirectory. There is no
  `codegen/<target>/hir/`.

## 2. File naming and content rules

**Rule 1 — one real Op = one file.** A real Op class lives in
`ir/<hir|tir>/<category>/<op_name>.py`. The file name is the
snake_case of the Op class CamelCase (`MatMul` → `matmul.py`,
`RMSNorm` → `rms_norm.py`). TIR effect Ops and TIR-owned Expr Ops
follow the same rule.

**Rule 1a — surface-alias schemas have no per-name file.** A surface
alias ([core-ir §2.3](./core-ir.md#surface-aliases-register_alias))
has no IR class — its builder routes to a kinded target Op. All
aliases for a category live together in `aliases.py` (e.g. the 19
HIR math sugar names `add` / `sub` / `cmp_eq` / `neg` / … all
register in `ir/hir/math/aliases.py`).

**Rule 1b — tag-dispatched IR classes.** `Binary` / `Unary` /
`Reduce` and other Op classes that fold many surface names through a
`kind` attribute live in one file per IR class
(`ir/hir/math/binary.py` / `ir/hir/math/unary.py` /
`ir/tir/arith.py` / `ir/tir/reduce.py`). This does not contradict
Rule 1: "one Op = one file" means **one IR class** per file; aliases
are not IR classes, so they go through Rule 1a.

**Rule 1c — target-specific IR nodes nest under the dialect.** IR is
**dialect-first**: its primary organizing axis is the dialect, and most
nodes are target-neutral. A node or descriptor that is specific to one
compilation target nests as `ir/{dialect}/{target}/{category}/<name>.py`;
target-neutral abstractions stay at `ir/{dialect}/{category}/`. For
example the whole MMA surface is target-owned — the `Mma` op, the
`MmaOpSpec` / `MmaAtom` descriptors, the CUDA SM80 instruction spec, and its
fragment layouts all live under `ir/tir/cuda/nn/` (`mma.py` + `mma_atom.py`),
and the HIR per-shape `Mma_SM80_*` / `Wgmma_SM90_*` ops under
`ir/hir/cuda/nn/mma.py` — because an MMA instruction fixes a concrete hardware
op. (`codegen/` and `runtime/` are **target-first** instead — their primary
axis is the target — so each tree is organized by its own primary axis.)

**Rule 2 — one (node, target) codegen = one file.** Each handler
lives at `codegen/<target>/tir/<category>/<name>.py`. Stmt emitters,
Expr-Op emitters, and tag-dispatched (`arith`, `reduce`) emitters
each get their own file. Codegen consumes TIR only.

**Rule 3 — what an IR-class file contains:**

- **HIR Op file** (`ir/hir/<cat>/<name>.py`): Op class +
  `@register_typeinfer(Op)` + `@register_costmodel(Op)` (if any).
- **TIR effect Op file** (`ir/tir/<cat>/<name>.py`): Op class +
  `@register_typeinfer(Op)` (returning `UnitType`) +
  `@register_verify_stmt(Op)`. The verify rule keys on the Op class
  even though the invocation is an `Evaluate(op, args)` Stmt — see
  [visitor-registry §5](./visitor-registry.md).
- **TIR-owned Expr Op file**
  (`ir/tir/memory/{alloc_tensor,ptr_of,memory_span,tensor_view}.py`,
  …): Op class + `@register_typeinfer(Op)` +
  `@register_costmodel(Op)` (if any). Call-position constraints
  (e.g. `AllocTensor` may only appear as `LetStmt.value`) are
  checked by the **enclosing Stmt's** `@register_verify_stmt`; there
  is no separate Op-level verify decorator for these.
- **`<category>/aliases.py` file** (Rule 1a): a list of
  `@register_alias(...)` declarations whose builders construct the
  target Op instance. Alias files hold no IR class and participate
  in neither typeinfer nor verify.

**Rule 4 — what a target codegen file contains:** the
`@register_codegen_<target>` for that (op / stmt) pair, and nothing
else.

**Rule 5 — `<category>/__init__.py` re-export rules:**

- Real Op submodules are re-exported via `from .<file> import <Cls>`
  (maintained by hand; new Ops add their import here).
- `aliases.py` is imported only for its `@register_alias`
  side-effects; nothing is re-exported from it (aliases have no
  class to expose).
- User imports go through the `tilefoundry.dsl.tf` /
  `tilefoundry.dsl.T` namespaces' `__getattr__`, not through the
  per-category `__init__.py`
  ([parser §2](./parser.md#2-dsl-namespace-package)).

**Rule 6 — one pass = one file.** A pass class lives in
`passes/transforms/<pass_name>.py` (snake_case file name = pass
class CamelCase in snake form: `HirToTirPass` → `hir_to_tir.py`).
Internal visitors / mutators stay in the same file.

**Rule 7 — what template files contain.**
`codegen/<target>/templates/*.j2` carry boilerplate assembly only
(kernel shells, host stubs, fixed-shape per-target wrappers).
Stmt-level and Op-level emitters are not template-driven; they live
in `codegen/<target>/tir/.../*.py` as Python walkers (see
[codegen](./codegen.md)).

## 3. Multi-agent parallelism guarantee

The lock granularity is a single `(node, target)` pair. The naming
rules in §2 imply that two agents working on different
`(node, target)` pairs touch disjoint files; cross-cutting changes
(`shard/` fields, kernel templates, pass framework) confine
themselves to the owning directory. Representative scenarios:

| Scenario | Files affected |
|---|---|
| Agent A adds `Conv2D`, Agent B adds `RMSNorm` | `ir/hir/nn/conv2d.py` + `ir/hir/nn/rms_norm.py` — no conflict |
| Agent A edits `MatMul.typeinfer`, Agent B adds `tir.cuda.nn.Mma` CUDA codegen | `ir/hir/nn/matmul.py` + `codegen/cuda/tir/nn/mma.py` — no conflict |
| Agent A adds a new target `cpu` | a fresh `codegen/cpu/` subtree — no conflict |

Anything that fits the "one Op = one file" / "one (node, target) =
one file" / "one pass = one file" rules above shares the same
property by construction.

## 4. DSL package layout

The author-facing surface is delivered as a namespace package:

```
src/tilefoundry/dsl/
  __init__.py           # exports `tf` and `T` sub-namespaces
  __main__.py           # `python -m tilefoundry.dsl regen` CLI
  _stub_gen.py          # `.pyi` generator (run by regen)
  py.typed              # PEP 561 marker
  tf/
    __init__.py         # module-level __getattr__ → OpSchema lookup
    __init__.pyi        # AUTO-GENERATED, gitignored
  T/
    __init__.py         # same pattern for TIR
    __init__.pyi        # AUTO-GENERATED, gitignored
```

The two sub-packages (`tf` and `T`) follow the same pattern:
`__getattr__(name)` looks `name` up in the OpSchema registry for the
corresponding dialect and returns either the Op class (for real-Op
schemas) or the alias builder fn (for surface-alias schemas).
Unknown names raise `AttributeError`.

### 4.1 Built-in op-class location convention

For `@register_op` to auto-derive `dialect` + `category`, an Op
class MUST live under:

```
src/tilefoundry/ir/<hir|tir>/<category>/<file>.py
```

with `cls.__module__` matching `tilefoundry.ir.<hir|tir>.<category>.*`.
The 4th segment of the dotted path is the category. Outside that
path, the decorator requires explicit `dialect=` and `category=`
kwargs.

`cls.__name__.lower()` is the default canonical Op name. When the
canonical name diverges from the class lowercase (e.g. `RMSNorm` →
`rms_norm`), pass `name="..."` explicitly.

### 4.2 `.pyi` stub regeneration

The `.pyi` stubs reflect the registered schemas only. After adding
a new `@register_op` / `@register_alias`, regenerate stubs via:

```
python -m tilefoundry.dsl regen
```

The CLI imports `tilefoundry.ir` (forcing every built-in schema to
register) and writes `tf/__init__.pyi` / `T/__init__.pyi`. Stubs
are gitignored — IDEs that need them locally SHOULD run `regen` on
package install.

## 5. DSL import surface

The author-facing exports route through `tilefoundry.dsl`:

```python
# canonical authoring imports
from tilefoundry import func, prim_func
from tilefoundry.dsl import tf, T, Tensor
```

- `Tensor` is the parser-owned DSL authoring-surface annotation
  sugar; it is owned by `tilefoundry.dsl` (defined under
  `tilefoundry.dsl._tensor`, re-exported as `tilefoundry.dsl.Tensor`). It
  is **not** the IR tensor type — the IR type carrier is
  `tilefoundry.ir.types.TensorType`. See [parser §1.4](./parser.md) for
  the annotation grammar.
- `DType` is **not** re-exported. dtype values use string form in
  DSL source (`Tensor[(8,), "bf16"]`, `zeros((1, 64), "bf16", ...)`);
  the parser converts strings to `DType.<name>` at attribute-binding
  time when the receiving `ParamDef` declares `annotation=DType`.
- For users who prefer bare Op names (`add(...)` / `relu(...)`),
  `from tilefoundry.dsl.tf import *` binds every registered HIR name
  into the call site's lexical scope. Without that import the
  parser requires the namespace form `tf.add(...)`.

The `tilefoundry.dsl.{tf, T}` modules expose `__all__` via their lazy
`__getattr__`, so a star-import sees every name registered against
the corresponding dialect, including custom Ops registered after
the DSL package first loaded.
