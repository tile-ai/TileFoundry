# TileFoundry Spec — Code Organization

Implementation guide (not an architecture spec). Defines the Python
source tree layout.

## 1. Directory skeleton

> This spec describes the **stable layers** only — whether a single
> `.py` file exists is decided by the naming rules in [§2](#2-file-naming-and-content-rules) and is not
> enumerated here (so adding a new Op does not require a spec edit).
> The current file list is whatever `git ls-files src/tilefoundry`
> reports.

The top-level package `src/tilefoundry/` is divided as follows. Each
directory has an owning spec — that spec is the single source of
truth for the directory's structure and invariants.

| Directory | Owning spec | Contents |
|---|---|---|
| `ir/core/` | [core-ir](./core-ir.md) | Shared node algebra: `Module` / `Expr` / `Var` / `Constant` / `Tuple` / `Op` / `Call` / `Stmt` (base class) / `OpSchema` / `ParamDef` / call-graph and ownership queries / typed metadata attach-detach and diagnostics / `@register_op` / `@register_alias` / `op_registry` / `errors`. |
| `ir/types/` | [types](./types.md) | Type-system root: `Type` / `TensorType` / `TupleType` / `UnitType` / `CallableType` / `DType` / `StorageKind` / `resolve_storage` / local projections (`local_type_of`) / tensor-leaf, byte-by-storage, and topology-extent queries / `dim.*` (with their typeinfer). |
| `ir/types/shard/` | [shard](./shard.md) | Shard / layout sublayer: `Topology` / `Mesh` / `Layout` / `ComposedLayout` / `ShardLayout` / `ShardAttr` (`Split` / `Broadcast` / `Dynamic` / `Partial`). The physical nesting reflects the spec's "sublayer" relationship. |
| `ir/constraints/` | [parser](./parser.md) | Authored `where(layout=..., mesh=..., storage=...)` constraint records: the shared base plus layout, mesh, and storage constraints, attached by the parser and read back by the Python printer. |
| `ir/visitor.py` | [visitor-mutator](./visitor-mutator.md) | `ExprFunctor` / `ExprVisitor` / `ExprWalker` / `ExprCollector` / `ExprCloner` / `BindingSubstitutionCloner` / `StmtVisitor` / `StmtMutator` / `StmtExprMutator`, plus `collect_exprs`, value-operand/function-value queries, and the canonical `PrimFunction` walk and rewrite entries. |
| `ir/hir/` | [hir](./hir.md) | HIR Op layer; one subdirectory per category (`math/` / `tensor/` / `nn/` / `shape/` / `sharding/`). One real Op per `.py` ([§2](#2-file-naming-and-content-rules) rule 1); surface-alias schemas have no per-name file and live in each category's `aliases.py` ([§2](#2-file-naming-and-content-rules) rule 5). |
| `ir/tir/` | [tir](./tir.md) | TIR layer: `stmt.py` re-exports the `Stmt` base from `ir/core/stmt.py`; `stmts.py` hosts the general TIR `Stmt` subclasses (`LetStmt` / `Evaluate` / `Sequential` / `MeshScope` / …), while specialized statement families such as `DispatchCall` may live in their own file; `prim_function.py`; effect Ops and TIR-owned Expr Ops by category (`memory/` / `nn/` / …); `launch.py` owns `Launch` and its authored launch-attribute descriptors; `arith.py` / `reduce.py` for tag-dispatched `Binary` / `Unary` / `Reduce`; `intrinsic.py` for the `@intrinsic` decorator. Target-specific nodes nest under `ir/tir/<target>/<category>/` (e.g. `ir/tir/cuda/nn/mma.py`) per [§2](#2-file-naming-and-content-rules) Rule 1c. |
| `parser/` | [parser](./parser.md) | DSL → IR parsing: `base.py` (shared visitor base + dispatch), `hir_parser.py` (`@func` body), `tir_parser.py` (`@prim_func` body), layout sugar / range-slice / dispatch modules. **Not under `ir/`** — the parser is a producer of IR, not an IR sublayer. |
| `analysis/` | [analysis](./analysis.md) | Fact layer over typed HIR: one module per analysis family. The compact public surface lives in `analysis/__init__.py`; per-target Facts projections live with their owning Target. |
| `passes/` | [passes](./passes.md) | Pass framework (`pass_base.py` / `pass_manager.py`) plus concrete transforms (`transforms/<pass_name>.py`, [§2](#2-file-naming-and-content-rules) rule 6). |
| `target/` | [target](./target.md) | Compilation Target classes, class registration, service selection, and architecture/device facts: `base.py` owns `Architecture` / `Device` / `Target` / `register_target` / `registered_targets`; `services.py` owns the immutable service descriptors; each backend owns its concrete Target. Authored code constructs Target values; there is no string resolver. |
| `target/cpu.py` | [target](./target.md) | The `CpuTarget` backend and its CPU code-generation service selection. |
| `target/hardware/` | [target](./target.md) | The installed hardware database and its generic machinery: the authored Architecture / Device documents, the envelope and evidence-leaf loader, `HardwareSpecRegistry`, and the exact-key schema reader. It fixes the envelope only; the fact namespace below `facts` belongs to the target package named by a document's `schema`. |
| `target/<backend>/spec.py` | [target](./target.md) | One backend's typed hardware schemas and the documents it installs, registered into the shared registry as an import side effect. This is where a fact path, its unit, and its cross-field invariants are validated, and where a document becomes an immutable `Architecture` / `Device` value. |
| `analysis/api.py` | [analysis](./analysis.md) | The public composed Analyze operation: shared authored-program check and normalization, one per-call `AnalyzeContext`, dependency closure, ordering, single execution per member, Metadata-ownership enforcement, and semantic result assembly. |
| `analysis/registry.py` | [analysis](./analysis.md) | The built-in Analyzer declarations. It re-exports the immutable `Analyzer` descriptor from `target/services.py` and holds no Target dispatch table. |
| `analysis/errors.py` | [analysis](./analysis.md) | `AnalysisError`, the one diagnostic the whole analysis layer raises, so catching an analysis failure catches every analysis failure rather than the subset the caller happened to import. |
| `analysis/visitor.py` | [analysis](./analysis.md) | The per-call `AnalyzeContext`, carrying the shared root/current lexical `Scope` while a family traverses its work. |
| `analysis/scope.py` | [analysis](./analysis.md) | The shared `Scope` tree and `Access` relations built once from normalized HIR; families query these views instead of constructing parallel structure. |
| `analysis/affine.py` | [analysis](./analysis.md) | The shared loop-affine term parser used by scope binding and authored-loop footprint binding, including constant loop strides and bounded invariant offsets. It does not introduce a second affine graph representation. |
| `analysis/footprint.py` | [analysis](./analysis.md) | Target-independent authored-loop access images, buffer-view folding, and deduplicated versus repeated byte readings. Requires no separate time map. |
| `analysis/report.py` | [analysis](./analysis.md) | Structured analysis report data, including record-family registration, field serialization, and target-aware report-only projections. It depends only on analysis/core modules; inspection consumes it to produce text and source annotations. |
| `analysis/check.py` | [analysis](./analysis.md) | The shared authored-program gate for analysis: authored-type re-derivation, authored validation, call-context validation, and checker-specific input checks. Established once per public call rather than per family. |
| `analysis/facts.py` | [analysis](./analysis.md) | The narrow Facts aggregates the analysis families declare — the memory hierarchy graph, the throughput rates, and the parallel capacity. It is the record of how much hardware each measurement rests on, and names no backend; a Fact shared across consumer families belongs in `target/facts.py`. |
| `analysis/metadata.py` | [analysis](./analysis.md) | The typed records the families leave on the IR, split by what each number depends on rather than by convenience. |
| `analysis/compute_cost.py` | [analysis](./analysis.md) | The `compute-cost` family: logical flops per DType and bytes per storage level, from the authored program alone. |
| `analysis/memory.py` | [analysis](./analysis.md) | The `memory` family: value lifetimes, per-level peaks, and the capacity comparisons against a target's hierarchy — failing on an over-full addressable level and advising on an over-full cache. |
| `analysis/roofline.py` | [analysis](./analysis.md) | The `roofline` family: the recorded work divided by the target's published rates, per Call and aggregated per Function. Adds no count of its own. |
| `analysis/performance.py` | [analysis](./analysis.md) | The `performance` family: occurrences projected from the shared `Scope` tree into flat timeline records and one function envelope, scaled by parallel capacity. It introduces no second scope tree. |
| `visitor_registry/` | [visitor-registry](./visitor-registry.md) | Shared registry instances and derived visitors: access-relation construction, contexts, ISL helpers, relation building, shard propagation, type inference, verification, code generation, and cost evaluation. |
| `visitor_registry/op_cost.py` | [analysis](./analysis.md) | Each operation's per-instance flops and bytes, registered into the shared cost-evaluator registry. Owned here rather than by any target package, because the work an operation asks for follows from its own semantics and operand types on every backend. |
| `inspection/analysis_report.py` | [inspection](./inspection.md) | Presentation of analysis-owned report data as text and annotated source. Analysis owns the structured report data and JSON dump; inspection owns how a human reads it. |
| `target/<backend>/facts.py` | [target](./target.md) | One backend's Facts projections selected by its Target's `get_facts`. They restate installed documents in the shape a family declared and measure nothing. |
| `target/facts.py` | [target](./target.md) | Facts used across consumer families, such as topology limits, plus validation for values returned by `Target.get_facts`: the requested frozen-dataclass shape and returned type. A Fact used by one family stays with that family; this module holds no projection registry. |
| `codegen/` | [codegen](./codegen.md) | Code generation: the immutable `CodeGenerator` service, linkable / linked products and linker, and concrete generators under `<target>/` (mirroring `ir/tir/` file layout — `tir/<category>/<name>.py` emitter, [§2](#2-file-naming-and-content-rules) rule 2). **Not under `ir/`** — codegen is a consumer of IR; `templates/` holds boilerplate only. |
| `codegen/registry.py` | [codegen](./codegen.md) | The compatibility re-export of the immutable `CodeGenerator` descriptor and source-order grouping by equal Target values. It rejects multiple unequal CUDA Target groups before emission and holds no emitter registry. |
| `runtime/` | [runtime](./runtime.md) | Runtime support (per-target headers, function templates, launch helpers). |
| `inspection/` | [inspection](./inspection.md) | IR visualisation: DOT, Python printer, web viewer. |
| `dump/` | [inspection](./inspection.md) | Dump flags, dynamically scoped dump contexts, and file/null dump sinks used by inspection and test integration. |
| `dsl/` | [parser](./parser.md) (authoring namespace) | User-facing import surface: `tf/` (HIR namespace) / `T/` (TIR namespace, including `_platforms.py`) / `_namespace.py` / `_stub_gen.py` / `storage.py` / `__main__.py`. The `tf/__init__.pyi` and `T/__init__.pyi` stubs are produced by `python -m tilefoundry.dsl regen` and are gitignored. |
| `compile.py` | [architecture](./architecture.md) | `tilefoundry.lower` / `build` / `compile` / `jit` top-level public verbs. |
| `module.py` | [parser](./parser.md) | The `@module` decorator entry point and module-level topology authoring constants. |
| `script.py` | [parser](./parser.md) | `@func` / `@prim_func` / `@intrinsic` decorator entry points. |
| `__init__.py` | [inspection](./inspection.md) | The top-level `view` convenience entry and re-exports of public compiler surfaces. |
| `utils/` | [code-organization](./code-organization.md) | Shared leaf machinery: a module here MUST import nothing from `ir/`, `parser/`, `passes/`, `codegen/`, `runtime/` or `cli/`, and MUST name no layer. It is depended on and depends on nothing, which is what lets a consumer outside the package — a pre-commit hook under an interpreter with nothing installed — load one of these modules by path and get the same implementation the package uses. A helper that needs to know a layer belongs in that layer; this is not a home for anything that did not fit. |

**Stage boundary.** The pipeline picture in
[architecture §1](./architecture.md#1-spec-relationship-map) places `parser/`
and `codegen/` outside `ir/` (front-end producer and back-end consumer); the
physical directory layout reflects that boundary directly.

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
- `analysis/` sits outside `ir/` because it derives facts about typed HIR
  rather than defining an IR layer. It reads the IR and the `Target` and
  decides nothing over what it measures
  ([architecture §5](./architecture.md#5-analysis--optimization)).
- `codegen/<target>/` consumes only TIR. The subtree mirrors
  `ir/tir/`: `prim_function` lives in `tir/`, Stmt emitters in
  `tir/stmts/`, and `memory/` / `nn/` / `arith/` / `reduce/` /
  `tensor/` each have their own subdirectory. There is no
  `codegen/<target>/hir/`.
- Authored launch attributes belong to `ir/tir/launch.py`; launch-geometry
  derivation (grid / block extents) is an internal `codegen/cuda/emit.py`
  helper (`_derive_launch_config`), consumed within codegen itself rather
  than carried past it as a runtime-owned metadata type. The two launch
  contracts are distinct even though both are consumed across the codegen
  boundary.

`ir/constraints/`, `visitor_registry/`, and `dump/` are cross-cutting packages;
their stable responsibilities are owned by [parser](./parser.md),
[visitor-registry](./visitor-registry.md), and [inspection](./inspection.md),
respectively. Their internal file layout is not a per-Op contract.

## 2. File naming and content rules

**Rule 1 — one real Op = one file.** A real Op class lives in
`ir/<hir|tir>/<category>/<op_name>.py`. The file name is the
snake_case of the Op class CamelCase (`MatMul` → `matmul.py`,
`RMSNorm` → `rms_norm.py`). TIR effect Ops and TIR-owned Expr Ops
follow the same rule.

**Rule 1a — surface-alias schemas have no per-name file.** A surface
alias ([core-ir §2.3](./core-ir.md#23-op))
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
fragment layouts all live under `ir/tir/cuda/nn/` (`mma.py` + `mma_atom.py`).
The backend-bound construction stays in TIR: HIR is the checking reference
side, and carrying the instruction name in that reference would make two GPU
targets require different HIR references. (`codegen/` and `runtime/` are
**target-first** instead — their primary axis is the target — so each tree is
organized by its own primary axis.)

**Rule 2 — one (node, target) codegen = one file.** Each handler
lives at `codegen/<target>/tir/<category>/<name>.py`. Stmt emitters,
Expr-Op emitters, and tag-dispatched (`arith`, `reduce`) emitters
each get their own file. Codegen consumes TIR only.

**Rule 3 — what an IR-class file contains:**

- **HIR Op file** (`ir/hir/<cat>/<name>.py`): Op class +
  `@register_typeinfer(Op)` + `@register_cost_evaluator(Op)` (if any).
- **TIR effect Op file** (`ir/tir/<cat>/<name>.py`): Op class +
  `@register_typeinfer(Op)` (returning `UnitType`) +
  `@register_verify_stmt(Op)`. The verify rule keys on the Op class
  even though the invocation is an `Evaluate(op, args)` Stmt — see
  [visitor-registry §5](./visitor-registry.md#5-instance-2--verify).
- **TIR-owned Expr Op file** (`ir/tir/memory/{alloc_tensor,ptr_of,memory_span,tensor_view}.py`,
  …): Op class + `@register_typeinfer(Op)` + `@register_cost_evaluator(Op)`
  (if any). Call-position constraints are checked by the enclosing Stmt's
  `@register_verify_stmt`.
- **`<category>/aliases.py` file** (Rule 1a): `@register_alias(...)`
  declarations whose builders construct the target Op instance.

**Rule 4 — what a target codegen file contains:** the
`@register_codegen_<target>` for that (op / stmt) pair, and nothing else.

**Rule 5 — `<category>/__init__.py` re-export rules:** real Op submodules are
re-exported; aliases are imported only for registration side effects; user imports
go through [parser §2](./parser.md#2-syntax-and-rules).

**Rule 6 — one pass = one file.** A pass class lives in
`passes/transforms/<pass_name>.py`; internal visitors / mutators stay in that file.

**Rule 7 — what template files contain.** `codegen/<target>/templates/*.j2`
carry boilerplate assembly only; emitters live in Python walkers.

## 3. Multi-agent parallelism guarantee

The lock granularity is a single `(node, target)` pair. The naming rules in
[§2](#2-file-naming-and-content-rules) imply that two agents working on different
`(node, target)` pairs touch disjoint files; cross-cutting changes confine
themselves to the owning directory.

## 4. DSL package layout

The author-facing surface is delivered as a namespace package. The two
sub-packages (`tf` and `T`) use the OpSchema registry for their corresponding
dialect and return an Op class or alias builder; unknown names raise `AttributeError`.

### 4.1 Built-in op-class location convention

For `@register_op` to auto-derive `dialect` + `category`, an Op class MUST live
under `src/tilefoundry/ir/<hir|tir>/<category>/<file>.py`, with its module name
matching that path. Outside it the decorator requires explicit values.

### 4.2 `.pyi` stub regeneration

The `.pyi` stubs reflect registered schemas only. After adding a new
`@register_op` / `@register_alias`, regenerate them with
`python -m tilefoundry.dsl regen`.

## 5. DSL import surface

The author-facing exports route through `tilefoundry.dsl`:

```python
# example
# Canonical authoring imports.
from tilefoundry import func, prim_func
from tilefoundry.dsl import tf, T, Tensor
```

- `Tensor` is the parser-owned DSL authoring-surface annotation
  sugar; it is owned by `tilefoundry.dsl` (defined under
  `tilefoundry.dsl._tensor`, re-exported as `tilefoundry.dsl.Tensor`). It
  is **not** the IR tensor type — the IR type carrier is
  `tilefoundry.ir.types.TensorType`. See [parser §2.1](./parser.md#21-syntax) for
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
