# TileFoundry Spec — Code Organization

Implementation guide (not an architecture spec). Defines the Python
source rules and the source tree layout.

## 0. Python source rules

These rules bind every Python module under `src/tilefoundry/` and
`targets/<vendor>/tilefoundry_<vendor>/`, except `src/tilefoundry/orchestrator/`
and the `models/` trees. A rule exists only where a checker enforces it; a
convention no tool can check is not listed. No source line carries a tool
directive: every exception is written in `pyproject.toml`
([§0.6](#06-exceptions)).

### 0.1 Reference style

The [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
is the reference for anything the rules below leave open. It is not a rule
itself, because conformance to the whole guide cannot be checked. This spec
overrides the guide in three places: "import modules only" and "no relative
imports" are replaced by C3, and the `_` prefix for internal names is replaced
by C3 visibility and C4.

### 0.2 Style rules

| Rule | Statement | Checker |
|---|---|---|
| S1 | No `@staticmethod`; write a module function. | `scripts/code_organization_lint.py` |
| S2 | `@classmethod` is used only for a named constructor: `from_<source>`, or `create` for a factory protocol whose subclass decides the implementation. | `scripts/code_organization_lint.py` |
| S4 | No nested function or class, unless it closes over a local of the enclosing function other than `self` / `cls`. | `scripts/code_organization_lint.py` |
| S5 | No custom metaclass, `__del__`, `__import__` or `importlib.import_module`. A plugin loading point that imports by name carries a per-file ignore ([§0.6](#06-exceptions)). `getattr` / `hasattr` are allowed for dispatch. | `scripts/code_organization_lint.py` |
| S6 | A parameter default is neither a mutable value nor a function call. | ruff `B006`, `B008` |
| S7 | A `@property` body is not only `return self.<field>`; a setter body is not only `self.<field> = value`. | `scripts/code_organization_lint.py` |
| S8 | A class that derives from a frozen dataclass and declares fields of its own is itself `@dataclass(frozen=True)`. A frozen dataclass's `__post_init__` writes only through `object.__setattr__(self, "<own field>", ...)`. | `scripts/code_organization_lint.py` |
| S9 | A named constructor that uses only its class's layer and lower ones is a `@classmethod` `from_<source>` (S2). One that uses a higher layer is a module function `make_<name>` whose return type is defined in another module. | `scripts/code_organization_lint.py`; the layer half by C2 |
| S10 | Classes are `CapWords`; functions, methods and variables are `snake_case`; constants are `UPPER_CASE`; an exception class name ends in `Error`. Method names dispatched by reflection on a class name (`<prefix>_<ClassName>`: `visit_*`, `visit_leaf_*`, `print_*` and the pattern printer's readers) and the DSL's authored names (`S`, `P`, `B`, …) are the `ignore-names` of `pyproject.toml`. | ruff `N` |
| S11 | A public class or function name does not repeat its package's name: the classes and functions of package `p`, and the names `p` exports, do not contain `p`, unless the name is exactly `p` (`cost.Cost`). Module file names are not constrained. | `scripts/code_organization_lint.py` |

S9 recognizes its subject by name (`make_*`), so a rename evades it.

### 0.3 Structure rules

| Rule | Statement | Checker |
|---|---|---|
| C1 | No import inside a function. A heavy third-party module is loaded lazily through `tilefoundry.utils.lazy`. | ruff `PLC0415` |
| C2 | The package import graph is acyclic and obeys the layer table of [§0.5](#05-layers-and-cycles). The module import graph within one directory is acyclic. | import-linter: the `layers` contract and the whole-graph contract in `scripts/importlinter_acyclic.py` |
| C3 | A module other than `__init__.py` has a literal `__all__`; an `__init__.py` only re-exports and writes no `__all__` ([§0.4](#04-exports-and-imports)). | `scripts/code_organization_lint.py` |
| C4 | No name starts with `_`: files, classes, functions, methods, attributes, variables, parameters and import aliases. Dunder names and a lone `_` are exempt. | `scripts/code_organization_lint.py` |
| C5 | An attribute, or an element of an attribute, is written only through the `self` / `cls` of the method that writes it. No `setattr`, `delattr`, `object.__setattr__`, `vars(x).update` or `x.__dict__` change is applied to another object; a method may change a copy of its own instance that it just made with `copy.copy(self)`. | `scripts/code_organization_lint.py` |
| C6 | No `global`. A module-level container is immutable (`tuple`, `frozenset`, `MappingProxyType`); `__all__` is exempt. Mutable runtime state lives in a class instance and is written by that instance's own methods (C5). | `scripts/code_organization_lint.py` |
| C8 | No comment is a tool directive (`noqa`, `type:`, `pragma:`, `ruff:`, `pylint:`, `pyright:`, `mypy:`, `fmt:`, `isort:`, `hygiene:`). | `scripts/comment_hygiene_lint.py` |
| C9 | Every module and package has a docstring. | ruff `D100`, `D104` |
| C10 | A namespace a module imports is not rebound in that module: no parameter, local, loop or `with`/`except` target, function or class definition, or module-level assignment takes its name. | `scripts/code_organization_lint.py` |
| C11 | Imports take the forms of [§0.4](#04-exports-and-imports). | `scripts/code_organization_lint.py` |

A key with no registration raises `MissingHandlerError` where it is looked up;
the tests cover the registrations. A registration is written next to its key's
owner: an Op's registrations in the Op's own module, a Target's in that Target's
package, a core HIR-to-TIR pairing in the `schedule` package.

### 0.4 Exports and imports

- Every module other than `__init__.py` has a literal `__all__`; it is the module's export
  list. An `__init__.py` holds only its docstring and `from` imports, and the names it
  imports are the package's exports, so it writes no `__all__`.
- Within one directory: `from .module import Name`, with `Name` in `module.__all__`.
- Across directories a module imports a package, or a module the package
  exports, as a namespace and reaches its members through it:
  `from tilefoundry.ir import types`, then `types.TensorType`. Importing a name
  from another directory (`from tilefoundry.ir.types import TensorType`) is never
  done. A package exports one of its own modules by `from . import mod` in its
  `__init__.py`.
- A module never imports its own package or an ancestor package, including a
  module placed directly in an ancestor package's directory. It may import a
  subpackage or an exported module of an ancestor as a namespace
  (`from tilefoundry.ir import pattern`), never a name the ancestor re-exports.
- An `__init__.py` holds only a docstring and `from .x import Name` lines whose `x`
  is a module or subpackage of its own directory.
- Imports use `from ... import ...`; `import tilefoundry.x` is not used.
- Each member reached through a namespace (`typeinfer.register`, `cpu.Target`) is
  exported by that package's `__init__.py` or listed in that module's `__all__`; deeper levels (`ir.utils.leaves_of`)
  are not constrained further. A module that provides members through a
  module-level `__getattr__` (`tf`, `T`) is exempt.
- A user entry point is a package whose `__init__.py` re-exports names of other
  directories: the root `tilefoundry` and `tilefoundry.dsl`. Its own C3 and C11
  findings carry a per-file ignore, and any module may import names, or `*`,
  from it, as user programs do. It keeps a literal `__all__`, which states what
  `import *` brings in.
- Third-party imports are outside this section.

| Visibility | Written as | Usable from |
|---|---|---|
| Module-private | Not in the module's `__all__` | The same file |
| Package-internal | In the module's `__all__`, not imported by the package's `__init__.py` | The modules of the same directory |
| Public | Imported by the package's `__init__.py` | Modules of other directories, subject to C2 |

Re-exporting through `__init__.py` is safe under C2 and this section together.
Python runs every ancestor `__init__.py` before a submodule, so an ancestor is
half initialized while its submodule loads and cannot be imported by it; a
package's `__init__.py` loads only lower layers, which finish first.

### 0.5 Layers and cycles

A package imports only packages on lower rows; the packages of one row do not
import each other. All rows live inside the distribution packages; a row is
not a separate distribution.

| Row | Packages (bottom to top) |
|---|---|
| 0 | `dispatch` (the generic `Registry`), `utils`, `dump` |
| 1 | `services` (service functions over `dispatch.Registry`) |
| 2 | `ir.expr`: `DType`, `StorageKind`, `Expr`, `Op`, the op registry and the `Dim*` operations as data |
| 3 | `ir.types`: type, layout and mesh definitions |
| 4 | `ir.dim`: the behaviour of dimensions (simplification, isl, normalization, substitution) and `MeshCoord` |
| 5 | `ir.utils`: helpers over types (leaves, sizes, bounds, compatibility, local views) |
| 6 | `target` |
| 7 | `ir.clause`, `ir.pattern`, `core` |
| 8 | `ir.core` (structure nodes and visitors) |
| 9 | `interfaces` (`valuation`, `typeinfer`, `verify`, `cost`, `access`, `evaluator`, `schedule`) |
| 10 | `ir.hir`, `ir.tir` |
| 11 | `verifier` (whole-function and whole-module verification) |
| 12 | `analysis`, `parser` (`parser.func`, `parser.prim_func`, `parser.module`: the authoring decorators) |
| 13 | `inspection`, `passes` (owns `passes.Options`, the options every pass is created with, and `passes.default_pipeline`) |
| 14 | `codegen`, the last stage of the pipeline |
| 15 | `schedule`, `runtime` |
| 16 | `compiler`: `build`, `compile`, `jit` and `compiler.Options`, which gathers the options of every stage |
| 17 | `dsl`, `cli`: user entry points; `dsl` re-exports the authoring decorators |
| 18 | `bootstrap`: loads the registering packages and the installed Targets, once, when `tilefoundry` is imported |
| 19 | `tilefoundry`, the root package, imported only by users; its `__init__.py` only re-exports |
| 20 | `tilefoundry_nvidia`, `tilefoundry_apple` |

A cycle is broken according to its cause:

| Cause | Fix |
|---|---|
| An import serves only a type annotation | Quote the annotation and drop the import |
| A catch-all module joins several concepts and becomes a hub | Split it by concept |
| A function of a lower-layer class uses a higher-layer concept | Move the function to the higher layer; the lower layer keeps the data |
| A lower layer must call an implementation that lives higher | Call it through `services`: the service function is declared on row 0, the implementing package registers its implementation at import, and `bootstrap` imports that package |
| A package spans two rows | Split the package by row |

### 0.6 Exceptions

Every rule applies to every Python file of the repository. An exception is a
path glob and the rule codes it ignores, written only in `pyproject.toml`, in the
section of the tool that checks the rule: `[tool.ruff.lint.per-file-ignores]`
(and `pep8-naming` for S10), `[tool.code_organization.per-file-ignores]`,
`[tool.comment_hygiene]` (`exempt`), and `[tool.importlinter]`. Tests and
scripts are not imported as library code, so of the rules this tool checks
only C10 and C11 bind them. `.pre-commit-config.yaml` declares the hooks and
carries no rule arguments.

## 1. Directory skeleton

> This table describes the **stable layers** only. Whether a single `.py` file
> exists is decided by the naming rules in [§2](#2-file-naming-and-content-rules)
> (so adding an Op needs no spec edit); the file list is `git ls-files
> src/tilefoundry`.

Each directory of `src/tilefoundry/` has an owning spec, the single source of
truth for its structure and invariants.

| Directory | Owning spec | Contents |
|---|---|---|
| `ir/expr/` | [core-ir](./core-ir.md) | The IR foundation, importing nothing above itself: `Expr` / `Var` / `Constant` / `Call` / `Tuple` (`expr.py`), typed metadata (`metadata.py`), `Op` / `OpSchema` / `ParamDef` and the op registry with `@register_op` (`op.py`, `op_schema.py`, `param_def.py`, `op_registry.py`, `register.py`), `VerifyError` (`errors.py`), the shared operation kinds (`kinds.py`), `DType` and `StorageKind` (`dtype.py`, `storage.py`), the `Dim*` operations as plain data with `DimVar` and the structural dimension queries (`dim.py`), and the memoizing `Functor` / `Visitor` dispatch over the four nodes above (`visitor.py`). The `+ - * // %` operators of `Call` and `DimVar` build dimension arithmetic through the `simplify_dim` service. |
| `ir/dim/` | [types §4](./types.md#4-dim--symbolic-shape-dimensions) | The behaviour of dimensions: `simplify.py` builds and recognizes dimension arithmetic and implements `simplify_dim`; `isl_interop.py` converts to and from isl, normalizes, bounds, and builds shape domains, implementing `normalize_dim` and `dim_range`; `substitute.py` binds dimension variables through types; `mesh_coord.py` holds the `MeshCoord` Op, whose typeinfer, evaluation, cost and access handlers live in the interface packages. |
| `ir/utils/` | [types](./types.md) | Helpers over types that no type definition needs: tensor leaves (`leaves.py`), compatibility of a declared and an actual type (`compatible.py`), element counts and byte sizes (`sizes.py`), dimension upper bounds and register strides (`bounds.py`), local views of a type (`local.py`), and which scope a statement stands inside and what it admits (`mesh_scope.py`: `device_layout`, `covered_by_scope`), and tiling helpers (`tiler.py`). Nothing below `ir.utils` imports it. |
| `ir/core/` | [core-ir](./core-ir.md) | The structure nodes both IR layers share and the visitors that read them: `Module`, `Function`, `PrimFunction`, `LoopRegion`, `MeshRegion`, `FunctionRef`, the TIR `ShapeOf` Expr (`shape_of.py`), `SSAValue` (`ssa_value.py`), `Printable` / `PrinterBase` (`inspection.py`), the shared static-AST evaluator (`static_eval.py`), `TotalAndPerUnit` / `TripInterval` (`values.py`), and the `Stmt` base (`stmt.py`) with one `Stmt` subclass per file (`sequential.py` / `let_stmt.py` / `for_stmt.py` / `while_stmt.py` / `if_stmt.py` / `mesh_scope.py` / `return_stmt.py` / `evaluate.py`). |
| `ir/core/visitor.py` | [visitor-mutator](./visitor-mutator.md) | `TypeFunctor` / `ExprVisitor` / `ExprWalker` / `ExprCollector` / `ExprCloner` / `BindingSubstitutionCloner` / `StmtVisitor` / `StmtMutator` / `StmtExprMutator`, plus `collect_exprs`, value-operand/function-value queries, and the canonical `PrimFunction` walk and rewrite entries. |
| `ir/pattern/` | [core-ir](./core-ir.md) | Operation-declaration predicates, reached as the namespace `pattern`: the pattern values and the formula language in `pattern.py`, the named arrangement predicates in `predicates.py`, the `Pattern` base with its matcher in `match.py`, formula evaluation and solving in `solve.py`, cross-operand relations in `constraint.py`, and pattern construction/specialization helpers in `utils.py`. |
| `ir/types/` | [types](./types.md) | Type-system root, definitions only: `Type` / `TensorType` / `TupleType` / `UnitType` / `CallableType` / `PointerType`, with the named constructors `TensorType.from_shape`, `Mesh.from_meshes`, `Swizzle.from_active` and `ShardLayout.from_logical`. It reads dimensions through `ir.expr` and the `simplify_dim`, `normalize_dim` and `dim_range` services, and never imports `ir.dim`. |
| `ir/types/{int_tuple,stride,layout,layout_algebra,swizzle_layout,shard_layout,mesh}.py` | [shard](./shard.md) | `Topology` / `Mesh` / `Layout` / `ComposedLayout` / `ShardLayout` / `ShardAttr` (`Split` / `Broadcast` / `Varying` / `Partial`), filed as CuTe files them: int tuples (`flatten` / `unflatten` / `repeat_like` / `product`), strides (`compact_major` / `idx2crd` / `crd2idx`), layouts, general layout algebra, and swizzle-specialized layout algebra each in their own module; mesh construction, separation, and topology-bound checking stay with `Mesh`. |
| `ir/clause/` | [parser](./parser.md) | Authored `where(layout=..., mesh=..., storage=...)` constraint records, reached as the namespace `clause`: the shared base plus layout, mesh, and storage constraints, attached by the parser and read back by the Python printer. |
| `ir/hir/` | [hir](./hir.md) | HIR Op layer; one subdirectory per multi-operation category (`math/` / `tensor/` / `nn/` / `sharding/`), with the standalone scheduling carrier in `schedule.py`. One real Op per `.py` ([§2](#2-file-naming-and-content-rules) rule 1); surface-alias schemas have no per-name file and are declared on their canonical Op class by `register_op(..., aliases=...)` ([§2](#2-file-naming-and-content-rules) rule 1a). The HIR structure nodes `Function` / `LoopRegion` / `MeshRegion` live in `ir/core/`; users import them from `ir.core`. `ir.hir` exports its subpackages and `schedule`, and the type-inference helpers its Ops share live in `interfaces/typeinfer/`. |
| `schedule/` | [schedule](./schedule.md) | Schedule finalization, instruction facts, candidate discovery over HIR, and the target-neutral candidate pairings (`pairings.py`) registered through `interfaces/schedule/`. |
| `ir/tir/` | [tir](./tir.md) | TIR layer: effect Ops and TIR-owned Expr Ops by category (`memory/` / `nn/` / …); `launch.py` owns `Launch` and its authored launch-attribute descriptors; `arith.py` / `reduce.py` for tag-dispatched `Binary` / `Unary` / `Reduce`; `intrinsic.py` for the `@intrinsic` decorator. The `Stmt` subclasses (`LetStmt` / `Evaluate` / `Sequential` / `MeshScope` / …), `PrimFunction`, `FunctionRef` and `ShapeOf` live in `ir/core/`; users import them from `ir.core`. `ir.tir` exports its modules and subpackages. Target-specific nodes live in `targets/<vendor>/tilefoundry_<vendor>/<target>/ir/tir/<category>/` per [§2](#2-file-naming-and-content-rules) Rule 1c. |
| `parser/` | [parser](./parser.md) | DSL → IR parsing: `ast_pattern.py` (executable AST patterns, the shared parse context and `ParseError`), `pattern_nodes.py` (the concrete pattern nodes), `parser_visitor.py` (`parse_function`, the function parser entry), `grammar_render.py` and `spec.py` (the generated grammar reference). `script.py` (`@func` / `@prim_func` / `@intrinsic` decorators) and `module_decorator.py` (the `@module` decorator and the typeinfer + verify loop it runs), exported as `parser.func`, `parser.prim_func`, `parser.module`. **Not under `ir/`** — the parser is a producer of IR, not an IR sublayer. |
| `analysis/` | [analysis](./analysis.md) | Fact layer over typed HIR: one module per analysis family. The compact public surface lives in `analysis/__init__.py`; per-target Facts projections live with their owning Target. |
| `passes/` | [passes](./passes.md) | Pass framework (`pass_base.py` / `pass_manager.py`, and `options.py` for `passes.Options`, `pipeline.py` for `passes.default_pipeline`) plus concrete transforms (`transforms/<pass_name>.py`, [§2](#2-file-naming-and-content-rules) rule 6). |
| `target/` | [target](./target.md) | Compilation Targets. `__init__.py` re-exports the public names and the namespaces `cpu` and `hardware`; authored code constructs Target values (`cpu.Target()`) and there is no string resolver. `target.py` owns `Target` / `Architecture` / `Device` / `UnsupportedCapabilityError` / `bind` and the `bindings` registry; `registry.py` owns the `Catalog` of registered Target classes (`register`, `unregister`, `registered`, `load`); `spec.py` owns `HardwareSpec` and the selection of a hardware document; `facts.py` owns every Facts aggregate and `facts_result`; `analyzer.py` owns the immutable `Analyzer` descriptor. |
| `target/cpu.py` | [target](./target.md) | The `cpu.Target` backend and its CPU code-generation service selection. |
| `target/hardware/` | [target](./target.md) | The installed hardware database and its generic machinery: the authored Architecture / Device documents, the envelope and evidence-leaf loader, and the exact-key schema reader. It fixes the envelope only; the fact namespace below `facts` belongs to the target package named by a document's `schema`. |
| `target/<backend>/spec.py` | [target](./target.md) | One backend's typed hardware schemas and the documents it installs, registered into the shared registry as an import side effect. This is where a fact path, its unit, and its cross-field invariants are validated, and where a document becomes an immutable `Architecture` / `Device` value. |
| `analysis/api.py` | [analysis](./analysis.md) | The public composed Analyze operation: shared authored-program check and normalization, one per-call `AnalyzeContext`, dependency closure, ordering, single execution per member, Metadata-ownership enforcement, and semantic result assembly. |
| `analysis/registry.py` | [analysis](./analysis.md) | The built-in Analyzer declarations, built on the immutable `target.Analyzer` descriptor of `target/analyzer.py`. It registers `builtin_analyzer` as a service and holds no Target dispatch table. |
| `analysis/errors.py` | [analysis](./analysis.md) | `Error`, the one diagnostic the whole analysis layer raises, so catching an analysis failure catches every analysis failure rather than the subset the caller happened to import. |
| `analysis/visitor.py` / `analysis/iteration_scope.py` | [analysis](./analysis.md) | The per-call `AnalyzeContext` carrying the shared root/current `IterationScope`, and the `IterationScope` tree built once from normalized HIR; families query it instead of constructing parallel structure. |
| `analysis/access.py` / `analysis/loop_domain.py` | [analysis](./analysis.md) | Access relations resolved against authored iteration scopes, retaining each input's original boundary index; isl iteration domains built from authored loop bounds. |
| `analysis/footprint.py` / `analysis/reach.py` | [analysis](./analysis.md) | Moving boundaries, reuse axes and reuse windows of a cache-backed level; `ReachedAddresses`, the evidence one boundary leaves before anything is counted, and `Footprint`, the unique bytes one wave touches. |
| `analysis/quantity.py` / `analysis/precision.py` | [analysis](./analysis.md) | `Spread` and `Breakdown` (one quantity in its logical, total and per-unit counting domains, grouped by resource kind), and `Precision` (the direction in which a quantity differs from its true value). |
| `analysis/check.py` | [analysis](./analysis.md) | The shared authored-program gate for analysis: authored-type re-derivation, authored validation, call-context validation, and checker-specific input checks. Established once per public call rather than per family. `program_dim_vars` publicly collects dimension declarations reached through program values and execution geometry. |
| `analysis/allocation.py` / `analysis/liveness.py` | [analysis](./analysis.md) | The buffer allocation and the value liveness the analysis families share. |
| `analysis/metadata.py` | [analysis](./analysis.md) | The typed records the families leave on the IR, split by what each number depends on rather than by convenience. |
| `analysis/compute_cost.py` | [analysis](./analysis.md) | The `compute-cost` family: logical flops per DType and bytes per storage level, from the authored program alone. |
| `analysis/memory.py` | [analysis](./analysis.md) | The `memory` family: per-Call movement plus Function-wide movement, value lifetimes, per-level peaks, and non-fatal capacity errors for over-full addressable levels. |
| `analysis/roofline.py` | [analysis](./analysis.md) | The `roofline` family: the recorded work divided by the target's published rates, per Call and aggregated per Function. Adds no count of its own. |
| `analysis/performance.py` | [analysis](./analysis.md) | The `performance` family: occurrences projected from the shared `IterationScope` tree into flat timeline records and one function envelope, scaled by parallel capacity. It introduces no second scope tree. |
| `dispatch/` | [visitor-registry §3](./visitor-registry.md#3-registry) | The generic `Registry` and `MissingHandlerError`. It imports nothing from the IR. |
| `services/` | [visitor-registry §3.2](./visitor-registry.md#32-services) | The service functions (`evaluate`, `prepare`, `render`, `report_json`, `verify_call`, `describe_refusal`, `refusal_reason`, `builtin_analyzer`, `simplify_dim`, `normalize_dim`, `dim_range`, `specialize`, `converter`) that a lower layer calls and a higher layer implements, and the registry that holds the implementations. It imports no IR. |
| `interfaces/` | [visitor-registry](./visitor-registry.md) | The interfaces a concrete Op implements, one subpackage each. A subpackage owns its registry, its registration function, its context and its visitor; an Op file registers its rules with `from tilefoundry.interfaces import typeinfer` and `@typeinfer.register(Op)`. It sits below `ir.hir` and `ir.tir`, so an Op imports its interfaces and an interface never imports an Op. |
| `interfaces/valuation/` | [visitor-registry §4](./visitor-registry.md#4-instance-1--typeinfer) | `valuation.Context` and `valuation.Visitor`: the lexical environment and call transitions every analysis builds on, the reach of definitions (`valuation.definitions`, `valuation.index_contexts`), region isolation, Function specialization (`specialize.py`), and the enumeration of declared functions (`call_paths.py`: `collect_functions`). |
| `interfaces/typeinfer/` | [visitor-registry §4](./visitor-registry.md#4-instance-1--typeinfer) | `typeinfer.registry` / `typeinfer.register`, `TypeInferContext`, `TypeInferVisitor`, shard propagation (`shard_propagate.py`), input reachability, and the typeinfer of the `Dim*` nodes. |
| `interfaces/verify/` | [visitor-registry §5](./visitor-registry.md#5-instance-2--verify) | `verify.registry` / `verify.register`, `verify.Context` and `verify.Visitor`, and `verify.require_layouts`, the layout check of a value that needs no registered rule. |
| `interfaces/cost/` | [visitor-registry §7](./visitor-registry.md#7-instance-4--cost) | `cost.registry` / `cost.register`, `cost.Context`, `Cost`, `TrafficBytes`, `cost.Visitor`, the construction helpers of `helpers.py`, and the cost of the `Dim*` nodes. Each Op's own per-instance flops and bytes are registered in that Op's file, not by any target package, because the work an operation asks for follows from its own semantics and operand types on every backend. |
| `interfaces/access/` | [visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation) | The access relation of every Op, one concept per module: `relation.py` (`access.Relation`, the carrier of one boundary), `registry.py` (`access.relation_registry`, `access.register_relation`, `access.relations_of`), `projection.py` (a boundary in the coordinates one reader addresses: `access.projected`, `access.local_relations_of`), `iteration.py` (`access.iterating`), `window.py` (windows and control operands), `reach.py` (`access.settled`, `access.reached_elements`: what one boundary reaches), and `builders.py` (`access.identity`, `access.broadcast`, `access.unread`, `access.state`, and the relations of contraction, gather and views). The buffer alias of a view lives in `alias.py` (`access.alias_registry`, `access.register_alias`). |
| `interfaces/evaluator/` | [evaluator](./evaluator.md) | The reference evaluator: `evaluator.registry` / `evaluator.register`, `EvaluateContext`, `evaluator.Visitor`, `evaluate`, and the `Value` family. |
| `interfaces/schedule/` | [visitor-registry §11](./visitor-registry.md#11-instruction-candidate-mappings) | `register_candidates` and the `schedule.pairings` registry it fills, candidate discovery, and the instruction check (`check_call`, `project`, `schedule.RefusalError`). |
| `verifier/` | [visitor-registry §5](./visitor-registry.md#5-instance-2--verify) | Whole-program verification (`checks.py`: `verify_function`, `verify_prim_function`, `verify_module`, `verify_value_layouts`, `input_params`). It sits above `ir.hir` and `ir.tir`, so neither layer owns the driver that reads both. |
| `inspection/analysis_data.py` | [inspection](./inspection.md) | Structured report data of an analysis result: each record projected under its family, its declared field names and the printed names of its values. |
| `inspection/analysis_report.py` | [inspection](./inspection.md) | Presentation of that report data as text and annotated source, and the `report_json` service that `analyze` calls for the `analysis.json` dump. Analysis leaves records on the IR; inspection owns how they are read. |
| `target/<backend>/facts.py` | [target](./target.md) | One backend's Facts projections selected by its Target's `get_facts`. They restate installed documents in the shape a family declared and measure nothing. |
| `target/facts.py` | [target](./target.md) | Every Facts aggregate a Target projects — topology limits, memory execution levels, the memory hierarchy graph, throughput rates and parallel capacity — plus validation for values returned by `Target.get_facts`: the requested frozen-dataclass shape and returned type. It names no backend; this module holds no projection registry. |
| `codegen/` | [codegen](./codegen.md) | Code generation: Target-resolved emission classes; Linkable products in `linkable_function.py` / `linkable_module.py`; Linked products in `linked_function.py` / `linked_module.py`; linking in `link_module.py`; the CPU generator under `cpu/`; every other target's generators live in its own package under `codegen/` (mirroring `ir/tir/` file layout — `tir/<category>/<name>.py` emitter, [§2](#2-file-naming-and-content-rules) rule 2). **Not under `ir/`** — codegen is a consumer of IR; `templates/` (and a target's own `templates/`) holds boilerplate only. |
| `runtime/` | [runtime](./runtime.md) | Runtime support (per-target headers, function templates, launch helpers). |
| `inspection/` | [inspection](./inspection.md) | IR visualisation: DOT, Python printer, web viewer. |
| `dump/` | [inspection](./inspection.md) | Dump flags, dynamically scoped dump contexts, and file/null dump sinks used by inspection and test integration. |
| `dsl/` | [parser](./parser.md) (authoring namespace) | User-facing import surface: `tf/` (HIR namespace) / `T/` (TIR namespace), each with a `dialect.py`, plus `namespace.py` / `stub_gen.py` / `tensor.py` / `storage.py` / `__main__.py`. `dsl/__init__.py` re-exports the authoring decorators of `parser`. The `tf/__init__.pyi` and `T/__init__.pyi` stubs are produced by `python -m tilefoundry.dsl regen` and are gitignored. |
| `compiler/` | [architecture](./architecture.md) | `compiler.build` / `compile` / `jit` (`compilation.py`, `jit_cache.py`) and `compiler.Options` (`options.py`), which gathers the options of every stage. The root re-exports them as `tilefoundry.build` / `compile` / `jit` / `CompilerOptions`. |
| `core/` | [evaluator](./evaluator.md) | Values shared by the evaluator and the runtime: `ShardTensor` and `local_type` (`tensor.py`). |
| `cli/` | [cli](./cli.md) | The `tilefoundry` command: argument parser and one module per command (`models`, `spec`, `tutorial`, `check`, `analyze`, `target`, `schedule`, `scaffold`). |
| `bootstrap.py` | [code-organization](./code-organization.md) | Imports the packages that register implementations and loads the installed Targets, once, when `tilefoundry` is imported. |
| `__init__.py` | [inspection](./inspection.md) | Re-exports of the public compiler surfaces, including `view` (`inspection.viewer.view`); it imports `bootstrap`. |
| `utils/` | [code-organization](./code-organization.md) | Shared leaf machinery: a module here MUST import nothing from `ir/`, `parser/`, `passes/`, `codegen/`, `runtime/` or `cli/`, and MUST name no layer. It is depended on and depends on nothing, which is what lets a consumer outside the package — a pre-commit hook under an interpreter with nothing installed — load one of these modules by path and get the same implementation the package uses. A helper that needs to know a layer belongs in that layer; this is not a home for anything that did not fit. |

**Reading notes:**

- `ir/` holds the IR proper and its sublayers only. `ir/types/` is the
  root of the type system, and the shard / layout types sit in it
  directly ([architecture §3](./architecture.md#3-type-system)): they
  are types, and a folder naming them a sublayer said nothing a reader
  could use. Their filing follows CuTe's, one module per thing the
  algebra is written against.
- `codegen/` and `parser/` sit outside `ir/`: by the pipeline of
  [architecture §1](./architecture.md#1-spec-relationship-map) they are the
  back-end consumer and front-end producer of IR, not IR sublayers.
- `analysis/` sits outside `ir/` because it derives facts about typed HIR
  rather than defining an IR layer. It reads the IR and the `Target` and
  decides nothing over what it measures
  ([architecture §5](./architecture.md#5-analysis--optimization)).
- A target package's `codegen/` consumes TIR; an HIR `Function` enters only
  for skeleton emission. Its TIR subtree mirrors `ir/tir/`, while `hir/`
  holds the skeleton emission of an HIR `Function`.
- Shared launch descriptors belong to `ir/tir/launch.py`; target-specific
  launch attributes live in their target package. The geometry a
  kernel is launched at is settled where the `Launch` is written
  ([passes §7.3](./passes.md#73-inserthostentrypass)) and read back off
  it by `codegen/context_builder.py`, rather than derived a second time inside
  codegen or carried past it as a runtime-owned metadata type. The two launch
  contracts are distinct even though both are consumed across the codegen
  boundary.

`ir/types/layout_algebra.py` has a mechanically checkable public surface:
every name in its `__all__` MUST have the same name in CuTe, with
`is_inverse_projectable` as its sole exception. Python's missing overload
dispatch requires the separately named `supports_composition`,
`supports_inverse`, and `inverse` adapters in `swizzle_layout.py`;
`NotProjectableError` in `layout.py` is the named Python diagnostic needed by
composed application and inversion. These are explicit exceptions, not a
license for unrelated helpers in the algebra modules.

`ir/pattern/`, `ir/clause/`, `dispatch/`, `services/`, `interfaces/`, and `dump/` are cross-cutting packages;
their stable responsibilities are owned by [core-ir](./core-ir.md),
[parser](./parser.md), [visitor-registry](./visitor-registry.md) (the middle
three), and [inspection](./inspection.md), respectively. Their internal file layout is not a per-Op contract.
`ir/pattern/` MUST NOT import `inspection/`, not even inside a function: a failed
match answers with the matcher's `Refusal`, and the consumer that reports it
turns that data into text with `PatternPrinter.refusal`. `inspection/` may
import `ir/pattern/`; the reverse edge does not exist. Likewise target-owned TIR
atoms never import `inspection/`: an atom is a `Printable` from `ir/core/`, a
printer hands itself to `atom.print(printer, ctx)`, and the printer does not
import an atom class.

## 2. File naming and content rules

**Rule 1 — one real Op = one file.** A real Op class lives in
`ir/<hir|tir>/<category>/<op_name>.py`. The file name is the
snake_case of the Op class CamelCase (`MatMul` → `matmul.py`,
`RMSNorm` → `rms_norm.py`). TIR effect Ops and TIR-owned Expr Ops
follow the same rule.

**Rule 1a — surface-alias schemas have no per-name file.** A surface
alias ([core-ir §2.3](./core-ir.md#23-op))
has no IR class — its builder routes to a kinded target Op. The
aliases of an Op are declared on its class by `register_op(..., aliases=...)`
(e.g. the HIR math sugar names `add` / `sub` / `cmp_eq` / … are declared on
`Binary` in `ir/hir/math/binary.py`, and `neg` / … on `Unary` in
`ir/hir/math/unary.py`).

**Rule 1b — tag-dispatched IR classes.** `Binary` / `Unary` /
`Reduce` and other Op classes that fold many surface names through a
`kind` attribute live in one file per IR class
(`ir/hir/math/binary.py` / `ir/hir/math/unary.py` /
`ir/tir/arith.py` / `ir/tir/reduce.py`). This does not contradict
Rule 1: "one Op = one file" means **one IR class** per file; aliases
are not IR classes, so they go through Rule 1a.

**Rule 1c — target-specific code lives in a plugin package.** A backend lives
under `targets/<vendor>/tilefoundry_<vendor>/<name>/`.
Its internal paths preserve the core categories: `target/`, `analysis/`,
`ir/hir/`, `ir/tir/`, `codegen/`, `passes/` and `schedule/` (the
target's candidate pairings, registered through `interfaces/schedule/`). A target
registers its other rules in the files of the nodes that own them, through the
registries of `interfaces/`. Target-owned IR
nodes nest as `ir/{dialect}/{category}/<name>.py` within that package;
target-neutral abstractions stay in core at `ir/{dialect}/{category}/`. A
target-specific rule for a target-neutral Op follows the same dialect and
category path, without redefining the Op class.
The backend-bound construction stays in TIR: HIR is the checking reference
side, and carrying the instruction name in that reference would make two
targets require different HIR references. (`codegen/` and `runtime/` are
**target-first** instead — their primary axis is the target — so each tree is
organized by its own primary axis.)

**Rule 1d — a single carrier may be its category.** When an IR category consists
of one carrier operation named for that category, it may live directly at
`ir/<hir|tir>/<category>.py` instead of the redundant
`ir/<hir|tir>/<category>/<category>.py`. It MUST declare its dialect and category
explicitly. `ScheduleOp` is this exception: it lives in `ir/hir/schedule.py` and
registers as dialect `tf`, category `schedule`.

**Rule 2 — one (node, target) codegen = one file.** Each handler
lives at `<target package>/codegen/tir/<category>/<name>.py`. Stmt emitters,
Expr-Op emitters, and tag-dispatched (`arith`, `reduce`) emitters
each get their own file. HIR skeleton emission is the sole exception to the
TIR input rule: its handler lives at `<target package>/codegen/hir/function.py` and
MUST NOT emit HIR operation bodies.

**Rule 3 — what an IR-class file contains:**

- **HIR Op file** (`ir/hir/<cat>/<name>.py`): Op class + the four rules every `tf` Op
  has ([§0.3](#03-structure-rules)):
  `@typeinfer.register(Op)`, `@evaluator.register(Op)`, `@cost.register(Op)` and
  an access relation (`access.register_relation(Op)`), plus
  `access.register_alias(Op)` when the Op is a view. The Op file imports the
  interface packages (`from tilefoundry.interfaces import typeinfer`), and the
  evaluator of the Op is written in the Op's own file.
- **TIR effect Op file** (`ir/tir/<cat>/<name>.py`): Op class +
  `@typeinfer.register(Op)` (returning `UnitType`) + an access relation +
  `@verify.register(Op)`, and `@evaluator.register(Op)` when a schedule can
  select the Op. The verify rule keys on the Op class
  even though the invocation is an `Evaluate(op, args)` Stmt — see
  [visitor-registry §5](./visitor-registry.md#5-instance-2--verify).
- **TIR-owned Expr Op file** (`ir/tir/memory/{alloc_tensor,ptr_of,memory_span,tensor_view}.py`,
  …): Op class + `@typeinfer.register(Op)` + `@cost.register(Op)`
  (if any). Call-position constraints are checked by the enclosing Stmt's
  `@verify.register`.

**Rule 4 — what a target codegen file contains:** the `codegen.register`
declaration naming that target, `Role.EMIT`, and that op / stmt class
([visitor-registry §6](./visitor-registry.md#6-instance-3--codegen)), and
nothing else. A target's own calling convention — how it declares a parameter
and how a caller passes one — lives in that target's `abi.py`, not in a
per-node file.

**Rule 5 — `<category>/__init__.py` re-export rules:** real Op submodules are
re-exported; user imports go through [parser §2](./parser.md#2-syntax-and-rules).

**Rule 6 — one pass = one file.** A pass class lives in
`passes/transforms/<pass_name>.py`; internal visitors / mutators stay in that file.

**Rule 7 — what template files contain.** A `templates/*.j2` under `codegen/` or a target's `codegen/`
carry boilerplate assembly only; emitters live in Python walkers.

### 2.1 Package export rule

Every package re-exports its public names by importing them in its
`__init__.py`. Other directories import only from the package
([§0.4](#04-exports-and-imports)).

## 3. Multi-agent parallelism guarantee

The lock granularity is a single `(node, target)` pair. The naming rules in
[§2](#2-file-naming-and-content-rules) imply that two agents working on different
`(node, target)` pairs touch disjoint files; cross-cutting changes confine
themselves to the owning directory.

## 4. DSL package layout

The author-facing `tf` and `T` modules delegate attribute lookup to one
`Dialect` instance each. Each level exposes registered operations, alias
builders, types and child namespaces; unknown names raise `AttributeError`.

### 4.1 Built-in op-class location convention

Core operations live under `src/tilefoundry/ir/<hir|tir>/<category>/`;
plugin operations use the same layout within their package. `@register_op`
requires explicit `dialect` and `category` arguments regardless of location.
Its optional `namespace` selects the DSL path independently of that category.

### 4.2 `.pyi` stub regeneration

The `.pyi` stubs reflect the dialect registry, including exported types and
nested namespaces. After changing registrations, regenerate them with
`python -m tilefoundry.dsl regen`.

A plugin's import is its registration boundary. Its package root exports target
values and imports every module that registers operations or implementations.
Hardware TOML, templates, runtime headers and `docs/spec/*.md` are package data;
core locates plugin resources through `importlib.resources`, without repository
path fallbacks. The distribution bundles core and the in-tree plugin packages
in one wheel; plugin packages are not distributed as separate wheels.
