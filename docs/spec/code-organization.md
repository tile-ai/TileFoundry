# TileFoundry Spec — Code Organization

Implementation guide (not an architecture spec). It states where a source file
lives, how modules import and export, and which checkers enforce the rules. The
conventions for writing code (S1–S11, C4–C6) are in
[develop §5](../develop.md#5-principles).

## 1. Source tree

The tree shows the source packages. It does not list single `.py` files: the
naming rules of [§2](#2-where-a-new-thing-goes) decide whether a file exists, so
adding an Op needs no spec edit. The file list is `git ls-files src/tilefoundry`.

Each directory has an owning spec, the single source of truth for its structure
and invariants. The number is the layer of that package in the `[tool.importlinter]`
contracts of `pyproject.toml`, counted from the bottom. A directory with no
number is a container or lies outside the contracts.

```text
src/tilefoundry/
  dispatch/       0   The generic Registry and MissingHandlerError. No IR.      visitor-registry §3
  utils/          0   Shared leaf machinery (see below).                        code-organization
  dump/           0   Dump flags, dump contexts and dump sinks.                 inspection
  services/       1   Functions a lower layer calls and a higher layer          visitor-registry §3.2
                      implements, and their registry. No IR.
  ir/                 The IR and its sublayers only.
    expr/         2   Expr, Op, the op registry, DType, StorageKind, DimVar,    core-ir
                      the Dim* operations as data, Functor / Visitor.
    types/        3   Types, layouts, meshes, and TypeFunctor / TypeVisitor /   types, shard,
                      TypeMutator. Files follow CuTe's filing.                  visitor-mutator
    dim/          4   The behaviour of dimensions: simplification, isl,         types §4
                      substitution, MeshCoord.
    utils/        5   Helpers over types: leaves, sizes, bounds,                types
                      compatibility, local views, mesh scope, tiling.
    clause/       7   Authored where(...) constraint records.                   parser
    pattern/      7   Operation-declaration predicates and the matcher.         core-ir
    core/         8   Structure nodes both IR layers share; Expr and Stmt       core-ir,
                      visitors.                                                 visitor-mutator
    hir/         10   HIR Ops by category; schedule.py holds ScheduleOp.        hir
    tir/         10   TIR Ops by category; launch.py holds Launch.              tir
  target/         6   Target, Architecture, Device, Facts, cpu.Target and the   target
                      hardware database (hardware/).
  core/           7   ShardTensor and local_type, shared by the evaluator and   evaluator
                      the runtime.
  interfaces/     9   One subpackage per interface: valuation, typeinfer,       visitor-registry,
                      verify, cost, access, evaluator, schedule.                evaluator
  verifier/      11   Whole-function and whole-module verification.            visitor-registry §5
  analysis/      12   One module per analysis family over typed HIR.            analysis
  parser/        12   DSL to IR parsing and the authoring decorators.           parser
  inspection/    13   Printers, DOT, the viewer and analysis reports.           inspection
  passes/        13   Pass framework, passes.Options, default_pipeline, and     passes
                      transforms/.
  codegen/       14   Code generation, linking, the CPU generator (cpu/) and    codegen
                      templates/.
  schedule/      15   Schedule finalization, candidate discovery and the core   schedule
                      pairings.
  runtime/       15   Runtime support and headers (include/).                   runtime
  compiler/      16   build, compile, jit and compiler.Options.                 architecture
  dsl/           17   The tf / T author namespaces (tf/, T/).                    parser §2
  cli/           17   The tilefoundry command, one module per command.          cli
  bootstrap.py   18   Imports the registering packages and loads the installed  code-organization
                      Targets, once, when tilefoundry is imported.
  __init__.py         Re-exports the public surfaces; it imports bootstrap.     inspection
  orchestrator/       Outside the source rules (§4).
targets/<vendor>/tilefoundry_<vendor>/<target>/
                 20   A backend, above tilefoundry. Its subdirectories are the  target §1.3
                      core categories it fills.
  nvidia: cuda/{analysis, codegen, docs, ir, models, passes, schedule, target}
  apple:  amx/{docs, evaluator, include, ir, runtime, target}
          metal/{docs, ir, passes, target}
```

A package imports only packages on lower rows; the packages of one row do not
import each other. All rows live inside the distribution packages; a row is
not a separate distribution.

`utils/` is shared leaf machinery. A module here MUST import nothing from `ir/`,
`parser/`, `passes/`, `codegen/`, `runtime/` or `cli/`. A module here MUST name
no layer. It depends on nothing. A consumer outside the package can load one of
these modules by path and get the same implementation the package uses. An
example is a pre-commit hook under an interpreter with nothing installed. A
helper that needs to know a layer belongs in that layer. This directory is not a home
for anything that did not fit.

## 2. Where a new thing goes

**Rule 1 — one real Op = one file.** A real Op class lives in
`ir/<hir|tir>/<category>/<op_name>.py`. The file name is the
snake_case of the Op class CamelCase (`MatMul` → `matmul.py`,
`RMSNorm` → `rms_norm.py`). TIR effect Ops and TIR-owned Expr Ops
follow the same rule.

Core operations live under `src/tilefoundry/ir/<hir|tir>/<category>/`;
plugin operations use the same layout within their package. `@register_op`
requires explicit `dialect` and `category` arguments regardless of location.
Its optional `namespace` selects the DSL path independently of that category.

**Rule 1a — surface-alias schemas have no per-name file.** A surface
alias ([core-ir §2.3](./core-ir.md#23-op))
has no IR class — its builder routes to a kinded target Op. An
Op declares its aliases on its class with `register_op(..., aliases=...)`.
Example: `Binary` in `ir/hir/math/binary.py` declares the HIR math sugar names
`add` / `sub` / `cmp_eq` / …. `Unary` in `ir/hir/math/unary.py` declares
`neg` / ….

**Rule 1b — tag-dispatched IR classes.** Some Op classes fold many surface names through a
`kind` attribute. `Binary` / `Unary` /
`Reduce` are examples. Each such class lives in its own file
(`ir/hir/math/binary.py` / `ir/hir/math/unary.py` /
`ir/tir/arith.py` / `ir/tir/reduce.py`). This does not contradict
Rule 1. "One Op = one file" means **one IR class** per file. Aliases
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
The backend-bound construction stays in TIR. HIR is the checking reference
side. An instruction name in that reference would make two
targets require different HIR references. `codegen/` and `runtime/` are
**target-first**: their primary axis is the target. Each tree follows its own
primary axis. Packaging, entry points and package
resources of a plugin follow [target §1.3](./target.md#13-target-loading-and-extensions).

**Rule 1d — a single carrier may be its category.** An IR category can consist
of one carrier operation named for that category. Then the operation MAY live
directly at `ir/<hir|tir>/<category>.py` instead of the redundant
`ir/<hir|tir>/<category>/<category>.py`. It MUST declare its dialect and category
explicitly. `ScheduleOp` is this exception: it lives in `ir/hir/schedule.py` and
registers as dialect `tf`, category `schedule`.

**Rule 2 — one (node, target) codegen = one file.** Each handler
lives at `<target package>/codegen/tir/<category>/<name>.py`. Stmt emitters,
Expr-Op emitters, and tag-dispatched (`arith`, `reduce`) emitters
each get their own file. HIR skeleton emission is the sole exception to the
TIR input rule. Its handler lives at `<target package>/codegen/hir/function.py`.
Its handler MUST NOT emit HIR operation bodies.

**Rule 3 — what an IR-class file contains:**

- **HIR Op file** (`ir/hir/<cat>/<name>.py`): Op class + the four rules every `tf` Op
  has ([develop Functions](../develop.md#functions)).
  The rules are `@typeinfer.register(Op)`, `@evaluator.register(Op)`,
  `@cost.register(Op)` and an access relation (`access.register_relation(Op)`).
  When the Op is a view, the file also has `access.register_alias(Op)`. The Op
  file imports the interface packages (`from tilefoundry.interfaces import typeinfer`).
  The Op's own file holds the evaluator of the Op.
  An Op that can be differentiated also has `@autodiff.register(Op)`
  ([visitor-registry §12](./visitor-registry.md#12-instance-5--autodiff)).
  `differentiate` refuses an Op that has none and names it.
- **TIR effect Op file** (`ir/tir/<cat>/<name>.py`): Op class +
  `@typeinfer.register(Op)` (returning `TupleType.Void`) + an access relation +
  `@verify.register(Op)`, and `@evaluator.register(Op)` when a schedule can
  select the Op. The verify rule keys on the Op class
  even though the invocation is an `Evaluate(op, args)` Stmt — see
  [visitor-registry §5](./visitor-registry.md#5-instance-2--verify).
- **TIR-owned Expr Op file** (`ir/tir/memory/{alloc_tensor,ptr_of,memory_span,tensor_view}.py`,
  …): Op class + `@typeinfer.register(Op)` + `@cost.register(Op)`
  (if any). Call-position constraints are checked by the enclosing Stmt's
  `@verify.register`.

The file of an Op registers its per-instance flops and bytes. No target package
registers them. The work an operation asks for follows from its own semantics
and operand types on every backend.

**Rule 4 — what a target codegen file contains:** the `codegen.register`
declaration naming that target, `Role.EMIT`, and that op / stmt class
([visitor-registry §6](./visitor-registry.md#6-instance-3--codegen)), and
nothing else. A target's calling convention lives in that target's `abi.py`, not in a
per-node file. The convention states how the target declares a parameter and
how a caller passes one.

**Rule 5 — `<category>/__init__.py` re-export rules:** real Op submodules are
re-exported; user imports go through [parser §2](./parser.md#2-syntax-and-rules).

**Rule 6 — one pass = one file.** A pass class lives in
`passes/transforms/<pass_name>.py`; internal visitors / mutators stay in that file.

**Rule 7 — what template files contain.** A `templates/*.j2` under `codegen/` or a target's `codegen/`
carry boilerplate assembly only; emitters live in Python walkers.

Shared launch descriptors belong to `ir/tir/launch.py`; target-specific launch
attributes live in their target package. A target package's `codegen/` consumes
TIR; its `hir/` holds only the skeleton emission of an HIR `Function`.

The internal file layout of the cross-cutting packages (`ir/pattern/`,
`ir/clause/`, `dispatch/`, `services/`, `interfaces/`, `dump/`) is not a per-Op
contract.

These rules give two agents that work on different `(node, target)` pairs
disjoint files; a cross-cutting change stays in its owning directory.

## 3. Imports and exports

| Rule | Statement | Checker |
|---|---|---|
| C1 | No import inside a function. A heavy third-party module is loaded lazily through `tilefoundry.utils.lazy`. | ruff `PLC0415` |
| C2 | The package import graph is acyclic and obeys the layer table of [§1](#1-source-tree). The module import graph within one directory is acyclic. | import-linter: the `layers` contract and the whole-graph contract in `scripts/importlinter_acyclic.py` |
| C3 | A module other than `__init__.py` has a literal `__all__`; an `__init__.py` only re-exports and writes no `__all__` ([§3](#3-imports-and-exports)). | `scripts/code_organization_lint.py` |
| C10 | A module does not rebind a namespace it imports. No parameter, local, loop or `with`/`except` target, function or class definition, or module-level assignment takes its name. | `scripts/code_organization_lint.py` |
| C11 | Imports take the forms of [§3](#3-imports-and-exports). | `scripts/code_organization_lint.py` |

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
  exported by that package's `__init__.py` or listed in that module's `__all__`.
  Deeper levels (`ir.utils.leaves_of`) are not constrained further.
  A module that provides members through a module-level `__getattr__` (`tf`, `T`) is exempt.
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

Every package re-exports its public names by importing them in its
`__init__.py`. Other directories import only from the package
([§3](#3-imports-and-exports)).

Re-exporting through `__init__.py` is safe under C2 and this section together.
Python runs every ancestor `__init__.py` before a submodule. So an ancestor is
half initialized while its submodule loads. The submodule cannot import it. A
package's `__init__.py` loads only lower layers. Lower layers finish first.

`ir/pattern/` MUST NOT import `inspection/`, not even inside a function. A failed
match answers with the matcher's `Refusal`. The consumer that reports it
turns that data into text with `PatternPrinter.refusal`. `inspection/` may
import `ir/pattern/`; the reverse edge does not exist. Likewise target-owned TIR
atoms never import `inspection/`. An atom is a `Printable` from `ir/core/`. A
printer hands itself to `atom.print(printer, ctx)`. The printer does not
import an atom class.

A cycle is broken according to its cause:

| Cause | Fix |
|---|---|
| An import serves only a type annotation | Quote the annotation and drop the import |
| A catch-all module joins several concepts and becomes a hub | Split it by concept |
| A function of a lower-layer class uses a higher-layer concept | Move the function to the higher layer; the lower layer keeps the data |
| A lower layer must call an implementation that lives higher | Call it through `services`. Declare the service function on row 1. The implementing package registers its implementation at import. `bootstrap` imports that package. |
| A package spans two rows | Split the package by row |

## 4. Checkers

These rules bind every Python module under `src/tilefoundry/` and
`targets/<vendor>/tilefoundry_<vendor>/`, except `src/tilefoundry/orchestrator/`
and the `models/` trees. A rule exists only where a checker enforces it; a
convention no tool can check is not listed. No source line carries a tool
directive: every exception is written in `pyproject.toml`
([§4](#4-checkers)). The same scope binds S1–S11 and C4–C6 of
[develop §5](../develop.md#5-principles).

The [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
is the reference for anything the rules below leave open. It is not a rule
itself, because conformance to the whole guide cannot be checked. This spec
overrides the guide in three places. C3 replaces "import modules only" and "no
relative imports". C3 visibility and C4 replace the `_` prefix for internal
names.

| Rule | Statement | Checker |
|---|---|---|
| C8 | No comment is a tool directive (`noqa`, `type:`, `pragma:`, `ruff:`, `pylint:`, `pyright:`, `mypy:`, `fmt:`, `isort:`, `hygiene:`). | `scripts/comment_hygiene_lint.py` |
| C9 | Every module and package has a docstring. | ruff `D100`, `D104` |

Every rule applies to every Python file of the repository. An exception is a
path glob and the rule codes it ignores. Write it only in `pyproject.toml`, in the
section of the tool that checks the rule. The sections are `[tool.ruff.lint.per-file-ignores]`
(and `pep8-naming` for S10), `[tool.code_organization.per-file-ignores]`,
`[tool.comment_hygiene]` (`exempt`), and `[tool.importlinter]`. Tests and
scripts are not imported as library code, so of the rules this tool checks
only C10 and C11 bind them. `.pre-commit-config.yaml` declares the hooks and
carries no rule arguments.
