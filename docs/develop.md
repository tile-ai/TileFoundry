# Developer

How to bring up a dev/test loop for TileFoundry, and how we work on it.

## 1. Env

Use a dedicated **`tilefoundry-dev`** conda env (Python ≥ 3.12 + torch
built against your local CUDA). Cloning an existing torch+CUDA env
is easiest. Do not install tilefoundry into `base`.

```sh
conda create --clone <torch_env> -n tilefoundry-dev
conda activate tilefoundry-dev
```

Record where your env lives in a gitignored `.dev-env` at repo root
(informational; nothing reads it automatically):

```sh
echo CONDA_ENV=tilefoundry-dev > .dev-env
```

## 2. Install

```sh
git submodule update --init --recursive   # third_party/cutlass
python -m pip install -e '.[test]' -e targets/nvidia -e targets/apple
```

When upgrading a checkout from before the distribution split, delete the
repository-root `tilefoundry.egg-info` before reinstalling all three editable projects.

The core distribution declares runtime deps (`apache-tvm-ffi`, `torch`) and
the `[test]` extra (`pytest`). Each target distribution provides its own package
data and registration entry point. After this `import tilefoundry` resolves to
`src/tilefoundry/...` and plain `pytest tests/` works from repo root.
**Do not set `PYTHONPATH`** — in particular `PYTHONPATH=tests:src`
shadows stdlib `types` via `tests/types/__init__.py`.

## 3. Tests

`pytest tests/` runs the **entire** suite — unit, codegen, nvcc-compiled,
and GPU end-to-end — with no marker gating. There is no `--gpu` flag or
`-m nvcc` opt-in. A machine without `nvcc` or a CUDA device fails the
compile and e2e tests. It does not skip them. End-to-end coverage is never
silently hidden. Per-test dump output lands under
`test_results/{file_stem}/{test_name}/...` (gitignored).

Set `TILEFOUNDRY_CACHE_DIR` to keep compile results across runs. The test session
defaults `CCACHE_DIR` to its `ccache/` subdirectory and `TORCH_EXTENSIONS_DIR` to
its `torch_extensions/` subdirectory, creating both directories. With `ccache`
on `PATH`, it defaults `CMAKE_CUDA_COMPILER_LAUNCHER` and
`CMAKE_CXX_COMPILER_LAUNCHER` to `ccache`. `CCACHE_BASEDIR` follows the process's
temporary directory (`TMPDIR`), and `CCACHE_NOHASHDIR` defaults to `1`. Existing
environment values take precedence. Only the main test process removes torch
extension locks older than 30 minutes. Without a cache root, torch extensions
stay under `<repo>/.torch_extensions` and no compiler launcher is configured.
CI uses `/__w/tf-cache` as its cache root.

Common inner loops:

```sh
pytest tests/core/passes tests/core/ir tests/nvidia/cuda/ir -q  # IR / passes edits
pytest tests/nvidia/cuda/codegen tests/nvidia/cuda/runtime -q  # codegen-text edits
pytest tests/nvidia/cuda/integration -q  # runtime header / host wrapper edits
```

## 4. Development workflow

- `main` is the integration branch. Develop on `task/<id>-<short>`
  branches and merge via PR.
- Discuss requirements / design / trade-offs before any code change.
- Plan: draft under `docs/plans/<name>.md` against `docs/plans/TEMPLATE.md`,
  then run `scripts/finalize_plan_context.py <plan>` to inject the
  plan-level Preflight + per-milestone policy ACs from
  `docs/policies/project-policy.json`.
- Reviewers read the finalized plan as the implementation contract.
- After approval, implement milestone-by-milestone: claim a task,
  cut the branch, code + test, post results, open the PR.
- One commit per milestone batch; commit messages carry the milestone
  tag. No `--amend`, no force-push to `main`, no `--no-verify`.
- Claiming done requires test evidence.
- Commit, issue and pull request titles, the pull request body, branch
  names and the review loop: [CONTRIBUTING.md](../CONTRIBUTING.md).

## 5. Principles

Cross-cutting rules every change MUST honour. Each section is a
short bullet list — keep it that way.

### Code comments

- Documentation lives in a module / class / function docstring. A `#` line is
  for a tool (`ruff:` / `noqa` / `type:` / `pragma:` / `mypy:` / `fmt:` /
  `isort:`) or a shebang, never for a reader. In C / C++ / CUDA the docstring's
  place is a Doxygen block above the declaration (`/** */`, `///`, trailing
  `///<`); `//` and `/* */` are not used.
- A docstring spends at most 8 prose lines and stays inside 100 columns.
  Google sections (`Args:` / `Returns:` / `Raises:` / `Yields:` /
  `Attributes:`) are structure, not prose, and do not count.
- Docstrings follow Google style, validated by ruff `D2,D3,D4` under
  `convention = "google"`.
- Describe local logic only.
- The spec is the only durable document a docstring may reference
  (`docs/spec/... §...`). Never reference `docs/plans/...`.
- A constraint that must guide code long-term belongs in the spec first.
- No milestone / version references.
- No PR / task / commit / issue / msg / thread coordinates.
- No agent or human names; no discussion / review narration.
- No prose cataloguing what the code does NOT do.
- A line that genuinely names one of these shapes says `hygiene:` and why. It
  excuses that line from the reference checks alone, never from placement,
  length or width.

### Scope

- A commit touches only what the current task requires.
- Adjacent / autoformat / submodule changes ship in separate commits.
- Out-of-scope changes that cannot be split MUST be called out in the
  commit body.

### Spec impact

- A milestone that changes a public contract lists its owning `docs/spec/*.md`
  files in its `#### Related Files`; one that changes none lists no spec path.
  There is no separate declaration to keep in step with that list.
- Contract-changing implementation and its specification ship in the same
  milestone.

### Forward references

- Break type-only import cycles with quoted forward-reference annotations.
- Do not guard type-only imports with `typing.TYPE_CHECKING`; the
  `forward-references` pre-commit hook rejects that shim. Ruff cannot be asked
  to do it — its own `TC001`-`TC003` push the other way.
- Import runtime dependencies normally or lazily at their point of use.
  Import torch through the proxies in `tilefoundry.utils.lazy` so
  `import tilefoundry` does not load torch.

### Tests

- Settle with the author how a milestone is accepted, while the plan is
  written, in its `##### Accepted by`.
  `scripts/finalize_plan_context.py` refuses to finalize a plan that leaves that choice open.
  It prints the contract from `docs/policies/project-policy.json`.
- Each milestone defines its own Target State Design before implementation.
  Show every part the milestone settles in its delivered form as code or compact
  pseudocode. Implementation and review match that design.
- A Target State Design covers every step its milestone lists. A step with no
  delivered shape is a reviewer's finding before implementation begins.
- One workflow may evidence several acceptance criteria. An acceptance criterion
  describes behaviour, not a one-test obligation.
- Lock contracts, not implementation detail. Do not test source text, AST shape,
  private calls, object identity, counts, or names merely to prevent a future
  refactor.

### DSL / HIR authoring

- A leading `@func` docstring is accepted; strings in nested blocks remain bare expressions.
- Use `tf.<op>` attribute path; do not alias individual ops.
- Variadic ops take one explicit list, tuple, or supported static list comprehension;
  attributes go by keyword.
- Every `@func` parameter MUST reach the return through real ops;
  dead `_ = expr` assignments do not count.

### C++ formatting

- C++/CUDA sources follow the repo `.clang-format` (clang-format ≥ 18, LLVM
  base with `IndentWidth: 4`, `ColumnLimit: 80`, `PointerAlignment: Right`,
  `Standard: c++17`). Run `pre-commit install` once; the versioned
  `.pre-commit-config.yaml` then formats every touched
  `*.h`/`*.hpp`/`*.cuh`/`*.cu`/`*.cpp`/`*.cc` with `clang-format -i` on commit.
  An equivalent manual check is `clang-format --dry-run -Werror <files>`.
- Formatting scope is **touched files only** — never run a tree-wide reformat in
  a feature commit (that belongs in its own isolated commit).
- Naming: **types** (`struct`/`class`/`enum`/`using` aliases) are `PascalCase`
  (`Topology`, `Mesh`, `ShardLayout`); **functions** are `snake_case`
  (`topology_shape`, `make_shard_tensor`); **template parameters** are
  `T`-prefixed `PascalCase` (`TLayout`, `TMesh`, `TAttrs`) — no trailing
  underscore.
- Headers: each target's preamble directly includes its package-owned runtime
  header. There is no umbrella header or target-selection macro. Device headers
  use `.cuh` where required by their compiler; ordinary C++ headers use `.h`.

### Naming

The terms, and the rules N1–N7 for using them in names, are in [terms](./spec/terms.md).

- S1–S11 and C4–C6 in this section bind the Python sources in the scope of [code-organization §4](spec/code-organization.md#4-checkers), with its exceptions.
- S10: Classes are `CapWords`; functions, methods and variables are `snake_case`; constants are `UPPER_CASE`; an exception class name ends in `Error`. Reflection dispatches some method names on a class name (`<prefix>_<ClassName>`: `visit_*`, `visit_leaf_*`, `print_*` and the pattern printer's readers).
  `pyproject.toml` lists these names in `ignore-names`.
  It lists the DSL's authored names (`S`, `P`, `B`, …) there too. (checker: ruff `N`)
- C4: No name starts with `_`: files, classes, functions, methods, attributes, variables, parameters and import aliases. Dunder names and a lone `_` are exempt. (checker: `scripts/code_organization_lint.py`)
- S11: A public class or function name does not repeat its package's name.
  The classes and functions of package `p`, and the names `p` exports, do not contain `p`.
  A name that is exactly `p` (`cost.Cost`) is allowed. Module file names are not constrained. (checker: `scripts/code_organization_lint.py`)
- `ir/types/layout_algebra.py` has a mechanically checkable public surface.
  Every name in its `__all__` MUST have the same name in CuTe. `is_inverse_projectable` is the sole exception.
  Python has no overload dispatch, so `swizzle_layout.py` needs the separately named `supports_composition`, `supports_inverse`, and `inverse` adapters.
  `NotProjectableError` in `layout.py` is the named Python diagnostic for composed application and inversion.
  These are explicit exceptions. They are not a license for unrelated helpers in the algebra modules.

### Data

- S8: A class that derives from a frozen dataclass and declares fields of its own is itself `@dataclass(frozen=True)`. A frozen dataclass's `__post_init__` writes only through `object.__setattr__(self, "<own field>", ...)`. (checker: `scripts/code_organization_lint.py`)
- C5: An attribute, or an element of an attribute, is written only through the `self` / `cls` of the method that writes it. No `setattr`, `delattr`, `object.__setattr__`, `vars(x).update` or `x.__dict__` change is applied to another object.
  A method may change a copy of its own instance that it just made with `copy.copy(self)`. (checker: `scripts/code_organization_lint.py`)
- C6: No `global`. A module-level container is immutable (`tuple`, `frozenset`, `MappingProxyType`); `__all__` is exempt. Mutable runtime state lives in a class instance and is written by that instance's own methods (C5). (checker: `scripts/code_organization_lint.py`)
- S7: A `@property` body is not only `return self.<field>`; a setter body is not only `self.<field> = value`. (checker: `scripts/code_organization_lint.py`)

### Construction

- S2: `@classmethod` is used only for a named constructor: `from_<source>`, or `create` for a factory protocol whose subclass decides the implementation. (checker: `scripts/code_organization_lint.py`)
- S9: A named constructor that uses only its class's layer and lower ones is a `@classmethod` `from_<source>` (S2). One that uses a higher layer is a module function `make_<name>` whose return type is defined in another module. (checker: `scripts/code_organization_lint.py`; the layer half by C2)
- S9 recognizes its subject by name (`make_*`), so a rename evades it.

### Functions

- S1: No `@staticmethod`; write a module function. (checker: `scripts/code_organization_lint.py`)
- S4: No nested function or class, unless it closes over a local of the enclosing function other than `self` / `cls`. (checker: `scripts/code_organization_lint.py`)
- S5: No custom metaclass, `__del__`, `__import__` or `importlib.import_module`. A plugin loading point that imports by name carries a per-file ignore ([code-organization §4](spec/code-organization.md#4-checkers)). `getattr` / `hasattr` are allowed for dispatch. (checker: `scripts/code_organization_lint.py`)
- S6: A parameter default is neither a mutable value nor a function call. (checker: ruff `B006`, `B008`)
- Data lives in the low layer and behaviour in the high layer. A lower layer calls a higher implementation through `services` ([code-organization §3](spec/code-organization.md#3-imports-and-exports)).
- A key with no registration raises `MissingHandlerError` where it is looked up; the tests cover the registrations. Write a registration next to its key's owner.
  Put an Op's registrations in the Op's own module. Put a Target's in that Target's package.
  Put a core HIR-to-TIR pairing in the `schedule` package.
- A walk over Exprs, Stmts or types uses a `Functor`, `Visitor` or `Mutator` of [visitor-mutator](spec/visitor-mutator.md), `TypeVisitor` and `TypeMutator` included.

## 6. Writing a target

Use [targets/apple](../targets/apple) as a minimal example and
[targets/nvidia](../targets/nvidia) as the complete compiler/runtime reference.
Follow the [target loading and extension contract](spec/target.md#13-target-loading-and-extensions)
in §1.3 for packaging, entry points, and registration.
Keep tests in `tests/<vendor>/<target>/`.
