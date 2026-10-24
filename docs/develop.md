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

## Writing a target

Start with [targets/apple](../targets/apple) as a small working example; use
[targets/nvidia](../targets/nvidia) for a complete compiler and runtime reference.
The Apple project keeps its `pyproject.toml`, `setup.py`, and `README.md` beside
`tilefoundry_apple/`. Its top-level `__init__.py` exports the public target values,
while the implementation lives in the `amx/` subpackage. The packaging entry point
is `amx = "tilefoundry_apple.amx"`.

Importing the target subpackage loads its registrations. The Apple example uses
`register_target` for its target class, `register_op` for instructions,
`register_typeinfer` for their inference rules, and `register_candidates` to offer
AMX instructions for a generic operation. For exported DSL types, use
`register_type`; its `namespace` parameter, like that of `register_op`, determines
the path under `T`. For example, NVIDIA's
[wgmma.py](../targets/nvidia/tilefoundry_nvidia/cuda/ir/tir/nn/wgmma.py) uses
`namespace="cuda.sm90"` to expose types under `T.cuda.sm90`. Use `register_codegen`
and `register_pass` when adding code generation and compiler passes.

Keep tests in `tests/<vendor>/<target>/` and specifications in the target
subpackage's `docs/spec/` directory.

Use [Target loading and extensions](spec/target.md#13-target-loading-and-extensions)
(§1.3) as the contract for package boundaries, entry points, registration, and
loading metadata. The Apple example publishes `tilefoundry-apple`; its `setup.py`
pins `tilefoundry==V` to the version `V` being built. Follow the same project
structure when adding a vendor distribution.

## 3. Tests

`pytest tests/` runs the **entire** suite — unit, codegen, nvcc-compiled,
and GPU end-to-end — with no marker gating. There is no `--gpu` flag or
`-m nvcc` opt-in: a machine without `nvcc` or a CUDA device fails (not
skips) the compile / e2e tests, so end-to-end coverage is never silently
hidden. Per-test dump output lands under
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

- How a milestone is accepted is settled with the author while the plan is
  written, in its `##### Accepted by`; `scripts/finalize_plan_context.py`
  refuses to finalize a plan that leaves that choice open, and prints the
  contract from `docs/policies/project-policy.json`.
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
  (`program_shape`, `make_shard_tensor`); **template parameters** are
  `T`-prefixed `PascalCase` (`TLayout`, `TMesh`, `TAttrs`) — no trailing
  underscore.
- Headers: each target's preamble directly includes its package-owned runtime
  header. There is no umbrella header or target-selection macro. Device headers
  use `.cuh` where required by their compiler; ordinary C++ headers use `.h`.
