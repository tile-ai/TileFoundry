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
python -m pip install -e '.[test]'
```

`pyproject.toml` declares runtime deps (`apache-tvm-ffi`, `torch`),
the `[test]` extra (`pytest`), and the jinja `*.j2` templates as
package data. After this `import tilefoundry` resolves to
`src/tilefoundry/...` and plain `pytest tests/` works from repo root.
**Do not set `PYTHONPATH`** — in particular `PYTHONPATH=tests:src`
shadows stdlib `types` via `tests/types/__init__.py`.

## 3. Tests

`pytest tests/` runs the **entire** suite — unit, codegen, nvcc-compiled,
and GPU end-to-end — with no marker gating. There is no `--gpu` flag or
`-m nvcc` opt-in: a machine without `nvcc` or a CUDA device fails (not
skips) the compile / e2e tests, so end-to-end coverage is never silently
hidden. Per-test dump output lands under
`test_results/{file_stem}/{test_name}/...` (gitignored).

Common inner loops:

```sh
pytest tests/passes tests/ir tests/ir_types -q   # IR / passes edits
pytest tests/codegen tests/runtime -q            # codegen-text edits
pytest tests/e2e -q                              # runtime header / host wrapper edits
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

## 5. Principles

Cross-cutting rules every change MUST honour. Each section is a
short bullet list — keep it that way.

### Code comments

- Describe local logic only.
- The spec is the only durable document a comment may reference
  (`docs/spec/... §...`). Never reference `docs/plans/...` — plans are
  working-process docs, not a stable contract.
- A constraint that must guide code long-term belongs in the spec
  first; the comment then cites the spec, never the plan that proposed
  it.
- No milestone / version references.
- No PR / task / commit / issue / msg / thread coordinates.
- No agent or human names; no discussion / review narration.
- No prose cataloguing what the code does NOT do.

### Scope

- A commit touches only what the current task requires.
- Adjacent / autoformat / submodule changes ship in separate commits.
- Out-of-scope changes that cannot be split MUST be called out in the
  commit body.

### Spec impact

- Every milestone declares `#### Spec Impact`, including documentation-only
  and internal-only milestones.
- A public-contract change lists its owning `docs/spec/*.md` files and repeats
  those files in the milestone's effective `#### Related Files`.
- A milestone without public-contract changes uses exactly one reasoned
  `N/A:` entry; it never mixes `N/A:` with spec paths.
- Contract-changing implementation and its specification ship in the same
  milestone.

### Forward references

- Break type-only import cycles with quoted forward-reference annotations.
- Do not guard type-only imports with `typing.TYPE_CHECKING`; Ruff rejects that
  shim.
- Import runtime dependencies normally or lazily at their point of use.

### Tests

- Write meaningful positive tests that exercise the intended path.
- Avoid excessive defensive / catch-all tests; failures on real paths
  are the signal to fix.
- Lock contracts, not implementation detail.

### DSL / HIR authoring

- No docstring in an `@func` body (parser rejects bare expressions).
- Use `tf.<op>` attribute path; do not alias individual ops.
- Variadic ops take positional inputs; attributes go by keyword.
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
- Headers: the umbrella entry header is `<tilefoundry/runtime.h>`; cuda target
  headers use `.cuh`, target-neutral / cpu headers use `.h`; exactly one
  `TILEFOUNDRY_TARGET_*` macro is defined per translation unit and the build
  injects it per target.
