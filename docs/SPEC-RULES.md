# TileFoundry Spec Rules

## Principles

The spec is the single source of truth for TileFoundry public contracts. If a
different implementation of the same public construct would need to know a fact,
that fact belongs in `docs/spec/*.md`, not in code comments or plan files.

Runtime ops use one public entry per op/family. Target runtime headers may split
the implementation into internal helpers, traits, or impl classes, but generated
code calls the public entry and does not select implementation tiers.

Code may carry local-mechanics comments. It does not mechanically backlink every
op definition to the spec. Add a short spec backlink only when a code path exists
solely because a specific spec rule requires it.

## Unified Entry Format

A spec entry shows the construct in its **source defining form**, with every
identifier spelled and cased exactly as in the source: a Python class
(including every HIR / TIR Op) appears as its `class Name(Base):` definition, a
Python function as its `def name(...) -> R:` signature, and a C++ construct as
its struct / class / enum / template declaration. A class MUST NOT be shown as
a call-form signature. Each entry is followed by a `- constraints:` list, where
every normative MUST / SHALL / SHOULD sentence lives.

The interface is concise, never a copy of the implementation: no decorators,
no registration machinery, no `ParamDef` plumbing, no method or function
bodies (a method appears as its `...`-terminated signature). An Op's inputs
and attributes appear as annotated fields, one per line, using the source
field names; attribute defaults are kept (they are interface).

Documentation inside a block follows the industry style of its language, so it
is mechanically checkable:

- **Python — Google docstring style** (validated by ruff's pydocstyle rules,
  `convention = "google"`). The class or function docstring opens with a
  one-line summary (for a value-producing Op, state what it produces; for an
  effect-form Op, say `effect form`). Field roles go in an `Attributes:`
  section (`name: input|attribute; role.`), function parameters in `Args:`,
  results in `Returns:`. Declaration lines carry no trailing role comments.
- **C++ — Doxygen** (`/** @brief ... @tparam ... @param ... @return ... */`
  above each declaration; aggregate members MAY use trailing `///<`).

A fenced block that exists to pin an ambiguous contract corner (usage, not an
interface) starts with a `# example` / `// example` marker line and is exempt
from the format checks. Example code never repeats an implementation. One
block may group a family of related signatures.

Consensus ops may be grouped when one external reference defines their
behavior. Custom TileFoundry ops and public runtime entries need their own
entry. A decorator-based mechanism appears only in the section that owns it —
the custom-op machinery (`@register_op` / `ParamDef`) in
[core-ir §2.3](./spec/core-ir.md), the visitor registries (`@register_*`) in
[visitor-registry](./spec/visitor-registry.md); anywhere else a decorator may
only appear inside an `# example`-marked block.

## Constraints

Spec text is English and records the current contract only. It must not mention
plan files, milestones, task IDs, PR numbers, commit hashes, chat message IDs,
agent names, version stamps, test plans, or future/TODO sections.

A construct has one owning section. Other specs link to that section instead of
restating its definition. Keep examples only when they pin down an otherwise
ambiguous contract corner.

## Entropy And Close Tracking

`scripts/spec_entropy_lint.py` guards Python code comments/docstrings from
re-growing long contract prose. `scripts/spec_rules_lint.py` checks
mechanically-forbidden spec tokens. These tools are review gates, not a
spec-code validator.

Review-led spec work tracks every source comment to one final state:
`implemented`, `verified no-op`, or `resolved by decision`.
