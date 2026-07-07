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

Spec entries for ops, runtime functions, public classes, and pass/helper classes
use one outer shell:

````md
#### Name
```text
signature / ClassName<...>
```
- kind: HIR op | TIR op | runtime func | C++ class | Python class/pass
- fields:
  - name: role or argument/member/effect description
- constraints:
  - contract rule
````

The `fields` content adapts to `kind`: class entries describe members/type
fields; function entries describe arguments, returns, effects, or runtime state.
Do not paste a full dataclass or re-list code-identical fields when a compact
signature and a few stable keys carry the contract.

Consensus ops may be grouped when one external reference defines their behavior.
Custom TileFoundry ops and public runtime entries need their own compact entry.

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
