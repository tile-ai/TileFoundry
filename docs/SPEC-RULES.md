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

Spec entries for ops, runtime functions, public classes, enums, and
pass/helper classes render the construct as a **source-code interface** in its
own language, followed by its constraints:

````md
#### Name
```python
class Name:
    field_a: TypeA                        # role of field_a
    field_b: TypeB                        # role of field_b
    def method(self, arg: T) -> R: ...    # role of method
```
- constraints:
  - contract rule
  - contract rule
````

Rules:

- Show the **interface only** — the class / enum / struct surface, field
  annotations, and method / function signatures. Never include a method body,
  traversal loop, registry / decorator implementation, or any other executable
  detail; a signature ends in `...`.
- Write the construct in its real language: `python` for Python classes /
  functions / decorators / enums, `cpp` for C++ / CUDA enums / classes /
  structs / types / functions.
- Describe each field's and method's role with a **short inline comment** on the
  same line (`# ...` in Python, `// ...` in C++). A longer note goes in a
  docstring / block comment that explains role only, never mechanics.
- An op-catalog entry renders as a callable signature
  (e.g. `Binary(kind, lhs, rhs) -> Tensor`) with inline comments; its owning
  dialect is conveyed by the file / namespace, not a separate label.
- A `- constraints:` bullet list follows the code block and carries the contract
  rules — including every normative MUST / SHALL / SHOULD sentence.
- One code block MAY list several signatures of the same family when that reads
  clearly.
- Do not re-list code-identical fields twice or paste a full implementation; the
  interface surface plus constraints carries the contract.
- Example code appears only to pin an otherwise-ambiguous contract corner (an
  edge case). It never replaces an entry's interface surface and never repeats a
  full implementation.

Consensus ops may be grouped when one external reference defines their behavior.
Custom TileFoundry ops and public runtime entries need their own entry.

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
