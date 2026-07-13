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

A spec entry shows the construct's **source-code interface** — the
class / enum / struct / function / op-callable surface with field annotations
and method signatures (ending in `...`, no bodies), each field or method given a
short inline comment for its role — followed by a `- constraints:` list:

````md
#### Name
```python
class Name:
    field_a: TypeA                        # role of field_a
    def method(self, arg: T) -> R: ...    # role of method
```
- constraints:
  - contract rule (every normative MUST / SHALL / SHOULD sentence lives here)
````

Interface only: no method body, traversal, or registry / decorator
implementation. Write the construct in its own language (`python` / `cpp`). One
block may group a family of related signatures. Example code appears only to pin
an ambiguous contract corner, never to repeat an implementation.

An **Op entry** (HIR / TIR) describes the Op class the IR defines: the
CamelCase class construction applied to its inputs and attributes, never a
lowercase function-style spelling. Parameter roles go in a docstring block
above the signature — one parameter per line, `;`-terminated except the last —
and the signature line carries no trailing parameter comment:

````md
##### Cast
```python
"""
x: input tensor;
dtype: target element dtype
"""
Cast(x, dtype) -> Tensor
```
- constraints:
  - contract rule
````

The same leading-docstring parameter block applies to every spec entry that
shows a Python callable (runtime API functions and helpers included).

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
