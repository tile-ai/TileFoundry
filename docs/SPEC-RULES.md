# TileFoundry Spec — Rules

How `docs/spec/*.md` is written, and how spec and code divide responsibility.
Principle first; constraints below. No fixed template — pick the form that fits
each construct.

## Principle

**The spec is the single source of truth for the public surface.** Every
contract and every long-lived design decision lives in `docs/spec/*.md` and
nowhere else. Code carries a one-line statement of purpose and comments that
explain only local mechanics. A later reader — human or agent — learns *what a
construct promises and why the system is shaped this way* by reading the spec,
not by reassembling it from docstrings scattered across the source.

There is **no spec↔code cross-validation**. The spec is authoritative; code is
neither checked against it nor restates it. This keeps a single, clean reading
entry point and stops the same fact from drifting in two places.

**Simplicity above all.** A spec section reads like a contract, not an
implementation log. If one sentence is enough, do not write a section.

A spec MUST be written in English. Existing zh-CN passages SHOULD be
rewritten on first touch.

## What the spec owns

The spec is the design document for the **public surface**. It owns exactly
three kinds of content:

1. **Op contracts** — for each HIR / TIR op, its fields, typing / verifier
   rules, and the worked examples that pin down otherwise-ambiguous corners.
2. **Runtime public-function contracts** — the interface a caller depends on
   for each public runtime entry (what it does and what it guarantees), not its
   internal helper signatures.
3. **Cross-function / system design principles** — the invariants and
   ownership decisions that span constructs (e.g. runtime-owned dispatch,
   cross-CTA fence ownership) that a reimplementation would still need to honor.

Litmus test: *would a different implementation of the same construct still need
to know this?* If yes, it is a contract or a design principle and belongs in the
spec. If it only explains why these particular lines are written this way, it is
a code comment.

Internal helpers, private utilities, and per-tier implementation signatures are
**not** part of the public surface and MUST NOT appear in the spec — describe
the one public entry and the principle by which it dispatches, never the tiers.

## Op catalog

An **Op** is spec'd as a catalog entry under a **namespace-leveled heading**
that mirrors the op's module namespace (e.g. `## tensor` → `### insert_slice`).
The op name is the pointer: a reader finds the contract by the op's name, and
code carries no back-link to it.

A **custom Op** (its own heading) records its **full contract** in that entry —
fields, typing / verifier rules, and worked examples. A **consensus Op**
(`add`, `relu`, `matmul`, ...) needs only a single sentence, or a grouped entry
with one stable external link covering the group (e.g.
`#### Reshape / Transpose / … — torch structural ops`); its behavior is defined
by that external reference, so no local contract is written.

Design / architecture guidance (dispatch principles, cross-layer ownership)
lives in the spec prose, outside the per-op entries.

## Section structure

This applies to **non-Op** constructs (e.g. `Function`,
`GridRegionExpr`, a `TensorType`); Ops follow the Op-catalog rule
above. A construct that is a `@dataclass` (or has the moral
equivalent — named fields with stable identity) MUST be introduced as:

1. one short opening sentence — what it is, what role it plays
2. the `@dataclass` code block as the definition of truth
3. a multi-level heading per field, in the order the fields appear
   in the dataclass, with sub-headings for each named sub-field that
   carries its own contract
4. additional sub-sections (cross-construct invariants, examples,
   an optional design-rationale section — see below) only after the
   field walk-through, never interleaved

Each field section MUST be written in formal style: bullet rules with
`MUST` / `MAY` / `SHOULD` (RFC 2119), equations, and set / index
notation. Outside the optional design-rationale section below, avoid
explanatory prose paragraphs ("here X means …", "the reason is …"); if a
paragraph reads like commentary rather than a rule or a definition, drop it.

A field section MAY include a short worked example after the rules,
when an example pins down a corner of the contract that the rules
alone leave ambiguous.

A construct's section MAY carry one **Design rationale** subsection,
placed after the field walk-through, when the *why* behind the contract
is not obvious from the rules and is worth preserving. Keep it to a
sentence or two that point at the intent — not a
restatement of the rules, not an essay. It is the one place commentary
is allowed; it MUST still be English. A routine construct needs none.

## Constraints

A spec section MAY reference other spec sections (`<file> §X.Y`) or
external public knowledge (with a stable URL). References are inline
where they are used; do not maintain a "Related specs" or "See also"
header / footer block.

A spec section MUST NOT reference any of:

- a plan under `docs/plans/`
- a milestone identifier (`M0`, `M1a`, `(M3 sync)`, ...)
- a task ID (`task #87`, `#73`, ...)
- a commit hash, pull-request number, or other VCS coordinate
- an agent name (Alice, Bob, ...) or human name
- a chat thread / message ID
- the literal `æ` annotation marker
- a version stamp (`V1`, `V2`). Spec records the
  current contract; previous shapes are not in scope.

A spec section MUST NOT carry milestone or sync markers (`(M0 sync)`
and the like) in its title.

A spec MUST NOT carry a `Non-goals` / 非目标 / "Out of scope" /
"Future / TODO" section, nor inline prose that catalogues what the
spec deliberately does not cover ("we do not consider X", "X is left
for future work", etc.). What is not in the repo is not part of the
contract; listing exclusions only invites confusion. State the
positive contract; if a boundary needs to be drawn, draw it inline
where the relevant construct is defined.

A spec MUST NOT carry a `测试要求` / `Tests` / `Testing` /
"Test plan" section, nor a list of test names that should pass. The
spec records the contract; tests live in `tests/` and are owned by
the test suite. If a test name appears in spec text, drop it.

A spec MUST NOT embed implementation detail or a recipe that does not
belong to its own owning surface. Each construct, and each cross-layer
translation recipe, lives in exactly one owning spec. A lowering
recipe lives in the lowering-pass spec when that pass owns the
translation (e.g. the cute MMA fragment → `ShardLayout` rewrite lives
in the `HirToTirPass` spec); a target-side emission detail lives in the
target / codegen spec.

A construct has exactly one owning section. Every other spec
references it and MUST NOT restate its definition or normative rules:
a shared fact lives once, at its owner, and is linked — never copied.

For finite enumerations (dtype, storage class, ...) state the rule
and a representative subset; exhaustive enumeration is not required.

Use **MUST** / **MUST NOT** / **SHOULD** / **MAY** (RFC 2119) only
inside contract sentences.

## Code side

A code module SHOULD open with a **one-line docstring** stating its purpose.
It MUST NOT restate the spec: no field-by-field contract, no recap of typing
rules, no design-principle essay. Comments explain **local mechanics only** —
why these specific lines are written this way — never a construct's contract or
a system-wide design decision.

There is **no per-op `Spec:` back-link** in code. The op / construct name is the
navigation pointer, and the reverse lookup is `grep` over `docs/spec/`. The spec
is not validated against code, and code maintains no per-op pointer back to it.

### Entropy control

To keep contracts from re-accreting in code, a single-directional lint guards
the code side (`scripts/spec_entropy_lint.py`): it flags a docstring or comment
block that is both long and carries contract / design vocabulary (`MUST`,
`MUST NOT`, `SHOULD`, `MAY`, `contract`, `invariant`, `dispatch principle`,
`single source`, ...) with "suspected contract — move to spec". It inspects code
only; it does not read or cross-check the spec. It is a backstop for the review
gate, not a validator.

The pre-commit hook runs it on the staged Python files under `src/**`, so a file
a change touches MUST stay clean; the tool takes explicit paths and does not scan
a default root. The Python tree is currently clean under a full run
(`spec_entropy_lint.py src`), and the per-file gate keeps it that way. The lint
covers Python source only; C++ runtime comments under `include/**` are not
machine-checked and rely on review.

Spec drives plan, not the other way round. When implementation reveals that a
contract is wrong, the fix lands on the spec first.
