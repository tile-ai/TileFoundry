# TileFoundry Spec — Rules

How `docs/spec/*.md` is written. Principle first; constraints below.
No fixed template — pick the form that fits each construct.

## Principle

**Simplicity above all.** A spec section reads like a contract, not an
implementation log. If one sentence is enough, do not write a section.

A spec MUST be written in English. Existing zh-CN passages SHOULD be
rewritten on first touch.

## Op catalog

An **Op** is spec'd as a catalog entry, not a field walk-through:

1. a **namespace-leveled heading** that mirrors the op's module
   namespace (e.g. `## tensor` → `### insert_slice`)
2. a **single sentence** naming what the op does; a consensus Op
   (`add`, `relu`, ...) adds a stable external link instead of prose

An Op that gets its **own** catalog heading (a custom Op) records its full
contract — fields, typing / verifier rules, worked examples — in the **Op-class
docstring** in code, which MUST carry the `Spec: <file> §X.Y` back-link (see
below). Consensus Ops listed together under a **grouped** entry (one heading +
an external link covering the whole group, e.g.
`#### Reshape / Transpose / … — torch structural ops`) are exempt from the
per-op back-link. Per-op contract or implementation detail MUST NOT live in the
spec; the spec op catalog is the namespaced index. Design / architecture guidance (dispatch
principles, cross-layer ownership) stays in the spec prose, outside the
per-op entries.

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

Ops follow the Op-catalog rule above: one namespaced heading + one
sentence in the spec, with the contract in the Op-class docstring.
Field-by-field tables and `ParamDef` listings stay in code.

For finite enumerations (dtype, storage class, ...) state the rule
and a representative subset; exhaustive enumeration is not required.

Use **MUST** / **MUST NOT** / **SHOULD** / **MAY** (RFC 2119) only
inside contract sentences.

## Code-side back references

A code module that implements a spec construct SHOULD carry a single
docstring line:

```
Spec: <file> §X.Y
```

An Op class that has its **own** catalog heading MUST carry this
`Spec:` line (it is the target of that entry and holds the op's full
contract); Ops under a grouped consensus entry are exempt (above). The
reverse index is `grep`. There is no central registry. Internal helpers
and private utilities MAY omit the anchor.

Spec drives plan, not the other way round. When implementation
reveals that a contract is wrong, the fix lands on the spec first.
