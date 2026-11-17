# TileFoundry Spec Rules

## Principles

- The spec records decisions that every implementation must preserve. These are
  public semantics, cross-module boundaries, and the internal design invariants
  that correctness, extensibility, or operability require.
- It does not repeat self-describing declarations or incidental details that may
  change without a design decision.

Runtime ops use one public entry per op/family. Target runtime headers may split
the implementation into internal helpers, traits, or impl classes. Generated
code calls the public entry. Generated code does not select implementation tiers.

Code may carry local-mechanics comments. It does not mechanically backlink every
op definition to the spec. Add a short spec backlink only when a code path exists
solely because a specific spec rule requires it.

## Unified Entry Format

A spec entry shows the construct in its **source defining form**. Every
identifier is spelled and cased exactly as in the source. A Python class
(including every HIR / TIR Op) appears as its `class Name(Base):` definition. A
Python function appears as its `def name(...) -> R:` signature. A C++ construct
appears as its struct / class / enum / template declaration. A class MUST NOT be
shown as a call-form signature. Each entry is followed by a `- constraints:` list, where
every normative MUST / SHALL / SHOULD sentence lives. A per-field table MAY
instead carry the rule that computes each field, one row per field. It is
normative. It is the owning statement for that rule. The same rule MUST NOT also
appear in the entry's `- constraints:` list. The list keeps every other
normative sentence.

The interface is concise. It is never a copy of the implementation. It has no
decorators, no registration machinery, no `ParamDef` plumbing, and no method or
function bodies. A method appears as its `...`-terminated signature. An Op's inputs
and attributes appear as annotated fields, one per line, using the source
field names; attribute defaults are kept (they are interface).

A block inside a `<!-- generated -->` region is read out of the source by a
script rather than written here, and states what the source states. It keeps
its decorators: nobody chose them for the page, and `@dataclass(frozen=True)`
or `@property` is a contract its caller is held to. Its first line names the
file it was read from, which is how it is recognised.

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
interface) starts with a `# example` / `// example` marker line. It is exempt
from the format checks. Example code never repeats an implementation. One
block may group a family of related signatures.

Consensus ops may be grouped when one external reference defines their
behavior. Custom TileFoundry ops and public runtime entries need their own
entry. A decorator-based mechanism appears only in the section that owns it. The
custom-op machinery (`@register_op` / `ParamDef`) appears in
[core-ir §2.3](./spec/core-ir.md). The visitor registries (`@register_*`) appear in
[visitor-registry](./spec/visitor-registry.md). Anywhere else a decorator may
only appear inside an `# example`-marked block.

## Constraints

Spec text is English and records the current contract only. It must not mention
plan files, milestones, task IDs, PR numbers, or commit hashes. It must not
mention chat message IDs, agent names, version stamps, test plans, or future/TODO
sections.

A construct has one owning section. Other specs link to that section instead of
restating its definition. A term has one owner: [terms](./spec/terms.md). Other specs
link there and do not restate a term's definition. Keep examples only when they pin down an otherwise
ambiguous contract corner.

## Referring To A Section

Every reference to a spec section is a markdown link:

    [<doc> §<number>](<path>#<github-anchor>)

*path* is relative inside a page (`./runtime.md`) and repository-root from
code (`docs/spec/runtime.md`). A renderer resolves a page's links beside the
page. A source file has no anchor but the root. The anchor MUST name the
section the display text numbers, never a heading nested under it. A bare
`§2.3` records which section was true once; the link breaks when the heading
moves, which is the point.

`scripts/spec_refs_lint.py` enforces this over `docs/spec/*.md`, `src/`,
`tests/` and `include/`; a `§` inside a fenced block is example text.
`tilefoundry.utils.spec_ref.spec_ref_render` renders one as
`spec runtime §1.1.2` for a refusal message.

**Append a section; do not renumber one.** A number is an address that code,
sibling specs and the `spec` command all reach a section by. Duplicates are
worse than gaps: two `3.1` headings make both unreachable by
`tilefoundry spec <topic> 3.1`. Gaps and out-of-order numbering are the
accepted cost — `docs/spec/runtime.md` numbers `2.10` before `2.9`. This part is
habit: the lint catches a renamed heading, not a renumbered one.

**A catalog of things that come and go is not numbered.** An op list and a primitive list are catalogs. Give the group a numbered
section. Give the members unnumbered headings under it, the way [hir §1.3](./spec/hir.md#13-op) does with
`#### ir/hir/<namespace>/` and `##### <OpName>`. Then adding one renumbers
nothing, and the relaxation this rule was once given — permitting a renumber
behind the refs lint — is not needed.

## Entropy And Close Tracking

`scripts/spec_entropy_lint.py` guards Python code comments/docstrings from
re-growing long contract prose. `scripts/spec_rules_lint.py` checks
mechanically-forbidden spec tokens. These tools are review gates, not a
spec-code validator.

Review-led spec work tracks every source comment to one final state:
`implemented`, `verified no-op`, or `resolved by decision`.

## Language

Spec text is written in ASD-STE100 Simplified Technical English.

- One sentence states one fact or gives one instruction.
- An instruction has 20 words or fewer. A description has 25 words or fewer.
- Use the active voice and the present tense.
- One word has one meaning, and one meaning has one word. The words for meshes,
  topologies and layouts are the terms of [terms](./spec/terms.md).
- Use a term with its defined meaning and part of speech. The defined verbs are: lowering issues; a unit executes.
- Do not replace a defined term with a synonym.
- Write an instruction as a command or with MUST. Write a description as a statement.
- A noun cluster has three words or fewer.
- Do not write history, rejected alternatives, or incidental descriptions of absent features.
- Keep prohibitions, refusal conditions, and invariants that define the current contract.

This rule applies to text that is added or changed. Existing text is rewritten
when its section is edited.
