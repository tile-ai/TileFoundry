---
type: FEAT | BUG | PERF | REFACTOR | DOCS | TEST | META
component: <component name>
target_repo: tilefoundry
---

<!-- Read this before writing any of the plan below.

     Pass 1 -- audit the repository and probe what is not yet known. Write
     `### Current state`, and nothing else yet.

     Pass 2 -- write each milestone's `##### Delivered` shape. Then stop and show
     the author those shapes. No prose exists yet, so a shape that is not what the
     author meant costs ten lines to redirect rather than a document to rewrite.

     Pass 3 -- write the rest, then run scripts/finalize_plan_context.py.

     Append to `### Decisions` as each question is settled. Never edit an accepted
     record; append one that supersedes it.

     Settle with the author anything an agent would otherwise decide alone at
     implementation time -- the delivered code and the way it is accepted alike.

     Use only the headings below. -->

# [TYPE][component] <short description>

## Description
<!-- The problem and the intended result. Only what the reader needs before the
     current state. -->

### Current state
<!-- State the mechanism the code implements today: what is there, and what is
     missing. No design and no proposal -- the gap is the finding, not the fix.

     One bullet per claim, each citing a path in backticks. A claim with nothing to
     point at is the one that turns out to be wrong, and the design resting on it
     moves once it does. -->
- `<dir>/<file>.py:<line>` <what it does today>
- `<dir>/<file>.py` <a number a probe produced, and what it means>

### Decisions
<!-- Append-only. One settled question per record, never edited once accepted:
     when the answer changes, append a record that supersedes the earlier one.
     `None.` when the plan settled nothing an implementer would otherwise choose.

     A record is where "why a new one rather than the existing one" gets answered.
     A new file, class, registry entry, analysis family, or test belongs here with
     its reason, because that is the choice a reader most often disagrees with. -->
- D1 <question> -- <choice>, because <reason>.
- D2 <question> -- <choice>, because <reason>. Supersedes D1: <what changed>.

## Milestones

### Milestone M0: <name>

#### Depends
<!-- `None`, or the milestones in this plan that must land first. -->
- None

#### Target State Design

##### Delivered
<!-- The surface this milestone designs, the behaviour it executes, and its
     output, in delivered form. Use code or compact pseudocode, in a fenced
     block tagged with a programming language. -->
```python
# <delivered code shape>
```

##### Accepted by
<!-- The one place acceptance is stated: first the means, then the behaviour.

     Means -- two answers are legal, and the first is not an exception:
       - nothing new; name the existing suites that must still pass unchanged
       - the test that earns its place; say which bar it clears and what breaks it
     Settled with the author while the plan is written.

     Behaviour -- one checkbox each, in terms a reviewer can check off. State what
     must be observable, not which suite shows it; the means above already said
     that. -->
- [ ] <observable completed behaviour>
- [ ] <observable completed behaviour>
<!-- policy_ac:start -->
- [ ] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
<!-- policy_ac:end -->

#### Related Files
<!-- Files this milestone touches. `filter_policies` reads them to choose which
     acceptance criteria this milestone inherits, so a missing path silently drops
     one. List owning docs/spec/*.md files when it changes a public contract. -->
- <path>

## Final Gate

<!-- Auto-filled repository-wide gates. -->
<!-- final_gate:start -->
<!-- final_gate:end -->
