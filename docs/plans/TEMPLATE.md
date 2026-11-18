---
type: FEAT | BUG | PERF | REFACTOR | DOCS | TEST | META
component: <component name>
target_repo: tilefoundry
---

<!-- How to discuss a plan with the author:

     Round 1 -- audit the code and write `### Current state`. Settle the open
     questions one by one into `### Decisions`, and write each milestone's
     `##### Delivered` as a sketch. Stop and show it to the author.

     Round 2 -- refine `##### Delivered` into the delivered code, and write
     `##### Accepted by`. Stop and show it to the author.

     Round 3 -- add the constraints and failure paths, then run
     scripts/finalize_plan_context.py.

     Settle with the author anything an implementer would otherwise decide alone.
     Use only the headings below. -->

# [TYPE][component] <short description>

## Description
<!-- The problem and the intended result. -->

### Current state
<!-- What the code does today and what is missing. No design. One bullet per
     claim, each with a path. -->
- `<dir>/<file>.py:<line>` <what it does today>

### Decisions
<!-- Append-only: to change one, append a record that supersedes it. Give the
     reason for each new file, class, registry entry, or test. `None.` when
     nothing was settled. -->
- D1 <question> -- <choice>, because <reason>.
- D2 <question> -- <choice>, because <reason>. Supersedes D1: <what changed>.

## Milestones

### Milestone M0: <name>

#### Depends
<!-- `None`, or earlier milestones of this plan. -->
- None

#### Target State Design

##### Delivered
<!-- The final form, with whichever fits each part:
     - a ```text tree of the files or modules, or the output a caller sees;
     - an ADT: the types and the signatures, in a ```python block;
     - a table: fields, states, rules, or a mapping;
     - a ```diff headed by `# <path>`, with `+`, `-`, and context lines, for a
       change to an existing file. At least one block MUST be a ```diff.
     Head each block with `# <path>`; these paths select the acceptance criteria
     the milestone inherits, so name every file it touches, the owning
     docs/spec/*.md included when a public contract changes. -->
```diff
# path/to/module.py
 def <the function this lands in>(...):
-    <the line that goes away>
+    <the line that replaces it>
```

##### Accepted by
<!-- First the means: existing tests that must still pass, or the new test and
     what failure it catches. Then one checkbox for each observable behaviour. -->
- [ ] <observable behaviour>
<!-- policy_ac:start -->
- [ ] Touched tests MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
<!-- policy_ac:end -->

## Final Gate

<!-- Auto-filled repository-wide gates. -->
<!-- final_gate:start -->
<!-- final_gate:end -->
