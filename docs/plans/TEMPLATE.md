---
type: FEAT | BUG | PERF | REFACTOR | DOCS | TEST | META
component: <component name>
target_repo: tilefoundry
---

<!-- A plan is written in two passes by two different people.

     Pass 1 — discussion (human + planner). Settles what "right" looks like:
     Description, Goal, Constraints, and each milestone's Golden Reference and
     Acceptance Criteria. `#### Plan` gets coarse steps only. The discipline
     here is one rule: DO NOT WRITE A QUANTIFIER, `file:line`, OR `§x.y` YOU
     HAVE NOT LOOKED UP. "the `reference.py` files that call `linear_weight`"
     is a fine coarse step; "the four dense models' `reference.py`" is not,
     unless the grep was run. Coarse means few steps, not vague ones.

     Pass 2 — dispatch (the reviewer, before the implementer writes anything).
     Explores the repository and corrects the plan: resolves every reference,
     settles every quantifier by running the grep, fixes `#### Related Files`,
     and turns the coarse steps into executable ones. The person writing the
     steps is the person who just read the code, so the premises are built
     rather than recalled. Then finalize and tell the implementer to start. -->

# [TYPE][component] <short description>

## Description

<!-- Free prose, in whatever shape this plan needs: what is observed, why it
     happens, what it would take. Nothing here is required and nothing reads it,
     so say the thing rather than filling a form. -->

## Milestones

### Milestone M0: <name>

#### Depends
- None

#### Golden Reference
<!-- State the source of truth before describing code: an external implementation,
     measured current behaviour, existing end-to-end workflow, or a precise public
     contract. List only the functional points this milestone must preserve or
     reach; naming the mechanism they land on is welcome — that is what makes the
     coarse plan below writable. A refactor's Golden Reference is the behaviour it
     preserves. Every reference in `Source:` must resolve; pass 2 completes it. -->
- Source: <reference implementation, document, contract, or existing workflow>
- Functional points: <observable behaviour determined by that source>

#### Related Files
<!-- Per-milestone touch surface. Drives the policy-AC injection into this milestone's
     `#### Acceptance Criteria → policy_ac` range. Use `- inherit: top-level` as an
     explicit fallback to the plan-level `### Related Files`; implicit inheritance is
     not allowed. List `docs/spec/*.md` here when this milestone changes a spec. -->
- <path>

#### Plan
<!-- Pass 1 leaves coarse steps; pass 2 makes them executable — real files, real call
     sites, real order. -->
- [ ] step 0.1 <action with affected files>
- [ ] step 0.2 <action>

#### Acceptance Criteria
<!-- State observable completed behaviour. An AC is not a request for its own test:
     one complete workflow may evidence several ACs. Do not specify source scans,
     private calls, object identities, line counts, or other implementation shape
     unless that form is itself a public contract. The evidence that an AC is met
     belongs in the gate request, not here — before implementation there is no
     receipt to write down. -->
- [ ] AC-0-1: <author-written, milestone-specific observable behaviour>
- [ ] AC-0-2: <author-written, milestone-specific observable behaviour>
<!-- policy_ac:start -->
- [ ] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [ ] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

### Milestone M1: <name>

#### Depends
- M0

#### Golden Reference
- Source: <reference implementation, document, contract, or existing workflow>
- Functional points: <observable behaviour determined by that source>

#### Related Files
- <path>
- `docs/spec/<name>.md`

#### Plan
- [ ] step 1.1 <action>

#### Acceptance Criteria
- [ ] AC-1-1: <author-written observable behaviour>
<!-- policy_ac:start -->
- [ ] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [ ] The gate request MUST show the Golden Reference's functional points exercised through the smallest real workflow, naming the retained evidence; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

## Final Gate

<!-- This block is auto-filled by `scripts/finalize_plan_context.py` once per
     plan. It holds repository-wide checks such as spec discipline and
     clang-format; do not repeat those gates in every milestone. -->
<!-- final_gate:start -->
- [ ] Spec section MUST NOT enumerate test names; the pre-commit `spec-rules-lint` and `english-only` hooks already reject forbidden section headers, plan / milestone / task / PR / commit references, agent names, and non-English text. <!-- policy_final: spec_discipline-0 -->
- [ ] No touched C++/CUDA files in this plan — clang-format gate N/A <!-- policy_final: clang_format-na -->
<!-- final_gate:end -->
