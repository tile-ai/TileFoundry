---
type: FEAT | BUG | PERF | REFACTOR | DOCS | TEST | META
component: <component name>
target_repo: tilefoundry
---

# [TYPE][component] <short description>

## Description

### Symptom / Motivation
<!-- What is observed or what motivates this change. Specific, not abstract. -->

### Root Cause Analysis
<!-- Why it happens — file paths, logic gaps, missing features. "N/A" for new features. -->

### Related Files
<!-- Plan-wide touch surface. Every file the plan expects to add/modify, repo-relative.
     `scripts/finalize_plan_context.py` reads this list to match path-scoped policies for
     the plan-level Execution Preflight block. -->
- <path>
- <path>

## Goal

<!-- One sentence. Measurable verb; no "improve" / "make better". -->

## Constraints

- <!-- Constraint discovered during exploration -->
- <!-- Boundary that distinguishes this plan from adjacent work -->

## Milestones

### Milestone M0: <name>

#### Depends
- None

#### Related Files
<!-- Per-milestone touch surface. Drives the policy-AC injection into this milestone's
     `#### Acceptance Criteria → policy_ac` range. Use `- inherit: top-level` as an
     explicit fallback to the plan-level `### Related Files`; implicit inheritance is
     not allowed. -->
- <path>

#### Spec Impact
<!-- List one or more owning `docs/spec/*.md` paths and repeat them in this
     milestone's effective Related Files. If no public contract changes, use exactly
     one reasoned entry: `- N/A: <reason>`. Do not mix paths and N/A. -->
- N/A: <reason this milestone does not change a public contract>

#### Golden Reference
<!-- State the source of truth before describing code: an external implementation,
     measured current behaviour, existing end-to-end workflow, or a precise public
     contract. List only the functional points this milestone must preserve or
     reach. A refactor's Golden Reference is the behaviour it preserves. -->
- Source: <reference implementation, document, contract, or existing workflow>
- Functional points: <observable behaviour determined by that source>

#### Plan
- [ ] step 0.1 <action with affected files>
- [ ] step 0.2 <action>

#### Acceptance Criteria
<!-- State observable completed behaviour. An AC is not a request for its own test:
     one complete workflow may evidence several ACs. Do not specify source scans,
     private calls, object identities, line counts, or other implementation shape
     unless that form is itself a public contract. -->
- [ ] AC-0-1: <author-written, milestone-specific observable behaviour>
- [ ] AC-0-2: <author-written, milestone-specific observable behaviour>
<!-- policy_ac:start -->
- [ ] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [ ] Verification MUST exercise the Golden Reference's functional points through the smallest real workflow; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

#### Verification
<!-- Prove the Golden Reference's functional points through the smallest real
     workflow. Extend an existing workflow before adding a test. A new test needs a
     short reason that no existing workflow reaches that point; do not add one merely
     to make an AC individually test-shaped. -->
- Golden point(s) exercised: <the point(s) above>
- Evidence: <command, test, or end-to-end path>
- New coverage: N/A — <why the existing workflow is sufficient>

### Milestone M1: <name>

#### Depends
- M0

#### Related Files
- <path>
- `docs/spec/<name>.md`

#### Spec Impact
- `docs/spec/<name>.md`

#### Golden Reference
- Source: <reference implementation, document, contract, or existing workflow>
- Functional points: <observable behaviour determined by that source>

#### Plan
- [ ] step 1.1 <action>

#### Acceptance Criteria
- [ ] AC-1-1: <author-written observable behaviour>
<!-- policy_ac:start -->
- [ ] Milestone MUST name a `#### Golden Reference` before implementation steps, with the source of truth and the observable functional points it determines. <!-- policy_ac: milestone_review-0 -->
- [ ] Verification MUST exercise the Golden Reference's functional points through the smallest real workflow; one workflow MAY cover several ACs, and a new test requires a stated reachability gap. <!-- policy_ac: milestone_review-1 -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-2 -->
<!-- policy_ac:end -->

#### Verification
- Golden point(s) exercised: <the point(s) above>
- Evidence: <command, test, or end-to-end path>
- New coverage: N/A — <why the existing workflow is sufficient>

## Execution Preflight

<!-- This block is auto-filled by `scripts/finalize_plan_context.py`.
     It surfaces the policy entries from `docs/policies/project-policy.json`
     whose `when.path_glob` matches the plan-level `### Related Files`
     above, so the implementer and reviewer can see the cross-cutting
     rules / knowledge for this plan in one place. It also validates each
     milestone's `Spec Impact`, `Golden Reference`, and `Verification`
     sections. Leave the marker pair below and run the finalizer; do not
     hand-edit the body. -->
<!-- policy_preflight:start -->

### Policy Rules & Knowledge

- Scope discipline — One commit touches only what the current task requires; unrelated edits / submodule bumps / autoformat go in separate commits or are called out explicitly. (see `docs/develop.md § Scope`) <!-- policy_rules: scope_discipline -->
- Milestone review — Each milestone names its Golden Reference, verifies its functional points through real workflows, and removes redundant test/comment scaffolding. (see `docs/develop.md § Tests`, `docs/develop.md § Code comments`) <!-- policy_rules: milestone_review -->
<!-- policy_preflight:end -->

## Final Gate

<!-- This block is auto-filled by `scripts/finalize_plan_context.py` once per
     plan. It holds repository-wide checks such as spec discipline and
     clang-format; do not repeat those gates in every milestone. -->
<!-- final_gate:start -->
- [ ] No touched C++/CUDA files in this plan — clang-format gate N/A <!-- policy_final: clang_format-na -->
<!-- final_gate:end -->
