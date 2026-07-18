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

#### Plan
- [ ] step 0.1 <action with affected files>
- [ ] step 0.2 <action>

#### Acceptance Criteria
- [ ] AC-0-1: <author-written, milestone-specific, verifiable>
- [ ] AC-0-2: <author-written, milestone-specific, verifiable>
<!-- policy_ac:start -->
<!-- policy_ac:end -->

### Milestone M1: <name>

#### Depends
- M0

#### Related Files
- <path>

#### Spec Impact
- `docs/spec/<name>.md`

#### Plan
- [ ] step 1.1 <action>

#### Acceptance Criteria
- [ ] AC-1-1: <author-written>
<!-- policy_ac:start -->
<!-- policy_ac:end -->

## Execution Preflight

<!-- This block is auto-filled by `scripts/finalize_plan_context.py`.
     It surfaces the policy entries from `docs/policies/project-policy.json`
     whose `when.path_glob` matches the plan-level `### Related Files`
     above, so the implementer and reviewer can see the cross-cutting
     rules / knowledge for this plan in one place. Leave the marker
     pair below and run the finalizer; do not hand-edit the body. -->
<!-- policy_preflight:start -->
<!-- policy_preflight:end -->
