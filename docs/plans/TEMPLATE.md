---
type: FEAT | BUG | PERF | REFACTOR | DOCS | TEST | META
component: <component name>
target_repo: tilefoundry
---

<!-- Pass 1 settles the intended result, each milestone's Target State Design,
     constraints, and acceptance criteria; Plan steps stay coarse. Pass 2 audits
     the repository, resolves references and quantifiers, fixes Related Files,
     makes the steps executable, and finalizes. Do not state a count or reference
     that has not been checked. -->

# [TYPE][component] <short description>

## Description

<!-- Explain the problem, intended result, and constraints in the shape this plan
     needs. -->

## Milestones

### Milestone M0: <name>

#### Depends
- None

#### Target State Design
<!-- Show every part this milestone designs in its delivered form. Use code or
     compact pseudocode. -->
```python
# <delivered code shape>
```

#### Related Files
<!-- Files this milestone touches. List owning docs/spec/*.md files when it
     changes a public contract. -->
- <path>

#### Plan
<!-- Pass 1 leaves coarse steps; pass 2 resolves real files, call sites, and order. -->
- [ ] step 0.1 <action with affected files>
- [ ] step 0.2 <action>

#### Acceptance Criteria
<!-- State observable completed behaviour, not one test per AC. Keep implementation
     shape in Target State Design; evidence belongs in the gate request. -->
- [ ] AC-0-1: <author-written, milestone-specific observable behaviour>
- [ ] AC-0-2: <author-written, milestone-specific observable behaviour>
<!-- policy_ac:start -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
<!-- policy_ac:end -->

### Milestone M1: <name>

#### Depends
- M0

#### Target State Design
<!-- Show every part this milestone designs in its delivered form. Use code or
     compact pseudocode. -->
```python
# <delivered code shape>
```

#### Related Files
- <path>
- `docs/spec/<name>.md`

#### Plan
- [ ] step 1.1 <action>

#### Acceptance Criteria
- [ ] AC-1-1: <author-written observable behaviour>
<!-- policy_ac:start -->
- [ ] Touched tests and comments MUST be reviewed for redundancy: remove ones superseded by the retained workflow, and do not add source-shape or hypothetical-refactor guards unless that form is a public contract. <!-- policy_ac: milestone_review-0 -->
<!-- policy_ac:end -->

## Final Gate

<!-- Auto-filled repository-wide gates. -->
<!-- final_gate:start -->
- [ ] Spec section MUST NOT enumerate test names; the pre-commit `spec-rules-lint` and `english-only` hooks already reject forbidden section headers, plan / milestone / task / PR / commit references, agent names, and non-English text. <!-- policy_final: spec_discipline-0 -->
- [ ] No touched C++/CUDA files in this plan — clang-format gate N/A <!-- policy_final: clang_format-na -->
<!-- final_gate:end -->
