---
type: FEAT | BUG | PERF | REFACTOR | DOCS | TEST | META
component: <component name>
target_repo: tilefoundry
---

<!-- Pass 1 settles the intended result and each milestone's Target State Design;
     Plan steps stay coarse. Pass 2 audits the repository, resolves references and
     quantifiers, fixes Related Files, makes the steps executable, and finalizes.
     Do not state a count or reference that has not been checked.

     What a milestone leaves behind is settled here, with the author, before it is
     dispatched -- the delivered code and the way it is accepted alike. Anything an
     agent would otherwise decide alone at implementation time is a question for
     pass 1.

     Only the headings below may appear. finalize_plan_context.py rejects any
     other one, so a thought with no home here belongs in the code, the spec, or a
     message -- not in the plan. -->

# [TYPE][component] <short description>

## Description
<!-- Explain the problem, intended result, and constraints in the shape this plan
     needs. -->

## Milestones

### Milestone M0: <name>

#### Depends
<!-- `None`, or the milestones in this plan that must land first. -->
- None

#### Target State Design

##### Delivered
<!-- The surface this milestone designs, the behaviour it executes, and its
     output, in delivered form. Use code or compact pseudocode. -->
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
<!-- Files this milestone touches. List owning docs/spec/*.md files when it
     changes a public contract. -->
- <path>

#### Plan
<!-- Pass 1 leaves coarse steps; pass 2 resolves real files, call sites, and order. -->
- [ ] step 0.1 <action with affected files>
- [ ] step 0.2 <action>

## Final Gate

<!-- Auto-filled repository-wide gates. -->
<!-- final_gate:start -->
<!-- final_gate:end -->
