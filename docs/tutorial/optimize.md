# Step two: make it fast

The authored HIR stays the reference. Everything here is written beside it.

1. **Write a runtime twin** of one Module. `@runtime_module` checks at decoration
   time that the twin's function names and child names are exactly the authored
   Module's, so a twin covers a whole Module at once —
   [runtime §1.1](../spec/runtime.md#11-runtimemodule).

2. **Check it.** `check` runs the implementation and its reference and says, output
   by output, whether it meets the bounds you stated. The reference is the authored
   Module through the evaluator, so the comparison is available from the twin's
   first line. `tilefoundry check --help` has the predicates and the arithmetic for
   choosing a tolerance.

3. **Ask for evidence when you want it.** `analyze` reports what the authored
   program costs: flops, traffic, roofline bounds, a predicted time. `schedule` proposes
   a plan for one topology level you name: placement, resharding, timing. Both read
   the authored source, both are optional, and neither decides anything.
   `tilefoundry analyze --help` and `tilefoundry schedule --help` say what each
   reports.

4. **Change the authored HIR when that is the answer.** Fusion is done by writing
   the fused form: two equations in one `@func`, two `@func`s merged into one, or
   work that crossed several Modules pulled into one. For a CUDA target one `@func`
   is one kernel symbol ([codegen §4.1](../spec/codegen.md#41-linkablefunction)), so
   a bigger `@func` is a bigger kernel. The fused Module's twin is then an ordinary
   twin, written the same way.

5. **Repeat wherever there is an opportunity.** There is no required order and no
   point at which some change becomes allowed.

## Working on a copy

The shipped model directory is the reference and lives in the installation. Copy it
before changing it:

```sh
source=$(tilefoundry models qwen3_5_35b_a3b --source | sed -n '1p')
cp -r "$source" mine
```

Every command takes `file:Class[.child...][.function]`, and the file is any file you
can read:

```sh
tilefoundry check mine/model.py:MyFused.fused --inputs random --out output \
    --fn allclose --atol 1e-6 --rtol 1e-6
tilefoundry analyze mine/model.py:MyFused --roofline
tilefoundry schedule mine/model.py:MyFused --topology cta --first-plan
```

`analyze` and `schedule` need a Module that reaches a declared target, so name it
from the root down.

## What check cannot do

A fused boundary replaces a composition of unfused ones. `check` compares one
implementation against one reference; it cannot take several composed unfused
boundaries on one side and one fused boundary on the other. That equivalence stays
yours to keep, at a boundary the commands can express.
