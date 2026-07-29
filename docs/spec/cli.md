# TileFoundry Command-Line Interface

This file defines the command-line contract for the two pieces of work an Agent
does through TileFoundry: translating a published model into authored HIR, and
turning that HIR into a high-performance runtime implementation. The commands
answer what only this project knows — which models have been described and how,
what a specification section says, whether an implementation agrees with its
reference, and what an authored program costs. Python authoring syntax and
grammar productions remain in the [parser specification](./parser.md); the
authored IR itself is the [HIR specification](./hir.md).

Naming no command MUST print an overview rather than an error: a one-line
project summary taken from installed package metadata, the usage form, the
commands in the order the work is done, and the options. The summary MUST NOT
be restated in the command surface, so there is one copy of it.

## Commands

```text
tilefoundry spec [TOPIC [SECTION]]

tilefoundry analyze model.py[:Module[.child_module...][.function]]
    [--roofline] [--footprint] [--timeline] [--dim NAME=EXTENT ...]

tilefoundry schedule model.py[:Module[.child_module...][.function]] --topology LEVEL [--json]
    [--dim NAME=EXTENT ...] [--solver-timeout SECONDS] [--solver-workers COUNT]
    [--first-plan]

tilefoundry inspect capabilities model.py[:Module[.child_module...][.function]]
```

`SOURCE` is a Python file followed optionally by
`:Module[.child_module...][.function]`. Without a selector, the source must
define exactly one top-level HIR Module or Function. A selector chooses the
named Module, or a Function reached through the chain of child Modules that own
it: each segment names a child Module of the one before it, and only the last
MAY name a Function. A leaf selected this way MUST keep resolving the Target and
hierarchy it inherits from the owners it was reached through.

A selector MAY name a top-level binding that is a bare `Function` — a source
whose `@func` declares no execution context binds one. Every verb here reads
hardware facts, so such a selection MUST be rejected naming the Module that
would declare its context, rather than analysed or scheduled against a default
([target §6](./target.md#6-target-ownership-and-compile-resolution)).

## Spec

`spec` discloses the installed specifications a step at a time: which documents
there are, what is in one, and one section of one. It MUST NOT print a document
whole; a reader who asked what a rule says is not asking for every rule.

- constraints:
  - With no `TOPIC`, output MUST list the documents that can be asked for,
    including any name that is another name for one of them.
  - With a `TOPIC` and no `SECTION`, output MUST be that document's outline —
    every section's key and title, indented by heading depth — and MUST NOT
    include the sections' bodies.
  - A section's key MUST be its own number when its heading carries one, and
    otherwise a name derived from its title. Numbers alone would leave most of a
    document unaddressable: a catalogue of operations numbers its container and
    not its entries, and a section that cannot be named cannot be read.
  - Keys MUST be unique within a document. Neither a number nor a title is unique
    on its own — a document may restart its numbering under a later heading, and
    may describe a field of the same name in two places — and a key naming two
    sections would make the refusal below unreachable and answer with whichever
    came first. A key that would repeat MUST take on the name of its enclosing
    heading, and the one above that, until the keys differ.
  - Headings MUST be recognised outside fenced code blocks only. A `#` line
    inside a fence is a comment in an example, and treating it as a section would
    both invent entries and cut the surrounding section short.
  - With a `SECTION`, output MUST be that section — its heading and the lines
    down to the next heading at its level or above — followed by the keys of the
    sections beside it: the previous and the next at its level, and the ones it
    contains. Naming the neighbours is what lets a reader walk the document
    without returning to the outline.
  - A `SECTION` the document does not have MUST be refused naming the keys it
    does have, so the next attempt can succeed.
  - A `TOPIC` with no installed document MUST be refused as that.
  - Documents MUST be read from the source tree when one is present, and from the
    installed data directory otherwise, so the command answers the same in an
    editable checkout as in an installed environment. The source tree comes first
    because an editable checkout is the one place the two can disagree, and there
    the working copy is what its author means.

## Analyze

`analyze` first runs deterministic type inference and then prints complete type
comments, regardless of analysis flags. It never performs candidate search,
layout enumeration, or automatic resharding.

Each flag names one root analysis. With no flag, `analyze` runs all of them. The
selected Module's resolved Target determines the hardware specification; there is
no ordinary `--target` option.

- constraints:
  - `analyze` MUST invoke the public operation once per requested root, because
    that operation takes one root per call
    ([analysis §3](./analysis.md#3-composed-analysis)). Requesting two analyses
    MUST NOT change what either reports.
  - A selection MUST resolve to a Module. A bare Function MUST be rejected
    naming the reason: it declares neither the Target its numbers are measured
    against nor the topology hierarchy they divide over.
  - `--json` MUST print the report as JSON instead of text. Both formats MUST
    carry the same conclusions ([analysis §2](./analysis.md#2-authored-hir-metrics)).
  - `--dim NAME=EXTENT` MUST bind one dimension the selection leaves open, and
    MUST be repeatable to bind several. It MUST be passed through as the
    operation's `dims` ([analysis §2.2](./analysis.md#2-authored-hir-metrics));
    the CLI MUST NOT specialise the selection itself, because then what it
    printed would be about a program the operation never saw.
  - A `--dim` argument that is not `NAME=EXTENT`, or whose extent is not an
    integer, MUST be rejected naming which argument and why.
  - Repeating `--dim` states another dimension. One dimension stated twice MUST
    be rejected naming that dimension, whether or not the two extents agree; the
    later occurrence MUST NOT win, because both came from the caller and choosing
    between them silently answers a request that has no answer.
  - With no `--dim`, the selection MUST be analysed as authored. A selection that
    leaves a dimension open MUST then fail naming the dimension: counting
    elements requires an extent, and a range is not one.
  - Every requested analysis MUST be reported together even when each was run at
    the stated extents, which builds one program per analysis. The report MUST
    accept those as one program when they were rebuilt from the same function at
    the same extents, and MUST refuse results rebuilt at different extents. The
    comparison MUST read the recorded extents and MUST NOT infer them from the
    resulting signature: a dimension occurring only in a loop bound or a body
    operation's attribute leaves the signature identical at every extent.
  - Output MUST report the analyses that were requested. A dependency that ran
    because a requested root needed it MUST appear in the executed list and MUST
    NOT have its own measurements reported.
  - On success, text output begins with the `#`-headed report followed by
    annotated HIR. On inference, verification, or analysis failure, stdout MUST
    be empty and stderr MUST report the source location, binding where
    available, and reason.

## Schedule

`schedule` makes one public Schedule call
([schedule §1](./schedule.md#1-the-public-schedule-operation)) and prints the
Plan that call produced. It composes nothing itself: which algorithm runs, what
it decides, and how the decision reads are owned by the algorithm registered for
the selected Module's target at the requested level.

The target is not a flag. It comes from the selected Module's resolved Target
([core-ir §1](./core-ir.md#1-module)): a kernel is authored against one target.
A selection that is a bare Function, or a Module whose owner chain declares no
Target, MUST be rejected -- `schedule` does not resolve an omission to a default
([target §6](./target.md#6-target-ownership-and-compile-resolution)).

- constraints:
  - `--topology` MUST be required and MUST name one level the selected Module
    declares. A level the Module does not declare, or one the target has no
    algorithm for, MUST be reported as that.
  - The command MUST call the public operation once and MUST NOT compose the
    algorithm's stages itself, so what it prints cannot drift from what the
    operation decided.
  - `--dim NAME=EXTENT` MUST behave as it does for `analyze`, passed through as
    the operation's `dims` ([schedule §2.2](./schedule.md#2-public-operation)).
  - `--solver-timeout SECONDS` and `--solver-workers COUNT` MUST state the search
    budget the operation is given, and either omitted MUST leave that part of the
    budget at the operation's own default. A solver that sizes itself to the
    machine is the right default for one schedule and the wrong one for several at
    once, so the caller running several MUST be able to say so; a budget that
    cannot be stated is a configuration nobody can reproduce.
  - `--first-plan` MUST ask for the first plan that satisfies the constraints
    rather than the best one within the budget, and MUST NOT lift the time limit:
    a search that has found nothing yet stays bounded by it. Omitted, the search
    MUST run as the operation's default does. The distinction is the caller's
    because a search that cannot prove its objective optimal spends its whole
    budget improving, so a caller who needs a plan and not the best plan otherwise
    pays the full budget for an answer it already had.
  - A search that ends without an answer MUST be reported as the search ending
    without one, and MUST NOT be reported as the selection having no schedule.
  - Output MUST be the Plan's own rendering: its `render()` by default, its
    `to_json()` under `--json`. The command MUST NOT impose a shape across
    algorithms, because two algorithms deciding different things have nothing to
    share a format for.
  - On any failure stdout MUST be empty and stderr MUST carry one
    `tilefoundry: error:` line naming the cause.

Scheduling this CUDA kernel at the level its Module divides over:

```python
# example
@func(target="cuda", topologies=(Topology("cta", 4),))
def blocked_matmul(
    x: Tensor[(64, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
) -> Tensor[(64, 64), "bf16"]:
    return matmul(x, w)
```

```text
# example
$ tilefoundry schedule model.py:blocked_matmul --topology cta
partition cta on nvidia.h200_sxm (OPTIMAL, makespan 35ns)
  MatMul x4 [0, 35)
```

The same Module scheduled at a level it also declares answers with that level's
own algorithm and that algorithm's own Plan, which reads differently because it
decided different things.

## Inspect Capabilities

`inspect capabilities` resolves the target from the selected Module and prints
the installed compact hardware capability record. It does not emit compiler
operation coverage. The record names both the architecture and the device
document behind the target, each with its content digest, then every recorded
fact by its path. A fact identifies its unit, the conditions it holds under,
its source, and its origin — vendor-published, measured on the described host,
cited from a reference, derived, or a reading no source states. A fact with no
usable value is reported as unavailable rather than given a placeholder number.
A target composed from a directly supplied value has no installed document to
report, and the command says so instead of naming the resource it resembles.
