# TileFoundry Command-Line Interface

This file defines the command-line contract for the two pieces of work an Agent
does through TileFoundry: translating a published model into authored HIR, and
turning that HIR into a high-performance runtime implementation. The commands
answer what only this project knows — which models have been described and how,
what a specification section says, whether an implementation agrees with its
reference, and what an authored program costs. Python authoring syntax and
grammar productions remain in the [parser specification](./parser.md); the
authored IR itself is the [HIR specification](./hir.md).

Naming no command at any command level MUST print that level's overview rather
than an error. The top-level overview MUST include a one-line project summary
taken from installed package metadata, the usage form, the commands in the order
the work is done, and the options. A command overview MUST include its
description, usage form, subcommands, and options, and its description MUST be
the one shown by the command above it. The project summary MUST NOT be restated
in the command surface, so there is one copy of it.

A command usage error MUST print the error followed by that command's complete
help to standard error and exit with status 2.

## Commands

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

The file in `SOURCE` is any readable Python file. Nothing privileges the model
sources this project ships: a reader who copies a shipped model directory, merges
two of a Module's functions into one and points a verb at the copy MUST reach the
same command surface, because coarsening a boundary is done by editing source and
there is no other mechanism for it. The installed directory named by `models
<name> --source` stays the reference to compare against, and it stays intact
because an installation is not where anybody edits — no verb enforces that, and
none should.

A command MUST load `SOURCE` as a Python module. While loading it, the directory
containing `SOURCE` MUST be first on the Python module search path, so a file beside
it MAY be imported by its module name. A command MUST capture and suppress standard
output that `SOURCE` emits while loading.

`check` reads the same `SOURCE` shape and one thing more: its selector MAY name a
runtime twin instead of an authored Module. A twin generated from an authored
Module states which Module that is ([runtime §1.1](./runtime.md#11-runtimemodule)),
so naming the implementation is enough to reach what it is judged against. A
runtime module that states none MUST be refused rather than compared against
something chosen for it.

## Check

`check` is the one command that reports agreement. It runs an implementation and,
when there is one, its reference, and says of every output whether it meets the
bounds the caller stated.

- constraints:
  - Repeated `--input` values MUST bind the function's inputs in parameter
    declaration order. Output names MUST come from return position: one tensor is
    `output`; a tuple's tensors are `output[0]`, `output[1]`, and so on in return
    order. These are positions, not names authored in the function.
  - One `--input` file MUST bind one parameter. Its value MAY be a bare tensor or
    an arbitrarily nested tuple or list of tensors; every leaf MUST be a tensor.
  - A target whose step is an orchestration method rather than a `@func` MUST
    refuse `--inputs random` and `--inputs real`, because its activation shapes
    and dtypes are not declared. The refusal MUST name the parameter count and
    names in order, and say that one `--input` file binds each parameter.
  - Every output MUST be judged by at least one predicate the caller states, and
    there MUST be no default predicate and no default bound. A bound nobody can
    meet is worse than none: a single `f32`→`bf16` rounding already measures
    `rel_l2` 1.66e-3, so a default of 1e-3 would teach its reader that FAIL is
    the normal state of a correct program.
  - Naming an output that was not produced, or leaving a produced output
    unjudged, MUST be refused. A comparison that silently skipped an output
    reports the same PASS as one that checked it.
  - An empty result MUST be an error, never a PASS: measuring nothing is not
    agreement.
  - A predicate MUST be refused on an output whose dtype it says nothing about.
    On a discrete output one wrong value is a total failure and a negligible
    numerical deviation, so an aggregate over indices MUST be refused pointing at
    exact comparison.
  - The reference MAY be stated as files, or MAY be the evaluator running the
    authored Module the implementation stands for. With no reference at all, only
    a predicate that judges the candidate alone is admissible; every two-sided
    predicate MUST be refused, because there is nothing to compare against.
  - Each output MUST report the norm of its reference. Near zero, a relative
    measure divides by nothing, so the report MUST state what it measured instead
    rather than a number with no scale to read it against.
  - Inputs MUST be stated: random, real weights from a checkpoint, or files, and
    no form MAY be the default. Weights MUST come from the same draw on both
    sides, and the report MUST say which form was used and what seed drew it.
    It MUST also say the actual and declared dtype of every activation and of
    every weight the selected Module declares, plus the tensor count and shape
    tree each `--input` file supplied.
  - A FAIL with `--inputs random` MUST state that the draw makes each activation
    independently; a target that relies on semantic relationships between
    activations MAY differ at ulp scale without either implementation being wrong,
    and `--inputs real` is the re-run that decides the comparison.
  - A FAIL measured against a reference MUST state that it proves disagreement,
    not which side is closer to truth. A reference MAY carry its own rounding, and
    establishing accuracy needs an independent high-precision reference that
    `check` does not run.
  - Reaching one leaf MUST read only that leaf Module's own weights. A comparison
    of one kernel MUST NOT materialise a whole model. A Module is the unit that
    loads, so what a run binds is everything the selected Module declares, not the
    subset the selected function names; the selector's child segments MUST scope
    the checkpoint by the same names they resolve the Module by, so the two cannot
    be addressed differently.
  - A dimension the target states as a range MUST be reported, along with the
    extent this run pinned it to; several extents MAY be stated for one dimension,
    and each MUST be run and reported. Where the extents select an implementation,
    the report MUST name the one selected and the range it covers. Naming it is what
    separates "it ran" from "it ran the intended program", so a run that only passed
    is not evidence that dispatch landed where the author meant.
  - Reporting a pin MUST also state both ways out of it: binding the dimension, and
    declaring a variant that covers the size.
  - An extent no declared variant covers MUST fail, naming the ranges that are
    covered. Choosing a neighbouring implementation instead would answer about a
    program nobody selected, and the failure is only actionable if the reader can
    see where the coverage stops.
  - Text and `--json` MUST carry the same facts.

## Tutorial

`tutorial` teaches the workflow: what to do, in what order, and at what
granularity.

- constraints:
  - It MUST point at `spec` for normative and reference material and at
    `check --help` for the predicate flags, and MUST NOT duplicate either. A
    second copy of a contract is a copy that goes stale, and the reader cannot
    tell which one is current.
  - Its pages ship as data beside the specifications and MUST be read from the
    same installed lookup, so a page is available to an installed wheel and not
    only to a checkout.
  - Where a page teaches by example, the example MUST be the shipped model source
    itself, selected by what a declaration is called rather than by where it sits.
    A copy pasted into prose is a second source that drifts, and a line range
    silently quotes the wrong lines as soon as the model above it changes.
  - `tutorial orchestrator` MUST list the shipped orchestrator families, and
    `tutorial orchestrator FAMILY` MUST show that family's shipped directory and
    every source file's leading docstring without importing or executing it. An
    unknown family MUST name the available families. Checkout and installed
    lookups MUST report the same shipped families and files.
  - Its workflow pages are `index`, `migrate`, and `optimize`; causal-LM decode
    sources are listed through `tutorial orchestrator`.
  - A family's list description MUST be the leading docstring of the first file
    in stable filename order.

## Models

`models` reports the models this project has described, and hands back one model's
authored source as a reference to copy from.

It reads a shipped catalog and MUST NOT import or execute a model's source. An
installed package carries the authored sources as read-only data and nothing that
could make them importable, so executing them is not available; and a reference
that runs before it can be read is a reference that decides what it describes.

- constraints:
  - With no `NAME`, output MUST list every described model with its counts. The
    catalog states what each model *is*, not how well it is verified: a ranking
    carried in shipped data is a claim about tests that the shipped artifact
    cannot check, and one that outlived the test it named is worse than none.
  - With a `NAME`, output MUST be that model's whole forest: every root it
    publishes, each Module's own functions with their signatures beneath it, and
    the leaf Modules marked. A leaf is a Module with no child Modules, not
    a function — a runtime twin is written per Module and MUST cover all of that
    Module's functions at once, so marking functions would state the work at a
    granularity nobody implements at.
  - A root MUST be a Module that declares the target its tree runs on, and a
    Module that declares none MUST NOT be listed as one. Declaring the machine is
    what makes a Module answerable on its own; a component or layer template
    reached through the root that owns it would otherwise be offered as a selector
    that Analyze and Schedule cannot answer.
  - A run of sibling Modules MAY be written once as the range it covers, and only
    when the run is adjacent, identically shaped down its whole subtree, named from
    one stem, and numbered consecutively. Such an entry MUST name every Module it
    stands for and MUST say how many there are: it is the complete tree written as
    ranges, not a tree with repetition left out. Distinct siblings MUST stay
    separate. Without this a stack states its one layer forty times and the reader
    is back to reading a dump.
  - The leaf-Module count and the function count MUST come from one traversal, so
    the numbers cannot disagree with the forest printed beside them, and they MUST
    count every Module a range stands for rather than the range as one.
  - `--source` MUST print the absolute path of the shipped model directory first,
    followed by one line for every file named in the package data manifest, in
    a stable filename order. Each file line MUST give its filename and the first line of
    its own docstring, or `-` when it has none. A checkout MUST read that manifest;
    an installation MUST read its model directory, and both MUST name the same
    files.
  - `--source` MUST parse a file's text to read its docstring and MUST NOT import
    or execute model source. It MUST NOT reformat, regenerate, or copy a shipped
    file: the installed directory is the reference, and a rendered copy is a
    different artifact wearing its name.
  - Catalog membership and `--source` describe a model and its shipped authored
    files; neither claims that the model has a runnable decode path. The package
    data manifest alone defines those files. Repository-only helpers such as
    `hf_alias.py` MUST NOT create an implicit requirement to ship `run.py` or
    `generation.py`.
  - A `NAME` the catalog does not have MUST be refused naming the models it does.
  - The forest and the counts MUST be generated from the models themselves rather
    than maintained beside them, because a hand-kept inventory of trees and numbers
    drifts silently from what it claims to describe.

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

Each flag names one root analysis. With no analysis flag, `analyze` requests no
family: it type-checks the selection and prints its complete inferred HIR. The
selected Module's resolved Target determines the hardware specification for an
explicit analysis; there is no ordinary `--target` option.

- constraints:
  - `analyze` MUST invoke the public operation once with every requested root,
    so their union dependency closure runs on one inlined Function view
    ([analysis §3](./analysis.md#3-composed-analysis)). Each closure member MUST
    run once, and requesting another root MUST NOT let one analyzer change
    Metadata owned by another.
  - A selection MUST resolve to a Module. A bare Function MUST be rejected
    naming the reason: it declares neither the Target its numbers are measured
    against nor the topology hierarchy they divide over.
  - `--json` MUST print the report as JSON instead of text. It MUST be refused as
    an argument-combination error when no analysis flag was supplied, naming that
    a report needs a requested root and printing the `analyze` usage. Both
    formats MUST carry the same conclusions
    ([analysis §2](./analysis.md#2-authored-hir-metrics)).
  - `--topology LEVEL` MUST be optional, passed through as the public analysis
    operation's `level`, and name the unit for per-unit figures. Its help MUST
    state the default and, for every family, which figure changes with the level
    and when to pass it, together with the global-traffic and observed-peak
    assumptions. Compute cost MUST name both `flops_per_unit` and
    `traffic_per_unit` as projected figures while keeping `flops` and `traffic`
    explicitly global.
    With no analysis flag it MUST be accepted and inert.
  - `--dim NAME=EXTENT` MUST bind one dimension the selection leaves open, and
    MUST be repeatable to bind several. One dimension MUST receive one extent;
    a comma-separated list of extents for one dimension MUST be rejected because
    several extents together are a `check` request. It MUST be passed through as the
    operation's `dims` ([analysis §2.2](./analysis.md#22-analysis-families));
    the CLI MUST NOT specialise the selection itself, because then what it
    printed would be about a program the operation never saw.
  - A `--dim` argument that is not `NAME=EXTENT`, or whose extent is not an
    integer, MUST be rejected naming which argument and why.
  - Repeating `--dim` states another dimension. One dimension stated twice MUST
    be rejected naming that dimension, whether or not the two extents agree; the
    later occurrence MUST NOT win, because both came from the caller and choosing
    between them silently answers a request that has no answer.
  - With no `--dim`, the selection MUST be analysed or type-checked as authored.
    A selection that leaves a dimension open MUST then fail naming the dimension,
    its declared `[lo, hi)` interval, and concrete extents inside that interval
    the caller can use: inferring its concrete program requires an extent, and a
    range is not one. The bare form MUST apply stated dimensions before running
    the public program check used by Analyze and Schedule, followed by the same
    authored-analysis readiness validation as Analyze; this is an internal CLI
    path, not a public typecheck operation. It MUST reject an unsupported declared
    topology level or an extent over that level's target limit, even though no
    analysis family was requested.
  - Every requested analysis MUST be reported together even when each was run at
    the stated extents, which builds one program per analysis. The report MUST
    accept those as one program when they were rebuilt from the same function at
    the same extents, and MUST refuse results rebuilt at different extents. The
    comparison MUST read the recorded extents and MUST NOT infer them from the
    resulting signature: a dimension occurring only in a loop bound or a body
    operation's attribute leaves the signature identical at every extent.
  - Output MUST report the analyses that were requested. A dependency that ran
    because a requested root needed it MUST appear in the executed list and, other
    than the bounded roofline support view defined by
    [analysis §2](./analysis.md#2-authored-hir-metrics), MUST NOT have its own
    measurements reported.
  - The report's `target` field MUST be the concrete Target value's `identity`,
    so two products served by one Target class remain distinguishable.
  - On success with at least one requested root, text output begins with the
    `#`-headed report followed by annotated HIR. With no analysis flag, output is
    the typed HIR alone, with no report header and no analysis Metadata comment.
    On inference, verification, or analysis failure, stdout MUST be empty and
    stderr MUST report the source location, binding where available, and reason.

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
    the operation's `dims` ([schedule §1](./schedule.md#1-the-public-schedule-operation)).
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
@func(
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 4),),
)
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

## Target

`target list` prints every Target value constructible in the current
environment. Each row contains its exact `identity` and a Python expression
that reconstructs an equal value; the required imports follow the rows. Values
made available by an explicit addition are marked, and the persistent entries
and their absolute sources are listed separately.

`target show IDENTITY` addresses those same rows by exact identity. For a
document-backed Target it prints the architecture and device documents retained
by the value, each with its digest, then every recorded fact by path. A fact
identifies its unit, conditions, source, and origin. A fact with no usable value
is reported as unavailable rather than given a placeholder number.

For a Target without retained documents, `target show` prints only its identity,
its reconstructing expression, and `facts: unavailable`. It does not query or
attempt to enumerate `get_facts`: a Target exposes facts on demand but has no
interface that claims to list every Facts projection it supports.

- constraints:
  - Target discovery MUST be an explicit `target add` operation. Installing a
    Python distribution MUST NOT implicitly register a Target through package
    metadata or entry points.
  - `target add --document PATH` MUST parse a complete hardware document, find
    the Target that owns its schema, validate and adopt it, and persist its
    absolute source path and content digest. Adding a device whose declared
    architecture is unavailable MUST fail naming the architecture to add first.
  - Without `--document`, `target add` MUST import a module name unless the
    argument ends in `.py` or names an existing file. A file source MUST be
    stored as an absolute path, loaded under its filename stem, executed on
    every command that loads the registry, and reported as such when added.
    Only one file source with a given stem may be added; a collision MUST fail
    naming the absolute path that already occupies that module name. A stem
    already occupied by any other importable module MUST likewise be rejected
    without replacing that module in `sys.modules`.
  - Every command MUST replay the registry before doing its own work, so an
    added Target is equally available to inspection, analysis, and scheduling.
    A missing or changed source MUST produce a warning naming that entry while
    valid entries continue to load and the requested command continues.
  - The default writable registry MUST be
    `<sys.prefix>/share/tilefoundry/registry.toml`. It MUST NOT use checkout
    data-file discovery or a user-global directory. `--registry PATH` MUST
    override it without reading or writing the default path.
  - A document entry MUST retain its original bytes at its source: replay MUST
    compare the content digest and MUST NOT silently adopt changed contents or
    store a private copy. Adding the document again is the explicit update.
  - Every available Target identity MUST be unique across registered provider
    classes and device documents. A document ID already in a `HardwareSpec`
    MUST retain the document-duplicate diagnostic; a Target identity collision
    MUST instead name the value and provider that already occupy it.
  - `target remove` MUST accept a document ID, a module name, or any identity
    loaded from that module. Removing a module MUST report every identity it
    removes, and the next command MUST no longer load those values.
  - Every expression printed by `target list` MUST be executable with the
    accompanying imports and MUST reconstruct the Target named by that row.
  - `target show` MUST accept every identity printed by `target list`. An
    unknown identity MUST fail naming the identities currently available.
  - `inspect` MUST NOT remain as a second Target inspection command.
