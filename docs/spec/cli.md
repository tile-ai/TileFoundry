# TileFoundry Command-Line Interface

This file defines the command-line contract for translating a published model into
authored HIR and turning that HIR into a runtime implementation. The commands answer
questions that only this project can answer. Which models are described? What does a
specification section say? Does an implementation agree with its reference? What
does an authored program cost? Authoring syntax is in the
[parser specification](./parser.md). The authored IR is the
[HIR specification](./hir.md).

Naming no command at any command level MUST print that level's `--help` page,
byte for byte, and MUST exit 0. The page content is the parser's to render and
is not restated here.

On a command usage error, the command MUST print its usage form and then the error
to standard error. It MUST exit with status 2.

`--version` MUST print the installed distribution's version and exit. The version
and the project summary MUST come from installed distribution metadata. With no
metadata available, the command surface MUST fail. It MUST NOT use a version or
summary restated in this project's source.

## Commands

`SOURCE` is a Python file followed optionally by
`:Module[.function]`. Without a selector, the source must define exactly one
top-level HIR Module or Function. A selector chooses the named Module and its
default entry, or one of that Module's own Functions. Child Module path segments
MUST be refused; a child used independently is named by its own top-level binding.

A selector MAY name a top-level binding that is a bare `Function`. A source whose
`@func` declares no execution context binds one. Every verb here reads hardware
facts. Such a selection MUST be rejected, naming the Module that would declare its
context. It MUST NOT be analysed against a default
([target §6](./target.md#6-target-ownership-and-compile-resolution)).

The file in `SOURCE` is any readable Python file. The command surface MUST NOT
privilege the model sources this project ships. A copy of a shipped model directory,
edited to coarsen a boundary, MUST reach the same command surface. The installed
directory named by `models <name> --source` is the reference to compare against.

A refusal for a shipped document name that does not exist MUST be the same at every
kind. It MUST name what was asked for and what exists, in the vocabulary the command
takes. It MUST NOT name an internal filename.

A command MUST load `SOURCE` as a Python module. While loading it, the directory
containing `SOURCE` MUST be first on the Python module search path. A file beside it
MAY then be imported by its module name. A command MUST capture and suppress
standard output that `SOURCE` emits while loading.

How much of `SOURCE` is executed depends on whether it named a root, and the two
boundaries are different:

- Without a selector, `SOURCE` MUST be executed as a whole. The first statement that
  fails MUST be reported as itself, because the caller asked about the file.
- With a selector, each top-level statement of `SOURCE` MUST be executed on its own
  and in source order. A statement that fails MUST be set aside. It MUST NOT end the
  load. No statement MAY be skipped unexecuted. What the selection needs is whatever
  it turns out to read. A reader cannot decide that from the text without deciding
  which names are a function's own.

  `SOURCE` MUST first be compiled whole. Every rule that the language applies to a
  module MUST still apply. A `__future__` import that is not at the beginning of the
  file MUST be refused here, as an interpreter refuses it. A selector asks
  the command to execute less of a file, not to accept more of the language.

  `SOURCE`'s leading `__future__` imports MUST then prefix each statement, so that
  they still govern it. Each statement MUST be compiled without inheriting the
  compiling code's own `__future__` flags. The module docstring MUST NOT be
  prefixed. It is the docstring because nothing precedes it. A load that put
  anything in front of it would leave the module with none. A file decides what it
  postpones. A file that postpones its annotations MUST keep them postponed. A file
  that does not postpone them MUST have them evaluated, exactly as loading it as a
  whole would.

  A statement that is the selection's own MUST NOT be set aside. That is a statement
  that binds the selected root, or that reads it while the file runs. Reads means
  free reads, not spellings. A name being written is not a reading of it. A function
  body is not read then. A class attribute or a parameter that shares the root's
  name is not a reading of it. A comprehension's variables are its own, every one of
  them. Only its first iterable is evaluated where the comprehension is written, so
  a name read there is a name from outside. Every later iterable is evaluated
  inside, where all the targets already belong to it.

  What decorates a function and what defaults its arguments IS read then. Whether
  its annotations are read is the file's own decision. Where `SOURCE` postpones its
  annotations, they stay strings and reach for nothing. Where it does not, they are
  read like anything else. Such a statement builds or reconfigures what was named.
  Its failure MUST be reported, even when an earlier statement also failed. It MUST
  also be reported when the root is already bound. Only an `Exception` MAY be set
  aside at all. Anything else is the interpreter unwinding rather than an unfinished
  program. It MUST propagate.

  A failure that follows from one already set aside is the exception. A statement
  reaching for a name that a set-aside statement would have bound fails for that
  reason, not its own. Reporting it would name the symptom instead of the cause.
  Such a failure MUST be set aside too.

  The selected root MUST then be resolved from what the load bound. If it is not
  bound, the load MUST fail with the first failure it set aside. What failed is
  something the selection needed. A root reached under a second name is that same
  root and keeps its own name.

A class body runs when the file does, so a class that refuses to be built refuses
there.

`check` reads the same `SOURCE` shape. Its selector MAY also name a runtime twin
instead of an authored Module. A twin generated from an authored Module states that
Module ([runtime §1.1](./runtime.md#11-runtimemodulepy)). Naming the implementation
is then enough to reach what it is judged against. A runtime module that states none
MUST be refused rather than compared against something chosen for it.

## Scaffold

```text
tilefoundry scaffold SOURCE OUT_DIR
```

The project is determined by the generator of the Target resolved by the selected Module.
`SOURCE` uses the shared authored-Module selector syntax described under
[Commands](#commands). The command accepts HIR or TIR and delegates Target
capability checks to codegen ([codegen §2.1](./codegen.md#21-target-resolution)).
`OUT_DIR` MUST be absent or an empty directory; existing nonempty output MUST
be rejected without overwriting files.

The command runs host-entry normalization
([passes §7.3](./passes.md#73-inserthostentrypass)), selects all CPU entries,
and calls skeleton `codegen/link_module.py:link_program`
([codegen §4.3](./codegen.md#43-linkedmodule)). It writes `CMakeLists.txt` and
the target-selected source files directly in `OUT_DIR`, compiles the project
and returns zero on success. The library is
`OUT_DIR/build/lib<root Module name>.so`. Generated device bodies are editable
placeholders; signatures, views, variant dispatch,
launchers and host registrations remain intact. The project can be rebuilt with
`cmake -G Ninja -S OUT_DIR -B BUILD_DIR` followed by `cmake --build BUILD_DIR`. Load the
library with `tvm_ffi.load_module`, then bind its dotted registrations with
`tvm_ffi.get_global_func`. The caller manages this library's lifetime. The caller
MUST retain it while its registrations may be called. This does not involve a
`runtime.Module`. Authored host orchestration remains in Python.

## Check

`check` is the one command that reports agreement. It runs an implementation and,
when there is one, its reference, and says of every output whether it meets the
bounds the caller stated.

- constraints:
  - `--inputs random` draws one activation per non-constant parameter in
    declaration order. `--inputs files:A.pt,B.pt` reads one file per activation
    parameter in that same order. Output names MUST come from return position. One tensor is `output`. A tuple's tensors are `output[0]`, `output[1]`, and so on in return
    order. These are positions, not names authored in the function.
  - `--weights` states where weights come from. `random` MUST draw each weight the
    first time it is asked for. `ckpt:DIR` MUST read the weights from a safetensors
    checkpoint. It is OPTIONAL: omitted, the run has no weight source. A weight
    reached with no source MUST be refused where it is first asked for. The refusal
    MUST name the Module that declares the weight and the weight. A weight MUST NOT
    be refused ahead of the run from what the selected Module declares. A Module
    declares only its own functions' weights, so what a run reaches is not that set.
  - One input file MUST bind one parameter. Its value MAY be a bare tensor or an
    arbitrarily nested tuple or list of tensors. Every leaf MUST be a tensor.
  - A target whose step is an orchestration method, not a `@func`, MUST be refused.
    The refusal MUST name the method and the HIR functions to check instead.
    Orchestration is host Python reused verbatim on both sides. `check` compares
    HIR, not one Python function object against itself.
  - Every output MUST be judged by at least one predicate that the caller states.
    There MUST be no default predicate and no default bound. A single `f32`→`bf16`
    rounding already measures `rel_l2` 1.66e-3.
  - Naming an output that was not produced, or leaving a produced output
    unjudged, MUST be refused.
  - An empty result MUST be an error, never a PASS.
  - A predicate MUST be refused on an output whose dtype it says nothing about.
    An aggregate over a discrete (index) output MUST be refused, pointing at
    exact comparison.
  - The reference MAY be stated as files, or MAY be the evaluator running the
    authored Module the implementation stands for. With no reference at all, only a
    predicate that judges the candidate alone is admissible. Every two-sided
    predicate MUST be refused, because there is nothing to compare against.
  - Each output MUST report the norm of its reference. Near zero, the report MUST
    state what it measured instead of a relative number with no scale.
  - Activations MUST be stated, random or files. No form MAY be the default. Weights
    MUST come from the same draw on both sides. The report MUST say which form was
    used and what seed drew it. The report MUST also say the actual and declared
    dtype of every activation. It MUST also say the tensor count and shape tree that
    each input file supplied.
  - `--device DEVICE` names where inputs and weights are built, and so where the
    run happens. Omitted, it is the device the selection's Target declares. Given,
    it is honoured as stated even when the declared hardware is unavailable,
    because the caller has said where to run. The evaluator picks no
    device of its own ([evaluator §2](./evaluator.md#2-parameters-and-inputs)).
  - A FAIL with `--inputs random` MUST state that each activation is drawn
    independently. It MUST also state that a target relying on relationships between
    activations MAY differ at ulp scale. It MUST also state that a
    `--inputs files:...` run decides the comparison.
  - A FAIL measured against a reference MUST state that it proves disagreement,
    not which side is closer to truth.
  - `--json PATH` MUST write the machine-readable report to `PATH`. It MUST NOT
    write any byte of that report to stdout. Without `--json`, the human report MUST
    remain on stdout. A `PATH` whose parent directory does not exist MUST be refused
    when the arguments are parsed, before any comparison runs. The refusal MUST name
    the path.
  - Reaching one leaf MUST read only that leaf Module's own weights. A comparison of
    one kernel MUST NOT materialise a whole model. A Module loads as a whole, so
    what a run binds is everything the selected Module declares, not the subset the
    selected function names. A run MUST read the checkpoint relative to the selected
    Module ([Commands](#commands)).
  - A dimension the target states as a range MUST be bound by `--dim`. A run MUST NOT choose an extent on the caller's behalf. Several extents MAY be
    stated for one dimension, and each MUST be run and reported. Where the
    extents select an implementation, the report MUST name the one selected and
    the range it covers.
  - An extent that no declared variant covers MUST fail. The failure MUST name the
    ranges that are covered. A neighbouring implementation MUST NOT be chosen
    instead.
  - Text and `--json` MUST carry the same facts.

## Tutorial

`tutorial` teaches the workflow: what to do, in what order, and at what
granularity.

- constraints:
  - It MUST point at `spec` for normative and reference material. It MUST point at
    `check --help` for the predicate flags. It MUST NOT duplicate either.
  - Its pages ship as data beside the specifications. Its pages MUST be read from
    the same installed lookup. Then a page is available to an installed wheel and
    not only to a checkout.
  - Where a page teaches by example, the example MUST be source that the page
    executes. A page that carries programs MUST open with the command that extracts
    them from the page itself. Every later command on the page MUST run against what
    that command produced. A copy pasted into prose is a second source that drifts.
  - A rendered page MUST be the output of executing its own blocks in order against
    the current tree. The extraction command MUST run first. Then a command whose
    flags changed cannot stay in a page unnoticed.
  - `tutorial orchestrator` MUST list the shipped orchestrator families.
    `tutorial orchestrator FAMILY` MUST show that family's shipped directory. It
    MUST also show every source file's leading docstring, without importing or
    executing the files. An unknown family MUST name the available families.
    Checkout and installed lookups MUST report the same shipped families and files.
  - Its pages are `index`, `migrate`, `optimize`, `schedule`, and `showcase`. The
    first four are the workflow. `showcase` is one kernel taken through six
    analyze-driven stages. It exercises the authoring surface that the other pages
    touch in part. Causal-LM decode sources are listed through
    `tutorial orchestrator`. A bare `tutorial` MUST print the `index` page and then
    its own help. The help names the pages a reader may ask for and `orchestrator`.
    `index` is that overview's source, not a page the command accepts.
  - A family's list description MUST be the leading docstring of the first file
    in stable filename order.

## Models

`models` reports the models this project has described, and hands back one model's
authored source as a reference to copy from.

It reads a shipped catalog. It MUST NOT import or execute a model's source. An
installed package carries the authored sources as read-only data, not as importable
code.

- constraints:
  - With no `NAME`, output MUST list every described model with its counts. The
    catalog states what each model *is*, not how well it is verified.
  - With a `NAME`, output MUST be that model's whole forest. The forest MUST show
    every root that the model publishes. It MUST show each Module's own functions
    with their signatures beneath it. It MUST mark the leaf Modules. A leaf is a
    Module with no child Modules, not a function. A runtime twin is written per
    Module. It MUST cover all of that Module's functions at once.
  - A catalog root MUST declare a target. A catalog root MUST NOT be held, directly
    or transitively, by another target-declaring Module in the same model package. A
    layer that declares its own target remains usable on its own. Such a layer MUST
    NOT be listed as another root when a published top-level model holds it.
  - A run of sibling Modules MAY be written once as the range it covers. The run
    MUST be adjacent, identically shaped down its whole subtree, named from one
    stem, and numbered consecutively. Such an entry MUST name every Module it stands
    for and MUST say how many there are. Distinct siblings MUST stay separate.
  - The leaf-Module count and function count MUST count definitions by object
    identity, once per model. Repeated instances of one definition MUST NOT
    increase either count. The forest MUST still show every instance, including
    the element count of an indexed slot when its instances are grouped.
  - `--source` MUST print the model directory's absolute path alone on the first
    line. It MUST then print one line per file except `__pycache__`, in stable
    filename order. Each file line MUST give its filename. It MUST also give the first line of its own docstring, or `-` when it has none. Checkout and installation MUST list the
    same files.
  - `--source` MUST parse a file's text to read its docstring. It MUST NOT import or
    execute model source. It MUST NOT reformat, regenerate, or copy a shipped file.
    The installed directory is the reference. A rendered copy is a different
    artifact with the same name.
  - Catalog membership and `--source` describe a model and its shipped authored files; neither
    claims that the model has a runnable decode path. Model directories MUST NOT publish `run.py` or
    `generation.py`; these MUST ship once in core's `orchestrator/causal_lm/` resources.
  - The catalog MUST merge each target's `models/catalog.json`, rejecting duplicate model
    names with an error naming both targets.
  - A `NAME` the catalog does not have MUST be refused naming the models it does.
  - The forest and the counts MUST be generated from the models themselves, not
    maintained beside them.

## Spec

`spec` discloses the installed specifications a step at a time: which documents
there are, what is in one, and one section of one. It MUST NOT print a document
whole. A reader who asked what a rule says is not asking for every rule.

- constraints:
  - With no `TOPIC`, output MUST list the documents that can be asked for. It MUST
    include any name that is another name for one of them.
  - With a `TOPIC` and no `SECTION`, output MUST be that document's outline. The
    outline MUST show every section's key and title, indented by heading depth. It
    MUST NOT include the sections' bodies.
  - A section's key MUST be its own number when its heading carries one. Otherwise
    the key MUST be a name derived from its title.
  - Keys MUST be unique within a document. A key that would repeat MUST take on the
    name of its enclosing heading. It MUST keep taking the next enclosing heading's
    name until the keys differ.
  - Headings MUST be recognised outside fenced code blocks only.
  - With a `SECTION`, output MUST be that section. The section is its heading and the lines down to the next heading at its level or above. Output MUST then give the keys of the
    sections beside it. These are the previous and the next at its level, and the
    ones it contains. Naming the neighbours lets a reader walk the document without
    returning to the outline.
  - A `SECTION` that the document does not have MUST be refused. The refusal MUST
    name the keys that the document does have, so the next attempt can succeed.
  - A `TOPIC` with no installed document MUST be refused as that.
  - Documents MUST be read from the source tree when one is present. Otherwise they
    MUST be read from the installed data directory. Then the command answers the
    same in an editable checkout as in an installed environment. The source tree
    comes first because in an editable checkout the working copy is what its author
    means.

## Analyze

`analyze` first runs deterministic type inference and then writes complete type
comments, regardless of analysis flags. It never performs candidate search,
layout enumeration, or automatic resharding.

Each flag names one root analysis. With no analysis flag, `analyze` requests no
family: it type-checks the selection and writes its complete inferred HIR. The
selected Module's resolved Target determines the hardware specification for an
explicit analysis; there is no ordinary `--target` option.

- constraints:
  - The whole `analyze` command MUST have a 300-second wall-clock budget. On
    expiry it MUST write `tilefoundry: error: analysis too complex, timed out
    after 300s` to standard error, flush that message, and exit nonzero. This
    budget is currently fixed and MUST NOT be exposed as a CLI flag or public
    analysis option.
  - `analyze` MUST invoke the public operation once with every requested root. Their
    union dependency closure then runs over one checked Module
    ([analysis §2](./analysis.md#2-composed-analysis)). Each closure member MUST run
    once. Requesting another root MUST NOT let one analyzer change Metadata owned by
    another.
  - A selection MUST resolve to a Module. A bare Function MUST be rejected. It
    declares neither the Target its numbers are measured against nor the topology
    hierarchy they divide over. The rejection MUST name that reason.
  - `analyze` MUST accept an optional `PATH` whether or not an analysis flag was
    supplied. With `PATH`, its typed HIR, text report, or JSON report MUST be
    written to that path. Successful execution MUST write none of the report to
    stdout. Without `PATH`, the same report bytes MUST be written to stdout.
  - `--json` MUST write the report as JSON, to `PATH` or stdout, instead of text.
    Its `source` field MUST carry each reached definition once with its records
    ([analysis §2](./analysis.md#2-composed-analysis)).
    `--operands` MUST NOT change `source`: it is always the annotated HIR from the
    render without operand annotations.
    The JSON flag with no analysis flag MUST be refused as an argument-combination
    error. The error MUST name that a report needs a requested root. It MUST
    print the `analyze` usage. Both formats MUST carry the same conclusions
    ([analysis §1](./analysis.md#1-authored-hir-metrics)).
  - `--operands` MUST add each operand's share of a call's traffic to that call's
    annotation. It MUST NOT change the JSON report, which carries that split either
    way. It is the traffic the annotation already states, one layer finer, so a
    reader who did not ask is not made to read it. With no analysis flag it MUST be
    accepted and inert.
  - `--topology TOPOLOGY` MUST be optional, passed through as the public analysis
    operation's `topology`, and name the unit for per-unit figures. Its help MUST
    state the default. For every family, its help MUST state which figure changes
    with the topology and when to pass it. Its help MUST also state the logical/total
    traffic distinction and the observed-peak assumption. Compute cost MUST
    distinguish `logical`, `total`, and each topology level's per-unit share.
    Movement MUST name the `MemoryMetadata.traffic.storage` per-unit account against
    `total`. With no analysis flag it MUST be accepted and inert.
  - `--dim NAME=EXTENT` MUST bind one dimension the selection leaves open, and MUST
    be repeatable to bind several. One dimension MUST receive one extent. A
    comma-separated list of extents for one dimension MUST be rejected, because
    several extents together are a `check` request. It MUST be passed through as the
    operation's `dims` ([analysis §1.2](./analysis.md#12-analysis-families)). The
    CLI MUST NOT specialise the selection itself. Then what it wrote would be about
    a program the operation never saw.
  - A `--dim` argument that is not `NAME=EXTENT` MUST be rejected. An argument whose
    extent is not an integer MUST be rejected. The rejection MUST name which
    argument and why.
  - Repeating `--dim` states another dimension. One dimension stated twice MUST be
    rejected, naming that dimension, whether or not the two extents agree. The later
    occurrence MUST NOT win. Both came from the caller, and choosing between them
    silently answers a request that has no answer.
  - With no `--dim`, the selection MUST be analysed or type-checked as authored. A
    selection that leaves a dimension open MUST then fail. The failure MUST name the
    dimension, its declared `[lo, hi]` interval, and concrete extents that the
    caller can use inside it. Inferring the concrete program requires an extent, and
    a range is not one. The bare form MUST apply stated dimensions before it runs
    the public program check used by Analyze. It MUST then run the same
    authored-analysis readiness validation as Analyze. This is an internal CLI path,
    not a public typecheck operation. It MUST reject an unsupported declared topology level. It MUST also reject an extent over that level's target limit, even though no
    analysis family was requested.
  - Every requested analysis MUST be reported together, even when each was run at
    the stated extents. That run builds one program per analysis. The report MUST
    accept those as one program when they were rebuilt from the same function at the
    same extents. It MUST refuse results rebuilt at different extents. The
    comparison MUST read the recorded extents. It MUST NOT infer them from the
    resulting signature. A dimension that occurs only in a loop bound or a body
    operation's attribute leaves the signature identical at every extent.
  - Output MUST report the analyses that were requested. A dependency that ran
    because a requested root needed it MUST appear in the executed list. Except for
    the bounded roofline support view defined by
    [analysis §1](./analysis.md#1-authored-hir-metrics), such a dependency MUST NOT
    have its own measurements reported.
  - The report's `target` field MUST be the concrete Target value's `identity`. Two
    products served by one Target class then remain distinguishable.
  - On success with at least one requested root, the report begins with the
    `#`-headed report followed by annotated HIR. With no analysis flag, it
    contains typed HIR alone, with no report header and no analysis Metadata
    comment.
    On inference, verification, or analysis failure, stdout MUST be empty. Stderr
    MUST report the source location, binding where available, and reason.

## Autodiff

```text
tilefoundry autodiff SOURCE --wrt NAMES [--recompute SELECTOR ...] [--saved] [PATH]
```

`autodiff` differentiates the selected Module with
[passes §7.4](./passes.md#74-differentiate). `--wrt` is required and names the
entry parameters, comma-separated. Each `--recompute` is a selector of
[passes §7.4](./passes.md#74-differentiate), passed to `differentiate` as
written: `Module[.function][:variable]` for each element of a definition,
`Root.slot[.index][.function][:variable]` for one element of a child slot of
the root, `Root.slot[.index]:None` for an element that saves what it computes.

- Without `--saved`, the command writes `<Name>Forward` and `<Name>Backward` as
  one DSL file: one header, and each Module class, child slot classes included,
  once. The file MUST load: `SOURCE:<Name>Forward` and `SOURCE:<Name>Backward`
  each select one, and printing the two again gives the same text. The one
  exception is an entry whose forward opens, at its top level, regions of more
  than one mesh: the backward then runs ops outside every mesh, which the
  parser refuses.
- With `--saved`, the command writes one row per tensor the forward keeps for
  the backward: its name, global shape, bytes on one device (the local shape,
  since memory is counted per device), the layout of each mesh axis, the
  buffer it keeps alive (`Saved.buffer` of
  [passes §7.4](./passes.md#74-differentiate), its own name when it is its
  own), and its origin. A tuple the forward keeps for a callee has one row per
  tensor, its name indexed. Two lines follow: `held`, the bytes on one device
  of each buffer that is not a parameter, counted once and whole even when
  only a view of it is saved; and `parameters`, the same for each parameter
  buffer. Both are 0 when nothing is saved.
- Without `PATH` the output goes to stdout.
- constraints:
  - A missing `--wrt` is a usage error (exit 2). A refusal of `differentiate`
    exits 1 with its message.

## Schedule

`schedule` owns the transition from checked scheduling HIR to executable TIR
and the two reports used to choose an instruction. Its subcommands are:

```text
tilefoundry schedule finalize SOURCE PATH [--json] [--dim NAME=EXTENT ...]
tilefoundry schedule facts [OP] --target TARGET PATH [--json]
tilefoundry schedule candidates SOURCE PATH [--json] [--dim NAME=EXTENT ...]
```

`analyze`, `candidates` and `finalize` share one front end: load the source, bind
one extent per `--dim`, and check the entry. `finalize` then runs the analyses
the Target's `ConvertHIRToTIR` names in `analyses`. It writes the lowered Module
as canonical TIR source following
[passes §7.1](./passes.md#71-converthirtotir), also when it holds a single
PrimFunction. With `--json`, `PATH` instead
contains an object whose `source` field is that same canonical TIR. `facts` describes the Op declarations admitted
by the exact `TARGET` identity, optionally narrowed to one fully-qualified
`OP` name. With no `OP`, it lists every registered TIR Op
that owns a class-level instruction-capability declaration. A named capability
must occur in the target architecture's capability set. A declaration with no
required capability imposes no named feature requirement. Execution mesh
Patterns are reported as declaration constraints and checked against the actual
call mesh. The facts inventory does not prove that a concrete call is valid. A carrier may associate more
than one named declaration with itself. The inventory lists that carrier once,
while each associated declaration remains addressable by its own exact name.
With or without a selected name, `facts` reports each declaration's capability,
execution mesh, parameters (including attribute ranges and defaults), operands,
and cross-operand constraints. The `ops` inventory contains the selected Op declarations.
Text and JSON read the original declarations directly; JSON dictionaries exist only
at the serialization boundary. The inventory JSON has `target` and `ops` fields;
a selected declaration is serialized on its own. Text and JSON use the same detailed declarations;
JSON inventory entries carry the same fields as a selected declaration. Supported named variants are included with their complete declarations, so
a whole-target report also contains each admitted atom's parameter and operand constraints. The
parameter and operand sections use `PatternPrinter.declaration`:
conditional constraints stay symbolic and are printed directly. Facts MUST NOT
enumerate Cartesian products of parameters or invent shared-subtree definitions. A carrier with an
associated attribute reports execution meshes as alternatives selected by each
declaration.
`candidates` reports the instructions whose access relation and operand
patterns can implement each unscheduled registered HIR site. A target-neutral
pairing registry decides which TIR carrier families are relevant to each HIR Op. A HIR Op with no pairing is not a candidate site. Rank-0 UMAT values
are unmaterialized scalars rather than instruction sites, regardless of their
HIR operation class. Target capabilities then
silently remove paired carriers unsupported by the selected target. Every remaining instruction uses the same match entry as explicit
`tf.schedule`, including field matching, tile projection, verifier, and
cross-operand checks. The registered callback supplies every operand type,
including the result, and instruction attributes. Each instruction has one
candidate or refusal entry; carrier variants use their associated declaration
name. Both kinds show the full ordered matching path. Distinct values across
accepted instances join with `|`, including all admitted tile extents. A
refusal places the failed operand/node reason on a second indented line.
JSON entries contain `path` and `reason`; a candidate has an empty reason.
There is no `needs` column. A site with no accepted instruction reports its
refusals, and a missing execution mesh is a refusal rather than an internal error.
An inventory Op with no access relation is not comparable and is skipped by
candidate discovery; explicitly selecting that Op in `tf.schedule` remains an
error.

- constraints:
  - `candidates --dim NAME=EXTENT` and `finalize --dim NAME=EXTENT` MUST bind
    one extent per dimension before analysis, with the same format and
    duplicate-dimension validation as Analyze.
    An unbound dimension MUST be named with its closed declared interval and
    suggested extents. The command MUST NOT write a report when binding fails.
  - `PATH` is required, is the only report destination, and MUST NOT be partly
    written when the operation fails. A successful command writes a trailing
    newline and writes nothing to stdout.
  - `finalize` MUST run lowering through `PassManager`, so every changed
    `PrimFunction` receives the manager's automatic TIR verification.
  - `--json` changes only representation. It MUST carry the same TIR source or
    report facts as the text form.
  - `facts --target` is required. `TARGET` and `OP`, when present,
    match exact identities; neither accepts a short alias or a best-effort
    fallback.
  - Six failures MUST print `tilefoundry: error: REASON` to stderr and exit 1. They
    are: a selected instruction with no access relation; an addressable result
    without an analyzed offset; an unsupported HIR call; an unknown instruction; an
    unknown target; a source with no candidate site. No partial output is permitted.

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
its reconstructing expression, and `facts: unavailable`. It does not query or attempt to enumerate `get_facts`. A Target exposes facts on demand. It has no interface that claims to list every Facts projection it supports.

- constraints:
  - Target discovery MUST be an explicit `target add` operation. Installing a
    Python distribution MUST NOT implicitly register a Target through package
    metadata or entry points.
  - `target add --document PATH` MUST parse a complete hardware document and find
    the Target that owns its schema. It MUST validate and adopt the document. It
    MUST persist the document's absolute source path and content digest. Adding a
    device whose declared architecture is unavailable MUST fail naming the
    architecture to add first.
  - Without `--document`, `target add` MUST import a module name unless the
    argument ends in `.py` or names an existing file. A file source MUST be stored
    as an absolute path and loaded under its filename stem. It MUST be executed on
    every command that loads the registry. It MUST be reported as such when added.
    Only one file source with a given stem may be added. A collision MUST fail,
    naming the absolute path that already occupies that module name. A stem already
    occupied by any other importable module MUST likewise be rejected without
    replacing that module in `sys.modules`.
  - Every command MUST replay the registry before doing its own work. Then an added
    Target is equally available to inspection and analysis. A missing or changed
    source MUST produce a warning naming that entry. Valid entries continue to load,
    and the requested command continues.
  - The default writable registry MUST be
    `<sys.prefix>/share/tilefoundry/registry.toml`. It MUST NOT use checkout
    data-file discovery or a user-global directory. `--registry PATH` MUST
    override it without reading or writing the default path.
  - A document entry MUST retain its original bytes at its source. Replay MUST
    compare the content digest. It MUST NOT silently adopt changed contents or store
    a private copy. Adding the document again is the explicit update.
  - Every available Target identity MUST be unique across registered provider
    classes and device documents. A document ID already in a `HardwareSpec` MUST
    retain the document-duplicate diagnostic. A Target identity collision MUST
    instead name the value and provider that already occupy it.
  - `target remove` MUST accept a document ID, a module name, or any identity
    loaded from that module. Removing a module MUST report every identity it
    removes, and the next command MUST no longer load those values.
  - Every expression printed by `target list` MUST be executable with the
    accompanying imports. The expression MUST also reconstruct the Target named by
    that row.
  - `target show` MUST accept every identity printed by `target list`. An
    unknown identity MUST fail naming the identities currently available.
  - `inspect` MUST NOT remain as a second Target inspection command.
