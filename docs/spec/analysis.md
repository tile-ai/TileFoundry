# TileFoundry Spec — analysis (polyhedral model + per-stage target facts)

This spec owns TileFoundry's fact layer: everything a later stage decides
*over*, and nothing that decides anything itself. It has two surfaces:

| Surface | Entry | What it states |
|---|---|---|
| Polyhedral model | `extract(hir) -> TileGraph` | one HIR `Function` body as isl domains, access relations and auto-inferred dependences — target-independent |
| Program check | `check_program(module, function, level=..., budget=..., analyzers=...)` | an inlined Function view after validating one authored program, its declared topology, and what each requested analysis needs of it |
| Composed measurement | `analyze(module, function, analysis=...)` | one or more root analyses and their union dependency closure, leaving typed Metadata on the IR |

Per-Op semantic derivation — typeinfer, the forward access relation, shard
propagation — is owned by [semantic-analysis](./semantic-analysis.md), and the
registries behind it by [visitor-registry](./visitor-registry.md); the
polyhedral model consumes the forward relation
([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation))
rather than restating it.

**Layering.** The decisions taken over these facts are owned by
[schedule](./schedule.md#4-kernel-schedule-construction). The dependency is
one-way: the schedule layer reads this layer's facts, and this layer MUST NOT
import or otherwise depend on the schedule layer. The atom catalogue and the
store a tile lives in are the schedule layer's own inputs and are owned there
([schedule §5](./schedule.md#5-scheduling-facts)).

## 1. Polyhedral model

One extraction models one HIR `Function` body as a set of *statements* at
**element** granularity. A statement is one compute op at one call site; its
iteration domain is the op's own element domain, prefixed by one dimension per
authored loop enclosing it. Nothing here retiles, reorders, or searches: the
model states what the authored program accesses, and what must run before what.

### 1.1 `TileUnit`

```python
class TileUnit:
    """One statement's identity.

    Attributes:
        name: attribute; isl tuple name shared by this statement's domain, reads, writes and deps pieces.
        op: attribute; The HIR Call (one op at one call site) that produced this statement.
    """

    name: str
    op: object
```

- constraints:
  - The structure MUST be immutable.
  - `name` MUST be a valid isl identifier and MUST be the tuple name of every
    `TileGraph` piece belonging to this statement.
  - `name` MUST be unique within one extraction: a name two statements would
    share MUST be disambiguated with a numeric suffix, and a statement
    contributed by a penetrated nested `Function` MUST additionally carry that
    call site's own prefix.
  - `op` MUST be the `Call` node itself, so a consumer can recover
    `op.target`, `op.args` and `op.type`; it MUST NOT be the bare `Op`.

### 1.2 `TileGraph`

```python
class TileGraph:
    """Provide the polyhedral model of one HIR Function body.

    Attributes:
        domain: attribute; Union of every statement's iteration domain, one named tuple per statement.
        deps: attribute; Auto-inferred read-after-write must-dependence between statement instances.
        reads: attribute; Union of every statement's read access relations, statement tuple to buffer tuple.
        writes: attribute; Union of every statement's write access relations.
        units: attribute; One TileUnit per statement, in dependence-respecting order.
        params: attribute; isl parameter name to the ShapeDim it stands for.
        buffer_dtypes: attribute; Buffer tuple name to the DType its elements carry.
        parallel_dims: attribute; Statement name to one flag per own domain dimension, set when that dimension carries no dependence.
    """

    domain: "isl.union_set"
    deps: "isl.union_map"
    reads: "isl.union_map"
    writes: "isl.union_map"
    units: tuple[TileUnit, ...]
    params: dict
    buffer_dtypes: dict = field(default_factory=dict)
    parallel_dims: dict = field(default_factory=dict)
```

- constraints:
  - The structure MUST be immutable; a stage that adds a fact MUST return a
    replaced copy rather than mutate one.
  - `domain`, `reads` and `writes` MUST be unions of per-statement pieces with
    one isl tuple name per statement (the domain, and the input side of the
    access relations) or per accessed buffer (their output side). The domain
    MUST NOT be a single unnamed set: the schedule tree needs one named tuple
    per statement.
  - `deps` MUST relate statement *instances* and MUST be derived from
    `reads` / `writes` by the extraction itself ([§1.3](#13-extract)), never supplied by a
    caller.
  - `units` MUST be ordered so that every dependence runs forwards: the order
    is the body's SSA-DAG postorder, which makes sequencing the statements in
    that order legal by construction.
  - `params` MUST resolve every isl parameter name appearing in `domain` back
    to its `ShapeDim`. One parameter name resolving to two different
    `ShapeDim`s across statements MUST raise.
  - `buffer_dtypes` MUST record an element `DType` for every buffer an access
    relation names, so a byte count over an access relation needs no second
    walk of the HIR.
  - `parallel_dims` MUST carry exactly one flag per own domain dimension of
    every statement, and MUST be measured from `domain` + `deps` ([§1.7](#17-parallel-dimensions)) rather
    than reported by a scheduler.
  - Schedule trees, ring depths, and decisions MUST remain schedule-owned state
    outside `TileGraph`; schedule program views pair those values with the
    immutable analysis graph without mutating it
    ([schedule §4](./schedule.md#4-kernel-schedule-construction)).

### 1.3 `extract`

```python
def extract(hir: Function) -> TileGraph: ...

class ExtractError(NotImplementedError):
    """A construct the polyhedral extraction does not model."""
```

`extract` walks `hir.body` in SSA-DAG postorder (dependencies before
dependents) and classifies every node it meets:

| Body node | How `extract` models it |
|---|---|
| `Call` of a compute op | one statement; or one statement per output branch when the outputs cannot share one domain (`RoPE`'s grouped-query `q` / `k`, whose head counts differ) |
| `Call` of `TupleGetItem` / `Reshape` / `IndexSelect` / `Slice` | structural view — no statement. It resolves to its source's buffer name, and the coordinate change it expresses is folded into every consumer's access map |
| `Call` of `Zeros` / `FullLike` | buffer declaration — no statement and no access relation: it names a fresh buffer and gives it a starting value |
| `Call` whose target is a `Function` | penetrated, not rejected: the callee's params bind to the caller's already-resolved argument expressions, its body is walked in place, and every statement and buffer it contributes is prefixed with the callee name plus a per-call-site index |
| `GridRegionExpr` | not a statement — it contributes one leading domain dimension to every statement it encloses ([§1.4](#14-authored-loops)) |
| `Tuple` | resolved through the same substitution table; no statement |

- constraints:
  - Each statement's access relations MUST come from the forward
    (input-type-driven) relation service
    ([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation)).
    `extract` MUST stamp the statement and buffer tuple names onto each
    returned map and restrict it to the paired domain, and MUST reuse each
    access map's own formula unchanged — no retiling happens here.
  - Before the relation is built, every argument type MUST be narrowed to its
    per-shard local shape when it carries a `ShardLayout`: each mesh `Split`'s
    target *tensor* axis divided by that mesh axis's extent, tensor rank
    preserved. A `Partial` / `Broadcast` / `Dynamic` mesh axis consumes no
    tensor axis. Narrowing is centralized in the extraction, so every
    registered relation is sharding-aware without knowing sharding exists.
  - A `Split`-sharded axis whose extent is not a static integer, or is not
    evenly divisible by its mesh extent, MUST raise `ExtractError`.
  - A loop-indexed `Slice` read MUST map result coordinate `u` to source
    coordinate `start + u * stride`. Its consumer statement domain contains
    only full windows (`start + size * stride <= source_extent`). No residual
    tail domain is returned. A non-affine start or non-static window step MUST
    raise `ExtractError`.
  - An op with no registered forward relation MUST raise `ExtractError` naming
    the op and the registration remedy. `extract` MUST NOT guess an access
    pattern and MUST NOT carry a per-op fallback.
  - A statement MUST write at least one value; a relation that produced no
    output map MUST raise `ExtractError`.
  - A statement with several outputs MUST write them under its own buffer name
    suffixed `_<index>`, matching what a downstream `TupleGetItem` read
    resolves to.
  - An output access map that is **not** injective MUST also be recorded as a
    read of the same buffer: two domain points writing one output cell is only
    sound as a read-modify-write accumulation, and recording that self-read is
    what lets the dependence inference discover a reduction carry. An injective
    output map MUST stay a pure write.
  - `deps` MUST be inferred by isl dependence analysis over `reads` (sink),
    `writes` (must-source) and an initial total execution order, keeping the
    resulting read-after-write must-dependence. That initial order MUST place
    the authored loop coordinates *before* the per-statement postorder index, so
    a value written at one iteration is seen by a read at the next; a
    statement-first order would lose every loop carry.
  - `extract` MUST raise `ExtractError`, naming the offender, for: a body that
    is `None`; a body with no compute op; a self-recursive nested call; a
    dispatch prototype (a callee carrying variants or no body); a nested call
    whose arity does not match the callee's params; and a `DimVar` that binds to
    conflicting shapes at one call site.

### 1.4 Authored loops

An authored loop is a `GridRegionExpr` ([hir §1.2](./hir.md#12-gridregionexpr)).
It is modelled as a domain dimension, not as a statement.

- constraints:
  - Each enclosing loop MUST prefix one dimension to the domain of every
    statement it encloses, outermost loop first, so the loop axes are the
    leading dimensions of that domain.
  - The dimension MUST range over the loop's own half-open `[start, extent)`
    and MUST carry the raw induction value rather than a normalised trip
    counter; a `step` other than `1` MUST appear as a stride constraint on it.
  - `start` and `step` MUST be static integers. `extent` MAY be a bare
    `DimVar`, which becomes a same-name isl parameter bound to its own
    `[lo, hi)` range — the same treatment a dynamic tensor axis gets.
  - Which statements a loop encloses MUST be decided by **variance**, not
    reachability: a value is inside a loop only when it transitively reads that
    loop's induction variable or one of its carried args. A loop-invariant value
    MUST lift out, and a value read after the loop MUST be outside it even
    though the loop's yield produced it.
  - A carried arg and the value yielded into it MUST share one buffer name. The
    dependence inference then reports the loop carry as a distance-1 dependence
    along that loop's dimension, read at iteration `i` and written at `i - 1`;
    nothing states the carry separately.
  - An `IndexSelect` whose one-element index is an enclosing loop's induction
    variable reshaped to `(1,)` MUST fold into the consumer's access map at the
    selected dim, so iteration `i` addresses slice `i`. Any other index MUST
    raise `ExtractError`: a data-dependent selection has no affine access map
    and MUST NOT be approximated.

### 1.5 Facts over a time relation

Four measurements take the time relation as data — one `isl.union_map` from
statement coordinates to a common time space, which the schedule layer owns and
this layer only reads.

```python
def time_extents(tg: TileGraph, time_map: "isl.union_map") -> tuple[int, ...]: ...

def statement_time_dims(tg: TileGraph, time_map: "isl.union_map") -> dict[str, tuple[int, ...]]: ...

def carried_distances(tg: TileGraph, time_map: "isl.union_map", n_dims: int) -> dict[str, tuple[int, ...]]: ...

def access_footprints(tg: TileGraph, time_map: "isl.union_map") -> tuple[AccessFootprint, ...]: ...
```

- constraints:
  - `time_extents` MUST return the per-dimension extent of the time relation's
    range over `tg.domain`. Every statement MUST share one time space, and every
    time dimension MUST start at `0` — tile counting assumes an origin-based
    extent — otherwise it MUST raise `ExtractError`.
  - `statement_time_dims` MUST report, per statement and per time dimension, the
    statement's own domain dimension that time dimension travels with, or `-1`
    where it is constant there. A time dimension mixing two domain dimensions (a
    skewed band) MUST raise `ExtractError`: no per-axis tile size describes it.
  - `carried_distances` MUST report, per buffer, the largest dependence distance
    isl reports along each of the first `n_dims` time dimensions. A dependence
    MUST be attributed to every buffer its source writes and its sink reads,
    which for a read-after-write must-dependence is exactly the memory it
    travels through.
  - `access_footprints` MUST return one `AccessFootprint` per read and per write
    of `tg`, expressed against the time relation's range so that a size per time
    dimension sizes it.

### 1.6 `AxisExtent` / `AccessFootprint`

```python
class AxisExtent:
    """One buffer dimension's reach inside one statement's whole access.

    Attributes:
        axes: attribute; Time dimensions that reach this dimension; empty when none does.
        extent: attribute; Number of elements of this dimension the access reaches.
    """

    axes: tuple[int, ...]
    extent: int

class AccessFootprint:
    """One (statement, buffer) access, sized per buffer dimension.

    Attributes:
        statement: attribute; isl tuple name of the accessing statement.
        buffer: attribute; isl tuple name of the accessed buffer.
        is_read: attribute; True for a read access, False for a write.
        dims: attribute; One AxisExtent per buffer dimension.
        elem_bytes: attribute; Bytes one element of the buffer occupies.
    """

    statement: str
    buffer: str
    is_read: bool
    dims: tuple[AxisExtent, ...]
    elem_bytes: int
```

- constraints:
  - Both structures MUST be immutable.
  - `extent` MUST be measured off the access relation, never derived from a tile
    size: a dimension read at a rate reaches fewer elements than the iterations
    that reach it.
  - `axes` carries no size, only the reuse fact. An empty `axes` MUST mean no
    time dimension reaches that dimension — it is re-read in full by every
    iteration.
  - The element count of one access MUST be the product over `dims`. That count
    is the bounding box of the access's range: exact for a box-shaped access, an
    upper bound for one that leaves holes inside its own box.
  - `elem_bytes` MUST be the buffer's element size in whole bytes, resolved from
    `TileGraph.buffer_dtypes`. A buffer with no recorded dtype MUST raise
    `ExtractError`.

### 1.7 Parallel dimensions

`TileGraph.parallel_dims` is the fact isl names `coincident`, measured here
rather than obtained from a scheduler.

- constraints:
  - Only a statement's **self**-dependence MAY constrain its own dimensions: the
    schedule layer sequences statements, so every cross-statement dependence is
    already satisfied by that order.
  - A dimension MUST be reported parallel when every self-dependence has
    distance `0` there, and MUST NOT be otherwise. A statement with no
    self-dependence MUST have every dimension reported parallel.

## 2. Authored-HIR metrics

The measurement entry is the composed operation ([§3](#3-composed-analysis)).
Each family below owns its record, field derivation, target facts, and rendered
forms. The command line composes one call per requested family and renders those
results together ([cli §Analyze](./cli.md#analyze)).

Typed records use the immutable `IRMetadata` and optional comment interface
defined by [core-ir §2](./core-ir.md#2-expr). Their attachment point says what
they describe: a record on a `Call` describes that call, while a record on a
`Function` describes the whole function.

- constraints:
  - One record type MUST mean the same quantity at every attachment point.
  - A `Function`-attached record MUST NOT be read as data the Function
    inherently carries. It states what one analysis found for one invocation,
    and there MUST be no cross-call cache behind it.

### 2.2 Analysis families

The first families are `compute-cost`, `memory`, `roofline`, and `performance`.
Each owns its record types and declares its dependencies and output additions.

| Selector | Requires | Owns | Attaches to | Rests on | Text summary adds | Annotates equations |
|---|---|---|---|---|---|---|
| `compute-cost` | - | `ComputeCostMetadata` | every measured Call and the Function | the authored program | `compute-cost` | every measured Call |
| `memory` | - | `MemoryMetadata`, `TrafficMetadata`, `LoopFootprintMetadata` | `MemoryMetadata` on the Function; `TrafficMetadata` on every measured Call and the Function; `LoopFootprintMetadata` on every `GridRegionExpr` | the authored program, `MemoryHierarchyFacts` | `peak-footprint`, `traffic`, `advisory` | none |
| `roofline` | `compute-cost`, `memory` | `RooflineMetadata` | every measured Call and the Function | `ThroughputFacts` | `roofline` | every measured Call |
| `performance` | `compute-cost`, `memory` | `PerformanceMetadata`, `PerformanceSummaryMetadata` | `PerformanceMetadata` on every Call with a modeled duration; `PerformanceSummaryMetadata` on the Function | `ThroughputFacts`, `ParallelCapacityFacts`, `MemoryHierarchyFacts` | `performance` | every Call with a modeled duration |

Every compact text summary begins with these two lines:

```text
# example
# analysis target=<target> module=<module> function=<function> topology=<level>
# selection requested=<selector>[,<selector>...] executed=<selector>[,<selector>...]
```

Every summary line is one record walked exactly as an annotated equation is
([inspection §2.8](./inspection.md#28-record-comment-forms)), so the two surfaces
cannot spell one value two ways. What the report is about and what was asked of it
are records of the report rather than of the IR; every other summary line is a
record of the selected Function.

The JSON report carries the same identity and selection in `target`, `module`,
`function`, `topology`, `requested`, and `executed`. Whole-function
projections are under `function_records`; `calls` is a value-ordered list whose
entries have a `value` label and one key per selected family. `loops` is the
corresponding authored-loop list, labelled by induction variable. When memory is
selected, a loop whose backing storage has a same-scope implicit cache also has
`cache-pressure`: one target-aware row per cache, computed from the loop's
device-wide access footprint. `totals` appears when the selected view includes
compute cost or roofline's bounded work evidence.

One result is rendered once, and every surface reads that rendering:

```python
def report(result: AnalysisResult) -> dict[str, object]:
    """Project one composed Analyze result into a shared rendering structure."""
    ...

def render_analysis(result: AnalysisResult) -> AnalysisRendering:
    """Render one annotated program and its report data in a single pass."""
    ...
```

- `report` MUST accept one `AnalysisResult` and read every requested family's
  records from that result's record-bearing Function. It MUST NOT merge
  independently rebuilt Functions by identity, origin, dimensions, or walk
  position.
- Text, JSON, and annotated HIR MUST come from one `render_analysis` call over
  the same result's Function and selected Metadata types. Each rendered Call
  record's `value` is `<left-hand-side>:<line>`, where `line` is the physical
  line containing that statement's `=`, even when its comment ends a later line
  of the same statement. Both surfaces MUST use the line locations collected by
  that one printer pass rather than recover them from names or text.
- A Call equation carrying a record MUST state it in that line's Metadata
  comment. A carry update is a name rebinding rather than a Call equation, so it
  receives neither a comment nor a report row.
- A parameterized loop occurrence MUST stay one record. Neither surface expands
  it into one entry per trip.
- A compute-cost comment MUST state one unit's share beside the whole quantity it
  is a share of, and JSON MUST expose all four quantities without reconstructing
  any of them from the others.

A value renders by the type its field holds, and those forms and their separators
are owned by [inspection §2.8](./inspection.md#28-record-comment-forms). What this
layer settles is which type a field holds and what its keys name:

- A mapping's key is a dtype, a storage level, or an operand position -- an
  argument integer, or `result` for the value the Call produces.
- `<resource>` is `compute`, `memory`, `balanced`, `unrated`, or `none`.
- Bytes moved are `TrafficBytes`; a whole quantity paired with one unit's share
  is `TotalAndPerUnit`; a Call's occurrence on the timeline is one
  `TripInterval`.

- constraints:
  - A family MUST obtain hardware only through a Facts aggregate it declares
    ([target §11](./target.md#11-target-facts-projection)). Common analysis code
    MUST NOT branch on a concrete Target type, MUST NOT call a complete Target
    analyzer, and MUST NOT resolve an undeclared Target to a default.
  - A family MUST read a dependency's record rather than recompute what it
    states. A number with two derivations has two answers.
  - Before any member of a requested union closure writes Metadata, Analyze MUST
    establish every requested root's family-specific readiness. Each family
    states its own through the checker its descriptor carries, and one
    metadata-free traversal of the derived program answers all of them.
    Performance readiness requires a positive `ParallelCapacityFacts` value for
    the selected topology, rates stated for that same level, and one valid
    execution placement for every occurrence that will take time. Where the buffers
    go is not a readiness question: it is decided with the schedule.
    Failing performance readiness MUST NOT make the same unplaced program invalid
    for `compute-cost`, `memory`, or `roofline`.
  - Global logical work, per-unit work, and lifetime order MUST remain
    target-independent. Physical capacity, hierarchy relationships, and
    throughput comparisons are target-aware.
  - A rendering MUST NOT be a field of the semantic result, and an analysis MUST
    NOT format one.
  - A rendering MUST report what the caller requested. Dependency records nobody
    requested MUST stay on the IR and MUST NOT be reported except for roofline's
    bounded evidence defined below. Record ownership MUST come from the
    Target-selected descriptor ([§3.1](#31-target-selected-analyzers)).
  - Every rendering of one run MUST select records through one shared decision
    and MUST show only records actually written.
  - Every reported quantity MUST come from a record, except a total that is the
    exact sum of records that state it. A quantity not derivable that way MUST be
    recorded by the analysis that computed it.
  - Text and JSON MUST be built from one intermediate report and MUST carry the
    same conclusions.
  - A family's JSON projection MUST be its record's fields under their own names,
    with nothing left out: a default, a `null`, and an empty mapping are each a
    fact a program branches on, and a key spelled by hand is a key that can drift
    from the field it reports. A field whose projection needs the expression the
    record is attached to MUST be declared as one, and MAY be absent where the
    program offers no such reading -- `operands` on a Function Call, which charges
    a callee total no operand position names. A comment over the same record MAY
    state fewer keys, or projected ones
    ([inspection §2.8](./inspection.md#28-record-comment-forms)), and what it
    leaves out MUST stay in the JSON projection.
  - A compact text summary MUST contain whole-function facts only. Per-value
    facts MUST stay on their annotated equations; JSON MAY retain operand names
    and types in its structured projection.

#### 2.2.1 `compute-cost`

`compute-cost` measures the logical work of each authored `Call` without reading
target hardware facts. What an occurrence moves is the memory family's answer
([§2.2.2](#222-memory)), read off the same registered evaluator.

```python
class ComputeCostMetadata(IRMetadata):
    """One Call's logical work, as the authored program states it.

    Attributes:
        flops: attribute; Flop count per compute DType name, sorted by name.
        flops_per_unit: attribute; Flop count performed by one unit of the analysed topology level.
        service: attribute; Result count per service kind, sorted by kind.
        service_per_unit: attribute; Result count per service kind for one unit of the analysed topology level.
    """

    flops: tuple[tuple[str, int], ...] = ()
    flops_per_unit: tuple[tuple[str, int], ...] = ()
    service: tuple[tuple[str, int], ...] = ()
    service_per_unit: tuple[tuple[str, int], ...] = ()
```

| Field | How it is computed | Reads the target |
|---|---|---|
| `flops` | For a primitive Call, run its registered cost evaluator over operand and result Types as written, then multiply by the enclosing recomputation factor. For a Function Call, take the callee's summed `flops` and multiply by the call site's factor. | No |
| `service` | For a primitive Call, take its cost evaluator's service counts -- the results it asks a machine for that are not floating point -- and multiply by the same factor. A Function Call takes the callee's summed `service`. | No |
| `service_per_unit` | The same evaluator over the same projected Types, multiplied by the same factor. A Function Call takes the equivalently projected callee total. | No; projection reads resolved Mesh and effective Module topology extents. |
| `flops_per_unit` | Use the same evaluator over Types projected through authored `Split`s at or coarser than the analysed level, then multiply by the same factor. A Function Call takes the equivalently projected callee total. | No; projection reads resolved Mesh and effective Module topology extents. |

Requesting this family adds one summary line, prefixed by `# `: the Function's own
record, stated exactly as a Call's is. The whole program's work is not a second
record.

```text
compute-cost flops=<dtype>:<int>@<int>[,...] service=<kind>:<int>@<int>[,...]
```

Every measured Call receives this annotation. Each key pairs the whole quantity
with one unit's share, so the two `*_per_unit` fields are not separate keys.

Each reported Call's JSON projection is under its `compute-cost` key:

```text
{"flops": {<dtype>: <int>},
 "flops_per_unit": {<dtype>: <int>},
 "service": {<kind>: <int>},
 "service_per_unit": {<kind>: <int>}}
```

- constraints:
  - A record MUST be attached to every reachable `Call` and to the Function.
    The Function record MUST include authored-loop repetition and therefore is
    not the direct sum of the one-occurrence Call records.
  - An op with no registered cost evaluator MUST raise `AnalysisError`.
  - Missing program geometry MUST NOT be replaced with a target capacity.
  - The enclosing recomputation factor MUST be the product of the authored loop
    trip counts for loops whose induction variable or carried argument the Call
    transitively reads. A loop-invariant Call MUST keep a factor of one. The
    same rule MUST apply to primitive and Function Calls.
  - Downstream families MUST read the already-scaled record and MUST NOT apply
    authored loop trip counts a second time.

#### 2.2.2 `memory`

`memory` measures whole-Function value lifetimes and footprints, decides where
each value's bytes live, and states what every occurrence moves and at which
level. The movement is read off the Op's own registered evaluator and the
amounts its access relations reach.

```python
class TrafficMetadata(IRMetadata):
    """What one Call moves, or what one Function moves over all its trips.

    Attributes:
        whole: attribute; TrafficBytes per storage level name.
        per_unit: attribute; TrafficBytes per storage level name for one unit of the analysed topology level.
        operands: attribute; TrafficBytes per operand, positional against (*call.args, call); present only for a direct primitive call.
    """

    whole: tuple[tuple[str, TrafficBytes], ...] = ()
    per_unit: tuple[tuple[str, TrafficBytes], ...] = ()
    operands: tuple[TrafficBytes, ...] = ()


class LevelFootprint:
    """How much of one memory level a function needs at its peak.

    Attributes:
        level: attribute; The memory level name.
        peak_bytes: attribute; The largest simultaneous claim on the level.
        persistent_bytes: attribute; The part that cannot be reclaimed.
        capacity_bytes: attribute; The stated capacity, or None when unknown.
    """

    level: str
    peak_bytes: int
    persistent_bytes: int
    capacity_bytes: int | None = None

class BufferFootprint:
    """Bytes one authored loop touches in one buffer at one storage level.

    Attributes:
        buffer: attribute; The stable value name of the source buffer.
        level: attribute; The storage level containing that buffer.
        bytes: attribute; Deduplicated bytes reached by one logical position.
        device_bytes: attribute; Deduplicated bytes in the union across positions.
        repeated_bytes: attribute; Per-position bytes without deduplicating repeated access.
    """

    buffer: str
    level: str
    bytes: int
    device_bytes: int
    repeated_bytes: int

class LoopFootprintMetadata(IRMetadata):
    """Known buffer accesses or a lower bound within one authored GridRegionExpr.

    Attributes:
        footprints: attribute; One row per source buffer and storage level.
        known: attribute; Whether every access had a representable relation.
    """

    footprints: tuple[BufferFootprint, ...]
    known: bool

class ValueLifetime:
    """One value's residency, as positions in the function's value order.

    Attributes:
        binding: attribute; The parameter or authored binding name, unique in the function.
        level: attribute; The memory level the value occupies.
        bytes: attribute; Bytes the value occupies at that level.
        defined_at: attribute; Position the value becomes resident.
        last_used_at: attribute; Position it may be released.
        persistent: attribute; Whether it is held for the whole function.
    """

    binding: str
    level: str
    bytes: int
    defined_at: int
    last_used_at: int
    persistent: bool = False

class AllocationMetadata:
    """What showing this function's buffers fit came to.

    Attributes:
        solver_status: attribute; `"optimal"` or `"feasible"`.
    """

    solver_status: str

class MemoryMetadata(IRMetadata):
    """One function's memory behaviour against one target's hierarchy.

    Attributes:
        footprint: attribute; One row per level the function places values in.
        lifetimes: attribute; One entry per value residency.
        advisories: attribute; Capacity findings that do not invalidate the program.
        allocation: attribute; What showing the addressable buffers fit came to.
    """

    footprint: tuple[LevelFootprint, ...] = ()
    lifetimes: tuple[ValueLifetime, ...] = ()
    advisories: tuple[str, ...] = ()
    allocation: AllocationMetadata | None = None
```

Every traffic amount here is what a boundary's own relation reaches. The Op's
evaluator says which way each boundary moves and whether it materialises
anything; it does not say how much, and an Op with no relation fails closed.

- constraints:
  - One relation MUST answer for the whole program and for one unit, from one
    registration; every boundary MUST be held to the iterations its participant
    performs. Projecting an operand's Type is not enough, because a value nobody
    sharded projects to the whole of itself: that is what makes a broadcast
    operand cost its own size and a `Reshard` the distinct coordinates it
    reaches rather than a full source per participant.
  - Each leaf's bytes are charged at the level that leaf sits at. A `UMAT` leaf
    in `Call.args` charges its own bytes at `rmem` and one appearing only in an
    Op attribute charges nothing, so `whole` and an `operands` entry MUST NOT be
    assumed equal for a Type whose leaves occupy several levels: the first groups
    leaves by level, the second is one boundary's amount over all of them. Where
    those bytes were placed enters neither.
  - Two operands MAY name the same value; the `operands` split MUST keep their
    positions distinct, and MUST omit an entry it cannot state rather than emit
    it empty.

Capacity is settled against the authored definition order, which fixes every
buffer's lifetime before any of them is measured, so the only open question is
whether the ones live at once fit together. An arrangement answers that question
without being reported: no address or per-value buffer identity is a conclusion
of this analysis.

- constraints:
  - Capacity MUST be settled for the addressable levels `gmem` and `smem` only,
    once per capacity domain that holds a buffer -- the whole target for a level
    owned target-wide, one per owning position otherwise -- with two buffers in
    one domain never live in the same bytes at once. Residency at another level
    MUST NOT make a program unplaceable, and a level owned per unit of a topology
    level other than the one being analysed MUST fail rather than be assumed.
    Domains holding the same buffers are one question, decided once.
  - `allocation` MUST be absent only when no level could be projected against. A
    function with nothing addressable MUST record a settled `allocation`: the
    question was asked and there was nothing to decide. An attached
    `solver_status` MUST be `"optimal"` or `"feasible"`; a domain that cannot
    fit, cannot be expressed, or does not settle in time MUST raise
    `AnalysisError` saying which of the three happened and leave no record. A
    domain that fits at once MUST be settled without searching, and one whose
    simultaneously live bytes exceed the capacity MUST be refused without
    searching.
  - A Call's `whole`, `per_unit` and `operands` MUST state one occurrence. Only
    the Function record counts an occurrence as often as its authored loops
    repeat it, and its `operands` MUST be empty: which operand moved what
    belongs to the occurrence, not to the total.
  - A capacity conclusion MUST NOT correct or invent a movement number. What an
    occurrence moves is counted once from its own boundaries, so a function with
    no `allocation` still carries traffic -- a different question from whether a
    time may be reported for it ([§2.2.4](#224-performance)) -- and a window
    whose start arrives at run time reads that start rather than becoming a full
    read of its source and a write of its result.

| Field | How it is computed | Reads the target |
|---|---|---|
| `BufferFootprint.buffer` | Use the source parameter or authored binding name, with the same definition-order suffixing as value lifetimes. | No |
| `BufferFootprint.level` | Read the source buffer's declared storage level. | No |
| `BufferFootprint.bytes` | Build relations from rank-preserving per-position Types, union the loop-prefixed access images, count the union's integer points, multiply by the dtype bit width, then round the whole buffer reading up to bytes. If the count is not an integer or exceeds `repeated_bytes`, that buffer reading is unavailable. | No |
| `BufferFootprint.device_bytes` | Repeat the same exact union measurement from authored Types without shard narrowing, giving the union across logical positions in bytes. | No |
| `BufferFootprint.repeated_bytes` | Multiply each operand's per-position element count by its enclosing trip counts, sum accesses to the same buffer, multiply by dtype bit width, then round the whole buffer reading up to bytes. | No |
| `LoopFootprintMetadata.footprints` | One `BufferFootprint` per known source buffer and storage level, sorted by buffer then level. When `known` is false these rows are the available lower bound rather than an empty replacement. | No |
| `LoopFootprintMetadata.known` | False when an access in the loop or a descendant loop lacks a representable forward relation, marking `footprints` as a lower bound; true otherwise. | No |
| `ValueLifetime.binding` | Use the parameter or authored binding name, or `<value N>` when unnamed; repeated names take the printer's numeric suffix in definition order. | No |
| `ValueLifetime.level` | Emit one lifetime per storage level occupied by the value's Type. | No |
| `ValueLifetime.bytes` | Project the Type through every authored split at or coarser than the explicit level's `owner`, then take its logical bytes; a target-owned or undeclared level remains global. | `MemoryHierarchyFacts.explicit_levels[].owner` |
| `ValueLifetime.defined_at` | Position in the order of parameters followed by body Calls and Constants in SSA postorder. | No |
| `ValueLifetime.last_used_at` | Greatest recorded consumer position; the last position for a parameter, and also for the Function body when that body is itself a recorded value. | No |
| `ValueLifetime.persistent` | True for parameters and false for body allocations. | No |
| `LevelFootprint.level` | Each storage level with at least one lifetime, sorted by name. | No |
| `LevelFootprint.peak_bytes` | Largest sum of simultaneously live bytes at that level over the value order. | No |
| `LevelFootprint.persistent_bytes` | Sum of persistent lifetimes at that level. | No |
| `LevelFootprint.capacity_bytes` | Capacity of the matching explicit level, or `None` when it is unknown or undeclared. | `MemoryHierarchyFacts.explicit_levels[].capacity_bytes` |
| `MemoryMetadata.footprint` | One `LevelFootprint` per occupied storage level. | As above |
| `MemoryMetadata.lifetimes` | Every value residency except a `Reshape` or a `Transpose`, each of which describes bytes its operand already holds. | As above |
| `MemoryMetadata.advisories` | Explicit peak overflow, cache/shared-capacity division, and same-scope authored-loop access-footprint findings. | `MemoryHierarchyFacts` |
| `TrafficMetadata.whole` | One occurrence's per-boundary movement asked of the Op's access relations in the whole program's window, charged to the storage levels its operand Types name and grouped by level. On a Function, summed over every reachable occurrence, each counted as often as its authored loops repeat it. A Type with leaves at several levels keeps those leaf bytes separate. A `UMAT` leaf has no residency of its own: when it appears in `Call.args`, charge its own bytes at the target's established `rmem` materialization level; when it appears only in an Op attribute, charge nothing. A Function Call takes the callee's grouped total. | No |
| `TrafficMetadata.per_unit` | The same one occurrence, asked of the same relations in the analysed level's window, charged at the levels the operand's projected Type names. On a Function, summed over occurrences with the same repetition. A Function Call takes the equivalently projected callee total. | No; projection reads resolved Mesh and effective Module topology extents. |
| `TrafficMetadata.operands` | One occurrence's per-boundary movement in order `(*call.args, call)`, the same relation-derived amounts `whole` groups. Empty on a Function and on a Function Call, neither of which has a split. | No |

The target-aware loop projection is report data rather than another metadata
record. `LoopFootprintMetadata` remains target-independent:

```text
"cache-pressure": [{"cache_level": <level>, "backing_level": <level>,
                    "device_bytes": <int>, "capacity_bytes": <int|null>,
                    "status": "fits"|"exceeds"|"lower-bound"|"unknown"}, ...]
```

The projection MUST use `device_bytes`, sum rows at the cache's ultimate explicit
backing level, and compare only levels whose capacity scopes agree. A missing
backing level or a scope mismatch MUST emit no row and MUST NOT fail analysis.
`lower-bound` means an incomplete footprint has not yet exceeded capacity;
`exceeds` remains conclusive when the lower bound alone exceeds it. A cache with
no usable capacity emits `unknown`. A buffer with `device_bytes < bytes` MUST be
removed before projection and its `LoopFootprintMetadata` MUST be marked
incomplete.

The family reads this target projection:

```python
class MemoryRelationKind(Enum):
    """How two memory levels are related."""

    CACHES = "caches"
    SHARES_CAPACITY_WITH = "shares_capacity_with"

class ExplicitMemoryLevelFacts:
    """A level a program places values in by name.

    Attributes:
        name: attribute; The storage level name.
        capacity_bytes: attribute; Stated capacity, or None when unknown.
        scope: attribute; The topology level the capacity is stated per.
        owner: attribute; The topology whose units own separate values, or target.
    """

    name: str
    capacity_bytes: int | None
    scope: str
    owner: str

class ImplicitMemoryLevelFacts:
    """A level traffic passes through without being placed there.

    Attributes:
        name: attribute; The cache level name.
        capacity_bytes: attribute; Stated capacity, or None when unknown.
        scope: attribute; The topology level the capacity is stated per.
    """

    name: str
    capacity_bytes: int | None
    scope: str

class MemoryLevelRelation:
    """One edge between two memory levels.

    Attributes:
        kind: attribute; Which relationship this edge states.
        near: attribute; The level closer to the compute units.
        far: attribute; The level on the other side of the edge.
        shared_capacity_bytes: attribute; Size of the divided block, on a sharing edge.
    """

    kind: MemoryRelationKind
    near: str
    far: str
    shared_capacity_bytes: int | None = None

class MemoryHierarchyFacts:
    """Every memory level of one target, as a flat graph.

    Attributes:
        explicit_levels: attribute; The levels a program names.
        implicit_levels: attribute; The levels traffic only passes through.
        relations: attribute; The edges between them.
    """

    explicit_levels: tuple[ExplicitMemoryLevelFacts, ...]
    implicit_levels: tuple[ImplicitMemoryLevelFacts, ...]
    relations: tuple[MemoryLevelRelation, ...]
```

Requesting memory adds the Function's own movement, one footprint line, and one
line per advisory:

```text
traffic traffic=<level>:r<int>/w<int>@r<int>/w<int>[,...]
peak-footprint=<level>:<int>[,<level>:<int>...]
advisory="<text>"
```

An empty footprint states the family name alone; each advisory is its own line
and is quoted and escaped
([inspection §2.8](./inspection.md#28-record-comment-forms)). The record's own
comment form projects the footprint it holds, and `lifetimes` is read from JSON:

```text
memory peak=<level>:<int>[,...] persistent=<int> advisories=<int>
```

Every measured Call also receives a `traffic` annotation, whose `operands` split
is emitted only when asked for ([cli Analyze](./cli.md#analyze)) and is absent
from a Function, which has no split:

```text
traffic traffic=<level>:r<int>/w<int>@r<int>/w<int>[,...][ operands=<position>:r<int>/w<int>[,...]]
```

Its JSON projection is under the reported value's `traffic` key, with `whole`,
`per_unit` and one `operands` entry per position carrying `read` and `write`.
The `analyze` equation printer emits no memory annotation because that record is
attached only to the Function. Its full JSON projection is under
`function_records.memory`:

```text
{"footprint": [{"level": <level>, "peak_bytes": <int>,
                "persistent_bytes": <int>, "capacity_bytes": <int|null>}, ...],
 "traffic": {<level>: {"read": <int>, "write": <int>}},
 "lifetimes": [{"binding": <name>, "level": <level>, "bytes": <int>,
                "defined_at": <int>, "last_used_at": <int>,
                "persistent": <bool>}, ...],
 "advisories": [<text>, ...]}
```

- constraints:
  - `MemoryMetadata` MUST be attached per reachable `Function`; a peak spans its
    live ranges and belongs to no single expression.
  - `Reshape` and `Transpose` describe bytes their operand already holds and
    MUST NOT receive independent lifetimes. Every other result, a window and a
    field of a tuple and one that overwrites a destination included, MUST
    allocate its own: landing in an operand's buffer is a fact about a plan,
    and no plan has been made here. Analysis uses operation semantics for this
    distinction rather than inferring aliasing from layouts.
  - A caller-owned parameter MUST NOT be reused. Donation is a contract with
    the caller, not a conclusion this family may draw.
  - Which boundaries move is the Op's evaluator's answer and MUST NOT be read
    off the lifetimes above: a boundary it reports no direction on moves
    nothing, which is what a `Reshape` and a `Slice` say of their tensor source
    and their result, while a `Transpose` shares its operand's bytes and still
    reads and writes them because its evaluator materialises the permutation.
    The numbers that place a window MUST be read like any other operand: one
    element per number, reached through the boundary's own relation onto the
    flat leaves the operand holds, and charged at each reached leaf's own width.
    An operation that writes at an address it is given reads that address the
    same way.
  - The memory levels MUST be two flat tuples with a separate relation edge list.
  - A GPU projection MUST cover the explicit levels a program can name and the
    caches traffic passes through, and MUST state that L1 caches L2, that L2
    caches global memory, and that L1 divides one physical block with shared
    memory. A target with no sharing MUST express that with no sharing edge.
  - An implicit level MUST NOT receive a fixed capacity where its usable capacity
    depends on the program; that capacity MUST be derived from the sharing edge
    and the sharing level's measured peak.
  - Every explicit level MUST carry an `owner` supplied by the Target. It MUST be
    a declared Target topology or `target`. An implicit cache MUST NOT carry an
    owner.
  - Analysis MUST NOT infer memory ownership from a storage level's name or
    capacity scope.
  - One value exceeding an explicit level's capacity MUST raise `AnalysisError`.
    An aggregate explicit peak or an authored-loop access footprint exceeding an
    implicit cache capacity MUST instead produce an advisory and MUST NOT fail
    the call.

#### 2.2.3 `roofline`

`roofline` converts recorded work into a lower time bound at the target's
published compute and memory rates.

```python
class RooflineMetadata(IRMetadata):
    """A lower bound on time, and which side of the machine sets it.

    Attributes:
        compute_ns: attribute; Time the flops imply at the target's rates.
        memory_ns: attribute; Time the traffic imply at the target's bandwidth.
        ideal_ns: attribute; The ideal bound the two imply.
        bound_by: attribute; Which resource set the bound.
    """

    compute_ns: int = 0
    memory_ns: int = 0
    ideal_ns: int = 0
    bound_by: str = "none"
```

| Field | How it is computed | Reads the target |
|---|---|---|
| `compute_ns` | For each recorded dtype with a published rate, round `flops * 1e9 / rate` up to ns and sum the dtype times. A Function uses its summed flops, not a sum of per-Call times. | `ThroughputFacts.peak_flops_per_second` |
| `memory_ns` | Add reads and writes at `bandwidth_level`, multiply by `1e9 / memory_bandwidth_bytes_per_second`, and round up to ns; zero when no bandwidth is published or no bytes move. A Function uses its summed traffic, not a sum of per-Call times. | `ThroughputFacts.bandwidth_level` and `memory_bandwidth_bytes_per_second` |
| `ideal_ns` | Maximum of `compute_ns` and `memory_ns`; one ns when the occurrence records nonzero flops or nonzero `bandwidth_level` traffic and neither published rate yields a bound, otherwise zero. Traffic at any other level is stated and does not earn a bound: no rate was published for it, so none is owed. | Through the two times |
| `bound_by` | `none` for no bound, which includes an occurrence whose only movement is at a level with no published bandwidth, `balanced` for equal nonzero times, `memory` when memory is greater, `compute` when compute is greater, and `unrated` for the one-ns bound owed by work this prices whose rate is missing. | Through the two times |

The family reads this target projection:

```python
class ThroughputFacts:
    """Carry the whole-device rates a bound divides the whole program's work by.

    Attributes:
        peak_flops_per_second: attribute; Published compute rates by dtype.
        memory_bandwidth_bytes_per_second: attribute; Published memory bandwidth.
        bandwidth_level: attribute; Memory level whose traffic the bandwidth measures.
    """

    peak_flops_per_second: tuple[tuple[DType, int], ...]
    memory_bandwidth_bytes_per_second: int | None
    bandwidth_level: str

    def peak_for(self, dtype: DType) -> int | None: ...
```

What one unit gets through is a separate projection, read by `performance`
rather than by `roofline`: work that is not floating point at all has its own
published instruction throughput and no dtype to be filed under.

```python
class PerformanceServiceFacts:
    """Carry everything one unit gets through, by the kind of work it is asked for.

    Attributes:
        unit_flops: attribute; One unit's floating-point rate, by dtype.
        unit_ops: attribute; One unit's rate for each named service kind.
        unit_bandwidth: attribute; One unit's rate for each memory level it moves at.
        unit: attribute; Topology level these throughputs describe.
    """

    unit_flops: tuple[tuple[DType, int], ...]
    unit_ops: tuple[tuple[str, int], ...]
    unit_bandwidth: tuple[tuple[str, int], ...]
    unit: str

    def flops(self, dtype: DType) -> int | None: ...
    def ops(self, kind: str) -> int | None: ...
    def bandwidth(self, level: str) -> int | None: ...
```

The service kinds a target states are `integer`, `predicate`, `select` and
`special`. The names are this project's, not any vendor's, so each MUST say in
its provenance which published row it was derived from. A service is work the
machine does, not movement it makes: bytes are priced by a bandwidth, never by
standing an instruction rate in for one.

Requesting roofline adds this verdict and the two quantities the bound divides
-- the summed `flops` and the bandwidth-level bytes -- as `totals`. Nothing else
its dependencies wrote is promoted; asking for `memory` is what states those
([§2.2.2](#222-memory)).

```text
roofline ideal-ns=<int> bound-by=<resource>
```

Every measured Call receives this annotation. `compute_ns` and `memory_ns` are
the two numbers the verdict was read off, so they are in JSON rather than on the
line:

```text
roofline ideal-ns=<int> bound-by=<resource>
```

Reported Call and Function records use the same projection under their
`roofline` keys:

```text
{"compute_ns": <int>, "memory_ns": <int>, "ideal_ns": <int>,
 "bound_by": <resource>}
```

When roofline is requested without its dependencies being requested, `totals`
carries only exact `flops` and `traffic` sums, and `function_records.memory`
carries only `level` and `peak_bytes` per footprint row. Persistent bytes,
capacities, advisories, lifetimes, operand splits, and dependency annotations do
not enter that view. Independently requesting a dependency selects its full form
as defined in that family's section.

- constraints:
  - `ThroughputFacts.peak_for` MUST return `None` for a dtype with no published
    rate; analysis MUST NOT substitute an assumed rate.
  - `PerformanceServiceFacts.flops`, `ops` and `bandwidth` MUST return `None`
    for an unstated dtype, kind or level. Performance MUST reject non-zero work
    of that dtype, kind or level and MUST NOT substitute the whole-device rate
    or another kind's rate.
  - `bandwidth_level` MUST select the traffic level divided by the published
    bandwidth rather than summing traffic across levels.
  - Performance local duration MUST divide `ComputeCostMetadata.flops_per_unit`
    by `unit_flops`, `service_per_unit` by `unit_ops`, and the `bandwidth_level`
    entry of `TrafficMetadata.per_unit` by `unit_bandwidth`. Compute and
    movement overlap within one occurrence, so its duration is the greater of
    the two sides rather than their sum.
  - Traffic at a level with no stated one-unit bandwidth MUST remain visible in
    `TrafficMetadata` and MUST NOT enter a duration: an instruction throughput
    standing in for a bandwidth prices a move as though it were arithmetic.
  - Having moved bytes and having work this can time are different questions.
    What decides the second is the quantities a rate exists for: nonzero
    `flops_per_unit`, nonzero `service_per_unit`, or nonzero
    `TrafficMetadata.per_unit` at `bandwidth_level`. An occurrence with none of
    them MUST take zero time, MUST NOT be required to carry an execution
    placement, and MUST still record its movement at any other level: it is
    untimed, not absent. Work of a dtype or kind the target states no one-unit
    throughput for MUST refuse rather than price at zero, because that would
    leave a hole inside a number the reader takes as whole.
  - A predicate MUST NOT be recorded as floating-point work. A comparison
    records `predicate` service and a selection records `select`; neither has a
    FLOP count, and neither MAY be priced at zero for want of one.
  - `flops` MUST stay the measure of arithmetic that really is a multiply or an
    add. An operation the machine answers on a separate unit at a separate
    published rate belongs in `service`, and MUST NOT also appear in `flops`:
    one operation is one quantity, and a roofline that counts a special-function
    result as one FLOP states a bound the unit cannot meet.
  - Rate-to-duration divisions MUST use exact integer ceiling division. They
    MUST NOT pass through floating-point arithmetic.
  - Roofline records MUST be attached to every reachable `Call` and `Function`.
  - Roofline MUST read `ComputeCostMetadata` rather than evaluate the program a
    second time.
  - A recorded compute dtype that is not a `DType` name MUST raise
    `AnalysisError`.
  - `ThroughputFacts.peak_for` MUST return `None` for an unpublished dtype rate,
    and analysis MUST NOT substitute an assumed rate.
  - Roofline reads only `ComputeCostMetadata`, so a roofline-only rendering
    reports only what roofline and that dependency wrote. No other family's
    conclusion is promoted into it.

#### 2.2.4 `performance`

`performance` places compute-cost-priced occurrences on a CTA-local nominal
timeline, holds the buffers they keep live to the levels this model addresses,
and scales the root schedule by a fixed physical parallel capacity. The records
it owns are named for the prediction they carry rather than for the selector, so
that the interval stays one nested value with one meaning wherever it appears.

```python
class TimelineMetadata:
    """One interval on the nominal timeline.

    Attributes:
        start_ns: attribute; Modeled start, in ns.
        end_ns: attribute; Modeled end, in ns.
        trips: attribute; Number of executions represented by this interval.
        stride_ns: attribute; Start-to-start distance between repeated executions.
    """

    start_ns: int = 0
    end_ns: int = 0
    trips: int = 1
    stride_ns: int = 0


class PerformanceMetadata(IRMetadata):
    """One occurrence's interval within one local wave of its Function.

    Attributes:
        timeline: attribute; That occurrence's CTA-local interval.
    """

    timeline: TimelineMetadata


class PerformanceSummaryMetadata(IRMetadata):
    """One Function's predicted time, and what reaching it took.

    Attributes:
        timeline: attribute; Whole-Function envelope from zero, in ns.
        waves: attribute; Physical waves required by the root topology.
    """

    timeline: TimelineMetadata
    waves: int
```

`TimelineMetadata` is a value, not a record: it MUST NOT be attached to a Call or
a Function on its own, and what it spans is stated by the record carrying it.

Occurrence fields are:

| Field | How it is computed | Reads the target |
|---|---|---|
| `start_ns` | Start of one occurrence in the authored-order local schedule, after its producers end and after the last occurrence sharing any of its participants. | No |
| `end_ns` | End of that occurrence's first execution. | No |
| `trips` | One outside a loop; within a loop, the enclosing loop trip count represented by the interval. | No |
| `stride_ns` | Zero outside a loop; within a loop, the makespan of one body execution. | No |

Function summary fields are:

| Field | How it is computed | Reads the target |
|---|---|---|
| `timeline` | `[0, local makespan * waves)`, where the local makespan is the end of the CTA-local schedule, or zero with no work. Its duration is the prediction. | Through `waves` |
| `waves` | `ceil(N / P)`, where `N` is the static extent of the root topology selected by `ParallelCapacityFacts.topology` and `P` is `parallel_units`. | `ParallelCapacityFacts` |

Occurrence intervals remain CTA-local. They are not copied once per wave, and
neither the root topology extent nor `parallel_units` changes them. The capacity
`P` is compiler policy for concurrent instances; it is distinct from the
per-unit rates in `ThroughputFacts` even when both projections derive from the
same physical unit count today.

The family reads this target projection:

```python
class ParallelCapacityFacts:
    """Carry the parallel capacity assumed by performance analysis.

    Attributes:
        topology: attribute; Topology level being scheduled.
        parallel_units: attribute; Instances admitted concurrently.
    """

    topology: str
    parallel_units: int
```

Requesting performance adds this Function verdict to the summary:

```text
performance root=<Module>::<Function> predicted-ns=<int> waves=<int>
```

`root` is the report's own identity, composed by inspection from the module and
function it already states; it is not a field of `PerformanceSummaryMetadata` and
does not appear in that record's JSON projection. `predicted-ns` is the duration of
the summary's own envelope, not a second measurement. `waves` is stated even when
it is one, because how many passes over the machine a plan takes is a conclusion
and one wave is an answer.

Every Call with a modeled duration receives this annotation, one interval whether
or not it repeats: a single trip states its own bounds, and a repeated occurrence
states them offset by the trip index, with the trip count as a suffix. The trip
count is not a second key, because a reader deriving the later intervals reads it
off the interval it is a coefficient in:

```text
performance=[<int>,<int>)
performance=[<int>t+<int>,<int>t+<int>)*<int>
```

Reported Call and Function records use distinct projections under their
`performance` keys:

```text
Call: {"timeline": {"start_ns": <int>, "end_ns": <int>, "trips": <int>,
                    "stride_ns": <int>}}
Function: {"timeline": {"start_ns": 0, "end_ns": <int>, "trips": 1,
                        "stride_ns": 0},
           "waves": <int>}
```

The summary envelope's duration is a deterministic comparison estimate, not a
runtime prediction. It deliberately excludes launch overhead, occupancy,
utilization, traffic volume, and other execution effects that this family does not
model.

- constraints:
  - A primitive Call is eligible for performance only when it was authored
    inside a `Mesh` naming the selected topology level, recorded as
    `ExecutionDomainMetadata` ([core-ir §2](./core-ir.md#2-expr)). That Mesh is
    where the work ran. The layout its result carries says where that result's
    bytes were put, which is a different question, so a result carrying no
    `ShardLayout` at the selected level MUST NOT unplace the occurrence that
    produced it. Where the result type does carry that level on every tensor
    leaf, the two MUST name the same participant set: a result placed where the
    work that made it never ran is a program this MUST refuse. An occurrence
    with no nonzero
    `flops_per_unit`, no nonzero `service_per_unit` and no nonzero
    `TrafficMetadata.per_unit` at `bandwidth_level` is structural to this
    model: it needs no execution placement and MUST receive no record, because
    an empty
    interval reads as a measurement rather than as the absence of one. Movement
    at another level does not change that and MUST NOT be dropped from
    `TrafficMetadata` because of it -- structural here means nothing to time,
    not nothing done. It still carries its producers' precedence to its
    consumers. Inputs MUST NOT supply placement for an unplaced occurrence.
  - The participant set MUST be the exact image of that Mesh's layout under
    [shard §5](./shard.md#5-mesh), not an extent inferred from a topology or an
    operand. A `Broadcast` shard attribute still names placement: attributes
    describe distribution while the Mesh describes which positions participate.
  - Every primitive occurrence has one fixed duration and occupies its exact
    participant set. An SSA consumer MUST start no earlier than its producers
    end. Two positive-duration occurrences whose participant sets intersect
    MUST NOT overlap; disjoint sets MAY overlap, while a partial intersection
    serializes each whole occurrence rather than splitting it by participant.
  - A `GridRegionExpr` MUST be represented as one structured performance node. Its
    body is solved once, from the time the loop itself begins rather than from
    zero, so a body occurrence's reported `[start_ns, end_ns)` is the interval it
    actually runs in and not one a reader has to offset. `stride_ns` is that
    body's local makespan and the t-th execution of a body occurrence with first
    interval `[start_ns, end_ns)` is
    `[start_ns + t*stride_ns, end_ns + t*stride_ns)`, for `0 <= t < trips`.
    The loop spans `trips * stride_ns`; a consumer of its yield MUST wait for
    that full span. Loop-invariant values remain single occurrences outside it.
  - What an occurrence waits for MUST be read off the program's own structure:
    the values it names, the loop it sits in, and the participants it runs on.
    An ordering MUST NOT be inferred from an allocation -- which values share
    bytes is a plan's decision and no plan has been made -- and no occurrence is
    held back for a write nobody proved happens in place.
  - Occurrences MUST be laid out in inline occurrence order. Reordering
    independent work is a schedule's decision, not an analysis's: what overlaps
    is what the program's own placement made independent, and the reported time
    is the time of the program as written. On a `Function`, the summary's
    `timeline` MUST start at zero and span the whole local plan, scaled by
    `waves`; there MUST be no second field restating the local makespan or the
    scaled estimate, and none restating how the layout was reached -- it is
    exact for the model it states.
  - `parallel_units` is compiler policy over hardware facts. It MUST NOT enter
    one-unit rates or the CTA-local layout, and is not a program rewrite.
  - The buffers a plan keeps live MUST have been placed by `memory` before a
    time is reported for it, which a `MemoryMetadata` carrying an `allocation`
    is the evidence of; one without it MUST fail with `AnalysisError`. A
    placement that failed never reaches here, because `memory` refuses it.
    Capacity therefore changes whether there is an answer, never which answer:
    two capacities that both admit a placement MUST produce the same intervals.
  - Performance is a modeled plan and MUST NOT be read as a guarantee about
    lowering, physical occupancy, or runtime performance.

## 3. Composed analysis

`tilefoundry.analysis.check_program` is the shared, reusable gate before an
analysis or schedule algorithm runs.

```python
def check_program(
    module: "Module",
    function: "Function",
    *,
    level: str | None = None,
    budget: int = _INLINE_NODES,
    analyzers: tuple["Analyzer", ...] = (),
) -> "Function": ...


class AnalysisCheckContext:
    """What every input check reads: the program, the machine, and the level.

    Costing here is a question, not a record: nothing a context computes is
    attached.
    """

    module: "Module"
    function: "Function"
    target: "Target"
    level: str | None
    whole: CostEvaluator
    local: CostEvaluator

```

- constraints:
  - The operation MUST infer types over the full reachable Function graph and
    validate its caller/callee execution context, and MUST NOT run an analysis
    or schedule algorithm or attach derived Metadata to the authored IR.
  - The reachable Function and Mesh geometry and every effective Module
    topology extent MUST be concrete before this operation runs. Public Analyze
    and Schedule calls with `dims` MUST resolve all three through one binding
    pass before calling this gate; a residual dimension expression MUST fail
    before any consuming algorithm runs.
  - Every effective Module topology MUST name a level the resolved Target
    supports. A resolved static extent MUST be positive and within that level's
    finite hardware limit. A rejection MUST name the level, its extent, and the
    reason.
  - A non-`None` `level` MUST name exactly one effective Module topology.
  - Analyze and Schedule MUST call this operation before any consuming
    algorithm. Analyze MUST pass the whole resolved dependency closure as
    `analyzers`, so every analysis about to run states its input contract here.
  - The `analyzers` checkers MUST be bound to the derived Function and run in
    closure order: every `check_target`, then every `check_call` over one
    traversal of the derived non-Function calls, then every `finish`. One
    program MUST be walked once for this however many analyses asked, and the
    first refusal MUST stop the gate before any analysis writes.
  - The returned Function MUST inline every reachable HIR Function call at its
    call site while retaining each `GridRegionExpr` as one loop. Its induction
    variable, carried values, and yields MUST NOT be replaced with iterations.
    The authored Module and Function MUST remain unchanged.
  - The returned Function parameters MUST be the authored entry parameters
    followed by the `ConstTensor` declarations needed by reachable Module
    readings. A promoted declaration MUST be named by the clean dot-joined
    Module path and weight name used by runtime checkpoint keys. Declarations
    MUST follow the Module tree's owner-before-children order, and within one
    Module are unioned by name in Function/parameter order. Separate attachment
    paths MUST remain separate resources. Unequal types for one `(module path,
    weight name)` MUST fail. No constant value enters the IR.
  - Every primitive Call in the returned view MUST have a deterministic unique
    binding.
  - `budget` MUST be a non-negative integer limiting the number of unique body
    expression nodes after inlining. An oversized view MUST fail with both its
    size and the limit and MUST NOT return a partial Function.
  - Authored-analysis readiness is not a program-level check. Analyze MUST
    separately reject schedule constraints and unresolved local layouts before
    running an analyzer; Schedule MAY consume or diagnose those inputs under
    its own algorithm contract.

`tilefoundry.analysis.api.analyze` is the dependency-composed measurement
operation. One call selects one or more root analyses by name; the operation
resolves their union dependency closure, runs each member once, and reports what
ran.
Its subject is one `Module` and one HIR `Function` that Module owns. Reachable
HIR callees are part of that selected invocation and do not become separate
launches because of Module ownership; the invocation rule is owned by
[hir §1.1](./hir.md#11-function). Analyze does not select, interpret, trace, or
dummy-run a plain Python orchestration method.

- constraints:
  - Analyze MUST validate every caller/callee edge the selected query reaches
    against the one-execution-context requirement
    ([hir §1.1](./hir.md#11-function)). Reaching is what is validated, so an
    attached child no call reaches has no edge here. Of the two resolved values
    only the topology hierarchy is compared: the `Target` needs no second check,
    because only a root declares one ([core-ir §1](./core-ir.md#1-module)).
  - The Module owning a reached `Function` MUST be answered within the supplied
    tree, by identity and recorded origin rather than by name. No owner, or more
    than one, is refused rather than assigned to the root.

```python
class AnalysisResult:
    """Record what one composed Analyze call computed.

    Attributes:
        module: attribute; Source Module.
        function: attribute; Function that received records.
        analyses: attribute; Requested root analyses in first-occurrence order.
        level: attribute; Topology level whose unit the per-unit quantities describe, or None.
        executed: attribute; Analyses executed in dependency order.
        metadata_types: attribute; Metadata classes actually written.
    """

    module: "Module"
    function: "Function"
    analyses: tuple[str, ...]
    level: str | None
    executed: tuple[str, ...]
    metadata_types: tuple[type[IRMetadata], ...]


def analyze(
    module: "Module",
    function: "Function",
    *,
    analysis: str | Iterable[str],
    level: str | None = None,
    options: object | None = None,
    dims: "Mapping[str, int] | None" = None,
) -> AnalysisResult: ...
```

- constraints:
  - One call MUST select one or more root analyses. It MUST preserve their
    first-occurrence order, resolve their union dependency closure, and execute
    every member once.
  - `level` MUST name one effective Module topology. When omitted, it MUST
    default to the coarsest effective topology; when the Module declares none,
    it MUST remain `None` and no per-unit projection divides. `AnalysisResult.level`
    MUST record the resolved answer.
  - The Function MUST be one the Module owns: one it declares, or a
    specialization variant of one it declares
    ([core-ir §1](./core-ir.md#1-module)). A Function derived by specialising one
    of these MUST be refused, so that ownership is settled before anything is
    rebuilt.
  - `dims` states one extent per dimension reached through the Function graph,
    its Mesh geometry, or the effective Module topology expressions. An analysis
    counts elements and holds them against a machine, and has no answer for a
    range in any of those positions, so the program MUST be analysed at a chosen
    size rather than as authored.
  - `dims=None` MUST behave as a call that states no size: the Function is
    analysed as authored before the shared program check builds the inlined
    view, and `AnalysisResult.function` MUST be that record-bearing view.
  - When `dims` is stated it MUST be non-empty; every key MUST name a dimension
    reached through the Function graph, its Mesh geometry, or the effective
    Module topology expressions; every value MUST be an integer inside that
    dimension's declared bounds; every dimension the Function selects a variant
    on MUST be given a value; and no dimension MAY remain a range after
    substitution. Each of these MUST fail with an Analysis domain error. A stated
    `dims` MUST NOT be silently ignored, including when the Function declares no
    range at all.
  - Variant resolution and substitution MUST happen after the ownership check and
    before the shared program check or any algorithm runs. Function types, Mesh
    geometry, and effective topology extents MUST use the same resolved binding.
    Exactly one variant MUST cover the stated size; none and more than one MUST
    both fail.
  - When `dims` is stated, `AnalysisResult.function` MUST be the inlined view of
    the concrete Function the records were written onto and MUST retain the
    specialised Function's origin and extents. `AnalysisResult.module` MUST
    remain the Module the caller supplied. A reader given the symbolic input
    would find no records on it.
  - The recorded extents MUST be what identifies which size a derived Function is
    at. They MUST NOT be inferred from its signature: a dimension occurring only
    in a loop bound, a body operation's attribute, or a nested callee leaves the
    signature identical at every extent, so two sizes would be indistinguishable
    to anything comparing signatures.
  - The operation MUST resolve the root's full transitive dependency closure,
    order it so every dependency precedes its dependants, and execute each
    member exactly once per call. `executed` MUST report that order, so a shared
    dependency appears once.
  - Dependencies MUST be resolved under the same exact concrete Target as the
    root, obtained from `Module.resolve_target()`.
  - A dependency cycle MUST fail and MUST name the path that closes it. A
    missing root and a missing dependency MUST be distinguishable: one is the
    caller's selector, the other a broken Target capability.
  - Type inference and validation MUST each run once per call, before any
    analysis. No analysis MAY run once either has rejected the IR, because an
    analysis reads inferred types and assumes a verified function.
  - Family-specific readiness MUST be checked on that inferred inlined view and
    MUST complete before the first analysis in the dependency closure runs. In
    particular, a performance request that lacks an execution placement MUST
    fail before dependency Metadata is written.
  - Re-running MUST recompute the closure and refresh the Metadata that closure
    owns. There MUST be no cross-call cache. Metadata owned by nothing in the
    closure MUST be left untouched.
  - `metadata_types` MUST list the Metadata types the call actually wrote onto
    the IR, in execution order and without repeats. An analysis that declares a
    type but writes no record for this function MUST NOT contribute it, so a
    renderer is never sent after records that are not there.
  - `AnalysisResult` MUST be semantic. Human text, JSON, and annotated HIR are
    renderings of it and of the Metadata on the IR, and MUST NOT be fields of
    it.

### 3.1 Target-selected Analyzers

```python
AnalysisCallable = Callable[
    [Module, Function, Target, str | None, object | None], None
]

class Analyzer:
    """Describe one Target-selected analysis.

    Attributes:
        selector: attribute; Public analysis selector.
        run: attribute; Analysis implementation.
        requires: attribute; Dependency selectors.
        produces: attribute; Owned metadata classes.
        input_checker: attribute; What this analysis needs of a program.
    """

    selector: str
    run: AnalysisCallable
    requires: tuple[str, ...] = ()
    produces: tuple[type[IRMetadata], ...] = ()
    input_checker: AnalysisInputChecker = NO_INPUT_CHECK


class AnalysisInputChecker(Protocol):
    """What one analysis requires before any analysis writes.

    Three questions answerable without reading a record: what the target must
    state, what each call must carry, and what the function must hold.
    """

    def check_target(self, ctx: AnalysisCheckContext) -> None: ...
    def check_call(self, call: Call, ctx: AnalysisCheckContext) -> None: ...
    def finish(self, function: Function, ctx: AnalysisCheckContext) -> None: ...


class Target:
    def get_analyzer(self, selector: str) -> Analyzer: ...
```

- constraints:
  - `AnalysisCallable` MUST receive the Module, Function, exact Target, resolved
    topology level, and caller options in that order. The level MAY be `None`
    only when the Module declares no topology; options MAY be `None`.
  - Analyze MUST obtain every root and dependency from the same exact Target
    instance through `get_analyzer`.
  - A Target subclass MUST inherit its base Analyzers through normal Python
    inheritance. It MAY override one selector and delegate the rest to
    `super()` or refuse inherited behavior that is invalid for its hardware.
  - There MUST be no public analysis registration step or exact-concrete-Target
    algorithm table. A custom provider registers only its Target class.
  - A declaration MUST be rejected when it requires itself, repeats a
    dependency, produces the same Metadata type twice, or names a `produces`
    entry that is not an `IRMetadata` subclass.
  - `input_checker` MUST default to one that requires nothing, so an analysis
    with no input contract is declared by leaving the field out and keeps
    working unchanged. A checker MUST NOT attach Metadata: it states what a
    program must already be, and every checker in a closure MUST have answered
    before any analysis in it writes.
  - An analysis MAY change only the Metadata types its Analyzer declares.
    Ownership MUST be enforced against what reached the IR rather than against
    what the analysis reports, and MUST cover addition, replacement, and
    removal alike: deleting another analysis's record changes the IR as much as
    overwriting it. An equal-valued overwrite of another analysis's record MUST
    also count as a violation.
