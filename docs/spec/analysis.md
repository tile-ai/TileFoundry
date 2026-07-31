# TileFoundry Spec — analysis (polyhedral model + per-stage target facts)

This spec owns TileFoundry's fact layer: everything a later stage decides
*over*, and nothing that decides anything itself. It has two surfaces:

| Surface | Entry | What it states |
|---|---|---|
| Polyhedral model | `extract(hir) -> TileGraph` | one HIR `Function` body as isl domains, access relations and auto-inferred dependences — target-independent |
| Composed measurement | `analyze(module, function, analysis=...)` | one root analysis and its dependency closure, leaving typed Metadata on the IR |

Per-Op semantic derivation — typeinfer, the forward access relation, shard
propagation — is owned by [semantic-analysis](./semantic-analysis.md), and the
registries behind it by [visitor-registry](./visitor-registry.md); the
polyhedral model consumes the forward relation
([visitor-registry §4.1](./visitor-registry.md#41-forward-relation-service--type_relation))
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
    """Polyhedral model of one HIR Function body, plus its schedule.

    Attributes:
        domain: attribute; Union of every statement's iteration domain, one named tuple per statement.
        deps: attribute; Auto-inferred read-after-write must-dependence between statement instances.
        reads: attribute; Union of every statement's read access relations, statement tuple to buffer tuple.
        writes: attribute; Union of every statement's write access relations.
        units: attribute; One TileUnit per statement, in dependence-respecting order.
        params: attribute; isl parameter name to the ShapeDim it stands for.
        buffer_dtypes: attribute; Buffer tuple name to the DType its elements carry.
        parallel_dims: attribute; Statement name to one flag per own domain dimension, set when that dimension carries no dependence.
        tree: attribute; isl schedule tree over this model, or None before one is built.
        ring: attribute; Buffer name to its decided ring depth; empty until atom selection decides it.
        decisions: attribute; Recorded per-statement decisions, or None until atom selection records them.
    """

    domain: "isl.union_set"
    deps: "isl.union_map"
    reads: "isl.union_map"
    writes: "isl.union_map"
    units: tuple[TileUnit, ...]
    params: dict
    buffer_dtypes: dict = {}
    parallel_dims: dict = {}
    tree: "isl.schedule | None" = None
    ring: dict = {}
    decisions: dict | None = None
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
    `reads` / `writes` by the extraction itself (§1.3), never supplied by a
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
    every statement, and MUST be measured from `domain` + `deps` (§1.7) rather
    than reported by a scheduler.
  - `tree`, `ring` and `decisions` MUST be empty (`None` / `{}`) as returned by
    `extract`, and MUST be filled only by the schedule stages that decide them
    ([schedule §4](./schedule.md#4-kernel-schedule-construction)).
  - `ring` and `decisions` MUST be plain fields. Neither MAY be carried as an
    isl mark payload: isl marks are process-global state and MUST NOT hold a
    Python object.

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
| `Call` of `TupleGetItem` / `Reshape` / `Gather` | structural view — no statement. It resolves to its source's buffer name, and the coordinate change it expresses is folded into every consumer's access map |
| `Call` of `Zeros` / `FullLike` | buffer declaration — no statement and no access relation: it names a fresh buffer and gives it a starting value |
| `Call` whose target is a `Function` | penetrated, not rejected: the callee's params bind to the caller's already-resolved argument expressions, its body is walked in place, and every statement and buffer it contributes is prefixed with the callee name plus a per-call-site index |
| `GridRegionExpr` | not a statement — it contributes one leading domain dimension to every statement it encloses (§1.4) |
| `Tuple` | resolved through the same substitution table; no statement |

- constraints:
  - Each statement's access relations MUST come from the forward
    (input-type-driven) relation service
    ([visitor-registry §4.1](./visitor-registry.md#41-forward-relation-service--type_relation)).
    `extract` MUST stamp the statement and buffer tuple names onto each
    returned map and restrict it to the paired domain, and MUST reuse each
    access map's own formula unchanged — no retiling happens here.
  - Before the relation is built, every argument type MUST be narrowed to its
    per-shard local shape when it carries a `ShardLayout`: each mesh `Split`'s
    target *tensor* axis divided by that mesh axis's extent, tensor rank
    preserved. A `Partial` / `Broadcast` / `Dynamic` mesh axis consumes no
    tensor axis; a launch-provided (deferred) mesh extent narrows its axis to
    `1`. Narrowing is centralized in the extraction, so every registered
    relation is sharding-aware without knowing sharding exists.
  - A `Split`-sharded axis whose extent is not a static integer, or is not
    evenly divisible by its mesh extent, MUST raise `ExtractError`.
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
  - A `Gather` on an enclosing loop's own induction variable MUST fold into the
    consumer's access map as the gathered axis, so iteration `i` addresses slice
    `i`. A batched gather, a non-scalar index, or an index that is not an
    enclosing loop's induction variable MUST raise `ExtractError`: a
    data-dependent gather has no affine access map and MUST NOT be
    approximated.

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

The measurement entry is the composed operation (§3). What a human or a tool
reads is a *rendering* of its semantic result and of the records it left on the
IR ([inspection](./inspection.md)); the command line composes one call per
requested analysis and renders the results together
([cli §Analyze](./cli.md#analyze)).

- constraints:
  - A rendering MUST NOT be a field of the semantic result, and an analysis MUST
    NOT format one. A family that rendered its own text would be deciding
    presentation, and two families would then disagree about it.
  - A rendering MUST report what the caller *requested*. A requested analysis
    pulls its dependencies in, so records reach the IR that nobody asked to see;
    those records MUST stay on the IR and MUST NOT be reported. Which records an
    analysis owns MUST be read from its registration (§3.1) rather than from a
    second table.
  - Every rendering of one run MUST make that selection through one shared
    decision. A summary and an annotated program are two views of the same run,
    and choosing separately is how one of them comes to show a dependency the
    caller never asked about.
  - A rendering MUST show only records that were actually written, so it never
    reports a measurement that did not happen.
  - Every reported quantity MUST come from a record, except a total that is the
    exact sum of records that state it. A quantity that is not derivable that
    way MUST be recorded by the analysis that computed it, not reassembled by a
    renderer.
  - Two formats over one report MUST carry the same conclusions. They MUST be
    built from one intermediate structure rather than formatted independently.

### 2.1 Metadata records

```python
class TrafficBytes:
    """Read and write byte counts for one memory hierarchy level.

    Attributes:
        read: attribute; Bytes read at this level.
        write: attribute; Bytes written at this level.
    """

    read: int = 0
    write: int = 0

class ComputeCostMetadata(IRMetadata):
    """One Call's logical work, as the authored program states it.

    Attributes:
        flops: attribute; Flop count per compute DType name, sorted by name.
        traffic: attribute; TrafficBytes per storage level name.
        execution_count: attribute; How many times the call runs.
        operands: attribute; TrafficBytes per operand, positional against (*call.args, call); present only for a direct primitive call.
    """

    flops: tuple[tuple[str, int], ...] = ()
    traffic: tuple[tuple[str, TrafficBytes], ...] = ()
    execution_count: int = 1
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

class ValueLifetime:
    """One value's residency, as positions in the function's value order.

    Attributes:
        binding: attribute; The parameter or authored binding name.
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

class MemoryMetadata(IRMetadata):
    """One function's memory behaviour against one target's hierarchy.

    Attributes:
        footprint: attribute; One row per level the function places values in.
        traffic: attribute; TrafficBytes per level, over the whole function.
        lifetimes: attribute; One entry per value residency.
        advisories: attribute; Capacity findings that do not invalidate the program.
    """

    footprint: tuple[LevelFootprint, ...] = ()
    traffic: tuple[tuple[str, TrafficBytes], ...] = ()
    lifetimes: tuple[ValueLifetime, ...] = ()
    advisories: tuple[str, ...] = ()

class RooflineMetadata(IRMetadata):
    """A lower bound on time, and which side of the machine sets it.

    Attributes:
        compute_ns: attribute; Time the flops imply at the target's rates.
        memory_ns: attribute; Time the traffic implies at the target's bandwidth.
        theoretical_ns: attribute; The bound the two imply.
        bound_by: attribute; Which resource set the bound.
    """

    compute_ns: int = 0
    memory_ns: int = 0
    theoretical_ns: int = 0
    bound_by: str = "none"

class TimelineMetadata(IRMetadata):
    """A modeled placement on the nominal timeline.

    Attributes:
        grid_units: attribute; Parallel-unit extent this placement covers.
        waves: attribute; Number of waves issued.
        start_ns: attribute; Modeled start, in ns.
        end_ns: attribute; Modeled end, in ns.
    """

    grid_units: int = 1
    waves: int = 1
    start_ns: int = 0
    end_ns: int = 0
```

- constraints:
  - Every record MUST be immutable and MUST render one single-line comment form,
    so a printer can attach it without knowing the record type
    ([inspection](./inspection.md)).
  - A record's attachment point MUST say what it is about, and one record type
    MUST mean the same quantity wherever it hangs. A `Call`-attached record
    describes that call; a `Function`-attached record describes that whole
    function. A type MUST NOT change meaning with its attachment point.
  - A `Function`-attached record MUST NOT be read as data the Function
    inherently carries. It states what one analysis found when a call reached
    that function, and there MUST be no cross-call cache behind it.
  - `ComputeCostMetadata` MUST be derivable from the authored program alone. Its
    flops MUST come from the op's registered cost evaluator
    ([visitor-registry](./visitor-registry.md)) scaled by the execution count,
    and its bytes from the logical types the operands and result carry. It MUST
    NOT read any Target fact, so one authored call carries the same record on
    every backend. An op with no registered cost evaluator MUST raise
    `AnalysisError`.
  - The execution count MUST be the product of the execution-topology extents the
    call's value meshes carry and the owning Module declares. Conflicting extents
    for one topology name MUST raise rather than be reconciled.
  - `operands` MUST be positional against `(*call.args, call)`: one entry per
    argument in order, then the result. Each entry MUST be the amount the op's
    cost evaluator reported for that operand, unmodified. Two operands MAY name
    the same value; the position is what distinguishes them.
  - Which level an operand's bytes are charged at MUST come from that operand's
    Type. Where the Type occupies one level, the per-level `traffic` is those
    same amounts regrouped. Where it spans several, the per-level `traffic` MUST
    charge the whole Type at each of them, in the directions the op reported
    movement in: no operand-level count states how the bytes divide, so the
    aggregate takes a conservative bound while the operand entry keeps the
    reported amount. The two MUST NOT be assumed equal in general.
  - Operand attribution is defined for a call on a primitive op only. A call
    into another `Function` MUST report that function's totals and MUST record
    no operands: that traffic is an aggregate over the callee's own operands,
    and no breakdown of this call's arguments describes it. A rendering MUST
    omit the operand breakdown for such a call rather than emit it empty.
  - `MemoryMetadata` MUST be attached per `Function`: a peak is a property of the
    whole function's live ranges and belongs to no single expression.
  - Parameters MUST be resident from the start of the value order. A parameter
    declared constant is a weight and MUST be `persistent`, held past its last
    reader for the whole function; every other value MUST be measured by first
    definition and last use. A pure view — a reshape or a transpose — MUST
    allocate nothing.
  - `RooflineMetadata` MUST be computed from the recorded work rather than from a
    second reading of the program. On a `Function` the compute and memory times
    MUST each be summed over the function before being compared, because
    aggregating per-Call bounds instead would charge the machine for rounding and
    for overlap it does not suffer.
  - A dtype or a level the target publishes no rate for MUST contribute nothing
    to the bound, and MUST NOT be filled in with an assumed rate. Work whose rate
    is unpublished MUST still report a non-zero bound rather than read as free.
  - `TimelineMetadata` MUST be attached per **execution unit** — the group of
    calls fused by compatible local placement and equal parallel extent — so
    every call of one unit MUST carry the same record. A unit whose extent exceeds
    the parallel capacity MUST be modelled as consecutive waves. On a `Function`
    the record MUST span the whole plan, from the origin to the solved makespan.
  - The timeline is a modeled plan. It MUST NOT be read as a guarantee about
    lowering, physical occupancy, or runtime performance.

### 2.2 Analysis families

The first families are `compute-cost`, `memory`, `roofline`, and `timeline`.
Each owns one record type and declares what it needs.

| Selector | Requires | Owns | Rests on |
|---|---|---|---|
| `compute-cost` | — | `ComputeCostMetadata` | the authored program only |
| `memory` | `compute-cost` | `MemoryMetadata` | `MemoryHierarchyFacts` |
| `roofline` | `memory`, `compute-cost` | `RooflineMetadata` | `ThroughputFacts` |
| `timeline` | `roofline` | `TimelineMetadata` | `ParallelCapacityFacts` |

- constraints:
  - A family MUST obtain hardware only through a Facts aggregate it declares
    ([target §11](./target.md#11-target-facts-projection)). Common analysis code
    MUST NOT branch on a concrete Target type, MUST NOT call a complete Target
    analyzer, and MUST NOT resolve an undeclared Target to a default.
  - A family MUST read a dependency's record rather than recompute what it
    states. A number with two derivations has two answers.
  - Logical work and lifetime MUST remain target-independent. Physical capacity,
    hierarchy relationships, and throughput comparisons are target-aware.

### 2.3 Memory hierarchy facts

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
    """

    name: str
    capacity_bytes: int | None
    scope: str

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

- constraints:
  - The levels MUST be two flat tuples and the structure MUST be a separate edge
    list. A hierarchy that is only ever a tree cannot state that a cache and an
    addressable level divide one physical block, which is the relationship that
    decides how much of that block either one gets.
  - A GPU projection MUST cover the explicit levels a program can name and the
    caches traffic passes through, and MUST state that L1 caches L2, that L2
    caches global memory, and that L1 divides one physical block with shared
    memory. A target with no such sharing MUST express that by having no such
    edge, not by a placeholder one.
  - An implicit level MUST NOT be given a capacity of its own where its usable
    capacity depends on the program. That capacity MUST be derived from the
    sharing edge and the sharing level's measured peak.
  - Exceeding an explicit level's stated capacity MUST raise `AnalysisError`: the
    program placed more there than the level holds. Exceeding an implicit level's
    capacity MUST be recorded as an advisory and MUST NOT fail the call, because
    a working set larger than a cache still runs.

## 3. Composed analysis

`tilefoundry.analysis.api.analyze` is the dependency-composed measurement
operation. One call selects one root analysis by name; the operation resolves
what that root transitively needs, runs each member once, and reports what ran.

```python
class AnalysisResult:
    """What one composed Analyze call computed."""

    module: "Module"
    function: "Function"
    analysis: str
    executed: tuple[str, ...]
    metadata_types: tuple[type[IRMetadata], ...]


def analyze(
    module: "Module",
    function: "Function",
    *,
    analysis: str,
    options: object | None = None,
    dims: "Mapping[str, int] | None" = None,
) -> AnalysisResult: ...
```

- constraints:
  - One call MUST select exactly one root analysis. A caller wanting several
    roots MUST call the operation once per root.
  - The Function MUST be one the Module owns: one it declares, or a
    specialization variant of one it declares
    ([core-ir §1](./core-ir.md#1-module)). A Function derived by specialising one
    of these MUST be refused, so that ownership is settled before anything is
    rebuilt.
  - `dims` states one extent per dimension the Function declares as a range. An
    analysis counts elements and holds them against a machine, and neither has an
    answer for a range, so a Function stating a range MUST be analysed at a
    chosen size rather than as authored.
  - `dims=None` MUST behave as a call that states no size: the Function is
    analysed as authored, and `AnalysisResult.function` MUST be the object the
    caller supplied.
  - When `dims` is stated it MUST be non-empty; every key MUST name a dimension
    the Function declares as a range; every value MUST be an integer inside that
    dimension's declared bounds; every dimension the Function selects a variant
    on MUST be given a value; and no dimension MAY remain a range after
    substitution. Each of these MUST fail with an Analysis domain error. A stated
    `dims` MUST NOT be silently ignored, including when the Function declares no
    range at all.
  - Variant resolution and substitution MUST happen after the ownership check and
    before any algorithm runs. Exactly one variant MUST cover the stated size;
    none and more than one MUST both fail.
  - When `dims` is stated, `AnalysisResult.function` MUST be the concrete Function
    the records were written onto, derived from the Function the caller supplied,
    and MUST record both that Function as the one it was specialised from and the
    extents it was specialised at. `AnalysisResult.module` MUST remain the Module
    the caller supplied. A reader given the symbolic input would find no records
    on it.
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
    caller's selector, the other a broken registration.
  - Type inference and validation MUST each run once per call, before any
    analysis. No analysis MAY run once either has rejected the IR, because an
    analysis reads inferred types and assumes a verified function.
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

### 3.1 Analysis registration

```python
class AnalysisAlgorithm:
    """One registered analysis: its identity, needs, and owned Metadata."""

    selector: str
    run: AnalysisCallable
    requires: tuple[str, ...]
    produces: tuple[type[IRMetadata], ...]


def register_analysis(
    target_type: type,
    selector: str,
    *,
    requires: tuple[str, ...] = (),
    produces: tuple[type[IRMetadata], ...] = (),
) -> "Callable[[AnalysisCallable], AnalysisCallable]": ...
```

- constraints:
  - An analysis MUST be registered under the exact
    `(Target concrete type, selector)` pair, in the shared algorithm registry
    contract ([code-organization](./code-organization.md)). A base-class
    registration MUST NOT serve a subclass, and there MUST be no default-Target
    fallback: two targets sharing a base can need different implementations.
  - A target-independent analysis MUST still be registered once per supported
    target, so the support matrix is read from the registrations rather than
    inferred from an inheritance chain.
  - A duplicate registration for one exact pair MUST fail rather than replace,
    so dispatch cannot depend on import order.
  - A declaration MUST be rejected when it requires itself, repeats a
    dependency, produces the same Metadata type twice, or names a `produces`
    entry that is not an `IRMetadata` subclass.
  - An analysis MAY change only the Metadata types its registration declares.
    Ownership MUST be enforced against what reached the IR rather than against
    what the analysis reports, and MUST cover addition, replacement, and
    removal alike: deleting another analysis's record changes the IR as much as
    overwriting it. An equal-valued overwrite of another analysis's record MUST
    also count as a violation.
