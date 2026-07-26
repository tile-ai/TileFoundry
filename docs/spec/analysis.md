# TileFoundry Spec — analysis (polyhedral model + per-stage target facts)

This spec owns TileFoundry's fact layer: everything a later stage decides
*over*, and nothing that decides anything itself. It has three surfaces:

| Surface | Entry | What it states |
|---|---|---|
| Polyhedral model | `extract(hir) -> TileGraph` | one HIR `Function` body as isl domains, access relations and auto-inferred dependences — target-independent |
| Per-stage target facts | `Analysis` (structural interface) | the atom candidates of one op and the tile store of one storage level, for one target at one stage |
| Authored-HIR metrics | `analyze(ir) -> AnalysisResult` | roofline / footprint / timeline records attached to authored HIR as metadata |

Per-Op semantic derivation — typeinfer, the forward access relation, shard
propagation — is owned by [semantic-analysis](./semantic-analysis.md), and the
registries behind it by [visitor-registry](./visitor-registry.md); the
polyhedral model consumes the forward relation
([visitor-registry §4.1](./visitor-registry.md#41-forward-relation-service--type_relation))
rather than restating it.

**Layering.** The decisions taken over these facts are owned by
[schedule](./schedule.md#4-kernel-schedule-construction). The dependency is
one-way: the schedule layer reads Analysis facts, and this layer MUST NOT
import or otherwise depend on the schedule layer.

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

## 2. Per-stage target facts

The polyhedral model is target-independent; the atom catalogue and the store a
tile lives in are not. That store belongs to the **level**, not to the device: a
tile at the AMX `core` level lives in that core's L1d, one at the CUDA `cta`
level in shared memory.

### 2.1 `AtomFact`

```python
class AtomFact:
    """One candidate atom's facts, as the deciding stage consumes them.

    Attributes:
        shape: attribute; The atom's own M, N and K extents.
        dtype: attribute; The atom's own a, b and c operand DTypes.
        duration: attribute; Nominal roofline estimate for one instance, in ns.
        compute_duration: attribute; The compute-side half of that estimate alone, in ns.
        storage: attribute; Per-role fragment occupancy in bytes.
        resource: attribute; Required thread-scope footprint, keyed by scope name.
        is_async: attribute; True when the instruction is asynchronous.
        atom: attribute; The target's own realized atom descriptor, carried through opaquely.
    """

    shape: tuple[int, int, int]
    dtype: tuple[DType, DType, DType]
    duration: float
    compute_duration: float
    storage: dict[str, int]
    resource: dict[str, int]
    is_async: bool
    atom: object
```

- constraints:
  - The structure MUST be immutable and MUST stay target-independent: `atom`
    MUST be kept opaque, so a target package can enumerate its own catalogue
    without this type knowing that catalogue's types.
  - `shape` / `dtype` MUST mirror the atom's own shape and operand dtypes, so a
    consumer can filter and granularise without unpacking `atom`.
  - `duration` MUST be a nominal estimate in ns for **one** atom instance, and
    `compute_duration` MUST be its compute-side half alone — for a consumer that
    models the surrounding traffic itself and would otherwise charge memory
    twice.
  - `atom` MUST be the realized descriptor a later fill or codegen stage needs,
    so that stage never re-resolves it from `shape` / `dtype`.

### 2.2 `Analysis`

`Analysis` is the structural interface a Target binds for one stage, alongside
that stage's `Schedule` ([target §1](./target.md#1-target)).

```python
class Analysis(Protocol):
    """Report one stage's target-dependent facts.

    Attributes:
        stage: attribute; Exact Target service key for this implementation.
        tile_capacity_bytes: attribute; Capacity of the store one tile of this level lives in.
    """

    stage: str
    tile_capacity_bytes: int

    def candidate_atoms(self, op: Call) -> list[AtomFact]: ...
```

- constraints:
  - An implementation MUST be bound under `(Analysis, stage)` on the Target and
    selected by `target.service(Analysis, stage)`; `stage` MUST equal that exact
    key.
  - `tile_capacity_bytes` MUST be the capacity of the store belonging to the
    bound level, not the whole device, and MUST be a positive integer.
  - `candidate_atoms` MUST hard-filter the target's catalogue and MUST NOT rank
    it: ordering is not a decision this interface makes.
  - `candidate_atoms` MAY return an empty list — a legitimate "no candidate
    covers this op" outcome, not an error — and MAY raise `NotImplementedError`
    for an op kind or target its catalogue does not cover.
  - The capacity is a fact to record against a footprint, not a gate: an
    implementation MUST NOT raise because a tile does not fit. A tile wider than
    its store still has a schedule, only a worse one.

## 3. Authored-HIR metrics

`analyze` is the authored-HIR measurement entry the command line exposes
([cli §Analyze](./cli.md#analyze)). It re-derives types over the authored
program, validates it, then attaches one metadata record per measured
expression.

```python
class AnalysisOptions:
    """Select which authored-HIR analyses run.

    Attributes:
        roofline: attribute; Attach RooflineMetadata per measured Call.
        footprint: attribute; Attach FootprintMetadata per measured value.
        timeline: attribute; Attach TimelineMetadata per measured Call.
    """

    roofline: bool = True
    footprint: bool = True
    timeline: bool = True

class AnalysisResult:
    """Carry the annotated IR and its overall summary.

    Attributes:
        ir: attribute; The same IR object that was analyzed, now carrying metadata.
        summary_lines: attribute; Overall summary, one stable line per measured total.
        metadata_types: attribute; The metadata record types this run attached.
    """

    ir: "Module | Function"
    summary_lines: tuple[str, ...]
    metadata_types: tuple[type, ...]

class AnalysisError(ValueError):
    """An authored program the analysis rejects, or a measurement that failed."""

def analyze(ir: "Module | Function", *, options: AnalysisOptions | None = None) -> AnalysisResult: ...
```

- constraints:
  - `AnalysisOptions` and `AnalysisResult` MUST be immutable. `options=None`
    MUST mean a fresh default `AnalysisOptions()` for that call, which selects
    all three analyses.
  - `analyze` MUST re-derive every authored value type before measuring, and
    MUST reject an authored program that carries a schedule constraint
    ([schedule §3](./schedule.md#3-constraint-metadata)) or whose local-storage
    value has an unresolved layout. Both rejections MUST name the source
    location and binding where available.
  - `analyze` MUST attach each record in place on the IR it returns, replacing
    any earlier record of the same type, and `AnalysisResult.ir` MUST be that
    same object.
  - `AnalysisResult.metadata_types` MUST list exactly the record types the
    selected analyses attached, so a printer can comment them without knowing
    the option set.
  - A nested `Function` call MUST measure as its callee's own totals; an
    unresolved or recursive call graph MUST raise `AnalysisError`.
  - `analyze` MUST accept a HIR `Function`, or a `Module` whose entry function
    is one; anything else MUST raise `AnalysisError`.

### 3.1 Metadata records

```python
class TrafficBytes:
    """Read and write byte counts for one memory hierarchy level.

    Attributes:
        read_bytes: attribute; Bytes read at this level.
        write_bytes: attribute; Bytes written at this level.
    """

    read_bytes: int = 0
    write_bytes: int = 0

class RooflineMetadata(IRMetadata):
    """Per-Call flop counts, traffic, and the bound they imply.

    Attributes:
        flops: attribute; Flop count per DType name, sorted by name.
        traffic: attribute; TrafficBytes per storage level name.
        theoretical_ns: attribute; The roofline bound in ns.
    """

    flops: tuple[tuple[str, int], ...] = ()
    traffic: tuple[tuple[str, TrafficBytes], ...] = ()
    theoretical_ns: int = 0

class FootprintMetadata(IRMetadata):
    """Live bytes per storage level at one program point.

    Attributes:
        live_bytes: attribute; Live byte count per storage level name, sorted by name.
    """

    live_bytes: tuple[tuple[str, int], ...] = ()

class TimelineMetadata(IRMetadata):
    """One execution unit's modeled placement on the nominal timeline.

    Attributes:
        grid_ctas: attribute; CTA extent this unit runs at.
        waves: attribute; Number of waves the unit's CTAs are issued in.
        start_ns: attribute; Modeled start of the unit's first wave, in ns.
        end_ns: attribute; Modeled end of the unit's last wave, in ns.
    """

    grid_ctas: int = 1
    waves: int = 1
    start_ns: int = 0
    end_ns: int = 0
```

- constraints:
  - Every record MUST be immutable and MUST render one single-line comment form,
    so a printer can attach it without knowing the record type
    ([inspection](./inspection.md)).
  - A `Call`'s flop and byte counts MUST come from that op's registered cost
    evaluator ([visitor-registry](./visitor-registry.md)) scaled by the call's
    execution count, which is the product of the execution-topology extents its
    value meshes and its function declare. An op with no registered cost
    evaluator MUST raise `AnalysisError`. Conflicting extents for one topology
    name MUST raise rather than be reconciled.
  - `theoretical_ns` MUST be the larger of the compute time implied by `flops`
    over the device's peak throughput per DType and the memory time implied by
    global traffic over the device's memory bandwidth. A target that publishes
    neither fact MUST report `0`.
  - `live_bytes` MUST be measured over the postorder live ranges of the
    function's values, and a pure view (a reshape or a transpose) MUST allocate
    nothing.
  - `TimelineMetadata` MUST be attached per **execution unit** — the group of
    calls fused by compatible local placement and equal CTA extent — so every
    call of one unit MUST carry the same record. A unit whose CTA extent exceeds
    the capacity one launch admits MUST be modelled as consecutive waves, and
    the reported makespan MUST respect both the unit order and that capacity.
  - The timeline is a modeled plan. It MUST NOT be read as a guarantee about
    lowering, physical occupancy, or runtime performance.
