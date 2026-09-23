# TileFoundry Spec — analysis (authored-HIR metrics + per-stage target facts)

This spec owns TileFoundry's fact layer: everything a later stage decides
*over*, and nothing that decides anything itself. It has two surfaces:

| Surface | Entry | What it states |
|---|---|---|
| Program check | `check_program(module, function, topology_level=..., budget=..., analyzers=...)` | an inlined Function view after validating one authored program, its declared topology, and what each requested analysis needs of it |
| Composed measurement | `analyze(module, function, analysis=...)` | one or more root analyses and their union dependency closure, leaving typed Metadata on the IR |

Per-Op semantic derivation — typeinfer, the forward access relation, shard
propagation — is owned by [semantic-analysis](./semantic-analysis.md), and the
registries behind it by [visitor-registry](./visitor-registry.md); the families
below consume the forward relation
([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation))
rather than restating it.

## 1. Authored-HIR metrics

The measurement entry is the composed operation ([§2](#2-composed-analysis)).
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

### 1.2 Analysis families

The first families are `compute-cost`, `memory`, `roofline`, and `performance`.
Each owns its record types and declares its dependencies and output additions.

| Selector | Requires | Owns | Attaches to | Rests on | Text summary adds | Annotates equations |
|---|---|---|---|---|---|---|
| `compute-cost` | - | `ComputeCostMetadata` | every measured Call and the Function | the authored program | `compute-cost` | every measured Call |
| `memory` | - | `MemoryMetadata`, `RegionMemoryMetadata` | `MemoryMetadata` on every measured Call; `RegionMemoryMetadata` on the Function | the authored program, `MemoryHierarchyFacts`, `ParallelCapacityFacts` | `memory`, `advisory` | every measured Call |
| `roofline` | `compute-cost`, `memory` | `RooflineMetadata` | every measured Call and the Function | `ThroughputFacts` | `roofline` | every measured Call |
| `performance` | `compute-cost`, `memory` | `PerformanceMetadata`, `PerformanceSummaryMetadata` | `PerformanceMetadata` on every Call with a modeled duration; `PerformanceSummaryMetadata` on the Function | `ThroughputFacts`, `ParallelCapacityFacts`, `MemoryHierarchyFacts` | `performance` | every Call with a modeled duration |

Every compact text summary begins with these two lines:

```text
# example
# analysis target=<target> module=<module> function=<function> topology=<level> wave=<counted>/<declared>
# selection requested=<selector>[,<selector>...] executed=<selector>[,<selector>...]
```

Every summary line is one record walked exactly as an annotated equation is
([inspection §2.8](./inspection.md#28-record-comment-forms)), so the two surfaces
cannot spell one value two ways. What the report is about and what was asked of it
are records of the report rather than of the IR; every other summary line is a
record of the selected Function.

The JSON report carries the same identity and selection in `target`, `module`,
`function`, `topology`, `wave`, `requested`, and `executed`. `wave` holds
`counted` and `declared`.
Whole-function
projections are under `function_records`; `calls` is a value-ordered list whose
entries have a `value` label and one key per selected family. `loops` is the
corresponding authored-loop list, labelled by induction variable. Memory does not
attach an empty record to a loop for a conclusion it has not computed. `totals`
appears when the selected view includes compute cost or roofline's bounded work
evidence.

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
    execution placement for every occurrence that will take time. Where the
    buffers go is not a readiness question: nothing here decides it.
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
    Target-selected descriptor ([§2.2](#22-target-selected-analyzers)).
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

#### 1.2.1 `compute-cost`

`compute-cost` measures the logical work of each authored `Call` without reading
target hardware facts. What an occurrence moves is the memory family's answer
([§1.2.2](#122-memory)), read off the same registered evaluator.

```python
class ComputeCostMetadata(IRMetadata):
    """Typed work in logical, expanded, and topology-unit domains."""

    topologies: tuple[str, ...] = ()
    flops: Breakdown[int] = Breakdown()
    other_ops: Breakdown[int] = Breakdown()
```


| Field | How it is computed | Reads the target |
|---|---|---|
| `topologies` | The effective Module topology levels, coarsest first. | No |
| `flops` | For a primitive Call, run its registered cost evaluator over operand and result Types as written; the total then multiplies by the enclosing recomputation factor and the number of positions in its execution scope, and each level's share is the same evaluator over Types projected through authored `Split`s at or coarser than that level. For a Function Call, take the callee's summed record and multiply by the call site's factor. | No; projection reads resolved Mesh and effective Module topology extents. |
| `other_ops` | The evaluator's non-floating-point operation counts (`integer`, `predicate`, `select`, `special`), totalled and shared the same way. These keys map directly to target one-unit operation-throughput keys; target-side service naming is unchanged. | No; projection reads resolved Mesh and effective Module topology extents. |

Requesting this family adds one summary line, prefixed by `# `: the Function's own
record, stated exactly as a Call's is. The whole program's work is not a second
record.

```text
compute-cost flops=<dtype>:<int>@logical,<int>@total,<int>@<level>[,...][;<dtype>:...] other-ops=<kind>:<int>@logical,<int>@total,<int>@<level>[,...][;<kind>:...]
```

Every measured Call receives this annotation. Each key pairs the whole quantity
with one unit's share, so the two `*_per_unit` fields are not separate keys.

Each reported Call's JSON projection is under its `compute-cost` key:

```text
{"topologies": [<level>, ...],
 "flops": {<dtype>: {"logical": <int>, "total": <int>,
                      "per_unit": [<int>, ...]}},
 "other_ops": {<kind>: {"logical": <int>, "total": <int>,
                         "per_unit": [<int>, ...]}}}
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

#### 1.2.2 `memory`

`memory` states what every occurrence moves and at which level, the unique
addresses one wave touches at a representative iteration, whole-Function value
lifetimes and placement peaks, and which repeated reads fit in cache. The movement
is read off the Op's own registered evaluator and the amounts its access
relations reach.

Every measured `Call` carries `MemoryMetadata`; every reachable `Function`
carries `RegionMemoryMetadata`. A `LoopRegion` carries neither an empty record
nor a footprint merely to reserve an attachment point.

```python
class MemoryMetadata(IRMetadata):
    """One primitive Call's memory behavior for one occurrence."""

    topologies: tuple[str, ...] = ()
    traffic: Traffic = Traffic()
    operands: tuple[TrafficBytes, ...] = ()
    footprint: Footprint | None = None


class ReuseWindow:
    """One buffer's re-reads, and what keeping it costs the cache."""

    buffer: str
    time: str = ""
    space: str = ""
    holds_bytes: int = 0
    reuse_bytes: int = 0
    fits: bool = True
    complete: bool = True


class RegionMemoryMetadata(IRMetadata):
    """One Function's aggregate memory conclusions."""

    solver_status: str
    topologies: tuple[str, ...] = ()
    traffic: Traffic = Traffic()
    footprint: Footprint | None = None
    reuse_windows: tuple[ReuseWindow, ...] = ()
    lifetimes: tuple[ValueLifetime, ...] = ()
    peaks: tuple[MemoryLevelPeak, ...] = ()
    errors: tuple[str, ...] = ()
    advisories: tuple[str, ...] = ()
```

##### `Traffic`

```python
class Spread[V]:
    """One quantity, whole and as one unit of each topology level holds it.

    Attributes:
        logical: attribute; What the authored operation asks before replication.
        total: attribute; What the whole execution asks.
        per_unit: attribute; What one unit of each level holds, in topology order.
    """

    logical: V
    total: V
    per_unit: tuple[V, ...] = ()


class Breakdown[V]:
    """One category's quantities, split by kind."""

    kinds: tuple[tuple[str, Spread[V]], ...] = ()


class Traffic:
    """Movement grouped by storage and communication boundary."""

    storage: Breakdown[TrafficBytes] = Breakdown()
    communication: Breakdown[TrafficBytes] = Breakdown()
```

Every traffic amount is what a boundary's own relation reaches. The Op's
evaluator says which way each boundary moves and whether it materialises
anything; it does not say how much, and an Op with no relation fails closed.

| Field | How it is computed | Reads the target |
|---|---|---|
| `MemoryMetadata.topologies` | The effective Module topology levels, coarsest first. | No |
| `MemoryMetadata.traffic` | One occurrence's per-boundary movement, grouped by storage and communication boundary. | No; projection reads resolved Mesh and topology extents. |
| `MemoryMetadata.operands` | One occurrence's movement in order `(*call.args, call)`. | No |
| `RegionMemoryMetadata.topologies` | The same effective Module topology levels. | No |
| `RegionMemoryMetadata.traffic` | Every reachable occurrence. `logical` multiplies only loops the value varies in; `total` and `per_unit` multiply every enclosing loop. | No |

- constraints:
  - One relation MUST answer for the whole program and for one unit, from one
    registration; every boundary MUST be held to the iterations its participant
    performs. Projecting an operand's Type is not enough, because a value nobody
    sharded projects to the whole of itself: that is what makes a broadcast
    operand cost its own size and a `Reshard` the distinct coordinates it
    reaches rather than a full source per participant.
  - Each leaf's bytes are charged at the level that leaf sits at. A `UMAT` leaf
    in `Call.args` charges its own bytes at `rmem` and one appearing only in an
    Op attribute charges nothing, so a whole traffic amount and an `operands`
    entry MUST NOT be assumed equal for a Type whose leaves occupy several
    levels. Where those bytes were placed enters neither.
  - Two operands MAY name the same value; the `operands` split MUST keep their
    positions distinct, and MUST omit an entry it cannot state rather than emit
    it empty.
  - Every `Spread` MUST state a share for each declared topology level, not only
    for the level the call selected, and the record MUST name those levels once
    in `topologies` rather than beside each share.
  - A movement has two coordinates and MUST be stated in both. `storage` names
    the level the bytes entered or left. `communication` names the topology
    level whose boundary they crossed, which no storage level can answer: data
    handed from one unit to another is the same storage at both ends and has
    still gone somewhere. The same bytes MUST appear under both, exactly as a
    move between two storage levels is read at one and written at the other.
  - Under `communication`, `read` and `write` are the unit's own view: what it
    received and what it sent. A unit finer than the boundary MUST state no
    share of it -- crossing is what the units on either side do, and dividing
    the move among the units inside one states a move nobody made.
  - A duration MUST take whichever of compute, storage movement and crossing is
    longest, and MUST NOT sum them: one movement spends two resources over one
    span of time. A crossing at a level the target publishes no rate for MUST
    be stated and left untimed.
  - A Call's `traffic` and `operands` MUST state one occurrence. Only the
    Function record counts an occurrence as often as its authored loops repeat
    it.
  - A capacity conclusion MUST NOT correct or invent a movement number. A
    window whose start arrives at run time reads that start rather than becoming
    a full read of its source and a write of its result.
  - Which boundaries move is the Op's evaluator's answer and MUST NOT be read
    off value lifetimes. A boundary it reports no direction on moves nothing.
    The numbers that place a window MUST be read like any other operand: one
    element per number, reached through the boundary's own relation onto the
    flat leaves the operand holds, and charged at each reached leaf's own width.
    An operation that writes at an address it is given reads that address the
    same way.

##### `Footprint`

A footprint counts the unique addresses a program touches in the memory level
a cache backs. It is stated per source buffer, named as a lifetime names one.

```python
class Footprint:
    """Unique bytes one wave touches, or the lower bound when incomplete."""

    buffers: tuple[tuple[str, Breakdown[int]], ...] = ()
    complete: bool = True
```

| Field | How it is computed | Reads the target |
|---|---|---|
| `Footprint.buffers` | Group reached address sets by source-buffer identity, union each group, count its elements, and pack the source dtype's bits into whole bytes. Each buffer has one memory-level kind whose `Spread` states the same count in `logical` and `total` and has no `per_unit` entries. | `MemoryHierarchyFacts` selects the level; [target §11](./target.md#11-target-facts-projection) supplies `ParallelCapacityFacts`. |
| `Footprint.complete` | False when any contributing boundary is not exact, has no finite count, or belongs to a refused scope; otherwise true. | No |
| `MemoryMetadata.footprint` | The unique addresses this occurrence's own boundaries reach at every enclosing loop's first iteration over one wave, or `None` when no wave can be stated. | As above |
| `RegionMemoryMetadata.footprint` | The union of every Call's reached addresses below the Function, deduplicated per buffer before counting, or `None` when no wave can be stated. | As above |

- constraints:
  - Overlapping ranges into one buffer MUST be unioned before they are counted,
    and the byte count MUST be `ceil(elements * dtype.bit_width / 8)`.
  - Both directions contribute: a load and a store to that level each occupy the
    cache. Only boundaries against that level are counted; a boundary against
    another level MUST NOT enter.
  - A boundary contributes only when its positional entry in
    `MemoryMetadata.operands` reports a nonzero read or write. This MUST use the
    same Op-evaluator answer as traffic; a boundary with no direction moves
    nothing and MUST NOT enter the footprint.
  - One window is defined, and no other: every enclosing loop's induction
    variable is held at its first iteration, over one wave of units. A Call
    states the footprint of its own boundaries; a Function states the union of
    every Call below it, deduplicated per buffer before any byte count is taken.
    A `LoopRegion` MUST NOT carry a footprint.
  - A loop start MAY depend on a unit coordinate. Holding such a loop at its
    first iteration MUST constrain it to that start expression, not to a
    constant; the reached range then still varies with the coordinate.
  - A unit coordinate MUST be eliminated by union over the wave, never by taking
    the largest single unit's count. The two differ whenever units reach
    different addresses.
  - A program declaring more units than the target holds MUST NOT have them all
    counted as concurrent. The wave is the first `wave_units` positions in the
    mesh's own linear order, taken through `Mesh.layout`'s strides.
  - Each buffer's count is one number. It MUST be stated in every counting
    domain a `Spread` carries, because a union over units divides back into no
    per-unit share.
  - `Footprint.complete` is the only completeness marker. It is false when any
    contributing boundary is not `AccessPrecision.EXACT`, has no finite count,
    or belongs to a refused scope, and the stated bytes are then a lower bound.
  - A footprint MUST be absent, not empty, when analysis cannot state the wave
    it is taken over.

##### `ValueLifetime`

```python
class ValueLifetime:
    """One value's residency, as positions in the Function's value order."""

    binding: str
    memory_level: str
    bytes: int
    defined_at: int
    last_used_at: int
    persistent: bool = False
```

One ordinary expression event uses its operands and defines its result. A
region adds separate binding and exit events: a mesh argument is used before
its parameter is defined, and a loop initial value is used before its induction
and carried parameters are defined. One representative loop-body iteration is
recorded without expanding the trip count; a carried parameter spans entry to
exit, and a yielded value remains live through the backedge event. Positions
are monotonic across the whole Function, including nested and sibling regions.

| Field | How it is computed | Reads the target |
|---|---|---|
| `ValueLifetime.binding` | Use the parameter or binding name, suffixed with `:` and the line of the value's source span when it has one. Repeated names differ by the printer's numeric suffix in definition order. A value with neither name nor span is `<value N>` in definition order. | No |
| `ValueLifetime.memory_level` | Emit one lifetime per storage level occupied by the value's Type. | No |
| `ValueLifetime.bytes` | Project the Type through every authored split at or coarser than the explicit level's `owner`, then take its logical bytes; a target-owned or undeclared level remains global. | `MemoryHierarchyFacts.explicit_levels[].owner` |
| `ValueLifetime.defined_at` | Definition event on the Function-wide structured SSA timeline. | No |
| `ValueLifetime.last_used_at` | Greatest ordinary-consumer, region-entry, loop-backedge, or region-exit use event; the final timeline event for a parameter. | No |
| `ValueLifetime.persistent` | True for parameters and false for body allocations. | No |
| `RegionMemoryMetadata.lifetimes` | Every value residency except a non-material view. | As above |

- constraints:
  - `Reshape` and `Transpose` describe bytes their operand already holds and
    MUST NOT receive independent lifetimes. Every other result, including a
    window, tuple field, or result that overwrites a destination, MUST allocate
    its own. Analysis MUST use operation semantics for this distinction rather
    than infer aliasing from layouts.
  - A caller-owned parameter MUST NOT be reused. Donation is a contract with
    the caller, not a conclusion this family may draw.

##### `MemoryLevelPeak`

Capacity is settled against authored definition order, which fixes every
buffer's lifetime before any is measured. For `gmem` and `smem`, exact
polyhedral access relations may let the solver overlap a dead pointwise operand
with its result or embed an `insert_slice` update in its result. Every logical
SSA box remains in the model. The concrete arrangement is not reported. `rmem`
is not address-solved and reports only the largest single projected logical
value.

```python
class MemoryLevelPeak:
    """How much of one memory level a Function needs at its peak."""

    memory_level: str
    peak_bytes: int
    persistent_bytes: int
    capacity_bytes: int | None = None
```

| Field | How it is computed | Reads the target |
|---|---|---|
| `MemoryLevelPeak.memory_level` | Each storage level with at least one lifetime or traffic entry, sorted by name. | No |
| `MemoryLevelPeak.peak_bytes` | For `gmem` and `smem`, the address high-water mark of the first feasible whole-Function placement. Exact pointwise relations and exact `insert_slice` partitions may permit overlap; widened or unknown relations do not. For `rmem`, the largest single projected logical value. | No |
| `MemoryLevelPeak.persistent_bytes` | Sum of persistent lifetimes at that level. | No |
| `MemoryLevelPeak.capacity_bytes` | Capacity of the matching explicit level, or `None` when unknown. | `MemoryHierarchyFacts.explicit_levels[].capacity_bytes` |
| `RegionMemoryMetadata.peaks` | One peak per occupied or moved storage level. | As above |
| `RegionMemoryMetadata.solver_status` | `"feasible"` after the whole-Function placement settles, including when there is no addressable value. | No |
| `RegionMemoryMetadata.errors` | Non-fatal placement-capacity and cache-capacity failures. | `MemoryHierarchyFacts` |
| `RegionMemoryMetadata.advisories` | Lower-severity target-aware memory findings recorded by this family. | `MemoryHierarchyFacts` |

- constraints:
  - `RegionMemoryMetadata` MUST be attached per reachable `Function`; a peak
    spans its live ranges and belongs to no single expression.
  - An access relation that keeps a parameter with a stated finite range is
    exact and MAY prove overlap. A widened relation, and one with an unbounded
    parameter, MUST NOT.
  - Placement MUST be settled for the addressable levels `gmem` and `smem` only,
    once per capacity domain that holds a buffer -- the whole target for a level
    owned target-wide, one per owning position otherwise -- with two buffers in
    one domain never live in the same bytes at once. Residency at another level
    MUST NOT make a program unplaceable, and a level owned per unit of a topology
    other than the one being analysed MUST fail rather than be assumed. Domains
    holding the same buffers are one question, decided once.
  - A domain that cannot be expressed or does not settle in time MUST raise
    `AnalysisError` and leave no record. The solver MUST stop at its first
    feasible assignment rather than prove a minimum. Capacity MUST NOT restrict
    the address space.
  - A solved explicit-level peak exceeding capacity MUST add a non-fatal
    `errors` entry, preserve the complete result, and MUST NOT fail the call.

##### Cache occupancy

A cache holds data because it will be read again, and whether it still holds it
is decided by what was touched in between. This analysis states one row per
buffer that is read again, naming the two axes a second read can come from: the
loop whose next iteration reads it, and the mesh axis whose units read it at
once. A buffer with neither states no row.

| Field | How it is computed | Reads the target |
|---|---|---|
| `ReuseWindow.buffer` | The source buffer's lifetime label. | No |
| `ReuseWindow.time` | The outermost enclosing loop whose different iterations reach the same addresses, named by its induction variable; empty when no loop supplies a second read. | No |
| `ReuseWindow.space` | The mesh axis whose different coordinates reach the same addresses; empty when no mesh axis supplies a second read. | No |
| `ReuseWindow.holds_bytes` | Unique bytes of every buffer the whole wave touches while this buffer must remain resident. | `MemoryHierarchyFacts` selects the backed level; [target §11](./target.md#11-target-facts-projection) supplies `ParallelCapacityFacts`. |
| `ReuseWindow.reuse_bytes` | `(time trips * space units - 1)` times this buffer's unique bytes in the window; an absent axis contributes one. | As above |
| `ReuseWindow.fits` | True exactly when `holds_bytes` is less than the cache capacity. | `MemoryHierarchyFacts.implicit_levels[]` |
| `ReuseWindow.complete` | False when any boundary contributing to `holds_bytes` is inexact, uncountable, or refused; otherwise true. | No |
| `RegionMemoryMetadata.reuse_windows` | One row for every buffer with a time or space reuse axis and nonzero savings. | As above |

- constraints:
  - The stated bytes are everything the whole wave touches while that buffer
    must stay resident, every buffer included, not only the one read again:
    the others are what evict it.
  - A read by another unit of the same wave counts. On a target whose units
    share one cache, data several units read at once is the common case, and a
    model counting only one unit returning later would state no reuse at all
    for a schedule giving each unit one output tile.
  - A mesh axis supplies reuse exactly when one unit reaches the same addresses
    as the wave union along that axis. The spatial repeat count is how many
    units in the wave map onto those same addresses.
  - The window is the loop axis when there is one, and the mesh axis alone
    otherwise, because units reading at once are already inside one iteration
    of the loop that carries the later read.
  - Rows MUST NOT be summed. One row's window lies inside another's whenever
    its axis is nested inside, so a row that fits implies those nested in it
    fit.
  - Rows MUST be ordered by `reuse_bytes`, largest first. Equal reuse amounts retain
    their derivation order.
  - Data read once states no row; its bytes still enter every row whose window
    contains it.
  - A row whose computed reuse is zero states no row.
  - A window above capacity MUST add one non-fatal `errors` entry, regardless
    of how many buffer rows share it, and MUST NOT fail the call.
  - The stated bytes are a lower bound when any contributing boundary is
    inexact, exactly as a footprint is.
  - The capacity is stated per one instance of the cache's `scope`, and this
    analysis compares one wave against one instance. A deployment that spreads
    one wave across several instances is not modelled.
  - Per-unit control flow is out of scope: HIR states no conditional region, so
    two units differ only by the iteration domain a coordinate gives them.
  - This is the capacity judgement of an idealised fully associative LRU cache.
    Miss counts, miss rates and replacement policy state nothing here.

The report identity, not `RegionMemoryMetadata`, states the program-dependent
machine context. Its wave is `min(declared units, units the target runs at
once)` over the topology named by `ParallelCapacityFacts`; it is the top-level
`wave` field in JSON and appears once on the `analysis` text line. Cache level
and capacity remain Target facts and appear in this report only when they decide
a cache-capacity finding.

##### Target facts

The family reads the following target hierarchy. Parallel capacity is stated
once in [target §11](./target.md#11-target-facts-projection), because `memory`
and `performance` both consume it.

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

- constraints:
  - The memory levels MUST be two flat tuples with a separate relation edge
    list.
  - A GPU projection MUST cover the explicit levels a program can name and the
    caches traffic passes through, and MUST state that L1 caches L2, L2 caches
    global memory, and L1 divides one physical block with shared memory. A target
    with no sharing MUST express that with no sharing edge.
  - An implicit level MUST NOT receive a fixed capacity where its usable
    capacity depends on the program; that capacity MUST be derived from the
    sharing edge and the sharing level's measured peak.
  - Every explicit level MUST carry an `owner` supplied by the Target. It MUST
    be a declared Target topology or `target`. An implicit cache MUST NOT carry
    an owner.
  - Analysis MUST NOT infer memory ownership from a storage level's name or
    capacity scope.

##### Printed and JSON surface

Requesting memory adds one Function line and one line per finding:

```text
memory traffic=<memory-level>:r<bytes>/w<bytes>@logical,r<bytes>/w<bytes>@total,r<bytes>/w<bytes>@<topology>[,...] footprint=<buffer>:<bytes>[;<buffer>:<bytes>] peak=<level>:<bytes>[,...] persistent=<level>:<bytes>[,...]
  buffer=<buffer> holds=<bytes> time=<loop|none> space=<mesh-axis|none> reuse=<bytes> fits=<yes|no>
  error="<text>"
  advisory="<text>"
```

The `<bytes>` form selects the largest of `B`, `KB`, `MB`, and `GB` whose unit
the value reaches, with 1024 between adjacent units. `B` is an integer; every
larger unit has two decimal places. Zero is `0`. This one form is used for every
human-readable byte count, including error text; JSON keeps raw byte integers.

Each buffer, error, and advisory is its own line indented beneath the `memory`
family line. Errors and advisories are quoted and escaped. Every measured Call
receives a `memory` annotation; `operands` is emitted only when asked for
([cli Analyze](./cli.md#analyze)):

```text
memory traffic=<memory-level>:r<bytes>/w<bytes>@logical,r<bytes>/w<bytes>@total,r<bytes>/w<bytes>@<topology>[,...] footprint=<buffer>:<bytes>[;<buffer>:<bytes>] [operands=<position>:r<bytes>/w<bytes>[;<position>:...]]
```

In the printed `footprint` field, a buffer uses the same value label as a
lifetime binding and that label may itself contain `:` (for example,
`v0:57:1.00KB`). The byte count is the formatted value after the last colon.

Missing optional conclusions omit their whole printed field. Call and Function
JSON projections are both under `memory`. The Function's full projection is
under `function_records.memory`:

```text
{"topologies": [<level>, ...],
 "traffic": {"storage": {<memory-level>: <spread>, ...},
             "communication": {<topology-level>: <spread>, ...}},
 "footprint": {"buffers": {<buffer>: {<memory-level>: <spread>}, ...},
               "complete": <bool>} | null,
 "reuse_windows": [{"buffer": <name>, "time": <loop|"">,
                     "space": <mesh-axis|"">, "holds_bytes": <int>,
                     "reuse_bytes": <int>, "fits": <bool>,
                     "complete": <bool>}, ...],
 "lifetimes": [{"binding": <name>, "memory_level": <level>, "bytes": <int>,
                "defined_at": <int>, "last_used_at": <int>,
                "persistent": <bool>}, ...],
 "peaks": [{"memory_level": <level>, "peak_bytes": <int>,
             "persistent_bytes": <int>, "capacity_bytes": <int|null>}, ...],
 "solver_status": "feasible",
 "errors": [<text>, ...],
 "advisories": [<text>, ...]}
```

- constraints:
  - Text and JSON MUST project analysis conclusions from these records. The
    report identity's `wave` is read from the resolved Module Target and the
    program's declared units and MUST NOT be copied into the memory record.
  - A missing footprint MUST be JSON `null`; it MUST NOT be represented by an
    empty `Footprint`.

#### 1.2.3 `roofline`

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
    def bandwidth(self, memory_level: str) -> int | None: ...
```

The service kinds a target states are `integer`, `predicate`, `select` and
`special`. The names are this project's, not any vendor's, so each MUST say in
its provenance which published row it was derived from. A service is work the
machine does, not movement it makes: bytes are priced by a bandwidth, never by
standing an instruction rate in for one.

Requesting roofline adds this verdict and the two quantities the bound divides
-- the summed `flops` and the bandwidth-level bytes -- as `totals`. Nothing else
its dependencies wrote is promoted; asking for `memory` is what states those
([§1.2.2](#122-memory)).

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
carries only exact `flops`, `traffic`, and `communication` sums. Dependency
records remain on the semantic result but do not enter `function_records`, Call
annotations, or Call report rows. Independently requesting a dependency selects
its full form as defined in that family's section.

- constraints:
  - `ThroughputFacts.peak_for` MUST return `None` for a dtype with no published
    rate; analysis MUST NOT substitute an assumed rate.
  - `PerformanceServiceFacts.flops`, `ops` and `bandwidth` MUST return `None`
    for an unstated dtype, kind or level. Performance MUST reject non-zero work
    of that dtype, kind or level and MUST NOT substitute the whole-device rate
    or another kind's rate.
  - `bandwidth_level` MUST select the traffic level divided by the published
    bandwidth rather than summing traffic across levels.
  - Performance local duration MUST divide one unit's share of
    `ComputeCostMetadata.flops` by `unit_flops`, of `service` by `unit_ops`, and
    the `bandwidth_level` entry of `MemoryMetadata.traffic.storage` by
    `unit_bandwidth`, all at the level it was asked about. Compute and
    movement overlap within one occurrence, so its duration is the greater of
    the two sides rather than their sum.
  - Traffic at a level with no stated one-unit bandwidth MUST remain visible in
    `MemoryMetadata` and MUST NOT enter a duration: an instruction throughput
    standing in for a bandwidth prices a move as though it were arithmetic.
  - Having moved bytes and having work this can time are different questions.
    What decides the second is the quantities a rate exists for: a nonzero
    share of `flops` or of `service`, or nonzero
    `MemoryMetadata.traffic.storage` at `bandwidth_level`. An occurrence with none of
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

#### 1.2.4 `performance`

`performance` places compute-cost-priced occurrences on a CTA-local nominal
timeline, holds the buffers they keep live to the levels this model addresses,
and scales the root timeline by a fixed physical parallel capacity. The records
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
| `start_ns` | Start of one occurrence on the authored-order local timeline, after its producers end and after the last occurrence sharing any of its participants. | No |
| `end_ns` | End of that occurrence's first execution. | No |
| `trips` | One outside a loop; within a loop, the enclosing loop trip count represented by the interval. | No |
| `stride_ns` | Zero outside a loop; within a loop, the makespan of one body execution. | No |

Function summary fields are:

| Field | How it is computed | Reads the target |
|---|---|---|
| `timeline` | `[0, local makespan * waves)`, where the local makespan is the end of the CTA-local timeline, or zero with no work. Its duration is the prediction. | Through `waves` |
| `waves` | `ceil(N / P)`, where `N` is the static extent of the root topology selected by `ParallelCapacityFacts.topology` and `P` is `parallel_units`. | `ParallelCapacityFacts` |

Occurrence intervals remain CTA-local. They are not copied once per wave, and
neither the root topology extent nor `parallel_units` changes them. The capacity
`P` is compiler policy for concurrent instances; it is distinct from the
per-unit rates in `ThroughputFacts` even when both projections derive from the
same physical unit count today. The aggregate itself is stated once in
[target §11](./target.md#11-target-facts-projection), because `memory` reads it
too.

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
  - A primitive Call is eligible for performance when it is reachable in the
    authored HIR. Its execution region is represented structurally by
    `MeshRegion`; the result layout remains an independent property. A result
    carrying no `ShardLayout` MUST NOT unplace the occurrence that produced it.
    An occurrence
    with no nonzero share of
    `flops`, none of `service` and no nonzero
    `MemoryMetadata.traffic.storage` at `bandwidth_level` is structural to this
    model: it needs no execution placement and MUST receive no record, because
    an empty
    interval reads as a measurement rather than as the absence of one. Movement
    at another level does not change that and MUST NOT be dropped from
    `MemoryMetadata` because of it -- structural here means nothing to time,
    not nothing done. It still carries its producers' precedence to its
    consumers. Inputs MUST NOT supply placement for an unplaced occurrence.
  - The global total for an occurrence is its per-unit quantity multiplied by
    the number of positions in the enclosing execution scope. This multiplier
    counts copied or replicated work: an unsharded operation inside a scope is
    performed independently by every position and therefore contributes once
    per position to the total. The per-unit quantity already includes work
    projected through finer levels.
  - A `MeshRegion` evaluates its boundary arguments outside the region and
    executes its body once per enclosing position. Argument work is charged at
    its defining site; body work is charged per enclosing position.
  - The participant set MUST be the exact image of that Mesh's layout under
    [shard §5](./shard.md#5-mesh), not an extent inferred from a topology or an
    operand. A `Broadcast` shard attribute still names placement: attributes
    describe distribution while the Mesh describes which positions participate.
  - Every primitive occurrence has one fixed duration and occupies its exact
    participant set. An SSA consumer MUST start no earlier than its producers
    end. Two positive-duration occurrences whose participant sets intersect
    MUST NOT overlap; disjoint sets MAY overlap, while a partial intersection
    serializes each whole occurrence rather than splitting it by participant.
  - A `LoopRegion` MUST be represented as one structured performance node. Its
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
    independent work is a later decision, not an analysis's: what overlaps
    is what the program's own placement made independent, and the reported time
    is the time of the program as written. On a `Function`, the summary's
    `timeline` MUST start at zero and span the whole local plan, scaled by
    `waves`; there MUST be no second field restating the local makespan or the
    scaled estimate, and none restating how the layout was reached -- it is
    exact for the model it states.
  - `parallel_units` is compiler policy over hardware facts. It MUST NOT enter
    one-unit rates or the CTA-local layout, and is not a program rewrite.
  - The buffers a plan keeps live MUST have been placed by `memory` before a
    time is reported for it. A successful dependency records
    `RegionMemoryMetadata.solver_status="feasible"`; a placement that failed
    never reaches performance because `memory` refuses it.
    Capacity therefore changes whether there is an answer, never which answer:
    two capacities that both admit a placement MUST produce the same intervals.
  - Performance is a modeled plan and MUST NOT be read as a guarantee about
    lowering, physical occupancy, or runtime performance.

## 2. Composed analysis

`tilefoundry.analysis.check_program` is the shared, reusable gate before an
analysis runs.

```python
def check_program(
    module: "Module",
    function: "Function",
    *,
    topology_level: str | None = None,
    budget: int = _INLINE_NODES,
    analyzers: tuple["Analyzer", ...] = (),
) -> "Function": ...


class AnalysisCheckContext:
    """What every input check reads: the program, the machine, and the topology level.

    Costing here is a question, not a record: nothing a context computes is
    attached.
    """

    module: "Module"
    function: "Function"
    target: "Target"
    topology_level: str | None
    whole: CostContext
    local: CostContext

```

- constraints:
  - The operation MUST infer types over the full reachable Function graph and
    validate its caller/callee execution context, and MUST NOT run an analysis
    or attach derived Metadata to the authored IR.
  - The reachable Function and Mesh geometry and every effective Module
    topology extent MUST be concrete before this operation runs. A public
    Analyze call with `dims` MUST resolve all three through one binding pass
    before calling this gate; a residual dimension expression MUST fail before
    any consuming algorithm runs.
  - Every effective Module topology MUST name a level the resolved Target
    supports. A resolved static extent MUST be positive and within that level's
    finite hardware limit. A rejection MUST name the level, its extent, and the
    reason.
  - A non-`None` `topology_level` MUST name exactly one effective Module topology.
  - Analyze MUST call this operation before any consuming algorithm, and
    MUST pass the whole resolved dependency closure as
    `analyzers`, so every analysis about to run states its input contract here.
  - The `analyzers` checkers MUST be bound to the derived Function and run in
    closure order: every `check_target`, then every `check_call` over one
    traversal of the derived non-Function calls, then every `finish`. One
    program MUST be walked once for this however many analyses asked, and the
    first refusal MUST stop the gate before any analysis writes.
  - The returned Function MUST inline every reachable HIR Function call at its
    call site while retaining each `LoopRegion` as one loop. Its induction
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
  - Authored-analysis readiness is not a program-level rejection. Analyze MUST
    NOT reject an authored `where(...)` constraint: it is an input to a later
    decision, and a program carrying one is measured as written. A value whose
    placement is deferred contributes its whole-program figures to the per-unit
    total, because a deferred layout states no distribution to project through;
    values with a resolved layout still project.

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
        topology_level: attribute; Topology level whose unit the per-unit quantities describe, or None.
        executed: attribute; Analyses executed in dependency order.
        metadata_types: attribute; Metadata classes actually written.
    """

    module: "Module"
    function: "Function"
    analyses: tuple[str, ...]
    topology_level: str | None
    executed: tuple[str, ...]
    metadata_types: tuple[type[IRMetadata], ...]


def analyze(
    module: "Module",
    function: "Function",
    *,
    analysis: str | Iterable[str],
    topology_level: str | None = None,
    options: object | None = None,
    dims: "Mapping[str, int] | None" = None,
) -> AnalysisResult: ...
```

- constraints:
  - One call MUST select one or more root analyses. It MUST preserve their
    first-occurrence order, resolve their union dependency closure, and execute
    every member once.
  - `topology_level` MUST name one effective Module topology. When omitted, it
    MUST default to the coarsest effective topology the target states a
    `ParallelCapacityFacts` for. A program may name a level the host places
    rather than the machine runs -- several cards are one deployment's shape,
    not one card's -- and measuring per such a level would ask the machine for
    a unit it publishes no rate for. When the target answers for none of the
    declared levels the coarsest MUST be left selected, so the refusal names
    the level rather than the absence of one; when the Module declares none,
    `topology_level` MUST remain `None` and no per-unit projection divides.
    `AnalysisResult.topology_level` MUST record the resolved answer.
  - A target MUST answer `PerformanceServiceFacts` and `ParallelCapacityFacts`
    for the level it is asked about, and MUST refuse a level it publishes no
    rate for rather than answering for a different one. What one unit gets
    through is the device peak over however many of that unit the device holds,
    so the same program measured at two levels states the same work against
    proportionally different rates.
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

### 2.1 Shared IterationScope and Access

The normalized HIR is visited once per `analyze()` call. That visit produces a
`IterationScope` tree parallel to Function/LoopRegion nesting and `Access` relations
for the narrow and device views. `IterationScope.domain` is the accumulated
authored loop domain; `IterationScope.accesses` and `IterationScope.refused` are
the shared family inputs for movement and footprint conclusions.
An input `Access` stores its original Call boundary index as well as its relation
and allocation expression; an output stores `input_index=None` and its own output
boundary index, which is the tuple field it answers for. Failed boundaries remain
absent without shifting the index on later successful inputs or outputs. Storage
level and element width are read from the allocation type. A refused descendant
makes its owning scope unknown for that view. Non-affine runtime indices retain
the widest legal access approximation.
Normalization clones each reached Function call site independently. Within one
call site, source expressions shared by identity remain one shared expression
in the clone; sharing never aliases the independently cloned body of another
call site.

`analysis.loop_terms` resolves HIR values to `LoopTerm` without depending on
isl. `analysis.access` turns those terms into isl constraints and owns access
widening; the two modules do not define a second affine graph representation.

- constraints:
  - A loop `start` or `extent` MAY be unit-dependent. Every runtime leaf in one
    MUST carry a half-open value range, and `IterationScope.domain` MUST keep the
    whole affine expression with each such leaf as one identity-deduplicated
    isl parameter constrained by that range. A leaf without a range MUST be
    refused.
  - `IterationScope.domain_params` MUST map each of those parameter names back
    to the value it stands for, because the name itself carries no meaning and
    a reader that must recognise a unit coordinate cannot re-derive it without
    restating how the name was made.
  - A loop `step` MUST be a literal; a parametric stride has no isl
    representation.
  - `cardinality` MUST enumerate every feasible integer point of a parameter box
    of at most `PARAM_POINT_LIMIT` points and return the maximum, and MUST
    report unknown for a larger box. It MUST count directly when every retained
    parameter is already fixed to one integer point. An empty parameter context
    MUST count as zero; a non-empty context with an unbounded parameter MUST
    report unknown.
  - `IterationScope.trips()` MUST fix child and parent domains to the same parameter
    point before dividing, and take the maximum of those ratios.

### 2.2 Target-selected Analyzers

```python
class AnalyzeContext:
    module: Module
    target: Target
    topology_level: str | None
    options: object | None
    root: IterationScope
    current: IterationScope


AnalysisCallable = Callable[
    [Function, AnalyzeContext], None
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
  - `AnalysisCallable` MUST receive the normalized Function graph and one
    `AnalyzeContext` carrying the exact Module, Target, resolved topology level,
    caller options, and the shared root/current `IterationScope` view. The
    `topology_level` MAY be `None` only when the Module declares no topology;
    options MAY be `None`.
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
