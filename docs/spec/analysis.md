# TileFoundry Spec — analysis (authored-HIR metrics + per-stage target facts)

This spec owns TileFoundry's fact layer: everything a later stage decides
*over*, and nothing that decides anything itself. It has two surfaces:

| Surface | Entry | What it states |
|---|---|---|
| Program check | `check_program(module, function, dims=..., topology=..., budget=..., analyzers=...)` | a checked Module and root Function; each derived Function carries internal `ExecutionMeshMetadata(mesh)` recording its unique enclosing mesh, or `None` for root |
| Composed measurement | `analyze(program, function, analysis=...)` | one or more root analyses and their union dependency closure, It writes typed Metadata in place on the program that `check_program` returns. It does not clone the program or bind dimensions. |

The caller binds dimensions through `check_program`, then passes both returned
objects to `analyze`. Local definition trees remain internal to analysis.
`analysis.Result.module` and `analysis.Result.function` identify the supplied program and root.

[semantic-analysis](./semantic-analysis.md) owns per-Op semantic derivation:
typeinfer, the forward access relation, and shard propagation.
[visitor-registry](./visitor-registry.md) owns the registries behind it. The
families below consume the forward relation
([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation)).
They do not restate it.

## 1. Authored-HIR metrics

The measurement entry is the composed operation ([§2](#2-composed-analysis)).
Each family below owns its record, field derivation, target facts, and rendered
forms. The command line composes one call per requested family and renders those
results together ([cli §Analyze](./cli.md#analyze)).

Typed records use the immutable `IRMetadata` and optional comment interface
defined by [core-ir §2](./core-ir.md#2-expr). The attachment point shows what a record describes. A record on a `Call`
describes that call. A record on a `MeshRegion` describes that execution stage.
A record on a `Function` describes the whole function.

- constraints:
  - One record type MUST mean the same quantity at every attachment point.
  - A `Function`-attached record MUST NOT be read as data the Function
    inherently carries. It states what one analysis found for one invocation.
    There MUST be no cross-call cache behind it.

### 1.2 Analysis families

The core families are `compute-cost`, `memory`, `roofline` and `performance`; a
Target adds its own through `get_analyzer`
([§2.2](#22-target-selected-analyzers)). Each owns its record types and declares
its dependencies and output additions.

| Selector | Requires | Owns | Attaches to | Rests on | Text summary adds | Annotates equations |
|---|---|---|---|---|---|---|
| `compute-cost` | - | `ComputeCostMetadata` | every measured Call and the Function | the authored program | `compute-cost` | every measured Call |
| `memory` | - | `MemoryMetadata`, `RegionMemoryMetadata` | `MemoryMetadata` on every measured Call; `RegionMemoryMetadata` on the Function and each occupied `MeshRegion` | the authored program, `MemoryHierarchyFacts`, `TopologyFacts` | `memory`, `advisory` | every measured Call |
| `roofline` | `compute-cost`, `memory` | `RooflineMetadata` | every measured Call and the Function | `ThroughputFacts` | `roofline` | every measured Call |
| `performance` | `compute-cost`, `memory`, `roofline` | `PerformanceMetadata`, `PerformanceSummaryMetadata` | `PerformanceMetadata` on every Call with a modeled duration; `PerformanceSummaryMetadata` on the Function | `ThroughputFacts`, `TopologyFacts`, `MemoryHierarchyFacts` | `performance` | every Call with a modeled duration |

Every compact text summary begins with these two lines:

```text
# example
# analysis target=<target> module=<module> function=<function> topology=<topology>
# selection requested=<selector>[,<selector>...] executed=<selector>[,<selector>...]
```

Each summary line is one record. The report walks it exactly as it walks an
annotated equation
([inspection §2.8](./inspection.md#28-record-comment-forms)). So the two surfaces
cannot spell one value two ways. The report subject and the request are records
of the report, not of the IR. Every other summary line is a record of the
selected Function.

The JSON report carries `target`, `module`, `topology`, `requested`, and
`executed` once. Its `functions` list
contains each reached definition once in root-first DFS order. Each entry has
`function` (`Module.fn`), `function_records`, operand-first `calls`, authored
`loops`, and `totals` when compute cost or roofline work evidence is selected.
Call rows carry `value` and selected-family records; loop rows use the induction
variable. Rows have no call-path field. Memory MUST NOT attach an empty loop
record for a conclusion it has not computed.

One result is rendered once, and every surface reads that rendering:

```python
def report(result: analysis.Result) -> dict[str, object]:
    """Project one composed Analyze result into a shared rendering structure."""
    ...

def render_analysis(result: analysis.Result) -> AnalysisRendering:
    """Render one annotated program and its report data in a single pass."""
    ...
```

- `report` MUST accept one `analysis.Result` and read every requested family's
  records from that result's record-bearing Function. It MUST NOT merge
  independently rebuilt Functions by identity, origin, dimensions, or walk
  position.
- Text, JSON, and annotated HIR MUST come from one `render_analysis` call over
  the same result's Function and selected Metadata types. The `value` of each
  rendered Call record is `<left-hand-side>:<line>`. Here `line` is the physical
  line that contains the statement's `=`. This holds even when its comment ends
  a later line of the same statement. Both surfaces MUST use the line locations
  that one printer pass collects. They MUST NOT recover locations from names or
  text.
- A Call equation carrying a record MUST state it in that line's Metadata
  comment. A carry update is a name rebinding rather than a Call equation, so it
  receives neither a comment nor a report row.
- A parameterized loop occurrence MUST stay one record. Neither surface expands
  it into one entry per trip.
- A compute-cost comment MUST state one unit's share beside the whole quantity.
  JSON MUST expose all four quantities. A reader MUST NOT need to reconstruct one
  from the others.

A value renders by the type its field holds.
[inspection §2.8](./inspection.md#28-record-comment-forms) owns those forms and
their separators. This layer settles which type a field holds and what its keys
name:

- A mapping's key is a dtype, a storage level, or an operand position -- an
  argument integer, or `result` for the value the Call produces.
- `<resource>` is `compute`, `memory`, `balanced`, `unrated`, or `none`.
- Bytes moved are `TrafficBytes`; a whole quantity paired with one unit's share
  is `TotalAndPerUnit`; a Call's occurrence on the timeline is one
  `TripInterval`.

- constraints:
  - A family MUST obtain hardware only through a Facts aggregate it declares
    ([target §11](./target.md#11-target-facts-projection)). Common analysis code
    MUST NOT branch on a concrete Target type. It MUST NOT call a complete Target
    analyzer. It MUST NOT resolve an undeclared Target to a default.
  - A family MUST read a dependency's record rather than recompute what it
    states. A number with two derivations has two answers.
  - Before any member of a requested union closure writes Metadata, Analyze MUST
    establish every requested root's family-specific readiness. Each family
    states its own through the checker its descriptor carries, and one
    metadata-free traversal of the derived program answers all of them.
    Performance readiness requires a selected topology whose
    `TopologyLimits.max_physical_units` is a positive integer and one-unit
    `PerformanceServiceFacts` stated for that same topology. The program
    states where an occurrence runs, so that checker visits no occurrence. Where
    the buffers go is not a readiness question. Failing performance readiness
    MUST NOT make the program invalid for `compute-cost`, `memory`, or `roofline`.
  - Global logical work, per-unit work, and lifetime order MUST remain
    target-independent. Physical capacity, hierarchy relationships, and
    throughput comparisons are target-aware.
  - A rendering MUST NOT be a field of the semantic result, and an analysis MUST
    NOT format one.
  - A rendering MUST report what the caller requested. Dependency records nobody
    requested MUST stay on the IR. A rendering MUST NOT report them, except for
    roofline's bounded evidence defined below. Record ownership MUST come from the
    Target-selected descriptor ([§2.2](#22-target-selected-analyzers)).
  - Every rendering of one run MUST select records through one shared decision.
    It MUST show only records actually written. It MUST carry the same
    conclusions.
  - Every reported quantity MUST come from a record. The only exception is a total
    that is the exact sum of records that state it. A quantity not derivable that way MUST be
    recorded by the analysis that computed it.
  - A family's JSON projection MUST be its record's fields under their own names,
    with nothing left out. A default, a `null`, and an empty mapping are each a
    fact a program branches on. A key spelled by hand can drift from the field it
    reports. A field whose projection needs the expression that carries the record
    MUST be declared as one. Such a field MAY be absent where the program offers
    no such reading. A comment over the same record MAY state fewer keys, or
    projected ones
    ([inspection §2.8](./inspection.md#28-record-comment-forms)). What the comment
    leaves out MUST stay in the JSON projection.
  - A compact text summary MUST contain whole-function facts only. Per-value
    facts MUST stay on their annotated equations; JSON MAY retain operand names
    and types in its structured projection.

#### 1.2.1 `compute-cost`

Analysis quantities share `analysis.Precision`: `EXACT`, `UPPER_BOUND`,
`LOWER_BOUND`, and `UNKNOWN`. Their JSON spellings are `exact`, `upper_bound`,
`lower_bound`, and `unknown`. When evidence combines, exact evidence is the
identity. Equal directions keep their direction. Opposing directions become
unknown. Unknown evidence absorbs every other direction.

`compute-cost` measures the logical work of each authored `Call` without reading
target hardware facts. The memory family answers what an occurrence moves
([§1.2.2](#122-memory)). It reads the answer off the same registered evaluator.

```python
class ComputeCostMetadata(IRMetadata):
    """Typed work in logical, expanded, and topology-unit domains."""

    topologies: tuple[str, ...] = ()
    flops: Breakdown[int] = Breakdown()
    other_ops: Breakdown[int] = Breakdown()
    precision: Precision = Precision.EXACT
```


| Field | How it is computed | Reads the target |
|---|---|---|
| `topologies` | The effective Module topologies, coarsest first. | No |
| `flops` | For a primitive Call, run its registered cost evaluator over operand and result Types as written for `logical`. Each topology's share is the same evaluator over Types projected through authored `Split`s at or coarser than that topology. `total` is the selected unit's share times its executing mesh units. Function accumulation multiplies `logical` by varying-loop trips. It multiplies `per_unit` and `total` by all enclosing loop trips. Retained calls carry the one-execution summary of their callee. Primitive work accumulates into root totals with full repeats. It accumulates into each enclosing call with repeats relative to that call scope. | No; projection reads resolved Mesh and effective Module topology extents. |
| `other_ops` | The evaluator's non-floating-point operation counts (`integer`, `predicate`, `select`, `special`), totalled and shared the same way. These keys map directly to target one-unit operation-throughput keys. Target-side service naming is unchanged. | No; projection reads resolved Mesh and effective Module topology extents. |
| `precision` | One-trip Call work is exact; Function work combines the precision of every enclosing loop's trip count. Coordinate-dependent maximum trip counts yield an upper bound. | No |

Requesting this family adds one summary line, prefixed by `# `: the Function's own
record, stated exactly as a Call's is. The whole program's work is not a second
record.

```text
compute-cost flops=<dtype>:<int>@logical,<int>@total,<int>@<topology>[,...][;<dtype>:...] other-ops=<kind>:<int>@logical,<int>@total,<int>@<topology>[,...][;<kind>:...] precision=<exact|upper_bound|lower_bound|unknown>
```

Every measured Call receives this annotation. Each key pairs the whole quantity
with one unit's share, so the two `*_per_unit` fields are not separate keys.

Each reported Call's JSON projection is under its `compute-cost` key:

```text
{"topologies": [<topology>, ...],
 "flops": {<dtype>: {"logical": <int>, "total": <int>,
                      "per_unit": [<int>, ...]}},
 "other_ops": {<kind>: {"logical": <int>, "total": <int>,
                         "per_unit": [<int>, ...]}},
 "precision": <"exact"|"upper_bound"|"lower_bound"|"unknown">}
```

- constraints:
  - A record MUST be attached to every reached `Call` and Function.
    The Function record MUST include authored-loop repetition. It is
    not the direct sum of the one-occurrence Call records.
  - An op with no registered cost evaluator MUST raise `analysis.Error`.
  - Dim arithmetic (`DimAdd`, `DimSub`, `DimMul`, `DimFloorDiv`, `DimMod`,
    `DimMin`, `DimMax`) is scalar address work, not tensor work.
    Its evaluator MUST state no flops, no service, and zero traffic in every
    operand slot.
  - Missing program geometry MUST NOT be replaced with a target capacity.
  - `logical` MUST multiply the trip counts of authored loops only. A loop
    counts only when the Call transitively reads its induction variable or
    carried argument.
    `per_unit` and `total` MUST multiply every enclosing loop's trip count.
    A Call written in a loop body executes on every iteration.
    The same rule MUST apply to primitive and Function Calls.
  - Downstream families MUST read the already-scaled record and MUST NOT apply
    authored loop trip counts a second time.

#### 1.2.2 `memory`

`memory` states what every occurrence moves and at which level. `memory` also
states the unique addresses one wave touches at a representative iteration.
`memory` also states whole-Function value lifetimes and memory peaks. `memory`
also states which repeated reads fit in cache. The analysis reads the movement
off the Op's own registered evaluator and the amounts its access
relations reach.

Every measured `Call` carries `MemoryMetadata`. Every reachable `Function`
carries `RegionMemoryMetadata`, as does each `MeshRegion` whose execution window
contains a live value. A `LoopRegion` and an empty `MeshRegion` carry no record
merely to reserve an attachment point.

```python
class MemoryMetadata(IRMetadata):
    """One primitive Call's memory behavior for one occurrence."""

    topologies: tuple[str, ...] = ()
    traffic: Traffic = Traffic()
    operands: tuple[TrafficBytes, ...] = ()
    footprint: Footprint | None = None
    buffer_bytes: int | None = None
    offsets: tuple[int, ...] = ()


class ReuseWindow:
    """One buffer's re-reads, and what keeping it costs the cache."""

    buffer: SSAValue
    ownership: tuple[str, ...] = ()
    time: str = ""
    space: str = ""
    holds_bytes: int = 0
    reuse_bytes: int = 0
    fits: bool = True
    precision: Precision = Precision.EXACT


class RegionMemoryMetadata(IRMetadata):
    """One Function's or MeshRegion stage's memory conclusions."""

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
    """One quantity, whole and as one unit of each topology holds it.

    Attributes:
        logical: property; What the authored operation asks before replication.
        total: attribute; What the whole execution asks.
        per_unit: attribute; What one unit of each topology holds, in topology order.
        logical_groups: attribute; Local logical values grouped by the induction
            variables and formals they depend on.
    """

    total: V
    per_unit: tuple[V, ...] = ()
    logical_groups: tuple[tuple[tuple[Var, ...], V], ...] = ()


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
anything. It does not say how much. An Op with no relation fails closed.
Movement and footprint counting for data-dependent lookup boundaries follow
[visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation).
Footprint retains the full possible address image for reuse decisions. Out-of-bounds
IndexSelect indices with `fill_value` are not statically distinguished. Traffic
counts one slice per index as an upper bound.

| Field | How it is computed | Reads the target |
|---|---|---|
| `MemoryMetadata.topologies` | The effective Module topologies, coarsest first. | No |
| `MemoryMetadata.traffic` | One occurrence's per-boundary movement, grouped by storage and communication boundary. | No; projection reads resolved Mesh and topology extents. |
| `MemoryMetadata.operands` | One occurrence's movement in order `(*call.args, call)`. | No |
| `MemoryMetadata.buffer_bytes` | The size in bytes of one physical copy of an address-placed `gmem` or `smem` tensor result. The size includes a carried phi. It also includes all contributions on Partial axes finer than its storage owner. It is `None` for tuple results or results that are not address-placed. | No |
| `MemoryMetadata.offsets` | The byte offset of each physical result slot: each copy of a staged tensor in copy order, or each tuple `AllocationValue` in flattened result-index order. An `rmem` result has no reported offsets. Staged tuple results MUST raise `analysis.Error`. | No |
| `RegionMemoryMetadata.topologies` | The same effective Module topologies. | No |
| `RegionMemoryMetadata.traffic` | Every reachable occurrence. `logical` multiplies only loops the value varies in; `total` and `per_unit` multiply every enclosing loop. | No |

- constraints:
  - A Call's traffic `total` MUST equal the counted unit's share multiplied by
    the number of units executing its mesh scope. The same rule holds for
    `compute-cost`.
    Storage MUST count the selected topology unit. Communication MUST count
    the topology unit named by its boundary. An undeclared communication
    boundary MUST raise `analysis.Error`. It MUST NOT use whole movement.
  - One relation MUST answer for the whole program and for one unit. It MUST do
    so from one registration. Every boundary MUST be held to the iterations its
    unit performs. Projecting an operand's Type is not enough. A value that
    nobody sharded projects to the whole of itself. So a broadcast operand costs
    its own size. A `Reshard` costs the distinct coordinates it reaches, not a
    full source per unit.
  - Each leaf's bytes are charged at the level that leaf sits at. A `UMAT` leaf
    in `Call.args` charges its own bytes at `rmem`. A leaf that appears only in an
    Op attribute charges nothing. Readers MUST NOT assume a whole traffic amount equals an `operands`
    entry for a Type whose leaves occupy several levels. Where those bytes were placed enters neither.
  - Two operands MAY name the same value. The `operands` split MUST keep their
    positions distinct. It MUST omit an entry it cannot state. It MUST NOT emit
    that entry empty.
  - Every `Spread` MUST state a share for each declared topology, not only
    for the topology the call selected. The record MUST name those topologies once in
    `topologies`, not beside each share.
  - A movement has two coordinates and MUST be stated in both. `storage` names
    the level the bytes entered or left. `communication` names the topology
    whose boundary they crossed. No storage level can answer that. Data
    handed from one unit to another is the same storage at both ends. It has
    still gone somewhere. The same bytes MUST appear under both. A move between
    two storage levels reads the bytes at one level and writes them at the other.
  - Under `communication`, `read` and `write` are the unit's own view: what it
    received and what it sent. A unit finer than the boundary MUST state no
    share of it. Crossing is what the units on either side do. Dividing
    the move among the units inside one states a move nobody made.
  - When a same-mesh `Reshard` resolves a `Partial` mesh axis across multiple
    units, communication MUST be recorded at that topology boundary. This holds
    even when every `Split` mesh axis is unchanged. An extent-one mesh axis contributes no communication.
  - A duration MUST take whichever of compute, storage movement and crossing is
    longest. It MUST NOT sum them. One movement spends two resources over one
    span of time. A crossing at a topology the target publishes no rate for MUST
    be stated and left untimed.
  - A Call's `traffic` and `operands` MUST state one occurrence. Only the
    Function record counts an occurrence as often as its authored loops repeat
    it.
  - A capacity conclusion MUST NOT correct or invent a movement number. A
    window whose start arrives at run time reads that start. It does not become
    a full read of its source and a write of its result.
  - The evaluator of the Op answers which boundaries move. The analysis MUST NOT
    read that off value lifetimes. A boundary with no reported direction moves
    nothing. The numbers that place a window MUST be read like any other
    operand. Read one element per number. Reach it through the boundary's own
    relation onto the flat leaves the operand holds. Charge it at each reached
    leaf's own width.
    An operation that writes at an address it is given reads that address the
    same way.

##### `Footprint`

A footprint counts the unique addresses a program touches in the memory level
a cache backs. It retains source-buffer identities; the printer supplies names.

```python
class Footprint:
    """Unique bytes one wave touches, with directional counting evidence."""

    groups: tuple[tuple[int, tuple[ReachedAddresses, ...]], ...] = ()
    # reached, buffers and precision are derived properties.
    # buffers yields (SSAValue, ownership, Breakdown[int]); no display strings.
```

| Field | How it is computed | Reads the target |
|---|---|---|
| `Footprint.buffers` | Group reached address sets by source-buffer identity, union each group, count its elements, and pack the source dtype's bits into whole bytes. Each buffer has one memory-level kind with one `Spread`. That spread states one window's unique bytes in `logical`. It states the executed windows' bytes in `total`. It has no `per_unit` entries. | `MemoryHierarchyFacts` selects the level; [target §11](./target.md#11-target-facts-projection) supplies `TopologyFacts`. |
| `Footprint.precision` | Combine each contributing boundary's direction: omitted boundaries or uncountable amounts are lower bounds, counted widened accesses are upper bounds, and refused boundaries have unknown direction. | No |
| `MemoryMetadata.footprint` | The unique addresses this occurrence's own boundaries reach at every enclosing loop's first iteration over one wave. `total` multiplies this window by the number of waves in the enclosing mesh; absent when no wave can be stated. | As above |
| `RegionMemoryMetadata.footprint` | The union of every Call's reached addresses below the Function in `logical`. For `total`, group Calls by their enclosing loop/mesh chain. Union each group per buffer. Multiply each group by its waves and by every enclosing loop's trips. Sum the groups. Absent when no wave can be stated. | As above |

- constraints:
  - May-access footprint counting MUST follow the cap stated in
    [visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation).
  - Overlapping ranges into one buffer MUST be unioned before they are counted,
    and the byte count MUST be `ceil(elements * dtype.bit_width / 8)`.
  - Both directions contribute: a load and a store to that level each occupy the
    cache. Only boundaries against that level are counted; a boundary against
    another level MUST NOT enter.
  - A boundary contributes only when its positional entry in
    `MemoryMetadata.operands` reports a nonzero read or write. This MUST use the same Op-evaluator answer as traffic. A boundary with no direction moves nothing. It MUST NOT enter the footprint.
  - One window is defined, and no other: every enclosing loop's induction
    variable is held at its first iteration, over one wave of units. A Call states the footprint of its own boundaries. A Function states the union of every Call below it. The union is deduplicated per buffer before any byte count.
    A `LoopRegion` MUST NOT carry a footprint.
  - A loop start MAY depend on a unit coordinate. Hold such a loop at its first iteration with that start expression. The held loop MUST NOT be constrained to a constant. The reached range then varies with the coordinate.
  - A unit coordinate MUST be eliminated by union over the wave, never by taking
    the largest single unit's count. The two differ whenever units reach
    different addresses.
  - A program declaring more units than the target holds MUST NOT have them all
    counted as concurrent. The wave is the first `wave_units` units in the
    mesh's own linear order, taken through `Mesh.layout`'s strides.
  - The wave count MUST be `ceil(mesh_units(topology) / wave_units)`; a
    partial last wave MUST count as a full wave. Reuse, cache capacity checks,
    and the printed `footprint` field MUST use the single-window `logical` bytes.
  - `Footprint.precision` MUST preserve direction throughout address merging
    and counting; no intermediate Boolean may discard it.
  - A footprint MUST be absent, not empty, when analysis cannot state the wave
    it is taken over.

##### `SSAValue`

```python
class SSAValue:
    expr: Expr
    index: int = 0
```

`SSAValue` identifies one expression's flattened tensor result. Equality and
hashing MUST use expression identity and the flattened leaf index, rather than
structural expression equality. A tensor result uses index zero. Nested tuple
fields use consecutive flattened tensor leaf indices. Scalar event identities
also use index zero, but do not create storage allocations.

`Access.target`, `ReachedAddresses.target`, `BufferAlias.target`, and
`ValueLifetime.value` MUST use this identity. Call boundary indices remain
separate: they identify operands or output boundaries, rather than storage.

##### `ValueLifetime`

```python
class ValueLifetime:
    """One value's residency, as positions in the Function's value order."""

    value: SSAValue
    defined_at: int
    last_used_at: int
    memory_level: str = ""
    bytes: int = 0
    persistent: bool = False
```

LivenessMetadata MUST produce the sole per-leaf interval representation, `ValueLifetime`.
Memory enriches that record with its storage level, byte span and
persistence. Allocation MUST NOT recompute leaf intervals from a second use
index. Reports MUST omit `value` and retain the existing binding, memory_level,
bytes, defined_at, last_used_at and persistent fields.

One ordinary expression event uses its operands and defines its result. A region adds separate binding and exit events. A mesh argument is used before its parameter is defined. A loop initial value is used before its induction and carried parameters are defined. Liveness records one representative loop-body iteration without expanding the trip count. A carried parameter spans entry to exit. A yielded value remains live through the backedge event. Positions
are monotonic across the whole Function, including nested and sibling regions.

Events follow a depth-first visit of the operands, except where a schedule
overwrites data in place. Follow forwards and views that write nothing to find the data a value may hold. A schedule's result is new data, even though it aliases its destination. A retained call's result that shares an
argument's provenance root ([§2.1](#21-shared-iterationscope-and-access)) also
holds that argument's data. A loop's result holds new data and what its body yields. It may also hold its initial data in two cases. In the first case, the body passes its parameter on. In the second case, the loop may run no iterations. Its step or its least extent past its start is not known to be positive. A loop without carries
forwards its body. When the body reads a carried parameter, the parameter holds its initial data and what the body yields for it. Across the backedge, this may be another carry's data. As a schedule's operand, a carried parameter stands for its own data only.

A schedule replaces the data its result operand holds in the storage it writes. Data that a loop's carry took out of that storage does not count. Two writes can have result operands that share a provenance root and elements that meet. Such writes keep their depth-first order. The later write also replaces the earlier write's result. Each read that may touch the data a schedule replaces must take an earlier event. The schedule's own reads and the operands its instruction only writes are exceptions. Where the read reaches that data through a loop's carry, the loop takes the earlier event instead. When the schedule is in that loop's body, the loop takes no earlier event. The schedule then uses that storage after the carry is defined. Allocation then gives the carry storage of its own and copies the data into it ([`AllocationValue`](#allocationvalue)). A read may touch the data when its elements meet the schedule's elements. The read's elements are those that its resolved device access reaches in the provenance root. The schedule's elements are those that its resolved device access writes. For a loop that takes the data, the read's elements are those that the carried value holds. Elements that are not resolved meet anything. Elements in another root meet anything. A retained call's reads are elements that are not resolved. In the innermost scope holding both, the child
holding the earlier one precedes the child holding the schedule. Each such
scope sorts its children by these constraints and their operands at once, ties
in depth-first order, and events then follow that order.

Liveness raises `analysis.Error` naming the constrained schedules of a scope
whose constraints form a cycle. A loop's yield hands its data to the next round
after every child of the body, and the function returns its result after every
child. Liveness also raises for a schedule that replaces data which the yield or the result hands on. The schedule is in that body or in the function. This applies where the elements it writes meet those that the handed value holds in its provenance root. Where a carry took that data first, that loop precedes the schedule instead. Where the order of a scope's calls and regions differs from the depth-first one, `LivenessMetadata.orders` holds it under the scope's owner. Call targets are not part of this order. Lowering emits them in that order ([passes §7.1](./passes.md#71-converthirtotir)). This ordering does not cover four cases. The first is a schedule in a loop that replaces data defined outside the loop and read again in a later round. The second is a schedule in a retained callee that writes a formal. The third is a tuple element taken by an index known only at run time. The fourth is the operands an instruction writes as scratch beside its result.

| Field | How it is computed | Reads the target |
|---|---|---|
| Report `binding` | Render `ValueLifetime.value` through the same printer used for the surrounding IR. Metadata retains the SSAValue identity, without storing a display name. | No |
| `ValueLifetime.memory_level` | Emit one lifetime per material `AllocationValue`, at the level occupied by its projected Type. | No |
| `ValueLifetime.bytes` | Project the Type through every authored split at or coarser than the explicit level's `owner`. Then take its logical bytes. A target-owned or undeclared level remains global. A `ScheduleOp` whose selected instruction has a write-only result occupies `buffers` copies of those bytes; a `READ \| WRITE` result occupies one. | `MemoryHierarchyFacts.explicit_levels[].owner` |
| `ValueLifetime.defined_at` | Definition event on the Function-wide structured SSA timeline. A staged (`buffers > 1`) result starts at the outermost enclosing loop's phi event rather than its own definition event. | No |
| `ValueLifetime.last_used_at` | Greatest ordinary-consumer, region-entry, loop-backedge, or region-exit use event belonging to this result index. Static tuple projections use only the selected result; other tuple uses use every result. | No |
| `ValueLifetime.persistent` | True for parameters and false for body allocations. | No |
| `RegionMemoryMetadata.lifetimes` | Every value residency except a non-material view; present on the entry Function only. | As above |

- constraints:
  - Optional reuse MUST NOT place an ordinary result in a caller-owned
    parameter buffer. Donation is a contract with the caller, not a conclusion
    this family may draw. Semantic same-address relations, including registered
    views, InsertSlice destinations, and retained-call formal-to-actual bindings,
    are address constraints rather than optional donation.

##### `BufferAlias`

```python
class BufferAlias:
    target: SSAValue              # the result whose bytes hold this value
    relation: access.Relation      # result coordinates -> target coordinates
    kind: AliasKind
    writes: bool = False          # a mutation in the composed callee chain
```

| Field | How it is computed | Reads the target |
|---|---|---|
| `BufferAlias.target` | A registered view: the operand `aliased_operand` names. An adopted InsertSlice update window: the destination resolved through region captures. A carried-phi tie: the initial or phi value; an accepted InsertSlice result tie: its destination. | No |
| `BufferAlias.relation` | A registered view: `renaming_relation`. An update window: the proven operand-to-result relation with parameters resolved to values visible at the producer. A tie: identity. | No |
| `BufferAlias.kind` | One `AliasKind` member for each structural link or accepted tie. | No |
| `BufferAlias.writes` | `True` when the composed callee chain mutates the target; otherwise `False`. | No |

- constraints:
  - An op registered with `access.register_alias(Op)` returns the viewed
    input's position or `None` for each Call. `buffer_alias()` returns
    `BufferAlias(SSAValue(buffer), relation, AliasKind.VIEW)`, describing that input's bytes and the
    result-to-input coordinates, and proves the declaration on every call.
    `Slice` and `Reshape` alias input zero and MUST NOT allocate independent
    buffers. `Tuple`, `MeshRegion`, and `LoopRegion` results and static tuple
    projections describe bytes already held by their selected body or carry.
    They MUST have their own AllocationValue intervals and require the operand's
    address through constraints. Static tuple projections are visitor structural rules, like
    `Tuple`, `MeshRegion`, and `LoopRegion`, rather than registered views.
    An InsertSlice result MUST use its destination's address. This includes a parameter destination. The exception is a result whose old value has a later ordinary use.
    Every other ordinary result, including `Transpose`, MUST allocate its own.
    Analysis MUST use operation semantics
    for this distinction rather than infer aliasing from layouts.
  - Reshard MUST alias its source if and only if three conditions hold. Storage is unchanged. The shape is literal. Every logical element retains its complete set of storage instances and addresses. Gmem has one copy per device. Smem has one per device and CTA. Rmem has one per thread. Broadcast replicas MUST be included: removing a
    replica changes the set. [shard §6.1](./shard.md#61-shard-attr-semantics) states which copies each end holds. Two ends that hold the same copies at the same addresses alias. Every other Reshard MUST retain its
    independent buffer.
  - Bitcast MUST alias its source and allocate no independent buffer. Its declared access relation MUST bijectively map result coordinates to source coordinates with the same physical address. Bitcast MUST require the source's address. Bitcast MUST record zero traffic.
  - A value can have a Partial mesh axis finer than its storage owner. `buffer_bytes` of that value MUST count one copy per unit of that mesh axis. Multiple such mesh axes multiply
    their copy counts; an unresolved extent MUST raise `analysis.Error`.
  - Fresh smem allocations MUST reserve a byte span. The span MUST cover the physical layout's addresses within the selected topology unit. It MUST include stride holes and composition offsets. Splits at coarser topologies restrict the local domain;
    finer splits remain in it. Swizzles MAY reserve complete low-bit periods. They preserve the unchanged high bits. This is an arena-capacity bound. It is not a unique-footprint or traffic count.
    Every staged slot MUST reserve that span at the required alignment.
    Invalid negative or unresolved physical address ranges MUST be refused.
  - Registered buffer aliases MUST create an `AllocationValue(lifetime, aliases)`
    with its own SSA interval and the operand's byte span. The alias MUST use the same address. A persistent operand MUST pin its alias there. Aliases MUST NOT
    receive independent `buffer_bytes` or `offsets`. Analysis MUST resolve structural tuple projections to their
    selected element without a coordinate-renaming proof. For other aliases,
    `buffer_alias()` MUST prove that the result-to-operand map is single-valued, injective, and
    contained in the operand's index box. The proof MUST use the call's own unmodified declared access relation via `renaming_relation`. The proof MUST NOT use its folded accesses.
    Failure to prove that declaration MUST raise `analysis.Error`. `Transpose`
    is not registered: it writes a new value, retains its access permutation,
    and receives its own allocation under the ordinary conflict rules.
  - Retained calls MUST translate every callee value in a formal parameter's
    address group into the corresponding actual argument's group. A call result
    returning a parameter or its view MUST use that actual's address and relation. The exception applies when two conditions hold. The composed callee chain writes into the formal, directly through InsertSlice or through a retained call. The old actual has a later ordinary use. Read-only parameter and view returns MUST alias despite later reads of
    the actual, independently of optional reuse of fresh results. InsertSlice
    and call-result aliases remain accepted links in the allocation summary.
  - Allocation MUST return storage for each SSA value and record adopted
    InsertSlice update windows and carried-phi ties in `AllocationValue.aliases`.
    An update window MUST be adopted only together with the InsertSlice's
    destination alias. A material tuple op's leaf MAY reuse the destination's window in analysis. Accepted links retain their SSA leaf identities. Lowering continues
    to reject unsupported material tuple results. An update window's buffer MUST be the destination resolved through region captures. Its relation MUST map producer coordinates to destination coordinates. The destination and SSA parameters MUST be defined in the
    producer's scope or an ancestor, before the producer's definition event.
    Constants and dimension variables are always visible;
    a mesh coordinate is visible only if an enclosing mesh region declares its
    topology. Lowering MUST execute this decision without reading
    the InsertSlice's parameters at the producer.
    A window adopted for a tied carried parameter MUST be recorded on its tie initial. The carried parameter MUST retain its tie alias.
    Ordinary `smem` aliases use their placed offsets. Scope construction MUST
    compute each value's source once
    per view in definition order, using only its operands' source entries.
    Registered views compose their result-to-input relation with the input's
    source. The table MUST store the fully composed `BufferAlias` under an
    `SSAValue` key, with precision in a parallel per-view table. All scopes in
    one definition share those tables; other definitions use independent tables.
    AllocationSummaryMetadata MUST expose the accepted `AllocationValue.aliases` and
    `address_decisions` unchanged. Each address decision retains its holder,
    accepted link, and storage representative in acceptance order. Analysis MUST
    NOT rebuild executable relations, bind loop prefixes for lowering, choose
    external output positions, or publish derived per-leaf aliases.
    ConvertHIRToTIR's StoragePlan interprets these raw decisions. It asks
    registered view ops for their direct-operand relation and InsertSlice for
    its window offsets, independently of the estimator's cached source table.
    An empty execution domain does not erase an accepted window or carry tie.
    Storage partners and optional reuse remain analysis decisions; a new
    executable address is a nested view of its partner, not a composed root map.
    Duplicate outputs and unsupported retained returns preserve lowering's
    refusal text; analysis reports remain available. Lowering retains zero-trip
    loops. Estimation, optional reuse, address assignment and reports continue to use the
    source tables above.
    A retained Call return leaf MUST project the callee's return source.
    Source identity describes data for footprint and traffic, independently of
    allocation ownership. Allocation gives each Call result its own storage
    root unless the callee summary returns a formal root, which forwards to the
    actual leaf. Allocation and source collection MUST share the direct operand leaf selection for structural forwarding. Allocation MUST preserve Call storage boundaries when using those operands and cached view relations.
    MeshRegion results use their body's source. Static tuple projections select a structural element or a material op's flattened result index. Loop results use their corresponding carried phi. Capture parameters use their arguments' sources. Every other source is the value itself and identity.
    `resolve_access` MUST perform one source lookup and one relation composition,
    rather than trace producers per access. Formal-dependent coordinates are retained only when the caller explicitly supplies the definition's formals. Resolving a boundary MUST NOT walk to a definition root to discover them. Boundary parameters MUST be eliminated
    before composition in the operand's coordinate space. View parameters MUST be eliminated when constructing that view's source. The elimination MUST happen in the input's coordinate space. It MUST use the input's type for widening. Source relations retain their defining loop prefix, aligned to the access's enclosing loops. Loops internal to an escaping region MUST be projected out.
  - Every tensor SSA value MUST have its own `AllocationValue`. This includes views, region parameters and results, and static tuple projections. Dynamic projections and their forwards, described below, are the exception. Intervals MUST NOT be merged or extended to include aliases. Every tuple value MUST have one
    `AllocationValue` identified by `lifetime.value` per flattened tensor result. Its interval runs from the tuple's definition to the last use belonging to that result. A static tuple projection uses only the selected result. Whole-tuple operations, region captures, synthetic loop uses, and returns use every result. A dynamic tuple projection MUST conservatively keep every possible tensor leaf resident until the function timeline ends. Liveness MUST record a synthetic use at `timeline_end` on each flattened leaf of the tuple operand. These are the tuple leaves' own uses; their
    structural source links keep the element storage resident. The projection's
    own interval MUST follow only its own uses. The projection and its forwards,
    including capture parameters and region results, MUST have no `AllocationValue` (storage is `None`). They MUST NOT participate in in-place reuse.

##### `AllocationValue`

Allocation collects `AllocationValue(lifetime, aliases)` records.
`AllocationValue.value` is a property returning `lifetime.value`; it MUST NOT
store a second copy of the SSA identity. Each alias is a `BufferAlias` whose
SSA target is looked up in the allocation table. Structural aliases and accepted
ties form value links; there is no separate partition or merged Buffer object.
An allocation root is found by following accepted links. The kind determines
the following order and behavior:

`AliasKind` is an `IntEnum`: VIEW=0, FORWARD=1, CARRY_INIT=2,
CARRY_YIELD=3, INSERT_SLICE=4, WINDOW=5, IN_PLACE=6. Links retain single
members in this order; multiple accepted ties remain separate links.

| Kind | Holder -> target | Acceptance |
|---|---|---|
| VIEW | View result -> storage source | Unconditional proved cached relation, preserving Call storage boundaries |
| FORWARD | Structural result -> selected operand | Unconditional shared leaf selection |
| CARRY_INIT | Phi -> initial | Ordinary tie condition |
| CARRY_YIELD | Yield -> phi | Loop-body condition below |
| INSERT_SLICE | Result -> destination | Ordinary tie condition |
| WINDOW | Update -> result | Window condition below |
| IN_PLACE | Result -> operand | Ordinary tie condition and zero-offset proof |

Structural source links MUST be installed first. Conditional relations MUST be
attempted in the table's order, then in value definition order. For an ordinary
tie, A and B are the roots of the holder and target. Equal roots are accepted.
Otherwise B MUST end no later than A starts. B MUST be neither persistent nor rooted at a borrowed formal. Both levels MUST agree. The size bytes(A) MUST be at most bytes(B). Acceptance adds a value link from A to B. Address decisions MUST
NOT change accepted links.

Every tie MUST read the same occupancy: a value's own lifetime plus translated
callee windows mapped to that value. Start is the earliest definition and end
the latest use among those windows; lifetimes themselves MUST NOT be changed.
Without Calls this occupancy is the value's own lifetime.
A root's extent MUST be the maximum bytes of every member's occupancy, including mapped return-producer windows. Its own lifetime.bytes remains the size of its projected Type. At equal endpoints,
a mapped callee window has no caller-local access proof. Only members of the formal covered by a callee IN_PLACE capability may borrow that proof. The check repeats in actual storage-root coordinates. Other mapped windows MUST end strictly
before the candidate starts. Global interference MUST compare individual
windows, never their occupancy envelope.

At equal endpoints, A's first defining operation MUST have a proved zero-offset `operand_to_result_relation`. This applies to every B member it reads at that event. The
InsertSlice destination identity branch is a valid proof. A matmul reading its
carry is not pointwise and MUST NOT acquire a carry-yield tie on that evidence.
IN_PLACE candidates MUST be limited to proved zero-offset, non-persistent operands at the same level. Each such operand's own last use MUST be no later than the result's definition. Nonzero-offset IN_PLACE candidates MUST NOT be registered.
IN_PLACE MUST use source-resolved
`operand_to_result_relation`, grouped by `Access.target`, so the proof is in
storage-root coordinates after VIEW/FORWARD composition. A Slice's local
identity MUST NOT turn a shifted backing-buffer read into zero-offset reuse.

At equal endpoints, and for WINDOW reads at its start, proof MUST compare isl relations in B's root coordinates. Each operation's read relation MUST equal its value's relation at the candidate output landing. The comparison MUST be restricted to that input's iteration domain. Merely belonging to one storage root is insufficient.
The proved InsertSlice destination identity includes its forwarded unchanged
coordinates, not just the update's write-traffic coordinates.

CARRY_YIELD for loop L uses content within one representative body iteration.
Between different storage roots, A's material producer (including a material
Call result) MUST be defined inside L's body. A body-local view or forward of
an outside root does not qualify. A candidate without any body-local member
definition MUST be rejected. Rejection uses the ordinary backedge copy.
Completion for candidates already sharing one root is unchanged.
Its B endpoint is the last use of any B member strictly between L's phi and
backedge events; inner-loop synthetic uses count. L's own backedge, exit, and
later uses do not read the old phi content. Its A start is the earliest member
definition inside L's body. The resulting end(B) MUST be at most start(A), with
the same zero-offset proof at equality. LivenessMetadata intervals themselves MUST
remain unchanged.

Candidates MUST be recorded as accepted when their holder and target already reach one root. This applies to CARRY_INIT, CARRY_YIELD, and INSERT_SLICE candidates, after accepting links. It includes the case where another accepted edge established that storage identity. This
completion MUST NOT change roots or addresses. Both links remain in the
summary, with their distinct partners and kinds. WINDOW is excluded from this completion.

WINDOW is the exception whose holder precedes its target. It MUST be accepted only when two conditions hold. The same InsertSlice's result and destination already share B. The update-to-result relation proves a window visible at the producer. A, the update root, MUST differ from B. A MUST be neither persistent nor a formal boundary. A MUST occupy the same level. Every member of A MUST have its last use no later than the InsertSlice result's definition. The update MUST fit in the result. No B member may be used strictly
between A's start and that definition; at A's start, any B read requires the
zero-offset proof above. Acceptance links A to B with the window relation. Window
acceptance MUST follow VIEW, FORWARD, and accepted CARRY_INIT edges from the update to its storage root. It MUST reverse-compose those coordinate relations.
The resulting producer-to-destination map MUST cover the producer's complete
coordinate box. It MUST use only parameters visible at that producer. Otherwise the WINDOW MUST be rejected. The map MUST be a same-rank translation supported by lowering. In it, destination axis k equals producer axis k plus a producer-visible start. Rank-changing maps and non-translations MUST be rejected, leaving an
independent update for lowering to copy at InsertSlice. Tuple links remain
indexed by flat leaf. When CARRY_YIELD and INSERT_SLICE both accept one value,
both links MUST remain in the summary. Rejected ties MUST contribute no accepted
link; lowering copies the corresponding carry or slice.

A callee root containing a formal or the returned storage root is a boundary. It MUST have no local address variable or published offsets. A read-only zero-offset operand proof MAY declare an IN_PLACE capability when formal occupancy ends no later than output occupancy starts. Such proofs include formal operands and VIEW/FORWARD paths to the return root. The capability holds an output index, a formal input leaf and a coordinate relation. The call site MUST register and judge these capabilities with the same IN_PLACE rules as ordinary operations. A later use of the actual MUST prevent reuse. These proofs MUST NOT create local
IN_PLACE aliases, link formal storage to a result, or publish metadata.

AllocationConstraintVisitor.run MUST return AllocationSummaryMetadata with
values, IN_PLACE capabilities and accepted address decisions. Its private
occupancy evidence retains translated windows; it MUST NOT retain a second
call graph. Reading a called summary through get_metadata(..., ctx=ctx) uses
the attached Scope and caches the bound result in its snapshot. Lifecycle
region events remain owned by LivenessMetadata. Time translation uses
`shifted(record, origin)`. Windows rooted at a formal MUST name the actual SSA leaf. Windows rooted at a return MUST name the Call result leaf. Internal SSA values
retain their identities and share one address across all translated windows.
The caller MUST NOT mutate AllocationValue aliases in a callee summary. Own
windows MUST reference the AllocationValue's lifetime object rather than copy it.
Only the entry solves address assignment, once, after collecting its Call leaves. Non-boundary callee offsets MUST be published from that solution. Repeated region windows contribute their
maximum occupancy without traversing the callee body again.

##### `MemoryLevelPeak`

Capacity is settled against authored definition order. That order fixes the
lifetime of every linked value before any lifetime is measured. A whole-Function
peak at `gmem` or `smem` uses exact polyhedral access relations to place
addressable roots. The Call results report their concrete offsets. A `MeshRegion` peak scans root
occupancy over its entry-to-exit window instead. It counts each root once. At
addressable levels, it uses the union of live address intervals. It covers only
the levels that the region holds. In the target's ordered topology hierarchy, a
region's outermost topology MUST be the target-declared owner or an enclosing
topology. A region strictly inside the owner is not eligible to report that level.
A target-owned level makes no `MeshRegion` eligible. Eligibility does not decide
attribution. Every eligible region whose mesh contains the authored holder mesh
of a value reports that value. An enclosing region therefore also counts values
that its nested regions hold. A value that declares no holder belongs to no
region. Analysis does not infer a holder from its layout or storage name.
Analysis does not solve addresses for `rmem`. The Function reports the maximum
of all stage peaks and of a whole-Function live-byte scan of values that no
`MeshRegion` claims.

At each event, `rmem` counts a root once, with its greatest live member span.
Window occupancy and persistent bytes at `gmem` and `smem` MUST count the union
of live physical address intervals. A repeated callee region reports the maximum
across visits.

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
| `MemoryLevelPeak.peak_bytes` | On a Function, `peak_bytes` is the address high-water mark of the first feasible global `gmem` or `smem` address assignment. On a Function, the `rmem` peak is the maximum of every `MeshRegion` stage peak. The maximum also includes the whole-Function live-byte peak of values no mesh region claims. On a `MeshRegion`, the peak is the greatest root occupancy in its inclusive entry-to-exit event window. Unsplit tensor values defined in its body or child regions contribute while their lifetimes intersect the window. No region-versus-owner filter applies to them. Split values retain owner eligibility and holder-mesh selection. Target-owned storage (`gmem`) never contributes to a region. | `MemoryHierarchyFacts.explicit_levels[].owner` for a `MeshRegion`; otherwise no. |
| `MemoryLevelPeak.persistent_bytes` | Union of persistent address intervals at `gmem`/`smem`; greatest persistent member span per root at other levels. | No |
| `MemoryLevelPeak.capacity_bytes` | Capacity of the matching explicit level, or `None` when unknown. | `MemoryHierarchyFacts.explicit_levels[].capacity_bytes` |
| `RegionMemoryMetadata.peaks` | One peak per occupied or moved storage level. Only the entry Function carries peaks. A callee Function states none. | As above |
| `RegionMemoryMetadata.solver_status` | `"feasible"` after the whole-Function address assignment of the entry settles, including when no value is addressable. `"not_run"` on a callee Function, which does not place. | No |
| `RegionMemoryMetadata.errors` | Non-fatal address-capacity failures and cache-capacity failures proved by exact or lower-bound counts. | `MemoryHierarchyFacts` |
| `RegionMemoryMetadata.advisories` | Cache-capacity findings whose upper-bound or unknown counts do not prove excess. | `MemoryHierarchyFacts` |

- constraints:
  - Analysis MUST attach `RegionMemoryMetadata` to each reachable `Function` and
    to each `MeshRegion` with an eligible live value. The contributions are those
    stated for `MemoryLevelPeak.peak_bytes`. A mesh-region record carries only
    its topology names and per-level peaks. Whole-Function traffic, footprints,
    reuse windows, lifetimes, errors, and advisories stay on the Function.
  - Every explicit level considered for a mesh-region peak MUST state an owner
    in `MemoryHierarchyFacts`. A missing owner is an error. It does not make the
    level shared by every region.
  - A split value MUST declare its holder mesh to belong to an eligible
    `MeshRegion`. Analysis MUST NOT infer a holder for it.
  - An access relation that keeps a parameter with a stated finite range is
    exact. Such a relation MAY prove overlap. A widened relation MUST NOT prove
    overlap. A relation with an unbounded parameter MUST NOT prove overlap.
  - Analysis MUST settle address assignment for the addressable levels `gmem` and
    `smem` only. Analysis MUST settle address assignment once per capacity domain
    that holds a buffer. A capacity domain is the whole target for a level that
    the whole target owns, and one domain per owning unit otherwise. Two buffers
    in one domain are never live in the same bytes at once. Residency at another
    level MUST NOT make a program unplaceable. Analysis MUST fail on a level
    owned per unit of a topology other than the one analysed. Analysis MUST NOT
    assume that level. Domains that hold the same buffers are one question.
    Analysis decides it once.
  - Carry, InsertSlice, window and in-place sharing MUST follow
    the pre-solve value links and acceptance conditions above. Simultaneously
    live values in overlapping address intervals MUST belong to the same root.
    Otherwise analysis MUST raise `analysis.Error`. The error MUST identify both
    bindings, intervals, offsets, and level.
  - Every placed offset MUST be aligned to the greater of 16 bytes and the
    result element width. The copies of a staged result MUST be contiguous. Copy
    `k` starts at the solved block offset plus `k * buffer_bytes`.
  - If mandatory aliases, alignment, and lifetime interference have no feasible
    joint address assignment, analysis MUST raise `analysis.Error`. Analysis MUST
    NOT silently discard a required alias.
  - A domain that cannot be expressed or does not settle in time MUST raise
    `analysis.Error` and leave no record. The solver MUST stop at its first
    feasible assignment rather than prove a minimum. Capacity MUST NOT restrict
    the address space.
  - A seed is a complete candidate address assignment. Analysis builds the seed
    before solving. A seed holds a starting byte address for every placed value,
    and the peak byte extent that the assignment reaches. A seed is not an input
    and not a random seed. The solver takes the addresses of the seed as a hint.
    It takes the peak of the seed as an upper bound on the search. With no seed,
    the solver solves the same model with every constraint kept. It has no hints
    and no seed-derived upper bound.
  - A seed that hints or bounds the search MUST meet every address constraint of
    the solver. The constraints are: one address per root, alignment, disjoint
    addresses for interfering values, pinned persistent addresses, and the
    address limit. A seed that does not meet them MUST NOT bound the search. A
    failure to construct a seed proves nothing about the model.
  - If a solved explicit-level peak exceeds capacity, analysis MUST add a
    non-fatal `errors` entry and MUST preserve the complete result. The call MUST
    NOT fail.

##### Cache occupancy

A cache holds data because the program reads it again. What was touched in
between decides whether the cache still holds it. This analysis states one row
per buffer that is read again. The row names the two axes that a second read can
come from. One axis is the loop whose next iteration reads the buffer. The other
axis is the mesh axis whose units read it at once. A buffer whose units partially
overlap also states a row, even when neither exact reuse axis exists. Its
savings are unknown. A buffer with none of these reuse relations states no row.

| Field | How it is computed | Reads the target |
|---|---|---|
| `ReuseWindow.buffer` | The source buffer identity, rendered by the expression printer. | No |
| `ReuseWindow.time` | The outermost enclosing loop whose different iterations reach the same addresses, named by its induction variable; empty when no loop supplies a second read. | No |
| `ReuseWindow.space` | The mesh axes whose different coordinates reach the same or partially overlapping addresses; empty when no mesh axis supplies a second read. | No |
| `ReuseWindow.holds_bytes` | Unique bytes of every buffer the whole wave touches while this buffer must remain resident. | `MemoryHierarchyFacts` selects the backed level. [target §11](./target.md#11-target-facts-projection) supplies `TopologyFacts`. |
| `ReuseWindow.reuse_bytes` | `(time trips * space units - 1)` times this buffer's unique bytes in the window. An absent axis contributes one. Unknown (`None`, JSON `null`) for partial overlap whose savings are not modelled. | As above |
| `ReuseWindow.fits` | True exactly when `holds_bytes` is less than the cache capacity. | `MemoryHierarchyFacts.implicit_levels[]` |
| `ReuseWindow.precision` | The precision of the footprint that supplies `holds_bytes`. It does not mix in the trip upper bound used for `reuse_bytes`. A partial-overlap row has `unknown` precision instead. | No |
| `RegionMemoryMetadata.reuse_windows` | One row for every buffer with a time or space reuse axis and nonzero savings, or with partial overlap and unknown savings. | As above |

- constraints:
  - Partial-overlap detection MUST run only when time and space reuse supply no
    axis. Such a row MUST retain its holds bytes and report `reuse=unknown` and
    `precision=unknown`. JSON MUST use `null`
    for its reuse bytes. Existing exact reuse rows MUST remain unchanged.
  - An unknown-precision row that exceeds capacity is an advisory. The row
    suppresses the first-iteration working-set finding only when their precision
    reliability matches. That finding MAY coexist with the advisory of the row.
  - The stated bytes count everything the whole wave touches while that buffer
    must stay resident. They include every buffer, not only the one read again.
    The other buffers evict it.
  - Cache lines are not modelled. Analysis packs reached addresses by element
    width. Two accesses to neighbouring words of one hardware cache line
    therefore remain two distinct touches here.
  - A read by another unit of the same wave counts. On a target whose units
    share one cache, data that several units read at once is the common case. A
    model that counts only one returning unit states no reuse for a schedule that
    gives each unit one output tile.
  - The parallel reuse axis is a TileFoundry extension. The reuse-distance
    model that it follows (Falcon, PLDI 2024) states a single-threaded program.
    A union of addresses across concurrent units has no published model. It has
    no hardware validation here.
  - A mesh axis supplies reuse exactly when one unit reaches the same addresses
    as the wave union along that axis. The spatial repeat count is how many
    units in the wave map onto those same addresses. The space comparison reads
    the wave relation from the device view and the one-unit relation from the
    unit view. The time axis reads the raw narrow relation. It retains mesh
    parameters and applies no wave limit. `holds_bytes` reads the narrow
    relation. It limits that relation to the wave and projects out mesh
    parameters.
  - Under nested mesh regions, a reuse row states its space mesh axis against the
    innermost enclosing mesh. Outer execution meshes are not composed into
    that mesh axis label.
  - The window is the loop axis when one exists. Otherwise the window is the mesh
    axis alone. Units that read at once are already inside one iteration
    of the loop that carries the later read.
  - A time window is a loop-granularity over-approximation. A true reuse
    distance starts at the previous touch of the same cache line. This window
    covers a whole iteration of the carrying loop. Precision describes the
    counted window, not hardware cache misses or true byte reuse distance.
  - Analysis MUST NOT sum rows. The window of one row lies inside the window of
    another row whenever its axis is nested inside. A row that fits therefore
    implies that the rows nested in it fit.
  - Analysis MUST order rows by `reuse_bytes`, largest first. Equal amounts keep
    derivation order.
  - Data read once states no row. Its bytes still enter every row whose window
    contains it.
  - A window at or above capacity MUST add one non-fatal diagnosis, however many
    buffer rows share it. The call MUST NOT fail. `EXACT` and
    `LOWER_BOUND` conclusions belong to `errors`. Their true holds are at least
    the stated capacity-exceeding amount. `UPPER_BOUND` and `UNKNOWN`
    conclusions belong to `advisories`. Exceeding capacity is not proved for
    them. The same directional rule applies to the Function working set.
  - The capacity is stated per one unit of the `scope` of the cache. This
    analysis compares one wave against one unit. It does not model a deployment
    that spreads one wave across several units.
  - Per-unit control flow is out of scope. HIR states no conditional region. Two
    units differ only by the iteration domain that a coordinate gives them.
  - This is the capacity judgement of an idealised fully associative LRU cache.
    This analysis does not model miss counts, miss rates, or hardware
    replacement policy.

Cache level and capacity are Target facts. They are not part of
`RegionMemoryMetadata`. They appear in this report only when they decide a
cache-capacity finding.

##### Target facts

The family reads the target hierarchy and the physical unit counts.
[target §11](./target.md#11-target-facts-projection) states them once. Both
`memory` and `performance` consume them.

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
        scope: attribute; The topology the capacity is stated per.
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
        scope: attribute; The topology the capacity is stated per.
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
  - A GPU projection MUST cover the explicit levels a program can name. It MUST
    cover the caches that traffic passes through. It MUST state that L1 caches L2 and that L2 caches global memory. It MUST
    state that L1 divides one physical block with shared memory. A target with no sharing MUST express that with no sharing
    edge.
  - An implicit level MUST NOT receive a fixed capacity where its usable
    capacity depends on the program. Analysis MUST derive that capacity from the
    sharing edge and the measured peak of the sharing level.
  - Every explicit level MUST carry an `owner` that the Target supplies. It MUST
    be a declared Target topology or `target`. An implicit cache MUST NOT carry
    an owner.
  - Analysis MUST NOT infer memory ownership from a storage level's name or
    capacity scope.

##### Printed and JSON surface

Requesting memory adds one Function line and one line per finding:

```text
memory traffic=<memory-level>:r<bytes>/w<bytes>@logical,r<bytes>/w<bytes>@total,r<bytes>/w<bytes>@<topology>[,...] footprint=<buffer>:<bytes>[;<buffer>:<bytes>] footprint-precision=<exact|upper_bound|lower_bound|unknown> peak=<level>:<bytes>[,...] persistent=<level>:<bytes>[,...]
  buffer=<buffer> holds=<bytes> time=<loop|none> space=<mesh-axis|none> reuse=<bytes> fits=<yes|no>
  error="<text>"
  advisory="<text>"
```

A Function's memory record counts that Function and everything it reaches. A
reader MUST NOT sum the per-Function lines of one report.

The `<bytes>` form selects the largest of `B`, `KiB`, `MiB`, and `GiB` whose unit
the value reaches. Adjacent units differ by a factor of 1024. `B` is an integer.
Every larger unit has two decimal places. Zero is `0`. Every human-readable byte
count uses this one form, including error text. JSON keeps raw byte integers.

Each buffer, error, and advisory is its own line, indented beneath the `memory`
family line. The text form quotes and escapes errors and advisories. Every
measured Call receives a `memory` annotation. The text form emits `operands`
only when asked for ([cli Analyze](./cli.md#analyze)):

```text
memory traffic=<memory-level>:r<bytes>/w<bytes>@logical,r<bytes>/w<bytes>@total,r<bytes>/w<bytes>@<topology>[,...] footprint=<buffer>:<bytes>[;<buffer>:<bytes>] footprint-precision=<exact|upper_bound|lower_bound|unknown> [operands=<position>:r<bytes>/w<bytes>[;<position>:...]]
```

In the printed `footprint` field, the value label of a buffer may itself contain
`:` (for example, `v0:57:1.00KiB`). The byte count is the formatted value after
the last colon.
A root that is a declared parameter keeps its own name. Otherwise use the
first capturing parameter, in program order, that resolves to that root.
Otherwise use the existing value label. Duplicate-label suffixes remain
unchanged.

Missing optional conclusions omit their whole printed field. Call and Function
JSON projections are both under `memory`. The Function's full projection is
under `function_records.memory`:

```text
{"topologies": [<topology>, ...],
 "traffic": {"storage": {<memory-level>: <spread>, ...},
             "communication": {<topology>: <spread>, ...}},
 "footprint": {"buffers": {<buffer>: {<memory-level>: <spread>}, ...},
               "precision": <"exact"|"upper_bound"|"lower_bound"|"unknown">} | null,
 "reuse_windows": [{"buffer": <name>, "time": <loop|"">,
                     "space": <mesh-axis|"">, "holds_bytes": <int>,
                     "reuse_bytes": <int>, "fits": <bool>,
                     "precision": <"exact"|"upper_bound"|"lower_bound"|"unknown">}, ...],
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
  - Text and JSON MUST project analysis conclusions from these records.
  - A missing footprint MUST be JSON `null`. It MUST NOT be an empty
    `Footprint`.

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
| `compute_ns` | For each recorded dtype with a published rate, round `flops * 1e9 / rate` up to ns and sum the dtype times. A Function uses its summed flops, not a sum of per-Call times. | `ThroughputFacts.peak_flops_per_second`, asked for the Function's active units |
| `memory_ns` | Sum footprint `total` bytes at `bandwidth_level`, multiply by `1e9 / memory_bandwidth_bytes_per_second`, and round up to ns. The value is zero when no bandwidth is published or no bytes are counted. Reads and writes are one count. When the footprint is absent or counts another level, use traffic `total` reads plus writes at `bandwidth_level`. A Function uses its accumulated footprint or traffic, not a sum of per-Call times. | `ThroughputFacts.bandwidth_level` and `memory_bandwidth_bytes_per_second`, asked for the Function's active units |
| `ideal_ns` | Maximum of `compute_ns` and `memory_ns`. One ns when the occurrence records nonzero flops or nonzero priced `bandwidth_level` bytes and neither published rate yields a bound. Otherwise zero. Bytes at any other level are stated. They earn no bound, because no rate is published for that level. | Through the two times |
| `bound_by` | `none` for no bound. That includes an occurrence whose only movement is at a level with no published bandwidth. `balanced` for equal nonzero times. `memory` when memory is greater. `compute` when compute is greater. `unrated` for the one-ns bound owed by work that roofline prices and whose rate is missing. | Through the two times |

The family reads this target projection:

```python
class ThroughputFacts:
    """Carry the rates a bound divides the whole program's work by.

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

Roofline asks for `ThroughputFacts` with the query
`(topology, active, inflight)`. Here `active` is the most units that the
meshes of the Function run at that topology. `inflight` is the bytes in flight of a
Call, as performance states them. The Function uses the most that any of its
own Calls keeps, so its bound stays a lower bound. A call into another
definition is not an access and keeps nothing in flight. That definition prices
its own. The rates are those of the units the program
runs. A query of no topology is the whole deployment. A separate projection states
what one unit gets through. `performance` reads it, and `roofline` does not.
Work that is not floating point at all has its own published instruction
throughput. It has no dtype to be filed under.

```python
class PerformanceServiceFacts:
    """Carry everything one unit gets through, by the kind of work it is asked for.

    Attributes:
        unit_flops: attribute; One unit's floating-point rate, by dtype.
        unit_ops: attribute; One unit's rate for each named service kind.
        unit_bandwidth: attribute; One unit's rate for each memory level it moves at.
        topology: attribute; Topology these throughputs describe.
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
`special`. The names are this project's, not any vendor's. The provenance of
each name MUST state which published row the name derives from. A service is
work the machine does, not movement it makes. A bandwidth prices bytes. Never
use an instruction rate to price bytes.

Requesting roofline adds this verdict. It also adds as `totals` the two
quantities that the bound divides: the summed `flops` and the bandwidth-level
bytes. Roofline promotes nothing else that its dependencies wrote. Asking for
`memory` states those ([§1.2.2](#122-memory)).

Every measured Call receives this annotation. `compute_ns` and `memory_ns` are
the two numbers behind the verdict. They are in JSON, not on the line:

```text
roofline ideal-ns=<int> bound-by=<resource>
```

Reported Call and Function records use the same projection under their
`roofline` keys:

```text
{"compute_ns": <int>, "memory_ns": <int>, "ideal_ns": <int>,
 "bound_by": <resource>}
```

When a request names roofline but not its dependencies, `totals`
carries only exact `flops`, `traffic`, and `communication` sums. Dependency
records remain on the semantic result. They do not enter `function_records`,
Call annotations, or Call report rows. A request for a dependency on its own
selects its full form, as that family's section defines.

- constraints:
  - `ThroughputFacts.peak_for` MUST return `None` for a dtype with no published
    rate. Analysis MUST NOT substitute an assumed rate.
  - `PerformanceServiceFacts.flops`, `ops` and `bandwidth` MUST return `None`
    for an unstated dtype, kind or level. Performance MUST reject non-zero work
    of that dtype, kind or level. It MUST NOT substitute the whole-device rate
    or the rate of another kind.
  - Roofline MUST price footprint `total` at `bandwidth_level` without applying
    any further repetition factor. A footprint that is absent MUST fall back to traffic `total` reads plus
    writes at that level. A footprint that counts another level MUST fall back
    the same way. A
    footprint with omitted boundaries MUST still be used as its stated lower
    bound. No other memory level may enter the bandwidth calculation.
  - Footprint pricing assumes ideal reuse within one wave and repeats bytes
    across windows. Counting a partial last wave as full is a pessimistic
    approximation. Deduplication within one wave is optimistic. Below the
    parallel topology of the device, analysis counts the windows of each
    parent unit separately. This overestimates bytes when parents share data.
  - Performance local duration MUST divide one unit's share of
    `ComputeCostMetadata.flops` by `unit_flops`. It MUST divide one unit's share
    of `service` by `unit_ops`. It MUST divide one unit's share of the bytes at
    each storage level by `unit_bandwidth`. All three divisions are at the topology
    that performance was asked about, for the active units of the Function.
    Performance asks for `PerformanceServiceFacts` with
    `(topology, active, inflight)`. It caps `active` at the
    `max_physical_units` of the topology. `inflight` is the bytes that one
    parallel unit keeps in flight for the occurrence. An asynchronous copy
    (`category="async"`, such as TMA or `cp.async`) keeps one unit's bytes at
    `bandwidth_level` times the slots that its buffer declares. Any other access
    keeps 16 bytes per thread of the unit. When the program states no thread
    count, the count is as many threads as the finest topology allows. At
    `bandwidth_level`, one unit's bytes are the footprint `logical` bytes of the
    occurrence, divided among the units that its window holds, and rounded up.
    Reads and writes are one count. The units that the window holds are its
    executing units, capped at `active`. The roofline reads the same footprint.
    Without a footprint, the share is the `bandwidth_level` entry of
    `MemoryMetadata.traffic.storage`. A window that every unit writes alike then
    costs the window over the units. Disjoint shards cost each
    unit's own shard. Compute and movement overlap within one occurrence.
    Its duration is the greater of the two sides, not their sum.
  - Traffic at a level with no stated one-unit bandwidth MUST remain visible in
    `MemoryMetadata` and MUST NOT enter a duration. An instruction throughput
    that stands in for a bandwidth prices a move as arithmetic.
  - Moved bytes and timeable work are different questions. A quantity that has
    a rate decides the second: a nonzero share of `flops` or of `service`, or
    nonzero `MemoryMetadata.traffic.storage` at `bandwidth_level`. An occurrence
    with none of them MUST take zero time. Performance MUST NOT require such an
    occurrence to carry an execution mesh. It MUST still record its movement at
    any other level. It is untimed, not absent. Performance MUST refuse work of a
    dtype or kind with no stated one-unit throughput. It MUST NOT price that work
    at zero. A zero price would leave a hole inside a number that the reader
    takes as whole.
  - Analysis MUST NOT record a predicate as floating-point work. A comparison
    records `predicate` service. A selection records `select`. Neither has a
    FLOP count. Neither MAY be priced at zero for want of a count.
  - `flops` MUST stay the measure of arithmetic that really is a multiply or an
    add. An operation that the machine answers on a separate unit at a separate
    published rate belongs in `service`. It MUST NOT also appear in `flops`. One
    operation is one quantity. A roofline that counts a special-function
    result as one FLOP states a bound that the unit cannot meet.
  - Rate-to-duration divisions MUST use exact integer ceiling division. They
    MUST NOT pass through floating-point arithmetic.
  - Analysis MUST attach roofline records to reached `Call`s and Functions.
  - Roofline MUST read the recorded work rather than evaluate the program a
    second time.
  - A recorded compute dtype that is not a `DType` name MUST raise
    `analysis.Error`.
  - Roofline reads only the `compute-cost` and `memory` records. A
    roofline-only rendering reports only what roofline and those dependencies
    wrote under the bounded `totals` rule above.

#### 1.2.4 `performance`

`performance` places occurrences that have a compute cost on a CTA-local nominal
timeline. It holds the buffers they keep live to the levels this model addresses.
It scales the root timeline by a fixed physical parallel capacity. Each record it
owns is named for the prediction it carries, not for the selector. The interval
then stays one nested value with one meaning wherever it appears.

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
        wave: attribute; Units the Function's meshes run, and the topology's physical units.
    """

    timeline: TimelineMetadata
    wave: tuple[int, int]
```

`TimelineMetadata` is a value, not a record. It MUST NOT be attached to a Call or
a Function on its own. The record that carries it states what it spans.

Occurrence fields are:

| Field | How it is computed | Reads the target |
|---|---|---|
| `start_ns` | The start of one occurrence on the authored-order local timeline. It is the end of the preceding timed occurrence in authored order. | No |
| `end_ns` | End of that occurrence's first execution. | No |
| `trips` | One outside a loop. Within a loop, the enclosing loop trip count that the interval represents. | No |
| `stride_ns` | Zero outside a loop. Within a loop, the makespan of one body execution. | No |

Function summary fields are:

| Field | How it is computed | Reads the target |
|---|---|---|
| `timeline` | `[0, local makespan * waves)`. The local makespan is the end of the CTA-local timeline, or zero with no work. On the entry Function, roofline raises it to the roofline `ideal_ns` when roofline ran. `waves` is `ceil(N / P)`. `N` is the most units the Function's meshes run at the topology selected by `TopologyFacts.parallel_topology`. `P` is the `max_physical_units` of that topology. The duration of the interval is the prediction. `waves` is not a field. | `TopologyFacts` |

Occurrence intervals remain CTA-local. They are not copied once per wave. The
capacity `P` is the machine-side unit count. The mapping of one CTA to one
SM is compiler policy. [target §11](./target.md#11-target-facts-projection) states
the aggregate once, because `memory` reads it too.

Requesting performance adds this Function verdict to the summary:

```text
performance root=<Module>::<Function> predicted-ns=<int> wave=<active>/<physical>
```

`root` is the report's own identity. Inspection composes it from the module and
function the report already states. It is not a field of `PerformanceSummaryMetadata`.
It does not appear in the JSON projection of that record. `predicted-ns` is the
duration of the summary's own envelope. It is not a second measurement. `wave` is
the summary's `wave`. `active` is the most units the Function's meshes run at the
selected topology and the coarser ones. It is 1 when the Function states no
mesh. `physical` is the `TopologyLimits.max_physical_units` of that topology. Forty
units on a 132-unit topology read `wave=40/132`. 8192 units read `wave=8192/132`.
It states where the program's units reach. It is not an occupancy guarantee.

Every Call with a modeled duration receives this annotation. It is one interval
whether or not the Call repeats. A single trip states its own bounds. A repeated
occurrence states its bounds offset by the trip index, with the trip count as a
suffix. The trip count is not a second key. A reader who derives the later
intervals reads it off the interval, where it is a coefficient:

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
           "wave": [<active>, <physical>]}
```

The duration of the summary envelope is a deterministic comparison estimate. It is
not a runtime prediction. It is not a guarantee about lowering. It excludes launch
overhead, occupancy, utilization, traffic volume, and other effects this family
does not model.

- constraints:
  - A primitive Call reachable in the authored HIR is eligible. Its execution
    region is the enclosing `MeshRegion`, independent of the result layout. An
    occurrence is structural to this model when it has no nonzero share of
    `flops`, none of `service` and no nonzero `MemoryMetadata.traffic.storage` at
    `bandwidth_level`. A structural occurrence MUST receive no record. An empty
    interval reads as a measurement, not as the absence of one. Movement at
    another level MUST NOT be dropped from `MemoryMetadata` because of this.
    Structural means nothing to time, not nothing done.
  - The global total for an occurrence is its per-unit quantity multiplied by
    the number of units in the enclosing execution scope. This multiplier
    counts copied or replicated work. Every unit performs an unsharded operation
    inside a scope independently. The operation therefore contributes once
    per unit to the total. The per-unit quantity already includes work
    projected through finer topologies.
  - A `MeshRegion` evaluates its boundary arguments outside the region and
    executes its body once per enclosing unit. The model charges argument work at
    its defining site. It charges body work per enclosing unit.
  - The Mesh layout of an occurrence ([shard §5](./shard.md#5-mesh)) gives the
    units it runs on. The model counts them at the selected topology and caps the
    count at `active` for the per-unit share. A `Broadcast` shard attribute does not
    change them. Attributes describe distribution. The Mesh describes which
    units execute it.
  - Every primitive occurrence has one fixed duration. The model lays occurrences
    end to end in authored order on one local timeline. An SSA consumer therefore
    starts no earlier than its producers end. No two occurrences overlap.
  - A Call's scheduled duration MUST multiply every enclosing authored loop's
    trip count, including loops in which the Call is invariant. Work explicitly
    defined outside a loop MUST keep its defining scope's repetition count.
    Mesh units MUST NOT multiply this duration: units execute in parallel.
  - A `LoopRegion` MUST be represented as one structured performance node. The
    model solves its body once, from the time the loop itself begins, not from
    zero. A body occurrence therefore reports `[start_ns, end_ns)` as the interval
    it actually runs in. A reader does not have to offset it. `stride_ns` is the
    local makespan of that body. The t-th execution of a body occurrence with first
    interval `[start_ns, end_ns)` is
    `[start_ns + t*stride_ns, end_ns + t*stride_ns)`, for `0 <= t < trips`.
    The loop spans `trips * stride_ns`. A consumer of its yield MUST wait for
    that full span. Loop-invariant values remain single occurrences outside the loop.
  - An ordering MUST NOT be inferred from an allocation. Which values share
    bytes is a plan's decision. No plan has been made.
  - Occurrences MUST be laid out in definition-relative order. Reordering
    independent work is a later decision, not an analysis's. The reported time
    is the time of the program as written. On a `Function`, the summary's
    `timeline` MUST start at zero and span the whole local plan, scaled by
    `waves`. There MUST be no second field restating the local makespan or the
    scaled estimate. There MUST be no field restating how the layout was
    reached. The layout is exact for the model it states.
  - `max_physical_units` MUST NOT enter the CTA-local layout. It is not a
    program rewrite. It enters one-unit rates only as the cap on `active`.
  - The buffers a plan keeps live MUST have been placed by `memory` before a
    time is reported for it. A successful dependency records
    `RegionMemoryMetadata.solver_status="feasible"`. An address assignment that failed
    never reaches performance, because `memory` refuses it.
    Capacity therefore changes whether there is an answer, never which answer.
    Two capacities that both admit an address assignment MUST produce the same intervals.

## 2. Composed analysis

`tilefoundry.analysis.check_program` is the shared, reusable gate before an
analysis runs.

```python
def check_program(
    module: "core.Module",
    function: "core.Function",
    *,
    topology: str | None = None,
    budget: int = NODE_BUDGET,
    analyzers: tuple["target.Analyzer", ...] = (),
    dims: "Mapping[str, int] | None" = None,
) -> tuple["core.Module", "core.Function"]: ...


class CheckContext(valuation.Context):
    """What every input check reads: the program, the machine, and the topology.

    The context is bound to the derived Function the analyses will read.
    Nothing a check reads is a derived analysis record.
    """

    module: "core.Module"
    function: "core.Function"
    target: "target.Target"
    topology: str | None

```

- constraints:
  - The operation MUST infer types over the full reachable Function graph. It
    MUST validate the caller/callee execution context. It MUST NOT run an analysis.
    It MUST NOT attach derived Metadata to the authored IR. Each derived body it
    returns is its own copy. The operation infers its types under the
    program's Target and arguments and stores them on it. Analyses therefore read the
    `Expr.type` of that view, not a stamp of a shared child body.
  - Dimension binding MUST follow the Analyze constraints below. It MUST use
    [hir §2](./hir.md#2-function-specialization-api)
    before ordinary type inference with ranges.
    Remaining symbolic dimensions MUST be named before any consuming algorithm runs.
  - Every effective Module topology MUST name a topology the resolved Target
    supports. Its extent MUST be positive and static, and within the finite
    hardware limit of that topology. A rejection names the topology, its extent, and
    the reason.
  - A non-`None` `topology` MUST name exactly one effective Module topology.
  - `check_program(analyzers=...)` and Analyze use the same checker gate. The gate
    runs `check_target_facts` of each checker, followed by `visit` on each reached
    definition. Analyze MUST run the checkers of the requested closure on the checked
    program before any metadata writes. The first refusal MUST stop the gate.
  - The returned Module MUST contain one typed derived function per reached
    definition, retaining calls and LoopRegions. Entry parameters and implicit
    child weights retain their names and roles; weights MUST NOT become root
    parameters. Authored definitions MUST remain unchanged.
  - Every primitive Call in each derived body MUST have a deterministic unique
    binding. `budget` MUST be a non-negative integer limiting the sum of unique
    derived body nodes, independent of occurrence count. Exceeding it MUST fail
    with the program name, total size, and limit.
  - Definition-local analysis MUST visit definitions in dependency postorder,
    once per definition. Calls are leaves whose completed callee metadata is
    resolved through the current family context and accumulated like primitive
    records. The entry is the final definition, without a second aggregation;
    enclosing trip counts multiply work and traffic. Recursion and a callee
    without a body MUST raise `VerifyError` naming the function and call path.
  - One definition MUST receive one argument-type signature and one current
    mesh. Different shape, dtype, layout, storage, or mesh MUST raise
    `analysis.Error` naming the callee, both call paths, and both signatures.
  - Check MUST attach internal `ExecutionMeshMetadata` to every derived
    Function, recording its unique enclosing mesh; root records `None`. This
    record is exported as `analysis.ExecutionMeshMetadata` but does not opt into printing.
  - Records MUST live on IR once per definition, in local coordinates. The
    definition's scope tree MUST stop at its own root, whose parent is `None`.
    Source entries MUST stop at formals; formals' variance dependencies MUST
    include their own identity. Only called definitions retain formal-dependent
    coordinate parameters. The selected root has no caller to bind them and
    MUST use the existing widest-range approximation with `UPPER_BOUND` precision.
    Each definition publishes its own one-execution summary; caller records
    compose those summaries at Call boundaries.
    Performance MUST schedule each definition once relative to its own origin.
    Call duration is the callee duration. Performance schedules it in the caller's
    current scope.
    Callee region peaks MUST take the maximum across occurrences.
  - Parameters and arguments MUST share storage. Argument liveness MUST extend
    through the callee's last use. Callee temporaries occupy the caller timeline.
    A separately materialized callee result of matching memory level and byte
    size MUST share the caller destination. A return without an independent
    material buffer requires a caller destination, which MAY increase peak memory.
  - Implicit weights MUST be persistent. The model counts them by `(slot key chain,
    weight)` and the union of reached instance-index tuples, clipped to slot
    bounds. Their cardinality scales weight bytes. Unknown indices reach the
    whole slot, and traffic uses loop trips without this multiplier. Caller
    induction variables map from the loop domain to those tuples; textual
    instance labels MUST NOT determine counts. Footprint coordinates prefix
    instance indices before unioning element addresses, and persistent bytes
    count each reached instance once.
  - Each internal SSA leaf MUST share one address across its occurrence windows.
    Callee parameters and results tied to caller destinations use caller pointers
    and carry no offsets. Internal values MUST NOT reuse parameter storage.
    A return group's proven reuse of a parameter applies only at the call,
    subject to the caller's persistent and last-use checks. Different alias
    relationships MUST raise `analysis.Error` naming the callee, binding,
    both binding paths, and relationships.
  - Authored-analysis readiness is not a program-level rejection. Analyze MUST
    NOT reject an authored `where(...)` constraint. It is an input to a later
    decision. Analyze measures a program that carries one as written. A value
    whose shard layout is deferred contributes its whole-program figures to the
    per-unit total. This is because a deferred layout states no distribution to
    project through. Values with a resolved layout still project.

`tilefoundry.analysis.api.analyze` is the dependency-composed measurement
operation. One call selects one or more root analyses by name. The operation
resolves their union dependency closure, runs each member once, and reports what
ran.
Its subject is one `Module` and one HIR `Function` that Module owns. Reachable
HIR callees are part of that selected invocation. They do not become separate
launches because of Module ownership. [hir §1.1](./hir.md#11-function) owns the
invocation rule. Analyze does not select, interpret, trace, or
dummy-run a plain Python orchestration method.

- constraints:
  - `check_program` MUST validate every caller/callee edge the selected query
    reaches against the one-execution-context requirement
    ([hir §1.1](./hir.md#11-function)). Reaching is what is validated, so an
    attached child no call reaches has no edge here. Both effective targets
    and complete topology hierarchies MUST match.

```python
class Result:
    """Record what one composed Analyze call computed.

    Attributes:
        module: attribute; Source Module.
        function: attribute; Function that received records.
        analyses: attribute; Requested root analyses in first-occurrence order.
        topology: attribute; Topology whose unit the per-unit quantities describe, or None.
        executed: attribute; Analyses executed in dependency order.
        metadata_types: attribute; Metadata classes actually written.
        functions: attribute; Reached derived functions in root-first DFS order.
    """

    module: "core.Module"
    function: "core.Function"
    analyses: tuple[str, ...]
    topology: str | None
    executed: tuple[str, ...]
    metadata_types: tuple[type[expr.IRMetadata], ...]
    functions: tuple["core.Function", ...]


def analyze(
    module: "core.Module",
    function: "core.Function",
    *,
    analysis: str | Sequence[str],
    topology: str | None = None,
    options: object | None = None,
) -> Result: ...
```

- constraints:
  - `topology` MUST name one effective Module topology. When omitted, it
    MUST default to the coarsest effective topology whose `TopologyLimits`
    states `max_physical_units`. A program may name a topology the host places
    rather than the machine runs. Several cards are one deployment's shape,
    not one card's. Measuring per such a topology asks the machine for
    a unit it publishes no rate for. When the target answers for none of the
    declared topologies, the coarsest MUST be left selected. The refusal then names
    the topology, not the absence of one. When the Module declares none,
    `topology` MUST remain `None`. No per-unit projection divides.
    `analysis.Result.topology` MUST record the resolved answer.
  - A target MUST answer `PerformanceServiceFacts` and state
    `TopologyLimits.max_physical_units` for the topology it is asked about.
    It MUST refuse a topology it publishes no rate for. It MUST NOT answer for a
    different one. The query is
    the topology name, or `(topology, active)` when the program runs `active` of its
    units. A target MAY ignore `active`. One unit gets the
    deployment's rate for the active units, shared among them. The same
    program measured at two topologies therefore states the same work against
    proportionally different rates.
  - The Function MUST belong to the checked Module, including its derived
    definitions. Analyze MUST write metadata in place without cloning or
    re-running type inference. A root without `ExecutionMeshMetadata` MUST be
    rejected with `analysis.Error("analyze needs the program returned by check_program")`.
    Dimension binding belongs to `check_program`.
  - Within `check_program`, stated `dims` MUST be a non-empty mapping. Its
    keys MUST name reached dimensions. Its integer extents MUST satisfy their bounds.
    Every dispatch dimension MUST be supplied. Exactly one variant MUST cover
    it. No dimension MAY remain symbolic after substitution in types, Meshes,
    loop bounds, attributes, or effective topologies. Invalid bindings and missing
    bodies MUST fail with their names before any consuming algorithm runs.
  - `analysis.Result.function` and `analysis.Result.module` MUST be the exact
    objects supplied by the caller. The checked Function retains its origin and
    bound extents; analysis does not return the symbolic authored input.
  - The recorded extents MUST be what identifies which size a derived Function is
    at. They MUST NOT be inferred from its signature. A dimension occurring only
    in a loop bound, a body operation's attribute, or a nested callee leaves the
    signature identical at every extent. Two sizes would then be indistinguishable
    to anything comparing signatures.
  - One call MUST select one or more root analyses. It MUST preserve their
    first-occurrence order. It MUST resolve their union dependency closure. It MUST
    order the closure so every dependency precedes its dependants. It MUST execute
    each member exactly once. `executed` MUST report that order, so a shared
    dependency appears once.
  - Dependencies MUST be resolved under the same exact concrete Target as the
    root, obtained from `Module.resolve_target()`.
  - A dependency cycle MUST fail and MUST name the path that closes it. A
    missing root and a missing dependency MUST be distinguishable. One is the
    caller's selector. The other is a broken Target capability.
  - Type inference and structural validation belong to `check_program` and
    MUST complete before Analyze receives the program. Analysis reads those
    inferred types and assumes a verified function.
  - Family-specific readiness MUST be checked on that derived program before the
    first analysis in the dependency closure runs. A target can state no physical
    units for a topology. A performance request for that topology MUST fail before
    dependency Metadata is written.
  - Re-running MUST recompute the closure and refresh the Metadata that closure
    owns. There MUST be no cross-call cache. Metadata owned by nothing in the
    closure MUST be left untouched.
  - `metadata_types` MUST list the Metadata types the call actually wrote onto
    the IR, in execution order and without repeats. An analysis that declares a
    type but writes no record for this function MUST NOT contribute it. A renderer
    is then never sent after records that are not there.
  - `analysis.Result` MUST be semantic. Human text, JSON, and annotated HIR are
    renderings of it and of the Metadata on the IR. They MUST NOT be fields of
    it.

### Visit context and local coordinates

`get_metadata(expr, kind, ctx=ctx)` delegates to `IRMetadata.resolve(ctx)`
and the already-bound Scope. Its default returns the record unchanged.
Metadata does not declare capability requirements.
Analysis contexts extend `TypeInferContext`, not `cost.Context`. The context
prepares the callee and argument bindings. The `enter_call` of the context calls the `attach` operation of the caller scope and
records the call edge. Reading `callee_scope` follows that existing edge.
It does not recursively bind a definition tree. Consumers query `ctx.scope`,
`ctx.scope.provenance`, and `ctx.scope.access` without an additional attach step.
`scope.root` identifies the current definition. `scope.parent` always means the
lexically enclosing scope. It MUST NOT be repurposed as a caller link.
`call_site` identifies a Call. `caller_context` identifies the caller environment.
`caller_scope` and `callee_scope` refer to scopes, not functions.

Analyze drives each family once per reached definition, dependencies first.
Definition-local snapshots remain symbolic and shared. A bound callee view
substitutes actuals without mutating the definition tree. Calls do not create
lexical scopes. Calls do not trigger a second analysis of their callee body.

| Record | Interpretation |
|---|---|
| ComputeCostMetadata | Local execution counts. Caller loop scaling occurs during accumulation. |
| MemoryMetadata | Local traffic. Retained Calls have no independent boundary footprint. Callee accesses are translated once while building caller scopes. |
| LivenessMetadata | Local event records. `at(origin)` uses the common `shifted` operation. |
| AllocationSummaryMetadata | Values, in-place capabilities and unchanged raw address decisions. The attached Scope interprets bound occupancy and caches it in its snapshot. Consumers do not recursively expand calls. |
| PerformanceMetadata / PerformanceSummaryMetadata | Local intervals. The performance family handles their origin, not Scope. |

`Spread.logical` is the sum of `logical_groups`, or the typed zero for empty
groups. Call groups name local induction variables and formals. Function
accumulation consumes local induction variables. It retains only the formals on
which the count depends. Caller Function accumulation multiplies the count by
`scope.dependency_trips` for that set. This preserves the joined precision. Call
resolution MUST NOT apply caller loop multiplicities to either work or traffic.
Renderers MUST show only `logical`, `total`, and `per_unit`. Renderers MUST NOT render
the groups.

Footprint evidence retains local reached sets, their `access.Relation.values`
expressions, and groups of reached sets with local window multiplicities.
The expression printer owns display names. Metadata references retain
SSAValue identities. They reuse the same printer bindings as the surrounding IR.
Cross-function references MUST use `function::local_name`. References within
the current function keep their local name. This prevents distinct buffers with
the same local name from overwriting one another in reports. Within a function,
ambiguous SSA references MUST append the printed definition line. A loop-carried
parameter uses the loop header line. The code keeps its original local names.
Analysis does not generate buffer display strings or cache a separate name map.
Loop coordinates MUST remain bounded parameters. Induction identity names them.
Their values refer to their induction Vars. Resolve substitutes formals and
retains callee loop parameters. Runtime scalar bindings remain symbolic until
the numeric window is selected. Function and Call footprint numbers hold every
loop parameter at its first iteration. A caller reuse window instead unions
all iterations inside its selected window, including complete callee executions.
Loops outside that window stay at their first iteration. Caller candidates
remain its own axes. These fields support substitution. They MUST NOT be rendered. They MUST NOT
retain a scope or context. Numeric views of unresolved
formal-dependent coordinates MUST use the existing widest legal bound.
They MUST state `UPPER_BOUND`. Analysis computes call-site counts after substitution.

Analysis MUST compute reuse windows once per definition. Callee internal loops are
candidates only on that callee's Function record. Caller records use only the
caller's own loops and mesh axes. Their window footprint includes resolved
callee reached sets. A callee's unresolved formal coordinates use their widest
legal bound and state `UPPER_BOUND`. ReuseWindow is not resolved through calls.
Local source values use local
variance. A callee-internal source absent from that table conservatively follows
the variance of the current Call. Implicit const weights remain invariant.
The reached image tests instance-index dependence.
This may omit
reuse for an internal temporary that depends only on constants.

LivenessMetadata is internal metadata, not a report family. Each caller reserves the
event span of a callee template. It records actual leaf uses at the translated
formal-use events and preserves synthetic flags. Resolve only shifts the local
template. It stores neither child templates nor a second timeline origin.
Allocation may resolve the callee template at a Call to obtain internal windows.
Standalone liveness for a function without Calls MUST work without a scope
tree. It MUST NOT use an attached LivenessMetadata record as an implicit result cache.

Callee call durations MUST use the resolved callee summary's duration divided
by its waves. Non-entry summaries use the local schedule without a roofline
floor. Only the entry prediction applies its roofline floor. Callee summary
waves still follow the checked definition mesh.

Root footprint, reuse windows, lifetimes and whole-program peaks MUST be
absolute records. MeshRegion peak rows MUST also appear in report JSON under
`meshes`, parallel to `calls` and `loops`. Callee rows display local records.
Call records are already resolved in their own definition. Reporting MUST read them directly.
Reporting MUST NOT rebuild scope trees or resolve them again.

### 2.1 Shared IterationScope and Access

Analysis visits each reached definition once to build its IterationScope tree.
`analysis/iteration_scope.py` defines `ValueProvenance` and `ScopeAccess` beside IterationScope.
Scope composes them as `scope.provenance` and `scope.access`.
The first implements source and dependency queries. The second implements access
and projection queries. Both operate on the single `snapshot` of the scope:

| Snapshot section | Ownership and lifetime |
|---|---|
| `definition` | Shared definition facts: value snapshots, access records, relations, refusal evidence, and operation locations. |
| `attachments` | Call edges in this calling environment, including the callee reference, arguments, and instance identity. |
| `resolved` | Value, parameter, and relation results in this fixed lexical and calling environment. |

Attach MUST NOT recursively copy the definition tree or project every child
domain. Analysis obtains a lexical entry only when needed. The entry shares definition facts
and keeps derived results separate from other calling environments. A binding
change requires a new environment. Analysis does not reuse stale results.
`scope.at(path)` follows existing call edges and definition location indexes.
It does not rescan the callee tree or initiate another callee analysis.
Scope itself owns lexical structure, the iteration domain, and multiplicities.

`snapshot.definition.values[SSAValue]` contains one SnapshotEntry with `root`, `narrow`, `device`,
`unit`, `varies`, and optional `forward`. `forward` is a direct structural operand,
not the data-identity root. Three views remain supported. A missing unit view is
unknown. Snapshots do not duplicate types or display names.
`domain` is an access.Relation of zero-dimensional tuples with constraints on
identity-named loop parameters. `first()` constrains each coordinate to its
loop's start. `compose`, `constrain`, and `project` maintain parameter mappings
in access.Relation. Consumers do not maintain parallel parameter-name tables.
`ir/core/ssa_value.py` defines `SSAValue`. `ssa_leaves(expr)` lists the values of an expression. `SSAValue.leaf` reads the type of one value.

An input `Access` stores its original Call boundary index, its relation,
and its allocation expression. An output stores `input_index=None`. An output also stores its own output
boundary index. That index is the tuple field the output answers for. Failed boundaries remain
absent. They do not shift the index of later successful inputs or outputs. Analysis reads storage
level and element width from `Access.target.expr` at
`Access.target.index`. Analysis MUST group sources by `SSAValue`. The label of a tuple result
is `<name>[index]`. A whole-tuple input boundary uses leaf-slot coordinates.
`resolve_access` MUST expand each possibly selected slot into a separate Access
in that leaf's legal tensor-element coordinates. These accesses retain the
original boundary index and may-access flag. Counting produces the precision. Their target remains
the definition-local tuple leaf. No sentinel leaf index represents a whole tuple. A refused descendant
makes its owning scope unknown for that view. Non-affine runtime indices retain
the widest legal access approximation.
Normalization shares one derived body and its records per reached definition
([§2](#2-composed-analysis)).

`access.affine_of` (`interfaces/access`, see
[visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation))
turns HIR values into isl affine forms. `analysis/loop_domain.py` builds the
iteration domain from them. `analysis/access.py` owns access widening. The two
modules do not define a second affine graph representation.

- constraints:
  - A loop `start` or `extent` MAY be unit-dependent. Every runtime leaf in such a loop bound
    MUST carry a half-open value range. `IterationScope.domain` MUST keep the
    whole affine expression. It MUST represent each such leaf as one identity-deduplicated
    isl parameter constrained by that range. A leaf without a range MUST be
    refused.
  - `IterationScope.domain.values` MUST map each of those parameter names back
    to the value it stands for. The name itself carries no meaning.
    A reader that must recognise a unit coordinate cannot re-derive it without
    restating how the name was made.
  - A loop `step` MUST be a literal. A parametric stride has no isl
    representation.
  - `cardinality` MUST enumerate every feasible integer point of a parameter box
    of at most `PARAM_POINT_LIMIT` points. It MUST return the maximum. It MUST
    report unknown for a larger box. It MUST count directly when every retained
    parameter is already fixed to one integer point. An empty parameter context
    MUST count as zero. A non-empty context with an unbounded parameter MUST
    report unknown. An unbounded set dimension MUST also report unknown.
    Named `union` parameters instead reduce by set union before maximizing over
    the other parameters. Small union boxes MUST fix each fiber before projecting
    those parameters. They MUST then union and coalesce the images. An oversized or
    unbounded union box MUST fall back to existential projection and counting.
    It MUST NOT approximate the reached set.
  - `IterationScope.trips()` MUST count a loop off its bounds, not its domain.
    The count is `max(1, ceil(max(stop - start) / step))`. It MUST refuse an unbounded span.
    `trips_precision` is `EXACT` where that span reaches one value and
    `UPPER_BOUND` where it reaches several. The count is then an upper bound.

### 2.2 Target-selected Analyzers

```python
class AnalyzeContext(typeinfer.TypeInferContext):
    module: core.Module
    function: core.Function
    target: target.Target
    topology: str | None
    options: object | None
    scope: IterationScope | None = None
    mesh_scope: types.Mesh | None = None

    @property
    def topologies(self) -> tuple[Topology, ...]: ...
    def type_of(self, expr: Expr) -> Type: ...
    def local_type_of(self, expr: Expr) -> Type: ...


AnalysisCallable = Callable[
    [core.Function, AnalyzeContext], None
]

# tilefoundry.target.Analyzer
class Analyzer:
    """Describe one Target-selected analysis; a frozen dataclass.

    Attributes:
        selector: attribute; Public analysis selector.
        run: attribute; Analysis implementation.
        requires: attribute; Dependency selectors.
        produces: attribute; Owned metadata classes.
    """

    selector: str
    run: AnalysisCallable
    requires: tuple[str, ...] = ()
    produces: tuple[type[expr.IRMetadata], ...] = ()

    def get_checker(self) -> "target.AnalysisChecker | None": ...


# tilefoundry.target.AnalysisChecker
class AnalysisChecker(Protocol):
    """What one analysis requires before any analysis writes.

    A checker states requirements and never attaches Metadata. It asks
    target-wide questions once, then visits each reached definition.
    """

    def check_target_facts(self, ctx: CheckContext) -> None: ...
    def visit(self, node: expr.Expr, ctx: CheckContext) -> None: ...


class Target:
    def get_analyzer(self, selector: str) -> Analyzer: ...
```

- constraints:
  - `AnalysisCallable` MUST receive the normalized Function graph and one
    `AnalyzeContext`. That context MUST carry the exact Module, Target, resolved topology,
    and caller options. It MUST also carry the shared `scope` view (`scope.root` for the definition). The
    `topology` MAY be `None` only when the Module declares no topology.
    Options MAY be `None`.
  - Analysis reads types and access relations through the same `AnalyzeContext`.
    Analyze creates it before the scope tree. The scope tree reads relations through
    it with no topology. Analyze binds its `scope` to that tree
    (`bind_scope`) before any analysis runs. `type_of` returns the `Expr.type` that
    `check_program` stored on the analyzed copy. `local_type_of` projects it to
    `topology` over the Module's topologies, or returns it whole without
    one. `mesh_scope` is `None`. Analysis asks relations outside any execution
    mesh.
  - Analyze MUST obtain every root and dependency from the same exact Target
    instance through `get_analyzer`.
  - A Target subclass MUST inherit its base Analyzers through normal Python
    inheritance. It MAY override one selector and delegate the rest to
    `super()`. It MAY refuse inherited behavior that is invalid for its hardware.
  - There MUST be no public analysis registration step or exact-concrete-Target
    algorithm table. A custom provider registers only its Target class.
  - A declaration MUST be rejected when it requires itself or repeats a
    dependency. A declaration MUST also be rejected when it produces the same Metadata type twice.
    A declaration MUST be rejected when it names a `produces`
    entry that is not an `IRMetadata` subclass.
  - `get_checker()` MUST default to `None`. An analysis with no input
    contract needs no override and keeps working unchanged. An Analyzer with an
    input contract is a subclass. The subclass overrides `get_checker` to return a fresh
    checker. A checker MUST NOT attach Metadata. It states what a
    program must already be. Every checker in a closure MUST have answered
    before any analysis in it writes.
  - An analysis MAY change only the Metadata types its Analyzer declares.
    Ownership MUST be enforced against what reached the IR rather than against
    what the analysis reports. Ownership enforcement MUST cover addition, replacement, and
    removal alike. Deleting another analysis's record changes the IR as much as
    overwriting it. An equal-valued overwrite of another analysis's record MUST
    also count as a violation.
