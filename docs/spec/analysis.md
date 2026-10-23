# TileFoundry Spec — analysis (authored-HIR metrics + per-stage target facts)

This spec owns TileFoundry's fact layer: everything a later stage decides
*over*, and nothing that decides anything itself. It has two surfaces:

| Surface | Entry | What it states |
|---|---|---|
| Program check | `check_program(module, function, dims=..., topology_level=..., budget=..., analyzers=...)` | a checked Module and root Function; each derived Function carries internal `ExecutionMeshMetadata(mesh)` recording its unique enclosing mesh, or `None` for root |
| Composed measurement | `analyze(program, function, analysis=...)` | one or more root analyses and their union dependency closure, writing typed Metadata in place on the program returned by `check_program`; no cloning or dimension binding |

The caller binds dimensions through `check_program`, then passes both returned
objects to `analyze`. Local definition trees remain internal to analysis.
`AnalysisResult.module` and `AnalysisResult.function` identify the supplied program and root.

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
they describe: a record on a `Call` describes that call, a record on a
`MeshRegion` describes that execution stage, and a record on a `Function`
describes the whole function.

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
| `memory` | - | `MemoryMetadata`, `RegionMemoryMetadata` | `MemoryMetadata` on every measured Call; `RegionMemoryMetadata` on the Function and each occupied `MeshRegion` | the authored program, `MemoryHierarchyFacts`, `TopologyFacts` | `memory`, `advisory` | every measured Call |
| `roofline` | `compute-cost`, `memory` | `RooflineMetadata` | every measured Call and the Function | `ThroughputFacts` | `roofline` | every measured Call |
| `performance` | `compute-cost`, `memory` | `PerformanceMetadata`, `PerformanceSummaryMetadata` | `PerformanceMetadata` on every Call with a modeled duration; `PerformanceSummaryMetadata` on the Function | `ThroughputFacts`, `TopologyFacts`, `MemoryHierarchyFacts` | `performance` | every Call with a modeled duration |

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
    Performance readiness requires positive `TopologyLevelFacts.max_physical_units` for
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
    program offers no such reading. A comment over the same record MAY
    state fewer keys, or projected ones
    ([inspection §2.8](./inspection.md#28-record-comment-forms)), and what it
    leaves out MUST stay in the JSON projection.
  - A compact text summary MUST contain whole-function facts only. Per-value
    facts MUST stay on their annotated equations; JSON MAY retain operand names
    and types in its structured projection.

#### 1.2.1 `compute-cost`

Analysis quantities share `AnalysisPrecision`: `EXACT`, `UPPER_BOUND`,
`LOWER_BOUND`, and `UNKNOWN`. Their JSON spellings are `exact`, `upper_bound`,
`lower_bound`, and `unknown`. Exact evidence is the identity when evidence is
combined; equal directions retain their direction, opposing directions become
unknown, and unknown evidence absorbs every other direction.

`compute-cost` measures the logical work of each authored `Call` without reading
target hardware facts. What an occurrence moves is the memory family's answer
([§1.2.2](#122-memory)), read off the same registered evaluator.

```python
class ComputeCostMetadata(IRMetadata):
    """Typed work in logical, expanded, and topology-unit domains."""

    topologies: tuple[str, ...] = ()
    flops: Breakdown[int] = Breakdown()
    other_ops: Breakdown[int] = Breakdown()
    precision: AnalysisPrecision = AnalysisPrecision.EXACT
```


| Field | How it is computed | Reads the target |
|---|---|---|
| `topologies` | The effective Module topology levels, coarsest first. | No |
| `flops` | For a primitive Call, run its registered cost evaluator over operand and result Types as written for `logical`; each level's share is the same evaluator over Types projected through authored `Split`s at or coarser than that level, and `total` is the selected unit's share times its executing mesh positions. Function accumulation multiplies `logical` by varying-loop trips, and `per_unit` and `total` by all enclosing loop trips. Retained calls carry their callee's one-execution summary. Primitive work accumulates into root totals with full repeats and into each enclosing call with repeats relative to that call scope. | No; projection reads resolved Mesh and effective Module topology extents. |
| `other_ops` | The evaluator's non-floating-point operation counts (`integer`, `predicate`, `select`, `special`), totalled and shared the same way. These keys map directly to target one-unit operation-throughput keys; target-side service naming is unchanged. | No; projection reads resolved Mesh and effective Module topology extents. |
| `precision` | One-trip Call work is exact; Function work combines the precision of every enclosing loop's trip count. Coordinate-dependent maximum trip counts yield an upper bound. | No |

Requesting this family adds one summary line, prefixed by `# `: the Function's own
record, stated exactly as a Call's is. The whole program's work is not a second
record.

```text
compute-cost flops=<dtype>:<int>@logical,<int>@total,<int>@<level>[,...][;<dtype>:...] other-ops=<kind>:<int>@logical,<int>@total,<int>@<level>[,...][;<kind>:...] precision=<exact|upper_bound|lower_bound|unknown>
```

Every measured Call receives this annotation. Each key pairs the whole quantity
with one unit's share, so the two `*_per_unit` fields are not separate keys.

Each reported Call's JSON projection is under its `compute-cost` key:

```text
{"topologies": [<level>, ...],
 "flops": {<dtype>: {"logical": <int>, "total": <int>,
                      "per_unit": [<int>, ...]}},
 "other_ops": {<kind>: {"logical": <int>, "total": <int>,
                         "per_unit": [<int>, ...]}},
 "precision": <"exact"|"upper_bound"|"lower_bound"|"unknown">}
```

- constraints:
  - A record MUST be attached to every reached `Call` and Function.
    The Function record MUST include authored-loop repetition and therefore is
    not the direct sum of the one-occurrence Call records.
  - An op with no registered cost evaluator MUST raise `AnalysisError`.
  - Dim arithmetic (`DimAdd`, `DimSub`, `DimMul`, `DimFloorDiv`, `DimMod`,
    `DimMin`, `DimMax`) is scalar address work, not tensor work: its evaluator
    MUST state no flops, no service, and zero traffic in every operand slot.
  - Missing program geometry MUST NOT be replaced with a target capacity.
  - `logical` MUST multiply only the authored loop trip counts for loops whose
    induction variable or carried argument the Call transitively reads.
    `per_unit` and `total` MUST multiply every enclosing loop's trip count,
    because a Call written in a loop body executes on every iteration.
    The same rule MUST apply to primitive and Function Calls.
  - Downstream families MUST read the already-scaled record and MUST NOT apply
    authored loop trip counts a second time.

#### 1.2.2 `memory`

`memory` states what every occurrence moves and at which level, the unique
addresses one wave touches at a representative iteration, whole-Function value
lifetimes and placement peaks, and which repeated reads fit in cache. The movement
is read off the Op's own registered evaluator and the amounts its access
relations reach.

Every measured `Call` carries `MemoryMetadata`; every reachable `Function`
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
    alias: BufferAlias | None = None


class ReuseWindow:
    """One buffer's re-reads, and what keeping it costs the cache."""

    buffer: SSAValue
    ownership: tuple[str, ...] = ()
    time: str = ""
    space: str = ""
    holds_bytes: int = 0
    reuse_bytes: int = 0
    fits: bool = True
    precision: AnalysisPrecision = AnalysisPrecision.EXACT


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
Movement and footprint counting for data-dependent lookup boundaries follow
[visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation).
Footprint retains the full possible address image for reuse decisions. Out-of-bounds
IndexSelect indices with `fill_value` are not statically distinguished; traffic
counts one slice per index as an upper bound.

| Field | How it is computed | Reads the target |
|---|---|---|
| `MemoryMetadata.topologies` | The effective Module topology levels, coarsest first. | No |
| `MemoryMetadata.traffic` | One occurrence's per-boundary movement, grouped by storage and communication boundary. | No; projection reads resolved Mesh and topology extents. |
| `MemoryMetadata.operands` | One occurrence's movement in order `(*call.args, call)`. | No |
| `MemoryMetadata.buffer_bytes` | The bytes in one physical copy of an address-placed `gmem` or `smem` tensor result, including a carried phi. `None` for tuple results or results that are not address-placed. | No |
| `MemoryMetadata.offsets` | The byte offset of each physical result slot: each copy of a staged tensor in copy order, or each tuple `AllocationValue` in flattened result-index order. An `rmem` result has no reported offsets. Staged tuple results MUST raise `AnalysisError`. | No |
| `RegionMemoryMetadata.topologies` | The same effective Module topology levels. | No |
| `RegionMemoryMetadata.traffic` | Every reachable occurrence. `logical` multiplies only loops the value varies in; `total` and `per_unit` multiply every enclosing loop. | No |

- constraints:
  - A Call's traffic `total` MUST equal the counted unit's share multiplied by
    the number of positions executing its mesh scope, as for `compute-cost`.
    Storage MUST count the selected topology unit; communication MUST count
    the topology unit named by its boundary. An undeclared communication
    boundary MUST raise `AnalysisError`, rather than use whole movement.
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
| `Footprint.buffers` | Group reached address sets by source-buffer identity, union each group, count its elements, and pack the source dtype's bits into whole bytes. Each buffer has one memory-level kind whose `Spread` states one window's unique bytes in `logical` and the executed windows' bytes in `total`, with no `per_unit` entries. | `MemoryHierarchyFacts` selects the level; [target §11](./target.md#11-target-facts-projection) supplies `TopologyFacts`. |
| `Footprint.precision` | Combine each contributing boundary's direction: omitted boundaries or uncountable amounts are lower bounds, counted widened accesses are upper bounds, and refused boundaries have unknown direction. | No |
| `MemoryMetadata.footprint` | The unique addresses this occurrence's own boundaries reach at every enclosing loop's first iteration over one wave. `total` multiplies this window by the number of waves in the enclosing mesh; absent when no wave can be stated. | As above |
| `RegionMemoryMetadata.footprint` | The union of every Call's reached addresses below the Function in `logical`. For `total`, group Calls by their enclosing loop/mesh chain, union each group per buffer, multiply by its waves and every enclosing loop's trips, and sum the groups. Absent when no wave can be stated. | As above |

- constraints:
  - May-access footprint counting MUST follow the cap stated in
    [visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation).
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
  - `logical` MUST retain the single-window union. `total` MUST count that
    union once per executed window, with no `per_unit` entries. The wave count
    MUST be `ceil(mesh_positions(topology_level) / wave_units)`; a partial last
    wave MUST count as a full wave. Function accumulation MUST union Calls
    sharing an enclosing loop/mesh chain before multiplying by that group's
    waves and every enclosing loop's trips, then sum the groups.
  - Reuse, cache capacity checks, and the printed `footprint` field MUST use
    the single-window `logical` bytes.
  - `Footprint.precision` MUST preserve direction throughout address merging
    and counting. A missing boundary MUST contribute `LOWER_BOUND`; a counted
    widened relation MUST contribute `UPPER_BOUND`; a refused boundary MUST
    contribute `UNKNOWN`. No intermediate Boolean may discard this direction.
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

One ordinary expression event uses its operands and defines its result. A
region adds separate binding and exit events: a mesh argument is used before
its parameter is defined, and a loop initial value is used before its induction
and carried parameters are defined. One representative loop-body iteration is
recorded without expanding the trip count; a carried parameter spans entry to
exit, and a yielded value remains live through the backedge event. Positions
are monotonic across the whole Function, including nested and sibling regions.

| Field | How it is computed | Reads the target |
|---|---|---|
| Report `binding` | Render `ValueLifetime.value` through the same printer used for the surrounding IR. Metadata retains the SSAValue identity, without storing a display name. | No |
| `ValueLifetime.memory_level` | Emit one lifetime per material `AllocationValue`, at the level occupied by its projected Type. | No |
| `ValueLifetime.bytes` | Project the Type through every authored split at or coarser than the explicit level's `owner`, then take its logical bytes; a target-owned or undeclared level remains global. A `ScheduleOp` whose selected instruction has a write-only result occupies `buffers` copies of those bytes; a `READ \| WRITE` result occupies one. | `MemoryHierarchyFacts.explicit_levels[].owner` |
| `ValueLifetime.defined_at` | Definition event on the Function-wide structured SSA timeline. A staged (`buffers > 1`) result starts at the outermost enclosing loop's phi event rather than its own definition event. | No |
| `ValueLifetime.last_used_at` | Greatest ordinary-consumer, region-entry, loop-backedge, or region-exit use event belonging to this result index. Static tuple projections use only the selected result; other tuple uses use every result. | No |
| `ValueLifetime.persistent` | True for parameters and false for body allocations. | No |
| `RegionMemoryMetadata.lifetimes` | Every value residency except a non-material view. | As above |

- constraints:
  - Optional reuse MUST NOT place an ordinary result in a caller-owned
    parameter buffer. Donation is a contract with the caller, not a conclusion
    this family may draw. Semantic same-address relations, including registered
    views, InsertSlice destinations, and retained-call formal-to-actual bindings,
    are address constraints rather than optional donation. Each SSA value remains
    a distinct value with its own interval, even when addresses are equal.

##### `BufferAlias`

```python
class BufferAlias:
    target: SSAValue              # the result whose bytes hold this value
    relation: AccessRelation      # result coordinates -> target coordinates
    kind: AliasKind
```

| Field | How it is computed | Reads the target |
|---|---|---|
| `BufferAlias.target` | A registered view: the operand `aliased_operand` names. An adopted InsertSlice update window: the destination resolved through region captures. A carried-phi tie: the initial or phi value; an accepted InsertSlice result tie: its destination. | No |
| `BufferAlias.relation` | A registered view: `renaming_relation`. An update window: the proven operand-to-result relation with parameters resolved to values visible at the producer. A tie: identity. | No |
| `BufferAlias.kind` | One `AliasKind` member for each structural link or accepted tie. | No |

- constraints:
  - An op registered with `@register_buffer_alias(Op)` returns the viewed
    input's position or `None` for each Call. `buffer_alias()` returns
    `BufferAlias(SSAValue(buffer), relation, AliasKind.VIEW)`, describing that input's bytes and the
    result-to-input coordinates, and proves the declaration on every call.
    `Slice` and `Reshape` alias input zero and MUST NOT allocate independent
    buffers. `Tuple`, `MeshRegion`, and `LoopRegion` results and static tuple
    projections describe bytes already held by their selected body or carry.
    They MUST have their own AllocationValue intervals and require the operand's
    address through constraints. Static tuple projections are visitor structural rules, like
    `Tuple`, `MeshRegion`, and `LoopRegion`, rather than registered views.
    An InsertSlice result MUST use its destination's address, including a
    parameter destination, unless the old value has a later ordinary use.
    Every other ordinary result, including `Transpose`, MUST allocate its own.
    Analysis MUST use operation semantics
    for this distinction rather than infer aliasing from layouts.
  - Reshard MUST alias its source iff storage is unchanged, the shape is
    literal, and every logical element retains its complete set of storage
    instances and addresses. Gmem has one instance per device; smem has one
    per device and CTA; rmem has one per thread. Broadcast replicas MUST be included: removing a
    replica changes the set. A plain end has a copy on every
    instance of the other end's mesh; two sharded ends retain their own mesh
    selections. Every other Reshard MUST retain its independent buffer.
  - Bitcast MUST alias its source and allocate no independent buffer. Its
    declared access relation MUST bijectively map result coordinates to source
    coordinates with the same physical address, requiring the source's
    address and recording zero traffic.
  - Fresh smem allocations MUST reserve a byte span covering the physical
    layout's addresses within the selected topology unit, including stride holes
    and composition offsets. Splits at coarser levels restrict the local domain;
    finer splits remain in it. Swizzles
    MAY reserve complete low-bit periods, preserving the unchanged high bits;
    this is an arena-capacity bound, not a unique-footprint or traffic count.
    Every staged slot MUST reserve that span at the required alignment.
    Invalid negative or unresolved physical address ranges MUST be refused.
  - Registered buffer aliases MUST create an `AllocationValue(lifetime, aliases)`
    with its own SSA interval and the operand's byte span, requiring the same
    address. A persistent operand MUST pin its alias at the same
    address. Aliases MUST NOT receive independent `buffer_bytes` or
    `offsets`. Analysis MUST resolve structural tuple projections to their
    selected element without a coordinate-renaming proof. For other aliases,
    `buffer_alias()` MUST prove that the result-to-operand map is single-valued, injective, and
    contained in the operand's index box, using the call's own unmodified
    declared access relation via `renaming_relation`, not its folded accesses.
    Failure to prove that declaration MUST raise `AnalysisError`. `Transpose`
    is not registered: it writes a new value, retains its access permutation,
    and receives its own allocation under the ordinary conflict rules.
  - Retained calls MUST translate every callee value in a formal parameter's
    address group into the corresponding actual argument's group. A call result
    returning a parameter or its view MUST use that actual's address and relation,
    unless the composed callee chain writes into the formal (directly through
    InsertSlice or through a retained call) and the old actual has a later ordinary
    use. Read-only parameter and view returns MUST alias despite later reads of
    the actual. This is independent of optional reuse of fresh results. InsertSlice
    and call-result aliases MUST remain accepted links in the allocation summary
    without independent bytes or offsets.
  - Allocation MUST return storage for each SSA value and record adopted
    InsertSlice update windows and carried-phi ties in `AllocationValue.aliases`.
    An update window MUST be adopted only together with the InsertSlice's
    destination alias. A material tuple op's leaf MAY reuse the destination's
    window in analysis; accepted links retain their SSA leaf identities. Lowering continues
    to reject unsupported material tuple results. An update window's buffer
    MUST be the destination resolved through
    region captures; its relation MUST map producer coordinates to destination
    coordinates. The destination and SSA parameters MUST be defined in the
    producer's scope or an ancestor, before the producer's definition event.
    Constants and dimension variables are always visible;
    a mesh coordinate is visible only if an enclosing mesh region declares its
    topology. Lowering MUST execute this decision without reading
    the InsertSlice's parameters at the producer.
    A window adopted for a tied carried parameter MUST be
    recorded on its tie initial; the carried parameter MUST retain its tie alias.
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
    loops. Estimation, optional reuse, placement and reports continue to use the
    source tables above.
    A retained Call return leaf MUST project the callee's return source.
    Source identity describes data for footprint and traffic, independently of
    allocation ownership. Allocation gives each Call result its own storage
    root unless the callee summary returns a formal root, which forwards to the
    actual leaf. Allocation and source collection MUST share the direct operand
    leaf selection for structural forwarding; allocation MUST preserve Call
    storage boundaries when using those operands and cached view relations.
    MeshRegion results use their body's source; static tuple projections
    select a structural element or a material op's flattened result index;
    loop results use their corresponding carried phi; capture parameters use
    their arguments' sources. Every other source is the value itself and identity.
    `resolve_access` MUST perform one source lookup and one relation composition,
    rather than trace producers per access. Formal-dependent coordinates are
    retained only when the caller explicitly supplies the definition's formals;
    resolving a boundary MUST NOT walk to a definition root to discover them. Boundary parameters MUST be eliminated
    before composition in the operand's coordinate space. View parameters MUST
    be eliminated when constructing that view's source, in the input's coordinate
    space and using the input's type for widening. Source relations retain their
    defining loop prefix, aligned to the access's enclosing loops; loops internal
    to an escaping region MUST be projected out.
  - LivenessMetadata MUST record each SSA value's own interval. Except for dynamic
    projections and their forwards described below, every tensor SSA value,
    including views, region parameters and results, and static tuple projections,
    MUST have its own `AllocationValue`; intervals MUST NOT be merged or extended
    to include aliases. Every tuple value MUST have one `AllocationValue` identified by `lifetime.value`
    per flattened tensor result. Its interval runs from the tuple's definition to
    the last use belonging to that result: a static tuple projection uses only
    the selected result, while whole-tuple operations, region captures, synthetic
    loop uses, and returns use every result. A dynamic tuple projection MUST
    conservatively
    keep every possible tensor leaf resident until the function timeline ends:
    liveness MUST record a synthetic use at `timeline_end` on each flattened
    leaf of the tuple operand. These are the tuple leaves' own uses; their
    structural source links keep the element storage resident. The projection's
    own interval MUST follow only its own uses. The projection and its forwards,
    including capture parameters and region results, MUST have no
    `AllocationValue` (storage is `None`) and MUST NOT participate in in-place reuse.

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
Otherwise B MUST end no later than A starts, B MUST be neither persistent nor
rooted at a borrowed formal, both levels MUST agree, and bytes(A) MUST be at
most bytes(B). Acceptance adds a value link from A to B. Address decisions MUST
NOT change accepted links.

Every tie MUST read the same occupancy: a value's own lifetime plus translated
callee windows mapped to that value. Start is the earliest definition and end
the latest use among those windows; lifetimes themselves MUST NOT be changed.
Without Calls this occupancy is the value's own lifetime.
A root's extent MUST be the maximum bytes of every member's occupancy, including
mapped return-producer windows; its own lifetime.bytes remains the size of its
projected Type. At equal endpoints,
a mapped callee window has no caller-local access proof: only members of the
formal covered by a callee IN_PLACE capability may borrow that proof, checked
again in actual storage-root coordinates. Other mapped windows MUST end strictly
before the candidate starts. Global interference MUST compare individual
windows, never their occupancy envelope.

At equal endpoints, A's first defining operation MUST have a proved zero-offset
`operand_to_result_relation` for every B member it reads at that event. The
InsertSlice destination identity branch is a valid proof. A matmul reading its
carry is not pointwise and MUST NOT acquire a carry-yield tie on that evidence.
IN_PLACE candidates MUST be limited to proved zero-offset, non-persistent
operands at the same level whose own last use is no later than the result's
definition. Nonzero-offset IN_PLACE candidates MUST NOT be registered.
IN_PLACE MUST use source-resolved
`operand_to_result_relation`, grouped by `Access.target`, so the proof is in
storage-root coordinates after VIEW/FORWARD composition. A Slice's local
identity MUST NOT turn a shifted backing-buffer read into zero-offset reuse.

At equal endpoints, and for WINDOW reads at its start, proof MUST compare isl
relations in B's root coordinates: each operation's read relation MUST equal
its value's relation at the candidate output landing, restricted to that
input's iteration domain. Merely belonging to one storage root is insufficient.
The proved InsertSlice destination identity includes its forwarded unchanged
coordinates, not just the update's write-traffic coordinates.

CARRY_YIELD for loop L uses content within one representative body iteration.
Between different storage roots, A's material producer (including a material
Call result) MUST be defined inside L's body. A body-local view or forward of
an outside root does not qualify; a candidate without any body-local member
definition MUST be rejected. Rejection uses the ordinary backedge copy.
Completion for candidates already sharing one root is unchanged.
Its B endpoint is the last use of any B member strictly between L's phi and
backedge events; inner-loop synthetic uses count. L's own backedge, exit, and
later uses do not read the old phi content. Its A start is the earliest member
definition inside L's body. The resulting end(B) MUST be at most start(A), with
the same zero-offset proof at equality. LivenessMetadata intervals themselves MUST
remain unchanged.

After accepting links, CARRY_INIT, CARRY_YIELD, and INSERT_SLICE candidates whose
holder and target already reach one root MUST be recorded as accepted,
including when another accepted edge established that storage identity. This
completion MUST NOT change roots or placement. Both links remain in the
summary, with their distinct partners and kinds. WINDOW is excluded from this completion.

WINDOW is the exception whose holder precedes its target. It MUST be accepted
only when the same InsertSlice's result and destination already share B, and
the update-to-result relation proves a window visible at the producer. A, the update root,
MUST differ from B, be neither persistent nor a formal boundary, occupy the same
level, and have every member's last use no later than the InsertSlice result's
definition. The update MUST fit in the result. No B member may be used strictly
between A's start and that definition; at A's start, any B read requires the
zero-offset proof above. Acceptance links A to B with the window relation. Window
acceptance MUST follow VIEW, FORWARD, and accepted CARRY_INIT edges from
the update to its storage root, reverse-composing those coordinate relations.
The resulting producer-to-destination map MUST cover the producer's complete
coordinate box and use only parameters visible at that producer; otherwise
the WINDOW MUST be rejected. The map MUST be a same-rank translation supported
by lowering: destination axis k equals producer axis k plus a producer-visible
start. Rank-changing maps and non-translations MUST be rejected, leaving an
independent update for lowering to copy at InsertSlice. Tuple links remain
indexed by flat leaf. When CARRY_YIELD and INSERT_SLICE both accept one value,
both links MUST remain in the summary. Rejected ties MUST contribute no accepted
link; lowering copies the corresponding carry or slice.

A callee root containing a formal or the returned storage root is a boundary
and MUST have no local address variable or published offsets. Read-only
zero-offset operand proofs, including formal operands and VIEW/FORWARD paths
to the return root, MAY declare an IN_PLACE capability (output index, formal
input leaf and coordinate relation) when formal occupancy ends no later than
output occupancy starts. The call site MUST register and judge these capabilities
with the same IN_PLACE rules as ordinary operations; a later use of the actual
MUST prevent reuse. These proofs MUST NOT create local
IN_PLACE aliases, link formal storage to a result, or publish metadata.

AllocationConstraintVisitor.run MUST return AllocationSummaryMetadata with
values, IN_PLACE capabilities and accepted address decisions. Its private
occupancy evidence retains translated windows; it MUST NOT retain a second
call graph. Reading a called summary through get_metadata(..., ctx=ctx) uses
the attached Scope and caches the bound result in its snapshot. Lifecycle
region events remain owned by LivenessMetadata. Time translation uses
`shifted(record, origin)`. Windows rooted at a formal MUST name the actual SSA leaf;
windows rooted at a return MUST name the Call result leaf. Internal SSA values
retain their identities and share one address across all translated windows.
The caller MUST NOT mutate AllocationValue aliases in a callee summary. Own
windows MUST reference the AllocationValue's lifetime object rather than copy it.
Implicit weights MUST use the union of instance indices. Only the entry solves
placement, once, after collecting its Call leaves; non-boundary callee offsets
MUST be published from that solution. Repeated region windows contribute their
maximum occupancy without traversing the callee body again.

##### `MemoryLevelPeak`

Capacity is settled against authored definition order, which fixes every
linked value's lifetime before any is measured. Whole-Function `gmem` and `smem`
peaks use exact polyhedral access relations to place addressable roots; the
Call results report their concrete offsets. A `MeshRegion` peak instead scans
root occupancy over its entry-to-exit window, counting each root once and
using the union of live address intervals for addressable levels, but only for
levels the region holds. In the target's ordered topology hierarchy, the region's
outermost level MUST be the target-declared owner or an enclosing level; a
region strictly inside the owner is not eligible to report that level, and a
target-owned level makes no `MeshRegion` eligible. Eligibility does not decide
attribution: every eligible region whose mesh contains the value's authored
holder mesh reports it, so an enclosing region also counts values held by its
nested regions. A value that declares no holder belongs to no region; analysis
does not infer a holder from its layout or storage name. `rmem` is not
address-solved: the Function reports the maximum of all stage peaks and a
whole-Function live-byte scan of values that no `MeshRegion` claims.

At each event, `rmem` counts a root once, using its greatest live member span.
`gmem` and `smem` window occupancy and persistent bytes MUST count the union of
live physical address intervals. A repeated callee region reports the maximum
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
| `MemoryLevelPeak.peak_bytes` | On a Function, the address high-water mark of the first feasible global `gmem` or `smem` placement; its `rmem` peak is the maximum of every `MeshRegion` stage peak and the whole-Function live-byte peak of values no mesh region claims. On a `MeshRegion`, the greatest root occupancy in its inclusive entry-to-exit event window. Unsplit tensor values defined in its body or child regions contribute while their lifetimes intersect the window, without region-versus-owner filtering. Split values retain owner eligibility and holder-mesh selection. Target-owned storage (`gmem`) never contributes to a region. | `MemoryHierarchyFacts.explicit_levels[].owner` for a `MeshRegion`; otherwise no. |
| `MemoryLevelPeak.persistent_bytes` | Union of persistent address intervals at `gmem`/`smem`; greatest persistent member span per root at other levels. | No |
| `MemoryLevelPeak.capacity_bytes` | Capacity of the matching explicit level, or `None` when unknown. | `MemoryHierarchyFacts.explicit_levels[].capacity_bytes` |
| `RegionMemoryMetadata.peaks` | One peak per occupied or moved storage level. | As above |
| `RegionMemoryMetadata.solver_status` | `"feasible"` after the whole-Function placement settles, including when there is no addressable value. | No |
| `RegionMemoryMetadata.errors` | Non-fatal placement-capacity failures and cache-capacity failures proved by exact or lower-bound counts. | `MemoryHierarchyFacts` |
| `RegionMemoryMetadata.advisories` | Cache-capacity findings whose upper-bound or unknown counts do not prove excess. | `MemoryHierarchyFacts` |

- constraints:
  - `RegionMemoryMetadata` MUST be attached per reachable `Function` and per
    `MeshRegion` with an eligible live value. Unsplit tensor values defined in
    the region body or child regions contribute over intersecting lifetimes,
    independently of the region's depth relative to the storage owner. Split
    values retain owner eligibility and declared-holder selection. Each root
    counts once; target-owned storage (`gmem`) is excluded. A mesh-region
    record carries only its topology names and per-level peaks; whole-Function
    traffic, footprints, reuse windows, lifetimes, errors, and advisories stay
    on the Function.
  - Every explicit level considered for a mesh-region peak MUST state an owner
    in `MemoryHierarchyFacts`. A missing owner is an error, not a reason to
    treat the level as shared by every region.
  - A split value MUST declare its holder mesh to belong to an eligible
    `MeshRegion`. No holder declaration MUST be inferred for a split value;
    unsplit tensor values use the definition and lifetime rule above.
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
  - Carry, InsertSlice, window and in-place sharing MUST follow
    the pre-solve value links and acceptance conditions above. Simultaneously
    live values in overlapping address intervals MUST belong to the same root;
    otherwise analysis MUST raise `AnalysisError` identifying both bindings,
    intervals, offsets, and level.
  - Every placed offset MUST be aligned to the greater of 16 bytes and the
    result element width. A staged result's copies MUST be contiguous: copy
    `k` starts at the solved block offset plus `k * buffer_bytes`.
  - Mandatory aliases, alignment, and lifetime interference that have no
    feasible joint placement MUST raise `AnalysisError`; analysis MUST NOT
    silently discard a required alias.
  - A domain that cannot be expressed or does not settle in time MUST raise
    `AnalysisError` and leave no record. The solver MUST stop at its first
    feasible assignment rather than prove a minimum. Capacity MUST NOT restrict
    the address space.
  - A seed is a complete candidate placement that analysis builds for itself
    before solving: a starting byte address for every placed value and the peak
    byte extent that placement reaches. It is not an input and not a random
    seed. The solver takes its addresses as a hint and its peak as an upper
    bound on the search. With no seed it solves the same model, every
    constraint kept, without those hints or that seed-derived upper bound.
  - A seed that hints or bounds the search MUST meet every address constraint
    the solver states: one address per root, alignment, disjoint
    addresses for interfering values, pinned persistent addresses, and the
    address limit. A seed that does not MUST NOT bound the search; failing to
    construct one proves nothing about the model.
  - A solved explicit-level peak exceeding capacity MUST add a non-fatal
    `errors` entry, preserve the complete result, and MUST NOT fail the call.

##### Cache occupancy

A cache holds data because it will be read again, and whether it still holds it
is decided by what was touched in between. This analysis states one row per
buffer that is read again, naming the two axes a second read can come from: the
loop whose next iteration reads it, and the mesh axis whose units read it at
once. A buffer whose units partially overlap also states a row even when
neither exact reuse axis exists; its savings are unknown. A buffer with none
of these reuse relations states no row.

| Field | How it is computed | Reads the target |
|---|---|---|
| `ReuseWindow.buffer` | The source buffer identity, rendered by the expression printer. | No |
| `ReuseWindow.time` | The outermost enclosing loop whose different iterations reach the same addresses, named by its induction variable; empty when no loop supplies a second read. | No |
| `ReuseWindow.space` | The mesh axes whose different coordinates reach the same or partially overlapping addresses; empty when no mesh axis supplies a second read. | No |
| `ReuseWindow.holds_bytes` | Unique bytes of every buffer the whole wave touches while this buffer must remain resident. | `MemoryHierarchyFacts` selects the backed level; [target §11](./target.md#11-target-facts-projection) supplies `TopologyFacts`. |
| `ReuseWindow.reuse_bytes` | `(time trips * space units - 1)` times this buffer's unique bytes in the window; an absent axis contributes one. Unknown (`None`, JSON `null`) for partial overlap whose savings are not modelled. | As above |
| `ReuseWindow.fits` | True exactly when `holds_bytes` is less than the cache capacity. | `MemoryHierarchyFacts.implicit_levels[]` |
| `ReuseWindow.precision` | The precision of the footprint supplying `holds_bytes`, without mixing in the trip upper bound used for `reuse_bytes`; a partial-overlap row instead has `unknown` precision. | No |
| `RegionMemoryMetadata.reuse_windows` | One row for every buffer with a time or space reuse axis and nonzero savings, or with partial overlap and unknown savings. | As above |

- constraints:
  - Partial-overlap detection MUST run only when time and space reuse supply no
    axis. Such a row MUST retain its holds bytes, report `reuse=unknown` and
    `precision=unknown`; JSON MUST use `null`
    for its reuse bytes. Existing exact reuse rows MUST remain unchanged.
  - An unknown-precision row exceeding capacity is an advisory. It suppresses
    the first-iteration working-set finding only when their precision reliability matches,
    so that finding MAY coexist with the row's advisory.
  - The stated bytes are everything the whole wave touches while that buffer
    must stay resident, every buffer included, not only the one read again:
    the others are what evict it.
  - Cache lines are not modelled. Reached addresses are packed by element
    width, so two accesses to neighbouring words of one hardware cache line
    remain two distinct touches here.
  - A read by another unit of the same wave counts. On a target whose units
    share one cache, data several units read at once is the common case, and a
    model counting only one unit returning later would state no reuse at all
    for a schedule giving each unit one output tile.
  - The parallel reuse axis is a TileFoundry extension. The reuse-distance
    model it follows (Falcon, PLDI 2024) states a single-threaded program;
    unioning addresses across concurrent units has neither a published model
    behind it nor hardware validation here.
  - A mesh axis supplies reuse exactly when one unit reaches the same addresses
    as the wave union along that axis. The spatial repeat count is how many
    units in the wave map onto those same addresses. The space comparison reads
    the wave relation from the device view and the one-unit relation from the
    unit view. The time axis reads the raw narrow relation, retaining mesh
    parameters and applying no wave limit. `holds_bytes` reads the narrow
    relation, limits it to the wave, and projects out mesh parameters.
  - Under nested mesh regions, a reuse row states its space axis against the
    innermost enclosing mesh. Outer execution meshes are not composed into
    that axis label.
  - The window is the loop axis when there is one, and the mesh axis alone
    otherwise, because units reading at once are already inside one iteration
    of the loop that carries the later read.
  - A time window is a loop-granularity over-approximation. A true reuse
    distance starts at the previous touch of the same cache line; this window
    covers a whole iteration of the carrying loop. Precision describes the
    counted window, not hardware cache misses or true byte reuse distance.
  - Rows MUST NOT be summed. One row's window lies inside another's whenever
    its axis is nested inside, so a row that fits implies those nested in it
    fit.
  - Rows MUST be ordered by `reuse_bytes`, largest first. Equal reuse amounts retain
    their derivation order.
  - Data read once states no row; its bytes still enter every row whose window
    contains it.
  - A row whose computed reuse is zero states no row.
  - A window at or above capacity MUST add one non-fatal diagnosis, regardless
    of how many buffer rows share it, and MUST NOT fail the call. `EXACT` and
    `LOWER_BOUND` conclusions belong to `errors`: their true holds are at least
    the stated capacity-exceeding amount. `UPPER_BOUND` and `UNKNOWN`
    conclusions belong to `advisories`: exceeding capacity is not proved.
    The same directional rule applies to the Function working set.
  - The capacity is stated per one instance of the cache's `scope`, and this
    analysis compares one wave against one instance. A deployment that spreads
    one wave across several instances is not modelled.
  - Per-unit control flow is out of scope: HIR states no conditional region, so
    two units differ only by the iteration domain a coordinate gives them.
  - This is the capacity judgement of an idealised fully associative LRU cache.
    Miss counts, miss rates and hardware replacement policy are not modelled.

Cache level and capacity are Target facts, not part of `RegionMemoryMetadata`,
and appear in this report only when they decide a cache-capacity finding.

##### Target facts

The family reads the target hierarchy and physical unit counts stated once in
[target §11](./target.md#11-target-facts-projection), because `memory` and
`performance` both consume them.

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
memory traffic=<memory-level>:r<bytes>/w<bytes>@logical,r<bytes>/w<bytes>@total,r<bytes>/w<bytes>@<topology>[,...] footprint=<buffer>:<bytes>[;<buffer>:<bytes>] footprint-precision=<exact|upper_bound|lower_bound|unknown> peak=<level>:<bytes>[,...] persistent=<level>:<bytes>[,...]
  buffer=<buffer> holds=<bytes> time=<loop|none> space=<mesh-axis|none> reuse=<bytes> fits=<yes|no>
  error="<text>"
  advisory="<text>"
```

A Function's memory record counts that Function and everything it reaches, so
the per-Function lines of one report MUST NOT be summed.

The `<bytes>` form selects the largest of `B`, `KiB`, `MiB`, and `GiB` whose unit
the value reaches, with 1024 between adjacent units. `B` is an integer; every
larger unit has two decimal places. Zero is `0`. This one form is used for every
human-readable byte count, including error text; JSON keeps raw byte integers.

Each buffer, error, and advisory is its own line indented beneath the `memory`
family line. Errors and advisories are quoted and escaped. Every measured Call
receives a `memory` annotation; `operands` is emitted only when asked for
([cli Analyze](./cli.md#analyze)):

```text
memory traffic=<memory-level>:r<bytes>/w<bytes>@logical,r<bytes>/w<bytes>@total,r<bytes>/w<bytes>@<topology>[,...] footprint=<buffer>:<bytes>[;<buffer>:<bytes>] footprint-precision=<exact|upper_bound|lower_bound|unknown> [operands=<position>:r<bytes>/w<bytes>[;<position>:...]]
```

In the printed `footprint` field, a buffer's value label may itself contain `:`
(for example, `v0:57:1.00KiB`). The byte count is the formatted value after the
last colon.
A root that is a declared parameter keeps its own name; otherwise use the
first capturing parameter, in program order, that resolves to that root;
otherwise use the existing value label. Duplicate-label suffixes remain
unchanged.

Missing optional conclusions omit their whole printed field. Call and Function
JSON projections are both under `memory`. The Function's full projection is
under `function_records.memory`:

```text
{"topologies": [<level>, ...],
 "traffic": {"storage": {<memory-level>: <spread>, ...},
             "communication": {<topology-level>: <spread>, ...}},
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
| `compute_ns` | For each recorded dtype with a published rate, round `flops * 1e9 / rate` up to ns and sum the dtype times. A Function uses its summed flops, not a sum of per-Call times. | `ThroughputFacts.peak_flops_per_second`, asked for the Function's active units |
| `memory_ns` | Sum footprint `total` bytes at `bandwidth_level`, multiply by `1e9 / memory_bandwidth_bytes_per_second`, and round up to ns; zero when no bandwidth is published or no bytes are counted. Reads and writes are one count. When the footprint is absent or counts another level, use traffic `total` reads plus writes at `bandwidth_level`. A Function uses its accumulated footprint or traffic, not a sum of per-Call times. | `ThroughputFacts.bandwidth_level` and `memory_bandwidth_bytes_per_second`, asked for the Function's active units |
| `ideal_ns` | Maximum of `compute_ns` and `memory_ns`; one ns when the occurrence records nonzero flops or nonzero priced `bandwidth_level` bytes and neither published rate yields a bound, otherwise zero. Bytes at any other level are stated and does not earn a bound: no rate was published for it, so none is owed. | Through the two times |
| `bound_by` | `none` for no bound, which includes an occurrence whose only movement is at a level with no published bandwidth, `balanced` for equal nonzero times, `memory` when memory is greater, `compute` when compute is greater, and `unrated` for the one-ns bound owed by work this prices whose rate is missing. | Through the two times |

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
`(topology_level, active, inflight)`, where `active` is the most positions the
Function's meshes run at that level and `inflight` is a Call's bytes in flight,
as performance states them; the Function uses the most any of its own Calls
keeps, so its bound stays a lower bound. A call into another definition is not
an access and keeps nothing in flight: that definition prices its own. The rates are those of the units the program
runs, and a query of no level is the whole deployment. What one unit gets through is a separate projection, read
by `performance` rather than by `roofline`: work that is not floating point at
all has its own published instruction throughput and no dtype to be filed under.

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
  - Roofline MUST price footprint `total` at `bandwidth_level` without applying
    any further repetition factor. An absent footprint or one counting another
    level MUST fall back to traffic `total` reads plus writes at that level. A
    footprint with omitted boundaries MUST still be used as its stated lower
    bound. No other memory level may enter the bandwidth calculation.
  - Footprint pricing assumes ideal reuse within one wave and repeats bytes
    across windows. Counting a partial last wave as full is a pessimistic
    approximation; deduplication within one wave is optimistic. Below the
    device's parallel topology level, each parent instance's windows are
    counted separately, overestimating bytes when parents share data.
  - Performance local duration MUST divide one unit's share of
    `ComputeCostMetadata.flops` by `unit_flops`, of `service` by `unit_ops`, and
    of the bytes at each storage level by `unit_bandwidth`, all at the level it
    was asked about, for the Function's active units: performance asks for
    `PerformanceServiceFacts` with `(topology_level, active, inflight)`, `active`
    capped at the level's `max_physical_units`. `inflight` is the bytes one
    parallel unit keeps in flight for the occurrence: an asynchronous copy
    (`category="async"`, such as TMA or `cp.async`) keeps one unit's bytes at
    `bandwidth_level` times the slots its buffer declares, and any other access
    16 bytes per thread of the unit, as many threads as the finest level allows
    when the program states none. At `bandwidth_level`, one unit's bytes are
    the occurrence's footprint `logical` bytes, reads and writes as one count,
    divided among the positions that window holds -- its executing positions,
    capped at `active` -- and rounded up; the roofline reads the same footprint.
    Without a footprint, the share is the `bandwidth_level` entry of
    `MemoryMetadata.traffic.storage`. A window every position writes alike then
    costs the window over the positions, and disjoint shards cost each
    position's own shard. Compute and movement overlap within one occurrence,
    so its duration is the greater of the two sides rather than their sum.
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
  - Roofline records MUST be attached to reached `Call`s and Functions.
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
        wave: attribute; Positions the Function's meshes run, and the level's physical units.
    """

    timeline: TimelineMetadata
    wave: tuple[int, int]
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
| `timeline` | `[0, local makespan * waves)`, where the local makespan is the end of the CTA-local timeline, or zero with no work, and `waves` is `ceil(N / P)`: `N` is the most positions the Function's meshes run at the level selected by `TopologyFacts.parallel_level` and `P` is that level's `max_physical_units`. Its duration is the prediction; `waves` is not a field. | `TopologyFacts` |

Occurrence intervals remain CTA-local. They are not copied once per wave, and
neither the executing positions nor `max_physical_units` changes them. The capacity
`P` is the machine-side position count; the mapping of one CTA to one SM is
compiler policy. The
aggregate itself is stated once in [target §11](./target.md#11-target-facts-projection),
because `memory` reads it too.

Requesting performance adds this Function verdict to the summary:

```text
performance root=<Module>::<Function> predicted-ns=<int> wave=<active>/<physical>
```

`root` is the report's own identity, composed by inspection from the module and
function it already states; it is not a field of `PerformanceSummaryMetadata` and
does not appear in that record's JSON projection. `predicted-ns` is the duration of
the summary's own envelope, not a second measurement. `wave` is the summary's
`wave`: `active` is the most positions any mesh the Function executes runs at the
selected topology level and the coarser ones, 1 when it states no mesh, and
`physical` is that level's `TopologyLevelFacts.max_physical_units`. Forty
positions on a 132-unit level read `wave=40/132`, and 8192 read `wave=8192/132`.
It states where the program's placement reaches, not an occupancy guarantee.

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
           "wave": [<active>, <physical>]}
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
  - A Call's scheduled duration MUST multiply every enclosing authored loop's
    trip count, including loops in which the Call is invariant. Work explicitly
    defined outside a loop MUST keep its defining scope's repetition count.
    Mesh positions MUST NOT multiply this duration: units execute in parallel.
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
  - Occurrences MUST be laid out in definition-relative order. Reordering
    independent work is a later decision, not an analysis's: what overlaps
    is what the program's own placement made independent, and the reported time
    is the time of the program as written. On a `Function`, the summary's
    `timeline` MUST start at zero and span the whole local plan, scaled by
    `waves`; there MUST be no second field restating the local makespan or the
    scaled estimate, and none restating how the layout was reached -- it is
    exact for the model it states.
  - `max_physical_units` MUST NOT enter the CTA-local layout, and is not a
    program rewrite; it enters one-unit rates only as the cap on `active`.
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
    dims: "Mapping[str, int] | None" = None,
    topology_level: str | None = None,
    budget: int = _NODE_BUDGET,
    analyzers: tuple["Analyzer", ...] = (),
) -> tuple["Module", "Function"]: ...


class AnalysisCheckContext(ValuationContext):
    """What every input check reads: the program, the machine, and the topology level.

    Costing here is a question, not a record: nothing a context computes is
    attached.
    """

    module: "Module"
    function: "Function"
    target: "Target"
    topology_level: str | None

```

- constraints:
  - The operation MUST infer types over the full reachable Function graph and
    validate its caller/callee execution context, and MUST NOT run an analysis
    or attach derived Metadata to the authored IR. Each derived body it returns
    is its own copy: its types are inferred and stored on it under the
    program's Target and arguments, so analyses read that view's `Expr.type`
    rather than a stamp of a shared child body.
  - Dimension binding MUST follow the Analyze constraints below, using
    [hir §2](./hir.md#2-function-specialization-api)
    before ordinary type inference with ranges. Remaining symbolic dimensions
    MUST be named before any consuming algorithm runs.
  - Every effective Module topology MUST name a level the resolved Target
    supports. A resolved static extent MUST be positive and within that level's
    finite hardware limit. A rejection MUST name the level, its extent, and the
    reason.
  - A non-`None` `topology_level` MUST name exactly one effective Module topology.
  - Callers MUST build the program before invoking Analyze. Analyze MUST run
    the requested closure's checkers on that same program before any metadata
    writes. `check_program(analyzers=...)` and Analyze use the same checker gate:
    each checker's `check_target_facts`, followed by `visit` on each reached
    definition. The first refusal MUST stop the gate before any family writes.
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
    `AnalysisError` naming the callee, both call paths, and both signatures.
  - Check MUST attach internal `ExecutionMeshMetadata` to every derived
    Function, recording its unique enclosing mesh; root records `None`. This
    record is not exported by `tilefoundry.analysis` and does not opt into printing.
  - Records MUST live on IR once per definition, in local coordinates. The
    definition's scope tree MUST stop at its own root, whose parent is `None`.
    Source entries MUST stop at formals; formals' variance dependencies MUST
    include their own identity. Only called definitions retain formal-dependent
    coordinate parameters. The selected root has no caller to bind them and
    MUST use the existing widest-range approximation with `UPPER_BOUND` precision.
    Each definition publishes its own one-execution summary; caller records
    compose those summaries at Call boundaries.
    Performance MUST schedule each definition once relative to its own origin;
    call duration is the callee duration, scheduled in the caller's current scope.
    Callee region peaks MUST take the maximum across occurrences.
  - Parameters and arguments MUST share storage and extend argument liveness
    through the callee's last use; callee temporaries occupy the caller timeline.
    A separately materialized callee result of matching memory level and byte
    size MUST share the caller destination. A return without an independent
    material buffer requires a caller destination, which MAY increase peak memory.
  - Implicit weights MUST be persistent. Their instance index tuples are the
    union of reached index images clipped to slot bounds; their cardinality
    scales weight bytes. Unknown indices reach the whole slot. Traffic uses
    loop trips without this multiplier.
  - Each internal SSA leaf MUST share one address across its occurrence windows.
    Callee parameters and results tied to caller destinations use caller pointers
    and carry no offsets. Internal values MUST NOT reuse parameter storage;
    a return group's proven reuse of a parameter is applied only at the call,
    subject to the caller's persistent and last-use checks. Different alias
    relationships MUST raise `AnalysisError` naming the callee, binding,
    both binding paths, and relationships.
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
    attached child no call reaches has no edge here. Both effective targets
    and complete topology hierarchies MUST match.

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
        functions: attribute; Reached derived functions in root-first DFS order.
    """

    module: "Module"
    function: "Function"
    analyses: tuple[str, ...]
    topology_level: str | None
    executed: tuple[str, ...]
    metadata_types: tuple[type[IRMetadata], ...]
    functions: tuple["Function", ...]


def analyze(
    module: "Module",
    function: "Function",
    *,
    analysis: str | Iterable[str],
    topology_level: str | None = None,
    options: object | None = None,
) -> AnalysisResult: ...
```

- constraints:
  - One call MUST select one or more root analyses. It MUST preserve their
    first-occurrence order, resolve their union dependency closure, and execute
    every member once.
  - `topology_level` MUST name one effective Module topology. When omitted, it
    MUST default to the coarsest effective topology whose `TopologyLevelFacts`
    states `max_physical_units`. A program may name a level the host places
    rather than the machine runs -- several cards are one deployment's shape,
    not one card's -- and measuring per such a level would ask the machine for
    a unit it publishes no rate for. When the target answers for none of the
    declared levels the coarsest MUST be left selected, so the refusal names
    the level rather than the absence of one; when the Module declares none,
    `topology_level` MUST remain `None` and no per-unit projection divides.
    `AnalysisResult.topology_level` MUST record the resolved answer.
  - A target MUST answer `PerformanceServiceFacts` and state
    `TopologyLevelFacts.max_physical_units` for the level it is asked about,
    and MUST refuse a level it publishes no rate for rather than answering for a different one. The query is
    the level name, or `(level, active)` when the program runs `active` of its
    units; a target MAY ignore `active`. What one unit gets through is the
    deployment's rate for the active units shared among them, so the same
    program measured at two levels states the same work against proportionally
    different rates.
  - The Function MUST belong to the checked Module, including its derived
    definitions. Analyze MUST write metadata in place without cloning or
    re-running type inference. A root without `ExecutionMeshMetadata` MUST be
    rejected with `AnalysisError("analyze needs the program returned by check_program")`.
    Dimension binding belongs to `check_program`.
  - Within `check_program`, stated `dims` MUST be a non-empty mapping whose
    keys name reached dimensions and whose integer extents satisfy their bounds.
    Every dispatch dimension MUST be supplied, exactly one variant MUST cover
    it, and no dimension MAY remain symbolic after substitution in types, Meshes,
    loop bounds, attributes, or effective topologies. Invalid bindings and missing
    bodies MUST fail with their names before any consuming algorithm runs.
  - `AnalysisResult.function` and `AnalysisResult.module` MUST be the exact
    objects supplied by the caller. The checked Function retains its origin and
    bound extents; analysis does not return the symbolic authored input.
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
  - Type inference and structural validation belong to `check_program` and
    MUST complete before Analyze receives the program. Analysis reads those
    inferred types and assumes a verified function.
  - Family-specific readiness MUST be checked on that inferred derived program and
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

### Visit context and local coordinates

`get_metadata(expr, kind, ctx=ctx)` delegates to `IRMetadata.resolve(ctx)`
and the already-bound Scope; its default
returns the record unchanged. Metadata does not declare capability requirements.
Analysis contexts extend `TypeInferContext`, not `CostContext`. The context
prepares the callee and argument bindings. The context's `enter_call` calls the caller scope's `attach` operation and
records the call edge. Reading `callee_scope` follows that existing edge;
it does not recursively bind a definition tree. Consumers query `ctx.scope`,
`ctx.scope.provenance`, and `ctx.scope.access` without an additional attach step.
`scope.root` identifies the current definition. `scope.parent` always means the
lexically enclosing scope; it MUST NOT be repurposed as a caller link.
`call_site` identifies a Call, `caller_context` the caller environment, and
`caller_scope`/`callee_scope` refer to scopes rather than functions.

Analyze drives each family once per reached definition, dependencies first.
Definition-local snapshots remain symbolic and shared. A bound callee view
substitutes actuals without mutating the definition tree. Calls do not create
lexical scopes or trigger a second analysis of their callee body.

| Record | Interpretation |
|---|---|
| ComputeCostMetadata | Local execution counts; caller loop scaling occurs during accumulation. |
| MemoryMetadata | Local traffic; retained Calls have no independent boundary footprint. Callee accesses are translated once while building caller scopes. |
| LivenessMetadata | Local event records; `at(origin)` uses the common `shifted` operation. |
| AllocationSummaryMetadata | Values, in-place capabilities and unchanged raw address decisions. Bound occupancy is interpreted through the attached Scope and cached in its snapshot; consumers do not recursively expand calls. |
| PerformanceMetadata / PerformanceSummaryMetadata | Local intervals; the performance family handles their origin, not Scope. |

`Spread.logical_groups` is
`tuple[tuple[tuple[Var, ...], V], ...] = ()`. `logical` is derived by summing
its groups, or the typed zero for empty groups. Call groups name local induction
variables and formals; Function accumulation consumes local induction variables
and retains only the formals on which the count depends. Caller Function
accumulation multiplies it by `scope.dependency_trips` for that set, preserving the joined precision. Call resolution MUST NOT apply caller
loop multiplicities to either work or traffic. Renderers MUST show only `logical`,
`total`, and `per_unit`, and MUST NOT render the groups.

Footprint evidence retains local reached sets, their `AccessRelation.values`
expressions, and groups of reached sets with local window multiplicities.
Display names belong to the expression printer. Metadata references retain
SSAValue identities and reuse the same printer bindings as the surrounding IR;
cross-function references MUST use `function::local_name`, while references within
the current function keep their local name. This prevents distinct buffers with
the same local name from overwriting one another in reports. Within a function,
ambiguous SSA references MUST append the printed definition line; a loop-carried
parameter uses the loop header line. The code keeps its original local names;
analysis does not generate buffer display strings or cache a separate name map.
Loop coordinates MUST remain bounded parameters, named by induction identity,
with values referring to their induction Vars. Resolve substitutes formals and
retains callee loop parameters. Runtime scalar bindings remain symbolic until
the numeric window is selected. Function and Call footprint numbers hold every
loop parameter at its first iteration. A caller reuse window instead unions
all iterations inside its selected window, including complete callee executions;
loops outside that window stay at their first iteration. Caller candidates
remain its own axes. These fields support substitution and MUST NOT be rendered. They MUST NOT
retain a scope or context. Numeric views of unresolved
formal-dependent coordinates MUST use the existing widest legal bound and
state `UPPER_BOUND`; call-site counts are computed after substitution.

Reuse windows MUST be computed once per definition. Callee internal loops are
candidates only on that callee's Function record; caller records use only the
caller's own loops and mesh axes. Their window footprint includes resolved
callee reached sets. A callee's unresolved formal coordinates use their widest
legal bound and state `UPPER_BOUND`. ReuseWindow retains its seven numeric and
label fields and is not resolved through calls. Local source values use local
variance. A callee-internal source absent from that table conservatively follows
the current Call's variance; implicit const weights remain invariant, and
instance-index dependence is tested through the reached image. This may omit
reuse for an internal temporary that depends only on constants.

LivenessMetadata is internal metadata, not a report family. Each caller reserves a
callee template's event span and records actual leaf uses at the translated
formal-use events, preserving synthetic flags. Resolve only shifts the local
template; it stores neither child templates nor a second timeline origin.
Allocation may resolve the callee template at a Call to obtain internal windows.
Standalone liveness for a function without Calls MUST work without a scope
tree. It MUST NOT use an attached LivenessMetadata record as an implicit result cache.

Implicit weights MUST be counted by `(slot key chain, weight)` and the union of
reached instance-index tuples. Caller induction variables map from the loop
domain to those tuples; textual instance labels MUST NOT determine counts.
Footprint coordinates prefix instance indices before unioning element addresses,
and persistent bytes count each reached instance once.

Callee call durations MUST use the resolved callee summary's duration divided
by its waves. Non-entry summaries use the local schedule without a roofline
floor; only the entry prediction applies its roofline floor. Callee summary
waves still follow the checked definition mesh.

Root footprint, reuse windows, lifetimes and whole-program peaks MUST be
absolute records. MeshRegion peak rows MUST also appear in report JSON under
`meshes`, parallel to `calls` and `loops`. Callee rows display local records.
Call records are already resolved in their own definition; reporting MUST read them directly without rebuilding scope
trees or resolving them again.

### 2.1 Shared IterationScope and Access

Each reached definition is visited once to build its IterationScope tree.
`ValueProvenance` and `AccessAnalysis` are defined beside IterationScope in
`analysis/iteration_scope.py` and composed as `scope.provenance` and `scope.access`.
The first implements source/dependency queries; the second implements access
and projection queries. Both operate on the scope's single `snapshot`:

| Snapshot section | Ownership and lifetime |
|---|---|
| `definition` | Shared definition facts: value snapshots, access records, relations, refusal evidence, and operation locations. |
| `attachments` | Call edges in this calling environment, including the callee reference, arguments, and instance identity. |
| `resolved` | Value, parameter, and relation results in this fixed lexical and calling environment. |

Attach MUST NOT recursively copy the definition tree or project every child
domain. A lexical entry is obtained only when needed, shares definition facts,
and keeps derived results separate from other calling environments. Binding
changes require a new environment rather than reuse of stale results.
`scope.at(path)` follows existing call edges and definition location indexes;
it does not rescan the callee tree or initiate another callee analysis.
Scope itself owns lexical structure, the iteration domain, and multiplicities.

`snapshot.definition.values[SSAValue]` contains one SnapshotEntry with `root`, `narrow`, `device`,
`unit`, `varies`, and optional `forward`. `forward` is a direct structural operand,
not the data-identity root. Three views remain supported; a missing unit view is
unknown. Types and display names are not duplicated in snapshots.
`domain` is an AccessRelation of zero-dimensional tuples with constraints on
identity-named loop parameters. `first()` constrains each coordinate to its
loop's start. `compose`, `constrain`, and `project` maintain parameter mappings
in AccessRelation; consumers do not maintain parallel parameter-name tables.
`SSAValue` is defined in ir/core/expr.py, and exposes `leaves` and `leaf`.

An input `Access` stores its original Call boundary index as well as its relation
and allocation expression; an output stores `input_index=None` and its own output
boundary index, which is the tuple field it answers for. Failed boundaries remain
absent without shifting the index on later successful inputs or outputs. Storage
level and element width are read from `Access.target.expr` at
`Access.target.index`. Sources MUST be grouped by `SSAValue`; a tuple result's
label is `<name>[index]`. A whole-tuple input boundary uses leaf-slot coordinates;
`resolve_access` MUST expand each possibly selected slot into a separate Access
in that leaf's legal tensor-element coordinates. These accesses retain the
original boundary index and may-access flag; precision is produced at counting time, and their target remains
the definition-local tuple leaf. No sentinel leaf index represents a whole tuple. A refused descendant
makes its owning scope unknown for that view. Non-affine runtime indices retain
the widest legal access approximation.
Normalization shares one derived body and its records per reached definition
([§2](#2-composed-analysis)).

`analysis.loop_terms` resolves HIR values to `LoopTerm` without depending on
isl. `analysis.access` turns those terms into isl constraints and owns access
widening; the two modules do not define a second affine graph representation.

- constraints:
  - A loop `start` or `extent` MAY be unit-dependent. Every runtime leaf in one
    MUST carry a half-open value range, and `IterationScope.domain` MUST keep the
    whole affine expression with each such leaf as one identity-deduplicated
    isl parameter constrained by that range. A leaf without a range MUST be
    refused.
  - `IterationScope.domain.values` MUST map each of those parameter names back
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
    report unknown. An unbounded set dimension MUST also report unknown.
    Named `union` parameters instead reduce by set union before maximizing over
    the other parameters. Small union boxes MUST fix each fiber before projecting
    those parameters, then union and coalesce the images. An oversized or
    unbounded union box MUST fall back to existential projection and counting;
    it MUST NOT approximate the reached set.
  - `IterationScope.trips()` MUST count a loop off its bounds, not its domain:
    `max(1, ceil(max(stop - start) / step))`, refusing an unbounded span.
    `trips_precision` is `EXACT` where that span reaches one value and
    `WIDENED` where it reaches several, the count then being an upper bound.

### 2.2 Target-selected Analyzers

```python
class AnalyzeContext:
    module: Module
    target: Target
    topology_level: str | None
    options: object | None
    scope: IterationScope | None = None
    current_mesh: Mesh | None = None

    @property
    def topologies(self) -> tuple[Topology, ...]: ...
    def type_of(self, expr: Expr) -> Type: ...
    def local_type_of(self, expr: Expr) -> Type: ...


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
    caller options, and the shared `scope` view (`scope.root` for the definition). The
    `topology_level` MAY be `None` only when the Module declares no topology;
    options MAY be `None`.
  - The same `AnalyzeContext` is what analysis reads types and access relations
    through. It is created before the scope tree, which reads relations through
    it with no topology level, and its `root`/`current` are bound to that tree
    before any analysis runs. `type_of` returns the `Expr.type` that
    `check_program` stored on the analyzed copy; `local_type_of` projects it to
    `topology_level` over the Module's topologies, or returns it whole without
    one. `current_mesh` is `None`: relations are asked outside any execution
    mesh.
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
