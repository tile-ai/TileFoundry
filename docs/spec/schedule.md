# TileFoundry Spec — Schedule

A Schedule is a Target-owned service over one explicitly selected stage of typed
HIR. It reads a `Module` and one root `Function`, then returns one `Module` plus
a stable objective summary. Whether that module is a materialization is the
stage's own contract: a stage that rewrites the program returns the rewritten
module, and a stage that only decides over it returns the module it was given.
Scheduling is separate from pass sequencing: callers select the service by stage
name, and the service owns the stage-specific algorithm.

The service surface (§1–§2) is the public `schedule` package. The construction
stages a service composes it from (§4) are imported from their own modules; they
read [analysis](./analysis.md) facts, and the dependency is one-way — nothing in
the analysis layer imports the schedule layer.

## 1. Direct service invocation

The Module's resolved Target is the sole owner of service selection. A caller
selects an exact stage and invokes the returned service directly:

```python
# example
service = module.resolve_target().service(Schedule, stage)
result = service.solve(module, root)
```

- constraints:
  - The caller MUST select a non-empty stage string explicitly.
  - Target service lookup MUST match the requested stage exactly and MUST NOT
    infer a stage from layouts, topology, or constraints.
  - The service MUST come from `module.resolve_target()`
    ([core-ir §1](./core-ir.md#1-module)); a call MUST NOT override the Module's
    resolved Target.
  - Scheduling lookup MUST NOT fall back to a default Target when no Module in
    the owner chain declares one ([target §6](./target.md#6-target-ownership-and-compile-resolution)).
  - `root` MUST be `module.entry_function()` for the CTA service.
  - A service MAY accept `options=None`; this MUST mean a fresh default
    `ScheduleOptions()` value for that invocation.

## 2. Public structures

### 2.1 `ScheduleOptions`

`ScheduleOptions` carries runtime controls shared by all schedule services.

```python
class ScheduleOptions:
    """Configure one schedule service call.

    Attributes:
        timeout_seconds: attribute; Wall-clock budget for the underlying solver.
        workers: attribute; Solver worker count, where zero selects the solver default.
        random_seed: attribute; Deterministic solver tie-break seed.
        debug_dump_dir: attribute; Directory for stage-private artifacts, or None.
    """

    timeout_seconds: float = 60.0
    workers: int = 0
    random_seed: int = 0
    debug_dump_dir: Path | None = None
```

- constraints:
  - The structure MUST be immutable.
  - `debug_dump_dir` MUST affect artifact emission only and MUST NOT change the
    selected result.

### 2.2 `ScheduleResult`

`ScheduleResult` is the complete public result of a service call.

```python
class ScheduleResult:
    """Carry one HIR module and its summary report.

    Attributes:
        module: attribute; HIR module the reported solve applies to.
        report: attribute; Stable cross-stage objective summary.
    """

    module: Module
    report: ScheduleReport
```

- constraints:
  - The structure MUST be immutable.
  - `module` MUST be the output of the same solve `report` represents: the
    materialized module for a stage that materializes, and the module the caller
    passed in for a stage that materializes nothing at its level.
  - A stage that materializes MUST return a verified module.

### 2.3 `ScheduleReport`

`ScheduleReport` is the stable cross-stage makespan summary. Stage-private
operation rows, use decisions, candidate costs, and solver-native models are
not report fields.

```python
class ScheduleReport:
    """Summarize the selected objective and proof state.

    Attributes:
        root: attribute; Scheduled root Function name.
        target: attribute; Root Target name.
        stage: attribute; Exact stage key that produced the result.
        status: attribute; Public solution status.
        objective_name: attribute; Primary objective name.
        unit: attribute; Unit of the primary objective and bound.
        selected: attribute; Selected primary-objective value.
        best_bound: attribute; Integer lower bound for the makespan objective.
        gap: attribute; Relative optimality gap for the makespan objective.
    """

    root: str
    target: str
    stage: str
    status: Literal["OPTIMAL", "FEASIBLE_NOT_PROVEN"]
    objective_name: Literal["makespan"]
    unit: Literal["ns"]
    selected: int
    best_bound: int
    gap: float

    def to_json(self) -> str: ...

    def to_markdown(self) -> str: ...
```

- constraints:
  - The structure MUST be immutable.
  - `selected`, `best_bound`, and `gap` MUST describe the one `makespan`
    objective in `ns`.
  - `objective_name` is a fixed literal: every stage reports the same one
    quantity, and the field names it rather than selecting it.
  - The reported value MUST NOT be read as a proof that a solver optimised it. A
    stage MAY minimize the makespan as a solver objective, and a stage MAY
    instead compute it after the fact — as the nominal roofline estimate of the
    decisions it recorded (§5.1). `status`,
    `best_bound` and `gap` are what distinguish the two: a stage that did not
    prove optimality MUST NOT report `selected` as its own bound.
  - A feasible incumbent MUST be reportable even when optimality is unproven.
  - JSON and Markdown rendering MUST contain every public field and MUST NOT
    expose stage-private solve state.
  - When the report summarizes a CTA execution blueprint, its selected value
    describes the modeled plan and MUST NOT be interpreted as a guarantee that
    current lowering preserves the planned intervals, reserves physical SM
    shares, promotes values to SRAM or registers, or proves runtime performance
    or out-of-memory safety.

### 2.4 `Schedule`

`Schedule` is the structural interface registered by a Target for one stage.

```python
class Schedule(Protocol):
    """Solve one named scheduling stage.

    Attributes:
        stage: attribute; Exact Target service key for this implementation.
    """

    stage: str

    def solve(
        self,
        module: Module,
        root: Function,
        options: ScheduleOptions | None = None,
    ) -> ScheduleResult: ...
```

- constraints:
  - `stage` MUST equal the exact key under which the service is registered.
  - `solve` MUST read the supplied HIR directly and MUST return one
    `ScheduleResult`.
  - Stage-specific candidate rows, cost data, solver state, and materialization
    helpers MUST remain private to the concrete service.

## 3. Constraint metadata

Hard schedule constraints are represented by one stage-neutral
`ScheduleConstraintMetadata` record attached to the constrained HIR
expression. The record contains zero or one `LayoutConstraint`,
`MeshConstraint`, and `StorageConstraint` value, represented by the existing
constraint base and source-location fields.

```python
class LayoutConstraint(ScheduleConstraint):
    """Fix a physical Layout pattern and ShardAttr bindings."""

    layout: Layout
    bindings: tuple[tuple[str, ShardAttr], ...]

class MeshConstraint(ScheduleConstraint):
    """Filter an eventual ShardLayout by one Mesh value."""

    mesh: Mesh

class StorageConstraint(ScheduleConstraint):
    """Filter a value by one current StorageKind."""

    storage: StorageKind
```

`LayoutConstraint.layout` is constraint-owned and may contain the private
wildcard sentinel. Its `bindings` reuse `Split`, `Broadcast`, and `Partial`
from [shard](./shard.md). A wildcard is never stored as `Layout(None)` and
never enters a `TensorType.layout`. Metadata is not part of expression
equality, hashing, or the printed `repr`.

These values are hard filters for later scheduling stages. They carry no
preferences, candidate rows, costs, solver state, or CTA capability
decisions, and they do not register a scheduling service on a `CudaTarget`.

## 4. Kernel schedule construction

A stage service composes its solve from three stages over one
`TileGraph` ([analysis §1.2](./analysis.md#12-tilegraph)). Each takes the graph
and returns it enriched; none of them re-derives a fact the analysis layer
already states.

```text
extract(root)  ──▶  build_schedule_tree  ──▶  select_atoms  ──▶  emit_scaffold
  (analysis)          tg.tree                 tg.tree tiled       Skeleton
                                              tg.ring             Swimlane
                                              tg.decisions        HoleContract...
```

Each stage lives in its own module of the `schedule` package and is imported
from there; the compact public package surface (§1–§2) carries the service
protocol only.

| Stage | Signature | Error |
|---|---|---|
| tree construction | `build_schedule_tree(tg: TileGraph) -> TileGraph` | `KernelScheduleError` |
| atom selection | `select_atoms(tg: TileGraph, target=None, stage="cta") -> TileGraph` | `AtomSelectionError` |
| scaffold emission | `emit_scaffold(tg: TileGraph) -> tuple[Skeleton, Swimlane, list[HoleContract]]` | `EmitScaffoldError` |

### 4.1 Tree construction

```python
def build_schedule_tree(tg: TileGraph) -> TileGraph: ...

def schedule_bands(tree: "isl.schedule") -> tuple["isl.schedule_node_band", ...]: ...

def band_statement(band: "isl.schedule_node_band") -> str: ...

def tile_band(band: "isl.schedule_node_band", sizes: tuple[int, ...]) -> "isl.schedule": ...

def tile_bands(tree: "isl.schedule", sizes: dict[str, tuple[int, ...]]) -> "isl.schedule": ...

class KernelScheduleError(RuntimeError):
    """A schedule tree the band operations cannot work on."""
```

- constraints:
  - Nothing in this stage solves. The tree MUST be **constructed** from the
    statement order the analysis layer already reports: one identity band per
    statement, sequenced in `tg.units` order. That order respects every
    dependence, so the result is legal by construction.
  - An affine scheduling solve MUST NOT be introduced here, and no objective MAY
    be smuggled in from isl's own schedule constraints: its
    dependence-distance goal is not the one this layer decides for.
  - The statements MUST NOT be fused into one band: their ranks differ, so one
    padded shared band member would mean a different loop in each of them.
  - Each band's `coincident` members MUST be written from
    `tg.parallel_dims` ([analysis §1.7](./analysis.md#17-parallel-dimensions)) —
    the flags are read, never recomputed.
  - `build_schedule_tree` MUST return `tg` with `tree` filled in and every other
    field unchanged. An empty `tg.units`, or a unit with no matching piece of
    `tg.domain`, MUST raise `KernelScheduleError`.
  - `schedule_bands` MUST return every band of the tree in top-down order, which
    for a constructed tree is `tg.units` order, and MUST raise when the tree
    carries no band.
  - `band_statement` MUST raise unless the band belongs to exactly one
    statement.
  - `tile_band` MUST split one band into a tile band over `sizes` plus a point
    band holding the remainder. A size count that does not match the band's
    member count, or a size below `1`, MUST raise.
  - `tile_bands` MUST tile every band by its own statement's sizes and MUST
    raise for a statement with no decided size.

### 4.2 Atom selection

`select_atoms` is the one real decision of this layer. Every operation carries
its own extent and that extent **is** its tile: what the author wrote is what one
hole computes, so no tile size is searched. The single choice per statement is
which candidate atom granularises it; everything else is measured off `tg` and
the scheduling facts projected for that stage (§5.2).

```python
def select_atoms(tg: TileGraph, target: "Target | str | None" = None, stage: str = "cta") -> TileGraph: ...

class AtomSelectionError(RuntimeError):
    """A TileGraph consistency precondition that did not hold, or a stage that exposes no fact to decide on."""
```

- constraints:
  - `tg` MUST already carry a tree from `build_schedule_tree`; `tg.tree is None`
    or an empty `tg.units` MUST raise `AtomSelectionError`.
  - The facts MUST be read off the untiled tree: one band per statement, and one
    band member per own domain dimension. A band that schedules anything other
    than its statement's own dimensions in order, a band count that does not
    match the statement count, or a `tg.parallel_dims` entry whose flag count
    does not match the statement's rank, MUST raise.
  - The scheduling facts MUST be projected from the resolved target at the
    requested stage. A stage the target does not schedule, or one whose
    `tile_capacity_bytes` is not positive, MUST raise `AtomSelectionError`
    rather than let the projection failure escape.
  - An op the target's catalogue does not cover MUST be downgraded to "no
    candidates" rather than abort the run.
  - The pick MUST be the first candidate the catalogue's own hard filter left,
    and every survivor MUST also be recorded: no cost model ranks two candidates
    at this stage.
  - An atom's shape MUST align to the **trailing** dimensions of the statement's
    domain; the leading dimensions take extent `1`. An atom shape wider than the
    domain's rank MUST raise.
  - Placement MUST be derived, never solved: a statement's dependence-free
    dimensions are the ones spread over lanes, and a statement with none is
    serial.
  - One buffer's ring depth MUST be measured, not searched for: a dependence
    carried `distance` iterations along a dimension tiled `tile` wide spans
    `ceil(distance / tile)` tiles, and the ring holds one slot more so the older
    tile stays alive. The depth MUST be the maximum over every statement holding
    that buffer, and at least `1`.
  - A buffer's recorded footprint MUST count each buffer dimension once, at the
    widest extent any access of that statement needs there, multiplied by the
    buffer's ring depth.
  - Capacity MUST be recorded against that footprint, never enforced: exceeding
    it MUST be reported as a flag on the statement and MUST NOT raise. A tile
    too wide for its store still has a schedule, only a worse one.
  - Durations MUST be recorded in integer duration units of one thousandth of a
    nanosecond, with one atom instance's estimate floored at one unit: a
    statement's nominal time is a sum over its instances, so recording whole
    nanoseconds would quantise every atom to the same value. A statement with no
    candidate atom MUST instead be charged its own domain volume at a nominal one
    nanosecond per element — its only "how much work" signal is its extent.
  - The recorded timeline MUST be a prefix sum over `tg.units` order, and every
    dependence isl reports between two distinct statements MUST run with that
    order; one that contradicts it MUST raise.
  - `select_atoms` MUST return `tg` with `tree` tiled by each statement's own
    extents, `ring` filled per buffer, and `decisions` recorded. Recorded
    decisions MUST cover, per statement: the picked atom and every candidate, the
    derived placement, the tile extents and tile count, the atom instance count,
    the nominal duration and its start and end on the timeline, the per-buffer
    footprint in bytes, and whether that footprint fits the capacity. Recorded
    decisions MUST also carry the overall status, the makespan, the capacity, and
    the ring depths.
  - The status MUST record that the decision space is a single point per
    statement: with one candidate order and one tile per statement, the recorded
    decisions are optimal by construction.

### 4.3 Scaffold emission

`emit_scaffold` renders the decided tree into what an authoring agent fills: a
holed loop nest, a human-readable swimlane, and one hole contract per statement.

```python
class Skeleton:
    """A holed, C-like loop-nest skeleton.

    Attributes:
        text: attribute; The generated loop nest, with one hole call per statement instance.
        holes: attribute; Every hole name in text, in first-appearance order.
    """

    text: str
    holes: tuple[str, ...]

class Swimlane:
    """A human-readable rendering of the decided schedule.

    Attributes:
        text: attribute; One Mermaid gantt section per statement, minimally unrolled.
    """

    text: str

class BufferAccess:
    """One buffer touched by one statement.

    Attributes:
        tensor_name: attribute; Buffer tuple name, as the TileGraph names it.
        index_map: attribute; Access map from this statement's coordinates to that buffer's elements.
        dtype: attribute; Recovered HIR element DType, or None when it could not be resolved.
    """

    tensor_name: str
    index_map: "isl.map"
    dtype: object | None

class HoleContract:
    """What one hole must compute.

    Attributes:
        name: attribute; The hole's own call name in the skeleton.
        op_ref: attribute; The HIR Call this hole stands for.
        inputs: attribute; Every buffer the statement reads, in source-call argument order.
        output: attribute; The single buffer the statement writes.
        coords: attribute; The schedule coordinates the hole is parametrised by.
    """

    name: str
    op_ref: object
    inputs: tuple[BufferAccess, ...]
    output: BufferAccess
    coords: tuple[str, ...]

def emit_scaffold(tg: TileGraph) -> tuple[Skeleton, Swimlane, list[HoleContract]]: ...

class EmitScaffoldError(RuntimeError):
    """A construct emit_scaffold does not render, or a TileGraph precondition that did not hold."""
```

- constraints:
  - Every structure MUST be immutable.
  - The skeleton MUST be isl code generation over `tg.tree`, with each naked
    statement call replaced by its hole call. `tg.tree is None` MUST raise
    `EmitScaffoldError`.
  - A hole call MUST name its inputs, its output, and its raw schedule
    coordinates, each behind its own marker, so the three groups are
    distinguishable without re-deriving them.
  - A read-modify-write self-read on the output buffer MUST appear among the
    inputs rather than be silently dropped.
  - A buffer whose decided ring depth is above `1` MUST be referenced through
    that ring, indexed by the innermost coordinate modulo the depth. Before atom
    selection has run `tg.ring` is empty, and every reference MUST then be the
    bare buffer name.
  - Exactly one `HoleContract` MUST be produced per statement, not per call site
    in the generated text, and its `coords` MUST come from the first occurrence.
  - A statement whose name has no matching `TileUnit`, or that writes more than
    one buffer, MUST raise `EmitScaffoldError`.
  - A hole whose statement call cannot be placed in the generated text MUST raise
    rather than be dropped.
  - `HoleContract` MUST be a pure function contract — inputs, output, coordinates
    — and MUST NOT carry indexing or synchronization: the skeleton already
    carries those. `op_ref` MUST be the HIR `Call`, so a later stage can fill the
    hole and diff it against the [evaluator](./evaluator.md)'s own result for
    that op subgraph.
  - The swimlane MUST be minimally unrolled — a prologue instance, a handful of
    steady-state instances, and an epilogue instance, with the elided count
    stated — never the full iteration count: a real kernel's domain runs to
    hundreds of millions of points.

## 5. Scheduling facts

The polyhedral model is target-independent; the atom catalogue and the store a
tile lives in are not. That store belongs to the **level**, not to the device: a
tile at the AMX `core` level lives in that core's L1d, one at the CUDA `cta`
level in shared memory. Both are obtained by projecting the Target
([target §11](./target.md#11-target-facts-projection)), so a scheduling stage
names the facts it needs and no stage calls into a target through a service
object whose shape it must know.

### 5.1 `AtomFact`

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

### 5.2 `TileStoreFacts` and `AtomCandidateFacts`

```python
class TileStoreFacts:
    """The store a tile of one scheduled level occupies.

    Attributes:
        stage: attribute; The topology level this capacity belongs to.
        tile_capacity_bytes: attribute; Capacity of that level's store.
    """

    stage: str
    tile_capacity_bytes: int

class AtomCandidateQuery:
    """Which operation's catalogue is being asked for, at which level.

    Attributes:
        stage: attribute; The topology level being scheduled.
        op: attribute; The HIR Call whose candidates are wanted.
    """

    stage: str
    op: "Call"

class AtomCandidateFacts:
    """The atoms one target admits for one operation.

    Attributes:
        candidates: attribute; Those atoms, in the order the target enumerated them.
    """

    candidates: tuple[AtomFact, ...]
```

- constraints:
  - `TileStoreFacts` MUST be queried by stage name and projected once per solve:
    the capacity is a property of the level and of the hardware, not of any one
    operation. `tile_capacity_bytes` MUST be the capacity of the store belonging
    to that level rather than of the whole device, and MUST be positive.
  - `AtomCandidateFacts` MUST be queried per `(stage, op)` pair and MUST carry
    only that operation's candidates. Two aggregates rather than one is
    deliberate: bundling a per-level capacity into a per-operation projection
    would give one type two meanings depending on how it was asked for.
  - A projection MUST hard-filter the target's catalogue and MUST NOT rank it:
    ordering is not a decision made here. The order it returns MUST be the
    target's own enumeration order, so a consumer that ranks candidates sees the
    same sequence every run.
  - An empty `candidates` tuple MUST be a legitimate "no candidate covers this
    operation" outcome rather than an error, and a projection MAY raise
    `NotImplementedError` for an operation kind its catalogue does not cover.
  - A stage the target does not schedule MUST be reported as that, and a
    consumer MUST surface it as its own scheduling diagnostic rather than let a
    projection failure escape.
  - The capacity is a fact to record against a footprint, not a gate: nothing
    MUST raise because a tile does not fit. A tile wider than its store still has
    a schedule, only a worse one.
