# TileFoundry Spec — Schedule

Scheduling decides how one Function is placed over one level of the parallel
hierarchy its Module declares. One public operation names the program and the
level; one registered algorithm answers with a Plan it owns entirely. What an
algorithm decides is its own vocabulary: two algorithms placing different
hardware over different levels do not decide the same things, and a shared result
schema would either describe neither or force both to pretend.

The public surface (§1–§2) is the `schedule` package. The construction stages an
algorithm composes its solve from (§4) are imported from their own modules; they
read [analysis](./analysis.md) facts, and the dependency is one-way — nothing in
the analysis layer imports the schedule layer.

## 1. The public Schedule operation

```python
def schedule(
    module: Module,
    function: Function,
    *,
    topology: str,
    options: ScheduleOptions | None = None,
) -> ScheduleResult: ...
```

- constraints:
  - The caller MUST supply the Module, the Function, and one non-empty topology
    level name. Nothing about the request MAY be inferred from layouts,
    constraints, or the shape of the program.
  - The Target MUST come from `module.resolve_target()`
    ([core-ir §1](./core-ir.md#1-module)); a call MUST NOT override it and MUST
    NOT fall back to a default Target when no Module in the owner chain declares
    one ([target §6](./target.md#6-target-ownership-and-compile-resolution)).
  - The level MUST be resolved from the Module's own effective hierarchy. A name
    the hierarchy does not declare MUST fail, and a level whose extent is known
    only at launch MUST fail: an algorithm places work across a level by counting
    it.
  - Target resolution, level resolution, and algorithm resolution MUST all
    complete before the algorithm runs, so a request that cannot be served never
    leaves a partial solve behind.
  - `options=None` MUST mean a fresh default `ScheduleOptions()` for that call.
  - The returned Plan MUST be verified (§2.3) before the result reaches the
    caller.
  - The common Schedule code MUST NOT import a concrete Target implementation.

### 1.1 Algorithm registration

An algorithm is registered for the exact `(concrete Target type, topology name)`
pair in the shared algorithm registry contract
([code-organization](./code-organization.md)).

- constraints:
  - Resolution MUST match both halves of the key exactly. A registration for a
    base class MUST NOT serve a subclass: two targets that share a base can need
    different algorithms, and inheriting one would silently run the wrong one.
  - Registering the same pair twice MUST fail, so which hardware is schedulable
    at which level is single-valued and readable off the registrations.
  - A Target MUST NOT own a scheduling method, a scheduling service object, or a
    `target.schedule()` wrapper. Registration is the only support declaration.
  - There MUST be no public stage selector, no automatic level selection, and no
    generic service-lookup facade for scheduling.
  - An algorithm MUST own its whole problem: its private program view, its Facts
    query, its constraint problem, its solve, and its Plan type. Those names MUST
    remain private to the algorithm's own package.

## 2. Public structures

### 2.1 `ScheduleOptions`

`ScheduleOptions` carries solver runtime controls, independent of which
algorithm runs.

```python
class ScheduleOptions:
    """Configure one schedule call.

    Attributes:
        timeout_seconds: attribute; Wall-clock budget for the underlying solver.
        workers: attribute; Solver worker count, where zero selects the solver default.
        random_seed: attribute; Deterministic solver tie-break seed.
        debug_dump_dir: attribute; Directory for algorithm-private artifacts, or None.
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

`ScheduleResult` is the complete public result of one call.

```python
class ScheduleResult:
    """Carry what was decided, and what it was decided against.

    Attributes:
        module: attribute; The Module the call was made on.
        function: attribute; The Function that was scheduled.
        topology: attribute; The resolved level of that Module's hierarchy.
        plan: attribute; The verified Plan the selected algorithm produced.
    """

    module: Module
    function: Function
    topology: Topology
    plan: SchedulePlan
```

- constraints:
  - The structure MUST be immutable.
  - `module` and `function` MUST be the same objects the caller supplied.
    Scheduling decides about a program; it MUST NOT return a rewritten one in
    their place. An algorithm whose decision *is* a rewritten program MUST carry
    that program in its own Plan.
  - `topology` MUST be the level as the Module declares it, not a normalized copy.

### 2.3 `SchedulePlan`

`SchedulePlan` is the extensible semantic base of every algorithm's result. It is
not a union, a shared schema, or a shared JSON envelope.

```python
class SchedulePlan:
    """One solve's decisions, owned by the algorithm that made them."""

    def verify(self, module: Module, function: Function, topology: Topology) -> None: ...

    def to_json(self) -> str: ...

    def render(self) -> str: ...
```

- constraints:
  - The base MUST expose exactly these three operations and MUST impose no
    concrete field, shared schema, or common rendering on a subtype.
  - The base MUST NOT carry a version field, a deserializer, a renderer registry,
    or a generic data-export accessor. A Plan is produced by the algorithm that
    solved for it, in the process that solved; reading one back from text would
    mean trusting a document to describe decisions nobody made in that run.
  - A subtype MUST own the whole of its JSON object and the whole of its human
    rendering.
  - `verify` MUST be a structural check of the exported plan against the request
    it answers: that it refers only to things that exist and that its own
    references agree. It MUST NOT re-solve, invoke a solver, or state anything
    about whether the schedule is good.
  - A Plan that does not hold together MUST raise `PlanVerificationError` and MUST
    NOT reach the caller.

### 2.4 `PipelineSchedulePlan`

```python
class PipelineSchedulePlan(SchedulePlan):
    """Export one closed pipeline schedule."""

    target: TargetSpecRef
    scaffold: str
    statements: tuple[ScheduledStatement, ...]
    buffers: tuple[ScheduledBuffer, ...]
    holes: tuple[KernelHole, ...]
```

- constraints:
  - `target` MUST record architecture and device IDs with their content digests.
  - Each `ScheduledStatement` MUST carry one stable ID, selected instruction,
    tile, resource assignment, and a half-open interval.
  - Each `ScheduledBuffer` MUST carry one stable ID, storage, positive ring
    depth, and typed producer and consumer statement IDs.
  - Each `KernelHole` MUST reference one stable statement ID and expose tuple
    inputs, tuple outputs, and serialized ISL relations. It MUST NOT expose an
    opaque HIR operation reference.
  - JSON and text rendering MUST be deterministic views of the same decisions.
  - Plan construction MUST finish Target Facts projection before creating the
    closed problem. The problem and solve MUST hold no Target object or callback.

### 2.5 `ScheduleReport`

`ScheduleReport` is the reusable objective summary: the part of an answer that
does not depend on what was decided. An algorithm whose Plan states an objective
carries one of these rather than restating the same fields in its own vocabulary.
Algorithm-private operation rows, use decisions, candidate costs, and
solver-native models are not report fields.

```python
class ScheduleReport:
    """Summarize the selected objective and proof state.

    Attributes:
        root: attribute; Scheduled Function name.
        target: attribute; Resolved Target name.
        topology: attribute; The topology level the result was produced at.
        status: attribute; Public solution status.
        objective_name: attribute; Primary objective name.
        unit: attribute; Unit of the primary objective and bound.
        selected: attribute; Selected primary-objective value.
        best_bound: attribute; Integer lower bound for the makespan objective.
        gap: attribute; Relative optimality gap for the makespan objective.
    """

    root: str
    target: str
    topology: str
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
  - `objective_name` is a fixed literal: every algorithm that reports an
    objective reports the same one quantity, and the field names it rather than
    selecting it.
  - The reported value MUST NOT be read as a proof that a solver optimised it. An
    algorithm MAY minimize the makespan as a solver objective, and an algorithm
    MAY instead compute it after the fact — as the nominal roofline estimate of
    the decisions it recorded (§5.1). `status`, `best_bound` and `gap` are what
    distinguish the two: an algorithm that did not prove optimality MUST NOT
    report `selected` as its own bound.
  - A feasible incumbent MUST be reportable even when optimality is unproven.
  - JSON and Markdown rendering MUST contain every public field and MUST NOT
    expose algorithm-private solve state.
  - When the report summarizes a CTA execution blueprint, its selected value
    describes the modeled plan and MUST NOT be interpreted as a guarantee that
    current lowering preserves the planned intervals, reserves physical SM
    shares, promotes values to SRAM or registers, or proves runtime performance
    or out-of-memory safety.

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
decisions, and they do not register a scheduling algorithm for a `CudaTarget`.

## 4. Kernel schedule construction

An algorithm composes its solve from three stages over one
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
from there; the compact public package surface (§1–§2) carries the public
operation and its results only.

| Stage | Signature | Error |
|---|---|---|
| tree construction | `build_schedule_tree(tg: TileGraph) -> TileGraph` | `KernelScheduleError` |
| atom selection | `select_atoms(tg: TileGraph, target, stage="cta") -> TileGraph` | `AtomSelectionError` |
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
def select_atoms(tg: TileGraph, target: "Target | str", stage: str = "cta") -> TileGraph: ...

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
  - The target MUST be named by the caller. There MUST be no default-Target
    fallback: which atoms are candidates and how wide a tile may be are
    properties of one machine, so a call that names none asks for decisions
    nobody specified.
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
([target §11](./target.md#11-target-facts-projection)), so an algorithm names
the facts it needs and never calls into a target through an object whose shape it
must know.

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
