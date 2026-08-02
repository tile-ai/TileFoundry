# TileFoundry Spec — Schedule

Scheduling decides how one Function is placed over one level of the parallel
hierarchy its Module declares. One public operation names the program and the
level; one registered algorithm answers with a Plan it owns entirely. What an
algorithm decides is its own vocabulary: two algorithms placing different
hardware over different levels do not decide the same things, and a shared result
schema would either describe neither or force both to pretend.

The public surface ([§1](#1-the-public-schedule-operation)–[§2](#2-public-structures)) is the `schedule` package. The construction stages an
algorithm composes its solve from ([§4](#4-kernel-schedule-construction)) are imported from their own modules; they
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
    dims: Mapping[str, int] | None = None,
) -> ScheduleResult: ...
```

- constraints:
  - The caller MUST supply the Module, the Function, and one non-empty topology
    level name. Nothing about the request MAY be inferred from layouts,
    constraints, or the shape of the program.
  - The Function MUST be one the Module owns: one it declares, or a
    specialization variant of one it declares
    ([core-ir §1](./core-ir.md#1-module)). A Function derived by specialising one
    of these MUST be refused, so that ownership is settled before anything is
    rebuilt.
  - `dims` states one extent per dimension the Function declares as a range. A
    solver places work across a level by counting it and holds a tile against a
    capacity in bytes, so a Function stating a range MUST be solved at a chosen
    size rather than as authored.
  - `dims=None` MUST behave as a call that states no size: the Function is
    solved as authored.
  - When `dims` is stated it MUST be non-empty; every key MUST name a dimension
    the Function declares as a range; every value MUST be an integer inside that
    dimension's declared bounds; every dimension the Function selects a variant
    on MUST be given a value; and no dimension MAY remain a range after
    substitution. Each of these MUST fail with a Schedule domain error. A stated
    `dims` MUST NOT be silently ignored, including when the Function declares no
    range at all.
  - Variant resolution and substitution MUST happen after the ownership check
    and before the algorithm runs. Exactly one variant MUST cover the stated
    size; none and more than one MUST both fail.
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
  - The returned Plan MUST be verified ([§2.3](#23-scheduleplan)) before the result reaches the
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
        stop_at_first_solution: attribute; Accept the first result satisfying the
            constraints instead of searching the budget for the best one.
        debug_dump_dir: attribute; Directory for algorithm-private artifacts, or None.
    """

    timeout_seconds: float = 60.0
    workers: int = 0
    random_seed: int = 0
    stop_at_first_solution: bool = False
    debug_dump_dir: Path | None = None
```

- constraints:
  - The structure MUST be immutable.
  - `debug_dump_dir` MUST affect artifact emission only and MUST NOT change the
    selected result.
  - `stop_at_first_solution` MUST change which satisfying result is selected and
    MUST NOT change what counts as one: a result accepted under it MUST satisfy
    every constraint a result accepted without it satisfies, so a plan obtained
    this way is verifiable on the same terms.
  - `stop_at_first_solution` MUST NOT lift `timeout_seconds`. An algorithm that has
    found no satisfying result yet stays bounded by it, so the option cannot turn a
    bounded search into an unbounded one.

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
  - `module` MUST be the same object the caller supplied. Scheduling decides
    about a program; it MUST NOT return a rewritten Module in its place. An
    algorithm whose decision *is* a rewritten program MUST carry that program in
    its own Plan.
  - When the call states no `dims`, `function` MUST be the same object the
    caller supplied.
  - When the call states `dims`, `function` MUST be the concrete Function the
    plan was solved for, derived from the Function the caller supplied. It MUST
    record that Function as the one it was specialised from, and the plan MUST
    verify against it. A caller returned its own symbolic input would hold a
    plan it cannot check.
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

Which hardware documents a decision was made against is the same question
whatever was decided, so every Plan states it the same way.

```python
class TargetSpecRef:
    """Stable identity of the installed target facts one plan relies on.

    Attributes:
        architecture_id: attribute; Installed architecture document ID, or the architecture's own name.
        architecture_digest: attribute; Content digest of that document, empty when none was installed.
        device_id: attribute; Installed device document ID, or the device's own name.
        device_digest: attribute; Content digest of that document, empty when none was installed.
    """

    architecture_id: str
    architecture_digest: str
    device_id: str
    device_digest: str
```

- constraints:
  - The structure MUST be immutable and MUST be shared by every Plan that names
    the hardware it relied on, so two plans cannot describe the same target
    differently.
  - A Target constructed directly rather than installed from documents MUST state
    an empty digest rather than a fabricated one.

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
    tile, resource assignment, a half-open interval, the bytes it holds, and
    whether the level's tile store holds them.
  - `ScheduledStatement.footprint_bytes` MUST count each buffer the statement
    touches at the ring depth that buffer was given, so it states what the
    statement occupies once the pipeline is deep enough to run.
  - `ScheduledStatement.fits_capacity` MUST record `footprint_bytes` against the
    tile capacity the level states. A statement that does not fit MUST still
    appear in the plan: the plan states what the program costs on the target,
    and a solve MUST NOT drop or shrink a statement to make a plan fit.
  - Each `ScheduledBuffer` MUST carry one stable ID, storage, positive ring
    depth, and typed producer and consumer statement IDs.
  - `ScheduledBuffer.ring_depth` MUST be derived from the dependence distance
    the buffer carries under the extents of each statement holding it, so that
    a buffer whose value survives into a later tile is given enough slots to
    keep the earlier tile alive. A buffer that carries no dependence MUST be
    given one slot.
  - Each `KernelHole` MUST reference one stable statement ID and expose tuple
    inputs, tuple outputs, and serialized ISL relations. It MUST NOT expose an
    opaque HIR operation reference.
  - JSON and text rendering MUST be deterministic views of the same decisions.
  - Plan construction MUST finish Target Facts projection before creating the
    closed problem. The problem and solve MUST hold no Target object or callback.

### 2.5 `PartitionSchedulePlan`

A partition plan states where each value was placed, which operations run over
which placements, and what the solve proved. It names both by identities derived
from the authored program rather than by the indexes its own problem allocated, so
an agent reading the plan can find what it refers to in the program it wrote.

```python
class PositionInterval:
    """The half-open range of parallel positions something occupies.

    Attributes:
        start: attribute; First position occupied.
        end: attribute; One past the last position occupied.
    """

    start: int
    end: int

class TimeInterval:
    """One operation's half-open execution interval.

    Attributes:
        start_ns: attribute; Start of the interval, in ns.
        end_ns: attribute; One past the end of the interval, in ns.
    """

    start_ns: int
    end_ns: int

class PlacedValue:
    """One tensor value, the type it was placed in, and who touches it.

    Attributes:
        id: attribute; Stable identity of this placement, derived from the program.
        type: attribute; The ordinary IR Type selected for it, carrying its layout and storage.
        producer_id: attribute; The operation that produces it, or None when the plan produces none.
        consumer_ids: attribute; Every operation that reads it.
        positions: attribute; The positions this placement occupies.
    """

    id: str
    type: Type
    producer_id: str | None
    consumer_ids: tuple[str, ...]
    positions: PositionInterval

class PartitionedOperation:
    """One operation that runs, where it runs, and when.

    Attributes:
        id: attribute; Stable identity of this operation, derived from the program.
        operation: attribute; The operation's own kind.
        synthesized: attribute; True when the algorithm introduced it rather than the author.
        input_ids: attribute; The placements it reads.
        output_ids: attribute; The placements it produces.
        positions: attribute; The positions it occupies, or None when it occupies none.
        interval: attribute; Its execution interval, or None when the model gives it none.
    """

    id: str
    operation: str
    synthesized: bool
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    positions: PositionInterval | None
    interval: TimeInterval | None

class PartitionProof:
    """What the solve proved about its own objective.

    Attributes:
        status: attribute; Whether optimality was proven or only feasibility reached.
        objective_ns: attribute; The selected makespan, in ns.
        best_bound_ns: attribute; The bound the solve established, in ns.
        proven_optimal: attribute; True when the two met.
    """

    status: Literal["OPTIMAL", "FEASIBLE_NOT_PROVEN"]
    objective_ns: int
    best_bound_ns: int
    proven_optimal: bool

class PartitionSchedulePlan(SchedulePlan):
    """The placement one partition solve committed to, and its proof."""

    topology: str
    extent: int
    target: TargetSpecRef
    values: tuple[PlacedValue, ...]
    operations: tuple[PartitionedOperation, ...]
    root_results: tuple[str, ...]
    proof: PartitionProof
```

- constraints:
  - Every structure MUST be immutable, and the plan MUST state the level it
    decided about, how many positions of it there were, and the identity of the
    installed documents it decided against.
  - An identity MUST be derived from the authored program and MUST be unique
    within the plan. One value MAY hold more than one placement at once -- that is
    what a Reshard connects -- so a placement, not a value, is what an identity
    names.
  - `PlacedValue.type` MUST be the ordinary IR Type that was selected, so the
    layout and storage a value was placed in are read off the type system rather
    than restated in a parallel vocabulary.
  - An operation the algorithm synthesized MUST appear among `operations`, marked
    as synthesized, with the placements it moves between. There MUST be no
    separate route, report, or debug channel through which a caller would learn
    that data moves.
  - An operation charged as traffic rather than as occupancy MUST state no
    position range. Giving it one would claim it excludes other work from those
    positions, which the decision does not.
  - `proof` MUST state the objective, the bound the solve established, and whether
    the two met. It is a result fact and MUST NOT become a generic report facade.
  - An edge MUST be named the same way from both of its ends: a placement's
    producer MUST list that placement among its outputs, its consumers MUST list
    it among their inputs, and every operation's outputs and inputs MUST be
    reflected back by the placements they name. An edge only one end claims is a
    claim about a decision nobody made, and a walk that followed it would report a
    program flow the operations do not implement.
  - `verify` MUST reject: a level or extent other than the one the plan decided;
    two placements or two operations sharing an identity; a reference to a
    placement or operation the plan does not carry; an edge whose two ends
    disagree; a placement in a type that is not addressable global memory; a synthesized move between two placements that
    are different logical tensors or that are identical; a position range outside
    the level; an interval that ends before it starts; two operations holding the
    same position at the same time; a root result the plan does not carry or
    cannot reach by following producers; and a bound above the stated objective.
    It MUST do all of that without rebuilding candidates and without invoking a
    solver.
  - JSON and text rendering MUST be deterministic and MUST state the same
    placements, operations, intervals, and proof.
  - A solver variable or solver-native value MUST NOT appear in the exported plan.
  - The plan MUST NOT carry a rewritten program, and producing it MUST NOT rewrite
    one: a partition decides where work and its tensors go, and applying that
    decision to HIR is a separate operation the caller asks for.

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

An algorithm composes its solve from these stages over one
`TileGraph` ([analysis §1.2](./analysis.md#12-tilegraph)). Each takes the graph
and returns it enriched; none of them re-derives a fact the analysis layer
already states.

```text
extract(root)  ──▶  build_schedule_tree  ──▶  emit_scaffold
  (analysis)          tg.tree                 Skeleton / Swimlane / HoleContract
```

Each stage lives in its own module of the `schedule` package and is imported
from there; the compact public package surface ([§1](#1-the-public-schedule-operation)–[§2](#2-public-structures)) carries the public
operation and its results only.

| Stage | Signature | Error |
|---|---|---|
| tree construction | `build_schedule_tree(tg: TileGraph) -> TileGraph` | `KernelScheduleError` |
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

### 4.2 Scaffold emission

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

The polyhedral model is target-independent; the atom catalogue, the rates work is
charged at, and the store a tile lives in are not. Each algorithm family declares
the facts it needs as its own aggregate, so what one family asks for cannot become
a shared vocabulary another family has to satisfy. All of them are obtained by
projecting the Target ([target §11](./target.md#11-target-facts-projection)), so
an algorithm names the facts it needs and never calls into a target through an
object whose shape it must know.

The level an algorithm is asked about and the level whose store bounds it need not
be the same one, and a projection MUST NOT collapse them. An AMX core both runs
the work and owns the L1d its tile lives in. A CUDA pipeline is asked about
`thread`, because what it decides is how the threads of one CTA overlap their
work, but the store they cooperate in is shared memory, which is a CTA-scoped
resource: reporting that capacity as a per-thread number would claim a limit no
hardware publishes.

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

### 5.2 `PartitionFacts`

```python
class PartitionFactsQuery:
    """The one topology level a projection is asked to describe.

    Attributes:
        topology: attribute; The level being divided.
    """

    topology: str

class PartitionFacts:
    """All concrete hardware information required to close one partition.

    Attributes:
        topology: attribute; The level being divided.
        spec: attribute; Identity of the installed documents these numbers came from.
        parallel_units: attribute; How many positions of that level the plan may occupy.
        memory_bandwidth_bytes_per_second: attribute; The rate traffic is charged at.
        memory_capacity_bytes: attribute; The capacity resident bytes are charged against.
        peak_flops_per_second: attribute; Dense peak rate per compute DType.
    """

    topology: str
    spec: TargetSpecRef
    parallel_units: int
    memory_bandwidth_bytes_per_second: int
    memory_capacity_bytes: int
    peak_flops_per_second: tuple[tuple[DType, int], ...]

    def peak_flops(self, dtype: DType) -> int: ...
```

- constraints:
  - The structure MUST be immutable and MUST contain every numerical fact the
    closed problem and its solve consume. After it is projected, neither the
    problem nor the solve MAY hold a Target, follow one through the program, or
    invoke a projection again.
  - A capacity MUST be stated once here rather than copied onto each candidate: a
    candidate states the demand it makes, and what that demand is compared against
    belongs to the hardware.
  - A DType the hardware publishes no rate for MUST fail rather than resolve to
    zero or to a neighbouring rate. Charging work at a rate no document supports
    would put an unsupported number in the plan.
  - A level the target does not divide MUST be reported as that, and the algorithm
    MUST surface it as its own scheduling diagnostic rather than let a projection
    failure escape.
  - Compiler policy MUST stay in `ScheduleOptions`, not in these facts: what the
    hardware is does not depend on how aggressively the compiler was asked to
    schedule it.
