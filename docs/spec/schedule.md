# TileFoundry Spec — Schedule

A Schedule is a Target-owned service that materializes one explicitly selected
stage of typed HIR. It reads a `Module` and one root `Function`, then returns a
materialized `Module` plus a stable objective summary. Scheduling is separate
from pass sequencing: callers select the service by stage name, and the service
owns the stage-specific algorithm.

## 1. Direct service invocation

The root Function's Target is the sole owner of service selection. A caller
selects an exact stage and invokes the returned service directly:

```python
# example
options = ScheduleOptions()
service = root.target.service(Schedule, stage)
result = service.solve(module, root, options)
```

- constraints:
  - The caller MUST select a non-empty stage string explicitly.
  - Target service lookup MUST match the requested stage exactly and MUST NOT
    infer a stage from layouts, topology, or constraints.
  - The service MUST come from `root.target`; a call MUST NOT override the
    root Function's Target.
  - `root` MUST be one of `module.functions`.

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
    """Carry a materialized module and its summary report.

    Attributes:
        module: attribute; Materialized HIR module produced by the service.
        report: attribute; Stable cross-stage objective summary.
    """

    module: Module
    report: ScheduleReport
```

- constraints:
  - The structure MUST be immutable.
  - `module` MUST contain the materialized output selected by the same solve
    represented by `report`.

### 2.3 `ScheduleReport`

`ScheduleReport` is the stable cross-stage summary. Stage-private operation
rows, use decisions, candidate costs, and solver-native models are not report
fields.

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
        baseline: attribute; Deterministic legal baseline value.
        selected: attribute; Selected primary-objective value.
        solver_phase: attribute; First unproven objective phase, or final phase.
        proven_objectives: attribute; Objective phases proven optimal in solve order.
        best_bound: attribute; Bound for solver_phase, or None when unavailable.
        gap: attribute; Relative optimality gap for solver_phase, or None.
    """

    root: str
    target: str
    stage: str
    status: Literal["OPTIMAL", "FEASIBLE_NOT_PROVEN"]
    objective_name: Literal["makespan"]
    unit: Literal["ns"]
    baseline: int
    selected: int
    solver_phase: Literal["makespan", "reshard_bytes", "resource_area"]
    proven_objectives: tuple[str, ...]
    best_bound: int | None
    gap: float | None

    def to_json(self) -> str: ...

    def to_markdown(self) -> str: ...
```

- constraints:
  - The structure MUST be immutable.
  - `baseline` and `selected` MUST always describe `objective_name`, regardless
    of the value of `solver_phase`.
  - `best_bound` and `gap` MUST describe `solver_phase`.
  - JSON and Markdown rendering MUST contain every public field and MUST NOT
    expose stage-private solve state.

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
        options: ScheduleOptions,
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

## 4. CTA input preflight

CTA input preflight is a private validation boundary over a root HIR
`Function` or a `Module` entry. The root MUST carry an explicit
`CudaTarget` and exactly one `Topology("cta", n)` with a static integer
`1 <= n <= device.sm_count`. A dynamic or missing CTA extent is invalid for
this boundary.

Other root topology declarations are retained on the HIR and are ignored by
CTA preflight. They do not participate in CTA validation. A helper function
with no target and no program topologies inherits the caller's effective
target without mutating its source value. An explicit helper target MUST match
that effective target, and a helper with program topologies is rejected as a
kernel boundary. Recursive helper calls, TIR kernel calls, and dispatch
prototypes are invalid HIR CTA inputs.

Preflight recursively visits expression arguments and nested
`GridRegionExpr` values. Region `start` MUST be non-negative, while
`extent` and `step` MUST be positive static integers. Dynamic bounds fail
with the owning function and root context. Successful preflight returns only
immutable root CTA facts and the reachable HIR function set; it does not
register or invoke a scheduling service.
