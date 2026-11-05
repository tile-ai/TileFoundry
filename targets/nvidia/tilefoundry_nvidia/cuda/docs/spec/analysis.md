# TileFoundry CUDA Spec — Analysis

## 1. `sync`

`CudaTarget.get_analyzer("sync")` adds this family to the core families. See
[analysis §1.2](../../../../../../docs/spec/analysis.md#12-analysis-families).
A Target that does not add it refuses the selector. See
[analysis §2.2](../../../../../../docs/spec/analysis.md#22-target-selected-analyzers).

| Selector | Requires | Owns | Attaches to | Rests on | Text summary adds | Annotates equations |
|---|---|---|---|---|---|---|
| `sync` | `memory` | `SyncPlanMetadata`, `SyncMetadata` | `SyncPlanMetadata` on the Function; `SyncMetadata` on each Call, `MeshRegion` or `LoopRegion` whose statements need steps | the memory family's records and the completion registry ([§2](#2-completion-registry)) | - | - |

The `sync` family decides where lowering emits commits, waits, proxy fences,
barriers and fences. These steps make each access to shared memory or to a
register follow the earlier accesses it depends on. If the family cannot do
this, it refuses the Function. `SyncPlanMetadata` marks each analysed Function
as planned. CUDA lowering emits the steps of a node where it lowers that node.
Lowering analyses nothing itself. See
[passes §7.1](./passes.md#71-converthirtotir).

```python
class StepKind(Enum):          # under one tag, steps run in the members' order
    COMMIT = auto()
    WAIT = auto()
    PROXY_FENCE = auto()
    BARRIER = auto()
    FENCE = auto()

class SyncStep:
    kind: StepKind
    mesh: Mesh | None = None   # the thread mesh a barrier runs over
    group: Group | None = None # the completion group a commit, a wait or a fence is for

class SyncMetadata(IRMetadata):
    steps: tuple[tuple[object, tuple[SyncStep, ...]], ...] = ()   # (tag, steps)
    def at(self, tag=None) -> tuple[SyncStep, ...]: ...

class SyncPlanMetadata(IRMetadata):
    shared: bool = False       # whether any of its actions touches shared memory
```

- constraints:
  - **Steps.** Each analysis replaces the `SyncMetadata` of every node of the
    Function. Under one tag, steps run in the order of `StepKind`, then of
    `Group`. A tag holds each step once. Steps for different groups stay apart.
    The tag tells where its steps run:

    | Tag | Node | Steps run |
    |---|---|---|
    | None | `Call`, `MeshRegion`, `LoopRegion` | before the node's own statements: the instruction, fill or retained call of a Call, or the entry into a region |
    | `"issue"` | `Call` | before each issue of its instruction, in the region that issues it |
    | `"after"` | `Call` | right after the issues of its instruction, in the region that issues them |
    | `"base"`, `"update"` | `InsertSlice` | before the base or the update copy that lowering adds |
    | `("entry", slot)`, `("backedge", slot)` | `LoopRegion` | before the entry or the backedge copy that lowering adds for the carried `slot` |
  - **Actions.** The actions of a node are what lowering emits for it. The
    `visit_*` methods of the family MUST correspond one to one with the visits
    that `CudaLowering` has or inherits. A change to one MUST change the other.
    The family dispatches a Call as `CudaLowering` does, by the first row that
    matches:

    | Call | Family visit | Actions |
    |---|---|---|
    | one that `lowers_to_dim` accepts | - | none |
    | a retained call | - | the retained call, which reads its arguments and writes its results |
    | `TupleGetItem`, `Slice`, `Reshape`, `Bitcast`, `Empty`, `IndexSelect`, or a `Reshard` whose result has an accepted VIEW alias | `visit_<HIR op>` | none |
    | any other `Reshard` | `visit_Reshard` | the actions of `visit_T` |
    | `Zeros`, `FullLike` | `visit_<HIR op>` | one `Fill` |
    | `InsertSlice` | `visit_InsertSlice` | the copies that `copies_insert_slice_base` and `copies_insert_slice_update` name |
    | any other Call | `visit_T` | the instruction that `selected_op` names |
  - **Sources.** The family reads each fact from the source below. Lowering
    calls the same functions on the same analysed program.

    | Fact | Source |
    |---|---|
    | the instruction a Call issues, its operand types, the mesh that owns it | `passes/transforms/convert_hir_to_tir.py` |
    | the copies lowering adds, the parameters a Call's arguments supply, the mesh an instruction is issued on | `cuda/analysis/emission.py` |
    | the completion of each effect | the completion registry, see [§2](#2-completion-registry) |
  - **Register storage.** Every register read and write of an action touches
    the storage that lowering gives its value. `register_storage` follows the
    result of a Call whose visit issues its selected instruction in place to
    its argument. It follows any other value along its first accepted alias.
  - **Scopes.** The family uses the scopes of the Function's `IterationScope`.
    See
    [analysis §2.1](../../../../../../docs/spec/analysis.md#21-shared-iterationscope-and-access).
    It builds no scope tree of its own. The scope of an expression is its
    recorded location. The family orders the children of a scope by their
    events on the liveness timeline. No position leaves the family.
  - **Shared memory bytes.** An access owns the bytes of its value's allocation
    leaf, as the memory family places it. They span the offset and size of each
    stage, for all stages. For a Tuple result, they span the leaf's own offset
    and size. A leaf owns unknown bytes when it has no placed offset or when
    the Function returns its allocation. Unknown bytes overlap all bytes. The
    plan orders whole allocations, not stages or windows.
  - **Order.** A later access conflicts with an earlier one and follows it
    after steps, as this table states:

    | | Shared memory | Register storage |
    |---|---|---|
    | Conflict | one of the two writes and their bytes overlap | the earlier access is in flight |
    | Conflict in one group | as above | only when one of the two writes and they share no non-None `chain` |
    | No conflict | the earlier access is complete, and both are operands of instructions without an execution mesh, with one `ShardLayout` that splits along each mesh axis whose threads share shared memory | - |
    | Steps | a commit and a wait for an earlier group effect, a proxy fence for an async access after a generic one, then a barrier over a scope whose mesh chain is a prefix of the later access's | a commit and a wait for the group of the earlier access, and no barrier |
  - **Group fence.** A `fence` effect needs the fence of its group before the
    action in two cases. First, its registers have no access yet. Second, an
    access to its registers since the last fence is not an effect of that group
    and `chain`. A `fence` effect without a `chain` needs the fence before
    every issue, under the tag `"issue"`. The family keeps the fence state per
    register storage. This state is exact only while every access to one
    register storage runs on the threads that hold its value. See
    [hir Reshard](../../../../../../docs/spec/hir.md#reshard).
  - **Units.** An action with a `unit` effect MUST run in a scope whose mesh is
    whole, aligned units that hold the issuing threads. Every loop around that
    scope MUST have bounds that the threads of each unit see alike. Such a
    bound is one that a barrier accepts, or a thread coordinate that reads one
    value across each unit. A step of that group needs the same from its scope.
    If that scope fails, the action completes in place. It also completes in
    place when such an effect can be in flight at the return. Then its group's
    commit and wait run under the tag `"after"`. The family plans the Function
    again. The accesses of the action stay recorded as complete.
  - **Placement.** An action runs on the mesh that lowering issues it on, or on
    its lexical scope's mesh when that is None. That mesh MUST lie within its
    lexical scope's mesh, as `within_scope` decides. The family places steps in
    the innermost scope that holds both accesses, before the child that holds
    the later one. A barrier runs over the mesh of that scope. That mesh MUST
    be a thread mesh that `classify` supports. Every loop around the position
    MUST have bounds that every thread sees alike. A barrier accepts constants,
    kernel parameters and their captures, and induction variables of such
    loops. It also accepts coordinates of mesh levels that its mesh does not
    span.
  - **Loops.** The family analyses loops to a fixed point. The head state joins
    every body end until it repeats.
  - **Callees.** The family plans callees before their callers. Each Function
    orders its own accesses. The register effects of a retained call complete
    on return. A retained call also reads and writes unknown shared memory
    bytes, unless the callee has a plan in which no action touches shared
    memory. These effects complete as a `cp.async` group. An access that
    conflicts with them waits for every `cp.async` group in flight.
  - **Refusals.** The family raises `AnalysisError` when an action or a barrier
    breaks a MUST of **Units** or **Placement**. It also raises it for a loop
    whose state does not settle within its limit.

## 2. Completion registry

One registry over classes states when each memory effect of an issued
instruction is complete for later accesses. The `sync` family asks it for
every action lowering emits.

```python
class Issued:
    op: Op
    params: tuple[ParamDef, ...]    # in issue order
    types: tuple                    # one per parameter, None where none is known

class Mechanism(Enum):
    ON_RETURN = auto()
    GROUP = auto()

class Proxy(Enum):
    GENERIC = auto()
    ASYNC = auto()

class Group(Enum):
    CP_ASYNC = auto()
    WGMMA = auto()

class Effect:
    param: str
    write: bool
    mechanism: Mechanism
    group: Group | None = None
    proxy: Proxy = Proxy.GENERIC
    unit: Mesh | None = None
    fence: bool = False
    chain: tuple[int, DType] | None = None   # a WGMMA's (n, dtype)

completion_registry: DispatchRegistry     # keyed by class
def effects(issued: Issued, **fields) -> tuple[Effect, ...]: ...
def completion_of(issued: Issued, *, key: type | None = None) -> tuple[Effect, ...]: ...
```

- constraints:
  - A handler answers one `Effect` for each read and each write of the issued
    parameters. `effects` builds them in parameter order, with the reads of a
    parameter first. A handler sees the parameter types.
  - `Mechanism.ON_RETURN` means the effect is complete for the issuing threads
    once the statement returns. `Mechanism.GROUP` means it completes when those
    threads wait on the `group` that holds it.
  - `proxy` is the path the effect takes to shared memory. A `unit` is the
    mesh of one set of threads that issue the instruction together. These
    threads also run the steps of its group together. A `fence` effect on
    registers needs the fence of its group after any other access to those
    registers. An earlier effect with the same non-None `chain` is the
    exception.
  - Lookup uses the exact class of `issued.op`, or `key` when it is given. A
    class with no registration answers `Mechanism.ON_RETURN` for every effect.
    A class does not inherit the answer of its base class. The emitter of such
    a class MUST complete its work before the statement returns.
  - The registry belongs to CUDA. Only the `sync` family reads it.
    `cuda/analysis/completion.py` registers every handler when it is imported.
    `cuda/analysis/sync.py` imports it. No Op file registers a completion. The
    handlers are:

    | Class | Its effects complete |
    |---|---|
    | `CopyAsync` | in `Group.CP_ASYNC` |
    | the WGMMA atom | as [tir MMA atoms](./tir.md#declarative-mma-atoms-and-ttiled_mma) states |
    | `TiledMma` | as the handler of its atom states, looked up with `key` |
