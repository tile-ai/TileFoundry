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
this, it refuses the Function. It attaches `SyncMetadata` to each HIR node
whose statements need steps. It attaches `SyncPlanMetadata` to each analysed
Function to mark it as planned. CUDA lowering emits the steps of a node where
it lowers that node. Lowering analyses nothing itself. See
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
  - **Steps.** Steps belong to the HIR node that lowering emits statements
    for. This node is a `Call`, a `MeshRegion` or a `LoopRegion`. The tag of a
    step tells where the step runs:

    | Tag | Node | Steps run |
    |---|---|---|
    | None | `Call`, `MeshRegion`, `LoopRegion` | before the node's own statements: the instruction, fill or retained call of a Call, or the entry into a region |
    | `"issue"` | `Call` | before each issue of its instruction, in the region that issues it |
    | `"after"` | `Call` | right after the issues of its instruction, in the region that issues them |
    | `"base"` | `InsertSlice` | before the base copy that lowering adds |
    | `"update"` | `InsertSlice` | before the update copy that lowering adds |
    | `("entry", slot)` | `LoopRegion` | before the entry copy that lowering adds for the carried `slot` |
    | `("backedge", slot)` | `LoopRegion` | before the backedge copy that lowering adds for the carried `slot` |

    Under one tag, steps run in the order of `StepKind`. Steps of one kind run
    in the order of `Group`. A tag holds each step once. The family does not
    merge steps for different groups. Each analysis replaces the
    `SyncMetadata` of every node of the Function.
  - **Actions.** The actions of a node are what lowering emits for it. An
    action is the selected instruction of a Call, a copy or fill that lowering
    adds, or a retained call. A retained call reads its arguments and writes
    its results. The family dispatches a Call as `CudaLowering` does. The
    first row that matches the Call applies:

    | Call | Family visit | Actions |
    |---|---|---|
    | one that `lowers_to_dim` accepts | - | none |
    | a retained call | - | the retained call |
    | `TupleGetItem`, `Slice`, `Reshape`, `Bitcast`, `Empty`, `IndexSelect` | `visit_<HIR op>` | none |
    | a `Reshard` whose result has an accepted VIEW alias | `visit_Reshard` | none |
    | any other `Reshard` | `visit_Reshard` | the actions of `visit_T` |
    | `Zeros`, `FullLike` | `visit_<HIR op>` | one `Fill` |
    | `InsertSlice` | `visit_InsertSlice` | the copies that `copies_insert_slice_base` and `copies_insert_slice_update` name |
    | any other Call | `visit_T` | the instruction that `selected_op` names |

    The `visit_*` methods of the family MUST correspond one to one with the
    visits that `CudaLowering` has or inherits. A change to one MUST change
    the other.
  - **Sources.** The family reads each fact from one source:

    | Fact | Source |
    |---|---|
    | the instruction a Call issues, its operand types, the mesh that owns it | `passes/transforms/convert_hir_to_tir.py` |
    | the copies lowering adds, the parameters a Call's arguments supply, the mesh an instruction is issued on | `cuda/analysis/emission.py` |
    | the completion of each effect | the completion registry, see [§2](#2-completion-registry) |

    Lowering calls the same functions on the same analysed program. The
    register effects of a retained call complete on return. This is because
    the callee's own plan completes its WGMMA groups before it returns. The
    shared memory effects of a retained call follow the **Callees** rule.
  - **Register storage.** A register access touches the storage that lowering
    gives its value. `register_storage` follows the result of an instruction
    written in place to its argument. In this rule, an instruction is a Call
    whose visit issues its selected instruction. `register_storage` follows
    any other value along its first accepted alias. The register reads and
    writes of every action take part, not only the asynchronous ones.
  - **Scopes.** The scopes are those of the `IterationScope` of the Function.
    See
    [analysis §2.1](../../../../../../docs/spec/analysis.md#21-shared-iterationscope-and-access).
    The scope of an expression is its recorded location. The family builds no
    scope tree of its own. It orders the children of each scope by their
    events on the liveness timeline of the Function. Lowering emits them in
    this order. No position leaves the family.
  - **Shared memory bytes.** An access owns the shared memory bytes of its
    value's allocation leaf, as the memory family places it. These bytes start
    at the offset of each stage and span the stage size, for all stages
    together. For a Tuple result, they are the leaf's own offset and size. A
    leaf owns unknown bytes when it has no placed offset. A leaf also owns
    unknown bytes when the Function returns its allocation. This is because
    that allocation can live in the output that its caller passes. Unknown
    bytes overlap all bytes. The plan orders whole allocations. It does not
    tell which stage or window an access selects.
  - **Shared memory order.** Two shared memory accesses conflict when one of
    them writes and their bytes overlap. A scope covers an access when the
    scope's mesh chain is a prefix of the access's. A later access follows an
    earlier one once a barrier has run over a scope that covers the later
    access. A commit and a wait for an earlier group effect run in that scope
    before the barrier. A later access through the async proxy after an
    earlier generic one also needs a proxy fence. The proxy fence runs after
    the wait and before the barrier.
  - **Thread-owned bytes.** Each thread issues an instruction without an
    execution mesh on its own shard. A shared memory `ShardLayout` can split
    along each mesh axis whose threads share shared memory. With that layout,
    one thread touches each byte of an operand of such an instruction. Two
    accesses with the same such layout touch each byte from the same thread.
    When the earlier access is complete, the later one follows it in the
    program order of that thread. It needs no step.
  - **Register order.** An earlier access to a register storage can be in
    flight in a group. While it is in flight, each later access to that storage
    conflicts with it. A read conflicts as much as a write. This is because PTX
    leaves access to the registers of a wgmma undefined before its wait. A
    later effect of the same group conflicts only when one of the two writes
    and their `chain`s differ. The later access follows after a commit and a
    wait for the group of the earlier one. It needs no barrier. The family
    keeps the fence state per register storage. This is exact only while every
    access to one register storage runs on the threads that hold its value.
    Operand reachability, `Slice` and `Reshard` keep this true. A `Reshard` of
    register storage is a view only when every element keeps its thread. See
    [hir Reshard](../../../../../../docs/spec/hir.md#reshard).
  - **Group fence.** A `fence` effect needs the fence of its group before the
    action in two cases. First, its registers have no access yet. Second, an
    access to its registers since the last fence is not an effect of that
    group and `chain`. A `fence` effect without a `chain` needs the fence
    before every issue, under the tag `"issue"`.
  - **Units.** An action with a `unit` effect MUST run in a scope whose mesh
    is whole, aligned units. These units MUST hold the issuing threads. Every
    loop around that scope MUST have bounds that the threads of each unit see
    alike. A bound that a barrier accepts is such a bound. A thread
    coordinate that reads one value across each unit is also such a bound.
    PTX `.aligned` asks this of the threads of one warpgroup. So two units can
    loop a different number of times. A step of that group needs the same
    from the scope that the family places it in. If that scope does not meet
    these needs, the action completes in place. The action also completes in
    place if such an effect can still be in flight when the Function returns.
    To complete in place, the commit and wait of its group run under the tag
    `"after"`. The family then plans the Function again. The accesses of the
    action stay recorded as complete. So later accesses from other threads
    still take their proxy fences and barriers.
  - **Placement.** An action runs on the mesh that lowering issues it on.
    When that mesh is None, the action runs on the mesh of its lexical scope.
    That mesh MUST lie within the mesh of its lexical scope. `within_scope`
    decides this. The family places steps in the innermost scope that holds
    both accesses. It places them before the child that holds the later
    access. So the issuing threads of both accesses run them. A barrier runs
    over the mesh of that scope. That mesh MUST be a thread mesh that
    `classify` supports. Every loop around the position MUST have bounds that
    every thread sees alike. A barrier accepts these bounds:
    - constants
    - kernel parameters and their captures
    - induction variables of such loops
    - coordinates of mesh levels that the mesh of the barrier does not span
  - **Loops.** The family analyses loops to a fixed point. The head state
    joins every body end until it repeats.
  - **Callees.** The family plans callees before their callers. Each Function
    orders its own accesses. A retained call also reads and writes unknown
    shared memory bytes. The exception is a callee whose plan says that none
    of its actions touches shared memory. A callee without a plan is not an
    exception. These effects complete as a `cp.async` group. This is because
    the callee can leave groups in flight. An access that conflicts with the
    call waits for all of these groups.
  - **Refusals.** The family raises `AnalysisError` for:
    - an action whose mesh is not provably within its scope
    - an action with a `unit` effect whose own scope is not whole units
    - an action with a `unit` effect in a loop whose bounds can differ between
      the threads of one unit
    - a barrier that no supported uniform scope can hold
    - a loop whose state does not settle within its limit

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
    parameter first. A handler sees the parameter types. So an effect can
    depend on where an operand lives.
  - `Mechanism.ON_RETURN` means the effect is complete for the issuing threads
    once the statement returns. `Mechanism.GROUP` means it completes when those
    threads wait on the `group` that holds it.
  - `proxy` is the path the effect takes to shared memory. A `unit` is the
    mesh of one set of threads that issue the instruction together. These
    threads also run the steps of its group together. A `fence` effect on
    registers needs the fence of its group after any other access to those
    registers. The exception is an earlier effect with the same `chain`, when
    that `chain` is not None.
  - Lookup uses the exact class of `issued.op`, or `key` when it is given.
    Every `DispatchRegistry` looks up this way. A class with no registration
    answers `Mechanism.ON_RETURN` for every effect. A class does not inherit
    the answer of its base class. So the emitter of such a class MUST complete
    its work before the statement returns.
  - The registry belongs to CUDA. Only the `sync` family reads it.
    `cuda/analysis/completion.py` registers every handler when it is imported.
    `cuda/analysis/sync.py` imports it. No Op file registers a completion. The
    handlers are:

    | Class | Its effects complete |
    |---|---|
    | `CopyAsync` | in `Group.CP_ASYNC` |
    | the WGMMA atom | as [tir MMA atoms](./tir.md#declarative-mma-atoms-and-ttiled_mma) states |
    | `TiledMma` | as the handler of its atom states, looked up with `key` |
