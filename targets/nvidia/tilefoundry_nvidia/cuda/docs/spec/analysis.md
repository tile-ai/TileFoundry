# TileFoundry CUDA Spec — Analysis

## 1. `sync`

`cuda.Target.get_analyzer("sync")` adds this family to the core families. See
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
    storage: StorageKind | None = None   # the memory a proxy fence publishes: GMEM, or None for smem

class SyncMetadata(IRMetadata):
    steps: tuple[tuple[object, tuple[SyncStep, ...]], ...] = ()   # (tag, steps)
    def at(self, tag=None) -> tuple[SyncStep, ...]: ...

class SyncPlanMetadata(IRMetadata):
    shared: bool = False       # whether any of its actions touches shared memory
    barriers: tuple[int, int] | None = None   # (offset, count) of the per-warp mbarrier words
    global_async: bool = False # whether any of its actions reaches gmem through the async proxy
    errors: tuple[str, ...] = ()   # non-fatal: the mbarrier words ending above capacity
```

- constraints:
  - **Steps.** Each analysis replaces the `SyncMetadata` of every node of the
    Function. Under one tag, steps run in the order of `StepKind`, then of
    `Group`, then of `storage`. A tag holds each step once. Steps for different
    groups or storages stay apart.
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
    | the copies lowering adds, the parameters a Call's arguments supply, the mesh an instruction is issued on, whether it is a tensor-map load | `cuda/analysis/emission.py` |
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
  - **Global memory roots.** An access to global memory touches the
    allocation root of its value. A Function other than the entry touches an
    unknown root for a value it returns, and any Function for a global value
    the memory family did not record. Unknown roots match all roots.
  - **Order.** A later access conflicts with an earlier one and follows it
    after steps, as this table states:

    | | Shared memory | Global memory | Register storage |
    |---|---|---|---|
    | Conflict | one of the two writes and their bytes overlap | one of the two writes, one of the two goes through the async proxy, and their roots match | the earlier access is in flight |
    | Conflict in one group | as above | as above | only when one of the two writes and they share no non-None `chain` |
    | No conflict | the earlier access is complete, and both are operands of instructions without an execution mesh, with one `ShardLayout` that splits along each mesh axis whose threads share shared memory | - | - |
    | Steps | a commit and a wait for an earlier group effect, a proxy fence for an async access after a generic one, then a barrier over a scope whose mesh chain is a prefix of the later access's | as for shared memory, with a proxy fence over global memory | a commit and a wait for the group of the earlier access, and no barrier |

    A proxy fence runs on every thread of its scope before the barrier, so
    the threads that made the generic accesses publish them before the async
    access issues. An async access needs no fence before a later generic one:
    the completion of a tensor-map copy is followed by an implicit
    generic-async proxy fence (PTX `cp.async.bulk`).
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
    the later one. A barrier runs over the mesh of that scope. When that mesh
    holds one thread, no barrier runs: the thread's own program order orders
    its plain accesses. Only the barrier is left out. The commit and wait of a
    group in flight and a proxy fence still run, since an asynchronous write
    reaches even its issuing thread only through them. Otherwise the mesh MUST
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
    conflicts with them waits for every `cp.async` group in flight. Unless the
    callee has a plan whose `global_async` is False, the call also reads and
    writes unknown global memory through the async proxy, the callee's own
    temporaries included.
  - **Barrier words.** A Function with a `CopyAsyncTensor` load from global to
    shared memory, or a retained call whose callee plan has `barriers`, gets
    `barriers = (offset, count)`: one 8-byte mbarrier word per warp of the CTA,
    `count = ceil(threads / W)` with `W` the target's
    `architecture.max_threads_per_warp`, at `offset`, the entry's shared-memory peak
    rounded up to 8. Only the entry places shared memory, so every Function it
    reaches shares the offset. Words that end above the target's capacity
    refuse nothing, as a peak above it refuses nothing. When the placed peak
    fits the capacity and the words do not, `errors` says so; a peak already
    above capacity is the memory family's error, and an unknown capacity adds
    none. `SyncPlanMetadata` is a structured record of the `sync` report
    family, so a JSON report that requests `sync` carries it.
  - **CTAs.** No step orders two CTAs of one launch. In an entry launched on
    more than one CTA, `ctas.py` pairs the global memory reads and writes of
    its actions where one of the two goes through the async proxy, one of the
    two writes and their roots match; each pairs with itself too. Each side is
    the device access relation of its action, constrained by the loops around
    it. The second side renames every coordinate except the entry's own
    parameters and module dimensions, so two iterations of one action pair as
    well. Each execution runs in one physical CTA of the launch, and every CTA
    coordinate its relation names is the coordinate that CTA reads, as the
    runtime's `mesh_coords` reads it; the two CTAs differ. A coordinate that
    cannot be bound this way leaves the pair unproven. Only the sliced CTA
    meshes around an access keep a CTA from running it, as the runtime's
    `contains` guard does; a mesh smaller than its topology repeats over the
    launch. A slice whose positions are not static keeps no CTA out. For two
    different CTAs, the first row that holds decides a pair:

    | Pair | Outcome |
    |---|---|
    | both are the same `CopyReduceAsyncTensor` kind | accepted |
    | a relation is unknown or only an upper bound of what it may access, the two relations are over different targets, or a CTA coordinate cannot be bound | refused: they may reach one element; an upper bound cannot confirm a conflict |
    | the two relations reach no common element | accepted |
    | they reach a common element and every name keeps its source | refused: they reach one element |
    | they reach a common element and a renamed name loses its source | refused: they may reach one element |

    A retained call's relations are unknown.
  - **Threads.** No step orders the threads of one issue. An instruction that
    states no execution mesh of its own is issued by every thread of the mesh
    it runs on, each on the operands as that thread sees them. A retained
    call, an instruction with a required execution mesh, and one that takes
    its execution mesh as an attribute, as `CopyAsyncTensor` does, state for
    themselves which threads issue. For every other action, `threads.py` pairs
    each shared memory write with itself the way **CTAs** pairs global
    accesses, with thread coordinates in place of CTA coordinates: the second
    side renames every thread coordinate, and only the sliced thread meshes
    around the action keep a thread from issuing it. A copy or fill that
    lowering adds runs in the regions `transfer_meshes` gives it. When no
    operand's mesh splits the threads and the destination is in shared
    memory, the innermost of them is a slice holding the first thread of the
    enclosing mesh, and it counts as a slice around the action. A write needs
    no relation when a layout splits it over a mesh whose positions are
    exactly the threads that issue it, those every thread slice around it
    holds, or every thread of the CTA without one. That layout is the
    destination's own or, for a `Copy`, that of the one operand
    that splits every thread axis, since `Copy` projects a plain operand
    through it. Two different threads that reach one element are otherwise
    decided by the rows of the **CTAs** table. The family checks **Threads**
    after **CTAs**.
  - **Word users.** The word of warp `k` has one user, thread `W * k`: a
    `CopyAsyncTensor` is issued by whole aligned warps, as its execution mesh
    pattern states ([tir](./tir.md#copyasynctensor)), so its first issuing
    thread starts a warp. The plan orders no word.
  - **Refusals.** The family raises `analysis.Error` when an action or a barrier
    breaks a MUST of **Units** or **Placement**. It also raises it for a loop
    whose state does not settle within its limit, and for a pair that
    **CTAs** or **Threads** refuses.

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

completion_registry: Registry     # keyed by class
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
    | `CopyAsyncTensor`, `CopyReduceAsyncTensor` | on return; the shared operand through `Proxy.ASYNC` ([passes §7.1](./passes.md#71-converthirtotir)) |
