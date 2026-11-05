# TileFoundry CUDA Spec — Analysis

## 1. `sync`

`CudaTarget.get_analyzer("sync")` adds this family to the core ones
([analysis §1.2](../../../../../../docs/spec/analysis.md#12-analysis-families));
a Target that does not add it refuses the selector
([analysis §2.2](../../../../../../docs/spec/analysis.md#22-target-selected-analyzers)).

| Selector | Requires | Owns | Attaches to | Rests on | Text summary adds | Annotates equations |
|---|---|---|---|---|---|---|
| `sync` | `memory` | `SyncPlanMetadata`, `SyncMetadata` | `SyncPlanMetadata` on the Function; `SyncMetadata` on each Call, `MeshRegion` or `LoopRegion` whose statements need steps | the memory family's records and the completion registry ([§2](#2-completion-registry)) | - | - |

The `sync` family decides where lowering emits the commits, waits, fences and
barriers that make every access to shared memory or to a register follow the
earlier accesses it depends on, or refuses the Function. It attaches
`SyncMetadata` to each HIR node whose statements need steps, and
`SyncPlanMetadata` to every analysed Function to mark it planned. CUDA lowering
emits a node's steps where it lowers that node and analyses nothing itself
([passes §7.1](./passes.md#71-converthirtotir)).

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
  - Steps belong to the HIR node lowering emits statements for: a `Call`, a
    `MeshRegion` or a `LoopRegion`. The tag None is before the node's own
    statements: the instruction, fill or retained call a Call emits, or
    entering a region. `"issue"` is before each issue of a Call's instruction,
    and `"after"` right after its issues, in the region that issues them. Any
    other tag names a copy lowering adds for the node: `"base"` and `"update"`
    of an InsertSlice, `("entry", slot)` and `("backedge", slot)` of a loop's
    carried slot. Under one tag the steps run in `StepKind`'s order, and steps
    of one kind in `Group`'s; a step is kept once per tag, and steps for
    different groups are kept apart. Each analysis replaces the `SyncMetadata`
    of every node of the Function.
  - The actions of a node are the ones lowering emits: the selected
    instruction of a Call, the copies and fills lowering adds, and a retained
    call, which reads its arguments and writes its results. The family
    dispatches a Call as `CudaLowering` does: a Call `lowers_to_dim` accepts has
    no action, and any other Call is a retained call, else goes to
    `visit_<HIR op>`, else to `visit_T`. Its `visit_*` MUST correspond one to
    one with the visits `CudaLowering` has or inherits, so a change to one
    MUST change the other: `TupleGetItem`, `Slice`, `Reshape`, `Bitcast`,
    `Empty` and `IndexSelect` have no action; a `Reshard` whose result has an
    accepted VIEW alias has none, and any other goes to `visit_T`; `Zeros` and
    `FullLike` are one `Fill`; `InsertSlice` is the copies
    `copies_insert_slice_base` and `copies_insert_slice_update` name; and
    `visit_T` issues the instruction `selected_op` names.
  - Which instruction a Call issues, its operand types and the mesh that owns
    it come from `passes/transforms/convert_hir_to_tir.py`; which copies lowering adds, which
    parameters a Call's arguments supply and the mesh an instruction is issued
    on come from `cuda/analysis/emission.py`. Lowering calls the same functions on
    the same analysed program. Each effect's completion comes from the
    completion registry ([§2](#2-completion-registry)); a retained
    call's register effects complete on return, since the callee's own plan
    completes its WGMMA groups before it returns. Its shared memory effects
    follow the rule for calls below.
  - A register access touches the storage lowering gives its value:
    `register_storage` follows the result of an instruction written in place,
    a Call whose visit issues its selected instruction, to its argument, and
    any other value along its first accepted alias. Every action's register
    reads and writes take part, not only the asynchronous ones.
  - Scopes are those of the Function's `IterationScope`
    ([analysis §2.1](../../../../../../docs/spec/analysis.md#21-shared-iterationscope-and-access)):
    the scope an expression sits in is its recorded location. The family
    builds no scope tree of its own. It orders each scope's children by their
    events on the Function's liveness timeline, the order lowering emits them
    in; no position leaves the family.
  - An access owns the shared memory bytes of its value's allocation leaf, as
    the memory family places it: each stage's offset and the stage size, all
    stages together, or for a Tuple result the leaf's own offset and size. A
    leaf without a placed offset, or whose allocation the Function returns and
    so may live in the output its caller passes, owns unknown bytes, which
    overlap everything.
    The plan orders whole allocations, so it says nothing about which stage or
    window an access selects.
  - Two shared memory accesses conflict when one of them writes and their bytes
    overlap. A later access follows an earlier one once a barrier has run over
    a scope whose mesh chain is a prefix of the later access's; before that
    barrier, an earlier group effect is committed and waited on in that scope.
    A later access through the async proxy after an earlier generic one also
    needs a proxy fence, run after the wait and before the barrier.
  - An instruction without an execution mesh is issued by each thread on its
    own shard, so an operand whose type is a shared memory `ShardLayout`
    splitting along every mesh axis whose units share shared memory has each
    byte touched by one thread. Two accesses with the same such layout touch
    every byte from the same thread, so once the earlier one is complete the
    later one follows it in that thread's program order and needs no step.
  - A register access conflicts with any earlier access to the same storage
    that is still in flight in a group, a read as much as a write, since PTX
    leaves touching a wgmma's registers before its wait undefined. A later
    effect of the same group conflicts only when one of the two writes and
    their `chain`s differ. The later follows once the earlier's group is
    committed and waited on; no barrier is needed. The fence state is kept
    per storage, which is exact only while every access to one register
    storage runs on the threads that hold its value. Operand reachability,
    Slice and Reshard keep that so: a Reshard of register storage is a view
    only when every element keeps its thread
    ([hir Reshard](../../../../../../docs/spec/hir.md#reshard)).
  - A `fence` effect needs its group's fence before the action when any access
    to its registers since the last fence is not an effect of that group with
    the same `chain`, or when its registers have no access yet. A `fence`
    effect without a `chain` needs it before every issue, under the tag
    `"issue"`.
  - An action with a `unit` effect MUST run in a scope whose mesh is whole,
    aligned units holding the issuing threads, with loop bounds the threads of
    each unit see alike: what a barrier accepts below, or a thread coordinate
    that reads one value across each unit. PTX `.aligned` asks this of the
    threads of one warpgroup, so two units may loop a different number of
    times. A step of that group needs the same of the scope it is placed in. When that scope does not have it, or when such an effect may still be
    in flight when the Function returns, the action completes in place: its
    group's commit and wait run under the tag `"after"`, and the Function is
    planned again. Its accesses stay recorded as complete, so later accesses
    from other threads still take their fences and barriers.
  - An action runs on the mesh lowering issues it on, or on its lexical
    scope's mesh when that is None. That mesh MUST lie within its lexical
    scope's mesh (`within_scope`). Steps are placed in the innermost scope that
    holds both accesses, before the child that holds the later one, so the
    issuing threads of both run them; a barrier runs over that scope's mesh.
    That mesh MUST be a thread mesh `classify` supports, and every loop around
    the position MUST have bounds every thread sees alike: constants, kernel
    parameters and their captures, induction variables of such loops, and
    coordinates of mesh levels the barrier's mesh does not span.
  - Loops are analysed to a fixed point: the head state joins every body end
    until it repeats.
  - Callees are planned before their callers, and each orders its own
    accesses. Unless the callee's plan says none of its actions touches shared
    memory, or when the callee has no plan, a retained call also reads and
    writes unknown shared memory bytes and completes as a `cp.async` group,
    since the callee may leave groups in flight: whatever conflicts with it
    waits for them all.
  - The family raises `AnalysisError` for an action whose mesh is not provably
    within its scope, for an action with a `unit` effect whose own scope is
    not whole units with loop bounds each unit's threads see alike, for a barrier no
    supported uniform scope can hold, and for a loop whose state does not
    settle within its limit.

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
  - A handler answers one `Effect` per read and per write of the issued
    parameters; `effects` builds them in that order, reads of a parameter
    first. It sees the parameter types, so an effect can depend on where an
    operand lives.
  - `Mechanism.ON_RETURN` means the effect is complete for the issuing threads
    once the statement returns. `Mechanism.GROUP` means it completes when those
    threads wait on the `group` that holds it.
  - `proxy` is the path the effect takes to shared memory. A `unit` names the
    mesh of one group of threads that issue the instruction, and run its
    group's steps, together. A `fence` effect on registers needs the group's
    fence after any other access to those registers, except an earlier effect
    with the same non-None `chain`.
  - Lookup uses the exact class of `issued.op`, or `key`, as every
    `DispatchRegistry` does. A class with no registration answers
    `Mechanism.ON_RETURN` for every effect; a base class's answer is not
    inherited. The emitter of such a class MUST therefore complete its work
    before the statement returns.
  - The registry belongs to CUDA and only the `sync` family reads it.
    `cuda/analysis/completion.py` registers every handler when it is imported,
    and `cuda/analysis/sync.py` imports it: `CopyAsync` completes in
    `Group.CP_ASYNC`, the WGMMA atom as
    [tir MMA atoms](./tir.md#declarative-mma-atoms-and-ttiled_mma) states, and
    `TiledMma` answers with its atom's handler, looked up with `key`. No Op
    file registers a completion.
