# TileFoundry CUDA Spec — Passes

### 7.1 `ConvertHIRToTIR`

```python
class CudaConvertHIRToTIR(ConvertHIRToTIR):
    """Lower scheduled HIR with the visitor that emits the sync steps."""

    visitor = CudaLowering
    analyses = ("memory", "sync")


class CudaLowering(Lowering):
    """The core lowering, emitting the steps the ``sync`` analysis attached to each node."""

    def run(self) -> PrimFunction: ...
    def visit_MeshRegion(self, region: MeshRegion, ctx): ...
    def visit_callee(self, call: Call, operands, ctx) -> Expr | View: ...
    def visit_T(self, call: Call, operands, ctx) -> Expr | View: ...
    def visit_LoopRegion(self, loop: LoopRegion, ctx) -> Expr: ...
    def visit_InsertSlice(self, call: Call, operands, ctx) -> Expr: ...
    def visit_Zeros(self, call: Call, operands, ctx) -> Expr | View: ...
    def visit_FullLike(self, call: Call, operands, ctx) -> Expr | View: ...
    def visit_T_CopyAsyncTensor(self, call: Call, op, operands, ctx) -> Expr: ...
    def default_visit_T(self, call: Call, op, operands, ctx, emit=None) -> Expr: ...
```

- constraints:
  - CUDA binds the pass with `@bind(CudaTarget, ConvertHIRToTIR)`; finalization
    and compile obtain it through `Target.resolve`.
  - Every function the pass lowers MUST carry `SyncPlanMetadata`
    ([analysis §1](./analysis.md#1-sync));
    otherwise lowering raises `LoweringError` and does not analyse again. The
    pass names both families in `analyses`: a caller prepares the Module with
    `analyze(module, entry, analysis=("memory", "sync"))` and passes the result
    to `finalize`; `tilefoundry schedule finalize` reads `analyses`.
  - Which copies the loop and `InsertSlice` visits add, which parameters a
    Call's arguments supply and the mesh an instruction is issued on come from
    `cuda/analysis/emission.py`, which the `sync` family also reads.
  - The visitor emits the steps a node's `SyncMetadata` holds for a tag, in
    order, where that tag's statements are lowered:

    | Visit | Tag | Emitted |
    |---|---|---|
    | `visit_MeshRegion`, `visit_callee`, `visit_T` | None | first, then the core visit; the operands are already lowered |
    | `visit_Zeros`, `visit_FullLike` | None | after the result storage is declared, before the `Fill` |
    | `visit_LoopRegion` | `("entry", slot)` | before the slot's entry copy |
    | `visit_LoopRegion` | None | after the entry copies, before the loop |
    | `visit_LoopRegion` | `("backedge", slot)` | inside the loop, before the slot's backedge copy |
    | `visit_InsertSlice` | `"base"`, `"update"` | before the base copy, before the update copy |
    | `default_visit_T`, `visit_T_CopyAsyncTensor` | `"issue"` | before each issue |
    | `default_visit_T`, `visit_T_CopyAsyncTensor` | `"after"` | after the issues, inside the region that issues them |

    `visit_T` also covers a `Reshard` that is not a view. The loop,
    `InsertSlice`, fill and instruction visits exist only here: the core
    visitor refuses those nodes ([passes §7.1](../../../../../../docs/spec/passes.md#71-converthirtotir)).
  - For `Group.CP_ASYNC` a commit is `cp_async_commit` and a wait
    `cp_async_wait(n=0)`; for `Group.WGMMA` a commit is `wgmma_commit_group`,
    a wait `wgmma_wait_group(n=0)` and a fence `wgmma_fence`. A
    `StepKind.PROXY_FENCE` is `fence_proxy_async` over the step's `storage` and
    a `StepKind.BARRIER` `T.sync` over the step's mesh.
  - `CopyAsyncTensor` and `CopyReduceAsyncTensor` register no asynchronous
    completion group: their effects complete on return, the shared operand
    through the async proxy ([analysis §2](./analysis.md#2-completion-registry)).
    Their emitters MUST finish each copy before the operation ends: an mbarrier
    wait for a load, a full bulk wait (not `.read`) for a store or reduction,
    each followed by a meeting of the issuing threads.
  - `visit_T_CopyAsyncTensor` lowers a load from global to shared memory to
    `CopyAsyncTensorTx` wherever `default_visit_T` would issue the load, with
    the words `SyncPlanMetadata.barriers` places as `barrier`: `count` words of
    8 bytes from `offset`. A store is lowered by `default_visit_T`.

### 7.3 `insert_default_host_entry`

```python
class CudaInsertHostEntryPass(InsertHostEntryPass):
    """Give launchable HIR Module entries a host-callable entry.

    Attributes:
        name: attribute; Stable dump and log name.
        requires: attribute; Ordered dependency assertion.
    """

    def run(self, module: Module) -> Module: ...


def insert_default_host_entry(module: Module) -> Module:
    """Preserve TIR and append host wrappers for launchable HIR entries.

    Args:
        module: Module to normalize.

    Returns:
        The unchanged or host-entry-normalized Module.
    """
    ...
```

- constraints:
  - The pass is the scheduled form of the transform and states no rule of its
    own. CUDA binds it to the core `InsertHostEntryPass` with
    `@bind(CudaTarget, InsertHostEntryPass)`; finalization, compile and scaffold
    obtain it through `Target.resolve`.
  - A CPU entry MUST pass through unchanged.
  - When the root Module entry receives a host entry, the pass sets `Module.entry`
    to that host entry.
  - With no CPU entry, an all-HIR tree MUST gain one CPU entry per Module
    whose `entry` is set, whose result leaves are all GMEM or UMAT, and whose
    call path crosses no indexed slot, root and children alike: Python begins a kernel
    invocation only by entering one ([hir §1.1](../../../../../../docs/spec/hir.md#11-function),
    [core-ir §1](../../../../../../docs/spec/core-ir.md#1-module)). Any other function gets no entry.
    The tree's resolved Target MUST be CUDA. Each entry launches the Module
    entry by its root-relative call path
    ([core-ir §1](../../../../../../docs/spec/core-ir.md#1-module)) at the owner's
    effective topology and forwards its own parameters: the declared inputs,
    then one destination per tensor result leaf in tuple leaf order. A
    destination keeps the leaf's shape and dtype, with GMEM storage and no
    layout, and is named `out`, else its first free `_1`, `_2`, ... suffix,
    with the declared names reserved. A unit result adds none; any other
    result type MUST be rejected. The pass checks these when it builds the
    `Launch`, so later stages read them and do not derive them again. The
    entry's IR name is `<fn.name>_host`, with the same suffix rule on collisions; its emitted C++
    and registration names follow the HIR callee's path
    ([codegen §4.4](../../../../../../docs/spec/codegen.md#44-signatures)).
  - An entry whose call path crosses an indexed slot MUST receive no host
    entry; indexed entry references await #235.
  - An HIR entry with an rmem or smem result leaf MUST receive no host entry,
    because a host cannot pass that storage.
  - Each synthesized `Launch` MUST state the entry function's analyzed
    shared-memory peak as `dynamic_smem`, and 0 when it carries no memory
    analysis. When the entry's `SyncPlanMetadata.barriers` is `(offset, count)`,
    it states `offset + 8 * count` instead.
  - Mixed HIR/TIR device trees MUST be rejected.
  - A TIR Module MUST pass through unchanged; it supplies its own CPU entry.
  - It MUST NOT rewrite the Target any function records. Which target a
    function runs on is what its author wrote.
  - The launch geometry MUST be settled here and written into the `Launch`, so
    nothing downstream derives it again.
  - It MUST reject a launch-provided dynamic CTA extent.
