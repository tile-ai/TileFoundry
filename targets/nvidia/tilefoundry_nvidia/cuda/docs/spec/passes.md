# TileFoundry CUDA Spec — Passes

### 7.1 `ConvertHIRToTIR`

```python
class CudaConvertHIRToTIR(ConvertHIRToTIR):
    """Lower scheduled HIR with the visitor that emits the sync plan."""

    visitor = CudaLowering


class CudaLowering(Lowering):
    """Emit each function's sync plan and check what lowering issues belongs to it."""

    def before(self, position: tuple[int, int], ctx, op: Op | None = None) -> None: ...
```

- constraints:
  - CUDA binds the pass with `@bind(CudaTarget, ConvertHIRToTIR)`; finalization
    and compile obtain it through `Target.resolve`.
  - Every function the pass lowers MUST carry `SyncPlanMetadata`
    ([analysis §1.2.5](../../../../../../docs/spec/analysis.md#125-sync)) whose
    `basis` is the allocation and liveness records it carries; otherwise
    lowering raises `LoweringError` and does not analyse again. A caller
    prepares the Module with `analyze(module, entry, analysis=("memory", "sync"))`
    and passes the result to `finalize`; `tilefoundry schedule finalize` does
    the same.
  - `before` emits the steps the plan holds for that position, in plan order:
    `cp_async_commit` for a commit, `cp_async_wait(n=0)` for a wait and
    `T.sync` over the step's mesh for a barrier.
  - Each planned child is reached once, in plan order, in its planned scope
    and for the planned instruction. An action's statements run until the
    next child: one register fill per staged parameter, then one issue, or
    one per issue point for an instruction with an execution mesh. Each issue
    equals the planned instruction, attributes included, and runs on the
    planned mesh.
  - Every shared memory operand of an action's statements carries the record
    of [passes §7.1](../../../../../../docs/spec/passes.md#71-converthirtotir);
    its leaf is an allocation root the planned role holds and, where the plan
    knows the bytes the role's values own in that allocation, its covered
    bytes lie within them. Unknown covered bytes there, a missing record or a
    root the role does not hold raise `LoweringError`. A staged parameter is issued with the scalar
    prepared for that action and parameter.
  - Statements outside actions MUST NOT take a shared memory operand.

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
    analysis.
  - Mixed HIR/TIR device trees MUST be rejected.
  - A TIR Module MUST pass through unchanged; it supplies its own CPU entry.
  - It MUST NOT rewrite the Target any function records. Which target a
    function runs on is what its author wrote.
  - The launch geometry MUST be settled here and written into the `Launch`, so
    nothing downstream derives it again.
  - It MUST reject a launch-provided dynamic CTA extent.
