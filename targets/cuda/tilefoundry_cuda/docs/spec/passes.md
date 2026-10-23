# TileFoundry CUDA Spec — Passes

### 7.3 `insert_default_host_entry`

```python
class InsertHostEntryPass(ModulePass):
    """Give a device-only module a host-callable entry.

    Attributes:
        name: attribute; Stable dump and log name.
        requires: attribute; Ordered dependency assertion.
    """

    name: str = "insert_host_entry"
    requires: tuple[str, ...] = ()

    def run(self, module: Module) -> Module: ...


def insert_default_host_entry(module: Module) -> Module:
    """Return a Module whose entry is host-callable.

    Args:
        module: Module to normalize.

    Returns:
        The unchanged or host-entry-normalized Module.
    """
    ...
```

- constraints:
  - The pass is the scheduled form of the transform and states no rule of its
    own. CUDA registers it at `PassPoint.BEFORE_CODEGEN`; compile and scaffold
    select it through `target_passes`.
  - A CPU entry MUST pass through unchanged.
  - With no CPU entry, an all-HIR tree MUST gain one CPU entry per Module
    whose `entry` is set, root and children alike: Python begins a kernel
    invocation only by entering one ([hir §1.1](../../../../../docs/spec/hir.md#11-function),
    [core-ir §1](../../../../../docs/spec/core-ir.md#1-module)). Any other function gets no entry.
    The tree's resolved Target MUST be CUDA. Each entry launches the Module
    entry by its root-relative call path
    ([core-ir §1](../../../../../docs/spec/core-ir.md#1-module)) at the owner's
    effective topology and forwards its own parameters: the declared inputs,
    then one destination per tensor result leaf in tuple leaf order. A
    destination keeps the leaf's shape and dtype, with GMEM storage and no
    layout, and is named `out`, else its first free `_1`, `_2`, ... suffix,
    with the declared names reserved. A unit result adds none; any other
    result type MUST be rejected. The pass checks these when it builds the
    `Launch`, so later stages read them and do not derive them again. The
    authored root entry remains unchanged. The entry's IR name is
    `<fn.name>_host`, with the same suffix rule on collisions; its emitted C++
    and registration names follow the HIR callee's path
    ([codegen §4.4](../../../../../docs/spec/codegen.md#44-signatures)).
  - HIR device calls into a direct child's function with implicit ConstTensor
    arguments MUST be rejected, naming caller, callee and omitted weights and
    requesting explicit weight parameters or orchestration calls.
  - Mixed HIR/TIR device trees MUST be rejected.
  - With no CPU entry and exactly one CUDA TIR device function, the transform MUST
    synthesize a CPU entry that mirrors that function's parameters and launches
    it. One rule covers both device shapes: whether the call reads as a launch
    or as a dispatch follows from the callee's own shape, a lone kernel and a
    specialization prototype taking the same route here.
  - It MUST NOT rewrite the Target any function records. Which target a
    function runs on is what its author wrote.
  - The launch geometry MUST be settled here and written into the `Launch`, so
    nothing downstream derives it again. A prototype states no body of its own,
    so its variants state its geometry and they MUST agree.
  - It MUST reject ambiguous device-function sets, a non-entry CPU function, or
    a launch-provided dynamic CTA extent.
