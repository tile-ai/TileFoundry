# TileFoundry Spec — passes

The pass framework (`Pass`, `ModulePass`, `FunctionPass`, `PrimFuncPass`,
`PassManager`), then the implemented passes and their contracts. A `Pass` is
the unit the compiler schedules over a `Module`.

```mermaid
flowchart TB
    Pass["<b>Pass</b> (ABC)"]
    ModulePass["<b>ModulePass</b> (Pass)"]
    FunctionPass["<b>FunctionPass</b> (Pass)"]
    PrimFuncPass["<b>PrimFuncPass</b> (Pass)"]
    PassManager["<b>PassManager</b><br/>holds list[Pass]"]

    Pass --> ModulePass
    Pass --> FunctionPass
    Pass --> PrimFuncPass
    PassManager -. schedules .-> Pass
```

## 1. Role

A `Pass` abstracts every Module-reading or Module-rewriting compile step.
`PassManager` runs the passes added with `PassManager.add(...)` sequentially,
in registration order. A pass has one of three granularities: `ModulePass`,
`FunctionPass` or `PrimFuncPass`.

## 2. `Pass` base class

```python
from abc import ABC, abstractmethod
from tilefoundry.ir import core

class Pass(ABC):
    """A Module → Module function.

    Attributes:
        name: attribute; Stable dump and log name.
        requires: attribute; Ordered dependency assertion.
    """

    name: str = ""
    requires: tuple[str, ...] = ()

    @abstractmethod
    def run(self, module: core.Module) -> core.Module: ...
```

- constraints:
  - `run(module)` returns the resulting `Module`. `Module` is mutable, and a
    pass MAY rewrite the Module tree in place (`Module.replace_functions`) and
    return the same object; child Modules are shared by identity. The default
    `FunctionPass.run` / `PrimFuncPass.run` return the input unchanged when no
    function changed, and otherwise a `dataclasses.replace` copy.
  - Passes MUST NOT depend on global state. All configuration enters
    through constructor parameters or pass-local attributes.
  - Pass failures raise named exceptions (e.g. `VerifyError`); they
    do not swallow errors.

## 3. Three pass granularities

### 3.1 `ModulePass`

Runs over the whole `Module` and may add, remove or reorder functions, for
example by synthesizing an entry the module does not declare
([§7.3](#73-inserthostentrypass)).

```python
class ModulePass(Pass):
    """Run over a complete Module."""

    @abstractmethod
    def run(self, module: core.Module) -> core.Module: ...
```

A `ModulePass` inherits the [§2](#2-pass-base-class) contract.

### 3.2 `FunctionPass`

Visits each `core.Function` in `module.functions`. The default `run` calls
`run_function` for HIR entries, passes other entries through, and reassembles
the `Module`.

```python
class FunctionPass(Pass):
    """Run independently over each HIR Function."""

    @abstractmethod
    def run_function(self, fn: core.Function, module: core.Module) -> core.Function: ...
    def run(self, module: core.Module) -> core.Module: ...
```

### 3.3 `PrimFuncPass`

Same shape as `FunctionPass`, but visits `core.PrimFunction` entries and
passes HIR entries through.

```python
class PrimFuncPass(Pass):
    """Run independently over each TIR PrimFunction."""

    @abstractmethod
    def run_prim_func(self, fn: core.PrimFunction, module: core.Module) -> core.PrimFunction: ...
    def run(self, module: core.Module) -> core.Module: ...
```

## 4. Transform pass idiom

Transform passes use the visitor / mutator base classes from
[visitor-mutator](./visitor-mutator.md) rather than hand-written `isinstance`
dispatch. A transform `PrimFuncPass` wraps a `StmtExprMutator` subclass:
`run_prim_func` runs the mutator over `fn.body`, returns `fn` when the returned
body is the same object, and otherwise `replace(fn, body=new_body)`. The
visit-and-rewrite contract, including the `visit_Evaluate` entry form for TIR
effect Ops, is owned by
[visitor-mutator §7](./visitor-mutator.md#7-visitor-entry-forms-for-evaluate).

## 5. `PassManager`

A linear scheduler that runs passes in registration order.

```python
class PassManager:
    """Ordered pass pipeline.

    Attributes:
        passes: attribute; the registered passes, run in registration order.
    """

    passes: list[Pass] = field(default_factory=list)

    def add(self, p: Pass) -> "PassManager": ...      # register a pass; returns self for chaining
    def run(self, module: core.Module) -> core.Module: ...      # run passes in registration order
```

- constraints:
  - `run` first checks `requires`: each name MUST be the `name` of a pass added
    earlier, otherwise `ValueError`. It is an ordering assertion, not a
    topological sort.

`PassManager` does not own a dump destination. When the caller wraps the run in
a `tilefoundry.dump.Scope` and `Flags.PASS_IR` is enabled, each pass writes
`before.txt` / `after.txt` under `<scope-root>/{NN}_{pass_name}/`. Otherwise
`dump(...)` is a no-op. The before/after representations MUST be passed as
no-argument producers, so `repr` is evaluated only when `PASS_IR` is enabled in
an active scope.

## 6. Top-level API

Two public verbs operate on `core.Module` only:

```python
class Options:  # compiler.Options; the root re-exports it as CompilerOptions
    """Gather the options of every stage, one part per stage.

    Attributes:
        target: attribute; Constructed compilation Target.
        passes: attribute; ``passes.Options`` handed to ``passes.default_pipeline``.
        codegen: attribute; ``codegen.Options`` handed to ``codegen.link_program``.
        dump: attribute; ``dump.Flags`` that choose what the pipeline dumps.
    """

    target: Target
    passes: passes.Options = passes.Options()
    codegen: codegen.Options = codegen.Options()
    dump: dump.Flags = dump.Flags.ALL


def build(mod: core.Module, /, *, target: Target | None = None, options: Options | None = None) -> runtime.Module: ...
def compile(mod: core.Module, /, *, target: Target | None = None, options: Options | None = None) -> runtime.Module: ...
```

`build` takes a TIR `Module` whose `Module.entry` is a CPU host entry, then
codegen → toolchain link → loader, and returns a `runtime.Module`
([runtime](./runtime.md)). It requires the Module to own its Target; an
explicit Target that disagrees with the declared one is an error.

`compile` runs `passes.default_pipeline(module, options.passes)` over the
Module, then `build`, which hands `options.codegen` to `codegen.link_program`;
both run inside a `dump.Scope` opened with `options.dump`, which only narrows
the scope already active. The pass list is [§7.2](#72-target-pass-points). The
`jit` convenience, which also accepts a bare Function, and its cache contract
are owned by [runtime §1.6](./runtime.md#16-compiler).

### Dirty-scope retype / verify

`typeinfer` and `verify` are not standalone pipeline stages, and there is no
`TypeInferPass`. The parser already runs eager typeinfer
([parser](./parser.md)), so a `Module` entering the pipeline has `Expr.type`
filled. After each pass, `PassManager` rechecks the pass's **dirty scope**: every
function on the Module's call paths whose object identity was not on them before
the pass, and every function that reaches one of those by a call or a launch at
any depth, through its variants too. Each reference resolves from the frame of
the function that states it; a reference that does not resolve raises.
Rechecking is per whole function.

- HIR: `verify_function`, then type inference with ranges, which refreshes
  `RangeMetadata` over the complete function.
- TIR: `verify_prim_function`, which retriggers `typeinfer` on the embedded Expr
  fields and refreshes their `.type`.

A rewrite that keeps a function's identity, and changes no function it reaches,
is therefore not rechecked.

## 7. Implemented passes

### 7.1 `ConvertHIRToTIR`

```python
class ConvertHIRToTIR(ModulePass):
    """Replace one selected scheduled HIR entry with verified TIR."""

    visitor: ClassVar[type[Lowering]] = Lowering
    options: passes.Options = passes.Options()
    name: str = "convert-hir-to-tir"
    requires: tuple[str, ...] = ()
    analyses: tuple[str, ...] = ("memory",)

    @classmethod
    def create(cls, target: Target, options: passes.Options) -> ConvertHIRToTIR: ...
    def run(self, module: core.Module) -> core.Module: ...
```

- constraints:
  - The pass lowers the device root of the Module it receives, in place, and
    its retained calls. The device root is the callee of the `Launch` in
    `Module.entry` when that entry is a CPU host, otherwise `Module.entry`
    itself. The root MUST carry the memory analysis
    (`AllocationSummaryMetadata`); otherwise the pass raises `LoweringError`. It
    never checks or analyzes again, and it takes nothing but configuration at
    construction. Generated names come from the `BindingMetadata` on that IR.
  - A Target MAY bind a subclass with `@bind(TargetClass, ConvertHIRToTIR)`;
    finalization obtains the pass through `Target.resolve(ConvertHIRToTIR)`. The
    entry function and every callee are lowered by the same `visitor` class.
  - `analyses` names the analysis families the pass's visitor reads; a
    subclass whose visitor reads more names them too. `tilefoundry schedule
    finalize` runs exactly these on the pass `Target.resolve` gives.
  - Whether a Call is a dimension, the instruction it issues and its operand
    types, the mesh that owns a copy or an instruction, and the accepted
    aliases lowering executes come from the module-level functions of
    `passes/transforms/convert_hir_to_tir.py`. A target analysis that follows
    lowering imports the same functions.
  - The visitor lowers each HIR node in its `visit_<Node>`. `visit_Call` lowers
    a Call that `lowers_to_dim` accepts as a dimension; for any other Call it
    lowers the values its storage's dimensions read, then its operands, and
    dispatches a retained call to `visit_callee`, else to `visit_<HIR op>`, else
    to `visit_T`, which dispatches the selected TIR op to `visit_T_<Name>`, else
    to `default_visit_T`. `visit_<HIR op>` and the Expr visits share one name
    space, so the visitor MUST NOT define a visit for an HIR op whose class name
    is also an Expr class (for example `ShapeOf`).
  - The core visitor lowers mesh regions, retained calls, the views
    `TupleGetItem`, `Slice`, `Reshape` and `Bitcast`, a `Reshard` whose result
    has an accepted VIEW alias, and `Empty`; any other `Reshard` goes to
    `visit_T`, and an unscheduled `IndexSelect` raises `LoweringError`. Its
    `default_visit_T` raises `LoweringError` naming the Call, the TIR op and the
    Target, and a node without a visit raises `LoweringError` naming the node
    and the Target. A target visitor overrides the visits it changes and
    implements the rest: loops, `InsertSlice`, `Zeros`, `FullLike` and
    instruction issue.
  - Entering the function body, and in `on_region_start` any region's body,
    the core visitor first lowers the children `LivenessMetadata.orders` lists
    for it, in that order ([analysis §1.2.2](./analysis.md#122-memory)), so
    statements follow the events the analysis planned with. A target visitor
    that lowers loops enters each loop's body through `on_region_start` inside
    the loop's builder region.
  - The pass lowers the device root: the callee of the `Launch` in `Module.entry`
    when that entry is a CPU host, otherwise `Module.entry` itself. It returns the
    Module unchanged under `passes.Options(lower=False)` or when the Module has no
    HIR function. When lowering HIR from a CPU host root, exactly one `Launch` MUST
    resolve to an HIR function; otherwise `LoweringError` names the host and the
    number of such launches.
  - Address placement comes only from `MemoryMetadata`: integer shared-memory
    addresses are byte offsets, and every analyzed offset becomes one concrete
    buffer view. Multiple offsets bind one view per offset and one tuple of
    those views outside the outermost enclosing HIR loop; each consuming scope
    binds the selected stage `stages[index]` once. The index is the linear
    iteration ordinal
    from outermost to innermost, modulo the number of buffers, when inner loop
    trip counts are constant. Otherwise selection uses a rank-0 integer register
    counter modulo the number of buffers. It is initialized to zero before the
    outermost staged loop and incremented after all uses in each completed
    innermost staged iteration; all uses in one iteration see the same slot.
    Instruction-generated loops do not enter this index.
    An adopted update window is written at its producer. `InsertSlice` skips
    copying only when the update's alias names that HIR destination, resolved
    through region captures.
    A loop-carried entry alias must name its slot's initial value; any other
    entry alias raises `LoweringError`. A accepted semantic alias identifies its
    destination as `BufferAlias.target: SSAValue`. Lowering MUST select the
    flattened target leaf, including nested tuple paths, before binding storage.
    Lowering MUST execute accepted ties at
    every storage level: a tied carry reuses its initial storage; otherwise it
    copies the initial into independent phi storage before the loop. An accepted
    yield-to-phi alias binds the producer output to phi storage, including rmem.
    The backedge needs no copy when the yield has that tie or its tied
    InsertSlice destination resolves through captures to the phi; otherwise
    lowering MUST copy yield to phi. A tied InsertSlice returns destination
    storage; an untied one MUST copy destination to result before writing the
    update window. An update without an adopted WINDOW alias MUST be copied
    into the result window. Allocation's peak-estimation storage links do not imply
    that lowering shares registers; only accepted semantic aliases state that sharing.
  - Schedule type inference and candidate admission MUST share one geometric
    projection implementation with lowering. It infers `repeat`, `order`,
    local issue shapes and `bind`. Type inference and candidates additionally
    call `check_call` to check the declared execution mesh, operand reachability,
    projected atom patterns, `between` relations and registered leaf verifiers.
    Lowering only consumes the projection; it does not repeat admission checks.
    An unscheduled HIR operation MUST have exactly one registered candidate;
    its operand hook supplies any explicit source view (including Transpose).
  - One `LoweringContext` carries read-only `StoragePlan` and `IssuePlan`
    answers, `Names`, and a `TirBuilder`. Only the builder writes TIR.
  - HIR mesh and loop regions become `MeshScope` and `For`; register allocations
    become `AllocTensor`; scheduled and ordinary value operations become
    effect-form TIR calls. Views follow their storage root, so a parameter view
    is addressed through `PtrOf` rather than an analyzed arena offset.
  - Lowering MUST obtain HIR lexical ownership, captures and enclosing execution
    meshes from the shared valuation visitor/context. Planning stores references
    to those contexts rather than constructing a separate region tree. TIR
    statement builders retain their own emitted scopes and destinations.
  - Generated region bindings MUST be reserved in lexical pre-order before
    statement emission, so an outer `scope` index precedes its nested indices.
    The root region reserves `cta`; all other mesh regions, including zero-only
    regions, reserve `scope` names.
    Every HIR `MeshRegion` becomes one `MeshScope`, with its arguments evaluated
    outside it. Non-atom instructions without an operand-declared mesh use
    the current scope, retaining unit axes. An instruction issued on the enclosing
    scope's mesh is emitted directly; another declared mesh opens a nested
    `MeshScope`. Meshes compare after resolving layouts, preserving unit axes.
    IR and type mesh names are retained; lowering MUST NOT rebuild mesh frames.
    Adjacent scopes are not merged. A view is bound in each consuming scope,
    never at its definition; its sharding refers to that scope's mesh.
    Storage declarations follow their HIR definition's scope and position.
    Constant-address `smem` and scratch-backed `gmem` views are materialized in
    each consuming scope. Visit memoizes each HIR node's lowered result in the
    context; forwarded SSA leaves reuse that result. Storage links MUST NOT
    introduce TensorView wrappers. HIR view operations retain their corresponding
    views, including authored identity views. Instruction operands use the same
    Visit path; only a required issue tile introduces a new operand window.
    A window that starts at zero and spans its operand's shape is the operand
    itself. Lowering issues that operand, or its own pointer where the window
    takes another type, and builds no `Slice` for it.
    Lowering MUST obtain storage answers from StoragePlan, the sole reader of analysis
    storage metadata in this pass. The context reads `MemoryMetadata.offsets`,
    accepted links in AllocationSummaryMetadata.values, and the unchanged
    AllocationSummaryMetadata.address_decisions. It chooses direct storage partners,
    output positions and refusals; analysis publishes no derived aliases.
    Each HIR view is formed on its visited operand using the registered op's
    own result-to-operand relation. StoragePlan computes physical view recipes;
    TirBuilder only declares, binds, materializes and emits them. Adopted update windows use InsertSlice's offsets
    at the storage representative; identity forwards need no inverse relation.
    Moving a decision through its structural chain, or placing a written
    producer on an output reached through a view, requires identity coordinates.
    A nonidentity step raises `LoweringError` naming the value and relation;
    lowering MUST NOT silently discard the required inverse relation.
    Copy decisions query accepted kind and partner together: INSERT_SLICE for
    destination copies, WINDOW for update copies, CARRY_INIT for entry copies,
    and CARRY_YIELD for backedge copies. Multiple ties remain distinct.
    Nested view text may differ from a flattened root view, but the physical
    address mapping MUST remain the same. The context derives origin and strides
    from the partner's physical layout; an image bounding box is not a layout.
    A non-affine access retains a partner ComposedLayout's composition; without
    that composition it raises LoweringError naming the relation. Existing
    refusals, including duplicate output storage, retain their text.
    Only `gmem` and `smem` have physical address placements. Optional `rmem`
    IN_PLACE sharing is a live-byte peak estimate: each result retains its own
    typed AllocTensor. Semantic views, forwards, InsertSlice destinations and
    carry ties share register storage. Placed `smem` views MUST state their dtype
    explicitly at the analyzed byte offset; TensorView does not reinterpret a
    typed pointer's dtype. Loop coordinates in those relations MUST bind to their
    own induction values in the visit context.
    Shared register storage is declared at the outermost definition among its
    users, retaining the HIR tensor type and declared mesh.
    Staged storage retains its placement outside the outermost HIR loop.
    Each return-type leaf becomes a new external output parameter with that
    leaf's type; `umat` becomes `gmem`, retaining its layout. A non-tensor leaf
    raises `LoweringError` naming its position and type. Outputs use the analyzed
    storage without a producer-kind whitelist or a structural HIR storage trace.
    Repeated output storage roots and leaves without a proved address raise
    `LoweringError`. Retained-call arguments and destinations MUST be Tensor or
    TensorView objects over the caller's analyzed storage, except that a returned
    call result uses the caller's output parameter. Along a return address chain,
    objects produced by writes and their VIEW/FORWARD results MUST be views of
    that return leaf's output parameter. Input objects and read-only views taken
    directly from them MUST retain the input parameter. A read-only return emits
    no write or copy; overlap between input and output remains the caller's job.
    If a callee writes into a formal but the caller cannot place its result on
    that actual's storage, lowering MUST retain the `cannot be placed on its output`
    refusal.
    Each external output's address is the caller's responsibility; host entry,
    automatic runtime outputs and the codegen signature add no alias contract.
    `Zeros` and `FullLike` share fill lowering; the scalar constant uses the
    destination dtype and requested value. Each fill is emitted in its producer's
    scope; `Empty` emits no fill. Lowering does not add output or retained-call
    copies. Carry entry and backedge copies execute the analyzed storage decisions.
    Slice starts MUST materialize on-demand Call results as TIR scalar operands,
    including Calls nested in dimension expressions. The materialized binding
    MUST be visible where the slice key is emitted.
    A region result must not reference variables bound inside the region,
    including variables in a view's base or start coordinates.
  - Any construct without a complete lowering MUST raise `LoweringError`; the
    pass never emits a partial `PrimFunction` or silently preserves HIR.

Scheduled lowering MUST retain loops and produce one `PrimFunction` per reached
definition, preserving shared child definitions. Tensor parameters retain their
HIR types, and output parameters follow the return-leaf rule above, including
`ShardLayout`. Default and explicit compact row-major layouts denote the same
physical layout. A returned retained call MUST receive the caller's output
parameter directly when it satisfies the same placement rule as `Empty`.
Call boundaries MAY create matching views of
existing buffers, but MUST NOT insert copies, scheduling, or storage transfers.
The original destination remains the call's result after passing its view.

A callee output MAY be in on-chip memory. An smem return MUST copy into the smem
output buffer supplied by its caller. Other callee returns not bound to that
output buffer MUST fail with `LoweringError` naming the callee and call path and
requiring an explicit HIR `tf.schedule` copy. Lowering MUST NOT repair such an
input or schedule logical HIR automatically. Existing root output copying is
unchanged. Codegen does not support these retained TIR calls and MUST refuse them
with `no codegen handler for Op <type>` when emission reaches one.

### 7.2 Target pass points

The target pass registry in `passes/pass_manager.py` exposes these fixed points:

```python
class PassPoint(Enum):
    BEFORE_LOWER = "before_lower"
    AFTER_LOWER = "after_lower"
    BEFORE_CODEGEN = "before_codegen"


registry: Registry  # keyed by (target class, PassPoint, pass class)


def register_pass(
    target: type[Target], point: PassPoint
) -> Callable[[type[ModulePass]], type[ModulePass]]: ...


def for_target(module: core.Module, point: PassPoint) -> list[ModulePass]: ...
```

- constraints:
  - `passes.for_target` MUST collect PrimFunction targets from each function and
    HIR targets from the corresponding `collect_functions` frame. A frame with
    no Target MUST contribute no target. It MUST order
    targets by registered name and preserve registration order within a target.
  - A pass class MUST be registered at most once per point; a duplicate MUST
    raise `ValueError` naming the Target and pass.
  - Finalize, compile and scaffold MUST run the same pass list
    (`passes.default_pipeline`): `BEFORE_LOWER` target passes,
    `InsertHostEntryPass`, `ConvertHIRToTIR`, `AFTER_LOWER` target passes,
    `BEFORE_CODEGEN` target passes. Scaffold passes
    `passes.Options(lower=False)` here and `codegen.Options(emit="skeleton")` to
    `link_program`.

### 7.3 `InsertHostEntryPass`

```python
class InsertHostEntryPass(ModulePass):
    name: str = "insert-host-entry"
    requires: tuple[str, ...] = ()

    @classmethod
    def create(cls, target: Target, options: passes.Options) -> InsertHostEntryPass: ...
    def run(self, module: core.Module) -> core.Module: ...
```

Core defines the pass with `create(target, options)`; its default `run` returns
its Module unchanged. A Target MAY bind a subclass with
`@bind(TargetClass, InsertHostEntryPass)`; `default_pipeline` obtains it through
`Target.resolve(InsertHostEntryPass, options)`. It runs after `BEFORE_LOWER`
target passes and before `ConvertHIRToTIR`. The target owns the supported device
shapes and launch geometry. A TIR Module receives no entry: one without a CPU
entry is rejected by compile.

- constraints:
  - An existing CPU entry MUST pass through unchanged.
  - A synthesized `Launch` MUST state the analyzed shared-memory peak as
    `dynamic_smem`, and 0 when the Module carries no memory analysis.
  - Normalization MUST preserve each authored function's Target.
  - A synthesized `Launch` MUST carry its settled geometry so downstream
    codegen reads it without deriving it again.

## 8. Directory layout

File layout is implementation-owned. Dispatch registry ownership is defined by
[visitor-registry](./visitor-registry.md).
