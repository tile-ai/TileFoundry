# TileFoundry Spec — tir (`@prim_func` imperative IR)

TIR is the imperative target IR. A `@tilefoundry.prim_func` body parses
into TIR. TIR has no value return; effect-form Ops carry
the work, structural Stmts carry control flow.

- **Container**: `tir.PrimFunction(name, params, body, output_count, target)`.
  `body` is a `Sequential`; the function returns no value.
- **Stmt tree**: function bodies are nested Stmts only. Exprs appear
  inside Stmt fields (e.g. `LetStmt.value`, `For.start`).
- **Effect Ops** (`Copy`, `Fill`, `Cast`, `Mma`, `ReLU`, `RMSNorm`, `Reduce`)
  are value-class Ops registered with `@register_op`; in Stmt
  position they are invoked as `Evaluate(op, args)`
  ([§1.4](#14-evaluate)).
- **Value Ops** (`AllocTensor`, `MemorySpan`, `PtrOf`, `TensorView`)
  are anchored by `LetStmt` so their result `Var` has stable
  identity.
- **No HIR Ops** reach TIR. A program is TIR before the pass pipeline
  ([passes](./passes.md)) reads it.

## 1. TIR Stmt hierarchy

### 1.1 `Stmt`

```python
class Stmt:
    """Provide the abstract base for every TIR statement.

    Attributes:
        loc: attribute; Optional non-semantic debug location.
    """

    loc: str | None = None
```

- constraints:
  - the abstract base of every TIR Stmt subclass; HIR has no `Stmt`.

```mermaid
flowchart TB
    Stmt["<b>Stmt</b>"]
    Sequential["<b>Sequential</b> (Stmt)"]
    CtrlStmts["<b>control-flow stmts</b><br/>For / While / If / MeshScope / LetStmt / Return"]
    PrimFunction["<b>PrimFunction</b> (Stmt)"]
    EvaluateStmt["<b>Evaluate</b> (Stmt)<br/>invokes an Op or function symbol"]

    Stmt --> Sequential
    Stmt --> CtrlStmts
    Stmt --> PrimFunction
    Stmt --> EvaluateStmt
```

### 1.2 Structural Stmts (`tir.stmts`)

```python
class Sequential(Stmt):
    """Contain a statement sequence.

    Attributes:
        body: attribute; Statements in execution order.
    """

    body: tuple[Stmt, ...]

class LetStmt(Stmt):
    """Bind a variable to an expression before a nested body.

    Attributes:
        var: attribute; Bound variable.
        value: attribute; Bound expression.
        body: attribute; Nested statements.
    """

    var: Var
    value: Expr
    body: Sequential

class For(Stmt):
    """Represent a counted loop.

    Attributes:
        induction_var: attribute; Loop variable.
        start: attribute; Initial value.
        stop: attribute; Exclusive bound.
        step: attribute; Increment.
        body: attribute; Loop body.
    """

    induction_var: Var
    start: Expr
    stop: Expr
    step: Expr
    body: Sequential

class While(Stmt):
    """Represent a conditional loop.

    Attributes:
        cond: attribute; Loop condition.
        body: attribute; Loop body.
    """

    cond: Expr
    body: Sequential

class If(Stmt):
    """Represent conditional control flow.

    Attributes:
        cond: attribute; Branch condition.
        then_body: attribute; Taken body.
        else_body: attribute; Untaken body.
    """

    cond: Expr
    then_body: Sequential
    else_body: Sequential

class MeshScope(Stmt):
    """Scope a mesh binding over a body.

    Attributes:
        mesh: attribute; Compile-time mesh.
        binding: attribute; Lexical mesh binding.
        body: attribute; Scoped statements.
    """

    mesh: Mesh
    binding: Var
    body: Sequential

class Return(Stmt):
    """Terminate a value-less primitive function."""
```

- constraints:
  - the structural (control-flow / binding) Stmt family; bodies are `Sequential`.

- `MeshScope.mesh` carries the `Mesh` object; the `binding` `Var`
  scopes the mesh inside `body`.

### 1.3 `PrimFunction`

```python
class PrimFunction(Stmt):
    """Contain one effect-only TIR function.

    Attributes:
        name: attribute; Function name.
        params: attribute; Parameters, with trailing outputs.
        body: attribute; Function body.
        output_count: attribute; Number of trailing output parameters.
        target: attribute; Compilation target for this function.
    """

    name: str
    params: tuple[Var, ...]
    body: Sequential
    output_count: int = 1
    target: Target = field(default_factory=default_target)
```

- constraints:
  - itself a `Stmt`, not a separate top-level node; returns no value.
    `verify_prim_function` enforces the rules below.

`tir.verify.verify_prim_function(fn, *, module_fns=())` enforces:

- **Param homogeneity**. All parameters' layouts MUST be uniformly
  `ShardLayout` or uniformly non-`ShardLayout`; mixing is rejected.
- **Fresh `Var` identity**. The same `Var` object MUST NOT be bound
  by more than one `LetStmt` / `For` / `MeshScope` across the
  function. Parameters seed the bound set.
- **`LetStmt` typing**. `LetStmt.var.type` MUST equal the typeinfer
  of `LetStmt.value`.
- **`AllocTensor` placement**. `Call(AllocTensor, ...)` MAY only
  appear directly as `LetStmt.value`. Nesting it inside any other
  Expr is rejected.
- **`MeshScope` mesh in scope**. Any embedded `ShardLayout` MUST
  reference a mesh on the active TIR `MeshScope` traversal cache or a parameter's
  `ShardLayout.mesh`.
- **`For` bound coordinates**. A `MeshCoord` read by `For.start` / `.stop` /
  `.step` MUST name a literal in-range axis of a mesh bound by an enclosing
  `MeshScope`.
- **`Evaluate.callable`**. When `callable` is a `SymbolRef`
  ([§2.1](#21-symbolref)), module-level resolution MUST find exactly one
  `PrimFunction` of that name in the enclosing `Module`, `args` length
  MUST match the resolved callee's `params`, and the `SymbolRef.type`
  MUST equal the resolved callee's `CallableType`. When `callable` is
  an `Op`, every input operand MUST match its `ParamDef.pattern` and every
  declared `between` relation MUST hold. A per-Op verifier registered via
  `@register_verify_stmt(Op)` MAY impose additional rules that the declaration
  does not express.

### 1.4 `Evaluate`

```python
class Evaluate(Stmt):
    callable: Op | SymbolRef    # an effect-form Op or a SymbolRef callee
    args: tuple[Expr, ...]      # the callable's operands in ParamDef / parameter order
```

- constraints:
  - TIR's single Stmt-position wrapper for a no-result invocation; verify and
    lowering dispatch on `type(callable)`.

The `callable` is one of:

- an effect-form `Op` (e.g. `tir.memory.Copy`, `tir.cuda.nn.Mma`,
  `tir.tensor.Reduce`, `tir.Launch` [§2.3](#23-tir-ops)). `args` are
  the Op's operands in `ParamDef` order. Verification runs its optional
  per-Op verifier, then the declared `between` relations and operand patterns;
  a context-dependent operand pattern MAY resolve itself against the callable
  before matching.
- a `SymbolRef` ([§2.1](#21-symbolref)) — a reference to a callee
  `PrimFunction` in the enclosing `Module`. `args` follow the callee's
  parameter order, the final `output_count` positions binding output
  buffers; the callee is resolved uniquely at module level
  ([§1.3](#13-primfunction)).

Per-Op verify and codegen handlers, when present, are keyed by `Op` type and
receive the Op together with `args`; an `Op` callable carries no result, so its
`Call` form is unit-typed.

The value-producing counterpart is the `Call(Op, args)` Expr
([core-ir.md §2.1](./core-ir.md#21-call)): it has a non-`Unit` result
type and is anchored by `LetStmt`. `Evaluate` is the unit-typed,
Stmt-position form and the only Stmt-position invocation wrapper.

**Effect Op vs. control Stmt.** A callable that is a single
unconditional invocation — an effect `Op` or a function `SymbolRef` —
is expressed as `Evaluate(callable, args)`. A construct that carries
its own control flow stays a first-class `Stmt`, not an `Evaluate`
callable. `Abort` ([§1.7](#17-abort)) is a terminator.

### 1.5 `Sync`

`Sync` is a mesh-scoped barrier. It is an **effect-form op** (`tir.sync.Sync`),
authored `T.sync(m)`, and appears in Stmt position wrapped by `Evaluate`
([§1.4](#14-evaluate)) like any other effect op. The surface is **only**
`T.sync(m)` / `T.sync(m[slice])` — there is no `m.sync()` receiver form.

```python
class Sync(Op):
    """Effect form; mesh-scoped barrier op ``tir.sync.Sync``, authored ``T.sync(m)``.

    Attributes:
        mesh: attribute; the (possibly sliced) mesh the barrier synchronizes.
    """

    mesh: Mesh
```

- constraints:
  - a mesh-scoped barrier; in Stmt position it is wrapped by `Evaluate`. The
    participant set, barrier mapping, and named-barrier id rules are below.

#### `mesh` — the participating threads

- `mesh` is the (possibly sliced) mesh `Sync` synchronizes. `T.sync(m)`
  synchronizes the whole mesh; `T.sync(m[1:3, :])` synchronizes the constant
  sub-mesh selected by the slice.
- A **mesh slice is a compile-time descriptor.** `m[...]` is evaluated at parse
  time via `Mesh.__getitem__` into a sub-`Mesh` whose `layout` is a
  **`ComposedLayout`** recording the participating sub-box (the affine "mesh
  scope" case `image(c) = offset + outer(c)`): the selected per-axis extents
  over the parent strides in `outer`, the slice origin (linear thread index of
  the first participant) in `offset`, identity `inner`. An un-sliced mesh's
  `layout` is a plain `Layout`. A sliced mesh is still a `Mesh`; the slice never
  becomes an IR/SSA value.
- The **participant set** is derived through the existing layout algebra
  (`shard.md`): the participating linear thread indices are `offset +
  outer(coord)` over `outer`'s domain (the plain `layout` at `offset 0` for an
  un-sliced mesh); `base` is the minimum, `count = size(outer)`, and the block
  domain is the product of the topology extents. `classify` / `participation`
  are the single source of truth shared by verify and codegen.
- **Legal-slice verification.** A sliced mesh is accepted only if its
  `ComposedLayout` `layout` reconstructs as a constant slice of an enclosing
  full mesh `e`: same strides, per-axis sub-extents bounded by `e`'s shape, an
  offset that decomposes into in-range per-axis starts, and the **full topology
  tuple** + names equal — the proof rebuilds `e[key]` and compares, so a forged
  slice cannot pass. A full mesh (plain-`Layout` `layout`) is accepted only by
  equality with an enclosing mesh.

#### Supported slices and the barrier mapping

The participant set MUST be a single contiguous thread interval `[base,
base+count)`. Verify MUST reject (never broaden or split):

- a non-contiguous slice (e.g. a lane subset spanning warps);
- a cross-warp range that is not warp-aligned (`base` and `count` not both
  multiples of 32);
- a dynamic / inconsistent / unsupported-topology mesh.

A valid participant set maps to exactly one hardware barrier:

| participant set | barrier |
|---|---|
| whole block, more than one warp (`base==0`, `count==domain`) | `__syncthreads()` |
| whole block that is one warp | `__syncwarp()` |
| a contiguous lane subset within one warp | `__syncwarp(mask)` under a participant predicate |
| a warp-aligned contiguous multi-warp subset | a named `bar.sync <id>, <count>` under a participant predicate |
| the full mesh over the `cta` topology (all CTAs of the grid) | the grid-wide software barrier ([runtime §2.6](./runtime.md#26-cudaops)) |

Codegen MUST guard the `__syncwarp(mask)` and `bar.sync` cases with the
participant predicate `base <= tid < base+count` (`tid =
program_id<thread>()`): a non-participant thread MUST NOT execute the barrier,
and every participant MUST execute the same id and count.

The first four rows synchronize threads **within one block**; their participant
set is the contiguous thread interval above. A mesh whose topologies are all the
`cta` topology instead synchronizes **CTAs across the grid** — `program_id<cta>`
ranges over the launch's blocks — and maps to the grid-wide software barrier.
Only the **full** cta mesh participates: a cta slice (a subset of CTAs) has no
supported barrier and MUST be rejected at verify. The grid barrier's correctness
requires every CTA of the launch to be co-resident; that co-residency is the
launch's occupancy contract, not something the barrier can enforce. The
grid-barrier device helper and its counter protocol are specified in
[runtime §2.6](./runtime.md#26-cudaops).

#### Named-barrier id allocation

A sub-CTA `bar.sync` MUST carry a named-barrier id, allocated implicitly during
codegen, per kernel. Id `0` is reserved for the whole-CTA barrier; sub-CTA syncs
draw ids from `1..15`. Each emitted `bar.sync` MUST take the next free id; a
sync op node emits once, so a loop body reuses its id. A kernel requiring more
than 15 distinct named barriers MUST error; an id MUST NOT be reused across
distinct sync sites.

#### Design rationale

A barrier's scope is a compile-time constant — *which* threads take part — so
the mesh, and any slice of it, is a compile-time descriptor rather than an SSA
value, and the barrier kind is derived from that set by one shared routine so
verify and codegen cannot disagree.

### 1.7 `Abort`

```python
class Abort(Op):
    message: str = ""
```

- constraints:
  - a terminating effect Op on believed-unreachable paths, anchored in Stmt
    position by `Evaluate`.

- The CUDA emitter renders `Abort` as `__trap();` in device contexts
  and `assert(false);` in host contexts so a runtime hit is loud
  rather than silent.
### 1.8 `@intrinsic` — user-defined effect Stmts

```python
# example
@intrinsic
def <name>(<param>: Expr, ...) -> None: ...    # decorated function's signature defines the synthesized Stmt subclass; its body becomes the verifier
```

- constraints:
  - synthesises a Stmt subclass, registers the body as its verifier, and wires
    parser dispatch under the snake-case name; parameters are annotated `Expr` and
    the return annotation is `None`.

## 2. TIR Expr and callable constructs

### 2.1 `SymbolRef`

```python
class SymbolRef(Expr):
    """Name a primitive-function callee.

    Attributes:
        name: attribute; Canonical callee name.
        nested: attribute; Nested path, empty for the flat Module symbol table.
    """

    name: str
    nested: tuple[str, ...] = ()
```

- constraints:
  - a leaf `Expr` naming a callee as an `Evaluate` / `Launch` target; resolution
    is module-level and unique. Per-field rules below.

`SymbolRef` is a leaf `Expr` naming a callee `PrimFunction` as a call
target: the `callable` of an `Evaluate(SymbolRef, args)`
([§1.4](#14-evaluate)) function invocation and `args[0]` of a `Launch`
([§2.3](#23-tir-ops)).

#### `name`
- MUST be the canonical name of a `PrimFunction` in the enclosing
  `Module` ([core-ir.md §1](./core-ir.md#1-module)), exactly as stored
  in `PrimFunction.name`. It MAY be a generated / mangled
  specialization name.

#### `nested`
- MUST be empty: the `Module` holds only top-level functions, so a
  non-empty `nested` is rejected.

#### `type`
- MUST be the resolved callee's IR-level `CallableType`
  ([types §7](./types.md#7-callabletype)): `parameters` are the callee
  `params` types in order; `return_type` is `UnitType`
  ([types §6](./types.md#6-unittype)) — a TIR `PrimFunction` returns
  no value, its outputs are trailing params ([§1.4](#14-evaluate)).
- MUST be set at construction from the callee in hand; a `SymbolRef`
  with a deferred or unresolved type MUST NOT enter constructed IR.
  `Expr` is frozen and verify MUST NOT mutate IR, so verify only
  checks `type` against the resolved callee ([§1.3](#13-primfunction)) —
  it never back-fills.

Resolution is module level: a unique lookup over the `Module`
([core-ir.md §1](./core-ir.md#1-module)) MUST map `name` to exactly
one `PrimFunction`; zero or more than one match is an error.
Specialization variants each carry a distinct canonical
`PrimFunction.name`, so a `SymbolRef` to a variant resolves
unambiguously; the unmangled dispatcher is represented by its prototype and
`variants`, not by a `SymbolRef`. Local typeinfer does not resolve a `SymbolRef`; it
carries its `type` directly.

### 2.2 `ShapeOf`

```python
class ShapeOf(Expr):
    """Produce one runtime tensor extent.

    Attributes:
        param: attribute; Enclosing primitive-function tensor parameter.
        axis: attribute; Tensor axis.
    """

    param: Var
    axis: int
```

- constraints:
  - `type` is a rank-0 `i32` `TensorType` (scalar): one runtime tensor extent.
  - `param` MUST resolve to a parameter `Var` of the enclosing
    `PrimFunction`; `axis` MUST be a valid axis index of `param.type`.
  - the type is the only record that an axis is open. A `PrimFunction` MUST
    NOT carry a second parameter standing for an extent its own types already
    state as a `DimVar`; the convention expands the tensor parameter into a
    pointer plus one extent per open axis
    ([codegen §4.4](./codegen.md#44-signatures)), so no parameter name is
    reserved and a user parameter named `<something>_shape_<n>` is an ordinary
    parameter.

### 2.3 TIR Ops

Value Ops MUST be anchored by `LetStmt.value` — their result `Var` is the only
handle. Effect Ops appear in Stmt position as `Evaluate(op, args)`
([§1.4](#14-evaluate)). Each Op's full contract lives here, in its catalog entry
below; code carries only a one-line purpose docstring
([SPEC-RULES](../SPEC-RULES.md)).

The canonical inspection surface covers every Op in this catalog. Printing and
re-importing a TIR program MUST reach a fixed point: value Ops remain assignment
forms, effect Ops remain statement forms, and enum-valued attributes use their
named enum members with the owning enum imported by the printed program.

- `TensorType.storage` is a `StorageKind` ([types §2](./types.md#2-tensortype)).
  A memory-resident TIR tensor MUST carry a concrete level; the unmaterialized
  `umat` ([types §2](./types.md#2-tensortype)) is an HIR-only value and MUST
  already be materialized to a concrete level in TIR — it never appears there.
- `Reshard` ([hir §1.3](./hir.md#13-op)) does not appear in TIR. A layout change
  is a `TensorView` here and allocates and copies nothing; a storage change is a
  `LetStmt(AllocTensor)` plus `Evaluate(Copy, ...)`.

#### Memory Ops (`tir.memory.*`)

##### AllocTensor
```python
class AllocTensor(Op):
    """Value form; allocate a tensor, anchored by ``LetStmt.value``.

    Attributes:
        tensor_type: attribute; the allocated tensor's result type.
    """

    tensor_type: TensorType
```
- constraints:
  - allocate a tensor; a value Op anchored by `LetStmt.value`.

##### MemorySpan
```python
class MemorySpan(Op):
    """Value form; re-interpret a memory region as a typed tensor.

    Attributes:
        x: input; the memory region being re-interpreted.
    """

    x: Tensor
```
- constraints: []

##### PtrOf
```python
class PtrOf(Op):
    """Value form; take the device address of a tensor.

    Attributes:
        tensor: input; the tensor whose device address is taken.
    """

    tensor: Tensor
```
- constraints:
  - returns `PointerType(tensor.dtype, tensor.storage)`; it does not preserve
    the tensor's shape or layout in the pointer type.

##### TensorView
```python
class TensorView(Op):
    """Value form; construct a logical tensor over a typed pointer.

    Attributes:
        pointer: input; a ``PointerType`` value or an smem byte offset.
        dtype: optional attribute; element type stated for a numeric address.
        storage: optional attribute; storage stated for a numeric address.
        layout: attribute; the view descriptor.
        shape: attribute; logical shape, optionally inherited from ``PtrOf``.
    """

    pointer: object
    dtype: str | None = None
    storage: StorageKind | None = None
    layout: object
    shape: tuple | None = None
```
- constraints:
  - An explicit `shape` is authoritative. If omitted, it is inherited only
    when the input syntax is exactly `T.ptr_of(tensor)`, from that tensor's
    `TensorType.shape`. No other pointer provenance and no layout is inspected
    to guess a shape.
  - An integer input is a byte offset from the kernel's dynamic shared-memory
    base. It MUST state `dtype`, `storage="smem"`, and `shape`; booleans and
    non-integer numeric addresses are invalid.
  - A `PointerType` input MAY restate `dtype` or `storage`, but any stated value
    MUST equal the pointer descriptor.
  - Trailing coordinate inputs are not supported on pointer views.
  - `T.ptr_of` MAY point at an allocated `ShardTensor`; the view rebuilds over
    the engine pointer rather than reusing the existing shard layout.

##### Copy
```python
class Copy(Op):
    """Effect form; byte-equivalent copy between two tensors.

    Attributes:
        src: input; Copy source.
        dst: input; Copy destination.
    """

    src: Tensor
    dst: Tensor
```
- constraints:
  - `src` declares `READ`; `dst` declares `WRITE`.
  - both operands are whole-byte tensors in gmem, smem, or rmem, with equal
    dtype. Their storages MAY be equal; same-storage copy is still a byte move.
  - `scope` optionally states any non-empty run of threads. `rmem_layout` and
    `smem_layout` optionally state the author's landing arrangements.

##### Fill
```python
class Fill(Op):
    """Effect form; broadcast a scalar value into a tensor.

    Attributes:
        tensor: input; destination tensor.
        value: input; rank-0 scalar broadcast into ``tensor``.
    """

    tensor: Tensor
    value: Tensor
```
- constraints:
  - `tensor` declares `WRITE`; `value` declares `READ` and MUST be scalar.
  - a nonconstant value's dtype MUST equal the destination dtype. Constant zero
    is convertible and MAY use the parser's default scalar dtype.
  - `scope` optionally states any non-empty run of threads.

##### Cast
```python
class Cast(Op):
    """Effect form; convert a register tile to another dtype."""

    src: Tensor
    dst: Tensor
    scope: Mesh | None = None
```
- constraints:
  - `src` declares `READ`; `dst` declares `WRITE`; both are rmem tensors.
  - the operands have equal shapes and distinct dtypes.
  - `scope` optionally states any non-empty run of threads.

#### NN Ops (`tir.nn.*`)

##### TiledMma
```python
class TiledMma(Op):
    """Effect form; matrix-multiply-accumulate ``acc += lhs @ rhs``.

    Attributes:
        acc: input; accumulator fragment.
        lhs: input; left-hand operand fragment.
        rhs: input; right-hand operand fragment.
        atom: attribute; required compile-time ``MmaAtom`` declaration.
        scope: attribute; optional warp-aligned thread scope.
    """

    acc: Tensor
    lhs: Tensor
    rhs: Tensor
    atom: MmaAtom
    scope: Mesh | None = None
```
- constraints:
  - matrix-multiply-accumulate `acc += lhs @ rhs`; per-target PTX lowering lives in
    [target](./target.md), the atom calling convention in
    [§2.3](#23-tir-ops).
  - `acc` declares `READ | WRITE`; `lhs` and `rhs` declare `READ`.

##### ReLU
```python
class ReLU(Op):
    """Effect form; pointwise ``max(src, 0)`` written into ``dst``.

    Attributes:
        src: input; input tensor.
        dst: input; destination tensor.
    """

    src: Tensor
    dst: Tensor
```
- constraints: []

##### RMSNorm
```python
class RMSNorm(Op):
    """Effect form; fused RMS normalisation written into ``dst``.

    Attributes:
        src: input; input tensor, reduced over its last axis.
        dst: input; normalised-output tensor.
        weight: input; 1-D scale multiplied onto the normalised output.
        eps: attribute; epsilon applied with rsqrt.
    """

    src: Tensor
    dst: Tensor
    weight: Tensor
    eps: float
```
- constraints: []

#### Tensor Ops (`tir.tensor.*`)

##### Reduce

```python
class Reduce(Op):
    """Effect form; generic axis reduction dispatched by the ``kind`` tag.

    Attributes:
        src: input; reduction source.
        dst: input; reduction destination.
        workspace: input; optional staging buffer sized by lowering.
        axes: attribute; reduced-axis tuple.
        kind: attribute; ``ReduceKind`` tag.
    """

    src: Tensor
    dst: Tensor
    workspace: Tensor | None = None
    axes: tuple
    kind: ReduceKind
```
- constraints:
  - `Reduce` carries no dispatch parameter; runtime selects the strategy.
  - `workspace` is present only when lowering sizes cross-warp staging.
  - All forms lower to the single public runtime entry
    `tilefoundry::ops::reduce<Op, Axes>(src, dst[, workspace])`.
  - Plain and sharded runtime extents/tiers are derived inside the runtime.

##### Dot

`dst = sum(lhs * rhs)` in one statement, and not an `elementwise` followed by a
`Reduce`: materialising the product first would cost a register per element of
the row, which is what makes that pair the wrong spelling here ([runtime
§2.6](./runtime.md#26-cudaops)).

```python
class Dot(Op):
    """Effect form; fused multiply-contract over the axes the meshes contract.

    Attributes:
        lhs: input; left operand.
        rhs: input; right operand.
        dst: input; the destination cell.
        workspace: input; optional shared staging buffer, one slot per warp.
    """

    lhs: Tensor
    rhs: Tensor
    dst: Tensor
    workspace: Tensor | None = None
```
- constraints:
  - **No axes attribute.** `Reduce` names its axes because
    `ops::reduce<Op, Axes>` takes them as a template argument; the axes `Dot`
    contracts are the ones the operands' meshes already contract, so restating
    them at the call site would be a second source for one fact.
  - `lhs` and `rhs` contract over the same number of *local* elements: the fold
    walks one operand's length and indexes the other with it. Their global
    shapes may differ, and in the canonical matrix-vector call they do — a row
    of the matrix is split over the mesh while the vector is broadcast.
  - `dst` is one cell. A contraction leaves a total and every participant leaves
    holding it, so a wider destination is not a wider result but cells the op
    never writes.
  - `workspace` is smem, and is present only in the form that contracts across
    the block: with none the contraction lives inside a warp, with one each warp
    posts a partial into a slot. `Dot` carries no dispatch parameter; the runtime
    selects the tier.
  - A `workspace` requires `lhs` to carry a `ShardLayout`. Both the count of
    warps to fold and the barrier to fold behind come off that mesh, so a
    workspace beside a plain operand asks for a block contraction with nothing
    saying which block.
  - The operands need not agree in dtype: accumulation is f32 whatever is
    loaded.
  - Both forms lower to the single public runtime entry
    `tilefoundry::ops::dot(lhs, rhs, dst[, workspace])`.

#### Generic kind-tagged effect Ops (`tir.arith`)

`Binary` / `Unary` are effect-form Ops that dispatch on a kind enum rather than
per-op classes; they appear as `Evaluate(op, args)`. `BinaryKind` /
`UnaryKind` / `ReduceKind` are compiler-wide tag enums shared across HIR and
TIR; lowering preserves the kind value without re-mapping. Their owning
definitions are [core-ir §4](./core-ir.md#4-shared-operation-kinds).

##### Clamp

```python
class Clamp(Op):
    """Effect form; clamp a source tensor into a destination.

    Attributes:
        min_val: attribute; Lower bound.
        max_val: attribute; Upper bound.
        src: input; Source tensor.
        dst: input; Destination tensor.
    """

    min_val: float
    max_val: float
    src: Tensor
    dst: Tensor
```

- constraints:
  - `src` and `dst` MUST carry the same dtype; the effect writes
    `min(max(src, min_val), max_val)` elementwise into `dst`.

##### Binary
```python
class Binary(Op):
    """Effect form; pointwise binary operation ``dst = lhs <kind> rhs``.

    Attributes:
        lhs: input; left-hand operand.
        rhs: input; right-hand operand.
        dst: input; destination operand.
        kind: attribute; ``BinaryKind`` tag.
    """

    lhs: Tensor
    rhs: Tensor
    dst: Tensor
    kind: BinaryKind
```
- constraints:
  - Lowers to the binary runtime family without per-kind TIR classes.

##### Unary
```python
class Unary(Op):
    """Effect form; pointwise unary operation ``dst = <kind>(src)``.

    Attributes:
        src: input; input operand.
        dst: input; destination operand.
        kind: attribute; ``UnaryKind`` tag, including rsqrt.
    """

    src: Tensor
    dst: Tensor
    kind: UnaryKind
```
- constraints:
  - Lowers to the unary runtime family without per-kind TIR classes.

#### `Launch`

Effect Op for a host-side launch of a device kernel (CPU entry only, no value);
the callee `SymbolRef` and grid/block extents flow through the `Evaluate` args,
the non-grid/block launch config through the Op attributes.

The authored launch-attribute descriptors are owned by
`tilefoundry.ir.tir.launch`:

```python
class CudaLaunchAttr(IntEnum):
    """Authored selector for a CUDA launch attribute."""
    ...


class LaunchAttrs:
    """Carry authored launch attribute selector/value pairs.

    Attributes:
        entries: attribute; Selector/value pairs interpreted by target lowering.
    """

    entries: tuple[tuple[CudaLaunchAttr, object], ...] = ()
```

`CudaLaunchAttr` identifies the CUDA launch-attribute values carried by
`LaunchAttrs.entries`; CUDA target lowering interprets them and rejects
unsupported values. These are authored-IR selectors, not a target registration
API. Launch geometry is derived inside codegen and emitted into the generated
host entry; it is not part of the `Launch` schema and is not carried as runtime
metadata.

```python
class Launch(Op):
    """Effect form; host launch of a device kernel, producing no value.

    Attributes:
        cluster: attribute; optional cluster extents.
        dynamic_smem: attribute; dynamic shared-memory byte count.
        stream: attribute; optional stream handle.
        attrs: attribute; remaining ``LaunchAttrs`` launch configuration.
    """

    cluster: tuple | None = None
    dynamic_smem: int = 0
    stream: object | None = None
    attrs: LaunchAttrs = LaunchAttrs()

# Evaluate(Launch(...), (SymbolRef(callee), grid_x, grid_y, grid_z, block_x, block_y, block_z, *forwarded_args))
```

- constraints:
  - appears only in a CPU (host) entry body; grid/block extents are launch config,
    not kernel parameters. Per-arg / per-attribute rules below.

`Launch` appears only in a CPU (host) entry body, as `Evaluate(Launch(...),
args)` with `args = (SymbolRef(callee), grid_x, grid_y, grid_z, block_x,
block_y, block_z, *forwarded_args)`:

- **callee**: `args[0]` MUST be a `SymbolRef` ([§2.1](#21-symbolref)) resolving to
  a device `PrimFunction` with a CUDA target.
- **grid / block**: `args[1:7]` are the grid then block extents in the fixed
  order `grid_x, grid_y, grid_z, block_x, block_y, block_z`. Each is an `Expr`
  — a `Constant` for a static extent, a `ShapeOf` ([§2.2](#22-shapeof)) for a
  launch-provided (dynamic) one, or a dim-arithmetic `Call` over those. They are
  launch configuration, not kernel parameters: the device observes geometry
  through `gridDim` / `blockIdx` (the codegen `program_dim` / `program_shape`
  accessors), never as arguments.
- **forwarded args**: the remaining `args` bind the callee's parameters, one
  each, in declaration order. An extent the callee's types leave open is not
  among them: it travels with the pointer the convention expands that
  parameter into ([codegen §4.4](./codegen.md#44-signatures)), read off the
  tensor the host holds.
- **attributes**: `cluster`, `dynamic_smem`, `stream`, and `attrs` carry the
  non-grid/block launch configuration. A `cluster` / `stream` / `attrs` value
  the active CUDA target does not support MUST be rejected in target lowering.

#### Declarative MMA atoms and `T.tiled_mma`

An MMA instruction is a target-owned `MmaAtom` declaration. The declaration
class states its authored parameters, required physical scope, target
capability, and the `TensorPattern` read for each `A`, `B`, and `C` role. An
instance binds the parameters for one call; it does not carry a second copy of
concrete fragment layouts that could drift from those patterns.

```python
class MmaAtom:
    namespace: str
    scope: Mesh
    capability: str
    A: TensorPattern | SwitchPattern
    B: TensorPattern | SwitchPattern
    C: TensorPattern | SwitchPattern
    parameters: tuple[ParamDef, ...]
    bindings: dict[str, object]
    mesh: Mesh | None

    def role(self, role: str) -> TensorPattern: ...
    def scope_pattern(self) -> MeshPattern: ...
```

- constraints:
  - `parameters` MUST preserve declaration order. Construction MUST reject an
    unknown binding and a value refused by its `ParamDef.pattern`; an omitted
    parameter MUST take the value implied by earlier bindings or its declared
    default, and otherwise construction MUST fail.
  - `role("A")`, `role("B")`, and `role("C")` MUST resolve the declaration's
    role pattern under the instance bindings. The logical TIR orientation is
    always A `(M,K)`, B `(K,N)`, C `(M,N)`; each role pattern separately states
    the fragment's physical arrangement.
  - `scope_pattern()` MUST require the declaration's exact participant count
    at an aligned offset. `mesh`, when present, binds the atom to one concrete
    frame and MUST match the active frame at verify.
  - `capability` MUST be present in the active CUDA architecture's
    `instruction_capabilities`.

The public declarations are `T.cuda.sm80.Mma()` (BF16 `16x8x16`, F32
accumulator, register A/B/C over one warp) and
`T.cuda.sm90.Wgmma(n=..., form=..., a_major=..., mesh=...)` (BF16
`64 x n x 16` over one warpgroup). `Form` and `Major` live beside `Wgmma`
under `T.cuda.sm90`.

##### Calling convention

Load, compute, and store are separate effect statements under an enclosing
`MeshScope` ([§1.2](#12-structural-stmts-tirstmts)). `T.tiled_mma` has exactly
three input operands plus one required compile-time `atom` attribute:

```python
T.tiled_mma(acc, lhs, rhs, atom=T.cuda.sm80.Mma())
```

- `acc` MUST match `atom.role("C")` and is read-write.
- `lhs` MUST match `atom.role("A")` and is read-only.
- `rhs` MUST match `atom.role("B")` and is read-only.
- A, B, and C shapes MUST be `(M,K)`, `(K,N)`, and `(M,N)` respectively;
  operands MUST use the atom's declared dtypes and fragment arrangements.
- The active physical mesh MUST satisfy `atom.scope_pattern()`. If the atom
  carries `mesh=...`, its affine offset and ordered lanes MUST equal the active
  frame.

There is one TIR MMA op: `T.tiled_mma`. There is no optional-atom or bare-MMA
path. Per-target emission dispatches on the atom ([target](./target.md)); the
current CUDA emitter accepts the SM80 declaration and rejects WGMMA because
this stage provides no WGMMA runtime emitter.

#### Async copy Ops (`tir.async.*`)

Non-blocking `cp.async` gmem→smem staging for warp-specialized pipelines: a
producer issues copies, groups them, and a consumer waits on the group queue.

##### CopyAsync

```python
ASYNC_WIDTHS = (4, 8, 16)


class CopyAsync(Op):
    """Effect form; async gmem→smem copy, non-blocking.

    Attributes:
        src: input; gmem staging source.
        dst: input; smem staging destination.
        smem_layout: attribute; optional landing arrangement.
    """

    src: Tensor
    dst: Tensor
    smem_layout: Layout | None = None
```
- constraints:
  - Lowers to `tilefoundry::ops::copy_async(src, dst)`.
  - `src` is gmem and `dst` is smem, with the same dtype. Each layout MUST
    admit the same width from `ASYNC_WIDTHS` and MUST walk the same tile
    mode at step 1. For a `ShardLayout`, vector width is read from the whole
    tile arrangement; its other strides ensure every participant's start is
    aligned. Two split layouts compare their tile modes only when their mesh
    and shard attrs are identical.
  - A later read of `dst` is ordered by `CpAsyncCommit` followed by
    `CpAsyncWait`.

##### CpAsyncCommit

```python
class CpAsyncCommit(Op):
    """Effect form; close the current in-flight async-copy group."""
```
- constraints:
  - Later `CpAsyncWait` counts committed groups.

##### CpAsyncWait

```python
class CpAsyncWait(Op):
    """Effect form; wait until at most ``n`` committed groups remain in flight.

    Attributes:
        n: attribute; most-recent committed groups allowed to remain in flight.
    """

    n: int = 0
```
- constraints:
  - `n` is a non-negative compile-time count.
  - `n = 0` drains every outstanding committed group.

##### CopyAsyncBulk

A staging copy whose completion lands on an mbarrier, and not a tier of
`CopyAsync`: there every thread issues its own load and a commit closes the
group, so the thread that issues is the thread that waits; here a consumer can
wait for a tile it did not fetch.

Which instruction carries it is the runtime's choice from the operand shard
layouts, not something this op names: a contiguous run takes `cp.async.bulk`,
anything else takes an element path ([runtime
§2.6](./runtime.md#26-cudaops)).
Carrying that on the op would be codegen selecting a tier, which
[§2.3](#23-tir-ops) forbids.

The tensor forms (`cp.async.bulk.tensor.Nd`) take a host-encoded `TensorMap` in
place of a size, which is a different operand list rather than a different tier,
and are outside this op.

```python
class CopyAsyncBulk(Op):
    """Effect form; gmem→smem staging copy completing on an mbarrier.

    Attributes:
        src: input; gmem source tile.
        dst: input; smem destination tile.
        barrier: input; smem mbarrier the completion lands on.
    """

    src: Tensor
    dst: Tensor
    barrier: Tensor
```
- constraints:
  - `src` is gmem, `dst` is smem, `barrier` is smem.
  - `src` and `dst` agree in dtype and shape; the copy moves bytes and does not
    convert them.
  - Nothing blocks: the copy may still be in flight when the issuing thread
    reaches the next statement.
  - Consumers wait with `MBarrierWaitParity`. The arrival that declares the
    transferred bytes is the implementation's, issued on the same instruction as
    the copy; a caller pairing this with its own `MBarrierArriveExpectTx` would
    be declaring a count the op already knows.
  - Lowers to `tilefoundry::ops::copy_async_bulk(src, dst, bar)`
    ([runtime §2.6](./runtime.md#26-cudaops)). `barrier` is a tensor here
    because that is what TIR names a piece of shared memory with, and a word to
    the runtime, so the emitted call hands over the word's own address.

##### CopyAsyncTensor

`T.copy_async_tensor` declares the SM90 tensor-map form separately from
`CopyAsyncBulk`: its operands describe a tensor-map global layout and the box
landed in shared memory, rather than carrying an explicit mbarrier operand.

```python
class CopyAsyncTensor(Op):
    src: Tensor
    dst: Tensor
    smem_layout: Layout | None = None
    scope: Mesh | None = None
```

- constraints:
  - Exactly one end is gmem and one is smem; dtype and logical shape agree.
  - The global end is a static tensor-map layout of at most five dimensions,
    with one contiguous mode and every other byte stride a multiple of 16.
    The shared end is an at-most-five-dimensional box, each extent at most
    256, optionally using a 32-, 64-, or 128-byte TMA swizzle.
  - Both ends walk the same tile modes. The issuing scope is one aligned warp.
  - The declaration requires the target's `tma` capability. CUDA codegen MUST
    reject it until host-encoded tensor-map construction exists; this stage
    does not silently lower it to another copy instruction.

##### LdMatrix

```python
class LdMatrix(Op):
    src: Tensor
    dst: Tensor
    scope: Mesh | None = None
```

- constraints:
  - `src` is a shared-memory `(16, 16)` bf16 tile and `dst` is exactly the
    register A fragment declared by `T.cuda.sm80.Mma()`; their dtype and shape
    agree under the ordinary `Copy` verifier.
  - One canonical SM80 warp issues the operation. It requires the target's
    `tensor_core` capability and lowers to `tilefoundry::ops::ldmatrix`.
  - The destination layout is the atom's declaration, not a caller-selectable
    `rmem_layout` attribute.

#### Barrier object Ops (`tir.sync.mbarrier_*`)

A Hopper mbarrier is a 64-bit shared-memory word carrying an arrival count, a
transaction-byte count and a phase parity. It is not `Sync`
([§1.5](#15-sync)): `Sync` is a whole-mesh rendezvous every participant reaches,
while these let a producer signal completion of work the consumer did not
perform — which is what an asynchronous copy needs, since the thread that issues
one is not the thread that waits for it.

**Each lowers to its instruction, not to a runtime entry, and the runtime
publishes no `ops::` entry for any of them:** an mbarrier is a shared-memory
word, so nothing here reads a `ShardLayout` and none of it is an op
([runtime §2.6](./runtime.md#26-cudaops)). Each entry below names the
`mbarrier.*` instruction its emitter writes at the call site, together with the
generic-to-shared conversion the instruction takes — they name `.shared::cta`
explicitly rather than leaving the assembler to redo that window conversion on
every use.

The group is what a `CopyAsyncBulk` ring needs and no more: arm the word, arrive on it
declaring bytes, wait on its phase, release it. A bare `mbarrier.arrive` is
absent because `ops::copy_async_bulk` issues its own for the strided tier, and a bare
`mbarrier.expect_tx` because nothing pairs with it.

##### MBarrierInit

```python
class MBarrierInit(Op):
    """Effect form; arm a barrier for a fixed number of arrivals.

    Attributes:
        barrier: input; smem barrier object.
        arrive_count: attribute; arrivals that complete one phase.
    """

    barrier: Tensor
    arrive_count: int
```
- constraints:
  - `barrier` is smem; the instructions take a shared-window address.
  - `arrive_count` is a positive compile-time count. A phase needing zero
    arrivals is complete before anything is produced, which makes every
    consumer's wait a no-op.
  - One thread initialises, and a `Sync` covering every thread that will use the
    barrier separates this from the first arrival or wait.
  - Lowers to `mbarrier.init.shared::cta.b64`, written at the call site with
    `arrive_count` as an inline operand: the runtime publishes no entry for it
    ([runtime §2.6](./runtime.md#26-cudaops)).

##### MBarrierArriveExpectTx

```python
class MBarrierArriveExpectTx(Op):
    """Effect form; arrive and declare asynchronous bytes in one instruction.

    Attributes:
        barrier: input; smem barrier object.
        tx_bytes: attribute; bytes the paired copy delivers to this phase.
    """

    barrier: Tensor
    tx_bytes: int
```
- constraints:
  - `barrier` is smem and `tx_bytes` is a positive compile-time count.
  - The phase completes when both the arrivals and the byte count are satisfied,
    so one wait covers a copy the waiting thread did not issue.
  - `tx_bytes` MUST equal the bytes the paired copy delivers. A phase expecting a
    different count never completes, and that failure presents as a hang rather
    than as a wrong value.
  - This is not paired with a `CopyAsyncBulk`, which declares its own bytes on the
    instruction that issues the copy. It belongs to a producer issuing one
    itself.
  - Lowers to `mbarrier.arrive.expect_tx.shared::cta.b64`, with the arrival
    token discarded: consumers wait on the phase parity, not on a token handed
    between threads. The runtime publishes no entry for it; `ops::copy_async_bulk`
    writes its own for the bulk tier.

##### MBarrierWaitParity

```python
class MBarrierWaitParity(Op):
    """Effect form; block until the barrier's phase parity reaches a value.

    Attributes:
        barrier: input; smem barrier object.
        phase: input; the parity waited for.
    """

    barrier: Tensor
    phase: Tensor
```
- constraints:
  - `barrier` is smem.
  - The parity alternates `0, 1, 0, ...` across successive completions, which is
    what lets a fixed ring of barriers serve a pipeline of any length: stage `t`
    of a ring of `n` waits on parity `(t // n) & 1`.
  - Lowers to a single `mbarrier.try_wait.parity.shared::cta.b64` under a **C++**
    loop, not a PTX one: a label inside inline asm is emitted once per
    instantiation and collides as soon as two of them land in one translation
    unit, and `try_wait` already parks the warp in hardware for a bounded
    interval, so the loop is not a busy spin on the issue pipe. The runtime
    publishes no entry for it.
  - `phase` is a value and not an attribute — the parity is a function of the
    stage index — and lowers through the same scalar-expression renderer as
    `If.cond`, so a loop induction variable or a constant reaches the
    instruction and anything else is refused at codegen.

##### MBarrierInvalidate

```python
class MBarrierInvalidate(Op):
    """Effect form; release the barrier's shared-memory word.

    Attributes:
        barrier: input; smem barrier object.
    """

    barrier: Tensor
```
- constraints:
  - `barrier` is smem.
  - Lowers to `mbarrier.inval.shared::cta.b64`, written at the call site: the
    runtime publishes no entry for it.
