# TileFoundry Spec — semantic-analysis (per-Op semantic derivation)

This spec owns the semantic contracts that derive types, access relations, and
shard layouts over the IR. Each service is a registry-backed derived visitor.
[visitor-registry](./visitor-registry.md) owns registration and dispatch,
including the carrier of the access relation
([§4.1](./visitor-registry.md#41-access-relation-service--access_relation)).
This file owns the requirements of each service and the semantic rules that it
enforces.

## 1. Type propagation

Each Op registers its type inference through
`typeinfer.register(<OpClass>)` ([visitor-registry §4](./visitor-registry.md#4-instance-1--typeinfer)).
A handler receives the op and a typeinfer context. It derives the output `Type`.
It reports violations through `ctx.error`. A `core.Function` call composes these
per-op rules under elaboration ([hir §1.1](./hir.md#11-function)).
Elaboration reconstructs the callee body. Each node re-derives its type under
the actual argument types of the call. So a relation-derived rule needs no
function-boundary case of its own.

- constraints:
  - A function boundary MUST NOT complete or reject a legal `Partial`.
  - A function return MAY carry a `Partial(reduction)` in a `TensorType`.
    A tuple return MAY carry it in any nested tensor field.
  - Call elaboration MUST preserve the `ShardLayout` mesh-axis position and
    reduction on the actual value. An explicit `Reshard` or allreduce completes
    it.

### 1.1 Relation-derived type behavior

An op's typeinfer MAY derive the output type from the Op's access relations
([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation))
instead of a hand-written rule. Every boundary states an access map from the
Op's own iteration space to the index space of that tensor. The relations carry
**no tensor shape**. The output shape is typeinfer-side data. It comes from the
op's shape rule or from the extents that the output relation reaches.

- A domain dim that appears in an input access map but **not** in the
  output access map is a **reduction** dim.
- A tensor axis whose access maps to a constant is a **broadcast** axis.
- A symbolic size is an isl parameter of the domain. The input types give the
  rank of the relation.

[§3.2](#32-relation-driven-shard-propagation) describes the shard consequences:
`Split`, `Broadcast` and `Partial` propagation and the reduction effect.

### 1.2 Domain construction and output shape derivation

The relation service holds every boundary's image rank to the Type of the value it
describes ([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation)).

## 2. Access relation analysis

The access relation is the boundary model that relation-derived type behavior,
shard propagation, dependence and movement share. For each boundary, it is an
affine access map from the Op's own iteration space to the index space of a
tensor. The section
[visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation)
defines its carrier `access.Relation`, the tuple that one Op states, and the
registry.
Type and shard rules read only the affine structure of the access
maps, which is the domain dim that each axis uses. They never read the domain
bounds. So they are size-agnostic and identical for static and dynamic shapes.

An effect-form TIR instruction uses the same tuple, although its call has
the void result `TupleType.Void`:

- One relation per input `ParamDef` comes first, in parameter order. This holds
  whether that operand is read, written or both.
- The relation after those is the one boundary of the void result, as for any
  Op.
- The `effect` of each parameter says which operands are read and which are
  written, not its position. The argument prefix is not the set of reads.

An HIR consumer such as [`tf.schedule`](./schedule.md) may use the relations
of written operands to derive its own value result. Instructions selected by
such a consumer MUST register their relation explicitly. They have no fallback
relation.

### 2.1 IR to isl conversion

Every analysis service converts IR values to isl through the functions that
[types §11](./types.md#11-isl-interoperability) owns. Every boundary carries
what its parameters stand for in the `values` of its `access.Relation`
([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation)).
The service side and the boundary side each use an `IslParamValues`
dictionary from isl parameter name to IR value.

- constraints:
  - A service MUST compose converted values as isl objects. It MUST NOT render
    an isl object back to text and splice that text into a new relation.
  - A loop domain MUST treat each enclosing induction variable as a dimension
    of the space, not as a parameter. It MUST treat each capture of such a
    variable the same way. Each other leaf of a loop bound MUST be a parameter
    with a stated range. A loop bound with an unbounded parameter is refused.
  - A parameter of an access relation that an enclosing scope already names
    MUST keep that name. A capture of that value MUST be the same parameter.
    So one value is one parameter across the scope.
  - A reader MUST find what a parameter stands for in `values`, not in its
    name. Boundaries of one Op MUST share a parameter only when they bind the
    same object. `iterating` establishes this condition before the Op's
    boundaries are combined.

## 3. Shard propagation

### 3.1 Logical shape to layout domain

- `TensorType.shape` is the logical shape.
- `layout` has its own domain shape.
- The interpretation is a canonical regroup. It linearizes along the row-major
  order of the logical shape. Then it reinterprets along the row-major order of
  the layout domain. Strides describe physical indices only. Strides MUST NOT
  participate in logical-coordinate correspondence. A nested domain associates
  its top-level modes with the corresponding logical axes only when two
  conditions hold. The number of top-level modes equals the logical rank. The
  size of each mode equals its axis extent. Otherwise, the flattened modes
  regroup.
- Axis grouping uses `compatible_profile(layout.shape, shape)` to construct a
  profile without splitting or coalescing authored modes. `unflatten` reads the
  layout against that profile. `mode_axes` names the logical axis of each flat
  mode.
- `layout_to_isl_map` ([types §11](./types.md#11-isl-interoperability))
  implements this regroup for a `ShardLayout`. A logical axis whose extent is
  the product of consecutive flattened modes maps onto them. So a
  symbolic extent stays representable. Any other regroup MUST be rejected where
  the type is stated. This includes a mode that merges logical axes. Type
  inference MUST check function parameters with the same axis-group rule. It
  MUST check every Call result with that rule. It MUST report the declaring
  source location. Canonical UMAT shape metadata uses the same compact
  `Layout(shape)` as other tensor values.
- A reader with no topology addresses logical coordinates. A reader at a
  given topology addresses the coords one unit of that topology holds. The
  reader splits a coord cut by mesh axes at that topology or coarser into digits. The
  digits follow the cutting mesh axes outermost first in mesh-axis order, then the
  residual. Each cutting mesh axis digit is that unit's `MeshCoord` on that mesh axis.
  The residual is the local coord within the unit and is not a `MeshCoord`.
- A mesh coord is an isl parameter while the layout map is built. The
  access relation service fixes it to the requested unit. The default is 0 on
  every mesh axis unless stated. The service removes it before any boundary is
  counted.

### 3.2 Relation-driven shard propagation

This section applies when the output `ShardLayout` of an op is derived from a
forward access relation ([§1.1](#11-relation-derived-type-behavior)). A single
rule determines the output `ShardAttr`s from the input shards and the access
maps of the relation. All ops share this rule.

```python
# types.partial_reductions_by_axis (ir/types/shard_layout.py)
def partial_reductions_by_axis(layout: object) -> tuple[str | None, ...]: ...

# typeinfer.derive_output_shard_layout (interfaces/typeinfer/shard_propagate.py)
def derive_output_shard_layout(
    input_types: tuple,
    relations,
    output_shape: tuple,
    *,
    partial_reduction_dims: frozenset[int] = frozenset(),
    complete_reduction_dims: frozenset[int] = frozenset(),
    reduction: str = "sum",
    fresh_strides: bool = False,
    ctx=None,
    storage=None,
): ...
```

- constraints:
  - `derive_output_shard_layout` reads only the input types, the requested
    output shape, the explicit reduction/stride controls, and
    `access.Relation.affine`, which is the Op-written mapping before
    iteration-domain intersection. It MUST NOT read the bounded
    `access.Relation.relation` or an already-derived output type.
  - `partial_reductions_by_axis` returns one entry per mesh axis. The entry is
    the carried reduction name for `Partial` and `None` for any other attr. For
    a non-sharded layout, it returns an empty tuple.
  - The service first projects every input onto the executing topologies, that
    is, each topology that any input distributes. So mesh axes keep one index
    across inputs. With no real input sharding, the service returns `None`.
    A result Varying on every mesh axis is `None` too, the plain layout, only
    when its mesh covers every unit; on a sliced mesh it keeps that mesh
    ([shard §6.1](./shard.md#61-shard-attr-semantics)).
    Inputs with real sharding on different meshes MUST fail (`ValueError`).
  - When every propagated physical layout agrees, the result keeps it.
    Otherwise the result is `ShardLayout.from_logical` over the output shape.
    `carrier_layout` returns `None` for anything short of an exact coordinate
    projection.
  - Both sets turn a reduced split into `Partial(reduction)`. Each unit holds
    its part. The whole value is an explicit `Reshard` away. A reduction
    never turns a split into `Broadcast`, which would hide that communication.
    `complete_reduction_dims` additionally collapses the reduced
    modes to extent one. The sets MUST NOT overlap.
  - `fresh_strides=True` uses `ShardLayout.from_modes` for fresh compact
    row-major strides. Statically unit leaves have stride zero. Symbolic strides
    remain symbolic.
    Split modes owned per instance by `ctx.target` for the result `storage` also
    have stride zero. Compacting the remaining
    modes drops only each Split's mesh factor and retains its local residual
    (extent one for a fully factored Split). The set comes from the target memory
    owner and every coarser topology. With no memory hierarchy it is empty.
    Otherwise a compatible propagated physical layout is preserved when possible.
  - `ShardLayout.attrs` is indexed by mesh axis. A `Partial(reduction)` is a
    value state at that exact index, not an unordered collection of reductions
    and not a mode. Every propagation decision MUST retain that
    index and its reduction independently of every other mesh axis.
  - The op declares a reduction dim in `partial_reduction_dims` or
    `complete_reduction_dims`. A reduction dim is a domain dim absent from the
    output access map. Both sets yield `Partial(reduction)` on a split mesh
    axis (step 3).
  - Propagation applies per input mesh axis:
    1. `Split(k)` maps mode `k` to the input's logical tensor axis and
       then to a domain dim through the input access map.
    2. If that domain dim appears in the output access map, the output carries
       `Split` on the output mode to which the domain dim maps.
    3. If that domain dim is reduced, the output mesh axis becomes
       `Partial(reduction)`. This `Partial` is a value state on that mesh axis
       and carries no mode.
    4. A `Partial(reduction)` input propagates on the same mesh axis only when
       the op's math is proven to commute with `reduction`:
       `op(reduction(x0..xn)) == reduction(op(x0)..op(xn))`. This service MUST
       NOT make that mathematical judgment. Each op's typeinfer rule owns it.
       On each mesh axis, two Partial inputs MUST be compatible with that op's
       rule. States on different mesh axes MUST NOT be compared as an unordered
       set. Only an explicit reduction/allreduce resolves `Partial` to
       `Broadcast`.
    5. A `Broadcast` input mesh axis contributes no `Split`.
    6. Two inputs giving one mesh axis different output states is an error.
  - A `Partial` MUST NOT be silently eliminated or carried through a
    non-commuting ordinary op.
  - An ordinary multi-input op MUST inspect every tensor input for a `Partial`.
    Typeinfer MUST reject a secondary input whose axis-preserving state the
    result type cannot represent. It MUST name the `Reshard` remedy.
  - An op that reads more than one operand's shard attrs passes operand types
    from `shard_read_types`, which reads a literal operand as `Broadcast`
    ([shard §6.1](./shard.md#61-shard-attr-semantics)). A single-operand reader
    needs no call: no other operand gives a mesh.
  - A fully-`Broadcast` input `ShardLayout` is replicated. It carries no real
    sharding. It contributes no `Split`/`Partial`. It does not pin a mesh. When
    no input carries real sharding, the output carries none.
  - Only a zero-offset, unit-coefficient access of one domain dim is a
    projection for shared ownership propagation. An input `Split` MUST fail
    closed (`ValueError`) in two cases. The first case is an access of any other
    form, including a nonzero affine translation. The second case is an
    output-surviving dim that is reachable only through such an access. Without
    domain bounds, the service has no owner-boundary alignment proof for a
    translated access.
  - `Split(axis)` indexes an output mode, not a logical tensor axis.

An op whose result type is one operand's type reads the other tensor operands
with one shared rule.

```python
# typeinfer.combine_operand_attrs (interfaces/typeinfer/shard_propagate.py)
def combine_operand_attrs(result: TensorType, others: tuple[TensorType, ...], ctx) -> TensorType: ...

# typeinfer.shard_read_types (interfaces/typeinfer/shard_propagate.py)
def shard_read_types(ctx, operands: tuple[Expr, ...]) -> tuple[Type, ...]: ...

# typeinfer.uniform (interfaces/typeinfer/shard_propagate.py)
def uniform(node: Expr | int, ctx, scope=None) -> bool: ...
```

- constraints:
  - Each operand is read on the result's shard mesh with the target's storage
    owners ([shard §6.1](./shard.md#61-shard-attr-semantics)). A plain operand
    reads as `Varying` on a topology that owns its storage per unit; a literal
    operand, and a value the same on every unit, reads as `Broadcast` (through
    `shard_read_types`).
  - On each mesh axis whose topology owns the result's storage per unit: a
    `Broadcast` result with a `Varying` operand becomes `Varying`. Where the
    result storage has one shared copy, `Broadcast` and `Varying` read alike and
    the result keeps its attr. A `Partial` result with a `Varying` operand MUST fail
    (`ValueError`). Every other pair leaves the result unchanged; the op's own
    checks own `Split` and `Partial`.
  - A result that this rule makes `Varying` on every mesh axis is the plain
    `Layout` over its shape when its mesh covers every unit; on a sliced mesh it
    keeps the mesh. A result the rule does not change, plain or not, is
    returned as written.
  - The mesh and the address layout of the result do not change.
  - `shard_read_types` returns each operand's type. A plain operand that
    `uniform` admits is given an all-`Broadcast` `ShardLayout` on the mesh of
    the first operand that has one, keeping its address layout. Without such
    an operand, every type is returned as stored.
  - `uniform` admits an integer, a literal, a tensor that reads as `Broadcast`
    on every mesh axis, a loop index whose start, extent and step it admits, a
    region capture whose argument it admits, and an `Op` call other than
    `MeshCoord` whose operands it admits. It refuses a `DimVar`, a carried
    value, a plain parameter, and a function call whose result is plain: the
    callee can read `MeshCoord`. It decides each value once per inference run.

### 3.3 Output storage and mesh/layout compatibility

- constraints:
  - A symmetric multi-input op (`Binary`, `MatMul`, `Concat`, `Stack`)
    resolves output storage with `resolve_anchor_storage` over the concrete
    residency among its operands ([types §2](./types.md#2-tensortype)). The
    result does not depend on operand order. The op uses no memory-level
    lattice.
  - An unmaterialized operand (`storage=umat`), including host shape metadata,
    abstains. One concrete storage, or several that agree, is the output
    storage. With none, the output is `umat`.
  - Several concrete operands that disagree cause typeinfer to `ctx.error`,
    unless the op defines its own destination/mixed-storage resolution. Typeinfer
    uses no operand-order tie-break.

`TensorType.layout` carries the mesh and layout of a tensor value.
`ShardLayout.mesh` names the mesh instance. The IR places no scope-based
restriction on values from different meshes that coexist. The registered
typeinfer of each op **owns** the layout and mesh compatibility that its
operands need, and its result layout. Nothing outside imposes a uniform cross-op
rule. `Reshard` is the explicit op that changes the layout and mesh of a value.
For example, `IndexSelect` owns whether an indexed access is a foldable
one-element slice or a materializing data-dependent selection.
