# TileFoundry Spec — semantic-analysis (per-Op semantic derivation)

This spec owns the semantic contracts that derive types, access relations, and
shard layouts over the IR. Each service is a registry-backed derived visitor:
registration and dispatch are owned by [visitor-registry](./visitor-registry.md),
including the access relation's carrier
([§4.1](./visitor-registry.md#41-access-relation-service--access_relation)); this
file owns each service's requirements and the semantic rules it enforces.

## 1. Type propagation

Type inference is registered per Op through
`typeinfer.register(<OpClass>)` ([visitor-registry §4](./visitor-registry.md#4-instance-1--typeinfer)).
A handler receives the op and a typeinfer context, derives the output `Type`, and
reports violations through `ctx.error`. A `core.Function` call composes these
per-op rules under elaboration ([hir §1.1](./hir.md#11-function)): the callee
body is reconstructed and each node re-derives under the call's actual argument
types, so a relation-derived rule needs no function-boundary case of its own.

- constraints:
  - A function boundary MUST NOT complete or reject a legal `Partial`.
  - A function return MAY carry a `Partial(reduction)` in a `TensorType`, and a
    tuple return MAY carry it in any nested tensor field.
  - Call elaboration MUST preserve the `ShardLayout` mesh-axis position and
    reduction on the actual value; completion remains the responsibility of an
    explicit `Reshard` or allreduce.

### 1.1 Relation-derived type behavior

An op's typeinfer MAY derive the output type from the Op's access relations
([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation))
rather than from a hand-written rule. Every boundary states an access map from
the Op's own iteration space to that tensor's index space. The relations carry
**no tensor shape**: the output shape is typeinfer-side data, derived from the
op's shape rule or from the extents the output relation reaches.

- A domain dim that appears in an input access map but **not** in the
  output access map is a **reduction** dim.
- A tensor axis whose access maps to a constant is a **broadcast** axis.
- A symbolic size is an isl parameter of the domain; the relation's rank is
  read from the input types.

The shard consequences ( `Split` / `Broadcast` / `Partial` propagation and the
reduction effect) are in [§3.2](#32-relation-driven-shard-propagation).

### 1.2 Domain construction and output shape derivation

The relation service holds every boundary's image rank to the Type of the value it
describes ([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation)).

## 2. Access relation analysis

The access relation is the boundary model shared by relation-derived type
behavior, shard propagation, dependence and movement: per boundary, an affine
access map from the Op's own iteration space to a tensor's index space. Its
carrier `access.Relation`, the tuple one Op states, and the registry are defined
in [visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation).
Type and shard rules read only the access maps' affine structure (which domain
dim each axis uses), never the domain bounds, so they are size-agnostic and
identical for static and dynamic shapes.

An effect-form TIR instruction uses the same tuple even though its call is
unit-typed:

- One relation per input `ParamDef`, in parameter order, comes first, whether
  that operand is read, written or both.
- Its relations after those are its Unit result's, as for any Op.
- Which operands are read and which are written is each parameter's `effect`,
  not its position: the argument prefix is not the set of reads.

An HIR consumer such as [`tf.schedule`](./schedule.md) may use the relations
of written operands to derive its own value result. Instructions selected by
such a consumer MUST register their relation explicitly; there is no fallback
relation.

### 2.1 IR to isl conversion

Every analysis service converts IR values to isl through the functions that
[types §11](./types.md#11-isl-interoperability) owns, and every boundary carries
what its parameters stand for in the `values` of its `access.Relation`
([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation)),
both as one `IslParamValues` dictionary from isl parameter name to IR value.

- constraints:
  - A service MUST compose converted values as isl objects. It MUST NOT render
    an isl object back to text and splice that text into a new relation.
  - A loop domain MUST treat each enclosing induction variable, and each
    capture of one, as a dimension of the space, not as a parameter. Each other
    leaf of a loop bound MUST be a parameter with a stated range; a loop bound
    with an unbounded parameter is refused.
  - A parameter of an access relation that an enclosing scope already names
    MUST keep that name, and a capture of that value MUST be the same
    parameter, so one value is one parameter across the scope.
  - A reader MUST find what a parameter stands for in `values`, not in its
    name. Boundaries of one Op MUST share a parameter only when they bind the
    same object, which `iterating` makes so before the Op's boundaries are
    combined.

## 3. Shard propagation

### 3.1 Logical shape to layout domain

- `TensorType.shape` is the logical shape.
- `layout` has its own domain shape.
- The interpretation is canonical regroup: linearize along the logical shape's
  row-major order, then reinterpret along the layout domain's row-major order. Strides describe physical positions only
  and MUST NOT participate in logical-coordinate correspondence. Nested
  domains associate their top-level modes with the corresponding logical axes
  only when the number of top-level modes equals the logical rank and each
  mode's size equals its axis extent; otherwise, the flattened modes regroup.
- Axis grouping uses `compatible_profile(layout.shape, shape)` to construct a
  profile without splitting or coalescing authored modes. `unflatten` reads the
  layout against that profile; `mode_axes` names each flat mode's logical axis.
- `layout_to_isl_map` ([types §11](./types.md#11-isl-interoperability))
  implements this regroup for a `ShardLayout`. A logical axis whose extent is
  the product of consecutive flattened layout positions maps onto them, so a
  symbolic extent stays representable. Any other regroup, including a mode
  merging logical axes, MUST be rejected where the type is stated. Type
  inference MUST check function parameters and every Call result with the same
  axis-group rule and report the declaring source location. Canonical
  UMAT shape metadata uses the same compact `Layout(shape)` as other tensor values.
- A reader with no topology level addresses logical coordinates. A reader at a
  named topology level addresses the positions one unit of that level holds:
  a position cut by mesh axes at that level or coarser is split into digits,
  the cutting axes outermost first in mesh-axis order and then the residual,
  and each such digit is that unit's mesh coordinate.
- A mesh coordinate is an isl parameter while the placement is built. The
  access relation service fixes it to the requested unit, 0 on every mesh axis
  unless stated, and removes it before any boundary is counted.

### 3.2 Relation-driven shard propagation

When an op's output `ShardLayout` is derived from a forward access
relation ([§1.1](#11-relation-derived-type-behavior)), the
output `ShardAttr`s are determined from the input shards and the
relation's access maps by a single rule, shared across ops.

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
  - `derive_output_shard_layout` reads only input types, `access.Relation.affine`
    (the Op-written mapping before iteration-domain intersection), the requested
    output shape, and the explicit reduction/stride controls. It MUST NOT read
    the bounded `access.Relation.relation` or an already-derived output type.
  - `partial_reductions_by_axis` returns one entry per mesh axis: the carried
    reduction name for `Partial`, `None` for any other attr, and an empty tuple
    for a non-sharded layout.
  - Every input is first projected onto the participating topology levels (each
    level any input distributes), so mesh axes keep one index across inputs.
    With no real input sharding it returns `None`. Inputs with real sharding on
    different meshes MUST fail (`ValueError`).
  - When every propagated physical layout agrees (`carrier_layout`, which
    returns `None` for anything short of an exact coordinate projection), the
    result keeps it; otherwise the result is `ShardLayout.from_logical` over the
    output shape.
  - Both sets turn a reduced split into `Partial(reduction)`: each unit holds
    its part, and the whole value is an explicit `Reshard` away. A reduction
    never turns a split into `Broadcast`, which would hide that communication.
    `complete_reduction_dims` additionally collapses the reduced layout
    positions to extent one. The sets MUST NOT overlap.
  - `fresh_strides=True` uses `ShardLayout.from_modes` for fresh compact row-major strides;
    statically unit leaves have stride zero, and symbolic strides remain symbolic.
    Split modes owned per instance by `ctx.target` for the result `storage` also
    have stride zero. Compacting the remaining
    modes drops only each Split's mesh factor and retains its local residual
    (extent one for a fully factored Split). The set comes from the target memory
    owner and every coarser topology level. With no memory hierarchy it is empty.
    Otherwise a compatible propagated physical layout is preserved when possible.
  - `ShardLayout.attrs` is indexed by mesh axis. A `Partial(reduction)` is a
    value state at that exact index, not an unordered collection of reductions
    and not a layout position. Every propagation decision MUST retain that
    index and its reduction independently of every other mesh axis.
  - A reduction dim (a domain dim absent from the output access map) is declared
    by the op in `partial_reduction_dims` or `complete_reduction_dims`; both
    yield `Partial(reduction)` on a split mesh axis (step 3).
  - Propagation applies per input mesh axis:
    1. `Split(k)` maps layout axis `k` to the input's logical tensor axis and
       then to a domain dim through the input access map.
    2. If that domain dim appears in the output access map, the output carries
       `Split` on the output layout axis to which the domain dim maps.
    3. If that domain dim is reduced, the output mesh axis becomes
       `Partial(reduction)`. This `Partial` is a value state on that mesh axis
       and carries no layout axis.
    4. A `Partial(reduction)` input propagates on the same mesh axis only when
       the op's math is proven to commute with `reduction`:
       `op(reduction(x0..xn)) == reduction(op(x0)..op(xn))`. This service MUST
       NOT make that mathematical judgment; each op's typeinfer rule owns it.
       On each mesh axis, two Partial inputs MUST be compatible with that op's
       rule. States on different mesh axes MUST NOT be compared as an unordered
       set. Only an explicit reduction/allreduce resolves `Partial` to
       `Broadcast`.
    5. A `Broadcast` input axis contributes no `Split`.
    6. Two inputs giving one mesh axis different output states is an error.
  - A `Partial` MUST NOT be silently eliminated or carried through a
    non-commuting ordinary op.
  - An ordinary multi-input op MUST inspect every tensor input for a `Partial`.
    When its result type cannot represent a secondary input's axis-preserving
    state, typeinfer MUST reject that input and name the `Reshard` remedy.
  - A fully-`Broadcast` input `ShardLayout` is replicated: it carries no real
    sharding, contributes no `Split`/`Partial`, and does not pin a mesh. When no
    input carries real sharding, the output carries none.
  - Only a zero-offset, unit-coefficient access of one domain dim is a
    projection for shared ownership propagation. An input `Split` that accesses
    any other form, including a nonzero affine translation, or an
    output-surviving dim reachable only through such an access, MUST fail
    closed (`ValueError`): without domain bounds the service has no
    owner-boundary alignment proof for a translated access.
  - `Split(axis)` indexes an output layout axis, not a logical tensor axis.

### 3.3 Output storage and mesh/layout compatibility

- constraints:
  - A symmetric multi-input op (`Binary`, `MatMul`, `Concat`, `Stack`)
    resolves output storage with `resolve_anchor_storage` over the concrete
    residency among its operands ([types §2](./types.md#2-tensortype)),
    independently of operand order and with no memory-level lattice.
  - An unmaterialized operand (`storage=umat`), including host shape metadata,
    abstains. One concrete storage, or several that agree, is the output
    storage; with none the output is `umat`.
  - Several concrete operands that disagree cause typeinfer to `ctx.error`,
    unless the op defines its own destination/mixed-storage resolution. There is
    no operand-order tie-break.

A tensor value's mesh / layout is carried by its `TensorType.layout`
(`ShardLayout.mesh` names the mesh instance), and the IR places no scope-based
restriction on values from different meshes coexisting. Each op's registered
typeinfer **owns** the operand layout / mesh compatibility it requires and its
result layout; no uniform cross-op rule is imposed from outside. `Reshard` is
the explicit op that changes a value's layout / mesh. For example, `IndexSelect`
owns whether an indexed access is a foldable one-element slice or a
materializing data-dependent selection.
