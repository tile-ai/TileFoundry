# TileFoundry Spec — semantic-analysis (per-Op semantic derivation)

This spec owns TileFoundry's concrete static analysis services — the semantic
contracts that derive types, access relations, and shard layouts over the IR.
Each service is a registry-backed derived visitor: the common registration and
dispatch mechanism is owned by [visitor-registry](./visitor-registry.md); this
file owns each service's requirements, handler shape, required context, and the
semantic rules it enforces. The access relation's own carrier is owned by
[visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation);
this file links to it rather than redefining it.

## 1. Type propagation

Type inference is registered per Op through
`@register_typeinfer(<OpClass>)` and enforces its constraints via `ctx.error(...)`
([visitor-registry §4](./visitor-registry.md#4-instance-1--typeinfer)). A handler receives the op and a
typeinfer context, derives the output `Type`, and reports violations through
`ctx.error`. A `hir.Function` call composes these per-op rules under
elaboration ([hir §1.1](./hir.md#11-function)): the callee body is
reconstructed and each of its nodes re-derives through the same per-op
rules under the call's actual argument types, so a relation-derived rule
never needs its own function-boundary case.

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
the Op's own iteration space to that tensor's index space, and what the Op walks
is the union of those domains. The relations carry **no tensor shape**: the
output shape is typeinfer-side data, derived from the op's shape rule or from
the extents the output relation reaches.

Within the relation:

- A domain dim that appears in an input access map but **not** in the
  output access map is a **reduction** dim (it is eliminated in the
  output).
- A tensor axis whose access maps to a constant (rather than a domain
  dim) is a **broadcast** axis.
- A symbolic size is an isl parameter of the domain; the relation's
  rank is fixed and is read from the input types.

The shard consequences of these structural facts (how `Split` /
`Broadcast` / `Partial` propagate, and the reduction effect) are
defined in [§3.2](#32-relation-driven-shard-propagation).

### 1.2 Domain construction and output shape derivation

Every boundary's image rank is held to the Type of the value it describes by the
relation service ([visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation)).

## 2. Access relation analysis

The access relation is the boundary model shared by relation-derived type
behavior, shard propagation, dependence and movement: per boundary, an affine
access map from the Op's own iteration space to a tensor's index space. Its
carrier `AccessRelations` and the registry that produces it are both defined in
[visitor-registry §4.1](./visitor-registry.md#41-access-relation-service--access_relation).
The rule reads only the access maps' affine structure (which domain dim each
axis uses), never the domain bounds, so it is size-agnostic and identical for
static and dynamic shapes.

## 3. Shard propagation

### 3.1 Logical shape to layout domain

- `TensorType.shape` is the logical shape.
- `layout` has its own domain shape.
- The current interpretation is canonical regroup: linearize first
  along the logical shape's row-major order, then reinterpret along
  the layout domain's row-major order.

### 3.2 Relation-driven shard propagation

When an op's output `ShardLayout` is derived from a forward access
relation ([§1.1](#11-relation-derived-type-behavior)), the
output `ShardAttr`s are determined from the input shards and the
relation's access maps by a single rule, shared across ops.

```python
def partial_reductions_by_axis(layout: object) -> tuple[str | None, ...]: ...

def derive_output_shard_layout(
    input_types: tuple,
    relation,
    output_shape: tuple,
    *,
    partial_reduction_dims: frozenset[int] = frozenset(),
    complete_reduction_dims: frozenset[int] = frozenset(),
    fresh_strides: bool = False,
): ...
```

- constraints:
  - `derive_output_shard_layout` reads only input types, forward-relation maps,
    the requested output shape, and the explicit reduction/stride controls. It
    MUST NOT read relation bounds or an already-derived output type.
  - `partial_reductions_by_axis` returns one entry per mesh axis: the carried
    reduction name for `Partial`, `None` for any other attr, and an empty tuple
    for a non-sharded layout.
  - With no real input sharding it returns `None`. Inputs with real sharding on
    different meshes MUST fail.
  - `partial_reduction_dims` turns a reduced split into `Partial("sum")`;
    `complete_reduction_dims` turns it into `Broadcast`. The sets MUST NOT
    overlap.
  - `fresh_strides=True` requests fresh canonical strides; otherwise a
    compatible propagated physical layout is preserved when possible.
  - `ShardLayout.attrs` is indexed by mesh axis. A `Partial(reduction)` is a
    value state at that exact index, not an unordered collection of reductions
    and not a layout position. Every propagation decision MUST retain that
    index and its reduction independently of every other mesh axis.
  - A reduction dim (a domain dim absent from the output access map) carries
    one of two effects declared by the op/relation: `partial` means the
    per-shard result still needs a cross-shard reduction; `complete` means the
    reduction is already complete within each shard.
  - Propagation applies per input mesh axis:
    1. `Split(k)` maps layout axis `k` to the input's logical tensor axis and
       then to a domain dim through the input access map.
    2. If that domain dim appears in the output access map, the output carries
       `Split` on the output layout axis to which the domain dim maps.
    3. If that domain dim is reduced, the output mesh axis becomes
       `Partial(reduction)` for a `partial` effect or `Broadcast` for a
       `complete` effect. The resulting `Partial` is a value state on that mesh
       axis and carries no layout axis.
    4. A `Partial(reduction)` input propagates on the same mesh axis only when
       the op's math is proven to commute with `reduction`:
       `op(reduction(x0..xn)) == reduction(op(x0)..op(xn))`. This service MUST
       NOT make that mathematical judgment; each op's typeinfer rule owns it.
       On each mesh axis, two Partial inputs MUST be compatible with that op's
       rule. States on different mesh axes MUST NOT be compared as an unordered
       set. Only an explicit reduction/allreduce resolves `Partial` to
       `Broadcast`.
    5. A `Broadcast` input axis contributes no `Split`.
    6. Two inputs binding the same domain dim to incompatible mesh axes is an
       error.
  - A `Partial` MUST NOT be silently eliminated or carried through a
    non-commuting ordinary op; only an explicit `Reshard`/allreduce from
    `Partial` to `Broadcast` completes it.
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
    closed. The shared service reads only the maps' affine structure, never the
    domain bounds, so it has no owner-boundary alignment proof for a translated
    access.
  - `Split(axis)` indexes an output layout axis, not a logical tensor axis. A
    reduction-induced `Partial` attaches to no layout axis; it remains a value
    state on the mesh axis that was reduced.

### 3.3 Output storage and mesh/layout compatibility

- constraints:
  - A symmetric multi-input op (`Binary`, `MatMul`, `Concat`, `Stack`)
    resolves output storage by anchoring on the concrete residency among its
    operands ([types §2](./types.md#2-tensortype)); the rule is independent of
    operand order.
  - An unmaterialized operand (`storage=umat`), including host shape metadata,
    abstains and does not constrain the output.
  - One concrete operand storage is the anchor; the output takes that storage.
  - Several concrete operands that agree on a storage produce that storage.
  - Several concrete operands that disagree on storage cause typeinfer to
    `ctx.error`, unless the op defines its own destination/mixed-storage
    resolution. There is no operand-order tie-break.
  - If all operands are unmaterialized, the output is unmaterialized (`umat`).

This resolution uses no memory-level lattice; output residency is a function
of the concrete anchor(s) alone. (The `rmem < smem < gmem` hierarchy is
a `Reshard`-*direction* notion and is unrelated to output-storage anchoring.)

A tensor value's mesh / layout is carried by its `TensorType.layout`
(`ShardLayout.mesh` names the mesh instance) — that type is the source of
truth, and the IR places no scope-based restriction on values from
different meshes coexisting. Each op's registered typeinfer **owns** the
operand layout / mesh compatibility it requires and its result layout;
there is no uniform cross-op rule imposed from outside typeinfer.
`Reshard` is the explicit op that changes a value's layout / mesh.

For example, `IndexSelect` owns whether an indexed access is a foldable
one-element slice or a materializing data-dependent selection.
