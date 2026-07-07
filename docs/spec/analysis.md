# TileFoundry Spec — analysis (static analysis services)

This spec owns TileFoundry's concrete static analysis services — the semantic
contracts that derive types, access relations, and shard layouts over the IR.
Each service is a registry-backed derived visitor: the common registration and
dispatch mechanism is owned by [visitor-registry](./visitor-registry.md); this
file owns each service's requirements, handler shape, required context, and the
semantic rules it enforces. TileGraph relation vocabulary is owned by
[tilegraph](./tilegraph.md); this file links to it rather than redefining it.

## 1. Type propagation

Type inference is registered per Op through
`@register_typeinfer(<OpClass>)` and enforces its constraints via `ctx.error(...)`
([visitor-registry §4](./visitor-registry.md)). A handler receives the op and a
typeinfer context, derives the output `IRType`, and reports violations through
`ctx.error`.

### 1.1 Relation-derived type behavior

An op's typeinfer MAY derive the output type from a forward access
relation ([visitor-registry §4.1](./visitor-registry.md#41-forward-relation-service--type_relation))
rather than from a hand-written rule. The relation describes one
shared iteration domain and, per boundary, an access map from that
domain to the tensor's index space. The relation carries **no tensor
shape**: the output shape is typeinfer-side data, derived from the
op's shape rule or (where implemented) from the relation by composing
the output access map over the domain.

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

The relation describes one shared iteration domain; a symbolic size is an isl
parameter of that domain, and the relation's rank is fixed and read from the
input types. Output shape is not carried by the relation — it is typeinfer-side
data, derived from the op's shape rule or, where implemented, by composing the
output access map over the domain. Output access-map arity is validated by the
relation service ([visitor-registry §4.1](./visitor-registry.md#41-forward-relation-service--type_relation)).

## 2. Access relation analysis

The forward access relation is the boundary model shared by relation-derived
type behavior and shard propagation: one iteration domain plus, per boundary, an
affine access map from that domain to a tensor's index space. The relation
vocabulary — access relations, opaque relations, and the access-relation result
carrier — is defined in [tilegraph §3.4](./tilegraph.md#34-accessrelation); the
relation service that produces it is registered as described in
[visitor-registry §4.1](./visitor-registry.md#41-forward-relation-service--type_relation).
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

**Reduction effect.** A reduction dim (a domain dim absent from the
output access map) carries one of two effects, declared by the
op/relation:

- `partial` — the per-shard result is a partial that still needs a
  cross-shard reduction (e.g. a contraction dim split across the mesh);
- `complete` — the reduction is already complete within each shard
  (e.g. an explicit reduce over a sharded axis).

**Propagation.** Per input mesh axis, by its attr:

1. `Split(k)` — map cute axis `k` to the input's logical tensor axis,
   then to a domain dim via the input access map.
2. If that domain dim appears in the output access map, the output
   carries `Split` on the **output layout axis** the domain dim maps to.
3. If that domain dim is a reduction dim, the output mesh axis becomes
   `Partial(reduction)` when the effect is `partial`, or `Broadcast`
   when the effect is `complete`. The resulting `Partial` carries no
   cute axis — it is a value state on that mesh axis.
4. `Partial(reduction)` input — propagates on the **same mesh axis**:
   propagate unchanged when the dataflow is homogeneous in `reduction`;
   resolve to `Broadcast` via an explicit reduction / allreduce over that
   axis; error on a non-homogeneous use or an unreduced function
   output / return. There is no cute-axis mapping for a `Partial`.
5. A `Broadcast` (size-1) input axis contributes no `Split`.
6. Two inputs binding the same domain dim to incompatible mesh axes is
   an error.

A `Partial` MUST NOT be silently eliminated by an ordinary op
(no silent loss); only an explicit `Reshard` / allreduce from `Partial`
to `Broadcast` completes it.

A fully-`Broadcast` input `ShardLayout` (every attr `Broadcast`) is
**replicated**: it carries no real sharding, so it contributes no
`Split` / `Partial` and does not pin a mesh — it MAY combine with an
input sharded on a different mesh. When no input carries real sharding
the output carries none.

An input `Split` that accesses a non-projection domain dim, or an
output-surviving dim reachable only through a non-projection output
access, MUST **fail closed** rather than guess a mapping. The rule
reads only the access maps' affine structure (which domain dim each
axis uses), never the domain bounds, so it is size-agnostic and
identical for static and dynamic shapes.

**Owner axis.** `Split(axis)` indexes an **output layout (cute) axis**,
not the logical tensor axis. A reduction-induced `Partial` attaches to no
cute axis — it is a value state on the mesh axis that was reduced.

### 3.3 Output storage and mesh/layout compatibility

A symmetric multi-input op (`Binary`, `MatMul`, `Concat`, `Stack`,
`Mma`) resolves its output `storage` by **anchoring** on the concrete
residency among its operands ([types §2](./types.md)). The rule does not
appeal to any ordering of storage kinds and is independent of operand order:

- An **unmaterialized** operand (`storage=umat`) does not constrain the
  output — it abstains.
- One concrete operand storage (alongside any unmaterialized operands) is
  the **anchor**; the output takes that storage.
- Several concrete operands that agree on a storage → the output takes that
  storage.
- Several concrete operands that disagree on storage → typeinfer MUST
  `ctx.error`, unless the op defines its own destination/mixed-storage
  resolution. There is no operand-order tie-break.
- All operands unmaterialized → the output is unmaterialized (`umat`).

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

Per-op typeinfer owns layout compatibility and result layout. For example,
`Gather` owns whether an indexed access is a pure slice or a layout-preserving
data-dependent gather.
