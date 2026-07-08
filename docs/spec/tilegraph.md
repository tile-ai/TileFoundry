# TileFoundry Spec — TileGraph

Architectural placement is in
[architecture §5](./architecture.md#5-analysis--optimization):
`TileGraph` is **not an IR layer**. It is a **pass-private
intermediate representation** used by certain optimisation passes
(polyhedral analysis, tile search) when they need to work on top of
HIR or TIR. The pass extracts a regular region from TIR, lifts it
into a `TileGraph`, runs analysis or search, and rewrites the result
back into TIR. `TileGraph` never appears on the `tir → codegen`
critical path. This spec covers only the `TileGraph` core objects.

## 1. Scope

`TileGraph` is the **per-level SSA DAG** carrier used inside a pass.
This spec defines its core objects and minimum semantic skeleton.

## 2. Admission boundary

A `TileGraph` does not accept arbitrary front-end code. It only
accepts a **regular region** extracted from the front end or from
`script`. A region MUST satisfy:

- it can map onto a single canonical `domain` at the current level,
- it can be expressed as a `DiGraph<ITileNode>`,
- the graph's boundary accesses can be expressed as structured
  `access relation`s,
- any subregion that needs another level can be expressed as a
  child `TileGraph` carrying its own `domain relation`.

The current entry path is:

```
surface op call → type-pattern dispatch → implementation body
                → regular region → TileGraph
```

Where:

- type-pattern dispatch belongs to the front-end boundary, not to
  `TileGraph` core,
- `Mesh` / sharding / `.local()` belong to the front-end surface and
  are resolved before lifting into a `TileGraph`.

## 3. Core objects

### 3.1 `Value`

`TileGraph` does not define its own `Tensor` type. The
`inputs` / `outputs` of a `TileGraph` and a `TileUnit` reuse the
existing IR value. In the current implementation that value is the
shared `SSAValue` whose `type` MUST be a `TensorType`.

### 3.2 `Domain`

```text
Domain = isl.set
```

- kind: Python class
- fields: the legal iteration domain of the current `TileGraph`, carried by `isl.set`
- constraints: none — carried directly by `isl.set`

`Domain` is carried directly by `isl.set`. It represents the legal iteration
domain of the current `TileGraph`.

### 3.3 `DomainRelation`

```text
DomainRelation = isl.multi_aff
```

- kind: Python class
- fields: the map from the parent `TileGraph.domain` onto the current `TileGraph.domain`, carried by `isl.multi_aff`
- constraints: none — `isl.multi_aff` is sufficient at this layer

`DomainRelation` is carried directly by `isl.multi_aff`. It maps the parent
`TileGraph.domain` onto the current `TileGraph.domain`. `isl.multi_aff` is
sufficient at this layer.

### 3.4 `AccessRelation`

```text
AccessRelation = isl.multi_aff
```

- kind: Python class
- fields: the affine map from the current `TileGraph.domain` to a boundary value's index space, carried by `isl.multi_aff`
- constraints:
  - `AccessRelation` itself does not carry a tensor or value reference, a read /
    write mode, or any further semantics; those bindings live on the surrounding
    `TileGraph`'s boundary fields.

`AccessRelation` is also carried by `isl.multi_aff`. It expresses the affine map
from the current `TileGraph.domain` to a boundary value's index space.

### 3.5 `ITileNode`

```text
ITileNode
  inputs:  Value[]
  outputs: Value[]
```

- kind: Python class
- fields:
  - inputs: boundary input `Value`s
  - outputs: boundary output `Value`s
- constraints:
  - `ITileNode` unifies graph-node boundaries only; it does not unify `domain`.

`ITileNode` is the abstract base of every node that participates in
a `TileGraph` body.

### 3.6 `DiGraph<TNode>`

```text
DiGraph<TNode>
  nodes
  edges
```

- kind: Python class
- fields:
  - nodes: the graph nodes
  - edges: the graph edges
- constraints: none — the in-memory API is not pinned by this spec (reading-style reference: networkx)

`TileGraph.body` is carried by a directed-graph object. Reading-style
reference: networkx. The in-memory API is not pinned by this spec.

### 3.7 `TileGraph`

```text
TileGraph : ITileNode
  domain: Domain
  domain_relation: DomainRelation        # optional; child graphs only
  inputs:  Value[]
  input_access_relations:  AccessRelation[]
  outputs: Value[]
  output_access_relations: AccessRelation[]
  body: DiGraph<ITileNode>
```

- kind: Python class
- fields:
  - `domain` is the iteration domain of the current graph.
  - `domain_relation` belongs to `TileGraph` only; ordinary `ITileNode`
    members do not carry one.
  - `input_access_relations` aligns positionally with `inputs`.
  - `output_access_relations` aligns positionally with `outputs`.
  - `body` is the directed graph at the current level.
- constraints:
  - the composite node; a per-level SSA DAG. `domain_relation` is present on
    child graphs only.

`TileGraph` is the composite node.

### 3.8 `TileUnit`

```text
TileUnit : ITileNode
  op_kind: str
  inputs:  Value[]
  outputs: Value[]
```

- kind: Python class
- fields:
  - op_kind: the unit's op kind
  - inputs: boundary input `Value`s
  - outputs: boundary output `Value`s
- constraints:
  - a `TileUnit` does not own its own `domain`,
  - it does not own its own `domain_relation`,
  - as the minimum unit it is interpreted under the `domain` of the
    enclosing `TileGraph`.

`TileUnit` is the minimum-unit node.

## 4. Semantic skeleton

The minimum semantic skeleton is:

```
parent graph domain
    --domain relation-->  current graph domain
    --access relation-->  boundary value indices
```

That is:

- `domain` belongs to a `TileGraph`,
- `domain relation` expresses only the mapping between a graph and
  its parent graph,
- `access relation` expresses only the affine map from the current
  graph's `domain` to a boundary value's index space,
- `TileUnit` introduces no new domain; it consumes and produces
  existing IR values.

In addition:

- a `TileGraph` is a per-level **SSA DAG**,
- `TileGraph.body` is a `DiGraph<ITileNode>`,
- `TileUnit` is the minimum compute unit inside the current graph.
