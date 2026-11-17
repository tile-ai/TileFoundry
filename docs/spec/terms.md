# TileFoundry Spec — Terms

This spec defines the words that name TileFoundry concepts in Python, C++ and the specs. Other specs link here and do not restate a definition.

## 1. Terms by example

```math
\underbrace{\mathcal{T}_{\mathrm{thread}}}_{\text{topology}} = \underbrace{(384)}_{\text{topology size}}
\qquad\qquad
\underbrace{0,\ 1,\ 2,\ \dots,\ 383}_{\text{unit ids}}
```

1. A **topology** is one layer of parallel resource. Its name is the kind of **unit**. Its **topology size** counts its units within one unit of its parent, when present. A **unit id** is the number of one unit in its topology. The **topology scope** is all units of the topology.

```math
\underbrace{\big(\ \underbrace{\mathrm{gpu}:1}_{\text{topology level }0},\quad \underbrace{\mathrm{cta}:132}_{\text{topology level }1},\quad \underbrace{\mathrm{thread}:384}_{\text{topology level }2}\ \big)}_{\text{topologies of the Module}}
```

2. A Module declares an ordered list of **topologies**. The **topology level** is the position of a topology in this list. Level 0 is the coarsest. The list is the Module's, not the target's: in `(cta, thread)` the cta is at level 0.

```math
\underbrace{\mathcal{M}}_{\text{mesh}} =
\underbrace{\Bigg[\ \underbrace{\begin{matrix}132@\mathrm{sm}\\384\end{matrix}}_{\text{cta group}}\ \Bigg|\ \underbrace{\begin{matrix}3@\mathrm{group} & 128@\mathrm{lane}\\128 & 1\end{matrix}}_{\text{thread group}}\ \Bigg]}_{\text{mesh layout}}
\qquad
\underbrace{\mathcal{M}_{\mathrm{cta}} = \begin{bmatrix}132@\mathrm{sm}\\1\end{bmatrix}}_{\text{sub mesh}}
```

3. A **mesh** puts the units of adjacent topologies on named **mesh axes**. Each column of the **mesh layout** is `extent@mesh axis` above its stride. Each topology has one group of columns. A **sub mesh** contains one topology. Its unit-id layout is `composition` of its group with `U_t`, the projection of the index onto that topology.

```math
\underbrace{(1,\ 1,\ 40)}_{\text{coord}}\ \mapsto\ 384\cdot 1 + 128\cdot 1 + 40 = \underbrace{552}_{\text{index}} = \underbrace{1}_{\text{unit id in cta}}\cdot 384 + \underbrace{168}_{\text{unit id in thread}}
```

4. The mesh layout maps a **coord** to the **index** of a unit. The index counts the units of all topologies of the mesh: `T = make_layout(sizes)`, row-major. `idx2crd(index, sizes)` gives one unit id for each topology.

```math
\underbrace{\mathcal{M}_{\mathrm{thread}}[1{:}3,\ :]}_{\text{sliced mesh}} = \begin{bmatrix}2@\mathrm{group} & 128@\mathrm{lane}\\128 & 1\end{bmatrix} + \underbrace{128}_{\text{offset}}
```

5. A **sliced mesh** keeps the topology and the mesh axes. Its layout is a `ComposedLayout`. The **offset** is its `ComposedLayout.offset`, the index of the first unit of the slice.

```math
\underbrace{\big(\ \mathcal{M}_1,\ \underbrace{\mathcal{M}_2}_{\text{innermost mesh}}\ \big)}_{\text{enclosing meshes}}
\qquad
\underbrace{\mathcal{M}_{\mathrm{scope}}}_{\text{mesh scope}} = \Bigg[\ \begin{matrix}132@\mathrm{sm}\\384\end{matrix}\ \Bigg|\ \begin{matrix}2@\mathrm{group} & 128@\mathrm{lane}\\128 & 1\end{matrix}\ \Bigg] + 128
```

6. Each mesh `with` adds one **enclosing mesh**. The first one is the outermost. The last one is the **innermost mesh**. The **mesh scope** composes the enclosing meshes in order, with the rules of [shard §5](./shard.md#5-mesh). Its units are in force at that point of the program. Here $\mathcal{M}_1$ is the mesh of all cta, and $\mathcal{M}_2$ is the sliced mesh of item 5.

```math
\underbrace{\mathrm{mma}}_{\text{instruction}}:\quad
\underbrace{\mathcal{M}_{\mathrm{exec}} = \begin{bmatrix}32@\mathrm{lane}\\1\end{bmatrix}}_{\text{execution mesh}}
\qquad
\mathrm{repeat} = (4,\ 2)\ \Rightarrow\ \underbrace{8}_{\text{issues}}
```

7. An **instruction** is a TIR op that declares an **execution mesh** and can be selected by scheduling. One **issue** is one application of the instruction to one tile. Lowering issues the instruction. The units of the execution mesh execute it.

```math
\underbrace{\mathcal{L}_C}_{\text{shard layout}} = \big(\ \underbrace{(256,\ 64):(64,\ 1)}_{\text{layout of the data}},\ \ \underbrace{(\mathrm{Broadcast},\ \mathrm{Split}(0),\ \mathrm{Split}(0))}_{\text{shard attrs}},\ \ \underbrace{\mathcal{M}_{\mathrm{scope}}}_{\text{mesh}}\ \big)
```

```math
\underbrace{(256,\ 64)}_{\text{shape of }C}\ \rightarrow\ \underbrace{(1,\ 64)}_{\text{local shape of one unit}}
```

8. A **shard layout** binds the layout of the data to a mesh. It has one **shard attr** for each mesh axis. **Local** is the part that one unit of the specified topology holds. The example gives the local shape at the thread topology.

```math
\underbrace{\{\ \underbrace{\mathrm{gpu}}_{\text{topology}} : \underbrace{0}_{\text{unit id}}\ \}}_{\text{placement}}
```

9. A **placement** gives a unit id for each topology that it names. It tells where a process or a piece of data is. It may name only some topologies.

The names `gpu`, `cta`, `thread`, `sm`, `group` and `lane` above are examples of one target and its mesh axes.

## 2. Execution

| Term | Definition | Source | Words that MUST NOT name it | Python | C++ |
|---|---|---|---|---|---|
| unit (execution unit) | One thing that executes: one gpu, one cta, one thread. | own | `program`, `instance`, `participant`, `position` (for a member) | `per_unit`, `max_physical_units` | — |
| topology | One layer of parallel resource: the kind of unit (its name) and the number of units. | existing | `level` (for this object) | `Topology(name, size)` | enum value only |
| topology size | The number of units of this topology in one unit of the topology above. | CuTe `size` | `program_dim`, `instances`, `extent` of a topology | `Topology.size` | `topology_size<L>()` |
| topology level | The position of a topology in the topologies that the Module declares, an `int`. 0 is the outermost. Python and C++ count the same list. | own | `topology_level: str` (a value that is a name) | `topology: str` for a name; `topology_level(topology, topologies)` gives the position | `cute::find(program_topologies(), cute::C<L>{})` gives the position; `enum class TopologyLevel` labels the topology |
| unit id | The number of a unit in its topology, in `[0, size)`. | CUDA `%tid`, `%ctaid` | `program_id`, `rank` | `Placement.ids`, `unit_ids()` | `unit_id<L>()`, `unit_ids()` |
| mesh | The units of one or more adjacent topologies, arranged by a layout on named mesh axes. | JAX | — | `Mesh(topologies, layout, names)` | `Mesh<TLayout, TopologyLevel...>` |
| sub mesh | The part of a mesh that is one topology. Its mode is `get(mesh.layout, t)`, strides in the index over all topologies. As a mesh of one topology, its layout is `utils.unit_layout(mesh, t)`. | CuTe sublayout, `get<I>` | `separate`, `level` (for this object), `sub-mesh` for the result of a slice | `get(mesh.layout, t)`, `utils.topology_mesh(mesh, topology)` | `cute::get<t>(mesh.layout)` |
| mesh axis | One axis of a mesh. | JAX `axis_names` | `axis` alone | `mesh_axis`, `Mesh.names` | — |
| mesh layout | The layout of a mesh. It maps a coord to the index over all topologies of the mesh. Mode `t` holds the group of topology `t`. The unit id of topology `t` at coord `c` is `idx2crd(L(c), sizes)[t]`. | CuTe | `arrangement` | `Mesh.layout` | `Mesh::layout` |
| coord | The location of a unit on the mesh axes. | CuTe | `position` (for a location), "device numbering" | `mesh_coords`, `MeshCoord` | `mesh_coords` |
| index | The linear number that a layout gives to a coord. The index of a mesh layout runs over `T = make_layout(sizes)` of its topologies. The index of the first unit is `L(0)`, `crd2idx(0, L)`. | CuTe | "device numbering" | `mesh.layout(0)` | `mesh.layout(cute::Int<0>{})` |
| offset | `ComposedLayout.offset`, CuTe `ComposedLayout::offset()`. Of a sliced mesh it is the index of its first unit. It is not a unit id; `idx2crd` of an index gives the unit ids. | CuTe | `starts` | `ComposedLayout.offset` | `layout.offset()` |
| placement | A map from topology names to unit ids. It can name only some topologies. It tells where a process or a piece of data is. | OneFlow | `ProgramMetaData`; `placement` for a shard attr; "memory placement" | `Placement` | `Placement` |
| topology scope | All units of one topology. | own | `scope` alone | — | — |
| mesh scope | The units in force at one point of the program. It is one mesh. It composes the meshes of all enclosing `with`. Its units come from the topology scopes. | own | `scope` alone; `current_mesh`, `active_mesh` | `ctx.mesh_scope`, `MeshScope` | — |
| enclosing meshes | The mesh of each enclosing `with`, outermost first. The last one is the **innermost mesh**. | own | `mesh_stack`, `opened_mesh`, `entered_mesh`, `region_mesh`, `enclosing_mesh` (singular) | `ctx.enclosing_meshes()`, `ctx.innermost_mesh()` | — |
| execution mesh | The units that execute an instruction, as a mesh. Its ISL form is the execution domain. | existing ([tir](./tir.md)) | `participant frame`, `instruction frame`, `execution frame`, `issue mesh` | `execution_mesh`, `find_execution_mesh` | — |
| root mesh | The mesh that a function gets from outside. | existing | — | `root_mesh` | — |
| instruction | A TIR op that declares an execution mesh and can be selected by scheduling. A hardware instruction is a machine instruction. | existing | `instruction` for any TIR op | `match_instruction` | — |
| issue | One application of an instruction to one tile. Lowering writes one call statement for it. Lowering issues; a unit executes. | architecture | `issuers`, "issuing threads" (these are the execution mesh); `issue` for execute | `Issue`, `IssuePlan`, `repeat` | — |
| context | `valuation.Context`: the environment of one function (module, function, target, topologies, mesh scope). | existing | `frame` (a Python stack frame only) | parameter `ctx` | — |
| wave | A set of units that run at the same time. | own | — | `wave_units` | — |
| program | The result of one compilation (one launch). It is not a unit. | existing | `program` for a unit | `check_program` | `program_topologies()` |

## 3. Data

| Term | Definition | Source | Words that MUST NOT name it | Python |
|---|---|---|---|---|
| shard layout | Binds the modes of a layout to a mesh. | existing | — | `ShardLayout` |
| shard attr | How data is distributed on one mesh axis: `Split`, `Broadcast`, `Varying`, `Partial`. | existing; `Varying` as in JAX | "placement sugar", "placement semantics" | `ShardAttr` |
| local | The part that one unit of the specified topology holds. | CuTe `local_tile`, PyTorch `to_local` | — | `to_local`, `local_type_of`, `Local` |

## 4. Layout

All layout terms are as in CuTe.

| Term | Definition | Words that MUST NOT name it |
|---|---|---|
| `Layout`, `shape`, `stride` | As in CuTe. | — |
| mode | One dimension of a layout. | "layout axis" |
| rank | The number of modes. It is not a number of a unit. | `rank` for a unit id |
| extent | One entry of a shape. | — |
| size, cosize | As in CuTe. | — |
| algebra | The layout functions are a port of CuTe: same names, argument order and semantics ([shard §9](./shard.md#9-layout-construction-and-algebra)). Row-major is the default order. | — |
| `ComposedLayout.inner`, `ComposedLayout.outer` | CuTe `layout_a` and `layout_b`. | — |

## 5. Polyhedral

| Term | Definition | Python |
|---|---|---|
| computation domain | The ISL set of iteration points. | `iteration_domain` |
| execution domain | The ISL set of units in polyhedral optimization. The coords of the mesh are its dimensions. | — |
| iteration scope | The loops in force at one point of the program. | `IterationScope` |

## 6. Other words

| Word | Use |
|---|---|
| `position` | Not a term. In layout and mesh code, write unit, unit id (index), mode or coord. |
| memory level, cache level | Keep the prefix. |
| memory placement, address placement | Write address. |
| unit of measurement (`unit="byte"`) | Write measurement unit. Only `target/hardware/` and the hardware fact files can write `unit` alone. |
| `unit: str` in analysis (a topology name, such as `cta`) | Write `topology: str`. |
| translation unit | Write the full term. |
| unit stride, unit step | Write stride-1, step-1. |

## 7. Rules

- One concept has one word in Python, C++ and the specs.
- To select a word, use CuTe first, then JAX. Use a word of our own only when neither has one.
- `level`, `scope`, `domain` and `axis` always have a qualifier: topology level, memory level, mesh scope, mesh axis. A class, function or module that handles one case only can omit it.
- Only `targets/nvidia` and the sync classification code use the CUDA names warp, lane, cluster and CTA in implementation code.
- Spec examples can use the names of a specific topology or mesh axis, for example `cta`, `thread` and `lane`.
- Do not keep an alias for an old name.
