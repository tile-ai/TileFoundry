# TileFoundry CUDA Spec — Runtime

This spec owns the CUDA runtime headers under
`targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/`.
They are package data, located through `importlib.resources`, and build on
CUTLASS / CuTe. The [core runtime](../../../../../../docs/spec/runtime.md) owns
Python APIs. `scripts/runtime_spec_surface.py` generates declaration blocks;
hand-written constraints state semantics and design reasons.

## 2. C++ Runtime Surface

### 2.2 `cuda/`

#### 2.2.1 `cuda/runtime.cuh`

<!-- generated: cuda-runtime -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/runtime.cuh
inline constexpr int kWarpSize;

template <class...> inline constexpr bool dependent_false_v;

enum class TopologyLevel {
    gpu,
    cta,
    thread,
    topology_count,
};

template <TopologyLevel L>
CUTE_HOST_DEVICE constexpr auto topology_size() noexcept;

template <TopologyLevel L>
CUTE_HOST_DEVICE constexpr auto topology_shape() noexcept;

CUTE_HOST_DEVICE constexpr TopologyLevel program_topology() noexcept;

CUTE_HOST_DEVICE constexpr auto program_topologies() noexcept;

struct Placement {
    int unit_id[int(TopologyLevel::topology_count)];
};

CUTE_HOST_DEVICE Placement &placement();

CUTE_HOST_DEVICE void placement(Placement const &given);

template <TopologyLevel L> CUTE_HOST_DEVICE size_t unit_id() noexcept;

CUTE_HOST_DEVICE auto unit_ids() noexcept;
```
<!-- /generated -->

**Terms.** The terms are those of [shard §0](../../../../../../docs/spec/shard.md#0-terms). A warp grouping is a mesh axis of a thread mesh, not a topology.

- constraints:
  - `unit_id<TopologyLevel::gpu>()` MUST equal the host `Placement`'s id
    ([runtime §1.10](../../../../../../docs/spec/runtime.md#110-shardtensor)).
    Devices have no register for their coord in the host's mesh.
  - The id travels in a by-value `Placement` kernel parameter, hidden
    from the user signature. Before divergence, one thread MUST initialize
    `placement` and synchronize readers. Its `__shared__` slot belongs to
    one block, so concurrent kernels cannot overwrite each other's metadata.
  - Every length indexed by topology level MUST derive from `topology_count`, including `unit_id`, `unit_ids()` and mesh coords. Refuse short coords. Do not read the id of an adjacent topology.
  - `program_topology()` is the coarsest named topology. `program_topologies()` lists the topologies from there to `topology_count`, coarsest first. Use CuTe's `rank` and `find` on that tuple. For a missing topology, `find` gives the rank. The following `get` MUST refuse it as `"Index out of range"`.
  - Coords are row-major: `(8, 32)` gives `(id / 32, id % 32)`. This is
    opposite to CuTe order ([§2.3.1](#231-layoutcute_extcuh)).
  - One `.cu` states the `topology_size<L>()` of one launch. `topology_shape<L>()` includes topology `L` and all topologies below it. A mesh recompiled in another translation unit belongs to the launch of that translation unit.

### 2.3 `cuda/layout/`

#### 2.3.1 `layout/cute_ext.cuh`

<!-- generated: layout-cute-ext -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/layout/cute_ext.cuh
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto reverse(cute::Layout<Shape, Stride> const &l);

template <class A, class O, class B>
CUTE_HOST_DEVICE constexpr auto
reverse(cute::ComposedLayout<A, O, B> const &l);
```
<!-- /generated -->

**Terms.** *CuTe order* is CuTe's own reading of a layout, with mode zero fastest. A mesh is written row-major, with the last mesh axis fastest. The two orders are reverses of each other. CuTe's algebra reads a mesh backwards until the mesh is reversed.

- constraints:
  - CuTe reads mode zero as the fastest. A mesh is written the other way round. Every CuTe function that means "next to" therefore reads a mesh backwards. `coalesce` leaves `(2,32):(32,1)` unfolded: it is 64 consecutive threads. `logical_divide` cuts a domain that runs the other way. The runtime reverses the mesh axes once. After that it uses plain CuTe.

#### 2.3.2 `layout/mesh.cuh`

<!-- generated: layout-mesh -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/layout/mesh.cuh
template <class TLayout, TopologyLevel... Topos> struct Mesh {
    using layout_type = TLayout;
    static constexpr int topology_count;
    static constexpr TopologyLevel first_topology;
    TLayout layout;
    /// The topology flat axis ``I`` runs at. A mesh naming one topology
    /// answers its own topology for every axis. A mesh naming several
    /// groups its layout one nest per topology, and the nest holding
    /// flat axis ``I`` states its topology.
    template <int I>
    CUTE_HOST_DEVICE static constexpr TopologyLevel mesh_axis_topology();
};

template <class L, TopologyLevel... Topos, class Coord>
CUTE_HOST_DEVICE constexpr bool contains(Mesh<L, Topos...> const &mesh,
                                         Coord const &coord);

template <class L, TopologyLevel... Topos, class Coord>
CUTE_HOST_DEVICE constexpr auto mesh_coords(Mesh<L, Topos...> const &mesh,
                                            Coord const &coord);

template <class L, TopologyLevel... Topos>
CUTE_HOST_DEVICE constexpr bool is_warped(Mesh<L, Topos...> const &mesh);

template <class L, TopologyLevel... Topos>
CUTE_HOST_DEVICE constexpr auto as_warped(Mesh<L, Topos...> const &mesh);

template <TopologyLevel S, class Extents>
CUTE_HOST_DEVICE constexpr auto make_mesh(Extents const &extents);
```
<!-- /generated -->

**Terms.** The terms are those of [shard §0](../../../../../../docs/spec/shard.md#0-terms). A *slice* is a mesh whose layout has a constant term. It covers part of a topology, and starts at that term. A *warped* mesh is a thread mesh whose ids run a whole number of warps. Its *lane* mesh axis is one warp. Its *warp* mesh axis steps between warps.

- constraints:
  - A mesh is the topologies it names and a layout whose values are their ids. The IR `Mesh` carries the same two facts beside its mesh axis names ([shard §5](../../../../../../docs/spec/shard.md#5-mesh)). The unit count and shape of a mesh are `cute::size` and `cute::shape` of its layout. Both read through a slice on their own.
  - A mesh that names several topologies groups its mesh axes, one nest per topology. The layout maps a coord to the index of a unit over all topologies of the mesh. That index runs over `T`, the `topology_size` of each topology, row-major. The nest of one topology is `cute::get` of the layout at that topology. Its strides stay in that index. The unit id of a topology at an index is `cute::idx2crd` of the index over `T`, read at that topology. The outermost topology keeps the quotient, so its `topology_size` is not read. A mesh that names several topologies without that grouping is refused. No rule says which mesh axes belong to which topology. A slice of such a mesh counts each topology from the unit id of its first unit. `contains` and `mesh_coords` read such a slice. The CUDA mesh emission does not emit it, and refuses it by name.
  - `mesh_coords` gives the coord of this unit in the mesh. It holds one `idx2crd` per topology. Each one takes the id of that topology less its id at the first unit. It multiplies that by the stride of the topology in `T` and reads it against the nest of that topology. The grouping is the grouping of the mesh. The result is congruent with `cute::shape` of the mesh layout, whether the mesh names one topology or several. A layout over that shape therefore indexes with it as it stands. It reads `unit_ids()`, one id for each topology that this program names. A coord of another length is refused by name.
  - `contains` and `mesh_coords` ask different questions. On a mesh narrower than its topology, they also give different answers. `idx2crd` is `(id / stride) % extent`. Thread 64 of a 128-thread block is therefore not in a 32-unit mesh. It does act as unit 0 of that mesh. The upper warps repeat what the lowest warp does. A slice does not repeat. A coord from outside a slice that reaches `mesh_coords` is a codegen fault. `contains` asks every topology that the mesh names, each against its own nest. A mesh with one nest per topology is one unit per topology together.
  - A thread mesh has warps only when its ids run a whole number of warps. `(32,..)` stepping by one is one warp. The mesh axis beyond it carries the warps. `(128,..)` holds four warps, and `as_warped` splits them out. Lanes shorter than a warp, repeated across warps, MUST be refused by name. Such lanes reach neither `bar.sync`, which counts whole warps, nor one `__syncwarp`.
  - `as_warped` is idempotent: a mesh whose fastest axis is already one warp comes back unchanged.
  - Only one shape of `cute::ComposedLayout` is a mesh: `cute::identity` over a static offset. A swizzle in the first slot, or a dynamic offset, gives a first unit that is not a compile-time number. `ops::sync` names its threads by the index of the first unit, so it refuses such a mesh by name.
  - The offset MUST NOT be dropped. A mesh layout maps a coord to a unit *within* the mesh. Only the offset turns that into a unit of the launch. A coord read off a raw id gives every unit of a slice the box that its neighbour owns. A plain `cute::Layout` is the whole topology at offset zero.

#### 2.3.3 `layout/shard_layout.cuh`

<!-- generated: layout-shard-layout -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/layout/shard_layout.cuh
template <class TLayout, class TAttrs, class TMesh> struct ShardLayout {
    using layout = TLayout;
    using attrs = TAttrs;
    using mesh = TMesh;
    TLayout layout_value;
    TMesh mesh_value;
};

template <int Axis> struct S {
    static constexpr int axis;
};

struct B {};

template <class Reduction> struct P {
    using reduction = Reduction;
};

template <size_t Ax, class L, class A, class M>
CUTE_HOST_DEVICE constexpr auto stride(ShardLayout<L, A, M> const &sl);

template <class SL> CUTE_HOST_DEVICE constexpr void check_shard_layout();

template <class Shape, class TMesh, class Attrs,
          __CUTE_REQUIRES(!cute::is_layout<Shape>::value)>
CUTE_HOST_DEVICE constexpr auto
make_shard_layout(Shape const &shape, TMesh const &mesh, Attrs const &);

template <class TLayout, class TMesh, class Attrs,
          __CUTE_REQUIRES(cute::is_layout<TLayout>::value)>
CUTE_HOST_DEVICE constexpr auto
make_shard_layout(TLayout const &layout, TMesh const &mesh, Attrs const &);
```
<!-- /generated -->

**Terms.** The terms are those of [shard §0](../../../../../../docs/spec/shard.md#0-terms). An attr says what its mesh axis does with the tensor. `S<k>` splits tensor axis `k` across it. `B` broadcasts. `P<R>` leaves an unreduced partial.

- constraints:
  - `TAttrs` is a `cute::tuple` of empty tags. The aggregate holds no attribute value. The attrs are in the type. The layout carries the data.
  - One attr per mesh axis. A shard layout that says nothing about an axis leaves the reader to guess whether that axis splits the tensor.

### 2.4 `cuda/tensor_view/`

#### 2.4.1 `tensor_view/shard_tensor.cuh`

<!-- generated: tensor-view-shard-tensor -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/tensor_view/shard_tensor.cuh
template <class TEngine, class TGlobalLayout, class TShardLayout>
struct ShardTensor {
    using engine_type = TEngine;
    using global_layout_type = TGlobalLayout;
    using shard_layout_type = TShardLayout;
    TEngine engine;
    TShardLayout shard_layout;
    /// As ``cute::Tensor`` has it: the engine's own pointer, before any
    /// unit's slice is projected out of it.
    CUTE_HOST_DEVICE auto data();
    CUTE_HOST_DEVICE auto data() const;
};

template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto make_shard_tensor(T const &tensor, GL, SL shard_layout);

template <class T>
concept ShardTensorLike =
    detail::is_shard_tensor<cute::remove_cvref_t<T>>::value;

template <class SL>
inline constexpr bool shard_layout_is_composed_v;

template <class T> CUTE_HOST_DEVICE decltype(auto) local_tensor(T &&t);

template <class T>
using local_view_t =
    cute::remove_cvref_t<decltype(local_tensor(std::declval<T const &>()))>;

template <class T> CUTE_HOST_DEVICE constexpr int shard_mesh_units();
```
<!-- /generated -->

**Terms.** A *shard tensor* is an engine, the global layout it came from, and a shard layout. *Projecting* a shard tensor resolves it to the slice that this unit owns. `local_tensor` projects. `local_view_t` is its type-level half. `ShardTensorLike` is the one test for "is this operand sharded". All three are `tilefoundry::` names, not `detail` names.

- constraints:
  - `engine` is a CuTe tensor or view, never a raw pointer. Residency lives on the engine type, and `data()` drops it. Its layout MUST be the declared concrete local shard layout. For rmem, it MUST use the shared register numbering. Codegen MUST NOT choose a separate default engine order. Storage spans the layout codomain, including any pitch. A rank-0 address is encoded in CuTe as one singleton mode with stride zero.
  - A [`Swizzle`](../../../../../../docs/spec/shard.md#41-swizzle) reaches the device as a shard layout whose layout is a `cute::ComposedLayout`. Projection keeps the function of such a layout. `shard_layout_is_composed_v` picks that path. `local_tensor` returns a tensor over the own pointer of the engine. Its layout is the same composition, with the origin of this unit folded into its offset. The offset MUST NOT move to the engine. Advancing the pointer would run the function on a slice-relative index and give two units one address.
  - A composed layout takes that path even when no mesh axis splits it. The composed path reconstructs the declared function and the unit offset over the engine pointer, including when every mesh attribute broadcasts.
  - `local_tensor`, `local_view_t` and `ShardTensorLike` are public. A step that every op takes is not an implementation detail. The word that an op writes its constraints in is not one either. `is_shard_tensor` decides `ShardTensorLike`. Nothing outside needs it, so it stays in `detail`.
  - A tensor whose layout is a `ShardLayout` has distributed semantics. It is the whole tensor. Each unit owns the slice that its shard layout gives it.
  - A shard layout has one attr per mesh axis, counted over the flattened mesh axes. A mesh that names several topologies groups its mesh axes, one nest per topology. The rank of that mesh is the number of topologies it names, not the number of mesh axes that the attrs cover.
  - Several mesh axes MAY cut one tensor axis. They can be two mesh axes of one topology, as a grid. They can be two topologies, where one takes a block of what the other left. Each cut divides what the previous cut left. The local extent of the tensor axis is the whole extent divided by all cuts. One step of a mesh axis clears everything that the mesh axes inside it hold.
  - The origin of the slice of a unit is read from a value that has the shape of the mesh. On each mesh axis, it holds how far one step moves the origin along the tensor. The attrs say which tensor axis a step clears, and how much of it. The tensor layout says what that costs in elements. The value is congruent with the shape of the mesh, grouping included. `mesh_coords` therefore indexes it as it stands, and reading it is `crd2idx`. The value is not the index `layout(0)` of the first unit that a sliced mesh covers.

### 2.5 `cuda/primitive/`

#### 2.5.1 `primitive/unary.h`

<!-- generated: primitive-unary -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/primitive/unary.h
struct rsqrt_op {
    template <class T> __device__ T operator()(T x) const;
};

struct neg_op {
    template <class T> __device__ T operator()(T x) const;
};

struct relu_op {
    template <class T> __device__ T operator()(T x) const;
};

struct square_op {
    template <class T> __device__ T operator()(T x) const;
};

struct sigmoid_op {
    template <class T> __device__ T operator()(T x) const;
};

struct silu_op {
    template <class T> __device__ T operator()(T x) const;
};

struct softplus_op {
    template <class T> __device__ T operator()(T x) const;
};

struct exp_op {
    template <class T> __device__ T operator()(T x) const;
};

struct log_op {
    template <class T> __device__ T operator()(T x) const;
};

struct identity_op {
    template <class T> __device__ T operator()(T x) const;
};

struct clamp_op {
    float min_val, max_val;
    template <class T> __device__ T operator()(T x) const;
};
```
<!-- /generated -->

**Terms.** A *primitive* is a callable tag an op instantiates. It takes scalars, not tensors, which is why it is not an op.

- constraints:
  - A primitive takes scalars, not tensors, so it is not an op. It has no shard layout to read a behaviour from. `ops::` closes on the entries that have one ([§2.6](#26-cudaops)).

#### 2.5.2 `primitive/binary.h`

<!-- generated: primitive-binary -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/primitive/binary.h
struct mul_op {
    template <class T> __device__ T operator()(T a, T b) const;
};

struct add_op {
    template <class T> __device__ T operator()(T a, T b) const;
};

struct sub_op {
    template <class T> __device__ T operator()(T a, T b) const;
};

struct max_op {
    __device__ float operator()(float a, float b) const;
    template <class T> __device__ T operator()(T a, T b) const;
};

struct min_op {
    __device__ float operator()(float a, float b) const;
    template <class T> __device__ T operator()(T a, T b) const;
};

struct div_op {
    template <class T> __device__ T operator()(T a, T b) const;
};
```
<!-- /generated -->

- constraints: none beyond the signature.

### 2.6 `cuda/ops/`

An op derives its geometry from `ShardTensor` or `Mesh` operands. An op is one indivisible dependency chain or one per-element map. Composing two ops does not create another entry. `tilefoundry::ops` contains only:

| entry | geometry source |
| --- | --- |
| `elementwise` | destination local domain; stride-zero source broadcasts |
| `copy` / `copy_async` | operand shapes, strides, shares and move width |
| `reduce` | destination-broadcast axes split by the source |
| `dot` | operand meshes' contraction axes |
| `mma` | one atom's per-lane fragments |
| `sync` | mesh scope, base and count |
| `copy_async_bulk` | contiguous whole tiles with matching element types |
| `ldmatrix` | declared shared-memory tile layout and destination fragment |

- constraints:
  - Each family MUST expose one entry. Derive implementation tiers at compile time inside that entry, from operand layouts and static geometry. Codegen MUST NOT select a tier or record it on TIR ([codegen §3](./codegen.md#3-runtime-owned-op-dispatch)). Internal optimized paths MAY differ without changing the result or public entry.
  - Publish no predicate or constant that reports a selected tier or move width. Such an answer needs the operands to exist, so it cannot help to construct them.
  - `elementwise` alone covers fill, cast, clamp, activation, arithmetic and their broadcasts. Its source pack, callable and stride-zero modes already state these choices. Separate `fill`, `cast`, `unary`, `binary` or broadcast entries would duplicate them.
  - Raw-pointer, scalar or type queries are utilities outside `ops::`.
    Warp primitives ([§2.7](#27-cudautility)) and individual `mbarrier`
    instructions have no operand layout from which to derive a tier.
  - Public entries, implementation functors and op tags MUST be `__device__`.
    Reserve `CUTE_HOST_DEVICE` for layout / tensor-view helpers genuinely
    capable of host compilation, such as `local_tensor` and `make_shard_tensor`.

#### 2.6.1 `ops/detail.cuh`

<!-- generated: ops-detail -->
```text
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/ops/detail.cuh
// This header declares no surface of its own.
```
<!-- /generated -->

**Terms.** `ops::detail` is the set of tensor-view *implementation* names that an op may reach for: `id_axes`, `is_partial_attr_v`, `is_shard_tensor`, `is_split_attr_v`, `plain_layout`, `shard_attrs_match_mesh`, `shard_layout_is_full_broadcast`. They are defined in `tilefoundry::detail`. This header states which of them cross into `ops`. *Projecting* an operand is `local_tensor`. It resolves a ShardTensor to the slice of this unit, and it returns whole an operand that the mesh never spread. The projecting function is not one of those names. It, `local_view_t` and `ShardTensorLike` are `tilefoundry::` names. An op calls them without the `detail` qualifier ([§2.4.1](#241-tensor_viewshard_tensorcuh)).

- constraints:
  - An op reaches for a tensor-view *implementation* name through `ops::detail` and nowhere else. One header therefore states what crosses that boundary. The gate holds only names that an op may want and no op must have. A step that no op may skip is public, because a thing that every op goes through is not an implementation detail.
  - An op projects every operand through `tilefoundry::local_tensor`, sharded or not. An operand with no shard layout is not refused. Every unit holds it entire, and the arithmetic of the op over it is the same. The projection makes this choice once, not each op.

#### 2.6.2 `ops/copy.cuh`

<!-- generated: ops-copy -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/ops/copy.cuh
template <class TPointer, class TLayout>
CUTE_HOST_DEVICE auto tensor_view(TPointer pointer, TLayout layout);

template <class TSrc, class TDst>
__device__ void copy(TSrc const &src, TDst &dst);

template <class TSrc, class TDst>
__device__ void copy_async(TSrc const &src, TDst &dst);

template <class ExecutionMesh, int Bytes, class TSrc, class TDst, class TIndex>
__device__ void copy_async(TSrc const &src, TDst &dst, TIndex const &index);

template <class ExecutionMesh, int Bytes, class TSrc, class TDst, class TIndex,
          class TFill>
__device__ void copy_async(TSrc const &src, TDst &dst, TIndex const &index,
                           TFill fill);
```
<!-- /generated -->

For an RMEM `ShardLayout`, the shared `ir.types.register_strides` reader MUST assign compact register-engine steps to non-Split, non-unit modes. Codegen and Dot access relations use this reader. The steps MUST follow the ascending order of the declared physical strides. Producers supply concrete strides before codegen. Split and unit modes have engine step zero. This numbering is independent of mode order in the logical-domain correspondence.

**`tensor_view`.** Rebuilds the CuTe tensor engine that a TIR `TensorView`
describes from its typed pointer and emitted layout. Residency stays on the
pointer engine; a later `make_shard_tensor` adds the logical global and shard
layouts without recovering the source tensor.

**`copy`.**

The entry takes tensors. It reads the transfer shape, the strides, the element types, the share that this unit owns and the move width from the operand `ShardLayout`s.

- constraints:
  - **The move width is a property of the layouts, not of the call.** The move width is the largest power of two over which both operands run contiguously. A cap keeps the move of each side within 16 bytes. The width is `max_common_vector` of the two *projected* layouts. [§2.4.1](#241-tensor_viewshard_tensorcuh) derives those layouts statically from the shard layout and the mesh.
  - A dtype change does not force the width to one element. The wide load still holds, and the conversion happens on the way out. The cap follows the wider of the two element types.
  - Register fragments take the element path. Addressing `&frag(i)` under a loop that the compiler cannot unroll spills the fragment to local memory. That costs more than the wider move saves.
  - A layout cannot state where the offset of the shard lands. The implementation therefore tests the two base pointers against the alignment of the width. It falls back to the element path when the test fails. That is one comparison, not a scan.
  - The call site names no vector width and no thread index. A copy split across the threads of a block is a thread-scoped mesh in the shard layouts of the operands. The entry then copies exactly the slice that `local_tensor` gives each thread.

**`copy_async`.**

- constraints:
  - The call is non-blocking; generated code orders later reads through
    `cp_async_commit` and `cp_async_wait`.
  - The name is the contract. If `cp.async` cannot move a pair of projected views in words of 4 bytes or more, the pair does not compile. Such a pair takes `ops::copy` instead. This entry has no hidden synchronous tier for the caller to wait on.
  - The two projected slices hold the same number of elements. When a unit owns part of a wider destination, the shard layout of the destination states that part. The runtime does not borrow the offset of the source for it.
  - Runtime implementation details such as vector width and tail handling live
    in code comments, not this spec entry.

#### 2.6.3 `ops/dot.cuh`

<!-- generated: ops-dot -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/ops/dot.cuh
template <class Lhs, class Rhs, class Dst,
          class Ws = reduce_impl::no_workspace_t>
__device__ inline void dot(Lhs const &lhs, Rhs const &rhs, Dst &dst,
                           Ws &&ws = {});
```
<!-- /generated -->

**`dot`.** `dst = sum(lhs * rhs)` without materializing a product tensor,
which would cost one register per row element.

- constraints:
  - Without workspace, one warp's butterfly finishes the contraction. With workspace, warps post block-wide partials. Every unit gets the total.
  - Derive load width from both projected layouts, capped by alignment and 16 bytes, as for `copy`. Publish no width or tier query ([§2.6](#26-cudaops)).
  - Use a fixed number of independent partial accumulators and a fixed combining tree. One dependent sum would serialize loads behind arithmetic. Accumulation is f32 regardless of operand type.

#### 2.6.4 `ops/elementwise.cuh`

<!-- generated: ops-elementwise -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/ops/elementwise.cuh
template <class Fn, class TOut, class... TIn>
__device__ void elementwise(TOut &dst, Fn fn, TIn const &...src);
```
<!-- /generated -->

**`elementwise`.** `dst(i) = fn(src(i)...)` over the destination's local
domain. A broadcast is a stride-zero mode on the source's own layout.

- constraints:
  - `dst` and `fn` precede the variadic sources. Zero sources fills, one maps,
    and two combine; all use the same entry.
  - `fn` MAY be any callable, including the primitive values that `reduce` and `dot` also use. There is no separate tag path and lambda path.
  - Derive extent from `size()` of the projected destination. A count parameter
    would duplicate a layout fact and could disagree with it.
  - Do not check shape or domain at runtime: mismatches are codegen bugs,
    because the compile side computes shapes.
  - Assignment performs the only implicit conversion. The runtime MUST NOT insert casts or choose compute precision. Callers state casts in `fn`, matching TIR `Cast` semantics.
  - Read broadcasts through the own layout of each operand, including strided or sharded operands. Composition MUST preserve where its values reside.

#### 2.6.5 `ops/mma.cuh`

<!-- generated: ops-mma -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/ops/mma.cuh
template <class TA, class TB, class TC>
__device__ void mma(TA const &a, TB const &b, TC &c);
```
<!-- /generated -->

**`mma`.**

`c += a @ b` for one atom. The operands are the already-gathered fragments of the calling lane. The lane executes one instruction through the entry.

- constraints:
  - The element type of `a` selects the atom: bf16 selects
    `SM80_16x8x16_F32BF16BF16F32_TN` and e4m3 selects
    `SM89_16x8x32_F32E4M3E4M3F32_TN`, both accumulating in f32.
  - `b` holds the element type of `a`. `c` holds f32. `a`, `b` and `c` contain exactly the values of that atom per lane: `(8, 4, 4)` for bf16 and `(16, 8, 4)` for e4m3. Other shapes and element types are compile-time errors.
  - The fragment map is stated by each operand's `ShardLayout`; the runtime
    publishes no fragment-coordinate function and no accumulator constructor.
  - Another instruction is another atom under the same entry, not another
    entry.

#### 2.6.6 `ops/reduce.cuh`

<!-- generated: ops-reduce -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/ops/reduce.cuh
struct mean_op {};

struct absmax_op {};

template <class Op, class Axes, class Src, class Dst,
          class Ws = reduce_impl::no_workspace_t>
__device__ inline void reduce(Src const &src, Dst &dst, Ws &&ws = {});
```
<!-- /generated -->

**`reduce`.**

- constraints:
  - `reduce` is the only public runtime reduce entry; tier names and helper
    functions are internal.
  - Sharded operands derive the active tier and warp grouping from `(src, dst)` shard layouts inside the runtime. `ws` is a *resource the layouts demand*. It is never the tier selector. A reduce mesh that stays inside one warp takes the intra-warp tier and needs no workspace. A reduce mesh that crosses warps does not compile without a workspace.
  - Both operands name one mesh and give one attr per mesh axis; the reduced
    set is `src[i]` split or partial against `dst[i]` broadcast.
  - The combine domain is `float`. `init`, `elem`, `combine` and `finalize` are all in it. The entry refuses an element type that `float` cannot hold exactly, such as a 4- or 8-byte integer or a packed key. It does not round that type.
  - Plain operands derive extents from the operand rank and size inside the
    runtime.
  - A reduction whose reduced axis crosses CTA boundaries is not supported.

#### 2.6.7 `ops/sync.cuh`

<!-- generated: ops-sync -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/ops/sync.cuh
template <int Id> struct BarrierId {
    static constexpr int value;
};

template <int Id> inline constexpr BarrierId<Id> bar_id;

struct no_resource_t {};

template <class T> struct barrier_id_traits {
    static constexpr bool value;
};

template <class TMesh, TopologyLevel... Topos, class TRes = no_resource_t>
__device__ inline void sync(Mesh<TMesh, Topos...> const &mesh,
                            TRes resource = {});
```
<!-- /generated -->

**`sync`.** The mesh selects a grid, block, warp or named barrier. A mesh that reaches none MUST be refused explicitly, not approximated.

- constraints:
  - Derive the tier from topology scope, layout size and slice offset ([§2.3.2](#232-layoutmeshcuh)). Never derive it from a call-site template argument. Classification MUST agree with `tilefoundry.ir.tir.sync.classify`. Unsliced `cta` meshes MUST use grid barriers. Whole blocks MUST use block barriers, or warp convergence for one-warp blocks. Subsets inside one warp MUST use convergence. Aligned cross-warp subsets MUST use named barriers.
  - Reject sliced `cta` meshes, because absent CTAs never arrive. Reject cross-warp subsets whose base or count is not divisible by 32, because partial-warp participation deadlocks.
  - Allocate no resources. Missing or incorrect resources MUST cause a compile error that names the requirement. A grid counter belongs to the launch. A free named barrier belongs to the whole kernel. One mesh cannot give either.
  - Named ids MUST be compile-time values in `1..15`. Reject `0`, reserved by
    `__syncthreads`, because subset arrivals could release that barrier early.
    Do not derive ids from meshes: equal bases with different counts collide.
  - Codegen emits only `sync`, never lower-level barrier helpers.
  - Grid barriers require every launch CTA to be co-resident and participate. Their module-owned counter pair MUST start at zero. The caller supplies the counter pair. Without counters, use the grid group of a cooperative launch.

#### 2.6.8 `ops/copy_async_bulk.cuh`

<!-- generated: ops-copy-async-bulk -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/ops/copy_async_bulk.cuh
template <class Src, class Dst>
__device__ inline void copy_async_bulk(Src const &src, Dst &dst,
                                       uint64_t *bar);
```
<!-- /generated -->

**`copy_async_bulk`.** Stages a tile into shared memory and signals an mbarrier when the tile is readable. Each thread executes `copy_async` for itself. A consumer of this entry can wait for data that another thread fetched.

- constraints:
  - Take tensors, not addresses or byte counts. Shape, strides and element types come from operand layouts ([§2.6](#26-cudaops)).
  - One elected thread executes nonblocking `cp.async.bulk`. Off-grain byte counts hand off at runtime to an element path executed by every block thread. Both paths complete the same barrier. Both paths are safe to call from every thread. Consumers only wait on phase.
  - Both paths MUST enforce these `static_assert`s. Mesh axes MUST leave both tiles whole. The destination MUST be a `ShardTensor`. Projections MUST be contiguous. Element types and slice sizes MUST match. The element path needs destination mesh participation. It cannot correctly address noncontiguous layouts. A layout that fails these checks MUST be refused, not routed to that path.
  - `bar` is a 64-bit shared-memory resource, not a sharded tensor. Individual arming, phase-test and release instructions are not ops. They have no runtime entries. Callers place them where they build the ring. For stage `t` of an `n`-stage ring, wait on phase `(t / n) & 1` so barriers can be reused.
  - Bulk arrival declares bytes on the copy instruction. The caller MUST NOT arrive separately. This keeps declared and delivered counts identical.
  - The instruction requires whole 16-byte grains. Projected extent is a runtime value. Off-grain transfers therefore use the element path behind the same entry, barrier and result. A compile-time tier choice does not select that path.

#### 2.6.9 `ops/ldmatrix.cuh`

<!-- generated: ops-ldmatrix -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/ops/ldmatrix.cuh
template <class Src, class Dst>
__device__ inline void ldmatrix(Src const &src, Dst &dst);
```
<!-- /generated -->

**`ldmatrix`.**

One warp loads a dense shared-memory `(16, 16)` bf16 tile into the per-lane A
fragment consumed by `ops::mma`'s SM80 atom.

- constraints:
  - All 32 lanes execute `ldmatrix.sync.aligned.m8n8.x4.shared.b16` together. Each lane contributes the 16-byte-aligned address that the declared shard layout of the source selects. A full-broadcast shard still uses that declaration. It does not use the incidental layout of the backing allocation.
  - The four returned registers are stored in the destination fragment's
    layout order so that `ops::mma` observes the PTX register order unchanged.
  - The TIR declaration fixes the source shape/dtype and destination fragment;
    the runtime entry does not choose an atom or a fragment layout.

### 2.7 `cuda/utility/`

Warp primitives operate on computed scalar values and lane participation,
not tensors or meshes, so they have no layout-selected tiers:

| entry | input |
| --- | --- |
| `shuffle_xor` | one lane's value, exchange bits and the lanes that execute |
| `shuffle_elect` | calling thread's id |
| `warp_reduce` | one lane's value, combine functor and lanes per fold |

A candidate that takes a tensor or mesh belongs in the closed op list
([§2.6](#26-cudaops)). A parameter that the hardware clamps to one value is not a
public choice. `ops::reduce` uses `warp_reduce` for its intra-warp tier, which avoids a duplicate butterfly. Scalar callers can use the primitive directly.
`utility/warp.cuh` MUST therefore precede `ops/reduce.cuh`.

#### 2.7.1 `utility/warp.cuh`

<!-- generated: utility-warp -->
```cpp
// targets/nvidia/tilefoundry_nvidia/cuda/codegen/include/tilefoundry/runtime/cuda/utility/warp.cuh
template <class T>
inline constexpr bool is_shuffle_native_v;

template <class T> struct ShuffleXor {
    __device__ T operator()(T value, int lane_mask,
    unsigned member_mask) const;
};

struct Elect {
    __device__ bool operator()() const;
};

template <class T>
__device__ inline T shuffle_xor(T value, int lane_mask,
                                unsigned member_mask = 0xFFFFFFFFu);

inline bool shuffle_elect();

template <class Combine, int Width = 32, class T>
__device__ inline T warp_reduce(T value);
```
<!-- /generated -->

**`shuffle_xor`.**

- constraints:
  - The exchange is warp-scoped: a lane's result is defined only where the lane
    it pairs with is in `member_mask`.
  - A `T` that the shuffle intrinsic does not take directly is exchanged word-wise. Such a type MUST fail to compile when its size is not a whole number of 4-byte words. Which types the intrinsic takes is a code-level fact. It lives in code comments, not this entry.

**`shuffle_elect`.**

- constraints:
  - It elects one thread of the **CTA**, not one lane of each warp. An mbarrier
    armed for a single arrival is correct only under that reading.
  - **It takes no width and no count of the threads that execute.** The elected thread is always in the first warp. A leading run wider than a warp elects what 32 threads would elect. Electing among fewer threads is a different instruction, not a parameter of this one.
  - The election asks the linearised thread id ([§2.2.1](#221-cudaruntimecuh)), not `threadIdx.x`. Ids `0..31` are therefore the first warp, whatever shape the block has. A thread outside that run answers `false` without reaching the warp-scoped election. That election defines its result only for the lanes in its member mask.

**`warp_reduce`.**

- constraints:
  - A butterfly over `Width` lanes takes `log2(Width)` shuffles. It leaves the result of the run in **every** lane of the run, not only in its first lane.
  - `Width` MUST be a power of two in `2..32`; a violation is a
    `static_assert`, not a run-time check.
  - A `Width` under 32 leaves the runs independent. A per-row reduction over a tile with several rows to a warp needs this.
  - `Combine` is invoked as `Combine{}(a, b)`. `ops::elementwise` and `ops::reduce` ([§2.6](#26-cudaops)) use the same spelling. A merge is therefore defined once for all three. The op tags are valid `Combine` values as they stand.
