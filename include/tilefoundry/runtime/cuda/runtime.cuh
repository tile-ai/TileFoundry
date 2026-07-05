// tilefoundry runtime — thin wrapper around CuTe (spec 010 §5 / §6).
//
// Provides our own `tilefoundry::Mesh` / `tilefoundry::TopologyScope` /
// `tilefoundry::ShardLayout` / `tilefoundry::ShardTensor` template surface.
// Re-exports CuTe primitives (`cute::copy`, `cute::make_tensor`, etc.)
// so codegen can emit real CuTe calls.

#pragma once

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cstdint>
#include <cuda_fp8.h>

namespace tilefoundry {

enum class TopologyScope {
    cta,
    warp,
    thread,
    scope_count, // sentinel; not a real topology level
};

// program_id<T>(): backend-tag-dispatched runtime query for the linearized
// scalar id of the current execution instance within topology `T`.
//
// The CUDA backend reads the corresponding hardware built-in (`blockIdx`
// for `cta`, `threadIdx` for `thread`) and linearizes that level's own
// multi-dimensional coordinate to a scalar — it does **not** mix with
// other topology levels (cf. nncase ntt::distributed::program_id_getter).
//
// Primary template is intentionally undefined; backend-specific
// specializations must be provided.
template <TopologyScope T> CUTE_HOST_DEVICE size_t program_id() noexcept;

template <> CUTE_HOST_DEVICE size_t program_id<TopologyScope::cta>() noexcept {
#if defined(__CUDA_ARCH__)
    return size_t(blockIdx.x) + size_t(blockIdx.y) * size_t(gridDim.x) +
           size_t(blockIdx.z) * size_t(gridDim.x) * size_t(gridDim.y);
#else
    return 0;
#endif
}

template <>
CUTE_HOST_DEVICE size_t program_id<TopologyScope::thread>() noexcept {
#if defined(__CUDA_ARCH__)
    return size_t(threadIdx.x) + size_t(threadIdx.y) * size_t(blockDim.x) +
           size_t(threadIdx.z) * size_t(blockDim.x) * size_t(blockDim.y);
#else
    return 0;
#endif
}

// the ``warp`` level lives between ``cta`` and ``thread`` in the CUDA
// hierarchy. A warp is 32 lanes, so a thread's warp id is its linear thread
// index in the CTA divided by 32. ``warp`` is not a program topology level —
// it is expressed inside a mesh layout — so no ``program_shape<warp>`` is
// emitted; only this ``program_id<warp>()`` accessor exists.
template <> CUTE_HOST_DEVICE size_t program_id<TopologyScope::warp>() noexcept {
#if defined(__CUDA_ARCH__)
    return size_t(threadIdx.x) + size_t(threadIdx.y) * size_t(blockDim.x) +
           size_t(threadIdx.z) * size_t(blockDim.x) * size_t(blockDim.y);
#else
    return 0;
#endif
}

// program_shape<T>(): compile-time shape of topology level `T`. The
// primary template is intentionally undefined; codegen injects a
// per-module specialization (one per topology level used by the
// module) that returns a `cute::Shape<cute::Int<...>...>`.
template <TopologyScope T> constexpr auto program_shape() noexcept;

// program_dim<T>() = size of topology level `T`. For a static level this folds
// to the compile-time product of ``program_shape<T>()``. For a launch-provided
// (dynamic) ``cta`` level, codegen injects a non-constexpr specialization that
// reads the count from ``gridDim`` at runtime (no ``program_shape<cta>`` exists
// in that case).
template <TopologyScope T> constexpr auto program_dim() noexcept {
    return cute::size(program_shape<T>());
}

// Topology is parameterised by its scope + total size at compile time.
template <TopologyScope Scope, int Size> struct Topology {
    static constexpr TopologyScope scope = Scope;
    static constexpr int size = Size;
};

// Mesh<topology, cute_layout>: binds a topology to a MeshLayout.
//
// ``layout_value`` carries the mesh layout as a runtime value so a mesh
// axis with a dynamic (runtime ``int``) extent — e.g. a launch-provided CTA
// count — holds its real value. For a fully static mesh the CuTe layout type
// is empty, so the member is zero-cost and reads back the same static extent.
template <class TTopo, class TMeshLayout> struct Mesh {
    using topology = TTopo;
    using layout = TMeshLayout;
    TMeshLayout layout_value;
};

// ShardLayout<layout, attrs_tuple, mesh>: spec 003 shard layout surface.
// `TAttrs` encodes per-axis ShardAttr via a cute::tuple of tag types.
template <class TLayout, class TAttrs, class TMesh> struct ShardLayout {
    using layout = TLayout;
    using attrs = TAttrs;
    using mesh = TMesh;
    // Runtime values for the global per-axis layout and the mesh. A dynamic
    // global dim / mesh extent holds its real runtime value here; static
    // CuTe layouts are empty types so these members are zero-cost.
    TLayout layout_value;
    TMesh mesh_value;
};

// Per-axis shard attributes.
namespace shard {
template <int Axis> struct S {
    static constexpr int axis = Axis;
};
struct B {}; // Broadcast
template <class Reduction> struct P {
    using reduction = Reduction;
};
struct Dynamic {};
} // namespace shard

// Legacy aliases in tilefoundry:: for backward compat during migration.
using shard::B;
using shard::Dynamic;
using shard::P;
using shard::S;

// ShardTensor<Engine, GlobalLayout, TShardLayout> — descriptor that
// interprets a global memory region through a shard layout. Does not
// own data; the engine (pointer) is the only stored state.
template <class TEngine, class TGlobalLayout, class TShardLayout>
struct ShardTensor {
    using engine_type = TEngine;
    using global_layout_type = TGlobalLayout;
    using shard_layout_type = TShardLayout;
    TEngine engine;
    // Runtime shard-layout value; ``local`` reads per-axis shape / stride and
    // the mesh extent from here so dynamic dims carry their real value. Zero
    // storage when every layout dim is a static CuTe ``Int``.
    TShardLayout shard_layout;

    // Underlying register/memory pointer of the wrapped cute tensor. Mirrors
    // ``cute::Tensor::data()`` so callers can treat a ShardTensor and a plain
    // cute tensor uniformly. NOTE: a raw pointer drops the gmem/smem/rmem
    // residency tag (which lives on the engine *type*), so ``data()`` is only
    // for sites where residency no longer matters (e.g. the per-thread MMA
    // register fragment). Residency-aware paths use ``local()`` instead.
    CUTE_HOST_DEVICE auto data() { return engine.data(); }
    CUTE_HOST_DEVICE auto data() const { return engine.data(); }
};

// make_shard_tensor(tensor, global_layout, shard_layout) → ShardTensor.
// The engine T must be a CuTe tensor/view with a tagged engine category
// (gmem / smem / rmem) — raw pointers lose the tag and will be rejected
// at compile time.
template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto make_shard_tensor(T const &tensor, GL /*global_layout*/,
                                        SL shard_layout) {
    using engine_t = cute::remove_cvref_t<T>;
    // Reject raw pointers: they have no CuTe engine category, and
    // cute::is_rmem<T*>::value is true (raw pointers are neither
    // gmem nor smem), so a gmem/smem/rmem check would pass them.
    static_assert(
        !std::is_pointer_v<engine_t>,
        "ShardTensor engine must be a CuTe tensor/view, not a raw pointer");
    return ShardTensor<T, GL, SL>{tensor, shard_layout};
}

// Full-broadcast case: empty attrs or rank mismatch — return original tensor
template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto local_impl(ShardTensor<T, GL, SL> const &st,
                                 std::true_type /*full_broadcast*/) {
    return st.engine;
}

// Slicing case: per spec §2.10, build the per-instance view as
//   offset = Σ_{m : attrs[m]=Split(k)}  coord[m] · S[k]
//   shape' = shard_layout_local_shape(sl)
//   return make_tensor(engine.data() + offset, Layout(shape', S))
// where ``S`` is the storage-physical strides emitted by typeinfer
// (spec shard §7.1.2) — consumed verbatim, no per_shard rescale.
//
// Spec: docs/spec/runtime.md §2.10
template <class T, class GL, class SL>
CUTE_HOST_DEVICE auto local_impl(ShardTensor<T, GL, SL> const &st,
                                 std::false_type /*full_broadcast*/) {
    using mesh_t = typename SL::mesh;
    using topo_t = typename mesh_t::topology;
    constexpr auto scope = topo_t::scope;
    using attrs_t = typename SL::attrs;
    using sl_layout_t = typename SL::layout;
    using m_layout_t = typename mesh_t::layout;

    // Read the per-axis shape / stride and the mesh extent from the runtime
    // ShardTensor value (not a default-constructed type): a dynamic dim
    // carries its real runtime ``int`` here, while a static ``cute::Int<N>``
    // reads back as ``N`` — so the static path is unchanged. Rank stays a
    // compile-time property of the layout types.
    auto const &sl_layout = st.shard_layout.layout_value;
    auto const &m_layout = st.shard_layout.mesh_value.layout_value;

    auto pid = program_id<scope>();
    // BUGFIX: use Layout::get_hier_coord
    // instead of cute::idx2crd(idx, layout) — the latter matches the
    // 2-arg overload that treats Layout as colexicographic shape
    // (ignoring C-order strides), producing wrong per-thread coords.
    auto crd = m_layout.get_hier_coord(pid);
    // ``remove_cvref_t``: ``cute::shape`` on a layout with a dynamic (runtime
    // ``int``) mode yields a reference; strip it so ``tuple_size`` (which needs
    // a bare type) sees the rank. A static layout already yields a value type,
    // so this is a no-op there.
    using sl_shape_t =
        cute::remove_cvref_t<decltype(cute::shape(sl_layout_t{}))>;
    using m_shape_t = cute::remove_cvref_t<decltype(cute::shape(m_layout_t{}))>;
    constexpr int t_rank = cute::tuple_size<sl_shape_t>::value;
    constexpr int m_rank = cute::tuple_size<m_shape_t>::value;

    // Global per-axis dims (= ShardLayout shape) and per-axis user-provided
    // strides — these are taken straight from the ShardLayout value, not
    // recomputed.
    auto const sl_shape = cute::shape(sl_layout);
    auto const sl_stride = cute::stride(sl_layout);
    auto const m_shape = cute::shape(m_layout);
    int g_dim[t_rank];
    int g_stride[t_rank];
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((g_dim[Is] = int(cute::get<Is>(sl_shape)),
          g_stride[Is] = int(cute::get<Is>(sl_stride))),
         ...);
    }(std::make_index_sequence<t_rank>{});

    int m_ext[m_rank], m_crd[m_rank];
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((m_ext[Is] = int(cute::get<Is>(m_shape)),
          m_crd[Is] = int(cute::get<Is>(crd))),
         ...);
    }(std::make_index_sequence<m_rank>{});

    // For each tensor axis, record which mesh axis (if any) splits it.
    int axis_to_mesh[t_rank];
    for (int i = 0; i < t_rank; ++i)
        axis_to_mesh[i] = -1;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        auto process = [&]<size_t I>(std::integral_constant<size_t, I>) {
            auto attr = cute::get<I>(attrs_t{});
            using A = decltype(attr);
            if constexpr (!std::is_same_v<A, shard::B>) {
                constexpr int ax = A::axis;
                axis_to_mesh[ax] = int(I);
            }
        };
        (process(std::integral_constant<size_t, Is>{}), ...);
    }(std::make_index_sequence<m_rank>{});

    // Walk axes once per spec §2.10: Split axes add ``coord[m] · S[k]`` to
    // offset (S is storage-physical, consumed verbatim) and contribute the
    // per-instance cute extent ``G[k] / M_a``; non-Split axes pass through.
    int loc_shape[t_rank];
    int loc_stride[t_rank];
    int offset = 0;
    for (int i = 0; i < t_rank; ++i) {
        int m = axis_to_mesh[i];
        if (m >= 0) {
            offset += m_crd[m] * g_stride[i];
            loc_shape[i] = g_dim[i] / m_ext[m];
            loc_stride[i] = g_stride[i];
        } else {
            loc_shape[i] = g_dim[i];
            loc_stride[i] = g_stride[i];
        }
    }
    // Materialise the per-thread cute layout / tensor. Building a dynamic
    // cute Layout (int-typed shape & stride) avoids having to specialise
    // the local rank at compile time.
    auto local_layout = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return cute::make_layout(cute::make_shape(loc_shape[Is]...),
                                 cute::make_stride(loc_stride[Is]...));
    }(std::make_index_sequence<t_rank>{});

    // ``cute::coalesce`` absorbs size-1 axis collapse and contiguous-stride
    // merging on the local layout. ``const_cast`` on the engine yields a
    // modifiable lvalue when the caller's ``dst`` is a non-const ShardTensor —
    // ``local_impl`` is a descriptor read, ``engine.data()`` returns a value
    // pointer either way.
    auto &engine_mut = const_cast<typename std::remove_const<
        typename std::remove_reference<decltype(st.engine)>::type>::type &>(
        st.engine);
    return cute::make_tensor(engine_mut.data() + offset,
                             cute::coalesce(local_layout));
}

// local(st) — dispatches via tag to full-broadcast or slicing impl.
// Spec: docs/spec/runtime.md §2.10
template <class T, class GL, class SL>
CUTE_HOST_DEVICE decltype(auto) local(ShardTensor<T, GL, SL> const &st) {
    // BUGFIX: engine.data() returns a raw pointer (e.g. T*) which cute
    // mis-classifies as rmem even for gmem tensors. Check the engine
    // type directly — cute gmem_ptr/smem_ptr carry the correct tag.
    using engine_t = cute::remove_cvref_t<decltype(st.engine)>;
    if constexpr (!cute::is_gmem<engine_t>::value &&
                  !cute::is_smem<engine_t>::value) {
        // rmem: the engine itself is already the per-thread view
        return const_cast<engine_t &>(st.engine);
    } else {
        using attrs_t = typename SL::attrs;
        using mesh_t = typename SL::mesh;
        using m_layout_t = typename mesh_t::layout;
        using m_shape_t =
            cute::remove_cvref_t<decltype(cute::shape(m_layout_t{}))>;
        constexpr bool full_bc = (cute::tuple_size<attrs_t>::value == 0) ||
                                 (cute::tuple_size<attrs_t>::value !=
                                  cute::tuple_size<m_shape_t>::value);
        return local_impl(st, std::bool_constant<full_bc>{});
    }
}
// shard-aware copy — dispatches on ShardTensor vs plain tensor.
// MVP: full-tensor copy via local() (which currently returns full tensor).
// TODO: implement per-program slicing in local(); add static_assert
// domain check for compile-time shapes.

// shard → plain
template <class T, class GL, class SL, class DT>
CUTE_HOST_DEVICE void copy(ShardTensor<T, GL, SL> const &src, DT &dst) {
    auto view = local(src);
    // ``local(src)`` returns a per-thread
    // view whose cute Layout is dynamic-int (runtime ``int`` shape /
    // stride) while ``dst`` may carry a compile-time ``cute::Int<N>``
    // hierarchy; ``cute::copy`` template deduction won't bridge the
    // two. Iterate element-wise via the linear operator() on both
    // sides — that bypasses the rank/hierarchy mismatch and matches
    // the reshard semantic "each thread writes its local fragment".
    int N = int(cute::size(view));
    for (int i = 0; i < N; ++i) {
        dst(i) = view(i);
    }
}

// plain → shard
template <class ST, class T, class GL, class SL>
CUTE_HOST_DEVICE void copy(ST const &src, ShardTensor<T, GL, SL> &dst) {
    auto view = local(dst);
    // same rationale as the shard→plain
    // overload — element-wise via linear ``operator()`` instead of
    // ``cute::copy`` to sidestep the dynamic/compile-time hierarchy
    // mismatch that pops out of ``local_impl``'s runtime-int layout.
    int N = int(cute::size(view));
    for (int i = 0; i < N; ++i) {
        view(i) = src(i);
    }
}

// shard → shard: both operands are ShardTensors —
// resolve the overload ambiguity that the shard→plain and
// plain→shard overloads above would otherwise create. Each thread
// reads from its src local view and writes its dst local view.
//
// Multi-dimensional copy: the gmem local view may be strided (e.g.
// multi-axis Split where cute::coalesce cannot flatten), and linear
// ``dv(i) = sv(i)`` misreads from wrong positions. Walk the local
// layout coordinates so both operands see the same logical access
// pattern regardless of stride.
template <class TS, class GLS, class SLS, class TD, class GLD, class SLD>
CUTE_HOST_DEVICE void copy(ShardTensor<TS, GLS, SLS> const &src,
                           ShardTensor<TD, GLD, SLD> &dst) {
    auto &&sv = local(src);
    auto &&dv = local(dst);
    int N = int(cute::size(dv));
    for (int i = 0; i < N; ++i) {
        dv(i) = sv(i);
    }
}

namespace ops {

// ── Grid-wide barrier ───────────────────────────────────────────────

// Software grid-wide barrier over a caller-provided gmem counter pair:
// bar[0] = arrival counter, bar[1] = release phase. Every CTA arrives,
// the last one resets the counter and bumps the phase; the rest spin on
// the phase. Requires ALL CTAs of the launch to be co-resident (the
// caller's occupancy contract). Reusable across launches and CUDA-graph
// replays: the counter self-resets each phase, the phase is monotone
// (u32 wrap is harmless). Prior gmem writes of every CTA are visible to
// every CTA after return (__threadfence on arrival orders them before
// the release; the atomic spin read acquires the release). The backing
// counter pair is defined per generated module (internal linkage), so a
// header include never introduces a shared/duplicated global symbol.
__device__ __forceinline__ void grid_barrier(unsigned int *bar) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int n_ctas = gridDim.x * gridDim.y * gridDim.z;
        unsigned int phase = atomicAdd(&bar[1], 0u);
        __threadfence();
        unsigned int arrived = atomicAdd(&bar[0], 1u) + 1u;
        if (arrived == n_ctas) {
            bar[0] = 0u;
            __threadfence();
            atomicAdd(&bar[1], 1u);
        } else {
            while (atomicAdd(&bar[1], 0u) == phase) {
            }
        }
    }
    __syncthreads();
}

// ── Op tags (functors for dispatch) ─────────────────────────────────

struct mul_op {
    template <class T> __device__ T operator()(T a, T b) const { return a * b; }
};
struct add_op {
    template <class T> __device__ T operator()(T a, T b) const { return a + b; }
};
struct sub_op {
    template <class T> __device__ T operator()(T a, T b) const { return a - b; }
};
struct div_op {
    template <class T> __device__ T operator()(T a, T b) const { return a / b; }
};

struct rsqrt_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(rsqrtf(static_cast<float>(x)));
    }
};
struct neg_op {
    template <class T> __device__ T operator()(T x) const { return -x; }
};
struct relu_op {
    template <class T> __device__ T operator()(T x) const {
        return x > T(0) ? x : T(0);
    }
};
struct square_op {
    template <class T> __device__ T operator()(T x) const { return x * x; }
};
struct clamp_op {
    float min_val, max_val;
    template <class T> __device__ T operator()(T x) const {
        return x < static_cast<T>(min_val)
                   ? static_cast<T>(min_val)
                   : (x > static_cast<T>(max_val) ? static_cast<T>(max_val)
                                                  : x);
    }
};

// ── Generic primitives ──────────────────────────────────────────────

namespace detail {

// when an op
// impl receives a ``ShardTensor`` it projects to the per-thread
// fragment via ``local()`` before reading / writing elements. Plain
// cute Tensors pass through unchanged. This keeps the
// shard-tensor-everywhere materialisation invariant from the codegen
// surface while every helper still operates on a flat cute Tensor.
template <class T> struct is_shard_tensor : std::false_type {};
template <class E, class GL, class SL>
struct is_shard_tensor<ShardTensor<E, GL, SL>> : std::true_type {};

template <class T> CUTE_HOST_DEVICE decltype(auto) to_local(T &&t) {
    if constexpr (is_shard_tensor<cute::remove_cvref_t<T>>::value) {
        return local(t);
    } else {
        return std::forward<T>(t);
    }
}

} // namespace detail

// Fill tensor with a scalar value
template <class TOut> CUTE_HOST_DEVICE void fill(TOut &dst, float val, int N) {
    auto &&d = detail::to_local(dst);
    for (int i = 0; i < N; ++i) {
        d(i) = static_cast<cute::remove_cvref_t<decltype(d(0))>>(val);
    }
}

// Runtime-N element-wise copy. Used by the dynamic-shape lowering when
// ``cute::copy`` would iterate the (compile-time) envelope upper bound
// instead of the runtime extent. Casts element-wise so source and dest
// dtypes may differ as long as ``static_cast`` is well-defined between
// them.
template <class TSrc, class TDst>
CUTE_HOST_DEVICE void copy_n(TSrc const &src, TDst &dst, int N) {
    auto s = detail::to_local(src);
    auto &&d = detail::to_local(dst);
    using value_type = cute::remove_cvref_t<decltype(d(0))>;
    for (int i = 0; i < N; ++i) {
        d(i) = static_cast<value_type>(s(i));
    }
}

// Unary pointwise: dst(i) = op(src(i))
template <class Op, class TIn, class TOut>
CUTE_HOST_DEVICE void unary(TIn const &src, TOut &dst, int N, Op op = {}) {
    auto s = detail::to_local(src);
    auto &&d = detail::to_local(dst);
    for (int i = 0; i < N; ++i) {
        d(i) = static_cast<cute::remove_cvref_t<decltype(d(0))>>(op(s(i)));
    }
}

// Binary pointwise: dst(i) = op(lhs(i), rhs(i))
template <class Op, class TL, class TR, class TOut>
CUTE_HOST_DEVICE void binary(TL const &lhs, TR const &rhs, TOut &dst, int N,
                             Op op = {}) {
    auto l = detail::to_local(lhs);
    auto r = detail::to_local(rhs);
    auto &&d = detail::to_local(dst);
    using value_type = cute::remove_cvref_t<decltype(d(0))>;
    for (int i = 0; i < N; ++i) {
        d(i) = static_cast<value_type>(
            op(static_cast<value_type>(l(i)), static_cast<value_type>(r(i))));
    }
}

// Binary multi-cell broadcast: ``dst`` and ``lhs`` hold ``n_dst`` cells
// of ``step`` elements each (one cell per row of a per-thread (n_dst,
// step) cute layout); ``rhs`` is a per-cell scalar
// (``size(rhs) == n_dst``).  Used for shape patterns like
// ``(1, 3, 4) op (1, 3, 1)`` where the broadcast is on the innermost
// axis — companion to the reduce tier-1 multi-cell impl.
//
// When the coalesced per-thread layout is multi-axis (factorised
// single-axis sugar — spec shard §7.1.2) we access via ``l(j, k)`` /
// ``d(j, k)`` so cells stay row-aligned regardless of cute's
// col-major linearisation; a 1-D coalesced view (the headline
// rmsnorm seq_1 path) keeps the linear ``l(j*step + k)`` access.
template <class Op, class TL, class TR, class TOut>
CUTE_HOST_DEVICE void binary_cell_bcast(TL const &lhs, TR const &rhs, TOut &dst,
                                        int n_dst, int step, Op op = {}) {
    auto l = detail::to_local(lhs);
    auto r = detail::to_local(rhs);
    auto &&d = detail::to_local(dst);
    using value_type = cute::remove_cvref_t<decltype(d(0))>;
    constexpr int l_rank = decltype(cute::rank(l))::value;
    constexpr int d_rank = decltype(cute::rank(d))::value;
    for (int j = 0; j < n_dst; ++j) {
        auto rj = static_cast<value_type>(r(j));
        if constexpr (l_rank > 1 && d_rank > 1) {
            for (int k = 0; k < step; ++k) {
                d(j, k) = static_cast<value_type>(
                    op(static_cast<value_type>(l(j, k)), rj));
            }
        } else {
            for (int k = 0; k < step; ++k) {
                const int i = j * step + k;
                d(i) = static_cast<value_type>(
                    op(static_cast<value_type>(l(i)), rj));
            }
        }
    }
}

// Binary broadcast col: (M,K) * (M,1) -> (M,K)
template <class Op, class TL, class TR, class TOut>
CUTE_HOST_DEVICE void binary_bcast_col(TL const &lhs, TR const &rhs, TOut &dst,
                                       int M, int K, Op op = {}) {
    using value_type = cute::remove_cvref_t<decltype(dst(0))>;
    for (int m = 0; m < M; ++m) {
        auto scale = static_cast<value_type>(rhs(m, 0));
        for (int k = 0; k < K; ++k) {
            dst(m, k) = static_cast<value_type>(
                op(static_cast<value_type>(lhs(m, k)), scale));
        }
    }
}

// Binary broadcast row: (M,K) * (K,) -> (M,K)
template <class Op, class TL, class TR, class TOut>
CUTE_HOST_DEVICE void binary_bcast_row(TL const &lhs, TR const &rhs, TOut &dst,
                                       int M, int K, Op op = {}) {
    using value_type = cute::remove_cvref_t<decltype(dst(0))>;
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            dst(m, k) =
                static_cast<value_type>(op(static_cast<value_type>(lhs(m, k)),
                                           static_cast<value_type>(rhs(k))));
        }
    }
}

// Binary broadcast scalar: (N,) * () -> (N,)
template <class Op, class TS, class TV, class TOut>
CUTE_HOST_DEVICE void binary_bcast_scalar(TS const &src, TV const &scalar,
                                          TOut &dst, int N, Op op = {}) {
    auto s = detail::to_local(src);
    auto sc = detail::to_local(scalar);
    auto &&d = detail::to_local(dst);
    using value_type = cute::remove_cvref_t<decltype(d(0))>;
    auto val = static_cast<value_type>(sc(0));
    for (int i = 0; i < N; ++i) {
        d(i) = static_cast<value_type>(op(static_cast<value_type>(s(i)), val));
    }
}

// Reduce: dst = reduce_kind(src) over axes
template <class Op, class TIn, class TOut>
CUTE_HOST_DEVICE void reduce(TIn const &src, TOut &dst, int M, int K,
                             Op op = {}) {
    using value_type = cute::remove_cvref_t<decltype(dst(0))>;
    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += static_cast<float>(src(m * K + k));
        }
        dst(m) = static_cast<value_type>(op(acc, float(K)));
    }
}

// Reduce 1D: scalar = reduce_kind(vec)
template <class Op, class TIn, class TOut>
CUTE_HOST_DEVICE void reduce_1d(TIn const &src, TOut &dst, int N, Op op = {}) {
    using value_type = cute::remove_cvref_t<decltype(dst(0))>;
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        acc += static_cast<float>(src(i));
    }
    dst(0) = static_cast<value_type>(op(acc, float(N)));
}

struct mean_op {
    template <class T> __device__ T operator()(T sum, T count) const {
        return sum / count;
    }
};
struct sum_op {
    template <class T> __device__ T operator()(T sum, T) const { return sum; }
};
struct absmax_op {
    template <class T> __device__ T operator()(T cur, T candidate) const {
        return fabsf(static_cast<float>(candidate)) >
                       fabsf(static_cast<float>(cur))
                   ? candidate
                   : cur;
    }
};

// ── Sharded reduce ───────────────────────────────
//
// ``tilefoundry::ops::reduce<Op, Axes>(src, dst, workspace)`` is the runtime
// entry point HIR→TIR lowering targets for cross-warp / intra-warp /
// intra-thread sharded reductions. The
// dispatch shape is "C++20 concepts + reduce impl pack" — the top-level
// template inspects ``src``'s ShardTensor type and chooses the correct
// staged impl:
//
//   - LOCAL: every thread folds its own contiguous slice (loop)
//   - WARP : intra-warp lane fold via ``__shfl_xor_sync`` butterfly
//   - CTA  : cross-warp fold staged through the shared-memory
//            ``workspace`` buffer the lowering allocated
//
// MEAN is decomposed into ``SUM`` over the staged chain plus a single
// final division by the total reduce extent so precision loss stays
// minimal.
//
// The scaffold below implements the headline RMSNorm path (full reduce
// over the trailing axis with mesh axes ``(warp, thread)``). It is
// intentionally narrow: each thread holds ``LocalCount`` contiguous
// scalars, intra-warp reduces 32 lanes, cross-warp reduces ``WarpCount``
// warps via ``workspace``. More general sharding patterns (partial axes,
// row-stride MEANs, multi-warp groups) are explicit TODOs and will land
// alongside their first use case.

namespace reduce_impl {

// Workspace tag used when no shared-memory staging is needed (every
// reduce mesh axis lives inside a single warp).
struct no_workspace_t {};
inline constexpr no_workspace_t no_workspace{};

// Per-thread local fold: max absolute value over N elements.
template <class SrcT>
__device__ float local_fold_maxabs(SrcT const &src, int N) {
    float best = 0.f;
    for (int i = 0; i < N; ++i) {
        best = fmaxf(best, fabsf(static_cast<float>(src(i))));
    }
    return best;
}

// Per-thread local fold of a contiguous tensor of size N into a single
// scalar accumulator (sum / max / min, etc.). Returns the
// f32-promoted accumulator so MEAN's final divide stays in f32.
template <class SrcT> __device__ float local_fold_sum(SrcT const &src, int N) {
    float acc = 0.f;
    for (int i = 0; i < N; ++i) {
        acc += static_cast<float>(src(i));
    }
    return acc;
}

// Intra-warp butterfly sum reduction (32 lanes → broadcast sum).
__device__ inline float warp_sum_butterfly(float val) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, delta);
    }
    return val;
}

// Intra-warp butterfly max reduction (32 lanes → broadcast max).
__device__ inline float warp_max_butterfly(float val) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, delta));
    }
    return val;
}

// Cross-warp sum reduction via a shared-memory workspace.
//
// ``workspace`` is sized to ``total_warps`` (all non-thread mesh
// positions).  ``warps_per_group`` (≤ total_warps) controls grouping:
// warps are partitioned into contiguous groups of ``warps_per_group``
// slots, and each thread only aggregates across its own group.
// When ``warps_per_group == total_warps`` (single group), this is
// equivalent to the original flat cross-warp reduce.
template <class WorkspaceT>
__device__ float cta_sum_via_workspace(float warp_partial,
                                       WorkspaceT &workspace,
                                       int warps_per_group) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        workspace(warp_id) = warp_partial;
    }
    __syncthreads();
    int group_id = warp_id / warps_per_group;
    int group_start = group_id * warps_per_group;
    float acc = 0.f;
    for (int w = 0; w < warps_per_group; ++w) {
        acc += static_cast<float>(workspace(group_start + w));
    }
    return acc;
}

} // namespace reduce_impl

// ── Reduce tier-2: cross-warp within a CTA (smem workspace) ───────
//
// Selected by the HIR→TIR analysis
// when the reduce spans warps within a single CTA but does not cross
// CTAs.  Each warp folds locally then writes its partial into a
// shared-memory workspace; a single ``__syncthreads()`` later, every
// thread sums its group's slots and broadcasts the result to every
// output cell it owns.  ``warps_per_group`` partitions the workspace
// so non-reduced Split mesh axes form independent reduction groups.
//
// MEAN divides by ``local_n × 32 × warps_per_group``.
template <class Op, class Axes, class SrcT, class DstT, class WorkspaceT>
__device__ inline void reduce_intra_cta(SrcT const &src, DstT &dst,
                                        WorkspaceT &workspace,
                                        int warps_per_group) {
    auto s = detail::to_local(src);
    auto &&d = detail::to_local(dst);
    auto &&ws = detail::to_local(workspace);
    using value_type = cute::remove_cvref_t<decltype(d(0))>;
    // local_n: elements folded per thread.  Per the tier-1 ``reduce``
    // contract (size(s) % size(d) == 0; size(d) chunks of step lanes),
    // each output cell takes ``size(s) / size(d)`` source elements.
    // For the headline rmsnorm path size(d) == 1 ⇒ local_n == size(s)
    // — folds the entire per-thread tensor. Taking only the last cute
    // axis here (as the prior heuristic did) silently dropped the
    // per-warp residual axes introduced by single-axis sugar
    // factorisation (spec shard §7.1.2 + parser/sugar.py), so the mean
    // reduce summed only a fraction of the row.
    const int n_src = static_cast<int>(cute::size(s));
    const int n_dst = static_cast<int>(cute::size(d));
    const int local_n = (n_dst == 0) ? n_src : (n_src / n_dst);

    float local, warp_partial, cta_partial;
    if constexpr (std::is_same_v<Op, absmax_op>) {
        local = reduce_impl::local_fold_maxabs(s, local_n);
        warp_partial = reduce_impl::warp_max_butterfly(local);
        cta_partial = reduce_impl::cta_sum_via_workspace(warp_partial, ws,
                                                         warps_per_group);
    } else {
        local = reduce_impl::local_fold_sum(s, local_n);
        warp_partial = reduce_impl::warp_sum_butterfly(local);
        cta_partial = reduce_impl::cta_sum_via_workspace(warp_partial, ws,
                                                         warps_per_group);
    }

    if constexpr (std::is_same_v<Op, mean_op>) {
        float total_n = float(local_n) * 32.f * float(warps_per_group);
        float result = cta_partial / total_n;
        for (int i = 0; i < static_cast<int>(cute::size(d)); ++i) {
            d(i) = static_cast<value_type>(result);
        }
    } else if constexpr (std::is_same_v<Op, sum_op>) {
        for (int i = 0; i < static_cast<int>(cute::size(d)); ++i) {
            d(i) = static_cast<value_type>(cta_partial);
        }
    } else if constexpr (std::is_same_v<Op, absmax_op>) {
        for (int i = 0; i < static_cast<int>(cute::size(d)); ++i) {
            d(i) = static_cast<value_type>(cta_partial);
        }
    } else {
        static_assert(sizeof(Op) == 0, "tilefoundry::ops::reduce: unsupported Op");
    }
}

// ── Reduce tier-3: cross-CTA (grid-level) — placeholder ───────────
//
// Reserved entry point for reductions
// whose mesh axes cross CTA boundaries.  No implementation yet — any
// instantiation traps at compile time, and the HIR→TIR analysis
// must reject cross-CTA reduce calls explicitly rather than fall
// back to ``reduce_intra_cta``.
template <class Op, class Axes, class SrcT, class DstT, class WorkspaceT>
__device__ inline void reduce_cross_cta(SrcT const &, DstT &, WorkspaceT &,
                                        int) {
    static_assert(sizeof(Op) == 0,
                  "tilefoundry::ops::reduce_cross_cta: cross-CTA reduce not "
                  "implemented yet");
}

// ── Reduce tier-1: intra-warp only, no smem workspace ─────────────
//
// This entry point is selected by the
// HIR→TIR analysis when the reduced cute axes are covered entirely
// by the ``thread`` topology — i.e. no cross-warp coordination is
// needed.  Each thread folds its per-cell slice locally, then a
// 32-lane ``__shfl_xor_sync`` butterfly broadcasts the partial.
//
// Per-thread cell decomposition: the per-thread cute tensor
// has ``size(s)`` source elements feeding ``size(d)`` destination
// cells; we require ``size(s) % size(d) == 0`` and treat the source
// as ``size(d)`` contiguous chunks of ``size(s) / size(d)`` lanes.
// Each chunk reduces to one output cell — this matches both the
// y-residue multi-cell case (size(s)=12, size(d)=3, step=4) and the
// single-cell rmsnorm case (size(s)=12, size(d)=3 with broadcast or
// size(s)=N, size(d)=1).
template <class Op, class Axes, class SrcT, class DstT>
__device__ inline void reduce(SrcT const &src, DstT &dst) {
    auto s = detail::to_local(src);
    auto &&d = detail::to_local(dst);
    using value_type = cute::remove_cvref_t<decltype(d(0))>;

    const int n_src = static_cast<int>(cute::size(s));
    const int n_dst = static_cast<int>(cute::size(d));
    const int n_cells = (n_dst == 0) ? 1 : n_dst;
    const int step = n_src / n_cells;

    // Cell-aligned 2-D access (``s(j, k)``) keeps each cell on a single
    // row of the per-thread cute layout, regardless of cute's
    // col-major linearisation. Linear ``s(j*step + k)`` mis-aligns
    // when the coalesced per-thread layout has multiple cute axes
    // (factorised single-axis sugar — spec shard §7.1.2). The
    // single-axis path (1-D ``s``) uses linear access — that's the
    // headline rmsnorm seq_1 case where the per-thread cute layout
    // collapses to a single contiguous dim after coalesce.
    constexpr int s_rank = decltype(cute::rank(s))::value;
    for (int j = 0; j < n_cells; ++j) {
        float local;
        if constexpr (std::is_same_v<Op, absmax_op>) {
            local = 0.f;
            if constexpr (s_rank > 1) {
                for (int k = 0; k < step; ++k) {
                    local = fmaxf(local, fabsf(static_cast<float>(s(j, k))));
                }
            } else {
                const int base = j * step;
                for (int k = 0; k < step; ++k) {
                    local =
                        fmaxf(local, fabsf(static_cast<float>(s(base + k))));
                }
            }
        } else {
            local = 0.f;
            if constexpr (s_rank > 1) {
                for (int k = 0; k < step; ++k) {
                    local += static_cast<float>(s(j, k));
                }
            } else {
                const int base = j * step;
                for (int k = 0; k < step; ++k) {
                    local += static_cast<float>(s(base + k));
                }
            }
        }
        float partial;
        if constexpr (std::is_same_v<Op, absmax_op>) {
            partial = reduce_impl::warp_max_butterfly(local);
        } else {
            partial = reduce_impl::warp_sum_butterfly(local);
        }
        if constexpr (std::is_same_v<Op, mean_op>) {
            const float total_n = float(step) * 32.f;
            d(j) = static_cast<value_type>(partial / total_n);
        } else if constexpr (std::is_same_v<Op, sum_op> ||
                             std::is_same_v<Op, absmax_op>) {
            d(j) = static_cast<value_type>(partial);
        } else {
            static_assert(sizeof(Op) == 0,
                          "tilefoundry::ops::reduce: unsupported Op");
        }
    }
}

// Legacy wrappers kept for transition
template <class TIn, class TOut>
CUTE_HOST_DEVICE void relu(TIn const &src, TOut &dst) {
    // ``unary`` iterates the per-thread local tensor, so the element count
    // must come from the local view — ``src`` may be a ShardTensor, which
    // has no ``cute::size``.
    unary(src, dst, int(cute::size(detail::to_local(src))), relu_op{});
}

template <class TIn, class TOut>
CUTE_HOST_DEVICE void cast(TIn const &src, TOut &dst, int N) {
    auto s = detail::to_local(src);
    auto &&d = detail::to_local(dst);
    for (int i = 0; i < N; ++i) {
        d(i) = static_cast<cute::remove_cvref_t<decltype(d(0))>>(s(i));
    }
}

// Element-wise clamp: dst(i) = min(max(src(i), min_val), max_val)
template <class TIn, class TOut>
CUTE_HOST_DEVICE void clamp(TIn const &src, TOut &dst, int N, float min_val,
                            float max_val) {
    auto s = detail::to_local(src);
    auto &&d = detail::to_local(dst);
    using value_type = cute::remove_cvref_t<decltype(d(0))>;
    auto mn = static_cast<value_type>(min_val);
    auto mx = static_cast<value_type>(max_val);
    for (int i = 0; i < N; ++i) {
        auto v = static_cast<value_type>(s(i));
        d(i) = (v < mn) ? mn : ((v > mx) ? mx : v);
    }
}

// Fused RMSNorm: reduces over the last axis (K), computes in f32.
// dst(m,k) = src(m,k) * rsqrt(mean(src(m,:)^2) + eps) * weight(k)
template <class TIn, class TOut, class TW>
CUTE_HOST_DEVICE void rmsnorm(TIn const &src, TOut &dst, TW const &weight,
                              int M, int K, float eps) {
    using value_type = cute::remove_cvref_t<decltype(dst(0))>;
    for (int m = 0; m < M; ++m) {
        float sum_sq = 0.0f;
        for (int k = 0; k < K; ++k) {
            float val = static_cast<float>(src(m * K + k));
            sum_sq += val * val;
        }
        float rms = rsqrtf(sum_sq / float(K) + eps);
        for (int k = 0; k < K; ++k) {
            float val = static_cast<float>(src(m * K + k)) * rms *
                        static_cast<float>(weight(k));
            dst(m * K + k) = static_cast<value_type>(val);
        }
    }
}

// ── Mma — SM80 16x8x16 BF16/F32 atom ──
//
// Per-thread input tensors carry their *own* coalesced cute layout
// produced by ``tilefoundry::ShardLayout`` reshard. For
// ``SM80_16x8x16_F32BF16BF16F32_TN`` each lane (warp lane = ``threadIdx.x``
// % 32, decomposed by mesh stride ``(1, 4)`` into ``tx = lane % 4`` /
// ``ty = lane / 4``) holds:
//
//   A: 8 bf16  (per cute row-major value layout shape (2, 2, 2),
//               strides (1, 8, 128) over the (M=16, K=16) tile)
//   B: 4 bf16  (shape (2, 2), strides (1, 8) over (K=16, N=8))
//   C: 4 f32   (shape (2, 2), strides (1, 8) over (M=16, N=8))
//
// The PTX ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``
// instruction expects A in 4 ``b16x2`` registers, B in 2 ``b16x2``
// registers, C/D in 4 ``f32`` registers, with a *specific* mapping
// from per-lane (M, K) / (K, N) / (M, N) coordinates to register
// positions (see ``cute/atom/mma_traits_sm80.hpp`` ALayout / BLayout /
// CLayout).
//
// The register-pack order below is derived from the row-major
// re-derivation of the cute MMA_Traits layout: cute value axes (v0, v1, v2) for
// A bind to row-major strides (1, 128, 8); after ``cute::coalesce`` the
// per-thread tensor is shape (2, 2, 2) strides (1, 8, 128), i.e.
// (v0, v2, v1) order. The pairing therefore is::
//
//   a_regs[0] = pack(t(0,0,0), t(1,0,0))   // v1=0, v2=0
//   a_regs[1] = pack(t(0,0,1), t(1,0,1))   // v1=1, v2=0
//   a_regs[2] = pack(t(0,1,0), t(1,1,0))   // v1=0, v2=1
//   a_regs[3] = pack(t(0,1,1), t(1,1,1))   // v1=1, v2=1
//
// (and analogously for B / C).

namespace mma_detail {

CUTE_HOST_DEVICE uint32_t pack_bf16x2(uint16_t lo, uint16_t hi) {
    return (uint32_t(hi) << 16) | uint32_t(lo);
}

template <class T> CUTE_HOST_DEVICE uint16_t as_u16(T const &x) {
    uint16_t out;
    __builtin_memcpy(&out, &x, sizeof(uint16_t));
    return out;
}

} // namespace mma_detail

// Effect-form Mma: ``c += a @ b`` (in place). Operands are per-thread
// register fragments produced by ``reshard`` into a fragment-shaped
// ``ShardLayout``. The acc operand is read-modify-write.
template <class TA, class TB, class TC>
__device__ void mma_sm80_16x8x16_bf16(TA const &a, TB const &b, TC &c) {
    using Atom = cute::SM80_16x8x16_F32BF16BF16F32_TN;
    using namespace mma_detail;

    // The per-shard reshard buffers may carry size-1 axes (the split
    // axes of the fragment ShardLayout). Sidestep cute's multi-dim
    // indexing by working directly on the underlying register array —
    // cute's ``ArrayEngine`` stores elements linearly, and per the
    // row-major fragment encoding the natural traversal order is::
    //
    //   A flat[0..7] = (v0=0..1) × (v2=0..1) × (v1=0..1) (col-major)
    //   B flat[0..3] = (v0=0..1) × (v1=0..1)
    //   C flat[0..3] = (v0=0..1) × (v1=0..1)
    //
    // The pairing into mma registers therefore is::
    //
    //   a_regs[0] = pack(A[0], A[1])    // v1=0, v2=0
    //   a_regs[1] = pack(A[4], A[5])    // v1=1, v2=0
    //   a_regs[2] = pack(A[2], A[3])    // v1=0, v2=1
    //   a_regs[3] = pack(A[6], A[7])    // v1=1, v2=1
    //   b_regs[0] = pack(B[0], B[1])    // v1=0
    //   b_regs[1] = pack(B[2], B[3])    // v1=1
    // A/B are reshard'd register fragments; C is the read-modify-write
    // accumulator. ``.data()`` yields the per-thread register array for both a
    // ShardTensor (TIR effect form) and a plain cute Tensor (HIR functional
    // form), so the two paths share one access.
    auto a_data = a.data();
    auto b_data = b.data();
    auto c_data = c.data();

    uint32_t a0 = pack_bf16x2(as_u16(a_data[0]), as_u16(a_data[1]));
    uint32_t a1 = pack_bf16x2(as_u16(a_data[4]), as_u16(a_data[5]));
    uint32_t a2 = pack_bf16x2(as_u16(a_data[2]), as_u16(a_data[3]));
    uint32_t a3 = pack_bf16x2(as_u16(a_data[6]), as_u16(a_data[7]));

    uint32_t b0 = pack_bf16x2(as_u16(b_data[0]), as_u16(b_data[1]));
    uint32_t b1 = pack_bf16x2(as_u16(b_data[2]), as_u16(b_data[3]));

    // Pass distinct ``d_*`` outputs vs ``c_*`` accumulator inputs so
    // the ``=f`` / ``f`` constraints don't alias.
    float c0 = c_data[0];
    float c1 = c_data[1];
    float c2 = c_data[2];
    float c3 = c_data[3];
    float d0, d1, d2, d3;

    Atom::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);

    c_data[0] = d0;
    c_data[1] = d1;
    c_data[2] = d2;
    c_data[3] = d3;
}

} // namespace ops

} // namespace tilefoundry
