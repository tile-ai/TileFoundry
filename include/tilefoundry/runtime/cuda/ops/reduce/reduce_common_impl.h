/// reduce common impl — shared per-tag traits, per-thread fold, cell
/// decomposition, and the sharded-reduce dispatch trait / layout detector.
///
/// Included in-context from ``ops/reduce.cuh`` (which is itself included inside
/// ``namespace tilefoundry::ops`` from runtime.cuh). This header therefore does
/// NOT open ``namespace tilefoundry`` / ``ops`` and does NOT pull in system
/// headers — cute/std and the surrounding names (``detail::to_local``,
/// ``shard::S``/``shard::B``, ``TopologyScope``) are already in scope.
#pragma once

namespace reduce_impl {

/// Workspace tag used when no shared-memory staging is needed (every
/// reduce mesh axis lives inside a single warp).
struct no_workspace_t {};
inline constexpr no_workspace_t no_workspace{};

/// What a reduction needs beyond the merge itself: ``init`` seeds the fold,
/// ``elem`` enters its domain, and ``finalize`` divides a mean by the count.
///
/// ``combine_op`` *names* the binary functor, it does not restate it: the merge
/// lives once, in ``ops::add_op`` / ``ops::max_op`` / the caller's own, and is
/// what ``binary`` and ``warp_reduce`` call too. Writing it out again here is
/// how ``reduce<add_op>`` and a warp butterfly came to be two answers to one
/// question, with nothing to keep them equal.
template <class Op> struct reduce_traits;
template <> struct reduce_traits<add_op> {
    using combine_op = add_op;
    static constexpr float init = 0.f;
    __device__ static float elem(float x) { return x; }
    __device__ static float finalize(float acc, float) { return acc; }
};
template <> struct reduce_traits<mean_op> {
    using combine_op = add_op;
    static constexpr float init = 0.f;
    __device__ static float elem(float x) { return x; }
    __device__ static float finalize(float acc, float n) { return acc / n; }
};
template <> struct reduce_traits<max_op> {
    using combine_op = max_op;
    static constexpr float init = -INFINITY;
    __device__ static float elem(float x) { return x; }
    __device__ static float finalize(float acc, float) { return acc; }
};
template <> struct reduce_traits<min_op> {
    using combine_op = min_op;
    static constexpr float init = INFINITY;
    __device__ static float elem(float x) { return x; }
    __device__ static float finalize(float acc, float) { return acc; }
};
template <> struct reduce_traits<absmax_op> {
    using combine_op = max_op;
    static constexpr float init = 0.f;
    __device__ static float elem(float x) { return fabsf(x); }
    __device__ static float finalize(float acc, float) { return acc; }
};

/// Merge two partials with this reduction's own functor.
template <class Op> __device__ float combine(float a, float b) {
    return typename reduce_traits<Op>::combine_op{}(a, b);
}

/// Whether every value of ``T`` survives a round trip through ``float``.
///
/// ``reduce_traits`` above is float in all four of its members -- ``init``,
/// ``elem``, ``combine``, ``finalize`` -- so *the op's domain is float*, and
/// that was true of the implementation while the declaration said nothing.
/// ``float`` carries 24 significand bits, so it holds every value of a 1- or
/// 2-byte integer and of any float no wider than itself; ``int``, ``long
/// long`` and ``double`` it does not, and a packed 64-bit key (a score in the
/// high bits and an index in the low) it destroys outright.
template <class T> CUTE_HOST_DEVICE constexpr bool folds_in_float() {
    if constexpr (std::is_integral_v<T>)
        return sizeof(T) <= 2;
    else if constexpr (std::is_floating_point_v<T>)
        return sizeof(T) <= sizeof(float);
    else
        /// ``__half`` / ``__nv_bfloat16`` / the fp8 pair are class types, and
        /// each is narrower than float in both the exponent and the
        /// significand -- so every value of one of them is a float.
        return sizeof(T) <= 2 && std::is_convertible_v<T, float>;
}

/// What both operands have to be for any tier to be able to run.
///
/// Asked once, at the entry, so that the reason a fold cannot carry something
/// is a sentence at the call site rather than a rounded answer (the domain) or
/// an error inside CuTe's ``get`` (the layout shape).
template <class Src, class Dst>
CUTE_HOST_DEVICE constexpr void check_reduce_domain() {
    using s_view = tilefoundry::detail::local_view_t<Src>;
    using d_view = tilefoundry::detail::local_view_t<Dst>;
    static_assert(
        folds_in_float<typename s_view::value_type>(),
        "ops::reduce: every tier enters the fold through static_cast<float> "
        "and combines in float, so the element type must be one float holds "
        "exactly -- a 4- or 8-byte integer (a packed key, say) would be "
        "rounded. Fold it yourself, or split it into pieces float can carry");
    static_assert(
        folds_in_float<typename d_view::value_type>(),
        "ops::reduce: the finalised value is a float, so the destination's "
        "element type must be one float can be stored into without loss");
    static_assert(
        !cute::is_composed_layout<typename s_view::layout_type>::value,
        "ops::reduce: the source's projected layout must be a plain "
        "cute::Layout -- the fold splits it into the kept modes and the "
        "reduced ones, and a composed (swizzled, or origin-shifted) layout has "
        "no such split. Materialise the slice first, then reduce it");
}

template <class Op>
inline constexpr bool is_supported_reduce_op_v =
    std::is_same_v<Op, add_op> || std::is_same_v<Op, mean_op> ||
    std::is_same_v<Op, absmax_op> || std::is_same_v<Op, max_op> ||
    std::is_same_v<Op, min_op>;

/// Whether ``Axis`` is one of the axes ``Axes`` names.
template <class Axes, int Axis> CUTE_HOST_DEVICE constexpr bool is_reduced() {
    bool found = false;
    [&]<size_t... Js>(std::index_sequence<Js...>) {
        ((found = found || int(cute::remove_cvref_t<decltype(cute::get<Js>(
                                   Axes{}))>::value) == Axis),
         ...);
    }(std::make_index_sequence<cute::tuple_size<Axes>::value>{});
    return found;
}

/// The modes of ``l`` that ``Axes`` names (``Want``), or the rest.
///
/// ``Axes`` is the op's own argument, and until now it was declared, threaded
/// through all four tiers, and never read. ``size(src) / size(dst)`` stood in
/// for it -- the right number only when the reduced axes are the fast ones and
/// the elements are grouped the way that division assumes.
///
/// The two halves are disjoint sets of modes, so an offset in one plus an
/// offset in the other is the element's offset: no division, and nothing
/// assumed about which axes sit where.
template <bool Want, class Axes, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto
pick_axes(cute::Layout<Shape, Stride> const &l) {
    auto idx = cute::filter_tuple(
        cute::make_seq<cute::rank(cute::Layout<Shape, Stride>{})>{},
        [](auto i) {
            if constexpr (is_reduced<Axes, decltype(i)::value>() == Want)
                return cute::make_tuple(i);
            else
                return cute::tuple<>{};
        });
    if constexpr (cute::tuple_size<decltype(idx)>::value == 0) {
        return cute::make_layout(cute::Int<1>{}, cute::Int<0>{});
    } else {
        return cute::apply(
            cute::transform(
                idx, [&](auto i) { return cute::get<decltype(i)::value>(l); }),
            [](auto const &...m) { return cute::make_layout(m...); });
    }
}

/// ``s`` re-moded as ``(kept, reduced)``: one tensor, indexed as a pair.
///
/// The two halves are disjoint sets of ``s``'s own modes carrying ``s``'s own
/// strides, so this is a rank-2 view of the same elements and ``t(j, k)`` is
/// the element the fold wants -- reached through the tensor, which is what
/// applies the layout. The fold used to compute ``s.data()[kept(j) + red(k)]``
/// instead: the layout's own arithmetic written a second time at the use site,
/// correct only while ``s``'s offsets add that way.
template <class Axes, class SrcT>
CUTE_HOST_DEVICE auto as_kept_reduced(SrcT const &s) {
    auto const l = cute::layout(s);
    return cute::make_tensor(
        s.data(),
        cute::make_layout(pick_axes<false, Axes>(l), pick_axes<true, Axes>(l)));
}

/// Fold the reduced axes at the kept-axis position ``j``.
template <class Op, class Axes, class SrcT>
__device__ float local_fold(SrcT const &s, int j) {
    using traits = reduce_traits<Op>;
    auto t = as_kept_reduced<Axes>(s);
    float acc = traits::init;
    CUTE_UNROLL
    for (int k = 0; k < int(cute::size<1>(t)); ++k)
        acc = combine<Op>(acc, traits::elem(static_cast<float>(t(j, k))));
    return acc;
}

/// How many outputs this instance produces, and how many elements each eats.
template <class Axes, class SrcT> CUTE_HOST_DEVICE constexpr int kept_cells() {
    return int(cute::size(pick_axes<false, Axes>(
        cute::remove_cvref_t<decltype(cute::layout(std::declval<SrcT>()))>{})));
}
template <class Axes, class SrcT>
CUTE_HOST_DEVICE constexpr int reduced_span() {
    return int(cute::size(pick_axes<true, Axes>(
        cute::remove_cvref_t<decltype(cute::layout(std::declval<SrcT>()))>{})));
}

/// Combine one partial per warp, with the reduction's own combine.
///
/// It used to add them whatever the reduction was, which is right for ``sum``
/// and ``mean`` and silently wrong for anything else -- an ``absmax`` over a
/// group of warps came back as the sum of their maxima.
///
/// ``workspace`` holds one slot per warp; ``warps_per_group`` partitions them
/// into contiguous groups, and each thread aggregates only its own.
template <class Op, class WorkspaceT>
__device__ float cta_combine_via_workspace(float warp_partial,
                                           WorkspaceT &workspace,
                                           int warps_per_group) {
    const int tid = int(program_id<TopologyScope::thread>());
    const int lane = tid % kWarpSize;
    const int warp_id = tid / kWarpSize;
    if (lane == 0) {
        workspace(warp_id) = warp_partial;
    }
    __syncthreads();
    int group_id = warp_id / warps_per_group;
    int group_start = group_id * warps_per_group;
    float acc = reduce_traits<Op>::init;
    for (int w = 0; w < warps_per_group; ++w) {
        acc = combine<Op>(acc, static_cast<float>(workspace(group_start + w)));
    }
    return acc;
}

/// ── Layered sharded-reduce dispatch ───────────────────────────────
/// Compile-time derivation of the reduction level and ``warps_per_group`` from
/// the operand shard layouts, consumed by the public ``reduce`` entry.
/// The attr-kind tests are the layout system's own (layout/shard_layout.cuh),
/// not a second copy: ``shard_offset`` and this dispatch have to agree about
/// what a ``P`` and a ``Dynamic`` are, and two tables cannot be made to.
using tilefoundry::detail::is_partial_attr_v;
using tilefoundry::detail::is_split_attr_v;

/// An axis whose instances hold pieces of one value, either kind.
///
/// ``S<a>`` splits the tensor, so each instance holds different elements;
/// ``P<r>`` splits the *sum*, so each holds a contribution to the same
/// elements. Which it is changes what a local view looks like and not at all
/// what has to happen across the mesh -- both need every instance's answer
/// combined before the value exists. Reading only ``S`` here treated a partial
/// as a broadcast and returned one instance's contribution as the whole.
template <class T>
inline constexpr bool is_reducible_attr_v =
    is_split_attr_v<T> || is_partial_attr_v<T>;

struct reduce_dispatch_info {
    bool lane_reduced;
    int warps_per_group;
    /// How many lanes hold a piece of one value -- the product of the reduced
    /// lane-axis extents, which is the mesh's answer and not necessarily the
    /// whole warp. A mesh laid out several rows to a warp reduces each row
    /// across its own lanes; butterflying over all 32 would mix the rows, and
    /// dividing a mean by 32 would divide by the wrong count.
    int lanes_reduced;
    /// Whether any mesh axis actually has to be crossed. False when nothing is
    /// split or partial, and when the destination asked to stay ``P``.
    bool mesh_reduced;
    /// Whether every reduced axis divides into whole lanes and whole warps.
    ///
    /// A reduced axis of 48 over a warp of 32 does not: its instances are 32
    /// lanes of one warp and 16 of the next, which neither a butterfly nor a
    /// per-warp workspace slot describes. The arithmetic below would answer
    /// ``warps_per_group == 1`` for it and the fold would stop after a warp.
    bool warp_aligned;
    bool needs_workspace;
    bool needs_counter;
    bool needs_bar_id;
};
using plan_t = reduce_dispatch_info;

/// Derive, from the (src, dst) operand ShardLayouts, the active reduction level
/// and its ``warps_per_group``. Pure compile-time so the caller can select the
/// tier with ``if constexpr`` — otherwise the untaken tier still instantiates
/// and, e.g., ``CrossWarp<mean_op>`` would trip its supported-op guard.
/// Requires a static mesh layout (the reduce mesh is a thread-scoped static
/// mesh); a reduced axis on a non-thread mesh scope yields the cross-warp tier.
template <class SrcSL, class DstSL>
CUTE_HOST_DEVICE constexpr reduce_dispatch_info reduce_dispatch() {
    using src_attrs = typename SrcSL::attrs;
    using dst_attrs = typename DstSL::attrs;
    using mesh_t = typename SrcSL::mesh;
    static_assert(
        tilefoundry::detail::mesh_names_only<mesh_t, TopologyScope::thread,
                                             TopologyScope::cta>(),
        "ops::reduce: mesh must name cta or thread");
    constexpr bool is_thread =
        tilefoundry::detail::mesh_names_level<mesh_t, TopologyScope::thread>();
    using m_layout_t =
        typename tilefoundry::detail::mesh_axes_of<mesh_t,
                                                   TopologyScope::thread>;
    constexpr int m_rank = cute::tuple_size<src_attrs>::value;
    static_assert(tilefoundry::detail::shard_attrs_match_mesh<SrcSL>() &&
                      tilefoundry::detail::shard_attrs_match_mesh<DstSL>(),
                  "ops::reduce: one attr per mesh axis on both operands -- the "
                  "reduced set is read off src[i] against dst[i], so each "
                  "operand has to say what every one of its mesh's axes does "
                  "with the tensor");
    static_assert(
        std::is_same_v<typename SrcSL::mesh, typename DstSL::mesh>,
        "ops::reduce: the two operands must name one mesh -- axis i "
        "of src's mesh and axis i of dst's are what the reduced set "
        "is the difference of, and two meshes make i mean two things");
    static_assert(
        is_thread ||
            tilefoundry::detail::mesh_names_level<mesh_t, TopologyScope::cta>(),
        "ops::reduce: a reduce mesh's scope must be cta or thread -- "
        "the warp level has no program_id to name a participant with "
        "and scope_count is a sentinel, so a warp-sized grouping is "
        "an axis of a thread mesh's layout");

    int m_ext[m_rank] = {};
    bool reduced[m_rank] = {};
    auto const m_shape = cute::shape(m_layout_t{});
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((m_ext[Is] = int(cute::get<Is>(m_shape))), ...);
        ((reduced[Is] =
              is_reducible_attr_v<
                  cute::remove_cvref_t<decltype(cute::get<Is>(src_attrs{}))>> &&
              std::is_same_v<
                  cute::remove_cvref_t<decltype(cute::get<Is>(dst_attrs{}))>,
                  shard::B>),
         ...);
    }(std::make_index_sequence<m_rank>{});

    /// How each mesh axis divides into lanes and warps.
    ///
    /// A thread mesh is row-major, so axis ``i``'s coordinate is multiplied by
    /// the product of the extents *after* it: the axis is lanes while that
    /// product times its own extent still fits in 32, warps once the product
    /// alone reaches 32, and otherwise **splits** -- ``32 / stride`` of it is
    /// lanes and the rest warps.
    int lanes_of[m_rank] = {};
    int warps_of[m_rank] = {};
    int stride = 1;
    /// The split is the case the greedy walk this replaces had no room for: it
    /// counted whole axes from the fast end while their running product fit in
    /// a warp and stopped at the first that did not, so a ``(256,)`` mesh --
    /// the shape ``program_shape<thread>()`` returns -- yielded no lane axes
    /// and its reduced axis counted as a warp axis: ``warps_per_group == 256``
    /// where the block has 8 warps, a workspace read at ``0..255``, and a mean
    /// divided by 32 times too much.
    for (int i = m_rank - 1; i >= 0; --i) {
        if (is_thread) {
            const int room =
                tilefoundry::detail::MeshWarpView<mesh_t>::lane_extent / stride;
            const int lanes =
                room <= 1 ? 1 : (m_ext[i] < room ? m_ext[i] : room);
            lanes_of[i] = lanes;
            warps_of[i] = m_ext[i] / lanes;
        } else {
            /// A CTA mesh has no lanes: every one of its instances is a
            /// separate block, so a reduced axis crosses whatever the level
            /// above a warp is and the cross-warp tier is the only answer.
            lanes_of[i] = 1;
            warps_of[i] = m_ext[i];
        }
        stride *= m_ext[i];
    }

    bool lane_reduced = false;
    bool mesh_reduced = false;
    bool warp_aligned = true;
    int warps_per_group = 1;
    int lanes_reduced = 1;
    for (int i = 0; i < m_rank; ++i) {
        if (!reduced[i])
            continue;
        mesh_reduced = true;
        if (m_ext[i] % lanes_of[i] != 0)
            warp_aligned = false;
        if (lanes_of[i] > 1) {
            lane_reduced = true;
            lanes_reduced *= lanes_of[i];
        }
        warps_per_group *= warps_of[i];
    }
    return {lane_reduced, warps_per_group,     lanes_reduced, mesh_reduced,
            warp_aligned, warps_per_group > 1, false,         false};
}

template <class SrcSL, class DstSL> CUTE_HOST_DEVICE constexpr plan_t plan() {
    return reduce_dispatch<SrcSL, DstSL>();
}

/// Detector for a nested ``typename T::shard_layout_type``. Selects the sharded
/// tiers vs. the plain (non-sharded) path in the public ``reduce`` entry.

}
