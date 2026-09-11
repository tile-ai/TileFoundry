/// reduce common impl — shared per-tag traits, per-thread fold, cell
/// decomposition, and the sharded-reduce dispatch trait / layout detector.
///
/// Included in-context from ``ops/reduce.cuh`` (which is itself included inside
/// ``namespace tilefoundry::ops`` from runtime.cuh). This header therefore does
/// NOT open ``namespace tilefoundry`` / ``ops`` and does NOT pull in system
/// headers — cute/std and the surrounding names (``tilefoundry::local_tensor``,
/// ``shard::S``/``shard::B``, ``TopologyScope``) are already in scope.
#pragma once

namespace reduce_impl {

/// Workspace tag used when no shared-memory staging is needed (every
/// reduce mesh axis lives inside a single warp).
struct no_workspace_t {};
inline constexpr no_workspace_t no_workspace{};

/// Reduction traits define initialization, merge, and finalization.
template <class Op> struct reduce_traits;
template <> struct reduce_traits<primitive::add_op> {
    using combine_op = primitive::add_op;
    static constexpr float init = 0.f;
    __device__ static float elem(float x) { return x; }
    __device__ static float finalize(float acc, float) { return acc; }
};
template <> struct reduce_traits<mean_op> {
    using combine_op = primitive::add_op;
    static constexpr float init = 0.f;
    __device__ static float elem(float x) { return x; }
    __device__ static float finalize(float acc, float n) { return acc / n; }
};
template <> struct reduce_traits<primitive::max_op> {
    using combine_op = primitive::max_op;
    static constexpr float init = -INFINITY;
    __device__ static float elem(float x) { return x; }
    __device__ static float finalize(float acc, float) { return acc; }
};
template <> struct reduce_traits<primitive::min_op> {
    using combine_op = primitive::min_op;
    static constexpr float init = INFINITY;
    __device__ static float elem(float x) { return x; }
    __device__ static float finalize(float acc, float) { return acc; }
};
template <> struct reduce_traits<absmax_op> {
    using combine_op = primitive::max_op;
    static constexpr float init = 0.f;
    __device__ static float elem(float x) { return fabsf(x); }
    __device__ static float finalize(float acc, float) { return acc; }
};

/// Merge two partials with this reduction's own functor.
template <class Op> __device__ float combine(float a, float b) {
    return typename reduce_traits<Op>::combine_op{}(a, b);
}

/// Whether every value of ``T`` survives a round trip through ``float``.
template <class T> CUTE_HOST_DEVICE constexpr bool folds_in_float() {
    if constexpr (std::is_integral_v<T>)
        return sizeof(T) <= 2;
    else if constexpr (std::is_floating_point_v<T>)
        return sizeof(T) <= sizeof(float);
    else
        return sizeof(T) <= 2 && std::is_convertible_v<T, float>;
}

/// What both operands have to be for any tier to be able to run.
template <class Src, class Dst>
CUTE_HOST_DEVICE constexpr void check_reduce_domain() {
    using s_view = tilefoundry::local_view_t<Src>;
    using d_view = tilefoundry::local_view_t<Dst>;
    static_assert(
        folds_in_float<typename s_view::value_type>(),
        "ops::reduce: the element type must be one float holds exactly");
    static_assert(
        folds_in_float<typename d_view::value_type>(),
        "ops::reduce: the destination's element type must be one a float "
        "can be stored into without loss");
    static_assert(
        !cute::is_composed_layout<typename s_view::layout_type>::value,
        "ops::reduce: the source's projected layout must be a plain "
        "cute::Layout");
}

template <class Op>
inline constexpr bool is_supported_reduce_op_v =
    std::is_same_v<Op, primitive::add_op> || std::is_same_v<Op, mean_op> ||
    std::is_same_v<Op, absmax_op> || std::is_same_v<Op, primitive::max_op> ||
    std::is_same_v<Op, primitive::min_op>;

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
/// Select kept or reduced layout modes from Axes.
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
/// Combine shared-workspace partials by warp group.
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
/// Classify each mesh axis from the operand shard attributes.

using detail::is_partial_attr_v;
using detail::is_split_attr_v;

/// An axis whose instances hold pieces of one value, either kind.
template <class T>
inline constexpr bool is_reducible_attr_v =
    is_split_attr_v<T> || is_partial_attr_v<T>;

struct reduce_dispatch_info {
    bool lane_reduced;
    int warps_per_group;
    /// Count lanes contributing to one value.
    int lanes_reduced;
    /// Whether the reduction crosses any mesh axis.
    bool mesh_reduced;
    /// Whether every reduced axis divides into whole lanes and whole warps.
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
/// Lanes of one warp this thread mesh hands out, and zero when its ids state
/// no ``(lanes, warps)`` at all.
template <class TMesh> CUTE_HOST_DEVICE constexpr int lanes_of_mesh() {
    if constexpr (TMesh::scope == TopologyScope::thread &&
                  tilefoundry::is_warped(TMesh{}))
        return kWarpSize;
    else
        return 0;
}

template <class SrcSL, class DstSL>
CUTE_HOST_DEVICE constexpr reduce_dispatch_info reduce_dispatch() {
    using src_attrs = typename SrcSL::attrs;
    using dst_attrs = typename DstSL::attrs;
    using mesh_t = typename SrcSL::mesh;
    static_assert(mesh_t::scope == TopologyScope::thread ||
                      mesh_t::scope == TopologyScope::cta,
                  "ops::reduce: mesh must name cta or thread");
    constexpr bool is_thread = mesh_t::scope == TopologyScope::thread;
    using m_layout_t = typename mesh_t::layout_type;
    constexpr int m_rank = cute::tuple_size<src_attrs>::value;
    static_assert(detail::shard_attrs_match_mesh<SrcSL>() &&
                      detail::shard_attrs_match_mesh<DstSL>(),
                  "ops::reduce: both operands need one attr per mesh axis");
    static_assert(std::is_same_v<typename SrcSL::mesh, typename DstSL::mesh>,
                  "ops::reduce: the two operands must name one mesh");
    static_assert(is_thread || mesh_t::scope == TopologyScope::cta,
                  "ops::reduce: a reduce mesh's scope must be cta or thread");

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
    int lanes_of[m_rank] = {};
    int warps_of[m_rank] = {};
    int stride = 1;
    for (int i = m_rank - 1; i >= 0; --i) {
        if (is_thread) {
            const int room = lanes_of_mesh<mesh_t>() / stride;
            const int lanes =
                room <= 1 ? 1 : (m_ext[i] < room ? m_ext[i] : room);
            lanes_of[i] = lanes;
            warps_of[i] = m_ext[i] / lanes;
        } else {
            /// CTA meshes reduce across CTAs rather than warp lanes.
            lanes_of[i] = 1;
            warps_of[i] = m_ext[i];
        }
        stride *= m_ext[i];
    }

    bool lane_reduced = false;
    bool mesh_reduced = false;
    /// A mesh whose ids state no (lanes, warps) divides into neither, so it
    /// fails the same check a ragged axis does and says the same thing.
    bool warp_aligned = !is_thread || lanes_of_mesh<mesh_t>() > 0;
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
