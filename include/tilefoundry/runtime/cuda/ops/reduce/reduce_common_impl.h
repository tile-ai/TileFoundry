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

/// Combine semantics shared by every reduction tier.
/// ``init`` seeds the fold, ``elem`` enters the fold domain, and ``combine``
/// merges values. ``finalize`` divides means by the element count while sum
/// and absmax preserve their accumulator.
template <class Op> struct reduce_traits;

template <> struct reduce_traits<sum_op> {
    static constexpr float init = 0.f;
    __device__ static float elem(float x) { return x; }
    __device__ static float combine(float a, float b) { return a + b; }
    __device__ static float finalize(float acc, float) { return acc; }
};

template <> struct reduce_traits<mean_op> {
    static constexpr float init = 0.f;
    __device__ static float elem(float x) { return x; }
    __device__ static float combine(float a, float b) { return a + b; }
    __device__ static float finalize(float acc, float count) {
        return acc / count;
    }
};

template <> struct reduce_traits<absmax_op> {
    static constexpr float init = 0.f;
    __device__ static float elem(float x) { return fabsf(x); }
    __device__ static float combine(float a, float b) { return fmaxf(a, b); }
    __device__ static float finalize(float acc, float) { return acc; }
};

template <class Op>
inline constexpr bool is_supported_reduce_op_v =
    std::is_same_v<Op, sum_op> || std::is_same_v<Op, mean_op> ||
    std::is_same_v<Op, absmax_op>;

/// Per-thread cell decomposition, shared by every reduce tier: the per-thread
/// cute tensor has ``size(s)`` source elements feeding ``size(d)`` destination
/// cells; the source is treated as ``n_cells`` contiguous chunks of ``step``
/// elements, each chunk reducing to one output cell (``n_cells == 1`` when
/// ``d`` is a single scalar cell, e.g. dst rank 0/1).
struct cell_decomp_t {
    int n_src;
    int n_dst;
    int n_cells;
    int step;
};

template <class SrcT, class DstT>
__device__ cell_decomp_t cell_decomp(SrcT const &s, DstT const &d) {
    const int n_src = static_cast<int>(cute::size(s));
    const int n_dst = static_cast<int>(cute::size(d));
    const int n_cells = (n_dst == 0) ? 1 : n_dst;
    const int step = n_src / n_cells;
    return {n_src, n_dst, n_cells, step};
}

/// Fold one logical cell's ``step`` source elements through ``reduce_traits``.
/// Whole-tensor folds use flat ``s(k)`` addressing across every mode.
/// Multi-cell folds use ``s(j, k)`` so CuTe column-major linearization cannot
/// mix rows; rank-one tensors use the unambiguous flat ``s(j*step+k)`` form.
///
/// See [shard §7.1.2](docs/spec/shard.md#712-layoutstrides).
template <class Op, class SrcT>
__device__ float local_fold(SrcT const &s, int j, int step) {
    using traits = reduce_traits<Op>;
    float acc = traits::init;
    constexpr int s_rank = decltype(cute::rank(s))::value;
    if constexpr (s_rank > 1) {
        if (step == static_cast<int>(cute::size(s))) {
            for (int k = 0; k < step; ++k) {
                acc = traits::combine(acc,
                                      traits::elem(static_cast<float>(s(k))));
            }
        } else {
            for (int k = 0; k < step; ++k) {
                acc = traits::combine(
                    acc, traits::elem(static_cast<float>(s(j, k))));
            }
        }
    } else {
        const int base = j * step;
        for (int k = 0; k < step; ++k) {
            acc = traits::combine(
                acc, traits::elem(static_cast<float>(s(base + k))));
        }
    }
    return acc;
}

/// Intra-warp butterfly reduction (32 lanes → broadcast combine), using
/// ``Op``'s combine (``+`` for sum/mean, ``fmaxf`` for absmax).
template <class Op> __device__ float warp_butterfly(float val) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        val = reduce_traits<Op>::combine(
            val, __shfl_xor_sync(0xFFFFFFFFu, val, delta));
    }
    return val;
}

/// Cross-warp SUM aggregation via a shared-memory workspace.
///
/// ``workspace`` is sized to ``total_warps`` (all non-thread mesh
/// positions).  ``warps_per_group`` (≤ total_warps) controls grouping:
/// warps are partitioned into contiguous groups of ``warps_per_group``
/// slots, and each thread only aggregates across its own group.
/// When ``warps_per_group == total_warps`` (single group), this is
/// equivalent to the original flat cross-warp reduce.
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

/// ── Layered sharded-reduce dispatch ───────────────────────────────
/// Compile-time derivation of the reduction level and ``warps_per_group`` from
/// the operand shard layouts, consumed by the public ``reduce`` entry.
template <class T> struct is_split_attr : std::false_type {};
template <int A> struct is_split_attr<shard::S<A>> : std::true_type {};

struct reduce_dispatch_info {
    bool lane_reduced;
    int warps_per_group;
};

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
    constexpr auto scope = mesh_t::topology::scope;
    using m_layout_t = typename mesh_t::layout;
    constexpr int m_rank = cute::tuple_size<src_attrs>::value;

    int m_ext[m_rank] = {};
    bool reduced[m_rank] = {};
    auto const m_shape = cute::shape(m_layout_t{});
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((m_ext[Is] = int(cute::get<Is>(m_shape))), ...);
        ((reduced[Is] =
              is_split_attr<cute::remove_cvref_t<decltype(cute::get<Is>(
                  src_attrs{}))>>::value &&
              std::is_same_v<
                  cute::remove_cvref_t<decltype(cute::get<Is>(dst_attrs{}))>,
                  shard::B>),
         ...);
    }(std::make_index_sequence<m_rank>{});

    int thread_axes = 0;
    if (scope == TopologyScope::thread) {
        int prod = 1;
        for (int i = m_rank - 1; i >= 0; --i) {
            if (prod * m_ext[i] > 32)
                break;
            prod *= m_ext[i];
            ++thread_axes;
        }
    }

    bool lane_reduced = false;
    int warps_per_group = 1;
    for (int i = 0; i < m_rank; ++i) {
        const bool is_lane = (i >= m_rank - thread_axes);
        if (is_lane) {
            if (reduced[i])
                lane_reduced = true;
            continue;
        }
        if (reduced[i])
            warps_per_group *= m_ext[i];
    }
    return {lane_reduced, warps_per_group};
}

/// Detector for a nested ``typename T::shard_layout_type``. Selects the sharded
/// tiers vs. the plain (non-sharded) path in the public ``reduce`` entry.
template <class T, class = void> struct has_shard_layout : std::false_type {};
template <class T>
struct has_shard_layout<T, std::void_t<typename T::shard_layout_type>>
    : std::true_type {};
template <class T>
inline constexpr bool has_shard_layout_v = has_shard_layout<T>::value;

}
