// reduce common impl — shared per-tag traits, per-thread fold, cell
// decomposition, and the sharded-reduce dispatch trait / layout detector.
//
// Included in-context from ``ops/reduce.cuh`` (which is itself included inside
// ``namespace tilefoundry::ops`` from runtime.cuh). This header therefore does
// NOT open ``namespace tilefoundry`` / ``ops`` and does NOT pull in system
// headers — cute/std and the surrounding names (``detail::to_local``,
// ``shard::S``/``shard::B``, ``TopologyScope``) are already in scope.
#pragma once

namespace reduce_impl {

// Workspace tag used when no shared-memory staging is needed (every
// reduce mesh axis lives inside a single warp).
struct no_workspace_t {};
inline constexpr no_workspace_t no_workspace{};

// ── Per-tag combine semantics ──────────────────────────────────────
// One trait per Op tag, shared by every reduce tier (plain, intra-warp,
// intra-cta, cross-warp). ``init`` seeds the fold; ``elem(x)`` maps a raw
// source element into the fold domain (identity for sum/mean, ``|x|`` for
// absmax); ``combine(a, b)`` merges two fold-domain values (``+`` for
// sum/mean, ``fmaxf`` for absmax — both have ``init`` as their identity
// element over this domain, since absmax's domain is always non-negative);
// ``finalize(acc, count)`` produces the per-cell result (mean divides by the
// reduced element count; sum/absmax pass the accumulator through unchanged).
template <class Op> struct reduce_traits;

template <> struct reduce_traits<sum_op> {
    static constexpr float init = 0.f;
    __device__ static float elem(float x) { return x; }
    __device__ static float combine(float a, float b) { return a + b; }
    __device__ static float finalize(float acc, float /*count*/) { return acc; }
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
    __device__ static float finalize(float acc, float /*count*/) { return acc; }
};

template <class Op>
inline constexpr bool is_supported_reduce_op_v =
    std::is_same_v<Op, sum_op> || std::is_same_v<Op, mean_op> ||
    std::is_same_v<Op, absmax_op>;

// Per-thread cell decomposition, shared by every reduce tier: the per-thread
// cute tensor has ``size(s)`` source elements feeding ``size(d)`` destination
// cells; the source is treated as ``n_cells`` contiguous chunks of ``step``
// elements, each chunk reducing to one output cell (``n_cells == 1`` when
// ``d`` is a single scalar cell, e.g. dst rank 0/1).
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

// Rank-aware per-thread local fold: folds the ``step`` source elements of
// logical cell ``j`` (out of ``cell_decomp``'s ``n_cells``) into a single
// scalar accumulator via ``reduce_traits<Op>``.
//
// ``step == cute::size(s)`` is the single-cell / whole-tensor contract (Plain
// with a scalar dst, and IntraCta — whose per-thread tensor has no dedicated
// cell axis at all): it flattens across cute's *entire* multi-mode domain via
// single-index addressing (``s(k)``), which stays correct however many cute
// axes the per-thread layout carries — including residual axes introduced by
// single-axis sugar factorisation (spec shard §7.1.2) — because ``s(k)``
// always visits every coordinate of ``s`` exactly once regardless of its mode
// structure.
//
// Otherwise (``n_cells > 1``, e.g. IntraWarp / CrossWarp, where each thread
// owns several distinct output cells), cell-aligned 2-D access (``s(j, k)``)
// keeps cell ``j`` on its own row of the per-thread cute layout, regardless
// of cute's col-major linearisation: a linear ``s(j*step+k)`` over the full
// tensor mis-aligns here because it reinterprets the (cells, step) mode pair
// as one flat dimension in the wrong order. The single-axis path (rank-1
// ``s``) uses ``s(j*step+k)`` directly in both cases — unambiguous, since
// there is only one mode to address.
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

// Intra-warp butterfly reduction (32 lanes → broadcast combine), using
// ``Op``'s combine (``+`` for sum/mean, ``fmaxf`` for absmax).
template <class Op> __device__ float warp_butterfly(float val) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        val = reduce_traits<Op>::combine(
            val, __shfl_xor_sync(0xFFFFFFFFu, val, delta));
    }
    return val;
}

// Cross-warp SUM aggregation via a shared-memory workspace.
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

// ── Layered sharded-reduce dispatch ───────────────────────────────
// Compile-time derivation of the reduction level and ``warps_per_group`` from
// the operand shard layouts, consumed by the public ``reduce`` entry.
template <class T> struct is_split_attr : std::false_type {};
template <int A> struct is_split_attr<shard::S<A>> : std::true_type {};

struct reduce_dispatch_info {
    bool lane_reduced;
    int warps_per_group;
};

// Derive, from the (src, dst) operand ShardLayouts, the active reduction level
// and its ``warps_per_group``. Pure compile-time so the caller can select the
// tier with ``if constexpr`` — otherwise the untaken tier still instantiates
// and, e.g., ``CrossWarp<mean_op>`` would trip its supported-op guard.
// Requires a static mesh layout (the reduce mesh is a thread-scoped static
// mesh); a reduced axis on a non-thread mesh scope yields the cross-warp tier.
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

    // Lane axes: rightmost warp-sized suffix of a thread-scoped mesh.
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

// Detector for a nested ``typename T::shard_layout_type``. Selects the sharded
// tiers vs. the plain (non-sharded) path in the public ``reduce`` entry.
template <class T, class = void> struct has_shard_layout : std::false_type {};
template <class T>
struct has_shard_layout<T, std::void_t<typename T::shard_layout_type>>
    : std::true_type {};
template <class T>
inline constexpr bool has_shard_layout_v = has_shard_layout<T>::value;

} // namespace reduce_impl
