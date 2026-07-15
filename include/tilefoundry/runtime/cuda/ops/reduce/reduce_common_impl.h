// reduce common impl — shared helpers, dispatch trait, and layout detector.
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
// and, e.g., ``CrossWarp<mean_op>`` would trip its SUM/MAX-only guard.
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
