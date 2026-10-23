/// tilefoundry reduce op — single public ``reduce`` entry.
///
/// This file is included IN-CONTEXT from runtime.cuh, at the point inside
/// ``namespace tilefoundry::ops`` where the reduce surface used to live. It
/// therefore does NOT open ``namespace tilefoundry`` / ``ops`` and pulls in
/// no system headers — cute/std and the surrounding names
/// (``local_tensor``, ``shard::S``/``shard::B``, ``TopologyScope``) are
/// already in scope. The op tags below must precede the impl-header includes
/// because ``reduce_impl::reduce_traits<Op>`` specializes on them.
#pragma once

/// Reduce combine-kind tags — pure compile-time markers. Semantics (init
/// value, elem/combine/finalize) live in one place per tag,
/// ``reduce_impl::reduce_traits<Op>`` (reduce/reduce_common_impl.h), consumed
/// uniformly by all four reduce tiers below.
struct mean_op {};
struct absmax_op {};

/// ── Sharded reduce ───────────────────────────────
/// Per-tier reduce building blocks; the public ``reduce`` entry below selects a
/// tier from the operand shard layouts. MEAN folds as SUM plus a final divide
/// by the total reduced extent.
#include "reduce/reduce_common_impl.h"
#include "reduce/reduce_intra_warp_impl.h"
#include "reduce/reduce_intra_cta_impl.h"
#include "reduce/reduce_cross_warp_impl.h"
#include "reduce/reduce_plain_impl.h"

/// Single public reduce entry: ``dst = reduce_kind(src)`` over ``Axes``.
///
/// Dispatch the reduction tier from operand shard layouts.
template <class Op, class Axes, class Src, class Dst,
          class Ws = reduce_impl::no_workspace_t>
__device__ inline void reduce(Src const &src, Dst &dst, Ws &&ws = {}) {
    reduce_impl::check_reduce_domain<Src, Dst>();
    if constexpr (tilefoundry::ShardTensorLike<Src>) {
        using SLs = typename cute::remove_cvref_t<Src>::shard_layout_type;
        using SLd = typename cute::remove_cvref_t<Dst>::shard_layout_type;
        constexpr reduce_impl::reduce_dispatch_info plan =
            reduce_impl::reduce_dispatch<SLs, SLd>();
        constexpr bool has_ws = !std::is_same_v<cute::remove_cvref_t<Ws>,
                                                reduce_impl::no_workspace_t>;
        static_assert(plan.warp_aligned,
                      "ops::reduce: a reduced mesh axis must divide into whole "
                      "lanes and whole warps");
        if constexpr (!plan.mesh_reduced) {
            /// Reduce without crossing mesh instances.
            reduce_impl::Plain<Op, Axes>{}(src, dst);
        } else if constexpr (plan.warps_per_group == 1) {
            /// Reduce within one warp.
            reduce_impl::IntraWarp<Op, Axes>{}(src, dst);
        } else if constexpr (!has_ws) {
            static_assert(
                dependent_false_v<Src>,
                "ops::reduce: these shard layouts spread one value over "
                "more than one warp");
        } else if constexpr (plan.lane_reduced) {
            reduce_impl::IntraCta<Op, Axes>{}(src, dst, ws,
                                              plan.warps_per_group);
        } else {
            reduce_impl::CrossWarp<Op, Axes>{}(src, dst, ws,
                                               plan.warps_per_group);
        }
    } else {
        reduce_impl::Plain<Op, Axes>{}(src, dst);
    }
}
