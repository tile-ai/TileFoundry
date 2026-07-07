// tilefoundry reduce op — single public ``reduce`` entry.
//
// This file is included IN-CONTEXT from runtime.cuh, at the point inside
// ``namespace tilefoundry::ops`` where the reduce surface used to live. It
// therefore does NOT open ``namespace tilefoundry`` / ``ops`` and pulls in no
// system headers — cute/std and the surrounding names (``detail::to_local``,
// ``shard::S``/``shard::B``, ``TopologyScope``) are already in scope. The op
// tags below must precede the impl-header includes because the tier impl
// classes switch on them.
#pragma once

struct mean_op {
    template <class T> __device__ T operator()(T sum, T count) const {
        return sum / count;
    }
};
struct sum_op {
    template <class T> __device__ T operator()(T sum, T) const { return sum; }
};
// Max-reduction tag. Consumed as a compile-time ``Op`` type by the sharded
// reduce templates (their ``if constexpr`` / ``static_assert`` branches switch
// on it); the reduce entry points support the SUM and MAX combines.
struct rmax_op {
    template <class T> __device__ T operator()(T acc, T) const { return acc; }
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
// Per-tier reduce building blocks; the public ``reduce`` entry below selects a
// tier from the operand shard layouts. MEAN folds as SUM plus a final divide by
// the total reduced extent.
#include "reduce/reduce_common_impl.h"
#include "reduce/reduce_intra_warp_impl.h"
#include "reduce/reduce_intra_cta_impl.h"
#include "reduce/reduce_cross_warp_impl.h"
#include "reduce/reduce_plain_impl.h"

// Single public reduce entry: ``dst = reduce_kind(src)`` over ``Axes``.
//
// Sharded operands (a nested ``shard_layout_type``) select an intra-warp,
// intra-CTA, or cross-warp tier: with no workspace the reduce mesh lives inside
// a single warp (intra-warp); otherwise the (src, dst) shard layouts pick the
// intra-CTA (lane-reduced) vs. cross-warp tier and its ``warps_per_group``.
// Non-sharded operands take the plain rank-aware fold.
template <class Op, class Axes, class Src, class Dst,
          class Ws = reduce_impl::no_workspace_t>
__device__ inline void reduce(Src const &src, Dst &dst, Ws &&ws = {}) {
    if constexpr (reduce_impl::has_shard_layout_v<cute::remove_cvref_t<Src>>) {
        using SLs = typename cute::remove_cvref_t<Src>::shard_layout_type;
        using SLd = typename cute::remove_cvref_t<Dst>::shard_layout_type;
        if constexpr (std::is_same_v<cute::remove_cvref_t<Ws>,
                                     reduce_impl::no_workspace_t>) {
            reduce_impl::IntraWarp<Op, Axes>{}(src, dst);
        } else {
            constexpr reduce_impl::reduce_dispatch_info info =
                reduce_impl::reduce_dispatch<SLs, SLd>();
            if constexpr (info.lane_reduced)
                reduce_impl::IntraCta<Op, Axes>{}(src, dst, ws,
                                                  info.warps_per_group);
            else
                reduce_impl::CrossWarp<Op, Axes>{}(src, dst, ws,
                                                   info.warps_per_group);
        }
    } else {
        reduce_impl::Plain<Op, Axes>{}(src, dst);
    }
}
