/// CUDA dot op implementation. Included in-context from ops/dot.cuh inside
/// namespace tilefoundry::ops.
#pragma once

namespace dot_impl {

/// The elements per wide read, from the two operands' own layouts.
///
/// CuTe's formula (``algorithm/copy.hpp``): the run both sides share, capped by
/// what both are aligned for and by the 16-byte hardware maximum, and cut down
/// until it divides the run -- the fold walks recast units and has no tail.
///
/// Computed here rather than shared with ``copy``, which no longer needs it:
/// ``cute::copy`` asks CuTe the same question internally. A fold has no CuTe
/// equivalent to ask, so this is the one place left that states the formula.
template <class AView, class BView>
CUTE_HOST_DEVICE constexpr int fold_width() {
    using AL = typename cute::remove_cvref_t<AView>::layout_type;
    using BL = typename cute::remove_cvref_t<BView>::layout_type;
    using a_val = typename cute::remove_cvref_t<AView>::value_type;
    using b_val = typename cute::remove_cvref_t<BView>::value_type;
    if constexpr (!cute::is_static<AL>::value || !cute::is_static<BL>::value ||
                  cute::is_rmem<cute::remove_cvref_t<AView>>::value ||
                  cute::is_rmem<cute::remove_cvref_t<BView>>::value) {
        return 1;
    } else {
        constexpr int run = int(
            decltype(cute::gcd(cute::max_common_vector(AL{}, BL{}),
                               cute::gcd(cute::max_alignment(AL{}),
                                         cute::max_alignment(BL{}))))::value);
        constexpr int wide =
            int(sizeof(a_val) > sizeof(b_val) ? sizeof(a_val) : sizeof(b_val));
        constexpr int n = int(decltype(cute::size(AL{}))::value);
        int v = 1;
        while (v * 2 <= run && v * 2 * wide <= 16 && n % (v * 2) == 0)
            v *= 2;
        return v;
    }
}

/// Independent partial sums, so a row is not one chain of dependent FMAs.
///
/// A single running accumulator makes the whole row serial and the loads cannot
/// run ahead of it: what the warp waits on is then the row's latency, not its
/// bytes. Eight is enough to cover the FMA pipeline and few enough not to cost
/// occupancy. The tree at the end is this op's summation order, so a result is
/// reproducible without knowing the row length.
struct Partials {
    static constexpr int kWays = 8;
    float p[kWays];

    __device__ Partials() {
        CUTE_UNROLL
        for (int i = 0; i < kWays; ++i)
            p[i] = 0.f;
    }
    __device__ float total() const {
        return ((p[0] + p[1]) + (p[2] + p[3])) +
               ((p[4] + p[5]) + (p[6] + p[7]));
    }
};

/// This lane's share of the product, folded.
///
/// Not named ``fold``: the operands are ``cute::Tensor``s, so ADL puts
/// ``cute::fold`` in the overload set, where its unconstrained three-argument
/// form wins and the error lands inside CuTe.
///
/// ``V`` comes from the layouts, so neither width nor stride appears at the
/// call site. The wide read ``recast``s the two views, not their pointers:
/// recasting reshapes the layout, so a view with a slow mode keeps it, where
/// casting the pointer reads the view as one byte run, past the fast mode.
template <class AView, class BView>
__device__ float contract(AView const &a, BView const &b, int n) {
    using a_val = typename AView::value_type;
    using b_val = typename BView::value_type;
    constexpr int V = fold_width<AView, BView>();
    Partials acc;
    if constexpr (V > 1) {
        auto av = cute::recast<cute::uint_bit_t<V *int(sizeof(a_val)) * 8>>(a);
        auto bv = cute::recast<cute::uint_bit_t<V *int(sizeof(b_val)) * 8>>(b);
        const int nv = int(cute::size(av));
        for (int i = 0; i < nv; ++i) {
            auto ai = av(i);
            auto bi = bv(i);
            auto const *ap = reinterpret_cast<a_val const *>(&ai);
            auto const *bp = reinterpret_cast<b_val const *>(&bi);
            CUTE_UNROLL
            for (int k = 0; k < V; ++k)
                acc.p[k % Partials::kWays] +=
                    static_cast<float>(ap[k]) * static_cast<float>(bp[k]);
        }
    } else {
        for (int i = 0; i < n; ++i)
            acc.p[i % Partials::kWays] +=
                static_cast<float>(a(i)) * static_cast<float>(b(i));
    }
    return acc.total();
}

/// The extent of the mesh axis a warp's lanes are: the last one, because a
/// thread mesh is row-major and its fastest axis is the one adjacent thread ids
/// walk.
template <class T> CUTE_HOST_DEVICE constexpr int lane_axis_extent() {
    using sl_t = typename cute::remove_cvref_t<T>::shard_layout_type;
    using shape_t = cute::remove_cvref_t<decltype(cute::shape(
        tilefoundry::detail::mesh_positions_t<typename sl_t::mesh::layout>{}))>;
    return int(cute::get<cute::tuple_size<shape_t>::value - 1>(shape_t{}));
}

/// The contraction lives inside a warp: one butterfly finishes it.
///
/// Which tier runs is decided by the caller's argument list -- a workspace
/// passed or not -- so the mesh shape a butterfly needs is this tier's own to
/// state.
struct Warp {
    template <class Lhs, class Rhs, class Dst>
    __device__ void operator()(Lhs const &lhs, Rhs const &rhs, Dst &dst) const {
        static_assert(tilefoundry::shard_mesh_instances<Lhs>() <= kWarpSize,
                      "ops::dot: a cross-warp mesh requires workspace");
        static_assert(lane_axis_extent<Lhs>() == kWarpSize,
                      "ops::dot (warp tier): the fastest axis of the operands' "
                      "mesh must be exactly one warp of 32 lanes -- the "
                      "butterfly here is 32 lanes wide, so a narrower axis "
                      "such as (2, 16) mixes two contraction groups in one "
                      "butterfly and leaves each holding both sums, and a "
                      "wider one such as a flat (256,) spreads one contraction "
                      "over warps the butterfly never crosses. Crossing them "
                      "is the block tier and its workspace");
        auto a = detail::to_local(lhs);
        auto b = detail::to_local(rhs);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        const float sum = tilefoundry::warp_reduce<add_op>(
            contract(a, b, int(cute::size(a))));
        d(0) = static_cast<value_type>(sum);
    }
};

/// The contraction spans the block: each warp posts a partial, then every
/// thread folds the posted ones.
///
/// The second fold is over one value per warp, which is short enough that a
/// serial loop beats a second butterfly, and it leaves the total in every
/// thread rather than in one. Both the barrier and the count of warps to fold
/// come off the operands' own mesh, since the warps that post are exactly the
/// ones it names: ``blockDim.x >> 5`` is the whole block, and folds slots no
/// warp wrote wherever the mesh is narrower -- ``ops::mma`` asks the same way.
struct Cta {
    template <class Lhs, class Rhs, class Dst, class Ws>
    __device__ void operator()(Lhs const &lhs, Rhs const &rhs, Dst &dst,
                               Ws &ws) const {
        auto a = detail::to_local(lhs);
        auto b = detail::to_local(rhs);
        auto &&d = detail::to_local(dst);
        auto &&slots = detail::to_local(ws);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        constexpr int instances = tilefoundry::shard_mesh_instances<Lhs>();
        static_assert(instances >= kWarpSize && instances % kWarpSize == 0,
                      "ops::dot (block tier): the operands' mesh must be a "
                      "whole number of warps -- each warp posts one partial "
                      "and the fold reads one slot per warp, so a fraction of "
                      "a warp rounds that count of posted partials to zero, "
                      "posts nothing, is summed as nothing, and leaves every "
                      "participant holding 0.f");
        constexpr int warps = instances / kWarpSize;
        const float part = tilefoundry::warp_reduce<add_op>(
            contract(a, b, int(cute::size(a))));
        const unsigned tid = unsigned(
            tilefoundry::program_id<tilefoundry::TopologyScope::thread>());
        if ((tid & unsigned(kWarpSize - 1)) == 0u)
            slots(int(tid / unsigned(kWarpSize))) = part;
        ops::sync(lhs.shard_layout.mesh_value);
        float sum = 0.f;
        for (int w = 0; w < warps; ++w)
            sum += float(slots(w));
        d(0) = static_cast<value_type>(sum);
    }
};

}
