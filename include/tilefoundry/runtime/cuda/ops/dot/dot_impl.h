/// CUDA dot implementation included in tilefoundry::ops.
#pragma once

namespace dot_impl {

/// The elements per wide read, from the two operands' own layouts.
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
struct Partials {
    static constexpr int kWays = 8;
    float p[kWays] = {};
};

/// The sums added back, in pairs, so the tree stays as shallow as the ways.
CUTE_HOST_DEVICE float total(Partials const &acc) {
    return ((acc.p[0] + acc.p[1]) + (acc.p[2] + acc.p[3])) +
           ((acc.p[4] + acc.p[5]) + (acc.p[6] + acc.p[7]));
}

/// This lane's share of the product, folded.
template <class AView, class BView>
__device__ float contract(AView const &a, BView const &b, int n) {
    using a_val = typename AView::value_type;
    using b_val = typename BView::value_type;
    constexpr int V = fold_width<AView, BView>();
    Partials acc{};
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
    return total(acc);
}

/// Extent of the thread mesh's fastest axis, as the mesh states it.
///
/// The axis, not the ids it reaches: a butterfly runs one axis, so a mesh
/// written ``(2,16)`` spans sixteen lanes twice even where those 32 threads
/// are one whole warp. ``id_axes`` folds that distinction away, so read the
/// mesh's own last axis -- row-major, so the fastest is last.
template <class T> CUTE_HOST_DEVICE constexpr int lane_axis_extent() {
    using mesh_t = typename cute::remove_cvref_t<T>::shard_layout_type::mesh;
    using axes_t = cute::remove_cvref_t<decltype(cute::flatten(
        detail::positions_of(typename mesh_t::layout_type{})))>;
    return int(cute::get<decltype(cute::rank(axes_t{}))::value - 1>(
        cute::shape(axes_t{})));
}

/// The contraction lives inside a warp: one butterfly finishes it.
struct Warp {
    template <class Lhs, class Rhs, class Dst>
    __device__ void operator()(Lhs const &lhs, Rhs const &rhs, Dst &dst) const {
        static_assert(tilefoundry::shard_mesh_instances<Lhs>() <= kWarpSize,
                      "ops::dot: a cross-warp mesh requires workspace");
        static_assert(lane_axis_extent<Lhs>() == kWarpSize,
                      "ops::dot (warp tier): the fastest axis of the operands' "
                      "mesh must be exactly one warp of 32 lanes");
        auto a = detail::local_tensor(lhs);
        auto b = detail::local_tensor(rhs);
        auto &&d = detail::local_tensor(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        const float sum = tilefoundry::warp_reduce<add_op>(
            contract(a, b, int(cute::size(a))));
        d(0) = static_cast<value_type>(sum);
    }
};

/// Fold one partial per warp through shared workspace.
struct Cta {
    template <class Lhs, class Rhs, class Dst, class Ws>
    __device__ void operator()(Lhs const &lhs, Rhs const &rhs, Dst &dst,
                               Ws &ws) const {
        auto a = detail::local_tensor(lhs);
        auto b = detail::local_tensor(rhs);
        auto &&d = detail::local_tensor(dst);
        auto &&slots = detail::local_tensor(ws);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        constexpr int instances = tilefoundry::shard_mesh_instances<Lhs>();
        static_assert(instances >= kWarpSize && instances % kWarpSize == 0,
                      "ops::dot (block tier): the operands' mesh must be a "
                      "whole number of warps");
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
