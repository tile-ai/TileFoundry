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

/// The modes of the static layout ``L``, flat, from the smallest stride to
/// the largest; ties keep their order.
template <class L> struct AscendingStrides {
    using strides_t = decltype(cute::stride(cute::flatten(L{})));
    static constexpr int rank = int(cute::tuple_size<strides_t>::value);
    static constexpr cute::array<int, rank> order =
        []<int... Is>(cute::seq<Is...>) {
            cute::array<int, rank> stride = {
                int(decltype(cute::get<Is>(strides_t{}))::value)...};
            cute::array<int, rank> sorted = {Is...};
            for (int i = 1; i < rank; ++i)
                for (int j = i;
                     j > 0 && stride[sorted[j]] < stride[sorted[j - 1]]; --j) {
                    int moved = sorted[j];
                    sorted[j] = sorted[j - 1];
                    sorted[j - 1] = moved;
                }
            return sorted;
        }(cute::make_seq<rank>{});
};

/// ``layout`` with its modes in register order, coalesced so a mode of one
/// element leaves no stride to recast; a swizzle keeps its place in front of
/// the affine layout it permutes.
template <class L>
CUTE_HOST_DEVICE constexpr auto in_register_order(L const &layout) {
    if constexpr (cute::is_composed_layout<L>::value) {
        return cute::make_composed_layout(layout.layout_a(), layout.offset(),
                                          in_register_order(layout.layout_b()));
    } else {
        using ascending = AscendingStrides<L>;
        return [&]<int... Is>(cute::seq<Is...>) {
            return cute::coalesce(
                cute::select<ascending::order[Is]...>(cute::flatten(layout)));
        }(cute::make_seq<ascending::rank>{});
    }
}

/// ``t`` indexed in register order, the order Dot pairs its operands in.
///
/// TIR numbers a unit's registers from the smallest physical stride up
/// (``register_strides``), while CuTe counts the first mode fastest. A view
/// with one mode is returned as it is.
template <class View> __device__ decltype(auto) register_order(View const &t) {
    using flat_t = decltype(cute::flatten(
        detail::plain_layout(typename View::layout_type{})));
    if constexpr (decltype(cute::rank(flat_t{}))::value <= 1) {
        return t;
    } else {
        static_assert(cute::is_static<decltype(cute::stride(flat_t{}))>::value,
                      "ops::dot: an operand of several modes needs static "
                      "strides to be read in register order");
        return cute::make_tensor(t.data(), in_register_order(t.layout()));
    }
}

/// Whether ``t`` starts on a boundary of ``V`` of its elements. A layout
/// cannot say where a shard's offset lands, so the pointer is asked.
template <int V, class View> __device__ bool starts_aligned(View const &t) {
    using value = typename cute::remove_cvref_t<View>::value_type;
    return reinterpret_cast<uintptr_t>(cute::raw_pointer_cast(t.data())) %
               (V * sizeof(value)) ==
           0;
}

/// This lane's share of the product, one element at a time.
template <class AView, class BView>
__device__ float contract_elements(AView const &a, BView const &b, int n) {
    Partials acc{};
    for (int i = 0; i < n; ++i)
        acc.p[i % Partials::kWays] +=
            static_cast<float>(a(i)) * static_cast<float>(b(i));
    return total(acc);
}

/// This lane's share of the product, folded.
template <class AView, class BView>
__device__ float contract(AView const &a_view, BView const &b_view, int n) {
    auto const &a = register_order(a_view);
    auto const &b = register_order(b_view);
    using a_val = typename AView::value_type;
    using b_val = typename BView::value_type;
    constexpr int V = fold_width<cute::remove_cvref_t<decltype(a)>,
                                 cute::remove_cvref_t<decltype(b)>>();
    if constexpr (V == 1) {
        return contract_elements(a, b, n);
    } else {
        if (!starts_aligned<V>(a) || !starts_aligned<V>(b))
            return contract_elements(a, b, n);
        Partials acc{};
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
        return total(acc);
    }
}

/// Extent of the thread mesh's fastest mesh axis, as the mesh states it.
///
/// The mesh axis, not the ids it reaches: a butterfly runs one mesh axis, so a
/// mesh written ``(2,16)`` spans sixteen lanes twice even where those 32
/// threads are one whole warp. ``id_axes`` folds that distinction away, so read
/// the mesh's own last mesh axis -- row-major, so the fastest is last.
template <class T> CUTE_HOST_DEVICE constexpr int lane_axis_extent() {
    using mesh_t = typename cute::remove_cvref_t<T>::shard_layout_type::mesh;
    using axes_t = cute::remove_cvref_t<decltype(cute::flatten(
        detail::plain_layout(typename mesh_t::layout_type{})))>;
    return int(cute::get<decltype(cute::rank(axes_t{}))::value - 1>(
        cute::shape(axes_t{})));
}

/// The contraction lives inside a warp: one butterfly finishes it.
struct Warp {
    template <class Lhs, class Rhs, class Dst>
    __device__ void operator()(Lhs const &lhs, Rhs const &rhs, Dst &dst) const {
        static_assert(tilefoundry::shard_mesh_units<Lhs>() <= kWarpSize,
                      "ops::dot: a cross-warp mesh requires workspace");
        static_assert(lane_axis_extent<Lhs>() == kWarpSize,
                      "ops::dot (warp tier): the operands' fastest mesh axis "
                      "must be exactly one warp of 32 lanes");
        auto a = tilefoundry::local_tensor(lhs);
        auto b = tilefoundry::local_tensor(rhs);
        auto &&d = tilefoundry::local_tensor(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        const float sum = tilefoundry::warp_reduce<primitive::add_op>(
            contract(a, b, int(cute::size(a))));
        d(0) = static_cast<value_type>(sum);
    }
};

/// Fold one partial per warp through shared workspace.
struct Cta {
    template <class Lhs, class Rhs, class Dst, class Ws>
    __device__ void operator()(Lhs const &lhs, Rhs const &rhs, Dst &dst,
                               Ws &ws) const {
        auto a = tilefoundry::local_tensor(lhs);
        auto b = tilefoundry::local_tensor(rhs);
        auto &&d = tilefoundry::local_tensor(dst);
        auto &&slots = tilefoundry::local_tensor(ws);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        constexpr int units = tilefoundry::shard_mesh_units<Lhs>();
        static_assert(units >= kWarpSize && units % kWarpSize == 0,
                      "ops::dot (block tier): the operands' mesh must be a "
                      "whole number of warps");
        constexpr int warps = units / kWarpSize;
        const float part = tilefoundry::warp_reduce<primitive::add_op>(
            contract(a, b, int(cute::size(a))));
        const unsigned tid = unsigned(
            tilefoundry::unit_id<tilefoundry::TopologyLevel::thread>());
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
