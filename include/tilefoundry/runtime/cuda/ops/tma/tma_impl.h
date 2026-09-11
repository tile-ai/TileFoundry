/// TMA op internals. Included in-context from ``ops/tma.cuh``.
#pragma once

namespace tma_impl {

/// The shared-window address of a generic pointer.
__device__ inline uint32_t smem_addr(void const *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

/// Whether an operand is one unbroken run of bytes.
template <class T>
inline constexpr bool one_run_v = [] {
    using L = typename cute::remove_cvref_t<T>::layout_type;
    return cute::is_static<L>::value && decltype(cute::cosize(L{}))::value ==
                                            decltype(cute::size(L{}))::value;
}();

template <class T>
using elem_t = cute::remove_cvref_t<decltype(tilefoundry::local_tensor(
    std::declval<T const &>())(0))>;

/// Whether an operand leaves the tile whole on every instance of its mesh.
template <class T> CUTE_HOST_DEVICE constexpr bool leaves_tile_whole() {
    if constexpr (tilefoundry::ShardTensorLike<T>)
        return detail::shard_layout_is_full_broadcast<
            typename cute::remove_cvref_t<T>::shard_layout_type>();
    else
        return true;
}

/// What this op needs of its operands, asked once at the entry.
template <class Src, class Dst>
CUTE_HOST_DEVICE constexpr void check_tma_operands() {
    using s_view = tilefoundry::local_view_t<Src>;
    using d_view = tilefoundry::local_view_t<Dst>;
    static_assert(
        leaves_tile_whole<Src>() && leaves_tile_whole<Dst>(),
        "ops::tma_copy: both operands must leave the tile whole on every "
        "instance");
    static_assert(tilefoundry::ShardTensorLike<Dst>,
                  "ops::tma_copy: the destination must name a mesh");
    static_assert(
        one_run_v<s_view> && one_run_v<d_view>,
        "ops::tma_copy: both projected views must be one static unbroken run");
    static_assert(std::is_same_v<elem_t<Src>, elem_t<Dst>>,
                  "ops::tma_copy: both operands need the same element type");
    static_assert(
        copy_impl::same_slice_size<s_view, d_view>(),
        "ops::tma_copy: the two projected views must hold the same number of "
        "elements");
}

/// The element loop `Bulk` hands off to: every instance strides the one run.
template <int Instances> struct StridedCopy {
    template <class SV, class DV>
    __device__ void operator()(SV const &sv, DV &dv) const {
        const int n = int(cute::size(dv));
        const int first =
            int(tilefoundry::program_id<tilefoundry::TopologyScope::thread>());
        for (int i = first; i < n; i += Instances)
            dv(i) = static_cast<cute::remove_cvref_t<decltype(dv(0))>>(sv(i));
    }
};

/// Every thread copies its share, then one arrival says the tile is readable.
struct Strided {
    template <class Src, class Dst>
    __device__ void operator()(Src const &src, Dst &dst, uint64_t *bar) const {
        auto s = tilefoundry::local_tensor(src);
        auto &&d = tilefoundry::local_tensor(dst);
        StridedCopy<tilefoundry::shard_mesh_instances<Dst>()>{}(s, d);
        __threadfence_block();
        ops::sync(dst.shard_layout.mesh_value);
        if (tilefoundry::shuffle_elect())
            asm volatile("{\n"
                         "  .reg .b64 state;\n"
                         "  mbarrier.arrive.shared::cta.b64 state, [%0];\n"
                         "}\n" ::"r"(smem_addr(bar)));
    }
};

/// ``cp.async.bulk`` global to shared, completing on the barrier.
struct Bulk {
    template <class Src, class Dst>
    __device__ void operator()(Src const &src, Dst &dst, uint64_t *bar) const {
        auto s = tilefoundry::local_tensor(src);
        auto &&d = tilefoundry::local_tensor(dst);
        using elem = cute::remove_cvref_t<decltype(d(0))>;
        constexpr bool static_layout = cute::is_static<
            typename cute::remove_cvref_t<decltype(cute::layout(s))>>::value;
        const unsigned bytes =
            unsigned(int(cute::size(s))) * unsigned(sizeof(elem));
        constexpr unsigned static_bytes =
            static_layout
                ? unsigned(int(cute::size(typename cute::remove_cvref_t<
                                          decltype(cute::layout(s))>{}))) *
                      unsigned(sizeof(elem))
                : 0u;
        if constexpr (static_layout && ((static_bytes & 15u) != 0u)) {
            Strided{}(src, dst, bar);
            return;
        } else if constexpr (!static_layout) {
            if ((bytes & 15u) != 0u) {
                Strided{}(src, dst, bar);
                return;
            }
        }
        if (tilefoundry::shuffle_elect()) {
            asm volatile(
                "{\n"
                "  .reg .b64 state;\n"
                "  mbarrier.arrive.expect_tx.shared::cta.b64 state, [%0], %1;\n"
                "}\n" ::"r"(smem_addr(bar)),
                "r"(bytes));
            asm volatile(
                "cp.async.bulk.shared::cluster.global"
                ".mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                    smem_addr(&d(0))),
                "l"(&s(0)), "r"(bytes), "r"(smem_addr(bar))
                : "memory");
        }
    }
};

}
