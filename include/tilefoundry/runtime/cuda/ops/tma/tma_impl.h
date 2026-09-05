/// TMA op tiers. Included in-context from ``ops/tma.cuh``.
#pragma once

namespace tma_impl {

/// The shared-window address of a generic pointer.
__device__ inline uint32_t smem_addr(void const *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

/// Whether a shard layout describes one contiguous run.
///
/// Read off the ``ShardLayout``'s own layout type, which is static; the
/// extents `local()` produces are not, and the tier must be a compile-time
/// choice. A layout that coalesces to rank 1 with unit stride is a run of
/// bytes, which is what ``cp.async.bulk`` moves.
template <class T, class = void> struct run_of_bytes : std::false_type {};
template <class T>
struct run_of_bytes<T, std::void_t<typename T::shard_layout_type>> {
    using L = decltype(cute::coalesce(
        typename cute::remove_cvref_t<T>::shard_layout_type::layout{}));
    static constexpr bool value = cute::is_static<L>::value &&
                                  decltype(cute::rank(L{}))::value == 1 &&
                                  int(cute::stride<0>(L{})) == 1;
};

template <class T>
using elem_t = cute::remove_cvref_t<decltype(detail::to_local(
    std::declval<T const &>())(0))>;

/// The bulk instruction needs both ends contiguous, the same element type, and
/// a transfer that is a whole number of 16-byte grains. The first two are the
/// layout's business and decided here; the grain depends on the projected
/// extent, which the shard divides at run time, so `Bulk` checks it and hands
/// off to `Strided` when it does not hold.
template <class Src, class Dst>
inline constexpr bool bulk_eligible_v =
    run_of_bytes<Src>::value && run_of_bytes<Dst>::value &&
    std::is_same_v<elem_t<Src>, elem_t<Dst>>;

/// The element loop both tiers fall back on: every thread strides the run.
struct StridedCopy {
    template <class SV, class DV>
    __device__ void operator()(SV const &sv, DV &dv) const {
        const int n = int(cute::size(dv));
        for (int i = int(threadIdx.x); i < n; i += int(blockDim.x))
            dv(i) = static_cast<cute::remove_cvref_t<decltype(dv(0))>>(sv(i));
    }
};

/// Every thread copies its share, then one arrival says the tile is readable.
///
/// The block-wide barrier is what makes the single arrival honest: without it
/// the elected thread could arrive while another still had stores in flight.
struct Strided {
    template <class Src, class Dst>
    __device__ void operator()(Src const &src, Dst &dst, uint64_t *bar) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        StridedCopy{}(s, d);
        __threadfence_block();
        ops::sync<ops::SyncKind::syncthreads>();
        if (ops::shuffle_elect<1024>())
            ops::mbarrier_arrive(bar);
    }
};

/// ``cp.async.bulk`` global to shared, completing on the barrier.
///
/// The elected thread declares the byte count on the arrival that issues the
/// copy, so the declared and delivered counts are one expression and cannot
/// drift. An extent the shard leaves off the 16-byte grain has no defined
/// behaviour here, so it takes `Strided` instead -- same entry, same barrier,
/// same result.
struct Bulk {
    template <class Src, class Dst>
    __device__ void operator()(Src const &src, Dst &dst, uint64_t *bar) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        using elem = cute::remove_cvref_t<decltype(d(0))>;
        const unsigned bytes =
            unsigned(int(cute::size(s))) * unsigned(sizeof(elem));
        if ((bytes & 15u) != 0u) {
            Strided{}(src, dst, bar);
            return;
        }
        if (ops::shuffle_elect<1024>()) {
            ops::mbarrier_arrive_expect_tx(bar, bytes);
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
