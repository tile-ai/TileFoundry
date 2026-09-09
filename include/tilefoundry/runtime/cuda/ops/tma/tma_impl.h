/// TMA op internals. Included in-context from ``ops/tma.cuh``.
#pragma once

namespace tma_impl {

/// The shared-window address of a generic pointer.
__device__ inline uint32_t smem_addr(void const *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

/// Whether an operand is one unbroken run of bytes, which is what
/// ``cp.async.bulk`` moves.
///
/// ``cosize == size``: the offsets a layout produces span exactly as many
/// elements as it has, so nothing is skipped. It is CuTe's own way of asking
/// and it reads the offsets rather than the mode order, which the
/// coalesce-the-reversed-layout version this replaces did not -- that one
/// called a column-major contiguous tile non-contiguous, and a wrong answer
/// here now refuses a copy the instruction could have carried.
template <class T>
inline constexpr bool one_run_v = [] {
    using L = typename cute::remove_cvref_t<T>::layout_type;
    return cute::is_static<L>::value && decltype(cute::cosize(L{}))::value ==
                                            decltype(cute::size(L{}))::value;
}();

template <class T>
using elem_t = cute::remove_cvref_t<decltype(detail::to_local(
    std::declval<T const &>())(0))>;

/// Whether an operand leaves the tile whole on every instance of its mesh.
///
/// A plain CuTe tensor answers true: it names no mesh, so nothing divides it.
/// A ShardTensor answers with ``shard_layout_is_full_broadcast``, which is the
/// same question ``local()`` asks before it hands the engine back unmoved.
template <class T> CUTE_HOST_DEVICE constexpr bool leaves_tile_whole() {
    if constexpr (tilefoundry::detail::ShardTensorLike<T>)
        return tilefoundry::detail::shard_layout_is_full_broadcast<
            typename cute::remove_cvref_t<T>::shard_layout_type>();
    else
        return true;
}

/// What this op needs of its operands, asked once at the entry.
///
/// Neither the instruction nor its fallback divides the tile: `Bulk` elects one
/// thread of the CTA and declares `size(src)` bytes, and `StridedCopy` strides
/// one run every instance addresses.
///
/// These sentences used to be a *tier*. `bulk_eligible_v` selected `Strided`
/// wherever they failed -- and what fails them is an operand a mesh divides,
/// for which `StridedCopy` walked the destination's whole length out of a slice
/// `Instances` times shorter: a wrong answer with no diagnostic.
template <class Src, class Dst>
CUTE_HOST_DEVICE constexpr void check_tma_operands() {
    using s_view = tilefoundry::detail::local_view_t<Src>;
    using d_view = tilefoundry::detail::local_view_t<Dst>;
    static_assert(
        leaves_tile_whole<Src>() && leaves_tile_whole<Dst>(),
        "ops::tma_copy: both operands must leave the tile whole on every "
        "instance -- one elected thread moves the run and the element "
        "fallback strides it, so neither divides it. An operand a mesh really "
        "does divide is ops::copy, one slice an instance");
    static_assert(
        tilefoundry::detail::ShardTensorLike<Dst>,
        "ops::tma_copy: the destination must name a mesh -- the element "
        "fallback synchronises the instances that stored the tile and strides "
        "over their count, and a plain CuTe tensor states neither. Wrap it in "
        "a shard layout that gives the block's one axis shard::B");
    static_assert(
        one_run_v<s_view> && one_run_v<d_view>,
        "ops::tma_copy: both projected views must be one static unbroken run "
        "-- cp.async.bulk moves a byte range from an origin, and the element "
        "fallback walks the same range. A view with a gap in it names no such "
        "range; materialise the slice first, then stage it");
    static_assert(std::is_same_v<elem_t<Src>, elem_t<Dst>>,
                  "ops::tma_copy: this stages bytes, it does not convert "
                  "them, so the two element types must be one type");
    static_assert(
        copy_impl::same_slice_size<s_view, d_view>(),
        "ops::tma_copy: the two projected views must hold the same number of "
        "elements -- the byte count is read off the source and the tile is "
        "written at the destination's origin, so a shorter destination is "
        "overrun and a longer one is left part-filled");
}

/// The element loop `Bulk` hands off to: every instance strides the one run.
///
/// Where an instance starts and how far it steps are one fact -- which
/// instances are running this -- so the start is ``program_id`` and the step
/// ``Instances``, the destination mesh's own count and what the barrier below
/// waits on, rather than ``blockDim.x``, a different set wherever the mesh is
/// narrower. The run is one every instance addresses, which is what
/// ``check_tma_operands`` makes true: a divided destination needs its whole
/// slice from one instance, and this gives it every ``Instances``-th.
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
///
/// The barrier before it is what makes the single arrival honest: without it
/// the elected thread could arrive while another still had stores in flight.
/// Whose stores those are is the destination mesh's answer, so the set to wait
/// for is read off it. The arrival is written here rather than called: it reads
/// nothing off a layout, so it is no op, and its token is discarded because
/// consumers wait on the phase parity and not on a token.
struct Strided {
    template <class Src, class Dst>
    __device__ void operator()(Src const &src, Dst &dst, uint64_t *bar) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
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
