/// ShardTensor-aware tilefoundry::copy helpers. Included in-context from
/// runtime.cuh inside namespace tilefoundry.
#pragma once

namespace detail {

// The gmem-side precondition both vector paths need: the view's first element
// is 16B aligned and the next N - 1 elements follow it contiguously. Only the
// register side is statically known to be a unit-stride run, so this half is a
// runtime check. `N` is a template parameter so the unroll stays compile-time.
template <int N, class View>
CUTE_HOST_DEVICE bool is_contiguous_16b(View const &v) {
    auto const *p = &v(0);
    bool ok = (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
    CUTE_UNROLL
    for (int i = 1; i < N; ++i)
        ok = ok && (&v(i) == p + i);
    return ok;
}

// A coalesced view is a single unit-stride run of statically known length —
// the shape a 16B vector access needs on the register side.
template <class View> struct StaticContigView {
    using layout_t =
        decltype(cute::coalesce(cute::remove_cvref_t<View>{}.layout()));
    static constexpr bool value =
        cute::is_static<layout_t>::value &&
        decltype(cute::rank(layout_t{}))::value == 1 &&
        (int(cute::size(layout_t{})) == 1 ||
         int(cute::stride<0>(layout_t{})) == 1);
};

template <bool SrcIsGmem, bool DstIsGmem, class SView, class DView>
CUTE_HOST_DEVICE void copy_fragment(SView const &sv, DView &dv) {
    using s_val_t = cute::remove_cvref_t<decltype(sv(0))>;
    using d_val_t = cute::remove_cvref_t<decltype(dv(0))>;
    using dvc_layout_t = typename StaticContigView<DView>::layout_t;
    using svc_layout_t = typename StaticContigView<SView>::layout_t;
    constexpr bool dst_static_contig = StaticContigView<DView>::value;
    constexpr bool src_static_contig = StaticContigView<SView>::value;

    // Store side: the destination is global and the register-side source is a
    // contiguous run. Without this a reshard back to gmem writes one narrow
    // access per element, so a kernel whose loads vectorize still stores
    // scalar.
    if constexpr (DstIsGmem && !SrcIsGmem && std::is_same_v<s_val_t, d_val_t> &&
                  src_static_contig) {
        constexpr int mcv = int(decltype(cute::max_common_vector(
            svc_layout_t{}, svc_layout_t{}))::value);
        constexpr int vec_bits = mcv * int(cute::sizeof_bits<d_val_t>::value);
        if constexpr (vec_bits >= 128) {
            constexpr int N = int(cute::size(svc_layout_t{}));
            constexpr int C = 16 / int(sizeof(d_val_t));
            constexpr int NV = N / C;
            d_val_t *dp = &dv(0);
            if (is_contiguous_16b<N>(dv)) {
                uint4 tmp[NV];
                d_val_t *tp = reinterpret_cast<d_val_t *>(tmp);
                CUTE_UNROLL
                for (int i = 0; i < NV * C; ++i)
                    tp[i] = sv(i);
                uint4 *vp = reinterpret_cast<uint4 *>(dp);
                CUTE_UNROLL
                for (int i = 0; i < NV; ++i)
                    vp[i] = tmp[i];
                CUTE_UNROLL
                for (int i = NV * C; i < N; ++i)
                    dv(i) = sv(i);
                return;
            }
        }
    }

    if constexpr (SrcIsGmem && std::is_same_v<s_val_t, d_val_t> &&
                  dst_static_contig) {
        constexpr int mcv = int(decltype(cute::max_common_vector(
            dvc_layout_t{}, dvc_layout_t{}))::value);
        constexpr int vec_bits = mcv * int(cute::sizeof_bits<d_val_t>::value);
        if constexpr (vec_bits >= 128) {
            constexpr int N = int(cute::size(dvc_layout_t{}));
            constexpr int C = 16 / int(sizeof(d_val_t));
            constexpr int NV = N / C;
            s_val_t const *sp = &sv(0);
            if (is_contiguous_16b<N>(sv)) {
                uint4 const *vp = reinterpret_cast<uint4 const *>(sp);
                uint4 tmp[NV];
                CUTE_UNROLL
                for (int i = 0; i < NV; ++i)
                    tmp[i] = vp[i];
                d_val_t const *tp = reinterpret_cast<d_val_t const *>(tmp);
                CUTE_UNROLL
                for (int i = 0; i < NV * C; ++i)
                    dv(i) = tp[i];
                CUTE_UNROLL
                for (int i = NV * C; i < N; ++i)
                    dv(i) = sv(i);
                return;
            }
        }
    }
    int N = int(cute::size(dv));
    for (int i = 0; i < N; ++i)
        dv(i) = static_cast<d_val_t>(sv(i));
}

}

template <class T, class GL, class SL, class DT>
CUTE_HOST_DEVICE void copy(ShardTensor<T, GL, SL> const &src, DT &dst) {
    auto view = local(src);
    int N = int(cute::size(view));
    for (int i = 0; i < N; ++i) {
        dst(i) = view(i);
    }
}

template <class ST, class T, class GL, class SL>
CUTE_HOST_DEVICE void copy(ST const &src, ShardTensor<T, GL, SL> &dst) {
    auto view = local(dst);
    int N = int(cute::size(view));
    for (int i = 0; i < N; ++i) {
        view(i) = src(i);
    }
}

template <class TS, class GLS, class SLS, class TD, class GLD, class SLD>
CUTE_HOST_DEVICE void copy(ShardTensor<TS, GLS, SLS> const &src,
                           ShardTensor<TD, GLD, SLD> &dst) {
    auto &&sv = local(src);
    auto &&dv = local(dst);
    constexpr bool src_gmem =
        cute::is_gmem<cute::remove_cvref_t<decltype(src.engine)>>::value;
    constexpr bool dst_gmem =
        cute::is_gmem<cute::remove_cvref_t<decltype(dst.engine)>>::value;
    detail::copy_fragment<src_gmem, dst_gmem>(sv, dv);
}
