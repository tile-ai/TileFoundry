// ShardTensor-aware tilefoundry::copy helpers. Included in-context from
// runtime.cuh inside namespace tilefoundry.
#pragma once

namespace detail {

template <bool SrcIsGmem, class SView, class DView>
CUTE_HOST_DEVICE void copy_fragment(SView const &sv, DView &dv) {
    using s_val_t = cute::remove_cvref_t<decltype(sv(0))>;
    using d_val_t = cute::remove_cvref_t<decltype(dv(0))>;
    using dvc_layout_t =
        decltype(cute::coalesce(cute::remove_cvref_t<DView>{}.layout()));
    constexpr bool dst_static_contig =
        cute::is_static<dvc_layout_t>::value &&
        decltype(cute::rank(dvc_layout_t{}))::value == 1 &&
        (int(cute::size(dvc_layout_t{})) == 1 ||
         int(cute::stride<0>(dvc_layout_t{})) == 1);
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
            bool ok = (reinterpret_cast<uintptr_t>(sp) & 0xF) == 0;
            CUTE_UNROLL
            for (int i = 1; i < N; ++i)
                ok = ok && (&sv(i) == sp + i);
            if (ok) {
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

} // namespace detail

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
    detail::copy_fragment<src_gmem>(sv, dv);
}
