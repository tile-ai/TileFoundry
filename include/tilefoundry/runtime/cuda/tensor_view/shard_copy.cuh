/// ShardTensor-aware tilefoundry::copy helpers. Included in-context from
/// runtime.cuh inside namespace tilefoundry.
#pragma once

namespace detail {

/// Convert one value between element types.
///
/// ``static_cast`` is not enough for the CUDA half types: a translation unit
/// built with ``__CUDA_NO_BFLOAT16_CONVERSIONS__`` (which is what a torch
/// extension build defines) has no conversion operator on them at all, so the
/// intrinsic is the only way across.
template <class D, class S> struct value_cast {
    CUTE_HOST_DEVICE static D apply(S const &x) { return static_cast<D>(x); }
};
template <> struct value_cast<float, __nv_bfloat16> {
    CUTE_HOST_DEVICE static float apply(__nv_bfloat16 const &x) {
        return __bfloat162float(x);
    }
};
template <> struct value_cast<__nv_bfloat16, float> {
    CUTE_HOST_DEVICE static __nv_bfloat16 apply(float x) {
        return __float2bfloat16(x);
    }
};
template <> struct value_cast<float, __half> {
    CUTE_HOST_DEVICE static float apply(__half const &x) {
        return __half2float(x);
    }
};
template <> struct value_cast<__half, float> {
    CUTE_HOST_DEVICE static __half apply(float x) { return __float2half(x); }
};

template <class D, class S> CUTE_HOST_DEVICE D convert(S const &x) {
    return value_cast<D, cute::remove_cvref_t<S>>::apply(x);
}

/// A load/store unit of ``Bytes``, for the widest move the layouts admit.
template <int Bytes> struct alignas(Bytes) raw_vec {
    char raw[Bytes];
};

template <class T>
using elem_of = cute::remove_cvref_t<decltype(std::declval<T const &>()(0))>;

/// Whether a view addresses memory a vector move can reach.
///
/// Register fragments are excluded on purpose: taking ``&frag(i)`` under a
/// loop the compiler cannot unroll spills the fragment to local memory, which
/// costs more than the wider move saves.
template <class V>
inline constexpr bool addressable_v =
    cute::is_gmem<cute::remove_cvref_t<V>>::value ||
    cute::is_smem<cute::remove_cvref_t<V>>::value;

/// Elements per move: the largest power of two both layouts run contiguously
/// over, capped so neither side's move exceeds 16 bytes.
///
/// ``max_common_vector`` is the layout question -- over how many consecutive
/// indices do both sides advance by one element -- and the cap is the hardware
/// one. A dtype change does not force this to 1: the wider load still holds,
/// and the conversion happens on the way out.
template <class SL, class DL, class SElem, class DElem>
constexpr int vector_elems() {
    if constexpr (!cute::is_static<SL>::value || !cute::is_static<DL>::value) {
        return 1;
    } else {
        constexpr int mcv =
            int(decltype(cute::max_common_vector(SL{}, DL{}))::value);
        constexpr int wide =
            int(sizeof(SElem) > sizeof(DElem) ? sizeof(SElem) : sizeof(DElem));
        int v = 1;
        while (v * 2 <= mcv && v * 2 * wide <= 16)
            v *= 2;
        return v;
    }
}

/// Move ``n`` elements, ``V`` at a time where the addresses allow it.
///
/// The alignment test is one comparison on the two base pointers, not a scan:
/// the layouts already said the run is contiguous, so the only thing left that
/// a type cannot know is where the shard's offset landed.
template <int V, class SView, class DView>
CUTE_HOST_DEVICE void vec_move(SView const &sv, DView &dv, int n) {
    using s_val = elem_of<SView>;
    using d_val = elem_of<DView>;
    int i = 0;
#if defined(__CUDA_ARCH__)
    if constexpr (V > 1) {
        using s_vec = raw_vec<V *int(sizeof(s_val))>;
        using d_vec = raw_vec<V *int(sizeof(d_val))>;
        if ((reinterpret_cast<uintptr_t>(&sv(0)) % alignof(s_vec)) == 0 &&
            (reinterpret_cast<uintptr_t>(&dv(0)) % alignof(d_vec)) == 0) {
            for (; i + V <= n; i += V) {
                s_vec in = *reinterpret_cast<s_vec const *>(&sv(i));
                d_vec out;
                auto const *from = reinterpret_cast<s_val const *>(&in);
                auto *to = reinterpret_cast<d_val *>(&out);
                CUTE_UNROLL
                for (int k = 0; k < V; ++k)
                    to[k] = convert<d_val>(from[k]);
                *reinterpret_cast<d_vec *>(&dv(i)) = out;
            }
        }
    }
#endif
    for (; i < n; ++i)
        dv(i) = convert<d_val>(sv(i));
}

/// The view type one instance sees: a ShardTensor's projection, or the tensor
/// itself when there is no shard layout to project through.
template <class T, class = void> struct local_view {
    using type = cute::remove_cvref_t<T>;
};
template <class T>
struct local_view<
    T, std::void_t<typename cute::remove_cvref_t<T>::shard_layout_type>> {
    using type =
        cute::remove_cvref_t<decltype(local(std::declval<T const &>()))>;
};
template <class T> using local_view_t = typename local_view<T>::type;

/// Elements per move for a ShardTensor pair, as a compile-time answer.
template <class Src, class Dst> constexpr int shard_vector_elems() {
    if constexpr (addressable_v<local_view_t<Src>> &&
                  addressable_v<local_view_t<Dst>>) {
        return vector_elems<operand_layout_t<Src>, operand_layout_t<Dst>,
                            elem_of<local_view_t<Src>>,
                            elem_of<local_view_t<Dst>>>();
    } else {
        return 1;
    }
}

/// Copy one instance's slice, at the width the two shard layouts admit.
template <class Src, class Dst, class SView, class DView>
CUTE_HOST_DEVICE void copy_fragment(SView const &sv, DView &dv) {
    vec_move<shard_vector_elems<Src, Dst>()>(sv, dv, int(cute::size(dv)));
}

}

/// Elements per move ``copy`` will use on these operands.
///
/// Exposed so a caller staging a tile can say what width it expects and have
/// the compiler check it, rather than reading the generated code.
template <class Src, class Dst>
inline constexpr int copy_vector_elems = detail::shard_vector_elems<Src, Dst>();

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
    detail::copy_fragment<ShardTensor<TS, GLS, SLS>, ShardTensor<TD, GLD, SLD>>(
        sv, dv);
}

/// Plain CuTe tensors on both ends: the same width question, asked of the
/// tensors' own layouts. ``ShardTensor`` overloads above are more specialised,
/// so a sharded operand never reaches here.
template <class ST, class DT>
CUTE_HOST_DEVICE void copy(ST const &src, DT &dst) {
    detail::copy_fragment<ST, DT>(src, dst);
}
