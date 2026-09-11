/// CUDA copy op implementation. Included in-context from ops/copy.cuh inside
/// namespace tilefoundry::ops.
#pragma once

namespace copy_impl {

/// ``copy`` is ``cute::copy`` with the operands resolved to this slice first.
struct Copy {
    template <class TSrc, class TDst>
    __device__ void operator()(TSrc const &src, TDst &dst) const {
        auto &&s = tilefoundry::local_tensor(src);
        auto &&d = tilefoundry::local_tensor(dst);
        cute::copy(s, d);
    }
};

/// Whether two projected views hold the same number of elements.
template <class SView, class DView>
CUTE_HOST_DEVICE constexpr bool same_slice_size() {
    using SL = typename cute::remove_cvref_t<SView>::layout_type;
    using DL = typename cute::remove_cvref_t<DView>::layout_type;
    if constexpr (!cute::is_static<SL>::value || !cute::is_static<DL>::value)
        return true;
    else
        return int(decltype(cute::size(SL{}))::value) ==
               int(decltype(cute::size(DL{}))::value);
}

/// Bytes in the largest common asynchronous move.
template <class SView, class DView>
CUTE_HOST_DEVICE constexpr int async_bytes() {
    using SL = typename cute::remove_cvref_t<SView>::layout_type;
    using DL = typename cute::remove_cvref_t<DView>::layout_type;
    using elem = typename cute::remove_cvref_t<DView>::value_type;
    if constexpr (!cute::is_static<SL>::value || !cute::is_static<DL>::value) {
        return int(sizeof(elem));
    } else {
        constexpr int run = int(
            decltype(cute::gcd(cute::max_common_vector(SL{}, DL{}),
                               cute::gcd(cute::max_alignment(SL{}),
                                         cute::max_alignment(DL{}))))::value);
        int b = int(sizeof(elem));
        while (b * 2 <= run * int(sizeof(elem)) && b * 2 <= 16)
            b *= 2;
        return b;
    }
}

/// The same move, issued asynchronously.
struct CopyAsync {
    template <class TSrc, class TDst>
    __device__ void operator()(TSrc const &src, TDst &dst) const {
        auto &&s = tilefoundry::local_tensor(src);
        auto &&d = tilefoundry::local_tensor(dst);
        using value_type =
            typename cute::remove_cvref_t<decltype(d)>::value_type;
        constexpr int bytes = async_bytes<decltype(s), decltype(d)>();
        static_assert(bytes > int(sizeof(value_type)),
                      "ops::copy_async: these two views share no run wide "
                      "enough for cp.async");
        static_assert(
            same_slice_size<decltype(s), decltype(d)>(),
            "ops::copy_async: the two projected slices must hold the same "
            "number of elements");
        constexpr int V = bytes / int(sizeof(value_type));
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
        auto s_v = cute::recast<cute::uint_bit_t<bytes * 8>>(s);
        const int nv = int(cute::size(s_v));
        for (int i = 0; i < nv; ++i)
            __pipeline_memcpy_async(&d(i * V), &s_v(i), bytes);
#else
        static_assert(dependent_false_v<TSrc>,
                      "ops::copy_async: cp.async requires sm_80 or newer");
#endif
    }
};

}
