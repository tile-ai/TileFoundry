// CUDA copy op implementation. Included in-context from ops/copy.cuh inside
// namespace tilefoundry::ops.
#pragma once

namespace copy_impl {

struct CopyN {
    template <class TSrc, class TDst>
    CUTE_HOST_DEVICE void operator()(TSrc const &src, TDst &dst, int N) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        for (int i = 0; i < N; ++i) {
            d(i) = static_cast<value_type>(s(i));
        }
    }
};

struct CopyAsync {
    template <class TSrc, class TDst>
    CUTE_HOST_DEVICE void operator()(TSrc const &src, TDst &dst) const {
        auto s = detail::to_local(src);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        const int n_src = int(cute::size(s));
        const int n_dst = int(cute::size(d));
        const int off = (n_dst > n_src) ? local_offset(src) : 0;
        constexpr int ES = sizeof(value_type);
        constexpr int C = (ES == 2 || ES == 4 || ES == 8) ? (16 / ES) : 1;
        int i = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        if (C > 1) {
            for (; i + C <= n_src; i += C)
                __pipeline_memcpy_async(&d(off + i), &s(i), C * ES);
        }
#endif
        for (; i < n_src; ++i)
            d(off + i) = static_cast<value_type>(s(i));
    }
};

} // namespace copy_impl
