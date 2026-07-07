// CUDA copy op public entries. Included in-context from runtime.cuh inside
// namespace tilefoundry::ops.
#pragma once

#include "copy/copy_impl.h"

template <class TSrc, class TDst>
CUTE_HOST_DEVICE void copy_n(TSrc const &src, TDst &dst, int N) {
    copy_impl::CopyN{}(src, dst, N);
}

template <class TSrc, class TDst>
CUTE_HOST_DEVICE void copy_async(TSrc const &src, TDst &dst) {
    copy_impl::CopyAsync{}(src, dst);
}
