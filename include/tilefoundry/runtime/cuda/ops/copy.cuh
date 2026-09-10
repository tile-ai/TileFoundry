/// CUDA copy op public entries. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "copy/copy_impl.h"

/// Copy one projected slice to another.
template <class TSrc, class TDst>
__device__ void copy(TSrc const &src, TDst &dst) {
    copy_impl::Copy{}(src, dst);
}

template <class TSrc, class TDst>
__device__ void copy_async(TSrc const &src, TDst &dst) {
    copy_impl::CopyAsync{}(src, dst);
}
