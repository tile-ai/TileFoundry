// CUDA copy op public entries. Included in-context from runtime.cuh inside
// namespace tilefoundry::ops.
#pragma once

#include "copy/copy_impl.h"

// ``copy_n`` is a plain per-element copy (with dtype conversion) — routed
// through the shared ``unary_impl::Unary`` skeleton via the ``identity_op``
// tag (unary.cuh), identically to ``cast`` (cast.cuh): the two public
// entries name the same operation for different call sites.
template <class TSrc, class TDst>
__device__ void copy_n(TSrc const &src, TDst &dst, int N) {
    unary_impl::Unary<identity_op>{}(src, dst, N);
}

template <class TSrc, class TDst>
__device__ void copy_async(TSrc const &src, TDst &dst) {
    copy_impl::CopyAsync{}(src, dst);
}
