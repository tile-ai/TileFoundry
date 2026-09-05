/// CUDA copy op public entries. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "copy/copy_impl.h"

/// ``copy_n`` is a plain per-element copy (with dtype conversion) — routed
/// through the shared ``unary_impl::Unary`` skeleton via the ``identity_op``
/// tag (unary.cuh), identically to ``cast`` (cast.cuh): the two public
/// entries name the same operation for different call sites.
template <class TSrc, class TDst>
__device__ void copy_n(TSrc const &src, TDst &dst, int N) {
    unary_impl::Unary<identity_op>{}(src, dst, N);
}

/// Copy ``src`` into ``dst``, at the width their shard layouts admit.
///
/// The shape, the strides, the element types and the share this instance owns
/// all come off the operands' ``ShardLayout``s, so the call site names no
/// vector width and no thread index: a copy the mesh splits across threads is
/// a thread-scoped shard layout, not a strided loop written here.
///
/// The width is a compile-time answer -- ``copy_vector_elems`` reports it --
/// and the only run-time question left is whether the shard's offset landed on
/// the alignment the width needs.
template <class TSrc, class TDst>
__device__ void copy(TSrc const &src, TDst &dst) {
    copy_impl::Copy{}(src, dst);
}

/// Elements per move ``copy`` will use on these operands.
template <class TSrc, class TDst>
inline constexpr int copy_vector_elems =
    tilefoundry::copy_vector_elems<TSrc, TDst>;

template <class TSrc, class TDst>
__device__ void copy_async(TSrc const &src, TDst &dst) {
    copy_impl::CopyAsync{}(src, dst);
}
