// CUDA unary op public entries and tags. Included in-context from runtime.cuh
// inside namespace tilefoundry::ops.
#pragma once

struct rsqrt_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(rsqrtf(static_cast<float>(x)));
    }
};
struct neg_op {
    template <class T> __device__ T operator()(T x) const { return -x; }
};
struct relu_op {
    template <class T> __device__ T operator()(T x) const {
        return x > T(0) ? x : T(0);
    }
};
struct square_op {
    template <class T> __device__ T operator()(T x) const { return x * x; }
};
// Identity tag: forwards its argument unchanged. ``Unary<identity_op>`` is
// therefore a plain per-element map — the shared skeleton ``ops::cast`` and
// ``ops::copy_n`` route through (cast.cuh, copy.cuh); the output-side
// ``static_cast`` in ``unary_impl::Unary`` performs the actual conversion.
struct identity_op {
    template <class T> __device__ T operator()(T x) const { return x; }
};

#include "unary/unary_impl.h"

template <class Op, class TIn, class TOut>
__device__ void unary(TIn const &src, TOut &dst, int N, Op op = {}) {
    unary_impl::Unary<Op>{}(src, dst, N, op);
}
