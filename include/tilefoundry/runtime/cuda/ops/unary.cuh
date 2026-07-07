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

#include "unary/unary_impl.h"

template <class Op, class TIn, class TOut>
CUTE_HOST_DEVICE void unary(TIn const &src, TOut &dst, int N, Op op = {}) {
    unary_impl::Unary<Op>{}(src, dst, N, op);
}

template <class TIn, class TOut>
CUTE_HOST_DEVICE void relu(TIn const &src, TOut &dst) {
    unary(src, dst, int(cute::size(detail::to_local(src))), relu_op{});
}
