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
// Transcendentals go through the precise `<math.h>` forms (`expf`, not
// `__expf`), so a compiled kernel matches the evaluator's torch oracle rather
// than the faster intrinsic's looser result.
struct exp_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(expf(static_cast<float>(x)));
    }
};
struct exp2_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(exp2f(static_cast<float>(x)));
    }
};
struct log_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(logf(static_cast<float>(x)));
    }
};
struct log2_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(log2f(static_cast<float>(x)));
    }
};
struct abs_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(fabsf(static_cast<float>(x)));
    }
};
struct ceil_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(ceilf(static_cast<float>(x)));
    }
};
// `rintf` rounds halfway cases to even, which is what the torch oracle does;
// `roundf` rounds them away from zero and would disagree at every .5.
struct round_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(rintf(static_cast<float>(x)));
    }
};

#include "unary/unary_impl.h"

template <class Op, class TIn, class TOut>
__device__ void unary(TIn const &src, TOut &dst, int N, Op op = {}) {
    unary_impl::Unary<Op>{}(src, dst, N, op);
}
