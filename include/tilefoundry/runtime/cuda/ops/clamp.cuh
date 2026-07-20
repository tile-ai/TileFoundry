// CUDA clamp op public entry and tag. Included in-context from runtime.cuh
// inside namespace tilefoundry::ops.
#pragma once

struct clamp_op {
    float min_val, max_val;
    template <class T> __device__ T operator()(T x) const {
        return x < static_cast<T>(min_val)
                   ? static_cast<T>(min_val)
                   : (x > static_cast<T>(max_val) ? static_cast<T>(max_val)
                                                  : x);
    }
};

// Routed through the shared ``unary_impl::Unary`` skeleton (unary.cuh):
// ``clamp_op`` carries its ``(min_val, max_val)`` bounds as functor state,
// which ``Unary``'s existing single-argument functor call (``op(s(i))``)
// already supports — no generalisation of ``Unary`` was needed.
template <class TIn, class TOut>
__device__ void clamp(TIn const &src, TOut &dst, int N, float min_val,
                      float max_val) {
    unary_impl::Unary<clamp_op>{}(src, dst, N, clamp_op{min_val, max_val});
}
