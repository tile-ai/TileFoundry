// CUDA cast op public entry. Included in-context from runtime.cuh inside
// namespace tilefoundry::ops.
#pragma once

// ``cast`` is a plain per-element static_cast — routed through the shared
// ``unary_impl::Unary`` skeleton via the ``identity_op`` tag (unary.cuh);
// the output-side cast in ``Unary::operator()`` performs the conversion.
template <class TIn, class TOut>
__device__ void cast(TIn const &src, TOut &dst, int N) {
    unary_impl::Unary<identity_op>{}(src, dst, N);
}
