/// Binary callable tags for ``elementwise``, also used by ``reduce`` and
/// ``dot``. Included in-context from runtime.cuh inside namespace
/// tilefoundry::ops.
///
/// Tags and nothing else: values a caller hands an op as its callable, so they
/// are not ops and do not sit beside the entries in ``ops/``. ``reduce`` and
/// ``dot`` specialise ``reduce_traits`` on some of the arity-2 tags, which is
/// what makes the elementwise maximum of two tensors and the maximum over an
/// axis one name rather than two that must be kept meaning the same thing.
#pragma once

struct mul_op {
    template <class T> __device__ T operator()(T a, T b) const { return a * b; }
};
struct add_op {
    template <class T> __device__ T operator()(T a, T b) const { return a + b; }
};
struct sub_op {
    template <class T> __device__ T operator()(T a, T b) const { return a - b; }
};
/// Also the reduce tags: ``reduce`` specialises ``reduce_traits`` on these, so
/// the elementwise maximum of two tensors and the maximum over an axis are one
/// name rather than two that must be kept meaning the same thing.
/// ``fmaxf``/``fminf`` where they apply: they are one instruction, while the
/// comparison is a predicate and a select. The generic form stays for the types
/// that have no intrinsic -- a packed key reduced with ``max_op`` takes it.
struct max_op {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
    template <class T> __device__ T operator()(T a, T b) const {
        return a > b ? a : b;
    }
};
struct min_op {
    __device__ float operator()(float a, float b) const { return fminf(a, b); }
    template <class T> __device__ T operator()(T a, T b) const {
        return a < b ? a : b;
    }
};
struct div_op {
    template <class T> __device__ T operator()(T a, T b) const { return a / b; }
};
