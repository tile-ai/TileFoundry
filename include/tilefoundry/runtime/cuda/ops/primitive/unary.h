/// Unary callable tags for ``elementwise``. Included in-context from
/// runtime.cuh inside namespace tilefoundry::ops.
///
/// Tags and nothing else: values a caller hands an op as its callable, so they
/// are not ops and do not sit beside the entries in ``ops/``. ``reduce`` and
/// ``dot`` specialise ``reduce_traits`` on some of the arity-2 tags, which is
/// what makes the elementwise maximum of two tensors and the maximum over an
/// axis one name rather than two that must be kept meaning the same thing.
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
/// The logistic curve, and the two things built on it.
///
/// Computed in float whatever the operand type is: the exponential of a bf16
/// argument rounds twice, once into the exponential and once out of it, and the
/// second rounding is the one that shows up in a norm.
struct sigmoid_op {
    template <class T> __device__ T operator()(T x) const {
        const float v = static_cast<float>(x);
        return static_cast<T>(1.0f / (1.0f + __expf(-v)));
    }
};
struct silu_op {
    template <class T> __device__ T operator()(T x) const {
        const float v = static_cast<float>(x);
        return static_cast<T>(v / (1.0f + __expf(-v)));
    }
};
/// ``log1pf(expf(x))``, taking the identity above 20 where the two agree to
/// well past float precision and ``expf`` would otherwise overflow.
struct softplus_op {
    template <class T> __device__ T operator()(T x) const {
        const float v = static_cast<float>(x);
        return static_cast<T>(v > 20.0f ? v : log1pf(expf(v)));
    }
};
struct exp_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(expf(static_cast<float>(x)));
    }
};
struct log_op {
    template <class T> __device__ T operator()(T x) const {
        return static_cast<T>(logf(static_cast<float>(x)));
    }
};
/// Forwards its argument unchanged, so ``elementwise(dst, identity_op{}, src)``
/// is a per-element map whose only effect is the conversion ``dst(i) = ...``
/// implies -- which is what a cast is.
struct identity_op {
    template <class T> __device__ T operator()(T x) const { return x; }
};
/// Bounds as functor state, which is why a clamp needs no entry of its own.
struct clamp_op {
    float min_val, max_val;
    template <class T> __device__ T operator()(T x) const {
        return x < static_cast<T>(min_val)
                   ? static_cast<T>(min_val)
                   : (x > static_cast<T>(max_val) ? static_cast<T>(max_val)
                                                  : x);
    }
};
