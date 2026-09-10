/// Unary callable tags for elementwise.
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
/// Stable softplus callable.
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
/// Identity callable.
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
