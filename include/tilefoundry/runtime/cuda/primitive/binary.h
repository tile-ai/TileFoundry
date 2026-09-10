/// Binary callable tags shared by elementwise, reduce, and dot.
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
/// Maximum and minimum also serve as reduce tags.
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
