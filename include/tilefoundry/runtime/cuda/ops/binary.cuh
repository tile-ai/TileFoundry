// CUDA binary op public entries and tags. Included in-context from runtime.cuh
// inside namespace tilefoundry::ops.
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
struct div_op {
    template <class T> __device__ T operator()(T a, T b) const { return a / b; }
};

#include "binary/binary_impl.h"

template <class Op, class TL, class TR, class TOut>
__device__ void binary(TL const &lhs, TR const &rhs, TOut &dst, int N,
                       Op op = {}) {
    binary_impl::Binary<Op>{}(lhs, rhs, dst, N, op);
}

template <class Op, class TL, class TR, class TOut>
__device__ void binary_cell_bcast(TL const &lhs, TR const &rhs, TOut &dst,
                                  int n_dst, int step, Op op = {}) {
    binary_impl::BinaryCellBcast<Op>{}(lhs, rhs, dst, n_dst, step, op);
}

template <class Op, class TL, class TR, class TOut>
__device__ void binary_bcast_col(TL const &lhs, TR const &rhs, TOut &dst, int M,
                                 int K, Op op = {}) {
    binary_impl::BinaryBcastCol<Op>{}(lhs, rhs, dst, M, K, op);
}

template <class Op, class TL, class TR, class TOut>
__device__ void binary_bcast_row(TL const &lhs, TR const &rhs, TOut &dst, int M,
                                 int K, Op op = {}) {
    binary_impl::BinaryBcastRow<Op>{}(lhs, rhs, dst, M, K, op);
}

template <class Op, class TS, class TV, class TOut>
__device__ void binary_bcast_scalar(TS const &src, TV const &scalar, TOut &dst,
                                    int N, Op op = {}) {
    binary_impl::BinaryBcastScalar<Op>{}(src, scalar, dst, N, op);
}
