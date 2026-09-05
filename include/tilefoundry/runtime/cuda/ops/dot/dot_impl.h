/// CUDA dot op implementation. Included in-context from ops/dot.cuh inside
/// namespace tilefoundry::ops.
#pragma once

namespace dot_impl {

/// The width both operands run contiguously over, as ``copy`` computes it.
template <class Lhs, class Rhs>
inline constexpr int width_v =
    tilefoundry::detail::shard_vector_elems<Lhs, Rhs>();

/// Independent partial sums, so a row is not one chain of dependent FMAs.
///
/// A single running accumulator makes the whole row serial and the loads cannot
/// run ahead of it: what the warp waits on is then the row's latency, not its
/// bytes. Eight is enough to cover the FMA pipeline and few enough not to cost
/// occupancy. The tree at the end is this op's summation order, so a result is
/// reproducible without knowing the row length.
struct Partials {
    static constexpr int kWays = 8;
    float p[kWays];

    __device__ Partials() {
        CUTE_UNROLL
        for (int i = 0; i < kWays; ++i)
            p[i] = 0.f;
    }
    __device__ float total() const {
        return ((p[0] + p[1]) + (p[2] + p[3])) +
               ((p[4] + p[5]) + (p[6] + p[7]));
    }
};

/// This lane's share of the product, folded.
///
/// ``V`` comes from the layouts, so a contiguous run is read a vector at a time
/// and a strided one element by element -- the same question ``copy`` asks, and
/// the reason neither the width nor the stride appears at the call site.
template <int V, class AView, class BView>
__device__ float fold(AView const &a, BView const &b, int n) {
    using a_val = tilefoundry::detail::elem_of<AView>;
    using b_val = tilefoundry::detail::elem_of<BView>;
    Partials acc;
    int i = 0;
    if constexpr (V > 1) {
        using a_vec = tilefoundry::detail::raw_vec<V *int(sizeof(a_val))>;
        using b_vec = tilefoundry::detail::raw_vec<V *int(sizeof(b_val))>;
        if ((reinterpret_cast<uintptr_t>(&a(0)) % alignof(a_vec)) == 0 &&
            (reinterpret_cast<uintptr_t>(&b(0)) % alignof(b_vec)) == 0) {
            for (; i + V <= n; i += V) {
                a_vec av = *reinterpret_cast<a_vec const *>(&a(i));
                b_vec bv = *reinterpret_cast<b_vec const *>(&b(i));
                auto const *ap = reinterpret_cast<a_val const *>(&av);
                auto const *bp = reinterpret_cast<b_val const *>(&bv);
                CUTE_UNROLL
                for (int k = 0; k < V; ++k)
                    acc.p[k % Partials::kWays] +=
                        tilefoundry::detail::convert<float>(ap[k]) *
                        tilefoundry::detail::convert<float>(bp[k]);
            }
        }
    }
    for (int k = 0; i < n; ++i, ++k)
        acc.p[k % Partials::kWays] +=
            tilefoundry::detail::convert<float>(a(i)) *
            tilefoundry::detail::convert<float>(b(i));
    return acc.total();
}

/// The contraction lives inside a warp: one butterfly finishes it.
struct Warp {
    template <class Lhs, class Rhs, class Dst>
    __device__ void operator()(Lhs const &lhs, Rhs const &rhs, Dst &dst) const {
        auto a = detail::to_local(lhs);
        auto b = detail::to_local(rhs);
        auto &&d = detail::to_local(dst);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        const float sum = ops::warp_reduce<ops::warp_sum>(
            fold<width_v<Lhs, Rhs>>(a, b, int(cute::size(a))));
        d(0) = static_cast<value_type>(sum);
    }
};

/// The contraction spans the block: each warp posts a partial, then every
/// thread folds the posted ones.
///
/// The second fold is over one value per warp, which is short enough that a
/// serial loop beats a second butterfly, and it leaves the total in every
/// thread rather than in one.
struct Cta {
    template <class Lhs, class Rhs, class Dst, class Ws>
    __device__ void operator()(Lhs const &lhs, Rhs const &rhs, Dst &dst,
                               Ws &ws) const {
        auto a = detail::to_local(lhs);
        auto b = detail::to_local(rhs);
        auto &&d = detail::to_local(dst);
        auto &&slots = detail::to_local(ws);
        using value_type = cute::remove_cvref_t<decltype(d(0))>;
        const int warps = int(blockDim.x) >> 5;
        const float part = ops::warp_reduce<ops::warp_sum>(
            fold<width_v<Lhs, Rhs>>(a, b, int(cute::size(a))));
        if ((threadIdx.x & 31) == 0)
            slots(int(threadIdx.x) >> 5) = part;
        ops::sync<ops::SyncKind::syncthreads>();
        float sum = 0.f;
        for (int w = 0; w < warps; ++w)
            sum += float(slots(w));
        d(0) = static_cast<value_type>(sum);
    }
};

}
