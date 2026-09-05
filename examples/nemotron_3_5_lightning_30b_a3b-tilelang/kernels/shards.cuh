/// The layouts this kernel's stages are written against.
///
/// Every thread mapping in the file is one of these, so a stage says which
/// shape it wants and never writes `threadIdx`. What a lane owns, how wide its
/// loads are and where its slice starts all come out of `local()`.
#pragma once

/// The widest block of a row one lane can own so that 32 of them tile it.
///
/// Capped at 16 bytes and then cut down until it divides the per-lane share.
/// A 4096-wide row leaves each lane 128 and loads 16 bytes; a 2688-wide one
/// leaves 84 and loads 8. That is the whole reason this is not a constant.
constexpr int lane_block(int n) {
    const int per = n / 32;
    int v = 1;
    while (v * 2 <= 8 && per % (v * 2) == 0) v *= 2;
    return v;
}

/// The widest block one thread of a `Threads`-wide mesh can own, and how much
/// of a run of `N` such a mesh covers exactly.
///
/// A mesh divides or it does not: 4096 over 256 threads is 16 each and there is
/// nothing left, while 2688 is 10.5 and something always is. The pair below is
/// what a caller needs to cover the whole run -- the split part, then the
/// remainder, which is shorter than the block and is one element a thread.
constexpr int block_vec(int n, int threads) {
    int best_v = 1, best_span = (n / threads) * threads;
    for (int v = 2; v <= 8; v *= 2) {
        const int span = (n / (threads * v)) * threads * v;
        if (span >= best_span) {
            best_span = span;
            best_v = v;
        }
    }
    return best_v;
}
constexpr int block_span(int n, int threads) {
    const int v = block_vec(n, threads);
    return (n / (threads * v)) * threads * v;
}

/// The block's threads as `(32 lanes, warps)` -- what a per-warp row wants.
template <int Threads> __device__ auto lane_warp_mesh() {
    auto layout = cute::make_layout(
        cute::make_shape(cute::Int<32>{}, cute::Int<Threads / 32>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<32>{}));
    return tilefoundry::Mesh<
        tilefoundry::Topology<tilefoundry::TopologyScope::thread, Threads>,
        decltype(layout)>{layout};
}

/// The block's threads as one flat axis -- what a whole-block pass wants.
template <int Threads> __device__ auto flat_mesh() {
    auto layout = cute::make_layout(cute::make_shape(cute::Int<Threads>{}),
                                    cute::make_stride(cute::Int<1>{}));
    return tilefoundry::Mesh<
        tilefoundry::Topology<tilefoundry::TopologyScope::thread, Threads>,
        decltype(layout)>{layout};
}

/// `Rows` rows of an `(rows, N)` matrix: one row a warp, split across its lanes.
///
/// This is the operand shape `ops::dot` folds a matrix-vector row with. The
/// mesh axes point at the tensor axes they divide, so warp `w` lands on row `w`
/// and lane `l` on its `l`-th block without either index being written.
template <int N, int Rows, int Threads, class Ptr>
__device__ auto row_tile(Ptr p) {
    constexpr int V = lane_block(N);
    static_assert(Rows == Threads / 32, "one row a warp");
    auto layout = cute::make_layout(
        cute::make_shape(cute::Int<V>{}, cute::Int<32>{},
                         cute::Int<N / (32 * V)>{}, cute::Int<Rows>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<V>{}, cute::Int<32 * V>{},
                          cute::Int<N>{}));
    auto mesh = lane_warp_mesh<Threads>();
    tilefoundry::ShardLayout<decltype(layout),
                             cute::tuple<tilefoundry::shard::S<1>,
                                         tilefoundry::shard::S<3>>,
                             decltype(mesh)>
        shard{layout, mesh};
    return tilefoundry::make_shard_tensor(cute::make_tensor(p, layout), layout,
                                          shard);
}

/// The vector a `row_tile` contracts against: the same lane split, and every
/// warp sees all of it.
template <int N, int Threads, class Ptr> __device__ auto lane_vector(Ptr p) {
    constexpr int V = lane_block(N);
    auto layout = cute::make_layout(
        cute::make_shape(cute::Int<V>{}, cute::Int<32>{},
                         cute::Int<N / (32 * V)>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<V>{}, cute::Int<32 * V>{}));
    auto mesh = lane_warp_mesh<Threads>();
    tilefoundry::ShardLayout<
        decltype(layout),
        cute::tuple<tilefoundry::shard::S<1>, tilefoundry::shard::B>,
        decltype(mesh)>
        shard{layout, mesh};
    return tilefoundry::make_shard_tensor(cute::make_tensor(p, layout), layout,
                                          shard);
}

/// A run of `N` split across all the block's threads, contiguous blocks each.
///
/// `N` must be what the mesh covers exactly -- `block_span` says how much that
/// is, and a caller with a remainder takes it separately.
template <int N, int Threads, class Ptr> __device__ auto block_run(Ptr p) {
    constexpr int V = block_vec(N, Threads);
    static_assert(N % (Threads * V) == 0,
                  "block_run wants a run the mesh divides; use block_span");
    auto layout = cute::make_layout(
        cute::make_shape(cute::Int<V>{}, cute::Int<Threads>{},
                         cute::Int<N / (Threads * V)>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<V>{},
                          cute::Int<Threads * V>{}));
    auto mesh = flat_mesh<Threads>();
    tilefoundry::ShardLayout<decltype(layout),
                             cute::tuple<tilefoundry::shard::S<1>>,
                             decltype(mesh)>
        shard{layout, mesh};
    return tilefoundry::make_shard_tensor(cute::make_tensor(p, layout), layout,
                                          shard);
}

/// The run a `Threads`-wide mesh covers once it is padded up to divide.
///
/// A pad is cheaper than a remainder pass: the padded elements are zeros a sum
/// ignores, while the alternative is a second, narrower shard for a tail
/// shorter than one block. A width that would pad by more than a fifth is
/// rejected -- the loads get wider but the work grows faster.
constexpr int padded_vec(int n, int threads) {
    int best = 1;
    for (int v = 2; v <= 8; v *= 2) {
        const int span = ((n + threads * v - 1) / (threads * v)) * threads * v;
        if (span * 5 <= n * 6) best = v;
    }
    return best;
}
constexpr int padded_span(int n, int threads) {
    const int v = padded_vec(n, threads);
    return ((n + threads * v - 1) / (threads * v)) * threads * v;
}

/// A rank-1 run every thread sees whole -- a bulk copy's two ends, a barrier's
/// tile, anything the block moves rather than divides.
template <int N, int Threads, class Ptr> __device__ auto whole_run(Ptr p) {
    auto layout = cute::make_layout(cute::make_shape(cute::Int<N>{}),
                                    cute::make_stride(cute::Int<1>{}));
    auto mesh = flat_mesh<Threads>();
    tilefoundry::ShardLayout<decltype(layout),
                             cute::tuple<tilefoundry::shard::B>, decltype(mesh)>
        shard{layout, mesh};
    return tilefoundry::make_shard_tensor(cute::make_tensor(p, layout), layout,
                                          shard);
}

/// A one-cell destination in registers, for the ops that write a scalar.
template <class T> __device__ auto cell(T *p) {
    return cute::make_tensor(cute::make_rmem_ptr(p), cute::Int<1>{});
}

/// Where element `u` of this thread's slice of `t` sits in the whole run.
///
/// The shard knows; the caller should not recompute it. Stages that need a
/// genuine index -- which head a channel belongs to, which row of a strided
/// side table -- ask here rather than rebuilding the mapping from `threadIdx`.
template <class View, class T>
__device__ int index_of(View const &v, int u, T const *base) {
    return int(&v(u) - base);
}

/// How many elements this thread's slice of `t` holds.
template <class T> __device__ int local_n(T const &t) {
    return int(cute::size(tilefoundry::local(t)));
}

/// Move a run of `N`, whatever `N` is.
///
/// `ops::copy` takes what the mesh divides and the remainder -- shorter than
/// one block -- is one element a thread. That second statement is the only
/// place a length like 2688 shows: downstream stages work on a padded buffer
/// and never see it.
template <int N, int Threads, class SPtr, class DPtr>
__device__ void copy_run(SPtr src, DPtr dst) {
    constexpr int span = block_span(N, Threads);
    auto s = block_run<span, Threads>(src);
    auto d = block_run<span, Threads>(dst);
    tilefoundry::ops::copy(s, d);
    if constexpr (N > span) {
        const int i = int(threadIdx.x);
        if (i < N - span) dst[span + i] = src[span + i];
    }
}
