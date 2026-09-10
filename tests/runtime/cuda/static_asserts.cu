/// The runtime's compile-time refusals, one per translation unit.
#include <tilefoundry/runtime/cuda/runtime.cuh>

namespace tilefoundry {
template <>
CUTE_HOST_DEVICE constexpr auto program_dim<TopologyScope::thread>() noexcept {
    return cute::Int<256>{};
}
template <>
CUTE_HOST_DEVICE constexpr auto program_dim<TopologyScope::cta>() noexcept {
    return cute::Int<8>{};
}
}

using namespace tilefoundry;
using namespace tilefoundry::ops;

template <int... Es> __device__ auto tmesh() {
    return make_mesh<TopologyScope::thread>(
        cute::make_shape(cute::Int<Es>{}...));
}

/// One register held by the instances of a (Groups, Lanes) mesh, with Folded
template <int Groups, int Lanes, int Folded>
__device__ auto fold_cell(float *reg) {
    auto mesh = tmesh<Groups, Lanes>();
    auto layout = cute::make_layout(
        cute::make_shape(cute::Int<(Folded >= 2 ? 1 : Groups)>{},
                         cute::Int<(Folded >= 1 ? 1 : Lanes)>{}),
        cute::GenRowMajor{});
    auto t = cute::make_tensor(cute::make_rmem_ptr(reg), cute::Int<1>{});
    if constexpr (Folded >= 2)
        return make_shard_tensor(
            t, layout,
            make_shard_layout(layout, mesh,
                              cute::make_tuple(shard::B{}, shard::B{})));
    else if constexpr (Folded == 1)
        return make_shard_tensor(
            t, layout,
            make_shard_layout(layout, mesh,
                              cute::make_tuple(shard::S<0>{}, shard::B{})));
    else
        return make_shard_tensor(
            t, layout,
            make_shard_layout(layout, mesh,
                              cute::make_tuple(shard::S<0>{}, shard::S<1>{})));
}

/// The same over a flat (Threads,) mesh: one axis, split or broadcast.
template <int Threads, bool Split_> __device__ auto flat_cell(float *reg) {
    auto mesh = tmesh<Threads>();
    auto layout = cute::make_layout(
        cute::make_shape(cute::Int<(Split_ ? Threads : 1)>{}));
    auto t = cute::make_tensor(cute::make_rmem_ptr(reg), cute::Int<1>{});
    if constexpr (Split_)
        return make_shard_tensor(
            t, layout,
            make_shard_layout(layout, mesh, cute::make_tuple(shard::S<0>{})));
    else
        return make_shard_tensor(
            t, layout,
            make_shard_layout(layout, mesh, cute::make_tuple(shard::B{})));
}

#if CASE == 0
/// Positive control for item 2: a flat (256,) thread mesh splits into 32 lanes
__global__ void k() {
    using src_sl = cute::remove_cvref_t<decltype(flat_cell<256, true>(
        nullptr))>::shard_layout_type;
    using dst_sl = cute::remove_cvref_t<decltype(flat_cell<256, false>(
        nullptr))>::shard_layout_type;
    constexpr auto p = reduce_impl::reduce_dispatch<src_sl, dst_sl>();
    static_assert(p.mesh_reduced && p.lane_reduced, "(256,) reduces");
    static_assert(p.lanes_reduced == 32, "32 of it is one warp's lanes");
    static_assert(p.warps_per_group == 8, "and the rest is 8 warps");

    /// A (2, 64) mesh: the fast axis straddles the warp boundary too.
    using s2 = cute::remove_cvref_t<decltype(fold_cell<2, 64, 0>(
        nullptr))>::shard_layout_type;
    using d2 = cute::remove_cvref_t<decltype(fold_cell<2, 64, 2>(
        nullptr))>::shard_layout_type;
    constexpr auto q = reduce_impl::reduce_dispatch<s2, d2>();
    static_assert(q.lanes_reduced == 32 && q.warps_per_group == 4, "(2,64)");

    /// And the shapes the kernel actually uses stay intra-warp.
    using s3 = cute::remove_cvref_t<decltype(fold_cell<1, 32, 0>(
        nullptr))>::shard_layout_type;
    using d3 = cute::remove_cvref_t<decltype(fold_cell<1, 32, 1>(
        nullptr))>::shard_layout_type;
    constexpr auto r = reduce_impl::reduce_dispatch<s3, d3>();
    static_assert(r.warps_per_group == 1 && r.lanes_reduced == 32, "(1,32)");

    using sparse_layout =
        cute::Layout<cute::Shape<cute::Int<2>, cute::Int<32>>,
                     cute::Stride<cute::Int<64>, cute::Int<1>>>;
    using sparse_mesh = Mesh<sparse_layout, TopologyScope::thread>;
    /// One id per level this program names, as program_ids() gives: cta,
    /// thread.
    constexpr auto sparse_ids = cute::make_tuple(0, 0);
    static_assert(
        tilefoundry::contains(sparse_mesh{sparse_layout{}}, sparse_ids));
    static_assert(tilefoundry::contains(sparse_mesh{sparse_layout{}},
                                        cute::make_tuple(0, 31)));
    static_assert(tilefoundry::contains(sparse_mesh{sparse_layout{}},
                                        cute::make_tuple(0, 64)));
    static_assert(tilefoundry::contains(sparse_mesh{sparse_layout{}},
                                        cute::make_tuple(0, 95)));
    static_assert(!tilefoundry::contains(sparse_mesh{sparse_layout{}},
                                         cute::make_tuple(0, 32)));
    static_assert(!tilefoundry::contains(sparse_mesh{sparse_layout{}},
                                         cute::make_tuple(0, 96)));

    /// A mesh naming two levels: grouped one nest per level, asked one level
    /// at a time, each nest already in that level's own numbering.

    using grouped =
        cute::Layout<cute::Shape<cute::Shape<cute::Int<4>, cute::Int<2>>,
                                 cute::Shape<cute::Int<32>>>,
                     cute::Stride<cute::Stride<cute::Int<2>, cute::Int<1>>,
                                  cute::Stride<cute::Int<1>>>>;
    using mesh = Mesh<grouped, TopologyScope::cta, TopologyScope::thread>;
    constexpr auto cta = tilefoundry::get<TopologyScope::cta>(mesh{});
    constexpr auto thread = tilefoundry::get<TopologyScope::thread>(mesh{});
    static_assert(decltype(cta)::level_count == 1, "one level each");
    static_assert(int(cute::size(cta.layout)) == 8, "8 CTAs of their own");
    static_assert(int(cute::size(thread.layout)) == 32, "32 threads of theirs");
    static_assert(tilefoundry::is_warped(thread), "and that is one warp");
}
#endif

#if CASE == 1
/// Mesh::offset: a swizzle where identity belongs.
__global__ void k() {
    using inner = cute::Swizzle<1, 0, 1>;
    using box =
        cute::Layout<cute::Shape<cute::Int<32>>, cute::Stride<cute::Int<1>>>;
    using ml = cute::ComposedLayout<inner, cute::Int<0>, box>;
    using mesh = Mesh<ml, TopologyScope::thread>;
    static_assert(tilefoundry::offset(mesh{}) == 0, "unreachable");
}
#endif

#if CASE == 2
/// Mesh::offset: a run-time slice origin.
__global__ void k() {
    using box =
        cute::Layout<cute::Shape<cute::Int<32>>, cute::Stride<cute::Int<1>>>;
    using ml = cute::ComposedLayout<cute::identity, int, box>;
    using mesh = Mesh<ml, TopologyScope::thread>;
    static_assert(tilefoundry::offset(mesh{}) == 0, "unreachable");
}
#endif

#if CASE == 3
/// local(): two attrs for a one-axis mesh, which used to read as a full
__global__ void k(float *p) {
    auto mesh = tmesh<256>();
    auto layout = cute::make_layout(cute::make_shape(cute::Int<256>{}));
    auto t = cute::make_tensor(cute::make_gmem_ptr(p), layout);
    ShardLayout<decltype(layout), cute::tuple<shard::S<0>, shard::B>,
                decltype(mesh)>
        sl{layout, mesh};
    auto st = make_shard_tensor(t, layout, sl);
    auto v = tilefoundry::detail::local_tensor(st);
    v(0) = 1.f;
}
#endif

#if CASE == 4
/// make_shard_layout: the same mismatch, refused at construction.
__global__ void k(float *p) {
    auto mesh = tmesh<8, 32>();
    auto layout = cute::make_layout(cute::make_shape(cute::Int<256>{}));
    auto sl = make_shard_layout(layout, mesh, cute::make_tuple(shard::S<0>{}));
    (void)sl;
    (void)p;
}
#endif

#if CASE == 5
/// the offset arithmetic itself, reached without going through local().
__global__ void k(float *p) {
    auto mesh = tmesh<8, 32>();
    auto layout = cute::make_layout(cute::make_shape(cute::Int<256>{}));
    ShardLayout<decltype(layout), cute::tuple<shard::S<0>>, decltype(mesh)> sl{
        layout, mesh};
    p[0] = float(cute::get<1>(tilefoundry::detail::local_layout_and_offset(
        sl,
        tilefoundry::mesh_coords(sl.mesh_value, tilefoundry::program_ids()))));
}
#endif

#if CASE == 6
/// mesh_axis_term: a Dynamic attr, which used to be dropped as a broadcast.
__global__ void k(float *p) {
    auto mesh = tmesh<256>();
    auto layout = cute::make_layout(cute::make_shape(cute::Int<256>{}));
    auto t = cute::make_tensor(cute::make_gmem_ptr(p), layout);
    ShardLayout<decltype(layout), cute::tuple<shard::Dynamic>, decltype(mesh)>
        sl{layout, mesh};
    auto st = make_shard_tensor(t, layout, sl);
    tilefoundry::detail::local_tensor(st)(0) = 1.f;
}
#endif

#if CASE == 7
/// Item 1 at the entry: a mesh spanning 8 warps, reduced with no workspace.
__global__ void k(float *p) {
    float a = p[0], b = 0.f;
    auto src = fold_cell<8, 32, 0>(&a);
    auto dst = fold_cell<8, 32, 2>(&b);
    reduce<add_op, cute::tuple<cute::Int<1>>>(src, dst);
    p[0] = b;
}
#endif

#if CASE == 8
/// Item 1 at the tier: IntraWarp asked to cross warps directly.
__global__ void k(float *p) {
    float a = p[0], b = 0.f;
    auto src = fold_cell<8, 32, 0>(&a);
    auto dst = fold_cell<8, 32, 2>(&b);
    reduce_impl::IntraWarp<add_op, cute::tuple<cute::Int<1>>>{}(src, dst);
    p[0] = b;
}
#endif

#if CASE == 9
/// Item 8: a packed 64-bit key cannot be folded in float.
__global__ void k(unsigned long long *p) {
    unsigned long long a = p[0], b = 0;
    auto src = cute::make_tensor(cute::make_rmem_ptr(&a), cute::Int<1>{});
    auto dst = cute::make_tensor(cute::make_rmem_ptr(&b), cute::Int<1>{});
    reduce<max_op, cute::tuple<cute::Int<0>>>(src, dst);
    p[0] = b;
}
#endif

#if CASE == 10
/// Item 3's guard: a composed source layout has no kept/reduced split.
__global__ void k(float *p) {
    float b = 0.f;
    auto cl = cute::make_composed_layout(
        cute::identity{}, cute::Int<0>{},
        cute::make_layout(cute::make_shape(cute::Int<4>{})));
    auto src = cute::make_tensor(cute::make_gmem_ptr(p), cl);
    auto dst = cute::make_tensor(cute::make_rmem_ptr(&b), cute::Int<1>{});
    reduce<add_op, cute::tuple<cute::Int<0>>>(src, dst);
    p[0] = b;
}
#endif

#if CASE == 11
/// reduce_dispatch: two operands, two different meshes.
__global__ void k(float *p) {
    float a = p[0], b = 0.f;
    auto src = fold_cell<8, 32, 0>(&a);
    auto dst = fold_cell<4, 32, 2>(&b);
    reduce<add_op, cute::tuple<cute::Int<1>>>(src, dst);
    p[0] = b;
}
#endif

#if CASE == 12
/// reduce_dispatch: a mesh whose scope is neither cta nor thread.
__global__ void k(float *p) {
    float a = p[0], b = 0.f;
    using wlayout =
        cute::Layout<cute::Shape<cute::Int<32>>, cute::Stride<cute::Int<1>>>;
    using wmesh = Mesh<wlayout, TopologyScope::scope_count>;
    wmesh mesh{wlayout{}};
    auto sl = cute::make_layout(cute::make_shape(cute::Int<32>{}));
    auto dl = cute::make_layout(cute::make_shape(cute::Int<1>{}));
    auto ta = cute::make_tensor(cute::make_rmem_ptr(&a), cute::Int<1>{});
    auto tb = cute::make_tensor(cute::make_rmem_ptr(&b), cute::Int<1>{});
    auto src = make_shard_tensor(
        ta, sl,
        ShardLayout<decltype(sl), cute::tuple<shard::S<0>>, wmesh>{sl, mesh});
    auto dst = make_shard_tensor(
        tb, dl,
        ShardLayout<decltype(dl), cute::tuple<shard::B>, wmesh>{dl, mesh});
    reduce<add_op, cute::tuple<cute::Int<0>>>(src, dst);
    p[0] = b;
}
#endif

#if CASE == 13
/// Item 9: an intra-CTA reduce whose source keeps two cells.
__global__ void k(float *p) {
    __shared__ float ws[8];
    auto mesh = tmesh<8, 32>();
    auto slay = cute::make_layout(
        cute::make_shape(cute::Int<2>{}, cute::Int<8>{}, cute::Int<32>{}),
        cute::make_stride(cute::Int<256>{}, cute::Int<32>{}, cute::Int<1>{}));
    auto st = make_shard_tensor(
        cute::make_tensor(cute::make_gmem_ptr(p), slay), slay,
        make_shard_layout(slay, mesh,
                          cute::make_tuple(shard::S<1>{}, shard::S<2>{})));
    auto dlay = cute::make_layout(cute::make_shape(cute::Int<2>{}));
    auto dt = make_shard_tensor(
        cute::make_tensor(cute::make_gmem_ptr(p + 512), dlay), dlay,
        make_shard_layout(dlay, mesh,
                          cute::make_tuple(shard::B{}, shard::B{})));
    auto slots = cute::make_tensor(cute::make_smem_ptr(ws), cute::Int<8>{});
    reduce<add_op, cute::tuple<cute::Int<1>, cute::Int<2>>>(st, dt, slots);
}
#endif

#if CASE == 14
/// Item 6: a pair of views cp.async cannot move a wide word of.
__global__ void k(float *g, int n) {
    __shared__ float s[64];
    auto src = cute::make_tensor(cute::make_gmem_ptr(g), cute::make_layout(n));
    auto dst = cute::make_tensor(cute::make_smem_ptr(s), cute::make_layout(n));
    copy_async(src, dst);
}
#endif

#if CASE == 15
/// Item 7: a destination wider than the source, with no offset to borrow.
__global__ void k(float *g) {
    __shared__ float s[64];
    auto src = cute::make_tensor(cute::make_gmem_ptr(g), cute::Int<4>{});
    auto dst = cute::make_tensor(cute::make_smem_ptr(s), cute::Int<8>{});
    copy_async(src, dst);
}
#endif

#if CASE == 16
/// dot's warp tier: a mesh whose lane axis is half a warp.
__global__ void k(float *p) {
    float o = 0.f;
    auto mesh = tmesh<2, 16>();
    auto lay = cute::make_layout(
        cute::make_shape(cute::Int<4>{}, cute::Int<16>{}, cute::Int<2>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<4>{}, cute::Int<64>{}));
    auto lhs = make_shard_tensor(
        cute::make_tensor(cute::make_gmem_ptr(p), lay), lay,
        make_shard_layout(lay, mesh,
                          cute::make_tuple(shard::S<2>{}, shard::S<1>{})));
    auto rlay =
        cute::make_layout(cute::make_shape(cute::Int<4>{}, cute::Int<16>{}),
                          cute::make_stride(cute::Int<1>{}, cute::Int<4>{}));
    auto rhs = make_shard_tensor(
        cute::make_tensor(cute::make_gmem_ptr(p + 128), rlay), rlay,
        make_shard_layout(rlay, mesh,
                          cute::make_tuple(shard::B{}, shard::S<1>{})));
    auto out = cute::make_tensor(cute::make_rmem_ptr(&o), cute::Int<1>{});
    dot(lhs, rhs, out);
    p[0] = o;
}
#endif

#if CASE == 17
/// dot's block tier: a mesh that is not a whole number of warps.
__global__ void k(float *p) {
    __shared__ float ws[8];
    float o = 0.f;
    auto mesh = tmesh<16>();
    auto lay =
        cute::make_layout(cute::make_shape(cute::Int<4>{}, cute::Int<16>{}),
                          cute::make_stride(cute::Int<1>{}, cute::Int<4>{}));
    auto lhs = make_shard_tensor(
        cute::make_tensor(cute::make_gmem_ptr(p), lay), lay,
        make_shard_layout(lay, mesh, cute::make_tuple(shard::S<1>{})));
    auto rhs = make_shard_tensor(
        cute::make_tensor(cute::make_gmem_ptr(p + 64), lay), lay,
        make_shard_layout(lay, mesh, cute::make_tuple(shard::S<1>{})));
    auto out = cute::make_tensor(cute::make_rmem_ptr(&o), cute::Int<1>{});
    auto slots = cute::make_tensor(cute::make_smem_ptr(ws), cute::Int<8>{});
    dot(lhs, rhs, out, slots);
    p[0] = o;
}
#endif

#if CASE == 18
/// mma: operands that are neither a tile nor the atom's fragments.
__global__ void k(float *p) {
    float a[3] = {}, b[3] = {}, c[3] = {};
    auto at = cute::make_tensor(cute::make_rmem_ptr(&a[0]), cute::Int<3>{});
    auto bt = cute::make_tensor(cute::make_rmem_ptr(&b[0]), cute::Int<3>{});
    auto ct = cute::make_tensor(cute::make_rmem_ptr(&c[0]), cute::Int<3>{});
    mma(at, bt, ct);
    p[0] = c[0];
}
#endif

#if CASE == 19
/// ops::sync: a mesh whose scope is neither cta nor thread names no barrier.
__global__ void k() {
    using wlayout =
        cute::Layout<cute::Shape<cute::Int<32>>, cute::Stride<cute::Int<1>>>;
    Mesh<wlayout, TopologyScope::scope_count> mesh{wlayout{}};
    sync(mesh);
}
#endif

#if CASE == 20
/// ops::sync: a CTA mesh handed no grid-barrier counter.
__global__ void k() {
    auto mesh = make_mesh<TopologyScope::cta>(cute::make_shape(cute::Int<8>{}));
    sync(mesh);
}
#endif

#if CASE == 21
/// mma's tile tier: an accumulator mesh that is not a whole number of warps.
__global__ void k(cute::bfloat16_t *p, float *q) {
    auto mesh = tmesh<16>();
    auto alay =
        cute::make_layout(cute::make_shape(cute::Int<16>{}, cute::Int<16>{}),
                          cute::make_stride(cute::Int<16>{}, cute::Int<1>{}));
    auto a = make_shard_tensor(
        cute::make_tensor(cute::make_smem_ptr(p), alay), alay,
        make_shard_layout(alay, mesh, cute::make_tuple(shard::B{})));
    auto blay =
        cute::make_layout(cute::make_shape(cute::Int<8>{}, cute::Int<16>{}),
                          cute::make_stride(cute::Int<16>{}, cute::Int<1>{}));
    auto b = make_shard_tensor(
        cute::make_tensor(cute::make_smem_ptr(p + 256), blay), blay,
        make_shard_layout(blay, mesh, cute::make_tuple(shard::B{})));
    auto clay = cute::make_layout(cute::make_shape(cute::Int<8>{}));
    auto c = make_shard_tensor(
        cute::make_tensor(cute::make_rmem_ptr(q), clay), clay,
        make_shard_layout(clay, mesh, cute::make_tuple(shard::B{})));
    mma(a, b, c);
}
#endif

#if CASE == 22
/// A reduced axis that divides into neither whole lanes nor whole warps.
__global__ void k(float *p) {
    float a = p[0], b = 0.f;
    auto src = flat_cell<48, true>(&a);
    auto dst = flat_cell<48, false>(&b);
    reduce<add_op, cute::tuple<cute::Int<0>>>(src, dst);
    p[0] = b;
}
#endif

#if CASE == 24
/// A ``tma_copy`` source a mesh really does divide.
__global__ void k(float *p, float *q, uint64_t *bar) {
    auto mesh = tmesh<128>();
    auto src_lay =
        cute::make_layout(cute::make_shape(cute::Int<4>{}, cute::Int<128>{}),
                          cute::make_stride(cute::Int<128>{}, cute::Int<1>{}));
    auto dst_lay = cute::make_layout(cute::make_shape(cute::Int<512>{}),
                                     cute::make_stride(cute::Int<1>{}));
    auto src = make_shard_tensor(
        cute::make_tensor(cute::make_gmem_ptr(p), src_lay), src_lay,
        make_shard_layout(src_lay, mesh, cute::make_tuple(shard::S<1>{})));
    auto dst = make_shard_tensor(
        cute::make_tensor(cute::make_smem_ptr(q), dst_lay), dst_lay,
        make_shard_layout(dst_lay, mesh, cute::make_tuple(shard::B{})));
    tilefoundry::ops::tma_copy(src, dst, bar);
}
#endif

#if CASE == 25
/// Warps 0 and 2 of a block: two whole warps, and a gap between them.
__global__ void k() {
    using layout = cute::Layout<cute::Shape<cute::Int<2>, cute::Int<32>>,
                                cute::Stride<cute::Int<64>, cute::Int<1>>>;
    using mesh = Mesh<layout, TopologyScope::thread>;
    static_assert(tilefoundry::is_warped(mesh{}), "whole warps, so warped");
    static_assert(int(cute::size(tilefoundry::as_warped(mesh{}).layout)) == 64);
    static_assert(sync_impl::warps_run_together<mesh>(),
                  "ops::sync: a mesh that skips warps names no barrier");
}
#endif
#if CASE == 26
/// The level a mesh does not name.
__global__ void k() {
    using mesh = Mesh<
        cute::Layout<cute::Shape<cute::Int<32>>, cute::Stride<cute::Int<1>>>,
        TopologyScope::thread>;
    constexpr auto cta = tilefoundry::get<TopologyScope::cta>(mesh{});
    static_assert(int(cute::size(cta.layout)) == 32, "unreachable");
}
#endif
