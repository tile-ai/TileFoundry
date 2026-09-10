/// CUDA MMA op implementation. Included in-context from ops/mma.cuh inside
/// namespace tilefoundry::ops.
#pragma once

namespace mma_detail {

__device__ inline uint32_t pack_bf16x2(uint16_t lo, uint16_t hi) {
    return (uint32_t(hi) << 16) | uint32_t(lo);
}

template <class T> __device__ uint16_t as_u16(T const &x) {
    uint16_t out;
    __builtin_memcpy(&out, &x, sizeof(uint16_t));
    return out;
}

/// Geometry of the supported atom, read from CuTe's canonical trait.
using AtomOp = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using AtomTraits = cute::MMA_Traits<AtomOp>;
using AtomCLayout = typename AtomTraits::CLayout;

struct AtomGeometry {
    using ShapeMNK = typename AtomTraits::Shape_MNK;
    static constexpr int kM = int(cute::get<0>(ShapeMNK{}));
    static constexpr int kN = int(cute::get<1>(ShapeMNK{}));
    static constexpr int kK = int(cute::get<2>(ShapeMNK{}));
    static constexpr int kAccPerLane =
        int(cute::size(AtomCLayout{})) / kWarpSize;

    __device__ static int c_coord(int lane, int value = 0) {
        return int(AtomCLayout{}(cute::make_coord(lane, value)));
    }

    __device__ static int row(int lane) { return c_coord(lane) % kM; }
    __device__ static int col(int lane) { return c_coord(lane) / kM; }
};

}

namespace mma_impl {

/// Execute the atom on this lane's gathered fragments.
struct Atom {
    template <class TA, class TB, class TC>
    __device__ void operator()(TA const &a, TB const &b, TC &c) const {
        using CuteAtom = mma_detail::AtomOp;
        using namespace mma_detail;

        auto a_data = a.data();
        auto b_data = b.data();
        auto c_data = c.data();

        uint32_t a0 = pack_bf16x2(as_u16(a_data[0]), as_u16(a_data[1]));
        uint32_t a1 = pack_bf16x2(as_u16(a_data[4]), as_u16(a_data[5]));
        uint32_t a2 = pack_bf16x2(as_u16(a_data[2]), as_u16(a_data[3]));
        uint32_t a3 = pack_bf16x2(as_u16(a_data[6]), as_u16(a_data[7]));

        uint32_t b0 = pack_bf16x2(as_u16(b_data[0]), as_u16(b_data[1]));
        uint32_t b1 = pack_bf16x2(as_u16(b_data[2]), as_u16(b_data[3]));

        float c0 = c_data[0];
        float c1 = c_data[1];
        float c2 = c_data[2];
        float c3 = c_data[3];
        float d0, d1, d2, d3;

        CuteAtom::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);

        c_data[0] = d0;
        c_data[1] = d1;
        c_data[2] = d2;
        c_data[3] = d3;
    }
};

/// A rank-2 static layout is a tile; anything else is a gathered fragment.
template <class T>
inline constexpr bool tile_v = [] {
    using L = typename cute::remove_cvref_t<T>::layout_type;
    return cute::is_static<L>::value && decltype(cute::rank(L{}))::value == 2;
}();

/// How many threads the accumulator's mesh spreads the tile over.
template <class TC> CUTE_HOST_DEVICE constexpr int acc_threads() {
    return tilefoundry::shard_mesh_instances<TC>();
}

/// ``acc += a @ b`` over a whole tile, the atom looped by the tile's shape.
struct Tile {
    template <class TA, class TB, class TC>
    __device__ void operator()(TA const &a, TB const &b, TC &c) const {
        using Geo = mma_detail::AtomGeometry;
        auto av = detail::to_local(a);
        auto bv = detail::to_local(b);
        auto &&cv = detail::to_local(c);
        using a_elem = cute::remove_cvref_t<decltype(av(0, 0))>;

        constexpr int threads = acc_threads<TC>();
        static_assert(threads >= kWarpSize && threads % kWarpSize == 0,
                      "ops::mma (tile tier): the accumulator's mesh must be a "
                      "whole number of warps");
        constexpr int warps = threads / kWarpSize;
        constexpr int M = int(cute::size<0>(
            typename cute::remove_cvref_t<TA>::shard_layout_type::layout{}));
        constexpr int K = int(cute::size<1>(
            typename cute::remove_cvref_t<TA>::shard_layout_type::layout{}));
        constexpr int N = int(cute::size<0>(
            typename cute::remove_cvref_t<TB>::shard_layout_type::layout{}));
        static_assert(M % Geo::kM == 0 && K % Geo::kK == 0,
                      "the A tile must be a whole number of atoms");
        static_assert(N % (Geo::kN * warps) == 0,
                      "every warp must get a whole number of N atoms");

        constexpr int m_atoms = M / Geo::kM;
        constexpr int n_atoms = N / (Geo::kN * warps);

        const unsigned tid = unsigned(
            tilefoundry::program_id<tilefoundry::TopologyScope::thread>());
        const int lane = int(tid & unsigned(kWarpSize - 1));
        const int warp = int((tid / unsigned(kWarpSize)) % unsigned(warps));
        const int row = Geo::row(lane);
        const int col = Geo::col(lane);

        a_elem af[8];
        a_elem bf[4];
        float acc[Geo::kAccPerLane];

        CUTE_UNROLL
        for (int mi = 0; mi < m_atoms; ++mi) {
            const int m0 = mi * Geo::kM + row;
            CUTE_UNROLL
            for (int ni = 0; ni < n_atoms; ++ni) {
                const int n0 = (warp * n_atoms + ni) * Geo::kN + row;
                const int base = (mi * n_atoms + ni) * Geo::kAccPerLane;
                CUTE_UNROLL
                for (int v = 0; v < Geo::kAccPerLane; ++v)
                    acc[v] = cv(base + v);

                for (int k0 = 0; k0 < K; k0 += Geo::kK) {
                    const int kc = k0 + col;
                    CUTE_UNROLL
                    for (int h = 0; h < 2; ++h) {
                        af[0 + h] = av(m0, kc + h);
                        af[2 + h] = av(m0, kc + 8 + h);
                        af[4 + h] = av(m0 + 8, kc + h);
                        af[6 + h] = av(m0 + 8, kc + 8 + h);
                        bf[0 + h] = bv(n0, kc + h);
                        bf[2 + h] = bv(n0, kc + 8 + h);
                    }
                    auto at = cute::make_tensor(&af[0], cute::Int<8>{});
                    auto bt = cute::make_tensor(&bf[0], cute::Int<4>{});
                    auto ct = cute::make_tensor(&acc[0],
                                                cute::Int<Geo::kAccPerLane>{});
                    Atom{}(at, bt, ct);
                }

                CUTE_UNROLL
                for (int v = 0; v < Geo::kAccPerLane; ++v)
                    cv(base + v) = acc[v];
            }
        }
    }
};

template <class TA, class TB, class TC>
inline constexpr bool tile_shaped_v =
    tile_v<tilefoundry::detail::local_view_t<TA>> &&
    tile_v<tilefoundry::detail::local_view_t<TB>>;

/// Recognize the atom's per-lane fragment shapes.
template <class T, int N>
inline constexpr bool frag_v = [] {
    using L = typename cute::remove_cvref_t<T>::layout_type;
    if constexpr (!cute::is_static<L>::value)
        return false;
    else
        return int(decltype(cute::size(L{}))::value) == N;
}();

/// The atom's own operand shapes, stated positively.
template <class TA, class TB, class TC>
inline constexpr bool atom_shaped_v =
    frag_v<tilefoundry::detail::local_view_t<TA>, 8> &&
    frag_v<tilefoundry::detail::local_view_t<TB>, 4> &&
    frag_v<tilefoundry::detail::local_view_t<TC>, 4>;

}
