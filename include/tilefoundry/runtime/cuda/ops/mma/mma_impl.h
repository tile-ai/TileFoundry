/// CUDA MMA op implementation. Included in-context from ops/mma.cuh inside
/// namespace tilefoundry::ops.
#pragma once

namespace mma_detail {

__device__ uint32_t pack_bf16x2(uint16_t lo, uint16_t hi) {
    return (uint32_t(hi) << 16) | uint32_t(lo);
}

template <class T> __device__ uint16_t as_u16(T const &x) {
    uint16_t out;
    __builtin_memcpy(&out, &x, sizeof(uint16_t));
    return out;
}

/// Geometry of the ``m16n8k16`` atom, in the order its PTX operands take.
///
/// Every index the tile tier computes comes from here rather than from a
/// literal at the use site, so the four places that agree on this map -- the A
/// gather, the B gather, the accumulator read and its write-back -- cannot
/// drift apart.
struct Sm80_16x8x16 {
    static constexpr int kM = 16;
    static constexpr int kN = 8;
    static constexpr int kK = 16;
    static constexpr int kAccPerLane = 4;

    __device__ static int row(int lane) { return lane >> 2; }
    __device__ static int col(int lane) { return (lane & 3) * 2; }
};

}

namespace mma_impl {

/// The single instruction: operands are this lane's fragments, already
/// gathered.
struct Atom {
    template <class TA, class TB, class TC>
    __device__ void operator()(TA const &a, TB const &b, TC &c) const {
        using CuteAtom = cute::SM80_16x8x16_F32BF16BF16F32_TN;
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
template <class T, class = void> struct is_tile : std::false_type {};
template <class T>
struct is_tile<T, std::void_t<typename T::shard_layout_type>> {
    using L = typename cute::remove_cvref_t<T>::shard_layout_type::layout;
    static constexpr bool value =
        cute::is_static<L>::value && decltype(cute::rank(L{}))::value == 2;
};

/// How many threads the accumulator's mesh spreads the tile over.
template <class TC> constexpr int acc_threads() {
    return int(
        cute::remove_cvref_t<TC>::shard_layout_type::mesh::topology::size);
}

/// ``acc += a @ b`` over a whole tile, the atom looped by the tile's shape.
///
/// Warps split N; every lane then walks its own atoms. Nothing here reads a
/// transpose flag: ``b`` is logically ``(N, K)`` and whether the buffer behind
/// it is k-major or n-major is a stride in its layout, which the indexing picks
/// up for free.
struct Tile {
    template <class TA, class TB, class TC>
    __device__ void operator()(TA const &a, TB const &b, TC &c) const {
        using Geo = mma_detail::Sm80_16x8x16;
        auto av = detail::to_local(a);
        auto bv = detail::to_local(b);
        auto &&cv = detail::to_local(c);
        using a_elem = cute::remove_cvref_t<decltype(av(0, 0))>;

        constexpr int threads = acc_threads<TC>();
        constexpr int warps = threads / 32;
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

        const int tid = int(threadIdx.x);
        const int lane = tid & 31;
        const int warp = (tid >> 5) % warps;
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
inline constexpr bool tile_shaped_v = is_tile<TA>::value && is_tile<TB>::value;

}
