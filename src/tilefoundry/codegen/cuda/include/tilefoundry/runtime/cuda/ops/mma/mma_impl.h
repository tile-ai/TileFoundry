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

/// Four 8-bit values as one register, the first in the low byte.
__device__ inline uint32_t pack_e4m3x4(uint8_t b0, uint8_t b1, uint8_t b2,
                                       uint8_t b3) {
    return uint32_t(b0) | (uint32_t(b1) << 8) | (uint32_t(b2) << 16) |
           (uint32_t(b3) << 24);
}

template <class T> __device__ uint8_t as_u8(T const &x) {
    uint8_t out;
    __builtin_memcpy(&out, &x, sizeof(uint8_t));
    return out;
}

/// The instruction an A element type selects, and the values each lane holds
/// of A, B and C. An element type with no instruction is not ``known``.
template <class T> struct AtomFor {
    static constexpr bool known = false;
};

template <> struct AtomFor<__nv_bfloat16> {
    static constexpr bool known = true;
    using Op = cute::SM80_16x8x16_F32BF16BF16F32_TN;
    static constexpr int a = 8, b = 4, c = 4;
};

template <> struct AtomFor<__nv_fp8_e4m3> {
    static constexpr bool known = true;
    using Op = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
    static constexpr int a = 16, b = 8, c = 4;
};

/// The element type an operand's lane fragment holds.
template <class T>
using element_t = typename tilefoundry::local_view_t<T>::value_type;

}

namespace mma_impl {

/// Execute the atom on this lane's gathered fragments.
struct Atom {
    template <class TA, class TB, class TC>
    __device__ void operator()(TA const &a, TB const &b, TC &c) const {
        using namespace mma_detail;
        using CuteAtom = typename AtomFor<element_t<TA>>::Op;

        auto a_data = a.data();
        auto b_data = b.data();
        auto c_data = c.data();

        uint32_t a0, a1, a2, a3, b0, b1;
        if constexpr (cute::is_same_v<element_t<TA>, __nv_fp8_e4m3>) {
            a0 = pack_e4m3x4(as_u8(a_data[0]), as_u8(a_data[1]),
                             as_u8(a_data[2]), as_u8(a_data[3]));
            a1 = pack_e4m3x4(as_u8(a_data[8]), as_u8(a_data[9]),
                             as_u8(a_data[10]), as_u8(a_data[11]));
            a2 = pack_e4m3x4(as_u8(a_data[4]), as_u8(a_data[5]),
                             as_u8(a_data[6]), as_u8(a_data[7]));
            a3 = pack_e4m3x4(as_u8(a_data[12]), as_u8(a_data[13]),
                             as_u8(a_data[14]), as_u8(a_data[15]));

            b0 = pack_e4m3x4(as_u8(b_data[0]), as_u8(b_data[1]),
                             as_u8(b_data[2]), as_u8(b_data[3]));
            b1 = pack_e4m3x4(as_u8(b_data[4]), as_u8(b_data[5]),
                             as_u8(b_data[6]), as_u8(b_data[7]));
        } else {
            a0 = pack_bf16x2(as_u16(a_data[0]), as_u16(a_data[1]));
            a1 = pack_bf16x2(as_u16(a_data[4]), as_u16(a_data[5]));
            a2 = pack_bf16x2(as_u16(a_data[2]), as_u16(a_data[3]));
            a3 = pack_bf16x2(as_u16(a_data[6]), as_u16(a_data[7]));

            b0 = pack_bf16x2(as_u16(b_data[0]), as_u16(b_data[1]));
            b1 = pack_bf16x2(as_u16(b_data[2]), as_u16(b_data[3]));
        }

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

/// Recognize the atom's per-lane fragment shapes.
template <class T, int N>
inline constexpr bool frag_v = [] {
    using L = typename cute::remove_cvref_t<T>::layout_type;
    if constexpr (!cute::is_static<L>::value)
        return false;
    else
        return int(decltype(cute::size(L{}))::value) == N;
}();

/// The operands of the atom the A element type selects, stated positively: B
/// holds the same element type as A, C holds f32, and each fragment has the
/// atom's length.
template <class TA, class TB, class TC>
inline constexpr bool atom_shaped_v = [] {
    using Element = mma_detail::element_t<TA>;
    using For = mma_detail::AtomFor<Element>;
    if constexpr (!For::known)
        return false;
    else
        return cute::is_same_v<mma_detail::element_t<TB>, Element> &&
               cute::is_same_v<mma_detail::element_t<TC>, float> &&
               frag_v<tilefoundry::local_view_t<TA>, For::a> &&
               frag_v<tilefoundry::local_view_t<TB>, For::b> &&
               frag_v<tilefoundry::local_view_t<TC>, For::c>;
}();

}
