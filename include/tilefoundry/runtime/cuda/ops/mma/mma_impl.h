// CUDA MMA op implementation. Included in-context from ops/mma.cuh inside
// namespace tilefoundry::ops.
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

} // namespace mma_detail

namespace mma_impl {

struct MmaSm80_16x8x16Bf16 {
    template <class TA, class TB, class TC>
    __device__ void operator()(TA const &a, TB const &b, TC &c) const {
        using Atom = cute::SM80_16x8x16_F32BF16BF16F32_TN;
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

        Atom::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);

        c_data[0] = d0;
        c_data[1] = d1;
        c_data[2] = d2;
        c_data[3] = d3;
    }
};

} // namespace mma_impl
