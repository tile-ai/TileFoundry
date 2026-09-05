/// warp op implementations — one per entry in ``ops/warp.cuh``.
///
/// Included in-context from ``ops/warp.cuh`` (see reduce_common_impl.h for the
/// in-context include contract). No tier selection lives here: each struct is
/// the single implementation of its entry.
#pragma once

namespace warp_impl {

/// Whether ``__shfl_xor_sync`` takes ``T`` directly. Everything else goes
/// through the word-wise path below.
template <class T>
inline constexpr bool is_shuffle_native_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
    std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long>;

/// ``__shfl_xor_sync`` for the types the intrinsic accepts, and a word-wise
/// exchange for anything else (a bf16 pair, a small aggregate).
template <class T> struct ShuffleXor {
    __device__ T operator()(T value, int lane_mask,
                            unsigned member_mask) const {
        if constexpr (is_shuffle_native_v<T>) {
            return __shfl_xor_sync(member_mask, value, lane_mask);
        } else {
            static_assert(
                sizeof(T) % sizeof(unsigned) == 0,
                "shuffle_xor: type size must be a multiple of 4 bytes");
            constexpr int kWords = int(sizeof(T) / sizeof(unsigned));
            T out = value;
            unsigned *dst = reinterpret_cast<unsigned *>(&out);
            for (int i = 0; i < kWords; ++i)
                dst[i] = __shfl_xor_sync(member_mask, dst[i], lane_mask);
            return out;
        }
    }
};

/// Elect exactly one thread among the leading ``Width`` threads of the CTA.
///
/// On sm_90 the elected lane comes from ``elect.sync``: one instruction, where
/// a ballot plus a find-first would be three and would make every lane read a
/// value it does not use. Below sm_90 lane 0 is the elected thread — the same
/// guarantee (exactly one) with a different choice of which.
///
/// Threads outside the first warp answer false without executing the
/// warp-scoped instruction at all: ``elect.sync``'s result is defined only for
/// the lanes named in its member mask.
template <int Width> struct Elect {
    __device__ bool operator()() const {
        static_assert(Width > 0, "shuffle_elect: Width must be positive");
        /// The elected thread is always in the first warp, so every thread past
        /// it answers false. ``Width`` therefore only discriminates below 32.
        constexpr unsigned kParticipants = Width < 32 ? unsigned(Width) : 32u;
        if (threadIdx.x >= kParticipants)
            return false;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        constexpr unsigned kMask =
            Width >= 32 ? 0xFFFFFFFFu : ((1u << unsigned(Width & 31)) - 1u);
        unsigned pred = 0u;
        asm volatile("{\n"
                     "  .reg .b32 elected_lane;\n"
                     "  .reg .pred is_elected;\n"
                     "  elect.sync elected_lane|is_elected, %1;\n"
                     "  selp.b32 %0, 1, 0, is_elected;\n"
                     "}\n"
                     : "=r"(pred)
                     : "n"(kMask));
        return pred != 0u;
#else
        return (threadIdx.x & 31u) == 0u;
#endif
    }
};

/// The five-step butterfly: after step ``k`` every lane holds the combination
/// of its ``2^(k+1)``-lane block, so after five steps every lane holds the
/// warp's. The result is in every lane, not only in lane 0.
template <class Combine, class T> struct Butterfly {
    __device__ T operator()(T value) const {
        for (int delta = 16; delta > 0; delta >>= 1)
            value = Combine::apply(value,
                                   ShuffleXor<T>{}(value, delta, 0xFFFFFFFFu));
        return value;
    }
};

}
