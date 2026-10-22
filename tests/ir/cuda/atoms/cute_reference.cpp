#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/copy_traits_sm90.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/mma_traits_sm89.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/tensor.hpp>

#include <iostream>
#include <string>
#include <utility>

using namespace cute;
namespace G = cute::SM90::GMMA;

template <class T> void json_tuple(T const &value) {
    if constexpr (is_tuple<T>::value) {
        std::cout << '[';
        for_each(make_seq<tuple_size<T>::value>{}, [&](auto i) {
            if constexpr (i > 0)
                std::cout << ',';
            json_tuple(get<i>(value));
        });
        std::cout << ']';
    } else {
        std::cout << int(value);
    }
}

template <class L> void json_layout(L layout) {
    std::cout << "{\"shape\":";
    json_tuple(layout.shape());
    std::cout << ",\"strides\":";
    json_tuple(layout.stride());
    std::cout << '}';
}

template <class Op> void emit_tv() {
    using Tr = MMA_Traits<Op>;
    std::cout << "{\"A\":";
    json_layout(typename Tr::ALayout{});
    std::cout << ",\"B\":";
    json_layout(typename Tr::BLayout{});
    std::cout << ",\"C\":";
    json_layout(typename Tr::CLayout{});
    std::cout << '}';
}

template <class T, int N> void emit_wgmma(std::string const &form) {
    using Tile = Shape<_64, Int<N>, Int<32 / sizeof(T)>>;
    if (form == "SS") {
        emit_tv<decltype(G::ss_op_selector<T, T, float, Tile>())>();
    } else {
        emit_tv<decltype(G::rs_op_selector<T, T, float, Tile>())>();
    }
}

template <class T, int... I>
void select_n(int n, std::string const &form,
              std::integer_sequence<int, I...>) {
    ((n == 8 * (I + 1) ? (emit_wgmma<T, 8 * (I + 1)>(form), 0) : 0), ...);
}

template <class L> void emit_smem(L layout) {
    using S = get_swizzle_t<L>;
    std::cout << "{\"swizzle\":[" << S::num_bits << ',' << S::num_base << ','
              << S::num_shft << "],\"layout\":";
    json_layout(get_nonswizzle_portion(layout));
    std::cout << '}';
}

template <class T, G::Major Major> void select_swizzle(int swizzle) {
    switch (swizzle) {
    case 0:
        emit_smem(G::Layout_INTER_Atom<T, Major>{});
        break;
    case 1:
        emit_smem(G::Layout_SW32_Atom<T, Major>{});
        break;
    case 2:
        emit_smem(G::Layout_SW64_Atom<T, Major>{});
        break;
    case 3:
        emit_smem(G::Layout_SW128_Atom<T, Major>{});
        break;
    }
}

template <class T>
void dispatch(std::string const &kind, std::string const &form, int n,
              std::string const &major, int swizzle) {
    if (kind == "tv") {
        select_n<T>(n, form, std::make_integer_sequence<int, 32>{});
    } else if (major == "MN") {
        select_swizzle<T, G::Major::MN>(swizzle);
    } else {
        select_swizzle<T, G::Major::K>(swizzle);
    }
}

/** Emit host-only CuTe golden layouts, one JSON entry per requested input line.
 */
int main() {
    std::string key, kind, dtype, form, major;
    int n, swizzle;
    std::cout << '{';
    bool first = true;
    while (std::cin >> key >> kind >> dtype >> form >> n >> major >> swizzle) {
        if (!first)
            std::cout << ',';
        first = false;
        std::cout << '"' << key << "\":";
        if (kind == "mma" && dtype == "bf16") {
            emit_tv<SM80_16x8x16_F32BF16BF16F32_TN>();
        } else if (kind == "mma" && dtype == "fp8e4m3") {
            emit_tv<SM89_16x8x32_F32E4M3E4M3F32_TN>();
        } else if (kind == "stmatrix") {
            using Tr = Copy_Traits<SM90_U32x4_STSM_N>;
            std::cout << "{\"src\":";
            json_layout(typename Tr::SrcLayout{});
            std::cout << ",\"dst\":";
            json_layout(typename Tr::DstLayout{});
            std::cout << '}';
        } else if (dtype == "bf16") {
            dispatch<bfloat16_t>(kind, form, n, major, swizzle);
        } else if (dtype == "f16") {
            dispatch<half_t>(kind, form, n, major, swizzle);
        } else if (dtype == "fp8e4m3") {
            dispatch<float_e4m3_t>(kind, form, n, major, swizzle);
        } else {
            return 1;
        }
    }
    std::cout << "}\n";
}
