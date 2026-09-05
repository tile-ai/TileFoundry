/// Static projection of a ShardLayout onto one mesh instance. Included
/// in-context from runtime.cuh inside namespace tilefoundry.
#pragma once

namespace detail {

/// Whether a shard attribute names a tensor axis (``S<n>``, not ``B``).
template <class A, class = void> struct attr_axis {
    static constexpr int value = -1;
};
template <class A> struct attr_axis<A, std::void_t<decltype(A::axis)>> {
    static constexpr int value = int(A::axis);
};

/// The mesh extent that divides tensor axis ``Axis``, or 1 when none does.
///
/// A fold rather than a loop: this has to be usable as a template argument.
template <class SL, int Axis, size_t... Is>
constexpr int mesh_divisor_over(std::index_sequence<Is...>) {
    using mesh_layout_t = typename SL::mesh::layout;
    using attrs_t = typename SL::attrs;
    int divisor = 1;
    ((attr_axis<cute::remove_cvref_t<decltype(cute::get<Is>(attrs_t{}))>>::
                  value == Axis
          ? (divisor = int(cute::get<Is>(cute::shape(mesh_layout_t{}))))
          : 0),
     ...);
    return divisor;
}

template <class SL, int Axis> constexpr int mesh_divisor() {
    return mesh_divisor_over<SL, Axis>(
        std::make_index_sequence<
            cute::tuple_size<typename SL::attrs>::value>{});
}

/// The layout of the slice ``local()`` hands one mesh instance, as a type.
///
/// ``local()`` divides each sharded axis by that axis's mesh extent and offsets
/// the pointer. Both the shard layout and the mesh layout are static, so the
/// slice's layout is too -- but the one ``local()`` returns is assembled from
/// runtime ints, and a compile-time choice (a vector width, a copy tier) cannot
/// be read off it. This recomputes the same projection in the type system.
///
/// A shard layout that is not static, or that broadcasts wholesale, projects to
/// itself: there is nothing to divide.
template <class SL, class = void> struct local_layout {
    using type = typename SL::layout;
};

template <class SL>
struct local_layout<
    SL, std::enable_if_t<cute::is_static<typename SL::layout>::value &&
                         !shard_layout_is_full_broadcast<SL>()>> {
    using layout_t = typename SL::layout;
    static constexpr int t_rank =
        int(cute::tuple_size<
            cute::remove_cvref_t<decltype(cute::shape(layout_t{}))>>::value);

    template <size_t... Is>
    static constexpr auto build(std::index_sequence<Is...>) {
        return cute::make_layout(
            cute::make_shape(
                cute::Int<int(cute::get<Is>(cute::shape(layout_t{}))) /
                          mesh_divisor<SL, int(Is)>()>{}...),
            cute::stride(layout_t{}));
    }

    using type = decltype(build(std::make_index_sequence<t_rank>{}));
};

/// The local layout of a ShardTensor; a plain CuTe tensor projects to its own.
template <class T, class = void> struct operand_layout {
    using type = typename cute::remove_cvref_t<T>::layout_type;
};
template <class T>
struct operand_layout<
    T, std::void_t<typename cute::remove_cvref_t<T>::shard_layout_type>> {
    using type = typename local_layout<
        typename cute::remove_cvref_t<T>::shard_layout_type>::type;
};

template <class T> using operand_layout_t = typename operand_layout<T>::type;

}
