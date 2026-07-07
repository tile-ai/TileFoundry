// ops::detail tensor-view projection helpers. Included in-context from
// runtime.cuh inside namespace tilefoundry::ops.
#pragma once

namespace detail {

template <class T> struct is_shard_tensor : std::false_type {};
template <class E, class GL, class SL>
struct is_shard_tensor<ShardTensor<E, GL, SL>> : std::true_type {};

template <class T> CUTE_HOST_DEVICE decltype(auto) to_local(T &&t) {
    if constexpr (is_shard_tensor<cute::remove_cvref_t<T>>::value) {
        return local(t);
    } else {
        return std::forward<T>(t);
    }
}

} // namespace detail
