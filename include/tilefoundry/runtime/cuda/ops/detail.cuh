/// The names ``ops::detail`` answers to. They are defined in
/// ``tilefoundry::detail``; this states which of them an op may reach for.
/// What every op must go through is not here -- ``local_tensor``,
/// ``local_view_t`` and ``ShardTensorLike`` are ``tilefoundry::`` names.
#pragma once

namespace detail {
using tilefoundry::detail::id_axes;
using tilefoundry::detail::is_partial_attr_v;
using tilefoundry::detail::is_shard_tensor;
using tilefoundry::detail::is_split_attr_v;
using tilefoundry::detail::positions_of;
using tilefoundry::detail::shard_attrs_match_mesh;
using tilefoundry::detail::shard_layout_is_full_broadcast;
}
