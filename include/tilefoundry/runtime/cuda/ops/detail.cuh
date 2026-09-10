/// The names ``ops::detail`` answers to. They are defined in
/// ``tilefoundry::detail``; this states which of them an op may reach for.
#pragma once

namespace detail {
using tilefoundry::detail::id_axes;
using tilefoundry::detail::is_partial_attr_v;
using tilefoundry::detail::is_shard_tensor;
using tilefoundry::detail::is_split_attr_v;
using tilefoundry::detail::local_tensor;
using tilefoundry::detail::local_view_t;
using tilefoundry::detail::positions_of;
using tilefoundry::detail::shard_attrs_match_mesh;
using tilefoundry::detail::shard_layout_is_full_broadcast;
}
