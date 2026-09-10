/// The names ``ops::detail`` answers to. They are defined in
/// ``tilefoundry::detail``; this states which of them an op may reach for.
#pragma once

namespace detail {
using tilefoundry::detail::is_shard_tensor;
using tilefoundry::detail::local_view_t;
using tilefoundry::detail::to_local;
}
