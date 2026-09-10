/// tilefoundry runtime — thin wrapper around CuTe.
///
/// Provides our own `tilefoundry::Mesh` / `tilefoundry::TopologyScope` /
/// `tilefoundry::ShardLayout` / `tilefoundry::ShardTensor` template surface.
/// Re-exports CuTe primitives (`cute::copy`, `cute::make_tensor`, etc.)
/// so codegen can emit real CuTe calls.

#pragma once

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cstdint>
#include <cassert>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <cuda_pipeline.h>
#include <type_traits>
#include <utility>

namespace tilefoundry {

/// Hardware warp width.
inline constexpr int kWarpSize = 32;

/// A ``false`` that only the compiler's instantiation of a template can see.
template <class...> inline constexpr bool dependent_false_v = false;

/// Program topology levels and sentinel.
enum class TopologyScope {
    gpu,
    cta,
    thread,
    scope_count,
};

/// How many instances the launch gives level T. One .cu is one launch, so
/// each states its own; a launch-provided grid states it at run time.
template <TopologyScope T>
CUTE_HOST_DEVICE constexpr auto program_dim() noexcept;

namespace detail {
/// Named levels as one shape. Each count is a dependent call, so the counts
/// are read where the shape is asked for and not where it is written -- a
/// translation unit states them after this header.
template <TopologyScope... Ts>
CUTE_HOST_DEVICE constexpr auto dims_of() noexcept {
    return cute::make_shape(program_dim<Ts>()...);
}
}

/// Level T and every level under it, one mode each. A mesh naming several
/// levels is indexed against this shape: an axis belongs to the level whose
/// extents its own multiply up to, and a stride counts the levels below it.
template <TopologyScope T>
CUTE_HOST_DEVICE constexpr auto program_shape() noexcept {
    if constexpr (T == TopologyScope::gpu)
        return detail::dims_of<T, TopologyScope::cta, TopologyScope::thread>();
    else if constexpr (T == TopologyScope::cta)
        return detail::dims_of<T, TopologyScope::thread>();
    else
        return detail::dims_of<T>();
}

/**
 * @brief What a launch is told about the program it belongs to.
 *
 * A card has no register naming which of the mesh's cards it is, so the host
 * that placed it says so. Every level whose position the device reads for
 * itself leaves its entry unused.
 */
struct ProgramMetaData {
    int program_id[int(TopologyScope::scope_count)];
};

/**
 * @brief The block's one copy of what the launch was told about this program.
 *
 * Shared rather than global: the name is an offset into whichever block is
 * running, so one name is one copy per block rather than one per device.
 */
CUTE_HOST_DEVICE ProgramMetaData &program_meta() {
#if defined(__CUDA_ARCH__)
    __shared__ ProgramMetaData held;
    return held;
#else
    static ProgramMetaData held{};
    return held;
#endif
}

/**
 * @brief Hand the block what this launch was told.
 *
 * Called once, before any divergence: one thread writes and the barrier
 * orders that write against every read of it.
 */
CUTE_HOST_DEVICE void program_meta(ProgramMetaData const &meta) {
#if defined(__CUDA_ARCH__)
    if (threadIdx.x == 0)
        program_meta() = meta;
    __syncthreads();
#else
    program_meta() = meta;
#endif
}

/// Linearized id within topology level T.
template <TopologyScope T> CUTE_HOST_DEVICE size_t program_id() noexcept {
    static_assert(T == TopologyScope::gpu || T == TopologyScope::cta ||
                      T == TopologyScope::thread,
                  "program_id: only gpu, cta and thread have an id");
#if defined(__CUDA_ARCH__)
    if constexpr (T == TopologyScope::gpu) {
        return size_t(program_meta().program_id[int(TopologyScope::gpu)]);
    } else if constexpr (T == TopologyScope::cta) {
        return size_t(blockIdx.x) + size_t(blockIdx.y) * size_t(gridDim.x) +
               size_t(blockIdx.z) * size_t(gridDim.x) * size_t(gridDim.y);
    } else {
        return size_t(threadIdx.x) + size_t(threadIdx.y) * size_t(blockDim.x) +
               size_t(threadIdx.z) * size_t(blockDim.x) * size_t(blockDim.y);
    }
#else
    return 0;
#endif
}

namespace detail {
/// Every scope's id in enum order, so the tuple is as long as there are
/// scopes rather than as long as somebody remembered to list.
template <size_t... Is>
CUTE_HOST_DEVICE auto ids_of(std::index_sequence<Is...>) noexcept {
    return cute::make_tuple(program_id<TopologyScope(Is)>()...);
}
}

/// Every level's id, indexed by TopologyScope.
CUTE_HOST_DEVICE auto program_ids() noexcept {
    return detail::ids_of(
        std::make_index_sequence<size_t(TopologyScope::scope_count)>{});
}

#include "layout/cute_ext.cuh"
#include "layout/mesh.cuh"
#include "layout/shard_layout.cuh"
#include "tensor_view/shard_tensor.cuh"
#include "utility/warp.cuh"

namespace ops {

#include "ops/detail.cuh"
/// Primitive callables precede entries that instantiate them.
#include "primitive/unary.h"
#include "primitive/binary.h"
#include "ops/sync.cuh"
#include "ops/elementwise.cuh"
#include "ops/copy.cuh"

#include "ops/tma.cuh"
#include "ops/reduce.cuh"
/// dot after reduce: it reuses reduce's no-workspace tag.
#include "ops/dot.cuh"
#include "ops/rmsnorm.cuh"
#include "ops/mma.cuh"

}
}
