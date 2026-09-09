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
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <cuda_pipeline.h>
#include <type_traits>
#include <utility>

namespace tilefoundry {

/// The warp is the hardware's. It is the one number in this runtime that is a
/// constant rather than something an operand's layout answers, which is why it
/// has a name: a bare 32 beside a mesh extent reads like another extent.
inline constexpr int kWarpSize = 32;

/// A ``false`` that only the compiler's instantiation of a template can see.
///
/// Every ``if constexpr`` tier chain below ends in an ``else`` that refuses,
/// rather than in one that stands for whichever case nobody thought about --
/// that trailing ``else`` is how a mesh with no tier came to take a
/// neighbouring one and answer with the wrong number. A plain
/// ``static_assert(false)`` cannot write it: in a discarded branch it is
/// ill-formed before C++23, so the refusal has to depend on a parameter of the
/// template it sits in.
template <class...> inline constexpr bool dependent_false_v = false;

/// The fixed enumeration of [runtime
/// §2.1](docs/spec/runtime.md#21-topologyscope). Two of the four are levels
/// this backend can project; ``warp`` and the ``scope_count`` sentinel are not,
/// which is why a dispatch on scope is exhaustive rather than trailing an
/// ``else`` that stands for ``thread``.
enum class TopologyScope {
    cta,
    warp,
    thread,
    scope_count,
};

/// program_shape<T>(): the shape the launch gives topology level ``T`` -- the
/// grid dimensions for ``cta``, the block's for ``thread``. Each module
/// specialises it; nothing here defines it.
template <TopologyScope T>
CUTE_HOST_DEVICE constexpr auto program_shape() noexcept;

template <TopologyScope T>
CUTE_HOST_DEVICE constexpr auto program_dim() noexcept {
    return cute::size(program_shape<T>());
}

/// program_id<T>(): the linearized scalar id of this execution instance within
/// topology level ``T``. The runtime's only spelling of "which instance am I".
///
/// One template and not a specialization per level, so the dispatch on scope is
/// exhaustive. The linearisation is written out even for a flat block: the
/// shorter ``threadIdx.x`` is right only where ``program_shape<T>()`` says the
/// leading dimension is the whole level, and asking it here would make every
/// module including this header state a thread level whether it has one.
template <TopologyScope T> CUTE_HOST_DEVICE size_t program_id() noexcept {
    static_assert(T == TopologyScope::cta || T == TopologyScope::thread,
                  "program_id: only the cta and thread levels have an id -- "
                  "the warp level has none and scope_count is a sentinel, so a "
                  "warp-sized grouping belongs as an axis of a thread mesh's "
                  "layout");
#if defined(__CUDA_ARCH__)
    if constexpr (T == TopologyScope::cta) {
        return size_t(blockIdx.x) + size_t(blockIdx.y) * size_t(gridDim.x) +
               size_t(blockIdx.z) * size_t(gridDim.x) * size_t(gridDim.y);
    } else {
        /// Both products are dead on a flat block, and the three
        /// special-register reads are uniform, so ptxas hoists them once for
        /// the kernel: what a call site bought by spelling ``threadIdx.x``
        /// itself was not paid per use.
        return size_t(threadIdx.x) + size_t(threadIdx.y) * size_t(blockDim.x) +
               size_t(threadIdx.z) * size_t(blockDim.x) * size_t(blockDim.y);
    }
#else
    return 0;
#endif
}

#include "layout/shard_layout.cuh"
#include "layout/mesh_warp_view.cuh"
#include "tensor_view/shard_tensor.cuh"
#include "utility/warp.cuh"

namespace ops {

#include "tensor_view/ops_detail.cuh"
/// Primitive callables must precede the elementwise, reduce and dot entries
/// that instantiate or specialise on them.
#include "ops/primitive/unary.h"
#include "ops/primitive/binary.h"
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
