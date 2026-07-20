// tilefoundry runtime — thin wrapper around CuTe (spec 010 §5 / §6).
//
// Provides our own `tilefoundry::Mesh` / `tilefoundry::TopologyScope` /
// `tilefoundry::ShardLayout` / `tilefoundry::ShardTensor` template surface.
// Re-exports CuTe primitives (`cute::copy`, `cute::make_tensor`, etc.)
// so codegen can emit real CuTe calls.

#pragma once

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_pipeline.h> // cp.async (__pipeline_memcpy_async)
#include <type_traits>
#include <utility>

namespace tilefoundry {

enum class TopologyScope {
    cta,
    warp,
    thread,
    scope_count, // sentinel; not a real topology level
};

// program_id<T>(): backend-tag-dispatched runtime query for the linearized
// scalar id of the current execution instance within topology `T`.
template <TopologyScope T> CUTE_HOST_DEVICE size_t program_id() noexcept;

template <> CUTE_HOST_DEVICE size_t program_id<TopologyScope::cta>() noexcept {
#if defined(__CUDA_ARCH__)
    return size_t(blockIdx.x) + size_t(blockIdx.y) * size_t(gridDim.x) +
           size_t(blockIdx.z) * size_t(gridDim.x) * size_t(gridDim.y);
#else
    return 0;
#endif
}

template <>
CUTE_HOST_DEVICE size_t program_id<TopologyScope::thread>() noexcept {
#if defined(__CUDA_ARCH__)
    return size_t(threadIdx.x) + size_t(threadIdx.y) * size_t(blockDim.x) +
           size_t(threadIdx.z) * size_t(blockDim.x) * size_t(blockDim.y);
#else
    return 0;
#endif
}

// No program_id<TopologyScope::warp> specialization: codegen only admits
// ("cta", "thread") program topology levels (target/cuda/target.py
// topology_levels()), so this is never instantiated. The TopologyScope::warp
// enumerator itself stays (spec runtime.md §2.1's fixed enumeration); a real
// warp-scoped program_id (thread id >> 5) belongs with whichever milestone
// wires warp-scoped meshes, together with a test.

template <TopologyScope T> constexpr auto program_shape() noexcept;

template <TopologyScope T> constexpr auto program_dim() noexcept {
    return cute::size(program_shape<T>());
}

#include "layout/shard_layout.cuh"
#include "tensor_view/shard_tensor.cuh"
#include "tensor_view/shard_copy.cuh"

namespace ops {

#include "tensor_view/ops_detail.cuh"
#include "ops/sync.cuh"
#include "ops/fill.cuh"
#include "ops/unary.cuh"
#include "ops/copy.cuh"
#include "ops/binary.cuh"
#include "ops/reduce.cuh"
#include "ops/cast.cuh"
#include "ops/clamp.cuh"
#include "ops/rmsnorm.cuh"
#include "ops/mma.cuh"

} // namespace ops

} // namespace tilefoundry
