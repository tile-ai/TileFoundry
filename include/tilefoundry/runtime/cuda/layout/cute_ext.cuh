/// What CuTe leaves out. Included in-context from runtime.cuh inside
/// namespace tilefoundry.
#pragma once

/// A layout with its modes in the other order; ``cute::reverse`` takes a
/// tuple and a layout is two of them.
///
/// A mesh is row-major, last axis fastest, and CuTe's algebra reads mode zero
/// as the fastest, so every CuTe function meaning "next to" reads a mesh
/// backwards: ``coalesce`` leaves ``(2,32):(32,1)``, 64 consecutive threads,
/// unfolded. Reverse first and CuTe is right again. Only the top-level modes
/// turn, so a nested layout is flattened first.
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto reverse(cute::Layout<Shape, Stride> const &l) {
    return cute::make_layout(cute::reverse(cute::shape(l)),
                             cute::reverse(cute::stride(l)));
}
