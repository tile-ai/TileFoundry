/// CUDA copy op public entries. Included in-context from runtime.cuh inside
/// namespace tilefoundry::ops.
#pragma once

#include "copy/copy_impl.h"

/// Rebuild a CuTe tensor view from the typed pointer carried by TIR.
template <class TPointer, class TLayout>
CUTE_HOST_DEVICE auto tensor_view(TPointer pointer, TLayout layout) {
    return cute::make_tensor(pointer, layout);
}

/// Copy one projected slice to another.
template <class TSrc, class TDst>
__device__ void copy(TSrc const &src, TDst &dst) {
    copy_impl::Copy{}(src, dst);
}

template <class TSrc, class TDst>
__device__ void copy_async(TSrc const &src, TDst &dst) {
    copy_impl::CopyAsync{}(src, dst);
}
