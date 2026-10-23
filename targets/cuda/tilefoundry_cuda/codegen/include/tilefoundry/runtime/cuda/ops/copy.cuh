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

template <class ExecutionMesh, int Bytes, class TSrc, class TDst, class TIndex>
__device__ void copy_async(TSrc const &src, TDst &dst, TIndex const &index) {
    copy_impl::CopyIndexedAsync<ExecutionMesh, false, Bytes>{}(src, dst, index,
                                                               0);
}

template <class ExecutionMesh, int Bytes, class TSrc, class TDst, class TIndex,
          class TFill>
__device__ void copy_async(TSrc const &src, TDst &dst, TIndex const &index,
                           TFill fill) {
    copy_impl::CopyIndexedAsync<ExecutionMesh, true, Bytes>{}(src, dst, index,
                                                              fill);
}
