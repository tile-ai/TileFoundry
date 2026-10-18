// Hand-written AMX projection runtime; independent of llama.cpp.
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>
#include "runtime.hpp"
#include <cstdint>
#include <mutex>
#include <stdexcept>
using tvm::ffi::TensorView;
static std::mutex mutex;
static void tensor(const TensorView &t, int ndim, int code, int bits) {
    auto dtype = t.dtype();
    if (t.device().device_type != kDLCPU || t.ndim() != ndim ||
        dtype.code != code || dtype.bits != bits || dtype.lanes != 1 ||
        !t.IsContiguous())
        throw std::invalid_argument("Wrong CPU tensor rank/dtype/strides");
}
static void *ptr(const TensorView &t) {
    return (char *)t.data_ptr() + t.byte_offset();
}
void linear(TensorView x, TensorView w, TensorView out) {
    std::lock_guard<std::mutex> guard(mutex);
    tensor(x, 2, kDLFloat, 32);
    tensor(w, 3, kDLFloat, 16);
    tensor(out, 2, kDLFloat, 32);
    if (w.size(0) <= 0 || w.size(0) > INT32_MAX / 64)
        throw std::invalid_argument("Invalid N panels");
    int64_t m = x.size(0), k = x.size(1), n = w.size(0) * w.size(2);
    if (m <= 0 || k <= 0 || m > INT32_MAX - 31 || k % 32 || w.size(1) != k ||
        (w.size(2) != 64 && w.size(2) != 32) || out.size(0) != m ||
        out.size(1) != n || n * k > INT32_MAX || m * n > INT32_MAX ||
        ((m + 31) / 32 * 32) * k > INT32_MAX)
        throw std::invalid_argument("Invalid linear shape");
    shared_fp16::Weight weight((const _Float16 *)ptr(w), n, k, w.size(2));
    shared_fp16::prefill((float *)ptr(x), m, weight, (float *)ptr(out),
                         w.size(2), true);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(linear, linear);
