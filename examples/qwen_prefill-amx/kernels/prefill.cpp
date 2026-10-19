/** TVM FFI entry for the supplied CPU/AMX model runtime. */
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/string.h>
#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "shared_fp16.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <cstring>

extern "C" void tf_amx_load_prepared(const char *, const char *, int, int);
extern "C" void tf_amx_clear_prepared();
using tvm::ffi::String;
using tvm::ffi::TensorView;
static std::mutex mutex;
struct Session {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    ggml_threadpool_t workers = nullptr;
    int capacity = 0;
    ~Session() {
        if (ctx) {
            llama_detach_threadpool(ctx);
            llama_free(ctx);
        }
        if (workers)
            ggml_threadpool_free(workers);
        if (model)
            llama_model_free(model);
    }
};
static std::unique_ptr<Session> session;
static std::string process_checkpoint;
static int64_t process_threads = 0;
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
int64_t load_model(String path, int64_t threads, int64_t capacity,
                   int64_t mode) {
    std::lock_guard<std::mutex> guard(mutex);
    if (session)
        throw std::runtime_error(
            "Close the loaded model before loading another");
    if (threads < 1 || threads > 12 || capacity < 128 || capacity > 8192 ||
        (mode != 0 && mode != 7))
        throw std::invalid_argument("Invalid model options");
    std::string filename(path), t = std::to_string(threads),
                                m = std::to_string(mode);
    if (!process_checkpoint.empty() &&
        (filename != process_checkpoint || threads != process_threads))
        throw std::runtime_error(
            "Use a separate process for another checkpoint/worker count");
    setenv("TF_AMX_THREADS", t.c_str(), 1);
    setenv("TF_AMX_MODE", m.c_str(), 1);
    setenv("TF_AMX_CHECKPOINT", filename.c_str(), 1);
    unsetenv("TF_SHARED_RESOURCE");
    ggml_backend_load_all();
    llama_backend_init();
    auto value = std::make_unique<Session>();
    value->capacity = capacity;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = true;
    value->model = llama_model_load_from_file(filename.c_str(), mp);
    if (!value->model)
        throw std::runtime_error("Cannot load GGUF");
    auto cp = llama_context_default_params();
    cp.n_ctx = capacity;
    cp.n_batch = capacity;
    cp.n_ubatch = capacity;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    value->ctx = llama_init_from_model(value->model, cp);
    if (!value->ctx)
        throw std::runtime_error("Cannot create model context");
    auto tp = ggml_threadpool_params_default(threads);
    tp.poll = 0;
    value->workers = ggml_threadpool_new(&tp);
    if (!value->workers)
        throw std::runtime_error("Cannot create worker pool");
    llama_attach_threadpool(value->ctx, value->workers, value->workers);
    int64_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(value->model));
    session = std::move(value);
    process_checkpoint = filename;
    process_threads = threads;
    return vocab;
}
void load_weight(String name, String path, int64_t n, int64_t k) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!session)
        throw std::runtime_error("Load the model first");
    if (n <= 0 || k <= 0 || n % 64 || k % 32 || n > INT32_MAX ||
        k > INT32_MAX || n * k > INT32_MAX)
        throw std::invalid_argument("Invalid weight dimensions");
    std::string a(name), b(path);
    tf_amx_load_prepared(a.c_str(), b.c_str(), n, k);
}
void close_model() {
    std::lock_guard<std::mutex> guard(mutex);
    session.reset();
    tf_amx_clear_prepared();
}
void prefill(TensorView tokens, TensorView logits) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!session)
        throw std::runtime_error("Model not loaded");
    tensor(tokens, 1, kDLInt, 32);
    tensor(logits, 1, kDLFloat, 32);
    int64_t length = tokens.size(0);
    if (length < 1 || length > session->capacity)
        throw std::invalid_argument("Token length out of bounds");
    int n = length,
        v = llama_vocab_n_tokens(llama_model_get_vocab(session->model));
    if (n < 1 || n > session->capacity || logits.size(0) != v)
        throw std::invalid_argument("Token/logit shape out of bounds");
    auto *ids = (int32_t *)ptr(tokens);
    for (int i = 0; i < n; i++)
        if (ids[i] < 0 || ids[i] >= v)
            throw std::invalid_argument("Invalid token ID");
    llama_memory_clear(llama_get_memory(session->ctx), true);
    auto *batch = llama_batch_ext_init(session->ctx);
    for (int i = 0; i < n; i++) {
        int index = llama_batch_ext_add_token(batch, 0, ids[i]);
        llama_pos p = i;
        llama_batch_ext_set_pos(batch, index, &p);
    }
    llama_batch_ext_set_output_logits(batch, n - 1, true);
    int status = llama_process(session->ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_batch_ext_free(batch);
    if (status)
        throw std::runtime_error("Prefill failed: " + std::to_string(status));
    std::memcpy(ptr(logits), llama_get_logits_ith(session->ctx, -1),
                v * sizeof(float));
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
int64_t token_count(String text) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!session)
        throw std::runtime_error("Model not loaded");
    std::string str(text);
    if (str.size() > INT32_MAX)
        throw std::invalid_argument("Text too long");
    return -llama_tokenize(llama_model_get_vocab(session->model), str.data(),
                           str.size(), nullptr, 0, true, true);
}
void tokenize(String text, TensorView out) {
    std::lock_guard<std::mutex> guard(mutex);
    if (!session)
        throw std::runtime_error("Model not loaded");
    tensor(out, 1, kDLInt, 32);
    std::string str(text);
    if (str.size() > INT32_MAX || out.size(0) > INT32_MAX)
        throw std::invalid_argument("Text too long");
    int actual = llama_tokenize(llama_model_get_vocab(session->model),
                                str.data(), str.size(), (llama_token *)ptr(out),
                                out.size(0), true, true);
    if (actual != out.size(0))
        throw std::invalid_argument("Token output length mismatch");
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(token_count, token_count);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tokenize, tokenize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(load_model, load_model);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(load_weight, load_weight);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(close_model, close_model);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prefill, prefill);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(linear, linear);
