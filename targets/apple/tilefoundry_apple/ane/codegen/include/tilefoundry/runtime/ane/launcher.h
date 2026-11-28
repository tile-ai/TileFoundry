#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

/**
 * @brief A feature map, its complete padded storage reservation and the host
 * tensor of `ndim` axes that holds the same elements in row-major order.
 */
struct TensorDesc {
    const char *dtype;
    int64_t shape[4];
    int64_t strides[4];
    size_t storage_bytes;
    int ndim;
    int64_t host_shape[4];
};

/** @brief A named, generated coefficient writer. */
struct AneWeight {
    const char *name;
    int (*write)(uint8_t *, const DLTensor *);
};

/** @brief Immutable compiled entry data shared by independent instances. */
struct AneEntry {
    const uint8_t *image;
    size_t image_len;
    size_t coefficient_offset;
    const TensorDesc *io;
    int n_inputs;
    int n_outputs;
    const AneWeight *weights;
    int n_weights;
};

struct AneProgram;

/** @brief Status returned across the C ABI. */
enum AneStatus {
    ANE_OK = 0,
    ANE_UNAVAILABLE = 1,
    ANE_NOT_PROVISIONED = 2,
    ANE_LOAD_FAILED = 3,
    ANE_BAD_ARGUMENT = 4,
    ANE_EVALUATE_FAILED = 5,
    ANE_UNLOAD_FAILED = 6,
};

extern "C" {
/** @brief Validate a host tensor of rank ndim and its offset arithmetic. */
bool ane_check_tensor(const DLTensor *tensor, const int64_t *shape, int ndim,
                      int bits, bool contiguous);
/** @brief Copy the compiled skeleton into a new independent instance. */
int ane_create(const AneEntry *entry, AneProgram **program);
/** @brief Dispatch a named writer into this instance's coefficient segment. */
int ane_set_weight(AneProgram *program, const char *name,
                   const DLTensor *tensor);
/** @brief Hash the complete image and return its daemon cache path. */
int ane_cache_path(const AneProgram *program, char *path, size_t capacity);
/** @brief Borrow the instance image until mutation or destruction. */
int ane_image(const AneProgram *program, const uint8_t **bytes, size_t *length);
/** @brief Load a provisioned model and retain it for subsequent evaluations. */
int ane_load(AneProgram *program);
/** @brief Evaluate the retained model over validated host tensors. */
int ane_launch(AneProgram *program, DLTensor **inputs, int n_inputs,
               DLTensor **outputs, int n_outputs);
/** @brief Unload the model, free the instance and clear the caller's handle. */
int ane_destroy(AneProgram **program);
}

/** @brief Raise an FFI runtime error for a failed native operation. */
void ane_check_status(int status, const char *operation);
/** @brief Create a library-owned instance and return its non-reused token. */
int64_t ane_token_create(const AneEntry *entry);
/** @brief Write one compiled weight into the instance selected by its token. */
void ane_token_set_weight(int64_t token, tvm::ffi::String name,
                          DLTensor *tensor);
/** @brief Load the instance selected by a valid library-local token. */
int ane_token_load(int64_t token);
/** @brief Copy the complete image owned by a valid library-local token. */
tvm::ffi::Bytes ane_token_image(int64_t token);
/** @brief Return the daemon cache path for a valid library-local token. */
tvm::ffi::String ane_token_cache_path(int64_t token);
/** @brief Evaluate one instance while retaining its ownership lock. */
void ane_token_launch(int64_t token, DLTensor **inputs, int n_inputs,
                      DLTensor **outputs, int n_outputs);
/** @brief Release a valid token; zero denotes an empty instance. */
void ane_token_destroy(int64_t token);
/** @brief Register one entry, its weight names and its instance lifecycle. */
void ane_register(const char *name, const AneEntry *entry,
                  tvm::ffi::Function launch);
