#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlpack/dlpack.h>

/** @brief A feature map and its complete padded storage reservation. */
struct TensorDesc {
    const char *dtype;
    int64_t shape[4];
    int64_t strides[4];
    size_t storage_bytes;
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
/** @brief Validate a host tensor and its offset arithmetic before copying. */
bool ane_check_tensor(const DLTensor *tensor, const int64_t *shape, int bits,
                      bool contiguous);
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
