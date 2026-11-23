#pragma once

#include <cstddef>
#include <cstdint>
#include <dlpack/dlpack.h>

/** @brief An input or output feature map in element units. */
struct IODesc {
    const char *name;
    const char *dtype;
    int64_t shape[4];
    int64_t strides[4];
};

/** @brief Invoke the image through the separately provided runtime
 * implementation. */
int ane_launch(const uint8_t *image, size_t image_len, const char *image_hash,
               const IODesc *io, int n_io, DLTensor **inputs,
               DLTensor **outputs);
