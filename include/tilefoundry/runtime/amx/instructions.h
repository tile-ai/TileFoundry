#pragma once
#include <cstdint>

namespace tilefoundry::amx {
/** Enable the current thread's Apple AMX state. */
inline void start() {
    asm volatile("nop\nnop\nnop\n.word 0x201220" ::: "memory");
}
/** Release the current thread's Apple AMX state. */
inline void stop() {
    asm volatile("nop\nnop\nnop\n.word 0x201221" ::: "memory");
}
/** Load one 64-byte vector into X register zero. */
inline void load_x(const void *ptr) {
    auto operand = reinterpret_cast<uint64_t>(ptr) & ((1ULL << 57) - 1);
    asm volatile(".word (0x201000 + 0%0 - ((0%0 >> 4) * 6))" ::"r"(operand)
                 : "memory");
}
/** Load one 64-byte vector into Y register zero. */
inline void load_y(const void *ptr) {
    auto operand = reinterpret_cast<uint64_t>(ptr) & ((1ULL << 57) - 1);
    asm volatile(".word (0x201020 + 0%0 - ((0%0 >> 4) * 6))" ::"r"(operand)
                 : "memory");
}
/** Accumulate the FP16 outer product into FP32 Z, optionally clearing Z first.
 */
inline void fma16(bool clear = false) {
    uint64_t operand = (1ULL << 62) | (uint64_t(clear) << 27);
    asm volatile(".word (0x2011e0 + 0%0 - ((0%0 >> 4) * 6))" ::"r"(operand)
                 : "memory");
}
/** Store one physical 64-byte row of Z. */
inline void store_z(void *ptr, uint64_t row) {
    auto operand =
        (reinterpret_cast<uint64_t>(ptr) & ((1ULL << 57) - 1)) | (row << 56);
    asm volatile(".word (0x2010a0 + 0%0 - ((0%0 >> 4) * 6))" ::"r"(operand)
                 : "memory");
}
}
