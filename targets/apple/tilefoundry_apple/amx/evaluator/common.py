"""Helpers translated from Peter Cawley's AMX emulate.h (MIT)."""

U64 = (1 << 64) - 1
FMA_WIDEN_16_32 = 1 << 62
FMA_VECTOR_PRODUCT = 1 << 63
BF16_ONE = 0x3F80
BF16_NAN = 0x7FC0


def read_uint(buf, offset, nbytes):
    if offset < 0 or offset + nbytes > len(buf):
        raise IndexError("reference buffer read outside bounds")
    return int.from_bytes(buf[offset : offset + nbytes], "little")


def read_sint(buf, offset, nbytes):
    value = read_uint(buf, offset, nbytes)
    return value - (1 << (8 * nbytes)) if value >> (8 * nbytes - 1) else value


def write_uint(buf, offset, nbytes, value):
    if offset < 0 or offset + nbytes > len(buf):
        raise IndexError("reference buffer write outside bounds")
    buf[offset : offset + nbytes] = (value & ((1 << (8 * nbytes)) - 1)).to_bytes(nbytes, "little")


def load_int(buf, offset, nbytes, signext):
    return (
        read_sint(buf, offset, nbytes) if signext or nbytes == 8 else read_uint(buf, offset, nbytes)
    )


def load_xy_reg(pool, offset):
    offset &= 511
    return bytes(pool[(offset + i) & 511] for i in range(64))


def xy_shuffle(dst, shuffle, ebytes):
    if shuffle:
        src = bytes(dst)
        step, soff = 64 >> shuffle, 0
        for doff in range(0, 64, ebytes):
            dst[doff : doff + ebytes] = src[soff : soff + ebytes]
            soff += step
            if soff & 64:
                soff += ebytes
            soff &= 63


def parse_writemask(val, g, maskbits):
    val &= 0xFFFFFFFF
    mode = (val >> 6) & 7 if maskbits >= 9 else (val >> 5) & 3
    if mode:
        val = (val * g) & 0xFFFFFFFF
    val &= 63
    allbits = U64
    if mode == 0:
        if val in (1, 2):
            allbits = (~(allbits << g) << (g & -(val & 1))) & U64
            g <<= 1
            while g < 64:
                allbits |= allbits << g
                g <<= 1
        return allbits & U64 if val < (6 if maskbits >= 9 else 3) else 0
    if mode == 1:
        return ((~(U64 << g)) << val) & U64
    if mode in (2, 4):
        return U64 if mode == 2 and val == 0 else (~(U64 << val)) & U64
    if mode in (3, 5):
        return U64 if mode == 3 and val == 0 else (~(U64 >> val)) & U64
    return 0


def bit_select(hi, lo, mask):
    return (hi & ~mask) | (lo & mask)
