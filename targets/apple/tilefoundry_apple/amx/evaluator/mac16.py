"""Translation of mac16.c (Peter Cawley, MIT)."""

from .common import (
    FMA_VECTOR_PRODUCT,
    FMA_WIDEN_16_32,
    load_xy_reg,
    parse_writemask,
    read_sint,
    write_uint,
)


def mac32_alu(x, y, z, operand):
    if operand & (1 << 61):
        x = ((x + 128) & 255) - 128
    if operand & (1 << 60):
        y = ((y + 128) & 255) - 128
    mode = (operand >> 28) & 3
    val = (x * y, x, y, 0)[mode]
    val >>= (operand >> 55) & 31
    return val if operand & (1 << 27) else val + z


def emulate_AMX_MAC16(state, operand):
    x, y = load_xy_reg(state.x, (operand >> 10) & 511), load_xy_reg(state.y, operand & 511)
    zr = (operand >> 20) & 63
    xe, ye = parse_writemask(operand >> 41, 2, 7), parse_writemask(operand >> 32, 2, 7)
    for i in range(32):
        if not (xe >> (i * 2)) & 1:
            continue
        if operand & FMA_VECTOR_PRODUCT:
            off, n = zr * 64 + i * 2, 2
            write_uint(
                state.z,
                off,
                n,
                mac32_alu(
                    read_sint(x, i * 2, 2),
                    read_sint(y, i * 2, 2),
                    read_sint(state.z, off, n),
                    operand,
                ),
            )
        else:
            for j in range(32):
                if not (ye >> (j * 2)) & 1:
                    continue
                if operand & FMA_WIDEN_16_32:
                    off, n = (j * 2 + (i & 1)) * 64 + (i >> 1) * 4, 4
                else:
                    off, n = (j * 2 + (zr & 1)) * 64 + i * 2, 2
                write_uint(
                    state.z,
                    off,
                    n,
                    mac32_alu(
                        read_sint(x, i * 2, 2),
                        read_sint(y, j * 2, 2),
                        read_sint(state.z, off, n),
                        operand,
                    ),
                )
