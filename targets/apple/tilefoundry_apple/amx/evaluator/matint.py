"""Translation of matint.c (Peter Cawley, MIT)."""

from .common import U64, bit_select, load_int, load_xy_reg, parse_writemask, write_uint, xy_shuffle
from .extract import load_xy_reg_indexed
from .vecint import vecint_alu, vecint_alu_mode4


def emulate_AMX_MATINT(state, operand):
    if (operand >> 55) & 3:
        return
    indexed = bool(operand & (1 << 53))
    if operand & (1 << 54) and not indexed:
        return
    zr = (operand >> 20) & 3
    omask = 0 if ((operand >> 32) & 511) == 3 else -1
    mode = ((operand & (1 << 54)) >> 51) if indexed else (operand >> 47) & 63
    shift, width = (operand >> 58) & 31, (operand >> 42) & 15
    mask_y = bool(operand & (1 << 25))
    if mode == 4:
        zb, sat = {3: (32, 16), 4: (32, 32), 10: (32, 8), 11: (16, 8)}.get(width, (16, 16))
        zn = zb // 8
        enable = parse_writemask(operand >> 32, zn, 9)
        for j in range(0, 64, zn):
            for i in range(0, 64, zn):
                if not (enable >> (j if mask_y else i)) & 1:
                    continue
                off = bit_select(j, zr, zn - 1) * 64 + i
                val = load_int(state.z, off, zn, 64 - zb if operand & (1 << 63) else 0)
                write_uint(state.z, off, zn, vecint_alu_mode4(val, sat, operand) & omask)
        return
    if mode in (5, 6):
        xb, yb, zb, shift = 16, 16, 16, 15
    elif mode == 8:
        xb, yb, zb = (
            (8, 8, 32)
            if width == 10
            else ((8, 16, 32) if state.version >= 3 else (8, 8, 16))
            if width == 12
            else (8, 8, 16)
        )
    elif mode == 9:
        xb, yb, zb = {3: (16, 16, 32), 4: (32, 32, 32)}.get(width, (16, 16, 16))
        shift = 64 - xb
    else:
        xb, yb, zb = (16, 16, 32) if width == 3 else (16, 16, 16)
    if mode == 7 or mode >= 10:
        return
    xn, yn, zn = xb // 8, yb // 8, zb // 8
    x, y = (
        bytearray(load_xy_reg(state.x, (operand >> 10) & 511)),
        bytearray(load_xy_reg(state.y, operand & 511)),
    )
    if indexed:
        iy = bool(operand & (1 << 47))
        pool = state.y if iy else state.x
        off = ((operand >> 49) & 7) * 64
        load_xy_reg_indexed(
            y if iy else x, pool[off : off + 64], 4 if operand & (1 << 48) else 2, yb if iy else xb
        )
    xy_shuffle(x, (operand >> 29) & 3, xn)
    xy_shuffle(y, (operand >> 27) & 3, yn)
    xe = U64 if mask_y else parse_writemask(operand >> 32, xn, 9)
    ye = parse_writemask(operand >> 32, yn, 9) if mask_y else U64
    if ((operand >> 38) & 7) == 0 and ((operand >> 32) & 63) in (4, 5):
        if mask_y:
            y[:] = bytes(64)
        else:
            x[:] = bytes(64)
    zm, step = zn // xn - 1, zn if xn == 1 else xn
    for j in range(0, 64, step):
        if not (ye >> j) & 1:
            continue
        for i in range(0, 64, xn):
            if not (xe >> i) & 1:
                continue
            xv = load_int(x, i, xn, 64 - xb if operand & (1 << 63) else 0)
            yv = load_int(y, j, yn, 64 - yb if operand & (1 << 26) else 0)
            off = bit_select(bit_select(j, zr, xn - 1), i // xn, zm) * 64 + (i & -zn)
            zv = load_int(state.z, off, zn, 64 - zb)
            write_uint(state.z, off, zn, vecint_alu(xv, yv, zv, mode, shift) & omask)
