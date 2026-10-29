"""Translation of vecint.c (Peter Cawley, MIT)."""

from .common import U64, bit_select, load_int, load_xy_reg, parse_writemask, write_uint, xy_shuffle
from .extract import load_xy_reg_indexed


def vecint_alu_mode4(val, satbits, operand):
    shift = (operand >> 58) & 31
    if shift and operand & (1 << 29):
        val += 1 << (shift - 1)
    val >>= shift
    if operand & (1 << 30):
        if operand & (1 << 26):
            satbits -= 1
        hi = 1 << satbits
        if operand & (1 << 63):
            val = max(-hi if operand & (1 << 26) else 0, min(hi - 1, val))
        elif (val & U64) >= hi:
            val = hi - 1
    return val


def vecint_alu(x, y, z, alumode, shift):
    val = x * y
    if alumode in (5, 6):
        val += 1 << (shift - 1)
    elif alumode in (2, 3):
        val = x + y
    elif alumode == 9:
        return z + (((~(x ^ y)) << shift) & U64).bit_count()
    elif alumode == 11:
        val = x
    elif alumode == 12:
        val = y
    val >>= shift
    if alumode in (1, 3, 6):
        val = -val
    if alumode != 10:
        val += z
    if alumode in (5, 6):
        val = max(-32768, min(32767, val))
    return val


def emulate_AMX_VECINT(state, operand):
    if (operand >> 54) & 7:
        return
    zr, zs, xs, ys, ximask = operand >> 20, 64, 64, 64, -1
    if state.version >= 2 and operand & (1 << 31):
        bm = (operand >> 32) & 7
        operand &= ~(511 << 32)
        if bm == 1:
            operand |= 3 << 32
        elif bm == 2:
            xs = 0
        elif bm == 3:
            ys = 0
        elif bm == 4:
            operand |= 4 << 32
        elif bm == 5:
            operand |= 5 << 32
        elif bm == 6:
            xs, ximask = 0, 0
        elif bm == 7:
            ys = 0
            operand |= 1 << 38
        zs = 16 if zr & 32 else 32
    zr &= zs - 1
    omask = 0 if ((operand >> 32) & 511) == 3 else -1
    broadcast = ((operand >> 38) & 7) == 1
    indexed = bool(operand & (1 << 53))
    mode = 0 if indexed else (operand >> 47) & 63
    shift = (operand >> 58) & 31
    width = (operand >> 42) & 15
    if mode == 4:
        zb, sat = {3: (32, 16), 4: (32, 32), 9: (8, 8), 10: (32, 8), 11: (16, 8)}.get(
            width, (16, 16)
        )
        n = zb // 8
        enable = U64 if broadcast else parse_writemask(operand >> 32, n, 9)
        for row in range(zr, 64, zs):
            for i in range(0, 64, n):
                if (enable >> i) & 1:
                    off = row * 64 + i
                    val = load_int(state.z, off, n, 64 - zb if operand & (1 << 63) else 0)
                    write_uint(state.z, off, n, vecint_alu_mode4(val, sat, operand) & omask)
        return
    if mode in (5, 6):
        xb, yb, zb, shift = 16, 16, 16, 15
    else:
        xb, yb, zb = {
            3: (16, 16, 32),
            10: (8, 8, 32),
            11: (8, 8, 16),
            12: (8, 16, 32),
            13: (16, 8, 32),
        }.get(width, (16, 16, 16))
    if not (state.version >= 2 and mode in (10, 11, 12)) and mode >= 7:
        return
    xn, yn, zn = xb // 8, yb // 8, zb // 8
    xo, yo = operand >> 10, operand
    ibits = 4 if operand & (1 << 48) else 2
    iy = bool(operand & (1 << 47))
    if indexed:
        if iy:
            ys = ys * ibits // yb
        else:
            xs = xs * ibits // xb
    if state.version >= 4 and zs < 64:
        if indexed and not iy:
            xo &= 0xFFFFFFC0 | ((-(512 * ibits // (xn * zs))) & 0xFFFFFFFF)
        elif ximask == 0:
            xo &= -xn
        else:
            xo &= 0xFFFFFFC0
        if indexed and iy:
            yo &= 0xFFFFFFC0 | ((-(512 * ibits // (yn * zs))) & 0xFFFFFFFF)
        elif broadcast:
            yo &= -yn
        else:
            yo &= 0xFFFFFFC0
    for row in range(zr, 64, zs):
        x, y = bytearray(load_xy_reg(state.x, xo & 511)), bytearray(load_xy_reg(state.y, yo & 511))
        xo, yo = xo + xs, yo + ys
        if indexed:
            pool = state.y if iy else state.x
            off = ((operand >> 49) & 7) * 64
            load_xy_reg_indexed(y if iy else x, pool[off : off + 64], ibits, yb if iy else xb)
        xy_shuffle(x, (operand >> 29) & 3, xn)
        xy_shuffle(y, (operand >> 27) & 3, yn)
        xe, ye = parse_writemask(operand >> 32, xn, 9), parse_writemask(operand >> 32, yn, 9)
        if broadcast:
            xe, ye = U64, U64
        elif ((operand >> 38) & 7) == 0:
            val = (operand >> 32) & 63
            if val == 4:
                x[:] = bytes(64)
            elif val == 5:
                y[:] = bytes(64)
        step = min(xn, yn)
        zm = zn // step - 1
        for i in range(0, 64, step):
            xi = i & -xn & ximask
            yj = ((operand >> 32) * yn) & 63 if broadcast else i & -yn
            if not ((xe >> xi) & 1 and (ye >> yj) & 1):
                continue
            xv = load_int(x, xi, xn, 64 - xb if operand & (1 << 63) else 0)
            yv = load_int(y, yj, yn, 64 - yb if operand & (1 << 26) else 0)
            off = bit_select(row, i // step, zm) * 64 + (i & -zn)
            zv = load_int(state.z, off, zn, 64 - zb)
            write_uint(state.z, off, zn, vecint_alu(xv, yv, zv, mode, shift) & omask)
