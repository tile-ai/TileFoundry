"""Translation of ldst.c (Peter Cawley, MIT); addresses are logical offsets."""

from .common import bit_select


def copy_bytes(dst, doff, src, soff, count=64):
    if min(doff, soff) < 0 or doff + count > len(dst) or soff + count > len(src):
        raise IndexError("reference memory access outside bounds")
    dst[doff : doff + count] = src[soff : soff + count]


def load_register(state, regs, operand, mask):
    rn, addr = (operand >> 56) & mask, operand & ((1 << 56) - 1)
    copy_bytes(regs, rn * 64, state.memory, addr)
    if operand & (1 << 62):
        rs = 1
        if state.version >= 3 and operand & (1 << 61) and mask <= 15:
            rs = 2 if operand & (1 << 60) else 4
        copy_bytes(regs, ((rn + rs) & mask) * 64, state.memory, addr + 64)
        if state.version >= 2 and operand & (1 << 60) and mask <= 15:
            for i in (2, 3):
                copy_bytes(regs, ((rn + rs * i) & mask) * 64, state.memory, addr + 64 * i)


def store_register(state, regs, operand, mask):
    rn, addr = (operand >> 56) & mask, operand & ((1 << 56) - 1)
    copy_bytes(state.memory, addr, regs, rn * 64)
    if operand & (1 << 62):
        copy_bytes(state.memory, addr + 64, regs, ((rn + 1) & mask) * 64)


def emulate_amx_ldx(s, o):
    load_register(s, s.x, o, 7)


def emulate_amx_ldy(s, o):
    load_register(s, s.y, o, 7)


def emulate_amx_ldz(s, o):
    load_register(s, s.z, o, 63)


def emulate_amx_stx(s, o):
    store_register(s, s.x, o, 7)


def emulate_amx_sty(s, o):
    store_register(s, s.y, o, 7)


def emulate_amx_stz(s, o):
    store_register(s, s.z, o, 63)


def emulate_amx_ldzi(s, o):
    rn, addr = (o >> 56) & 63, o & ((1 << 56) - 1)
    half = (rn & 1) << 3
    for i in range(16):
        copy_bytes(s.z, bit_select(rn, i, 1) * 64 + (half + (i >> 1)) * 4, s.memory, addr + i * 4, 4)


def emulate_amx_stzi(s, o):
    rn, addr = (o >> 56) & 63, o & ((1 << 56) - 1)
    half = (rn & 1) << 3
    for i in range(16):
        copy_bytes(s.memory, addr + i * 4, s.z, bit_select(rn, i, 1) * 64 + (half + (i >> 1)) * 4, 4)


__all__ = ["emulate_amx_ldx", "emulate_amx_ldy", "emulate_amx_ldz", "emulate_amx_ldzi", "emulate_amx_stx", "emulate_amx_sty", "emulate_amx_stz", "emulate_amx_stzi"]
