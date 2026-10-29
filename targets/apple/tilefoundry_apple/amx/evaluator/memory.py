"""Translation of ldst.c (Peter Cawley, MIT); addresses are logical offsets."""

from .common import bit_select


def _copy(dst, doff, src, soff, count=64):
    if min(doff, soff) < 0 or doff + count > len(dst) or soff + count > len(src):
        raise IndexError("reference memory access outside bounds")
    dst[doff : doff + count] = src[soff : soff + count]


def _ld(state, regs, operand, mask):
    rn, addr = (operand >> 56) & mask, operand & ((1 << 56) - 1)
    _copy(regs, rn * 64, state.memory, addr)
    if operand & (1 << 62):
        rs = 1
        if state.version >= 3 and operand & (1 << 61) and mask <= 15:
            rs = 2 if operand & (1 << 60) else 4
        _copy(regs, ((rn + rs) & mask) * 64, state.memory, addr + 64)
        if state.version >= 2 and operand & (1 << 60) and mask <= 15:
            for i in (2, 3):
                _copy(regs, ((rn + rs * i) & mask) * 64, state.memory, addr + 64 * i)


def _st(state, regs, operand, mask):
    rn, addr = (operand >> 56) & mask, operand & ((1 << 56) - 1)
    _copy(state.memory, addr, regs, rn * 64)
    if operand & (1 << 62):
        _copy(state.memory, addr + 64, regs, ((rn + 1) & mask) * 64)


def emulate_AMX_LDX(s, o):
    _ld(s, s.x, o, 7)


def emulate_AMX_LDY(s, o):
    _ld(s, s.y, o, 7)


def emulate_AMX_LDZ(s, o):
    _ld(s, s.z, o, 63)


def emulate_AMX_STX(s, o):
    _st(s, s.x, o, 7)


def emulate_AMX_STY(s, o):
    _st(s, s.y, o, 7)


def emulate_AMX_STZ(s, o):
    _st(s, s.z, o, 63)


def emulate_AMX_LDZI(s, o):
    rn, addr = (o >> 56) & 63, o & ((1 << 56) - 1)
    half = (rn & 1) << 3
    for i in range(16):
        _copy(s.z, bit_select(rn, i, 1) * 64 + (half + (i >> 1)) * 4, s.memory, addr + i * 4, 4)


def emulate_AMX_STZI(s, o):
    rn, addr = (o >> 56) & 63, o & ((1 << 56) - 1)
    half = (rn & 1) << 3
    for i in range(16):
        _copy(s.memory, addr + i * 4, s.z, bit_select(rn, i, 1) * 64 + (half + (i >> 1)) * 4, 4)
