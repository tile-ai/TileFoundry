"""Translation of extr.c and GENLUT's indexed-load helper (Peter Cawley, MIT)."""
from .common import *

def load_xy_reg_indexed(dst, table, ibits, ebits):
    tmp = bytes(dst[:40])
    ebytes, imask, doff, soff = ebits//8, (1 << ibits)-1, 0, 0
    while doff < 64:
        # Eight indices consume exactly ibits bytes. The C helper reads a
        # uint64, whose unused high bytes may cross tmp[40] on the last group.
        bits = read_uint(tmp, soff, ibits)
        soff += ibits
        for _ in range(8):
            toff = ((bits & imask)*ebytes)&63
            dst[doff:doff+ebytes] = table[toff:toff+ebytes]
            bits >>= ibits
            doff += ebytes

def extr_alu(val, operand, outbits):
    shift = (operand >> 58)&31
    if operand & (1 << 63):
        if shift >= 16:
            f = val & 0xffffffff
            if (f & 0x7fffffff) > 0x7f800000:
                return BF16_NAN
            low = f & 0xffff
            f >>= 16
            return f + (low + (f & 1) > 0x8000)
        from .floating import convert_bits
        return convert_bits(val & 0xffffffff, 32, 16)
    if shift and operand & (1 << 54):
        val += 1 << (shift-1)
    val >>= shift
    if operand & (1 << 55):
        if operand & (1 << 56):
            outbits -= 1
        hi = 1 << outbits
        if operand & (1 << 57):
            lo = -hi if operand & (1 << 56) else 0
            val = max(lo, min(hi-1, val))
        elif (val & U64) >= hi:
            val = hi-1
    return val

def _extract(state, operand, vertical):
    row = operand >> 20
    step, enable, stride = 64, U64, 0
    offset = operand
    hv = bool(operand & (1 << 26))
    if hv:
        dst = state.y if operand & (1 << 10) else state.x
        mode = ((operand >> 63) << 4) | ((operand >> 11)&15)
        xybytes, zbytes, stride = {0:(1,1,0),8:(4,4,0),9:(2,4,1),10:(2,4,2),11:(1,4,1),13:(1,2,1),17:(8,8,0),24:(4,4,0)}.get(mode,(2,2,0))
        if mode in (25,26) and state.version >= 2:
            xybytes,zbytes,stride = 2,4,mode-24
        if state.version >= 2 and operand & (1 << 31):
            operand &= ~(511 << 32)
            step = 16 if row & 32 else 32
            if state.version >= 4:
                offset &= 0xffffffc0
        enable &= parse_writemask(operand >> 32, xybytes, 9)
    elif operand & (1 << 27):
        if vertical:
            state.y[((operand>>6)&7)*64:(((operand>>6)&7)+1)*64] = state.x[(row&7)*64:((row&7)+1)*64]
        else:
            state.x[((operand>>16)&7)*64:(((operand>>16)&7)+1)*64] = state.y[(row&7)*64:((row&7)+1)*64]
        return
    else:
        dst = state.y if vertical else state.x
        if not vertical:
            offset = operand >> 10
        xybytes = 8 >> ((operand >> 28)&3)
        if xybytes == 1:
            xybytes = 2
            enable &= 0x5555555555555555
        enable &= parse_writemask(operand >> (32 if vertical else 41), xybytes, 7)
        zbytes = xybytes
    signext = 64-zbytes*8 if operand & (1 << 57) else 0
    for row in range(row & (step-1),64,step):
        buffer = bytearray(64)
        for i in range(0,64,xybytes):
            zoff = (i & (zbytes-1))//xybytes*stride
            zr = bit_select(i if vertical else row, row+zoff, zbytes-1)
            off = row & -zbytes if vertical else i & -zbytes
            val = load_int(state.z,zr*64+off,zbytes,signext)
            if stride:
                val = extr_alu(val,operand,xybytes*8)
            write_uint(buffer,i,xybytes,val)
        if (vertical or hv) and ((operand >> 32)&511) == 3:
            buffer[:] = bytes(64)
        for i in range(64):
            if (enable >> i)&1:
                dst[(offset+i)&511] = buffer[i]
        offset += 64

def emulate_AMX_EXTRX(state, operand):
    _extract(state,operand,False)

def emulate_AMX_EXTRY(state, operand):
    _extract(state,operand,True)
