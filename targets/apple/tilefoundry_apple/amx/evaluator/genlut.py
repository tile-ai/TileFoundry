"""Translation of genlut.c (Peter Cawley, MIT)."""
from .common import *
from .extract import load_xy_reg_indexed

def emulate_AMX_GENLUT(state, operand):
    mode = (operand >> 53)&15
    source = state.y if operand & (1 << 10) else state.x
    tablepool = state.y if operand & (1 << 59) else state.x
    toff = ((operand >> 60)&7)*64
    table = tablepool[toff:toff+64]
    xy = bytearray(load_xy_reg(source,operand&511))
    ibits,ebits = [(4,32),(5,16),(4,64),(4,32),(5,16),(4,32),(5,16),(2,32),(2,16),(2,8),(4,64),(4,32),(4,16),(4,8),(5,16),(5,8)][mode]
    if mode <= 6:
        from .floating import fp_compare
        n = ebits//8
        values = []
        for i in range(64//n):
            v = 0
            while v < 64//n:
                if mode <= 2:
                    tv,xv = read_uint(table,v*n,n),read_uint(xy,i*n,n)
                    bits = ebits
                    if mode == 1 and state.version >= 2 and operand & (1 << 30):
                        tv,xv,bits = tv << 16,xv << 16,32
                    greater = fp_compare(tv,xv,bits) == 1
                else:
                    read = read_sint if mode in (3,4) else read_uint
                    greater = read(table,v*n,n) > read(xy,i*n,n)
                if greater:
                    break
                v += 1
            values.append((v-1)&255)
        dst, soff = bytearray(64), 0
        imask = 7 if ebits == 64 else (1 << ibits)-1
        for etotal in range(0,64,ebits):
            packed = sum((values[etotal//ebits*8+i]&imask) << (i*ibits) for i in range(8))
            write_uint(dst,soff,8,packed)
            soff += ibits
        dst[soff:] = bytes(64-soff)
        xy = dst
        operand &= ~(1 << 26)
    else:
        load_xy_reg_indexed(xy,table,ibits,ebits)
    dest = state.z if operand & (1 << 26) else state.y if operand & (1 << 25) else state.x
    doff = ((operand >> 20)&(63 if operand & (1 << 26) else 7))*64
    dest[doff:doff+64] = xy
