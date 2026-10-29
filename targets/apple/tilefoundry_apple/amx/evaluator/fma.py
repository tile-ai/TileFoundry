"""Python port of corsix/amx fma.c (Copyright (c) 2022 Peter Cawley; MIT; see reference license)."""
from .common import read_uint, write_uint, load_xy_reg, parse_writemask
from .floating import convert_bits, fp_isnan, fp_fma, fp_add, fp_mul

def _alu(x,y,z,bits,operand,subtract=False):
    mode=(operand>>27)&7; sign=1<<(bits-1)
    if subtract:
        if mode==1: z=sign
        elif mode==2: return fp_add(z,x^sign,bits)
        elif mode==3: return x^sign
        elif mode==4: return fp_add(z,y^sign,bits)
        elif mode==5: return y^sign
        elif mode==6: return z
        elif mode==7: return sign
        return fp_fma(x^sign,y,z,bits)
    if mode==1: return fp_mul(x,y,bits)
    if mode==2: return fp_add(z,x,bits)
    if mode==3: return x
    if mode==4: return fp_add(z,y,bits)
    if mode==5: return y
    if mode==6: return z
    if mode==7: return 0
    return fp_fma(x,y,z,bits)

def _emulate(state,operand,bits,subtract=False):
    n=bits//8; lanes=64//n; zr=(operand>>20)&63
    xe=parse_writemask(operand>>41,n,7); ye=parse_writemask(operand>>32,n,7)
    xb=load_xy_reg(state.x,(operand>>10)&511); yb=load_xy_reg(state.y,operand&511)
    def load(buf,i,half):
        v=read_uint(buf,i*n,2 if half else n)
        if half:
            v=convert_bits(v,16,32)
            if subtract and fp_isnan(v,32): v^=0x80000000
        return v
    x=[load(xb,i,bits==32 and bool(operand&(1<<61))) for i in range(lanes)]
    y=[load(yb,i,bits==32 and bool(operand&(1<<60))) for i in range(lanes)]
    for i in range(lanes):
        if not (xe>>(i*n))&1: continue
        js=[i] if operand&(1<<63) else range(lanes)
        for j in js:
            if not operand&(1<<63) and not (ye>>(j*n))&1: continue
            wide=bits==16 and not operand&(1<<63) and bool(operand&(1<<62))
            row=zr if operand&(1<<63) else j*n+(zr&(n-1))
            index=i; outbits=bits; xv=x[i]; yv=y[j]
            if wide:
                row=j*2+(i&1); index=i>>1; outbits=32
                xv=convert_bits(xv,16,32); yv=convert_bits(yv,16,32)
                if subtract:
                    if fp_isnan(xv,32): xv^=0x80000000
                    if fp_isnan(yv,32): yv^=0x80000000
            size=outbits//8; off=row*64+index*size
            z=read_uint(state.z,off,size)
            write_uint(state.z,off,size,_alu(xv,yv,z,outbits,operand,subtract))

def emulate_AMX_FMA16(state,operand): _emulate(state,operand,16)
def emulate_AMX_FMA32(state,operand): _emulate(state,operand,32)
def emulate_AMX_FMA64(state,operand): _emulate(state,operand,64)
