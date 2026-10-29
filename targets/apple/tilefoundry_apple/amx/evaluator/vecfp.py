"""Python port of corsix/amx vecfp.c (Copyright (c) 2022 Peter Cawley; MIT; see reference license)."""
from .common import read_uint,write_uint,load_xy_reg,xy_shuffle,parse_writemask,bit_select
from .floating import (fp_fma,fp_mul,fp_add,fp_compare,fp_minmax,convert_bits,
                       bf16_fma,bf16_to_f32,fp_isnan)

def _alu(x,y,z,bits,mode,bf16=False):
    if bf16:
        if mode==0: return bf16_fma(x,y,z)
        if mode==1: return bf16_fma(x^0x8000,y,z)
        if mode==4: return 0 if x==0 or (x-65536 if x&32768 else x)<=-128 else y
        if mode in (5,7):
            if fp_isnan(x,'bf16') or fp_isnan(z,'bf16'): return 0x7fc0
            def order(v):
                v=v-65536 if v&32768 else v
                return v^((v>>15)&0x7fff)
            return x if (order(x)>order(z) if mode==7 else order(x)<order(z)) else z
        if mode==10: return bf16_fma(x,y,0x8000)
        if mode==11: return bf16_fma(x,0x3f80,z)
        if mode==12: return bf16_fma(0x3f80,y,z)
        return z
    if mode==0: return fp_fma(x,y,z,bits)
    if mode==1: return fp_fma(x^(1<<(bits-1)),y,z,bits)
    if mode==4:
        c=fp_compare(x,0,bits)
        return 0 if c is not None and c<=0 else y
    if mode in (5,7): return fp_minmax(x,z,bits,mode==7)
    if mode==10: return fp_mul(x,y,bits)
    if mode==11: return fp_add(z,x,bits)
    if mode==12: return fp_add(z,y,bits)
    return z

def _types(state,operand):
    mode=(operand>>42)&15
    if mode in (0,1): return 16,(32 if mode==1 and state.version>=2 else 16),state.version>=2
    if mode==3: return 16,32,False
    if mode==4: return 32,32,False
    if mode==7: return 64,64,False
    return 16,16,False

def _load(state,operand,xoff,yoff,xybits):
    from .extract import load_xy_reg_indexed
    x=bytearray(load_xy_reg(state.x,xoff&511)); y=bytearray(load_xy_reg(state.y,yoff&511))
    if operand&(1<<53):
        reg=(operand>>49)&7; ibits=4 if operand&(1<<48) else 2
        pool=state.y if operand&(1<<47) else state.x
        load_xy_reg_indexed(y if operand&(1<<47) else x,bytes(pool[reg*64:reg*64+64]),ibits,xybits)
    xy_shuffle(x,(operand>>29)&3,xybits//8); xy_shuffle(y,(operand>>27)&3,xybits//8)
    return x,y

def _value(buf,lane,xybits,zbits,bf16):
    v=read_uint(buf,lane*(xybits//8),xybits//8)
    if xybits!=zbits: return bf16_to_f32(v) if bf16 else convert_bits(v,xybits,zbits)
    return v

def emulate_AMX_VECFP(state,operand):
    if (operand>>54)&7: return
    operand&=~(1<<37)
    mode=0 if operand&(1<<53) else (operand>>47)&63
    if mode not in (0,1,4,5,7) and not (mode in (10,11,12) and state.version>=2): return
    xybits,zbits,bf16=_types(state,operand); n=xybits//8
    zr=operand>>20; zs=xs=ys=64; ximask=-1; xo=operand>>10; yo=operand
    ibits=4 if operand&(1<<48) else 2
    if state.version>=2 and operand&(1<<31):
        bm=(operand>>32)&7; operand&=~(511<<32)
        if bm==1: operand|=3<<32
        elif bm==2: xs=0
        elif bm==3: ys=0
        elif bm==4: operand|=4<<32
        elif bm==5: operand|=5<<32
        elif bm==6: xs=0; ximask=0
        elif bm==7: ys=0; operand|=1<<38
        zs=16 if zr&32 else 32
        if state.version>=4:
            if operand&(1<<53) and not operand&(1<<47): xo&= (-64 & 0xffffffff)|(-(512*ibits//(n*zs)) & 0xffffffff)
            elif bm==6: xo&=-n & 0xffffffff
            else: xo&=-64 & 0xffffffff
            if operand&(1<<53) and operand&(1<<47): yo&=(-64 & 0xffffffff)|(-(512*ibits//(n*zs)) & 0xffffffff)
            elif bm==7: yo&=-n & 0xffffffff
            else: yo&=-64 & 0xffffffff
    if operand&(1<<53):
        if operand&(1<<47): ys=ys*ibits//xybits
        else: xs=xs*ibits//xybits
    for row in range(zr&(zs-1),64,zs):
        x,y=_load(state,operand,xo,yo,xybits); xo+=xs; yo+=ys
        xe=parse_writemask(operand>>32,n,9); broadcast=((operand>>38)&7)==1; omask=True
        if broadcast: xe=(1<<64)-1
        elif ((operand>>38)&7)==0:
            val=(operand>>32)&63
            if val==3: omask=False
            elif val==4: x[:]=bytes(64)
            elif val==5: y[:]=bytes(64)
        for i in range(64//n):
            if not (xe>>(i*n))&1: continue
            j=((operand>>32)&((64//n)-1)) if broadcast else i
            rr=bit_select(row,i,1) if zbits==32 and xybits==16 else row
            ii=i>>1 if zbits==32 and xybits==16 else i
            off=rr*64+ii*(zbits//8)
            xv=_value(x,i&ximask,xybits,zbits,bf16); yv=_value(y,j,xybits,zbits,bf16)
            z=read_uint(state.z,off,zbits//8)
            write_uint(state.z,off,zbits//8,_alu(xv,yv,z,zbits,mode,bf16 and zbits==16) if omask else 0)

vecfp_alu16=lambda x,y,z,alumode:_alu(x,y,z,16,alumode)
vecfp_alu32=lambda x,y,z,alumode:_alu(x,y,z,32,alumode)
vecfp_alu64=lambda x,y,z,alumode:_alu(x,y,z,64,alumode)
vecfp_alu_bf16=lambda x,y,z,alumode:_alu(x,y,z,16,alumode,True)
