"""Python port of corsix/amx matfp.c (Copyright (c) 2022 Peter Cawley; MIT; see reference license)."""
from .common import read_uint,write_uint,parse_writemask,bit_select
from .vecfp import _alu,_types,_load,_value

def emulate_AMX_MATFP(state,operand):
    if (operand>>54)&7: return
    operand&=~((1<<37)|(1<<63))
    mode=0 if operand&(1<<53) else (operand>>47)&63
    if mode in (2,3) or mode>=5: return
    xybits,zbits,bf16=_types(state,operand); n=xybits//8
    x,y=_load(state,operand,(operand>>10)&511,operand&511,xybits)
    xe=parse_writemask(operand>>32,n,9)
    ye=parse_writemask((((operand>>23)&7)<<6)|(operand>>58),n,9)
    omask=True
    if ((operand>>38)&7)==0:
        val=(operand>>32)&63
        if val==3: omask=False
        elif val in (4,5): x[:]=bytes(64)
    if ((operand>>23)&7)==0:
        val=(operand>>58)&63
        if val==3: omask=False
        elif val in (4,5): y[:]=bytes(64)
    zr=(operand>>20)&7
    for j in range(64//n):
        if not (ye>>(j*n))&1: continue
        for i in range(64//n):
            if not (xe>>(i*n))&1: continue
            if zbits==16: row=bit_select(j*2,zr,1); index=i
            elif xybits==16: row=bit_select(j*2,i,1); index=i>>1
            else: row=bit_select(j*n,zr,n-1); index=i
            off=row*64+index*(zbits//8)
            xv=_value(x,i,xybits,zbits,bf16); yv=_value(y,j,xybits,zbits,bf16)
            z=read_uint(state.z,off,zbits//8)
            write_uint(state.z,off,zbits//8,_alu(xv,yv,z,zbits,mode,bf16 and zbits==16) if omask else 0)
