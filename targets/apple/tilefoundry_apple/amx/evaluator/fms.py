"""Python port of corsix/amx fms.c (Copyright (c) 2022 Peter Cawley; MIT; see reference license)."""
from .fma import _emulate

def emulate_AMX_FMS16(state,operand): _emulate(state,operand,16,True)
def emulate_AMX_FMS32(state,operand): _emulate(state,operand,32,True)
def emulate_AMX_FMS64(state,operand): _emulate(state,operand,64,True)
