"""Python port of corsix/amx fms.c (Copyright (c) 2022 Peter Cawley; MIT; see reference license)."""

from .fma import emulate_fma


def emulate_amx_fms16(state, operand):
    emulate_fma(state, operand, 16, True)


def emulate_amx_fms32(state, operand):
    emulate_fma(state, operand, 32, True)


def emulate_amx_fms64(state, operand):
    emulate_fma(state, operand, 64, True)


__all__ = ["emulate_amx_fms16", "emulate_amx_fms32", "emulate_amx_fms64"]
