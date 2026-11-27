"""What a neural-engine task applies to its result."""

import struct
from enum import Enum
from functools import partial
from types import MappingProxyType

from tilefoundry import utils
from tilefoundry.ir import expr
from tilefoundry_apple.ane.ir.tir import evaluation

SAMPLES = 33
ORIGIN = 16
CENTRED = 1


@expr.register_type(dialect="T", namespace="ane")
class Nonlinear(Enum):
    """What a neural-engine task applies after its arithmetic."""

    NONE = "none"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SILU = "silu"


TABLES = MappingProxyType(
    {
        Nonlinear.SIGMOID: bytes.fromhex(
            "f8c829480000003c7f0d87107713261610192b1cda1ea0219b248127122adb2c"
            "a12fd6314e340a360038fb38d9398a3a0c3b653b9f3bc43bdb3be93bf23bf83b"
            "fb3bfd3bfe3bff3bff3bac090c173e16e63b0100010000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
        ),
        Nonlinear.SILU: bytes.fromhex(
            "1ac9007c0000007ccd985e9c659eefa07ba3a9a53ba845aa8eac85ae88b010b2"
            "a9b370b460b484b2fb96be34d039e43c0b3f9f40b741cc42dc437344f8447a45"
            "fc457e46fe467f47ff475d9491a1003c009c0100010000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
        ),
    }
)


def lookup(table: bytes, x):
    """Evaluate a centred lookup table at x.

    The table is 64 little-endian halfwords: fp16 ``x_lo`` and ``x_hi``, the outputs
    at or below ``x_lo`` and at or above ``x_hi``, 33 fp16 samples, the left and
    right lines as fp16 slope and offset, then the integer shift and mode. Sample i
    sits at ``x * 2**shift + 16 == i``; between samples the value is interpolated
    linearly and rounded once to fp16. At or left of sample 0 and right of sample 32
    a line gives ``f16(f16(slope * x) + offset)``.
    """
    torch = utils.torch
    halves = torch.frombuffer(bytearray(table), dtype=torch.float16).double()
    low, high, below, above = halves[:4]
    samples = halves[4 : 4 + SAMPLES]
    left, left_offset, right, right_offset = halves[4 + SAMPLES : 8 + SAMPLES]
    shift, mode = struct.unpack_from("<2H", table, 2 * (8 + SAMPLES))
    if mode != CENTRED:
        raise ValueError(f"lookup-table mode {mode:#x} is not decoded yet")
    x = x.double()
    position = x * 2**shift + ORIGIN
    index = position.clamp(0, SAMPLES - 2).floor().nan_to_num().long()
    fraction = position.clamp(0, SAMPLES - 1) - index
    y = evaluation.f16(samples[index] + (samples[index + 1] - samples[index]) * fraction)
    y = torch.where(position <= 0, evaluation.f16(evaluation.f16(left * x) + left_offset), y)
    y = torch.where(
        position > SAMPLES - 1, evaluation.f16(evaluation.f16(right * x) + right_offset), y
    )
    y = torch.where(x <= low, below, y)
    return torch.where(x >= high, above, y)


KINDS = MappingProxyType(
    {
        Nonlinear.NONE: lambda value: value,
        Nonlinear.RELU: lambda value: value.relu(),
        **{kind: partial(lookup, table) for kind, table in TABLES.items()},
    }
)


def apply(kind: Nonlinear, x):
    """x, max(x, 0), or the kind's lookup table at x."""
    return KINDS[kind](x)


__all__ = ["Nonlinear", "TABLES", "apply", "lookup"]
