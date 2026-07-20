"""Closed, process-lifetime DType descriptors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    name: str
    bit_width: int

    @classmethod
    def _members(cls) -> dict[str, "DType"]:
        """Descriptor singletons keyed by surface name (internal — the
        spec closes the Enum-style iteration surface; ``from_name`` is
        the one public resolution entry)."""
        return {v.name: v for v in vars(cls).values() if isinstance(v, DType)}

    @classmethod
    def from_name(cls, name: str) -> "DType":
        """Resolve a surface dtype string to its descriptor.

        The single string resolution surface: an unknown string is
        rejected with the valid names in the message.
        """
        members = cls._members()
        member = members.get(name)
        if member is None:
            raise ValueError(
                f"DType: unknown value {name!r}; valid: {sorted(members)}"
            )
        return member


@dataclass(frozen=True)
class FloatDType(DType):
    exponent_bits: int
    mantissa_bits: int


@dataclass(frozen=True)
class IntegerDType(DType):
    signed: bool


@dataclass(frozen=True)
class BoolDType(DType):
    pass


DType.f32 = FloatDType(name="f32", bit_width=32, exponent_bits=8, mantissa_bits=23)
DType.f16 = FloatDType(name="f16", bit_width=16, exponent_bits=5, mantissa_bits=10)
DType.bf16 = FloatDType(name="bf16", bit_width=16, exponent_bits=8, mantissa_bits=7)
DType.fp8e4m3 = FloatDType(name="fp8e4m3", bit_width=8, exponent_bits=4, mantissa_bits=3)
DType.f8e8m0 = FloatDType(name="f8e8m0", bit_width=8, exponent_bits=8, mantissa_bits=0)
DType.f4e2m1 = FloatDType(name="f4e2m1", bit_width=4, exponent_bits=2, mantissa_bits=1)
DType.i32 = IntegerDType(name="i32", bit_width=32, signed=True)
DType.i64 = IntegerDType(name="i64", bit_width=64, signed=True)
DType.bool = BoolDType(name="bool", bit_width=1)


__all__ = ["BoolDType", "DType", "FloatDType", "IntegerDType"]
