"""Closed, process-lifetime DType descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["BoolDType", "DType", "FloatDType", "IntegerDType", "Member", "dtype_members"]


class Member:
    """A ``DType`` of the class that holds it, built on first access and then shared.

    The member names its kind by a string because the kinds are subclasses of the
    class whose body states the member.
    """

    def __init__(self, kind: str, **fields: int | bool) -> None:
        """State the kind and the fields of the member."""
        self.kind = kind
        self.fields = fields
        self.name = ""
        self.value: DType | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Take the attribute name as the descriptor's name."""
        self.name = name

    def __get__(self, instance: object, owner: type) -> DType:
        """The one descriptor this member stands for."""
        if self.value is None:
            self.value = KINDS[self.kind](name=self.name, **self.fields)
        return self.value


@dataclass(frozen=True)
class DType:
    """A data type: its surface name and its width in bits."""

    name: str
    bit_width: int

    f64 = Member("float", bit_width=64, exponent_bits=11, mantissa_bits=52)
    f32 = Member("float", bit_width=32, exponent_bits=8, mantissa_bits=23)
    f16 = Member("float", bit_width=16, exponent_bits=5, mantissa_bits=10)
    bf16 = Member("float", bit_width=16, exponent_bits=8, mantissa_bits=7)
    fp8e4m3 = Member("float", bit_width=8, exponent_bits=4, mantissa_bits=3)
    f8e8m0 = Member("float", bit_width=8, exponent_bits=8, mantissa_bits=0)
    f4e2m1 = Member("float", bit_width=4, exponent_bits=2, mantissa_bits=1)
    i8 = Member("integer", bit_width=8, signed=True)
    u8 = Member("integer", bit_width=8, signed=False)
    i16 = Member("integer", bit_width=16, signed=True)
    u16 = Member("integer", bit_width=16, signed=False)
    i32 = Member("integer", bit_width=32, signed=True)
    u32 = Member("integer", bit_width=32, signed=False)
    i64 = Member("integer", bit_width=64, signed=True)
    bool = Member("bool", bit_width=1)

    @classmethod
    def from_name(cls, name: str) -> DType:
        """Resolve a surface dtype string to its descriptor.

        The single string resolution surface: an unknown string is
        rejected with the valid names in the message.
        """
        members = dtype_members(cls)
        member = members.get(name)
        if member is None:
            raise ValueError(f"DType: unknown value {name!r}; valid: {sorted(members)}")
        return member


def dtype_members(cls: type = DType) -> dict[str, DType]:
    """The descriptors the class *cls* itself holds, keyed by surface name."""
    return {
        held.name: getattr(cls, held.name) for held in vars(cls).values() if isinstance(held, Member)
    }


@dataclass(frozen=True)
class FloatDType(DType):
    """A floating-point type: its exponent and mantissa widths."""

    exponent_bits: int
    mantissa_bits: int


@dataclass(frozen=True)
class IntegerDType(DType):
    """An integer type: whether it is signed."""

    signed: bool


@dataclass(frozen=True)
class BoolDType(DType):
    """The boolean type."""


KINDS = MappingProxyType({"float": FloatDType, "integer": IntegerDType, "bool": BoolDType})
