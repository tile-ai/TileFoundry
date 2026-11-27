"""H14 HWX symbols encoding."""

from tilefoundry_apple.ane.isa import arith

from .bindings import TensorBinding
from .macho import PAGE, SYMBOL, TEXT_VM_SIZE, VM_BASE

N_SECT = 0x0E
N_EXT = 0x01
ELEMENT_CATALOG = (
    "void:t1=1",
    "int8:t2=r2;0;127",
    "uint8:t3=r1;0;255",
    "int16:t4=r1;-32768;32767",
    "float16:t5=r1;2;0",
    "float:t6=r1;4;0",
    "raw10:t7=r1;-512;511",
    "lut:t8=",
    "uint4:t9=r1;0;15",
    "uint6:t10=r1;0;63",
    "uint1:t11=r1;0,1",
    "uint2:t12=r1;0;3",
    "packed10:t13=r1;-512;511",
    "y10:t14=r1;-512;511",
    "uint16:t15=r1;0;65535",
    "e4m3:t16=r1;1;0",
    "int32:t17=r1;-2147483648;2147483647",
    "uint32:t18=r1;0;4294967295",
    "int64:t19=r1;-9223372036854775808;9223372036854775807",
    "uint64:t20=r1;0;18446744073709551615",
    "raw12:t21=r1;-512;511",
    "y12:t22=r1;-512;511",
    "int4:t23=r1;-8;7",
)
FIRST_SURFACE_TYPE = 1 + len(ELEMENT_CATALOG)
AXES = "nchw"
CATALOG_TYPE = 0x80
SHAPE_TYPE = 0x20


def shape_stab(binding: TensorBinding, base: int) -> str:
    """Encode one surface's element type and its per-axis byte strides.

    The record is ``name:t<base>=`` followed by one ``ar1;0;<extent>;<id>=s<stride><axis>``
    group per axis (batch, channel, height, width) and the element code last.
    """
    desc = binding.desc
    row, plane, batch, _ = desc.canonical_geometry
    shape = desc.shape
    strides = (batch, plane, row, desc.element_bytes)
    groups = []
    for axis, extent, stride in zip(AXES, shape, strides):
        groups.append(f"ar1;0;{extent};{base + 1 + len(groups)}=s{stride}{axis}")
    return f"{binding.name}:t{base}=" + ":".join(groups) + f":{desc.element_code}"


def symbol_table(
    image: "Image", bindings: tuple[TensorBinding, ...], kernel_section: int | None = None
) -> tuple[bytes, bytes]:
    """Encode the surface symbols, the element catalog and the shape records.

    Three groups follow each other: one external symbol per surface in section
    order, the element-type catalog shared by every image, and one shape record
    per surface whose type ids continue past the catalog.
    """
    names = bytearray(b"\0")
    symbols = bytearray()
    for index, binding in enumerate(bindings):
        symbols.extend(SYMBOL.pack(len(names), N_SECT | N_EXT, index + 1, 0, binding.address))
        names.extend(binding.name.encode("ascii") + b"\0")
    if image.coefficients:
        if kernel_section is None:
            raise ValueError("a coefficient blob needs its section ordinal")
        address = (
            VM_BASE
            + arith.align_up(image.scratch_bytes, PAGE)
            + sum(binding.size for binding in bindings)
            + TEXT_VM_SIZE
        )
        token = image.kernel_digest or image.kernel_symbol
        for index, coefficient in enumerate(image.coefficients):
            layout = coefficient.layout
            lanes, stride, base = layout.lanes, layout.stride, layout.base
            prefix = f"{token}_{index}" if len(image.coefficients) > 1 else token
            for lane in range(lanes):
                symbols.extend(
                    SYMBOL.pack(
                        len(names),
                        N_SECT | N_EXT,
                        kernel_section,
                        0x0002 if lane == 0 else 0,
                        address + base + lane * stride,
                    )
                )
                names.extend(f"{prefix}_ne_{lane}".encode("ascii") + b"\0")
    for entry in ELEMENT_CATALOG:
        symbols.extend(SYMBOL.pack(len(names), CATALOG_TYPE, 0, 0, 0))
        names.extend(entry.encode("ascii") + b"\0")
    for index, binding in enumerate(bindings):
        base = FIRST_SURFACE_TYPE + 5 * index
        symbols.extend(SYMBOL.pack(len(names), SHAPE_TYPE, 0, base & 0x1F, 0))
        names.extend(shape_stab(binding, base).encode("ascii") + b"\0")
    return bytes(symbols), bytes(names)


__all__ = ["symbol_table"]
