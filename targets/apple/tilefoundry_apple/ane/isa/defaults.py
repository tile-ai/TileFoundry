"""The configuration image a descriptor starts from.

The hardware starts every descriptor from a per-architecture default image of all
186 configuration words before any layer programs them, and only the words whose
value differs from that image are emitted.  The consequence
is visible in every reference container: a descriptor never writes a word that
still holds its default (`din = 1`, `dout = 1`, `scale = 1`, `pre_scale = 1`,
`clamp = [0, inf]`, an all-zero L2 source word, ...).

Seeding the register file with this image therefore reproduces the compiler's
elision.  Only the words the ANE emitters of this target can produce are listed;
each value is the one the compiler left unprogrammed in the reference
descriptors that program the same op class.

``0x003C`` is deliberately absent: the compiler writes it in every descriptor
even when it holds its default, so it must never be elided.
"""

DEFAULT_IMAGE: dict[int, int] = {
    0x0004: 0x00000001,  # Common din
    0x0008: 0x0000002A,  # Common input, second-source and output format
    0x000C: 0x00000010,  # Common input channels
    0x0010: 0x00000010,  # Common output channels
    0x0018: 0x00000001,  # Common dout
    0x0500: 0x00000000,  # L2 padding mode and source ReLU
    0x0508: 0x00000000,  # L2 source base address
    0x0900: 0x00000000,  # PE pool mode, op, clamps, sources
    0x0904: 0x3C000000,  # PE scale 1.0, bias 0.0
    0x0908: 0x3C000000,  # PE pre-scale 1.0, reduction epsilon 0.0
    0x0910: 0x7C000000,  # PE clamp [0, +inf]
    0x0D04: 0x00000000,  # NE op, kernel mode, activation
}

__all__ = ["DEFAULT_IMAGE"]
