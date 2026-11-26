"""Serialize intrinsic words into task descriptors.

Packet framing (RUN/MASK) is the ISA layer's job; this module decides which words
a descriptor states and in which descriptor they land.

One descriptor is one VLIW instruction: the load, the arithmetic and the store of
a single engine pass share one configuration image, so the serializer bundles the
words of that pass into a single task descriptor.  A pass that does not fit the
descriptor word limit is split into consecutive descriptors by the caller.

The hardware configuration is persistent between descriptors, so a word is only
emitted when it differs from what the previous descriptor left behind, and the
first descriptor is compared against the default image.
"""

from collections.abc import Iterable, Mapping

from tilefoundry_apple.ane import isa

from . import header

#: Words that carry an operand value and must be restated whenever a descriptor
#: reprograms them, even if the raw value is unchanged.
STICKY = frozenset(
    field.addr for field in (isa.PE.scale, isa.PE.pre_scale, isa.PE.final_scale, isa.PE.clamp_low)
)

#: The configuration image every descriptor starts from.  The hardware seeds the
#: register file with this per-architecture default before any layer programs it,
#: and only words whose value differs from it are emitted -- which reproduces the
#: compiler's word elision (a descriptor never restates ``din = 1``, ``scale = 1``,
#: an all-zero L2 source word, ...).  ``0x003C`` is deliberately absent: the
#: compiler writes it in every descriptor even at its default, so it is never
#: elided.  Only the words this target's emitters can produce are listed.
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


class State:
    """The configuration image the next descriptor starts from."""

    def __init__(self, defaults_image: Mapping[int, int] | None = None) -> None:
        self.defaults = dict(DEFAULT_IMAGE if defaults_image is None else defaults_image)
        self.tds: list[isa.TD] = []

    def emit(self, image: Mapping[int, int]) -> isa.TD:
        """Append one descriptor, eliding words that hold their reset value.

        Each descriptor is encoded from the reset image alone: a later descriptor
        repeats whatever its own handlers set, even when the previous descriptor
        already carried the same value.  Carrying values forward would emit fewer
        words than the compiler does, so no state is inherited across descriptors.
        """
        base = dict(self.defaults)
        td = isa.TD.build(dict(image), base=base)
        if td.size_words() > isa.MAX_TASK_WORDS:
            raise ValueError(f"ANE task_size {td.size_words()} exceeds {isa.MAX_TASK_WORDS} words")
        self.tds.append(td)
        return td


def to_program(passes: Iterable[Iterable[Mapping[int, int]]]) -> isa.Program:
    """Bundle one descriptor per pass, in schedule order."""
    state = State()
    todo = [isa.splice(*parts) for parts in passes]
    if not todo:
        raise ValueError("the schedule emits no passes")
    todo[-1] = isa.splice(todo[-1], isa.end_of_network())
    for image in todo:
        state.emit(image)
    if not state.tds:
        raise ValueError("the schedule emits no passes")
    for index, td in enumerate(state.tds):
        td.tid = index
    return header.apply(isa.Program(tuple(state.tds)))


__all__ = ["DEFAULT_IMAGE", "STICKY", "State", "to_program"]
