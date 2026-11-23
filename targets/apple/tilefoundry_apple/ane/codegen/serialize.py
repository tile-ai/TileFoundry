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
from tilefoundry_apple.ane.isa import defaults

#: Words that carry an operand value and must be restated whenever a descriptor
#: reprograms them, even if the raw value is unchanged.
STICKY = frozenset(
    field.addr
    for field in (isa.PE.scale, isa.PE.pre_scale, isa.PE.final_scale, isa.PE.clamp_low)
)

#: Words that a descriptor restates whenever its image differs at all.
VOLATILE = frozenset(
    addr
    for block in isa.CONFIG_BLOCKS
    for addr in range(block.base, block.end, 4)
    if addr not in STICKY
)


class State:
    """The configuration image the next descriptor starts from."""

    def __init__(self, defaults_image: Mapping[int, int] | None = None) -> None:
        self.image: dict[int, int] = {}
        self.defaults = dict(defaults.DEFAULT_IMAGE if defaults_image is None else defaults_image)
        self.tds: list[isa.TD] = []

    def emit(self, image: Mapping[int, int]) -> isa.TD:
        """Append one descriptor, eliding words that already hold their value."""
        base = dict(self.defaults)
        base.update({addr: word for addr, word in self.image.items() if addr not in VOLATILE})
        td = isa.TD.build(dict(image), base=base)
        if td.size_words() > isa.MAX_TASK_WORDS:
            raise ValueError(f"ANE task_size {td.size_words()} exceeds {isa.MAX_TASK_WORDS} words")
        self.image.update(td.writes())
        self.tds.append(td)
        return td


def to_program(parts: Iterable[Mapping[int, int]]) -> isa.Program:
    """Bundle one engine pass into a single-descriptor program."""
    state = State()
    state.emit(isa.splice(*parts))
    return isa.Program(tuple(state.tds))


__all__ = ["STICKY", "VOLATILE", "State", "to_program"]
