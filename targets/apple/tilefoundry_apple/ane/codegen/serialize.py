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

from . import header

#: Words that carry an operand value and must be restated whenever a descriptor
#: reprograms them, even if the raw value is unchanged.
STICKY = frozenset(
    field.addr for field in (isa.PE.scale, isa.PE.pre_scale, isa.PE.final_scale, isa.PE.clamp_low)
)


class State:
    """The configuration image the next descriptor starts from."""

    def __init__(self, defaults_image: Mapping[int, int] | None = None) -> None:
        self.defaults = dict(defaults.DEFAULT_IMAGE if defaults_image is None else defaults_image)
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


__all__ = ["STICKY", "State", "to_program"]
