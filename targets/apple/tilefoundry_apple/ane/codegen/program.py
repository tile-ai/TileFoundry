"""Bundle intrinsic register maps into immutable descriptors using the reset image."""

from collections.abc import Iterable, Mapping

from tilefoundry_apple.ane import isa

from . import header

DEFAULT_IMAGE = isa.RESET_IMAGE


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
        td = isa.TD.from_values(dict(image), base=base, tid=len(self.tds))
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
    return header.apply(isa.Program(tuple(state.tds)))


__all__ = ["DEFAULT_IMAGE", "State", "to_program"]
