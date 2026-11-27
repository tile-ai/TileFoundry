"""Derive immutable descriptor headers from stream position and engine usage."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from tilefoundry_apple.ane import isa
from tilefoundry_apple.ane.isa import fields

POSITION_MASKS: Mapping[str, tuple[int, int, int]] = MappingProxyType(
    {
        "single": (0x2A, 0x00FFF868, 0),
        "first": (0x08, 0x00FFF868, 0),
        "middle": (0x00, 0x00FFF800, 0),
        "last": (0x22, 0x00FFF868, 0),
    }
)
ENGINE_ENABLES = MappingProxyType({0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4})
ENGINE_NODES = 16
STATUS_BIT = 1 << 0


def engine_enables(nodes: int) -> int:
    """Return the engine-enable code for ``nodes`` active engine nodes."""
    if nodes <= 0:
        return 0
    if nodes > 8:
        return 5
    return ENGINE_ENABLES[nodes]


def position_of(index: int, count: int) -> str:
    """Name the stream position of descriptor ``index`` of ``count``."""
    if count == 1:
        return "single"
    if index == 0:
        return "first"
    if index == count - 1:
        return "last"
    return "middle"


def task_type_of(td) -> int | None:
    """Read the task type a descriptor selects, if it states one."""
    word = td.writes().get(fields.Common.task_type.addr)
    return None if word is None else int(fields.Common.task_type.get(word))


DEPENDENCY_BIT = 1 << 1


def header(index: int, count: int, td, *, nodes: int = ENGINE_NODES) -> dict[int, int]:
    """Return the header words for one descriptor of a stream."""
    position = position_of(index, count)
    log_mask, debug_mask, dram_word = POSITION_MASKS[position]
    task = task_type_of(td)
    active = nodes if task == fields.TaskType.NE else 0
    control = STATUS_BIT | (engine_enables(active) << 16)
    words = {
        0x04: 0,
        0x08: log_mask,
        0x0C: 0,
        0x10: debug_mask,
        0x14: 0,
        0x18: dram_word,
        0x1C: control,
    }

    if fields.DEPENDENCY_ID.addr in td.header:
        words[0x1C] = control | DEPENDENCY_BIT
        words[fields.DEPENDENCY_ID.addr] = td.header[fields.DEPENDENCY_ID.addr]
    return words


def apply(program, *, nodes: int = ENGINE_NODES):
    """Replace every descriptor header with the header for its stream position."""
    count = len(program.tds)
    return isa.Program(
        tuple(
            replace(td, header=header(index, count, td, nodes=nodes))
            for index, td in enumerate(program.tds)
        )
    )


__all__ = [
    "ENGINE_NODES",
    "ENGINE_ENABLES",
    "POSITION_MASKS",
    "apply",
    "engine_enables",
    "header",
    "position_of",
    "task_type_of",
]
