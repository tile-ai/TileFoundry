"""Fill the per-descriptor header words from the descriptor's stream position.

The header carries three kinds of state that the configuration packets do not:

``0x04``  an estimated cycle count for the task;
``0x08``  the log-event mask, and ``0x10`` its debug counterpart;
``0x18``  the DRAM log-event word;
``0x1C``  the control flags, whose bits 16..18 enable engine nodes.

``0x08``, ``0x10`` and ``0x18`` are masks of *event bits*: the compiler turns an
event on by setting its bit, and it turns the per-task boundary events on only
for the first and the last descriptor of a stream (a lone descriptor carries
both).  The control word keeps its task-status bit and, for a task that runs on
the convolution engine, the engine-enable code; tasks that run on the planar
engine leave the engine-enable field clear.

The engine-enable code is a function of how many engine nodes the task uses:
1..8 nodes map to ``3``..``4`` and more than eight nodes map to ``5``, while a
task that uses none leaves the field at ``0``.

Two header words are deliberately left at zero until their source is derived
from the compiler's own arithmetic instead of from sampled outputs:

``0x04``  the task's estimated cycle count.  The compiler returns zero for a task
          whose input chain lives in L2 and otherwise truncates a per-layer
          estimate to 16 bits, so a real value needs the cost model that produces
          that estimate.
``0x18``  the DRAM log-event mask.  Compiled images carry ``2**k - 1`` on every
          convolution or matrix task except the last of a stream, and zero
          elsewhere; ``k`` follows the network rather than the descriptor, so the
          quantity that indexes it is still unknown.
"""

from collections.abc import Mapping

from tilefoundry_apple.ane.isa import fields
from tilefoundry_apple.ane.isa.td import DEPENDENCY_ID

#: Log and debug mask words for the four stream positions.
POSITION_MASKS: Mapping[str, tuple[int, int, int]] = {
    # position -> (log-event mask, debug-log-event mask, dram log word)
    "single": (0x2A, 0x00FFF868, 0),
    "first": (0x08, 0x00FFF868, 0),
    "middle": (0x00, 0x00FFF800, 0),
    "last": (0x22, 0x00FFF868, 0),
}

#: Observed variant of the debug mask (``0x68`` instead of ``0xFFF868``).  The
#: value selects which event groups the layer enables, so it follows the layer
#: kind rather than the stream position; it is a telemetry field that the loader
#: rewrites, so it is left to the positional default until the event selector is
#: reproduced.
DEBUG_MASK_WITHOUT_GROUPS = 0x68

#: Engine-enable code per active node count.
ENGINE_ENABLES = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4}

#: Default node count for a task that runs on the convolution engine.  The
#: compiler derives it from the rasterization of the tile; until that pass is
#: reproduced the codegen passes the count in explicitly.
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


#: Registers that carry a coefficient base address (paired and single-word forms).
COEFF_SLOT_ADDRS = frozenset({0x1908, 0x19F8, 0x1A00, 0x19E8, 0x19EC, 0x19F4})


def loads_coefficients(td) -> bool:
    """Report whether the descriptor addresses the coefficient blob.

    The kernel-DMA block is also used for the stream-armed word, so the test is
    the presence of a coefficient *base address*, not of the block itself.
    """
    return any(
        addr in COEFF_SLOT_ADDRS for packet in td.packets for addr, _ in packet.writes
    )


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
        0x04: 0,  # estimated cycles: deferred,
        0x08: log_mask,
        0x0C: 0,
        0x10: debug_mask,
        0x14: 0,
        0x18: dram_word,  # DRAM log mask: deferred (see the module docstring),
        0x1C: control,
    }
    # A descriptor that waits on another task states the wait in bit 1 and adds
    # the waited-for id as a ninth header word; the policy must not drop it.
    if DEPENDENCY_ID.addr in td.header:
        words[0x1C] = control | DEPENDENCY_BIT
        words[DEPENDENCY_ID.addr] = td.header[DEPENDENCY_ID.addr]
    return words


def apply(program, *, nodes: int = ENGINE_NODES):
    """Replace every descriptor header with the header for its stream position."""
    count = len(program.tds)
    for index, td in enumerate(program.tds):
        td.header = header(index, count, td, nodes=nodes)
    return program


__all__ = ["ENGINE_NODES", "ENGINE_ENABLES", "DEBUG_MASK_WITHOUT_GROUPS", "POSITION_MASKS",
           "apply", "engine_enables", "header", "loads_coefficients", "position_of", "task_type_of"]
