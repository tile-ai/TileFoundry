"""Byte-addressed state for the AMX C-reference translation."""
from dataclasses import dataclass, field

@dataclass
class AMXState:
    x: bytearray = field(default_factory=lambda: bytearray(512))
    y: bytearray = field(default_factory=lambda: bytearray(512))
    z: bytearray = field(default_factory=lambda: bytearray(4096))
    version: int = 2
    memory: bytearray = field(default_factory=lambda: bytearray(4096))
    enabled: bool = False

    def __post_init__(self):
        self.x, self.y, self.z, self.memory = map(bytearray, (self.x, self.y, self.z, self.memory))
        if (len(self.x), len(self.y), len(self.z)) != (512, 512, 4096):
            raise ValueError("AMX register banks must have 512, 512, and 4096 bytes")

    def clone(self):
        return AMXState(self.x, self.y, self.z, self.version, self.memory, self.enabled)

    def to_bytes(self):
        return bytes(self.x + self.y + self.z)


def emulate_AMX_SET(state, operand=0):
    if state.enabled:
        raise RuntimeError("AMX is already enabled")
    state.x[:] = bytes(512)
    state.y[:] = bytes(512)
    state.z[:] = bytes(4096)
    state.enabled = True


def emulate_AMX_CLR(state, operand=0):
    state.enabled = False
