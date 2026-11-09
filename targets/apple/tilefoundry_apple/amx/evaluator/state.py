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

    def enable(self):
        """Clear every register and mark the engine enabled."""
        if self.enabled:
            raise RuntimeError("AMX is already enabled")
        self.x[:] = bytes(512)
        self.y[:] = bytes(512)
        self.z[:] = bytes(4096)
        self.enabled = True

    def disable(self):
        self.enabled = False

    def restore_enabled(self, enabled):
        self.enabled = enabled

    def copy_register(self, source, source_index, target, target_index):
        """Copy one 64-byte register between the X and Y banks."""
        banks = {"x": self.x, "y": self.y}
        banks[target][target_index * 64 : (target_index + 1) * 64] = banks[source][
            source_index * 64 : (source_index + 1) * 64
        ]


def emulate_amx_set(state, operand=0):
    state.enable()


def emulate_amx_clr(state, operand=0):
    state.disable()


__all__ = ["AMXState", "emulate_amx_clr", "emulate_amx_set"]
