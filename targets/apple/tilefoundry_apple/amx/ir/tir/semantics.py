"""Type inference, access, cost and Python evaluation shared by the AMX Ops.

Logical AMX fragment adapters to the sole Python instruction semantics.

Physical register state is internal scratch. Public operands remain typed logical
fragments; their instruction placement describes byte locations, not arithmetic
conversions. This module never loads a C oracle or dispatches native AMX.
"""

from tilefoundry import utils
from tilefoundry.interfaces import access, cost, evaluator, schedule, typeinfer
from tilefoundry.ir import expr
from tilefoundry_apple.amx import evaluator as amx_evaluator

from .common import access as common_access
from .common import check_types, encode, infer, placements
from .common import cost as common_cost


def byte_addresses(binding, count, nbytes):
    """Expand typed element starts, with genuine circular X/Y addressing."""
    bank = binding["bank"]
    offsets = tuple(binding["offsets"])
    if len(offsets) != count:
        raise ValueError("AMX placement element count differs from logical operand")
    for start in offsets:
        if not isinstance(start, int):
            raise TypeError("AMX placement offsets must be integer byte addresses")
        for lane in range(nbytes):
            address = start + lane
            yield address % 512 if bank in ("x", "y") else address


def buffers(state):
    return {
        "x": state.x,
        "y": state.y,
        "z": state.z,
        "memory": state.memory,
        "lifecycle": bytearray([int(state.enabled)]),
    }


def evaluate(ctx):
    op = ctx.op
    types = tuple(value.type for value in ctx.args)
    check_types(op, types)
    bindings = placements(op, types)
    params = schedule.call_params(op, len(ctx.args))
    roles = tuple(p.name for p in params)
    values = dict(zip(roles, ctx.args, strict=True))
    if set(bindings) != set(roles):
        raise ValueError("AMX placements must describe every logical operand")
    state = amx_evaluator.AMXState(version=op.generation)
    banks = buffers(state)
    assigned = {}
    addresses = {}
    read_roles = {p.name for p in params if p.effect & expr.MemoryEffect.READ}
    for role, value in values.items():
        if value.data.device.type != "cpu":
            raise ValueError("AMX Python evaluation requires CPU logical operands")
        if value.data.dtype != evaluator.to_torch_dtype(value.type.dtype):
            raise TypeError("AMX tensor data dtype differs from declared arithmetic dtype")
        nbytes = value.type.dtype.bit_width // 8
        locations = tuple(byte_addresses(bindings[role], value.data.numel(), nbytes))
        bank_name = bindings[role]["bank"]
        if bank_name not in banks:
            raise ValueError(f"AMX placement selects unknown bank {bank_name}")
        bank = banks[bank_name]
        raw = value.data.contiguous().view(utils.torch.uint8).reshape(-1).tolist()
        if len(raw) != len(locations):
            raise ValueError("AMX placement byte count differs from logical operand")
        for address, byte in zip(locations, raw, strict=True):
            if not 0 <= address < len(bank):
                raise IndexError("AMX logical fragment lies outside scratch capacity")
            if role not in read_roles:
                continue
            key = (bank_name, address)
            if key in assigned and assigned[key] != byte:
                raise ValueError(f"AMX operands disagree at overlapping physical byte {key}")
            assigned[key] = byte
            bank[address] = byte
        addresses[role] = locations
    state.restore_enabled(bool(banks["lifecycle"][0]))
    getattr(amx_evaluator, "emulate_amx_" + op.mnemonic.lower())(state, encode(op))
    banks = buffers(state)
    writes = tuple(op.writes)
    for role in writes:
        value = values[role]
        bank = banks[bindings[role]["bank"]]
        raw = [bank[address] for address in addresses[role]]
        data = utils.torch.tensor(raw, dtype=utils.torch.uint8).view(value.data.dtype).reshape(value.data.shape)
        value.data.copy_(data)
    return values[writes[0]] if writes else None


def register_semantics(ops):
    """Register inference, access, cost and evaluation for the Ops of one module."""
    for op in ops:
        typeinfer.register(op)(infer)
        access.register_relation(op)(common_access)
        cost.register(op)(common_cost)
        evaluator.register(op)(evaluate)


__all__ = ["evaluate", "register_semantics"]
