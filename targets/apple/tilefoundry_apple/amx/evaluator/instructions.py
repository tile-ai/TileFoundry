"""Logical AMX fragment adapters to the sole Python instruction semantics.

Physical register state is internal scratch. Public operands remain typed logical
fragments; their instruction placement describes byte locations, not arithmetic
conversions. This module never loads a C oracle or dispatches native AMX.
"""
from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.visitor_registry.instruction import call_params
from tilefoundry.evaluator.value import to_torch_dtype
from tilefoundry.utils.lazy import torch
from .state import AMXState
from . import (
    emulate_AMX_SET, emulate_AMX_CLR,
    emulate_AMX_LDX, emulate_AMX_LDY, emulate_AMX_LDZ, emulate_AMX_LDZI,
    emulate_AMX_STX, emulate_AMX_STY, emulate_AMX_STZ, emulate_AMX_STZI,
    emulate_AMX_FMA16, emulate_AMX_FMA32, emulate_AMX_FMA64,
    emulate_AMX_FMS16, emulate_AMX_FMS32, emulate_AMX_FMS64,
    emulate_AMX_MAC16, emulate_AMX_MATFP, emulate_AMX_VECFP,
    emulate_AMX_MATINT, emulate_AMX_VECINT,
    emulate_AMX_EXTRX, emulate_AMX_EXTRY, emulate_AMX_GENLUT,
)
from ..ir.tir.common import check_types, encode, placements
from ..ir.tir import AMX_OPS


def _byte_addresses(binding, count, nbytes):
    """Expand typed element starts, with genuine circular X/Y addressing."""
    bank = binding['bank']
    offsets = tuple(binding['offsets'])
    if len(offsets) != count:
        raise ValueError('AMX placement element count differs from logical operand')
    for start in offsets:
        if not isinstance(start, int):
            raise TypeError('AMX placement offsets must be integer byte addresses')
        for lane in range(nbytes):
            address = start + lane
            yield address % 512 if bank in ('x', 'y') else address


def _buffers(state):
    return {
        'x': state.x, 'y': state.y, 'z': state.z, 'memory': state.memory,
        'lifecycle': bytearray([int(state.enabled)]),
    }


def evaluate(ctx):
    op = ctx.op
    types = tuple(value.type for value in ctx.args)
    check_types(op, types)
    bindings = placements(op, types)
    params = call_params(op, len(ctx.args))
    roles = tuple(p.name for p in params)
    values = dict(zip(roles, ctx.args, strict=True))
    if set(bindings) != set(roles):
        raise ValueError('AMX placements must describe every logical operand')
    state = AMXState(version=op.generation)
    banks = _buffers(state)
    assigned = {}
    addresses = {}
    read_roles = {p.name for p in params if p.effect & MemoryEffect.READ}
    for role, value in values.items():
        if value.data.device.type != 'cpu':
            raise ValueError('AMX Python evaluation requires CPU logical operands')
        if value.data.dtype != to_torch_dtype(value.type.dtype):
            raise TypeError('AMX tensor data dtype differs from declared arithmetic dtype')
        nbytes = value.type.dtype.bit_width // 8
        locations = tuple(_byte_addresses(bindings[role], value.data.numel(), nbytes))
        bank_name = bindings[role]['bank']
        if bank_name not in banks:
            raise ValueError(f'AMX placement selects unknown bank {bank_name}')
        bank = banks[bank_name]
        raw = value.data.contiguous().view(torch.uint8).reshape(-1).tolist()
        if len(raw) != len(locations):
            raise ValueError('AMX placement byte count differs from logical operand')
        for address, byte in zip(locations, raw, strict=True):
            if not 0 <= address < len(bank):
                raise IndexError('AMX logical fragment lies outside scratch capacity')
            if role not in read_roles:
                continue
            key = (bank_name, address)
            if key in assigned and assigned[key] != byte:
                raise ValueError(f'AMX operands disagree at overlapping physical byte {key}')
            assigned[key] = byte
            bank[address] = byte
        addresses[role] = locations
    state.enabled = bool(banks['lifecycle'][0])
    # The one semantic implementation executes exactly one decoded instruction.
    globals()['emulate_AMX_' + op.mnemonic](state, encode(op))
    banks = _buffers(state)
    writes = tuple(op.writes)
    for role in writes:
        value = values[role]
        bank = banks[bindings[role]['bank']]
        raw = [bank[address] for address in addresses[role]]
        data = torch.tensor(raw, dtype=torch.uint8).view(value.data.dtype).reshape(value.data.shape)
        value.data.copy_(data)
    return values[writes[0]] if writes else None


for _op in AMX_OPS:
    register_schedule_eval(_op)(evaluate)
