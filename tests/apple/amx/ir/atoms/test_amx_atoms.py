"""Representative public AMX Op cases against one independent C simulator."""

import json
from collections import Counter

import pytest

from tilefoundry.evaluator.context import EvaluateContext
from tilefoundry.evaluator.registry import schedule_eval_registry
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.param_def import collect_param_defs
from tilefoundry.ir.pattern import SwitchPattern
from tilefoundry.ir.pattern.utils import fixed_pattern_value, selected_pattern
from tilefoundry.ir.types import DType, Layout, StorageKind, TensorType, UnitType
from tilefoundry.visitor_registry.instruction import call_params
from tilefoundry.visitor_registry.typeinfer import inference_type
from tilefoundry_apple.amx.ir.tir import AMX_OPS
from tilefoundry_apple.amx.ir.tir.common import (
    ExtractMode,
    FPMode,
    IndexMode,
    IntALU,
    IntLane,
    LutMode,
    ProductMode,
    Repeat,
    TransferCount,
)

from ._c_reference import OP_NAMES, _c_operand, _prepare, _tensor_bytes


def _same(actual, expected, label, case):
    if actual != expected:
        offsets = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        first = offsets[0] if offsets else min(len(actual), len(expected))
        pytest.fail(
            f"{case}: {label} differs at byte {first}; actual={actual[first : first + 16].hex()} expected={expected[first : first + 16].hex()}; differing bytes={len(offsets)}"
        )


def _bound_type(param, bindings):
    """Resolve one actual instance's operand, like WGMMA's _bound_fragment."""

    def value(item):
        if isinstance(item, (DType, StorageKind)):
            return item
        if isinstance(item, SwitchPattern):
            return value(dict(item.branches)[bindings[item.param]])
        return fixed_pattern_value(item, bindings)

    pattern = selected_pattern(param.pattern, bindings)
    layout = selected_pattern(pattern.layout, bindings)
    result = TensorType(
        shape=fixed_pattern_value(pattern.shape, bindings),
        dtype=value(pattern.dtype),
        storage=value(pattern.storage),
        layout=Layout(
            fixed_pattern_value(layout.shape, bindings),
            fixed_pattern_value(layout.strides, bindings),
        ),
    )
    assert pattern.match(result, dict(bindings)) is not None, (param.name, bindings, result)
    return result


_OPS = {op.__name__: op for op in AMX_OPS}
_BASE_CASES = tuple(
    (op, {"mode": LutMode.F32, "dtype": DType.f32} if op.mnemonic == "GENLUT" else {})
    for op in AMX_OPS
)
_EXTRA_CASES = (
    ("Fma16", {"acc_dtype": DType.f16}),
    ("Fma16", {"mode": ProductMode.VECTOR, "acc_dtype": DType.f16}),
    ("Fma32", {"x_dtype": DType.f16, "y_dtype": DType.f16}),
    ("Fma64", {"mode": ProductMode.VECTOR}),
    ("Fms16", {"mode": ProductMode.VECTOR, "acc_dtype": DType.f16, "skip_y": True, "skip_z": True}),
    ("Fms32", {"mode": ProductMode.VECTOR, "skip_x": True, "skip_z": True}),
    (
        "Mac16",
        {
            "mode": ProductMode.VECTOR,
            "acc_dtype": DType.i16,
            "x_dtype": DType.i8,
            "y_dtype": DType.i8,
        },
    ),
    ("LdxV2", {"count": TransferCount.FOUR}),
    ("LdxV3", {"count": TransferCount.FOUR, "nonconsecutive": True}),
    ("LdyV3", {"count": TransferCount.TWO, "nonconsecutive": True}),
    (
        "ExtrxV2",
        {
            "mode": ExtractMode.FLOAT,
            "src_dtype": DType.f32,
            "dst_dtype": DType.f16,
            "repeat": Repeat.TWO,
        },
    ),
    (
        "ExtryV3",
        {
            "mode": ExtractMode.INTEGER,
            "src_dtype": DType.i32,
            "dst_dtype": DType.i16,
            "repeat": Repeat.FOUR,
            "rounding": True,
            "saturate": True,
            "destination_offset": 73,
        },
    ),
    ("MatfpV2", {"fp_mode": FPMode.BF16_F32, "dtype": DType.bf16, "acc_dtype": DType.f32}),
    (
        "VecfpV2",
        {
            "fp_mode": FPMode.F16_F32,
            "dtype": DType.f16,
            "acc_dtype": DType.f32,
            "index_mode": IndexMode.X2,
        },
    ),
    ("VecfpV3", {"repeat": Repeat.FOUR, "broadcast_mode": 2, "x_offset": 73, "y_offset": 177}),
    (
        "MatintV2",
        {
            "alu": IntALU.WIDE_PRODUCT,
            "lane_mode": IntLane.I8_I16_I32,
            "x_dtype": DType.i8,
            "y_dtype": DType.i16,
            "acc_dtype": DType.i32,
        },
    ),
    (
        "VecintV2",
        {
            "lane_mode": IntLane.I8_I8_I32,
            "x_dtype": DType.i8,
            "y_dtype": DType.i8,
            "acc_dtype": DType.i32,
        },
    ),
    ("VecintV3", {"repeat": Repeat.FOUR, "x_offset": 73, "y_offset": 177}),
    ("GenlutV2", {"mode": LutMode.LOOKUP16_5, "dtype": DType.bf16}),
    ("GenlutV2", {"mode": LutMode.F16, "bf16": True, "dtype": DType.bf16}),
)
CASES = _BASE_CASES + tuple((_OPS[name], attrs) for name, attrs in _EXTRA_CASES)


def _check_case(declaration, attributes, c_oracle):
    """Construct one authored parameter case and compare all declared writes."""
    op = declaration(**attributes)
    inputs = tuple(p for p in collect_param_defs(declaration) if p.kind == "input")
    active = tuple(p for p in inputs if selected_pattern(p.pattern, op.bindings) is not None)
    assert tuple(p.name for p in call_params(op, len(active))) == tuple(p.name for p in active)
    types = tuple(_bound_type(param, op.bindings) for param in active)
    expressions = tuple(Var(type=t, name=p.name) for p, t in zip(active, types, strict=True))
    call = Call(type=UnitType(), target=op, args=expressions)
    assert isinstance(inference_type(call), UnitType)
    expected, banks, locations, args = _prepare(op, active, types)
    before = tuple(_tensor_bytes(value.data) for value in args)
    case = f"{declaration.__name__}({attributes!r})"
    lifecycle = op.mnemonic in ("SET", "CLR")
    if not lifecycle:
        scratch_before = {bank: bytes(data) for bank, data in banks.items()}
        c_oracle.execute(op.mnemonic, expected, _c_operand(op))
        declared_writes = {bank: set() for bank in banks}
        for role in op.writes:
            bank, addresses = locations[role]
            declared_writes[bank].update(addresses)
        for bank, data in banks.items():
            escaped = [
                address
                for address, (old, new) in enumerate(zip(scratch_before[bank], data, strict=True))
                if old != new and address not in declared_writes[bank]
            ]
            assert not escaped, (case, bank, escaped[:8])
    else:
        banks["lifecycle"][0] = int(op.mnemonic == "SET")
    output_index = tuple(p.name for p in active).index(op.writes[0])
    output_type = types[output_index]
    handler = schedule_eval_registry.lookup(declaration)
    assert handler is not None, case
    result = handler(EvaluateContext(op=op, args=args, result_type=output_type))
    assert isinstance(result, TensorValue) and result.type == output_type, case
    for index, (param, value) in enumerate(zip(active, args, strict=True)):
        if param.name in op.writes:
            bank, addresses = locations[param.name]
            wanted = bytearray(banks[bank][address] for address in addresses)
        else:
            wanted = before[index]
        _same(_tensor_bytes(value.data), wanted, param.name, case)
    _same(
        _tensor_bytes(result.data), _tensor_bytes(args[output_index].data), "returned output", case
    )
    return lifecycle


def test_amx_atoms(c_oracle, record_property):
    """Keep 59 explicit cases: every static Op plus twenty representative modes."""
    assert len(CASES) == 59
    assert {op.mnemonic for op in AMX_OPS} == set(OP_NAMES) | {"SET", "CLR"}
    counts = Counter()
    lifecycle_count = 0
    for declaration, attributes in CASES:
        lifecycle_count += _check_case(declaration, attributes, c_oracle)
        counts[declaration.__name__] += 1
    report = {
        "case_count": len(CASES),
        "c_reference_count": len(CASES) - lifecycle_count,
        "lifecycle_count": lifecycle_count,
        "per_op_counts": dict(counts),
    }
    for key, value in report.items():
        record_property("amx_" + key, json.dumps(value, sort_keys=True))
    print("AMX_ATOM_COVERAGE " + json.dumps(report, sort_keys=True))
