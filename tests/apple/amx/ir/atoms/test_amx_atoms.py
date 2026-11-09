"""Representative public AMX Op cases against one independent C simulator."""

import json
from collections import Counter

import pytest

from tilefoundry.interfaces import evaluator, schedule, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry_apple.amx.ir import tir

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
        if isinstance(item, (expr.DType, expr.StorageKind)):
            return item
        if isinstance(item, pattern.Switch):
            return value(dict(item.branches)[bindings[item.param]])
        return pattern.fixed_value(item, bindings)

    declared = pattern.selected(param.pattern, bindings)
    layout = pattern.selected(declared.layout, bindings)
    result = types.TensorType(
        shape=pattern.fixed_value(declared.shape, bindings),
        dtype=value(declared.dtype),
        storage=value(declared.storage),
        layout=types.Layout(
            pattern.fixed_value(layout.shape, bindings),
            pattern.fixed_value(layout.strides, bindings),
        ),
    )
    assert declared.match(result, dict(bindings)) is not None, (param.name, bindings, result)
    return result


_OPS = {op.__name__: op for op in tir.AMX_OPS}
_BASE_CASES = tuple(
    (op, {"mode": tir.LutMode.F32, "dtype": expr.DType.f32} if op.mnemonic == "GENLUT" else {})
    for op in tir.AMX_OPS
)
_EXTRA_CASES = (
    ("Fma16", {"acc_dtype": expr.DType.f16}),
    ("Fma16", {"mode": tir.ProductMode.VECTOR, "acc_dtype": expr.DType.f16}),
    ("Fma32", {"x_dtype": expr.DType.f16, "y_dtype": expr.DType.f16}),
    ("Fma64", {"mode": tir.ProductMode.VECTOR}),
    ("Fms16", {"mode": tir.ProductMode.VECTOR, "acc_dtype": expr.DType.f16, "skip_y": True, "skip_z": True}),
    ("Fms32", {"mode": tir.ProductMode.VECTOR, "skip_x": True, "skip_z": True}),
    (
        "Mac16",
        {
            "mode": tir.ProductMode.VECTOR,
            "acc_dtype": expr.DType.i16,
            "x_dtype": expr.DType.i8,
            "y_dtype": expr.DType.i8,
        },
    ),
    ("LdxV2", {"count": tir.TransferCount.FOUR}),
    ("LdxV3", {"count": tir.TransferCount.FOUR, "nonconsecutive": True}),
    ("LdyV3", {"count": tir.TransferCount.TWO, "nonconsecutive": True}),
    (
        "ExtrxV2",
        {
            "mode": tir.ExtractMode.FLOAT,
            "src_dtype": expr.DType.f32,
            "dst_dtype": expr.DType.f16,
            "repeat": tir.Repeat.TWO,
        },
    ),
    (
        "ExtryV3",
        {
            "mode": tir.ExtractMode.INTEGER,
            "src_dtype": expr.DType.i32,
            "dst_dtype": expr.DType.i16,
            "repeat": tir.Repeat.FOUR,
            "rounding": True,
            "saturate": True,
            "destination_offset": 73,
        },
    ),
    ("MatfpV2", {"fp_mode": tir.FPMode.BF16_F32, "dtype": expr.DType.bf16, "acc_dtype": expr.DType.f32}),
    (
        "VecfpV2",
        {
            "fp_mode": tir.FPMode.F16_F32,
            "dtype": expr.DType.f16,
            "acc_dtype": expr.DType.f32,
            "index_mode": tir.IndexMode.X2,
        },
    ),
    ("VecfpV3", {"repeat": tir.Repeat.FOUR, "broadcast_mode": 2, "x_offset": 73, "y_offset": 177}),
    (
        "MatintV2",
        {
            "alu": tir.IntALU.WIDE_PRODUCT,
            "lane_mode": tir.IntLane.I8_I16_I32,
            "x_dtype": expr.DType.i8,
            "y_dtype": expr.DType.i16,
            "acc_dtype": expr.DType.i32,
        },
    ),
    (
        "VecintV2",
        {
            "lane_mode": tir.IntLane.I8_I8_I32,
            "x_dtype": expr.DType.i8,
            "y_dtype": expr.DType.i8,
            "acc_dtype": expr.DType.i32,
        },
    ),
    ("VecintV3", {"repeat": tir.Repeat.FOUR, "x_offset": 73, "y_offset": 177}),
    ("GenlutV2", {"mode": tir.LutMode.LOOKUP16_5, "dtype": expr.DType.bf16}),
    ("GenlutV2", {"mode": tir.LutMode.F16, "bf16": True, "dtype": expr.DType.bf16}),
)
CASES = _BASE_CASES + tuple((_OPS[name], attrs) for name, attrs in _EXTRA_CASES)


def _check_case(declaration, attributes, c_oracle):
    """Construct one authored parameter case and compare all declared writes."""
    op = declaration(**attributes)
    inputs = tuple(p for p in expr.collect_param_defs(declaration) if p.kind == "input")
    active = tuple(p for p in inputs if pattern.selected(p.pattern, op.bindings) is not None)
    assert tuple(p.name for p in schedule.call_params(op, len(active))) == tuple(p.name for p in active)
    type_list = tuple(_bound_type(param, op.bindings) for param in active)
    expressions = tuple(expr.Var(type=t, name=p.name) for p, t in zip(active, type_list, strict=True))
    call = expr.Call(type=types.UnitType(), target=op, args=expressions)
    assert isinstance(typeinfer.inference_type(call), types.UnitType)
    expected, banks, locations, args = _prepare(op, active, type_list)
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
    output_type = type_list[output_index]
    handler = evaluator.registry.get(declaration, None)
    assert handler is not None, case
    result = handler(evaluator.EvaluateContext(op=op, args=args, result_type=output_type))
    assert isinstance(result, evaluator.TensorValue) and result.type == output_type, case
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
    assert {op.mnemonic for op in tir.AMX_OPS} == set(OP_NAMES) | {"SET", "CLR"}
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
