"""AMX Op Pattern variants compared with the independent original C emulator."""

from dataclasses import fields, is_dataclass
import json

import pytest

from tilefoundry.evaluator.context import EvaluateContext
from tilefoundry.evaluator.registry import schedule_eval_registry
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.param_def import collect_param_defs
from tilefoundry.ir.pattern import SwitchPattern, WildcardPattern
from tilefoundry.ir.pattern.utils import fixed_pattern_value, selected_pattern, variants
from tilefoundry.ir.types import DType, Layout, StorageKind, TensorType, UnitType
from tilefoundry.visitor_registry.instruction import call_params
from tilefoundry.visitor_registry.typeinfer import inference_type
from tilefoundry_apple.amx.ir.tir import AMX_OPS

from ._c_reference import (
    OP_NAMES,
    _c_operand,
    _prepare,
    _tensor_bytes,
)


def _same(actual,expected,label,case):
    if actual!=expected:
        offsets=[i for i,(a,b) in enumerate(zip(actual,expected)) if a!=b]
        first=offsets[0] if offsets else min(len(actual),len(expected))
        pytest.fail(f'{case}: {label} differs at byte {first}; actual={actual[first:first+16].hex()} expected={expected[first:first+16].hex()}; differing bytes={len(offsets)}')


def _parameter_values(param):
    """Small physical representatives; Enum/DType domains come from patterns."""
    if param.name=='register': return (0,7,63)
    if param.annotation is bool: return (False,True)
    if param.annotation is int:
        # Modes belong to enums; raw byte offsets/masks are not 64-bit exhaustive.
        return (param.default,) if param.has_default else (0,)
    return (param.default,) if param.has_default else None

def _type_parameters(attributes,inputs):
    """Only public attributes reached by actual input Patterns and dependencies."""
    public={param.name:param for param in attributes}
    def references(root):
        found=set();seen=set()
        def walk(node):
            if id(node) in seen: return
            seen.add(id(node))
            if isinstance(node,WildcardPattern) and node.name: found.add(node.name)
            if isinstance(node,SwitchPattern): found.add(node.param)
            # Predicate arithmetic can lower a Wildcard to a variable Term.
            if getattr(node,'op',None) in ('variable','bits') and getattr(node,'args',()):
                if node.args[0]: found.add(node.args[0])
            if is_dataclass(node) and not isinstance(node,type):
                for item in fields(node): walk(getattr(node,item.name))
            elif isinstance(node,(tuple,list)):
                for item in node: walk(item)
            elif isinstance(node,dict):
                for key,value in node.items(): walk(key);walk(value)
        walk(root)
        return found
    dependencies={param.name:references(param.pattern) for param in attributes}
    providers={name:{param for param,refs in dependencies.items() if name in refs}
               for name in set().union(*dependencies.values())-public.keys()}
    pending=[]
    for param in inputs: pending.extend(references(param.pattern))
    names=set();seen=set()
    while pending:
        name=pending.pop()
        if name in seen: continue
        seen.add(name)
        if name in public:
            names.add(name);pending.extend(dependencies[name]-seen)
        else:
            pending.extend(providers.get(name,set())-seen)
    return tuple(param for param in attributes if param.name in names)

def _bound_type(param,bindings):
    """Resolve one actual instance's operand, like WGMMA's _bound_fragment."""
    def value(item):
        if isinstance(item,(DType,StorageKind)): return item
        if isinstance(item,SwitchPattern):
            return value(dict(item.branches)[bindings[item.param]])
        return fixed_pattern_value(item,bindings)
    pattern=selected_pattern(param.pattern,bindings)
    layout=selected_pattern(pattern.layout,bindings)
    result=TensorType(
        shape=fixed_pattern_value(pattern.shape,bindings),
        dtype=value(pattern.dtype),
        storage=value(pattern.storage),
        layout=Layout(fixed_pattern_value(layout.shape,bindings),
                      fixed_pattern_value(layout.strides,bindings)),
    )
    assert pattern.match(result,dict(bindings)) is not None, (param.name,bindings,result)
    return result

@pytest.mark.parametrize('declaration', AMX_OPS, ids=lambda op: op.__name__)
def test_amx_atoms(declaration, c_oracle, record_property):
    """Instantiate public parameter domains and compare actual Evaluator values."""
    assert {op.mnemonic for op in AMX_OPS}==set(OP_NAMES)|{'SET','CLR'}, 'AMX declaration inventory differs from original instruction entrypoints'
    counts={}
    domains={}
    lifecycle_count=0
    c_count=0
    # 1. Expand each Op's public parameter Patterns, as in the WGMMA atom gate.
    parameters=collect_param_defs(declaration)
    attributes=tuple(p for p in parameters if p.kind=='attribute')
    assert not {'version','amx_generation'}.intersection(p.name for p in attributes)
    inputs=tuple(p for p in parameters if p.kind=='input')
    selected=_type_parameters(attributes,inputs)
    varied={param.name for param in selected}
    domains[declaration.__name__]={
        'varied':[param.name for param in selected],
        'integer_representatives':{param.name:list(_parameter_values(param)) for param in selected if param.annotation is int},
        'defaulted':{param.name:repr(param.default) for param in attributes if param.name not in varied},
    }
    assert all(param.has_default for param in attributes if param.name not in varied), (declaration,domains[declaration.__name__])
    print('AMX_FAMILY_START '+declaration.__name__+' '+json.dumps(domains[declaration.__name__],sort_keys=True),flush=True)
    count=0
    for bindings in variants(selected,vary_defaulted=True,values=_parameter_values):
        # 2. Instantiate and type-check this exact instruction and its active inputs.
        op=declaration(**{param.name: bindings[param.name] for param in selected})
        active=[]
        for param in inputs:
            pattern=selected_pattern(param.pattern,op.bindings)
            if pattern is None:
                assert param.optional and param.has_default and param.default is None, (declaration,bindings,param.name)
            else:
                active.append(param)
        active=tuple(active)
        assert tuple(p.name for p in call_params(op,len(active)))==tuple(p.name for p in active), (declaration,bindings)
        types=tuple(_bound_type(param,op.bindings) for param in active)
        expressions=tuple(Var(type=type,name=param.name) for param,type in zip(active,types))
        call=Call(type=UnitType(),target=op,args=expressions)
        assert isinstance(inference_type(call),UnitType), (declaration,bindings)
        # 3. Pack one shared input state independently for C and the Evaluator.
        expected,banks,locations,args=_prepare(op,active,types)
        before=tuple(_tensor_bytes(value.data) for value in args)
        name=op.mnemonic
        case=f'{name}({op.bindings!r})'
        if name not in ('SET','CLR'):
            scratch_before={bank:bytes(data) for bank,data in banks.items()}
            c_oracle.execute(name,expected,_c_operand(op))
            # A too-small output Pattern must not pass by comparing only
            # the declared subset of the C instruction's actual writes.
            declared_writes={bank:set() for bank in banks}
            for role in op.writes:
                bank,addresses=locations[role]
                declared_writes[bank].update(addresses)
            for bank,data in banks.items():
                escaped=[address for address,(old,new) in enumerate(zip(scratch_before[bank],data,strict=True))
                         if old!=new and address not in declared_writes[bank]]
                assert not escaped, (f'{case}: C changed {bank} byte {escaped[0] if escaped else None} '
                                     f'outside declared WRITE fragment; escaped bytes={len(escaped)}')
            c_count+=1
        else:
            # No original C lifecycle emulator: assert logical token only.
            banks['lifecycle'][0]=int(name=='SET')
            lifecycle_count+=1
        # 4. Run the actual Evaluator, then check writes, untouched inputs and return.
        output_index=tuple(p.name for p in active).index(op.writes[0])
        output_type=types[output_index]
        handler=schedule_eval_registry.lookup(declaration)
        assert handler is not None, f'{case}: missing actual Evaluator registration'
        result=handler(EvaluateContext(op=op,args=args,result_type=output_type))
        assert isinstance(result,TensorValue), f'{case}: expected CUDA-style TensorValue'
        assert result.type==output_type, f'{case}: wrong Evaluator output type'
        for index,(param,value) in enumerate(zip(active,args)):
            actual=_tensor_bytes(value.data)
            if param.name in op.writes:
                bank,addresses=locations[param.name]
                wanted=bytearray(banks[bank][address] for address in addresses)
            else:
                wanted=before[index]
            _same(actual,wanted,param.name,case)
        _same(_tensor_bytes(result.data),_tensor_bytes(args[output_index].data),'returned output',case)
        count+=1
    assert count, f'{declaration.__name__}: public parameter domain produced no instruction'
    counts[declaration.__name__]=count
    print('AMX_FAMILY_DONE '+declaration.__name__+' count='+str(count),flush=True)
    report={'declaration':declaration.__name__,'semantic_generation':declaration.generation,
            'case_count':sum(counts.values()),'c_reference_count':c_count,
            'lifecycle_count':lifecycle_count,'per_op_counts':counts,'parameter_domains':domains}
    for key,value in report.items():
        record_property('amx_'+key,json.dumps(value,sort_keys=True))
    # Each pytest node reports its own family; sum these records for the full gate.
    print('AMX_ATOM_COVERAGE '+json.dumps(report,sort_keys=True))

