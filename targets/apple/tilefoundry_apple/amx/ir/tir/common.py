"""Binding, physical fragment layouts, and encoding for explicit AMX Ops.

Operand layouts contain relative physical element offsets. Named selectors add
byte origins; X/Y wrap modulo512. No public operand is the whole AMX state.
"""
from copy import copy
from dataclasses import fields
from enum import IntEnum
from math import prod
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import ParamDef, collect_param_defs
from tilefoundry.ir.pattern import Pattern, PatternMatcher, matched, SwitchPattern, TensorPattern, LayoutPattern, OrPattern, WildcardPattern, RangePattern
from tilefoundry.ir.pattern.utils import fixed_pattern_value,selected_pattern,thread_execution_mesh
from tilefoundry.ir.types import DType,Layout,Mesh,StorageKind,TensorType,Topology,UnitType
from tilefoundry.ir.core.register import register_op, register_type

@register_type(dialect='T',namespace='amx')
class ProductMode(IntEnum): MATRIX=0; VECTOR=1
@register_type(dialect='T',namespace='amx')
class IndexMode(IntEnum): NONE=0; X2=1; Y2=2; X4=3; Y4=4
@register_type(dialect='T',namespace='amx')
class TransferCount(IntEnum): ONE=1; TWO=2; FOUR=4
@register_type(dialect='T',namespace='amx')
class Repeat(IntEnum): ONE=1; TWO=2; FOUR=4
@register_type(dialect='T',namespace='amx')
class Shuffle(IntEnum): NONE=0; PAIRS=1; QUADS=2; OCTETS=3
@register_type(dialect='T',namespace='amx')
class FPMode(IntEnum): F16=2; F16_F32=3; F32=4; F64=7; BF16=0; BF16_F32=1
@register_type(dialect='T',namespace='amx')
class IntLane(IntEnum): NARROW16=2; WIDE16=3; I32=4; I8=9; I8_I8_I32=10; I8_I8_I16=11; I8_I16_I32=12; I16_I8_I32=13
@register_type(dialect='T',namespace='amx')
class FPALU(IntEnum): ADD_PRODUCT=0; SUB_PRODUCT=1; POSITIVE_SELECT=4; MIN=5; MAX=7; PRODUCT=10; ADD_X=11; ADD_Y=12
@register_type(dialect='T',namespace='amx')
class IntALU(IntEnum): ADD_PRODUCT=0; SUB_PRODUCT=1; ADD_SUM=2; SUB_SUM=3; SHIFT=4; SQRDMLAH=5; SQRDMLSH=6; WIDE_PRODUCT=8; POPCOUNT=9; PRODUCT=10; ADD_X=11; ADD_Y=12
@register_type(dialect='T',namespace='amx')
class Bank(IntEnum): X=0; Y=1; Z=2
@register_type(dialect='T',namespace='amx')
class ExtractMode(IntEnum): COPY=0; LEGACY=1; INTEGER=2; FLOAT=3
@register_type(dialect='T',namespace='amx')
class LutMode(IntEnum): F32=0; F16=1; F64=2; I32=3; I16=4; U32=5; U16=6; LOOKUP32_2=7; LOOKUP16_2=8; LOOKUP8_2=9; LOOKUP64_4=10; LOOKUP32_4=11; LOOKUP16_4=12; LOOKUP8_4=13; LOOKUP16_5=14; LOOKUP8_5=15

FLOATS=(DType.f16,DType.bf16,DType.f32,DType.f64)
TYPES=(DType.u8,DType.i8,DType.u16,DType.i16,DType.u32,DType.i32,DType.f16,DType.bf16,DType.f32,DType.f64)
I16=(DType.i16,DType.u16); I8=(DType.i8,DType.u8); I32=(DType.i32,DType.u32)


class AmxOp(Op):
    """An instruction with static reference semantics, independent of a Target value."""

    generation = 1
    execution_mesh = Mesh((Topology("thread", 1),), Layout((1,), (1,)), ("worker",))
    execution_mesh_pattern = staticmethod(thread_execution_mesh)


def instruction_versions(template, generations, *, name, category):
    """Register static ISA variants once, specializing independent declaration copies."""
    from tilefoundry.ir.pattern.predicates import Formula, Term, evaluate

    variants = []
    for version, generation in enumerate(generations, 1):
        memo = {}

        def specialize(value):
            if isinstance(value, Term) and value.op == 'variable' and value.args == ('amx_generation',):
                return Term('constant', (generation,))
            if not isinstance(value, (ParamDef, Pattern, Term, tuple)):
                return value
            cached = memo.get(id(value))
            if cached is not None:
                return cached
            if isinstance(value, SwitchPattern) and value.param == 'amx_generation':
                result = specialize(dict(value.branches)[generation])
            elif isinstance(value, tuple):
                result = tuple(specialize(item) for item in value)
            else:
                values = {field.name: specialize(getattr(value, field.name))
                          for field in fields(value)}
                if 'predicates' in values:
                    values['predicates'] = tuple(
                        item for item in values['predicates']
                        if not (isinstance(item, Formula) and evaluate(item, {}) is True)
                    )
                result = copy(value)
                for field_name, specialized in values.items():
                    object.__setattr__(result, field_name, specialized)
                if isinstance(result, Formula):
                    if result.op in ('and', 'or'):
                        left, right = result.args
                        left_truth, right_truth = evaluate(left, {}), evaluate(right, {})
                        if result.op == 'and':
                            if left_truth is True:
                                result = right
                            elif right_truth is True:
                                result = left
                        else:
                            if left_truth is False:
                                result = right
                            elif right_truth is False:
                                result = left
                    known = evaluate(result, {})
                    if known is not None:
                        result = Formula('eq', (
                            Term('constant', (0,)), Term('constant', (0 if known else 1,)),
                        ))
            memo[id(value)] = result
            return result

        attributes = {param.name: specialize(param) for param in collect_param_defs(template)}
        attributes.update(
            __module__=template.__module__, generation=generation,
            capability=OpCapability(f'amx_{name}_v{version}'),
            hir_matches=tuple(template.__dict__.get('hir_matches', ())),
        )
        variant = type(f'{template.__name__}V{version}', (template,), attributes)
        variants.append(register_op(dialect='T', namespace='amx', category=category,
                                    name=f'{name}_v{version}')(variant))
    return tuple(variants)


def bind(op,attrs):
    Op.__init__(op,**attrs)
    held={}
    for param in collect_param_defs(type(op)):
        if param.kind!='attribute': continue
        value=getattr(op,param.name)
        if value is None and param.optional:
            held[param.name]=None
            continue
        if param.annotation is DType and isinstance(value,str): value=DType.from_name(value); setattr(op,param.name,value)
        if isinstance(param.annotation,type) and issubclass(param.annotation,IntEnum) and not isinstance(value,param.annotation): raise TypeError(f'{param.name} requires {param.annotation.__name__}')
        match=matched(param.pattern,value,held)
        if match is None: raise ValueError(f'{type(op).__name__}.{param.name}={value!r} violates {param.pattern!r}')
        held.update(match.captures); held[param.name]=value
    op.bindings=held
    validate(op)

def validate(op):
    n=op.mnemonic
    if n.startswith(('LD','ST')):
        if op.count==TransferCount.FOUR and (n not in ('LDX','LDY') or op.generation<2): raise ValueError('four-register transfer requires M2+ LDX/LDY')
        if op.nonconsecutive and (n not in ('LDX','LDY') or op.generation<3 or op.count==TransferCount.ONE): raise ValueError('nonconsecutive transfer requires M3+ multiple LDX/LDY')
        if n.endswith('I') and op.count!=TransferCount.ONE: raise ValueError('interleaved transfer is one64-byte fragment')
    if hasattr(op,'repeat') and op.repeat!=Repeat.ONE and op.generation<2: raise ValueError('repetition requires M2+')
    if hasattr(op,'index_mode') and op.index_mode!=IndexMode.NONE and int(op.alu) not in ((0,8) if n=='MATINT' else (0,)): raise ValueError('indexed mode requires product ALU')
    if hasattr(op,'fp_mode') and op.fp_mode in (FPMode.BF16,FPMode.BF16_F32) and op.generation<2: raise ValueError('BF16 requires M2+')
    if hasattr(op,'alu') and int(op.alu)>=10 and op.generation<2: raise ValueError('ALU mode requires M2+')
    if n=='GENLUT' and op.bf16 and (op.mode!=LutMode.F16 or op.generation<2): raise ValueError('BF16 generation requires M2+ F16 generation mode')
    if n=='GENLUT' and int(op.mode)<=6 and op.destination==Bank.Z: raise ValueError('index generation writes X or Y')
    if n=='GENLUT' and op.destination!=Bank.Z and op.destination_register>7: raise ValueError('X/Y destination register must be0..7')

def check_types(op,types):
    matcher=PatternMatcher(op.bindings)
    from tilefoundry.visitor_registry.instruction import call_params
    params=call_params(op,len(types))
    if hasattr(op,'index_mode'):
        expected=1 if op.mnemonic in ('MATINT','VECINT') and op.alu==IntALU.SHIFT else 3+int(op.index_mode!=IndexMode.NONE)
        if len(types)!=expected: raise TypeError(f'{op.mnemonic}: this mode requires {expected} logical operands')
    for param,t in zip(params,types):
        if not matcher.match(param.pattern,t): raise TypeError(f'{op.mnemonic}.{param.name}: invalid logical fragment {t!r}')
    if matcher.solve() is not True: raise TypeError('AMX fragment constraints cannot be solved')
    return UnitType()

def bound_value(value,bindings):
    if isinstance(value,SwitchPattern):
        return bound_value(dict(value.branches)[bindings[value.param]],bindings)
    if isinstance(value,(DType,StorageKind)): return value
    return fixed_pattern_value(value,bindings)


def operand_types(op):
    out=[]
    for p in collect_param_defs(type(op)):
        if p.kind!='input': continue
        pattern=selected_pattern(p.pattern,op.bindings)
        if pattern is None and p.optional: continue
        shape=fixed_pattern_value(pattern.shape,op.bindings)
        dtype=bound_value(pattern.dtype,op.bindings)
        storage=bound_value(pattern.storage,op.bindings)
        lp=selected_pattern(pattern.layout,op.bindings)
        layout=Layout(fixed_pattern_value(lp.shape,op.bindings),fixed_pattern_value(lp.strides,op.bindings))
        out.append(TensorType(shape,dtype,layout,storage))
    check_types(op,tuple(out)); return tuple(out)

def vector_layout(lanes,step=1): return LayoutPattern((lanes,),(step,))
def xy_matrix_layout(lanes,vertical,step=1): return LayoutPattern((lanes,1) if vertical else (1,lanes),(step,0) if vertical else (0,step))
def z_matrix_layout(lanes,xybytes,zbytes):
    ratio=zbytes//xybytes
    if ratio==1: return LayoutPattern((lanes,lanes),(64//xybytes*xybytes,1))
    return LayoutPattern((lanes,(ratio,lanes//ratio)),(64//xybytes, (64//zbytes,1)))

def _mode_offsets(shape,strides,index):
    if isinstance(shape,int): return index*strides
    result=0
    for extent,stride in zip(shape,strides):
        size=prod(_flatten(extent))
        result+=_mode_offsets(extent,stride,index%size); index//=size
    return result

def _flatten(x):
    return tuple(v for item in x for v in _flatten(item)) if isinstance(x,tuple) else (x,)
def layout_offsets(t):
    shape=t.shape; layout=t.layout
    count=prod(shape); result=[]
    for flat in range(count):
        coords=[]; left=flat
        for dim in reversed(shape): coords.append(left%dim); left//=dim
        coords.reverse()
        result.append(sum(_mode_offsets(s,st,c) for s,st,c in zip(layout.shape,layout.strides,coords))*t.dtype.bit_width//8)
    return tuple(result)

def encode(op):
    return encode_operand(op)

def infer(call,ctx):
    return check_types(call.target,tuple(ctx.type_of(a) for a in call.args))


def placements(op,types):
    """Byte starts of logical elements; arithmetic is exclusively in evaluator semantics."""
    n=op.mnemonic; out={}
    index=getattr(op,'index_mode',IndexMode.NONE)
    repeat=int(getattr(op,'repeat',Repeat.ONE)); step=64//repeat
    from tilefoundry.visitor_registry.instruction import call_params
    for param,t in zip(call_params(op,len(types)),types):
        role=param.name
        bank=op.register_roles[role]
        if bank=='indexed_bank': bank='y' if index in (IndexMode.Y2,IndexMode.Y4) else 'x'
        elif bank in ('source_bank','table_bank','destination'): bank=getattr(op,bank).name.lower()
        elif bank=='extract_source': bank=('x' if n=='EXTRY' else 'y') if op.mode==ExtractMode.COPY else 'z'
        offsets=layout_offsets(t)
        origin=0
        if n in ('SET','CLR'): pass
        elif n.startswith(('LD','ST')):
            if bank=='memory': origin=op.memory_offset
            elif n.endswith('I'):
                origin=(op.register&~1)*64+(op.register&1)*32
            else: origin=op.register*64
            capacity=4096 if bank=='z' else 512 if bank in ('x','y') else 1<<56
            offsets=tuple((origin+v)%capacity for v in offsets)
            out[role]={'bank':bank,'offsets':offsets}; continue
        elif n=='GENLUT':
            origin=op.source_offset if role=='source' else op.table_register*64 if role=='table' else op.destination_register*64
        elif n in ('EXTRX','EXTRY'):
            if op.mode==ExtractMode.COPY:
                origin=(op.z_selector if role=='src' else op.destination_register)*64
            elif role=='dst':
                origin=op.destination_offset & (-64 if op.generation>=4 and repeat>1 else -1)
            elif n=='EXTRY':
                origin=(op.z_selector&(step-1)) & -(op.src_dtype.bit_width//8)
            elif op.mode==ExtractMode.LEGACY: origin=op.z_selector*64
            else: origin=((op.z_selector&(step-1)) & -(op.src_dtype.bit_width//8))*64
        elif role=='table': origin=op.index_register*64
        elif role in ('lhs','rhs'):
            is_y=role=='lhs'; origin=op.y_offset if is_y else op.x_offset
            if n in ('VECFP','VECINT') and repeat>1 and op.generation>=4:
                dt=getattr(op,'dtype',None) or (op.y_dtype if is_y else op.x_dtype)
                size=dt.bit_width//8
                indexed=index in ((IndexMode.Y2,IndexMode.Y4) if is_y else (IndexMode.X2,IndexMode.X4))
                broadcast=op.broadcast_mode in ((3,7) if is_y else (2,6))
                if indexed:
                    ib=4 if index in (IndexMode.X4,IndexMode.Y4) else 2
                    origin &= -64 | -(512*ib//(size*step))
                else: origin &= -size if broadcast else -64
        elif role=='acc':
            if n.startswith(('FMA','FMS')) or n=='MAC16':
                width=int(n[3:]) if n!='MAC16' else 16
                if op.mode==ProductMode.VECTOR: origin=op.z_row*64
                elif width==16 and t.dtype.bit_width==32: origin=0
                else: origin=(op.z_row&(width//8-1))*64
            elif n=='MATFP':
                size=op.dtype.bit_width//8
                origin=0 if t.dtype.bit_width>op.dtype.bit_width else (op.z_row&(size-1))*64
            elif n=='MATINT':
                xb=op.x_dtype.bit_width//8; zb=t.dtype.bit_width//8
                origin=(op.z_row & ((zb-1) if op.alu==IntALU.SHIFT else (xb-1)&~(max(1,zb//xb)-1)))*64
            else:
                if n=='VECFP': ratio=t.dtype.bit_width//op.dtype.bit_width
                else: ratio=1 if op.alu==IntALU.SHIFT else t.dtype.bit_width//min(op.x_dtype.bit_width,op.y_dtype.bit_width)
                origin=((op.z_row&(step-1)) & -max(1,ratio))*64
        out[role]={'bank':bank,'offsets':tuple(origin+v for v in offsets)}
    return out


def encode_operand(op):
    n=op.mnemonic
    if n in ('SET','CLR'): return 0
    if n.startswith(('LD','ST')):
        return op.memory_offset|(op.register<<56)|(int(op.count!=TransferCount.ONE)<<62)|(int(op.count==TransferCount.FOUR)<<60)|(int(op.nonconsecutive)<<61)
    if n.startswith(('FMA','FMS')) or n=='MAC16':
        width=int(n[3:]) if n!='MAC16' else 16
        out=op.y_offset|(op.x_offset<<10)|(op.z_row<<20)|(int(op.skip_z)<<27)|(int(op.skip_y)<<28)|(int(op.skip_x)<<29)|(op.y_mask<<32)|(op.x_mask<<41)|(int(op.acc_dtype.bit_width==32 and width==16)<<62)|(int(op.mode)<<63)
        if n=='MAC16': return out|(int(op.x_dtype.bit_width==8)<<61)|(int(op.y_dtype.bit_width==8)<<60)|(op.shift<<55)
        return out|(int(width==32 and op.y_dtype==DType.f16)<<60)|(int(width==32 and op.x_dtype==DType.f16)<<61)
    if n in ('MATFP','VECFP','MATINT','VECINT'):
        integer=n.endswith('INT'); vector=n.startswith('VEC')
        repeat=int(op.repeat); step=64//repeat
        zr=(op.z_row&(step-1)) | (32 if repeat==4 else 0)
        out=op.y_offset|(op.x_offset<<10)|(zr<<20)
        out|=int(op.lane_mode if integer else op.fp_mode)<<42
        if op.index_mode==IndexMode.NONE: out|=int(op.alu)<<47
        else:
            out|=1<<53
            out|=int(op.index_mode in (IndexMode.Y2,IndexMode.Y4))<<47
            out|=int(op.index_mode in (IndexMode.X4,IndexMode.Y4))<<48
            out|=op.index_register<<49
            if n=='MATINT' and op.alu==IntALU.WIDE_PRODUCT: out|=1<<54
        if vector and repeat>1: out|=(1<<31)|(op.broadcast_mode<<32)
        else: out|=op.mask<<32
        if integer:
            out|=op.shift<<58
            if op.alu==IntALU.SHIFT:
                out|=int(op.acc_dtype.name.startswith('i'))<<63
                out|=int(op.signed_output)<<26
                out|=int(op.saturate)<<30|int(op.rounding)<<29
            else:
                out|=int(op.x_dtype.name.startswith('i'))<<63|int(op.y_dtype.name.startswith('i'))<<26
                out|=int(op.shuffle_x)<<29|int(op.shuffle_y)<<27
            if not vector: out|=int(op.mask_y)<<25
        else:
            out|=int(op.shuffle_x)<<29|int(op.shuffle_y)<<27
            if not vector: out|=(op.y_mask&63)<<58|(op.y_mask>>6)<<23
        return out
    if n in ('EXTRX','EXTRY'):
        if op.mode==ExtractMode.COPY: return (1<<27)|(op.z_selector<<20)|(op.destination_register<<(6 if n=='EXTRY' else 16))
        if op.mode==ExtractMode.LEGACY:
            lane={64:0,32:1,16:3 if op.low_byte_only else 2}[op.src_dtype.bit_width]
            return (op.z_selector<<20)|(lane<<28)|(op.destination_offset<<(0 if n=='EXTRY' else 10))|(op.mask<<(32 if n=='EXTRY' else 41))
        sb=op.src_dtype.bit_width; db=op.dst_dtype.bit_width
        if op.mode==ExtractMode.FLOAT:
            lane=1 if sb==64 else 8 if sb==db==32 else (10 if op.alternating else 9) if sb==32 else 2
        else:
            lane=0 if sb==8 else 8 if sb==db==32 else (10 if op.alternating else 9) if sb==32 and db==16 else 11 if sb==32 else 13 if db==8 else 2
        step=64//int(op.repeat); selector=(op.z_selector&(step-1))|(32 if op.repeat==Repeat.FOUR else 0)
        shift=16 if op.dst_dtype==DType.bf16 else op.shift
        return op.destination_offset|(int(op.destination==Bank.Y)<<10)|(lane<<11)|(selector<<20)|(1<<26)|(int(op.repeat!=Repeat.ONE)<<31)|(op.mask<<32)|(int(op.rounding)<<54)|(int(op.saturate)<<55)|(int(op.dst_dtype.name.startswith('i'))<<56)|(int(op.src_dtype.name.startswith('i'))<<57)|(shift<<58)|(int(op.mode==ExtractMode.FLOAT)<<63)
    if n=='GENLUT':
        out=op.source_offset|(int(op.source_bank==Bank.Y)<<10)|(op.destination_register<<20)|(int(op.destination==Bank.Z)<<26)|(int(op.bf16)<<30)|(int(op.mode)<<53)|(int(op.table_bank==Bank.Y)<<59)|(op.table_register<<60)
        if op.destination!=Bank.Z: out|=int(op.destination==Bank.Y)<<25
        return out
    raise NotImplementedError(n)


def z_vector_layout(mode,repeat):
    bits={FPMode.F16:16,FPMode.BF16:16,FPMode.F16_F32:16,FPMode.BF16_F32:16,FPMode.F32:32,FPMode.F64:64}[mode]
    zb=4 if mode in (FPMode.F16_F32,FPMode.BF16_F32) else bits//8
    ratio=zb//(bits//8); lanes=512//bits; rows=int(repeat)
    if ratio==1: return LayoutPattern((rows,lanes),((64//rows)*64//zb,1))
    return LayoutPattern((rows,(ratio,lanes//ratio)),((64//rows)*64//zb,(64//zb,1)))


def vector_input_layout(lanes,repeat,is_y):
    return SwitchPattern('broadcast_mode',{
        b:LayoutPattern((int(repeat),lanes),(0 if b in ((3,7) if is_y else (2,6)) else lanes,1))
        for b in range(8)
    })


def packed_input_layout(count,repeat,is_y):
    return SwitchPattern('broadcast_mode',{
        b:LayoutPattern((int(repeat),count),(0 if b in ((3,7) if is_y else (2,6)) else count,1))
        for b in range(8)
    })


def int_matrix_layout(xbytes,ybytes,zbytes):
    step=zbytes if xbytes==1 else xbytes
    ratio=max(1,zbytes//xbytes)
    lanes=64//xbytes
    if ratio==1: return LayoutPattern((64//step,lanes),(step*64//zbytes,1))
    return LayoutPattern((64//step,(ratio,lanes//ratio)),(step*64//zbytes,(64//zbytes,1)))


def int_vector_layout(xbytes,ybytes,zbytes,repeat):
    step=min(xbytes,ybytes); ratio=max(1,zbytes//step); lanes=64//step
    if ratio==1: return LayoutPattern((int(repeat),lanes),((64//int(repeat))*64//zbytes,1))
    return LayoutPattern((int(repeat),(ratio,lanes//ratio)),((64//int(repeat))*64//zbytes,(64//zbytes,1)))


def extract_source_shape(vertical,mode,src,dst,repeat):
    b=src.bit_width//8
    if mode==ExtractMode.COPY: return (1,64//b)
    if mode==ExtractMode.LEGACY: return (int(repeat),64) if vertical else (1,1,64//b)
    return (int(repeat),64//(dst.bit_width//8))


def extract_source_layout(vertical,mode,src,dst,repeat,alternating=False):
    b=src.bit_width//8; r=int(repeat); step=64//r
    shape=extract_source_shape(vertical,mode,src,dst,repeat)
    if mode==ExtractMode.COPY: return LayoutPattern(shape,(0,1))
    if mode==ExtractMode.LEGACY:
        return LayoutPattern(shape,(step//b,64//b) if vertical else (step*64//b,64//b,1))
    db=dst.bit_width//8; ratio=max(1,b//db); lanes=64//db
    row=step//b if vertical else step*64//b
    stride=2 if alternating and b==4 and db==2 else 1
    if ratio==1: return LayoutPattern(shape,(row,64 if vertical else 1))
    return LayoutPattern((r,(lanes//ratio,ratio)),(row,(64 if vertical else 1,stride*64//b)))


def access(call,ctx):
    import isl
    from tilefoundry.visitor_registry.access_relation import AccessRelation,matmul_relations,identity_access,iterating
    op=call.target; types=tuple(ctx.type_of(a) for a in call.args)
    if op.mnemonic.startswith(('FMA','FMS')) or op.mnemonic=='MAC16' or op.mnemonic in ('MATFP','MATINT'):
        if getattr(op,'mode',ProductMode.MATRIX)==ProductMode.MATRIX and len(types)==3:
            _,a,b=types
            *operands,result=matmul_relations(a.shape,b.shape,(-2,-1,-1,-2))
            return (result,*operands,result)
    if types and all(t.shape==types[0].shape for t in types):
        return iterating(types[0].shape, tuple(identity_access(len(t.shape)) for t in (*types, types[op.roles.index(op.writes[0])])) )
    def covered(t):
        names=tuple('d'+str(i) for i in range(len(t.shape)))
        bounds=' and '.join(f'0<={name}<{int(size)}' for name,size in zip(names,t.shape)) or 'true'
        return AccessRelation(isl.map('{ [issue] -> ['+','.join(names)+'] : issue=0 and '+bounds+' }'),may_access=True)
    return tuple(covered(t) for t in types)+(covered(types[op.roles.index(op.writes[0])]),)


def cost(call,ctx):
    from tilefoundry.target.base import UnsupportedCapabilityError
    from tilefoundry.visitor_registry.contexts import Cost,TrafficBytes
    op=call.target; n=op.mnemonic
    types=tuple(ctx.type_of(arg) for arg in call.args)
    check_types(op,types)
    if n.startswith(('LD','ST')):
        count=int(op.count)*64
        return Cost({},(TrafficBytes(read=count),TrafficBytes(write=count),TrafficBytes()))
    if n in ('SET','CLR'):
        return Cost({},(TrafficBytes(read=1,write=1),TrafficBytes()))
    if n.startswith(('FMA','FMS')):
        from ...evaluator.common import parse_writemask
        width=int(n[3:]); size=width//8; lanes=64//size
        xm=parse_writemask(op.x_mask,size,7)
        ym=parse_writemask(op.y_mask,size,7)
        nx=sum((xm>>(i*size))&1 for i in range(lanes))
        ny=1 if op.mode==ProductMode.VECTOR else sum((ym>>(i*size))&1 for i in range(lanes))
        active=nx*ny; skip=int(op.skip_z)|(int(op.skip_y)<<1)|(int(op.skip_x)<<2)
        work=(2,1,1,0,1,0,0,0)[skip]*active
        zb=types[0].dtype.bit_width//8
        xread=0 if op.skip_x or not active else nx*types[2].dtype.bit_width//8
        yread=0 if op.skip_y or not active else (nx if op.mode==ProductMode.VECTOR else ny)*types[1].dtype.bit_width//8
        traffic=(TrafficBytes(read=0 if op.skip_z else active*zb,write=active*zb),TrafficBytes(read=yread),TrafficBytes(read=xread),TrafficBytes())
        service={'negate':active} if n.startswith('FMS') and skip in (1,3,5) else {}
        return Cost({types[0].dtype:work},traffic,service=service)
    raise UnsupportedCapabilityError(f'{n}: exact logical cost is unknown')
