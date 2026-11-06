"""Representative MLX-native contracts; no numerical C++ simulation."""
from dataclasses import replace
from math import prod
import isl
import pytest
from tilefoundry.ir.core import Call,Var,Constant
from tilefoundry.ir.core.param_def import collect_param_defs,MemoryEffect as E
from tilefoundry.evaluator import EvaluateContext,TensorValue,to_torch_dtype
from tilefoundry.evaluator.registry import schedule_eval_registry
from tilefoundry.utils.lazy import torch
from tilefoundry.ir.isl_interop import layout_to_isl_map,shape_to_isl_set
from tilefoundry.ir.pattern import OrPattern,ShardLayoutPattern,PatternMatcher
from tilefoundry.ir.pattern.utils import fixed_pattern_value,selected_pattern
from tilefoundry.ir.types import DType,Layout,Mesh,ShardLayout,StorageKind,TensorType,Topology,UnitType
from tilefoundry.ir.types.int_tuple import flatten
from tilefoundry.visitor_registry.contexts import VerifyContext
from tilefoundry.visitor_registry.instruction import check_call
from tilefoundry.visitor_registry.typeinfer import inference_type
from tilefoundry_apple import MetalTarget
from tilefoundry_apple.metal.ir.tir import (
    METAL_OPS,LdMatrix,StMatrix,MatMul,Mma8x8x8,Reduce,ReductionKind as K,Shuffle,
    ShuffleKind as S,ShuffleFill,PrefixSum,PrefixProduct,Barrier,Fence,MemoryFence,
    AtomicLoad,AtomicStore,AtomicRmw,AtomicCas,AtomicKind as A,AtomicOrder,
    LdMatrix16,StMatrix16,MatMul16x32x16,Mma16x32x16,
)
def simd_mesh(groups=2,width=32):
    return Mesh((Topology("thread",groups*width),),Layout((groups,width),(width,1)))

# Every Op plus one case per material native mode/precision/layout boundary.
CASES=tuple((op,{},32) for op in METAL_OPS)+(
    (LdMatrix,{"dtype":DType.f16,"transpose":True},32),
    (AtomicRmw,{"dtype":DType.i32},32), (Mma8x8x8,dict.fromkeys(("lhs_dtype","rhs_dtype","acc_dtype","dst_dtype"),DType.f16),32),
    (Reduce,{"kind":K.PRODUCT},32), (Reduce,{"kind":K.ALL,"dtype":DType.bool},32),
    (Reduce,{"kind":K.ANY,"dtype":DType.bool},32),
    (Reduce,{"kind":K.XOR,"dtype":DType.u32},32),
    (Reduce,{"kind":K.AND,"dtype":DType.u16},32),
    (Reduce,{"kind":K.OR,"dtype":DType.u32},32),
    (Reduce,{"kind":K.MIN,"dtype":DType.f16},32),
    (Reduce,{"kind":K.MAX,"dtype":DType.i32},32),
    (Reduce,{},4),
    *((Shuffle,{"kind":kind},32) for kind in (S.BROADCAST,S.XOR,S.UP,S.DOWN,S.ROTATE_UP,S.ROTATE_DOWN)),
    (Shuffle,{"uniform":False},32),
    (ShuffleFill,{"kind":S.DOWN,"modulo":8,"slots":2},32),
    (PrefixProduct,{"dtype":DType.f16,"inclusive":True},32),
    (PrefixSum,{"dtype":DType.u16,"inclusive":True},32),
    (Barrier,{"fence":MemoryFence.NONE},32),
    *((AtomicRmw,{"kind":kind},32) for kind in (A.AND,A.OR,A.MIN,A.MAX)),
    (AtomicRmw,{"dtype":DType.f32},32), (AtomicCas,{"dtype":DType.f32},32),
    (MatMul16x32x16,{"a_dtype":DType.f16,"b_dtype":DType.f16},32),
    (Mma16x32x16,{"a_dtype":DType.bf16,"b_dtype":DType.bf16},32),
    (Mma16x32x16,{"b_dtype":DType.f16},32),
    (LdMatrix16,{"columns":32,"dtype":DType.bf16},32),
    (StMatrix16,{"columns":32},32),
)


def _bound(param,bindings):
    p=selected_pattern(param.pattern,bindings)
    def value(x):
        if isinstance(x,(DType,StorageKind)): return x
        if isinstance(x,OrPattern): return value(x.patterns[0])
        return fixed_pattern_value(x,bindings)
    layout=p.layout
    if isinstance(layout,ShardLayoutPattern):
        inner,mesh=layout.layout,layout.mesh.layout
        shape,strides=value(mesh.shape),value(mesh.strides)
        layout=ShardLayout(Layout(value(inner.shape),value(inner.strides)),layout.attrs,
            Mesh(tuple(Topology(n,prod(flatten(shape))) for n in layout.mesh.topologies),Layout(shape,strides)))
    else: layout=Layout(value(layout.shape),value(layout.strides))
    return TensorType(value(p.shape),value(p.dtype),layout,value(p.storage))


def _types(op,groups=2,width=32):
    nax=isinstance(op,(LdMatrix16,StMatrix16,MatMul16x32x16,Mma16x32x16))
    columns=getattr(op,"columns",16) if nax else 8
    matcher=PatternMatcher()
    for param in collect_param_defs(type(op)):
        if param.kind=="attribute":
            value=getattr(op,param.name)
            matcher.match(param.pattern,value)
            matcher.bindings[param.name]=value
    bindings={"groups":groups,"width":width,"requests":groups*width,"capacity":256,
        "row_stride":columns,"batch_stride":(16 if nax else 8)*columns,**matcher.bindings}
    return tuple(_bound(p,bindings) for p in collect_param_defs(type(op)) if p.kind=="input")


def _check_mlx(t,groups,nax):
    values={}
    actual=layout_to_isl_map(t.shape,t.layout,values,divided=lambda _:range(5))
    actual=actual.intersect_domain(shape_to_isl_set(t.shape,values))
    for name,coordinate in values.items():
        actual=actual.set_dim_name(isl.dim_type.PARAM,actual.find_dim_by_name(isl.dim_type.PARAM,name),f"o{coordinate.args[0].value}")
    strides=flatten(t.layout.layout.strides)
    dims=",".join(f"p{i}" for i in range(len(strides)))
    slot="+".join(f"{s}*p{i}" for i,s in enumerate(strides) if s) or "0"
    actual=actual.apply_range(isl.map(f"{{ [{dims}] -> [{slot}] }}"))
    if nax:
        count=8 if t.shape[-1]==16 else 16
        row="4*o4+o2+8*((s floord 4) mod 2)".replace("s floord 4","floor(s/4)")
        col="8*o3+4*o1+(s mod 4)+16*floor(s/8)"
    else: count,row,col=2,"4*o4+o2","4*o3+2*o1+s"
    expected=isl.map(f"""[o0,o1,o2,o3,o4] -> {{[g,r,c] -> [s]:
        g=o0 and r={row} and c={col} and 0<=s<{count}
        and 0<=o0<{groups} and 0<=o1<2 and 0<=o2<4 and 0<=o3<2 and 0<=o4<2 }}""")
    assert actual.is_equal(expected),actual


def test_special_instruction_patterns():
    for declaration,attrs,width in CASES:
        op=declaration(**attrs);types=_types(op,width=width)
        args=tuple(Var(f"v{i}",type=t) for i,t in enumerate(types))
        assert isinstance(inference_type(Call(op,args,type=None)),UnitType)
        mesh=simd_mesh(2,width)
        target=None if isinstance(op,(LdMatrix16,StMatrix16,MatMul16x32x16,Mma16x32x16)) else MetalTarget("apple.m2_pro.gpu")
        check_call(op,types,mesh,frame=VerifyContext(target=target,current_mesh=mesh))
        _evaluate_declared_case(op,types)
        if isinstance(op,(LdMatrix,StMatrix,MatMul,Mma8x8x8,LdMatrix16,StMatrix16,MatMul16x32x16,Mma16x32x16)):
            for t in types:
                if isinstance(t.layout,ShardLayout): _check_mlx(t,2,isinstance(op,(LdMatrix16,StMatrix16,MatMul16x32x16,Mma16x32x16)))
    selected=Mesh((Topology("thread",64),),Layout((32,),(1,)))
    check_call(Barrier(),(),selected,frame=VerifyContext(target=MetalTarget("apple.m2_pro.gpu")))


def test_rejected_contracts():
    mesh=simd_mesh()
    target=MetalTarget("apple.m2_pro.gpu")
    frame=VerifyContext(target=target,current_mesh=mesh)
    for cls,attrs in (
        (Mma8x8x8,{"lhs_dtype":DType.bf16}),
        (Reduce,{"kind":K.SUM,"dtype":DType.bool}),
        (Reduce,{"kind":K.ALL,"dtype":DType.bool,"slots":2}),
        (Shuffle,{"kind":S.UP,"uniform":False}),
        (ShuffleFill,{"modulo":3}),
        (AtomicRmw,{"kind":A.MIN,"dtype":DType.f32}),
        (AtomicLoad,{"order":AtomicOrder.SEQ_CST}),
        (MatMul16x32x16,{"relaxed_precision":False}),
    ):
        op=cls(**attrs)
        types=_types(cls())
        with pytest.raises((TypeError,ValueError)):
            check_call(op,types,mesh,frame=frame)
    op=MatMul();types=_types(op)
    with pytest.raises((TypeError,ValueError)):
        check_call(op,(replace(types[0],layout=Layout((2,8,8))),*types[1:]),mesh,frame=frame)
    with pytest.raises((TypeError,ValueError)):
        check_call(op,types,simd_mesh(1),frame=frame)
    for cls in (LdMatrix16,StMatrix16,MatMul16x32x16,Mma16x32x16):
        op=cls()
        with pytest.raises(ValueError):
            check_call(op,_types(op),mesh,frame=frame)
    for layout in (Layout((16,),(1,)),Layout((32,),(2,))):
        with pytest.raises(ValueError):
            check_call(Barrier(),(),Mesh((Topology("thread",64),),layout),frame=frame)
    op=Shuffle();types=_types(op)
    args=[Var(str(i),type=t) for i,t in enumerate(types)]
    args[-1]=Constant(32,type=types[-1])
    with pytest.raises(ValueError):
        check_call(op,tuple(args),mesh,frame=frame)


def _evaluate_declared_case(op,types):
    """Evaluate all declarations and compare declared writes and read-only inputs."""
    params=tuple(p for p in collect_param_defs(type(op)) if p.kind=="input")
    args=tuple(TensorValue(torch.zeros(t.shape,dtype=to_torch_dtype(t.dtype)),t) for t in types)
    if isinstance(op,AtomicRmw):
        args[-1].data.copy_(torch.arange(args[-1].data.numel(),dtype=args[-1].data.dtype))
        initial=(2**32-1 if op.dtype==DType.u32 else 2**31-1) if op.kind is A.ADD and op.dtype!=DType.f32 else 7
        args[1].data.fill_(initial)
        args[2].data.fill_(2 if op.kind is A.ADD else 3)
    if isinstance(op,AtomicCas) and op.dtype==DType.f32:
        args[-1].data.copy_(torch.arange(args[-1].data.numel(),dtype=args[-1].data.dtype))
        args[1].data.view(torch.int32).fill_(0x7fc00001)
        args[2].data.view(torch.int32).fill_(0x7fc00001)
        args[3].data.fill_(9)
    if isinstance(op,(MatMul16x32x16,Mma16x32x16)):
        args[1].data.copy_(torch.eye(16,dtype=args[1].data.dtype).expand_as(args[1].data)*2)
        args[2].data.copy_(torch.arange(args[2].data.numel(),dtype=torch.float32).reshape(args[2].data.shape))
        if isinstance(op,Mma16x32x16):
            args[0].data.fill_(3)
    before=tuple(v.data.clone() for v in args)
    handler=schedule_eval_registry.lookup(type(op))
    assert handler is not None
    result=handler(EvaluateContext(op=op,args=args,result_type=types[0] if types else UnitType()))
    for index,(param,value,old) in enumerate(zip(params,args,before,strict=True)):
        expected=old
        if isinstance(op,(MatMul16x32x16,Mma16x32x16)) and index==0:
            expected=before[2].float()*2
            if isinstance(op,Mma16x32x16):
                expected=expected+before[0]
        if isinstance(op,AtomicRmw):
            count=args[-1].data.numel()
            if param.name=="dst":
                expected=before[1][:count].clone()
            elif param.name=="object":
                expected=old.clone()
                for r in range(count):
                    left=before[1][r].item();right=before[2][r].item()
                    new={A.ADD:lambda:left+right,A.AND:lambda:int(left)&int(right),A.OR:lambda:int(left)|int(right),A.MIN:lambda:min(left,right),A.MAX:lambda:max(left,right)}[op.kind]()
                    if op.dtype!=DType.f32:
                        new=int(new)&0xffffffff
                        if op.dtype==DType.i32 and new>=2**31:new-=2**32
                    expected[r]=new
        if isinstance(op,AtomicCas) and op.dtype==DType.f32 and param.name=="object":
            expected=old.clone();expected[:args[-1].data.numel()]=9
        if isinstance(op,PrefixProduct) and not op.inclusive and param.name=="dst":
            expected=old.clone()
            expected[:,0]=1
        if isinstance(op,AtomicCas) and param.name=="success":
            expected=torch.ones_like(old)
        assert torch.equal(value.data.reshape(-1).view(torch.uint8),expected.reshape(-1).view(torch.uint8)),(type(op).__name__,param.name)
        if not param.effect & E.WRITE:
            assert torch.equal(value.data.reshape(-1).view(torch.uint8),old.reshape(-1).view(torch.uint8))
    if args:
        assert isinstance(result,TensorValue)
        assert torch.equal(result.data,args[0].data)
    else:
        assert result is None


def test_memory_bounds_and_cas():
    from tilefoundry.evaluator import EvalError
    for declaration in (AtomicLoad,AtomicStore,AtomicRmw,AtomicCas):
        op=declaration();types=_types(op)
        args=tuple(TensorValue(torch.zeros(t.shape,dtype=to_torch_dtype(t.dtype)),t) for t in types)
        args[-1].data.fill_(256)
        handler=schedule_eval_registry.lookup(declaration)
        with pytest.raises(EvalError):
            handler(EvaluateContext(op=op,args=args,result_type=types[0]))
    op=AtomicCas();types=_types(op)
    args=tuple(TensorValue(torch.zeros(t.shape,dtype=to_torch_dtype(t.dtype)),t) for t in types)
    args[-1].data.copy_(torch.arange(args[-1].data.numel(),dtype=args[-1].data.dtype))
    args[1].data.fill_(7)
    args[2].data.fill_(3)
    args[3].data.fill_(9)
    handler=schedule_eval_registry.lookup(AtomicCas)
    handler(EvaluateContext(op=op,args=args,result_type=types[0]))
    assert not args[0].data.any()
    assert (args[2].data==7).all() and (args[1].data==7).all()
    handler(EvaluateContext(op=op,args=args,result_type=types[0]))
    assert args[0].data.all()
    assert (args[1].data[:args[0].data.numel()]==9).all()


@pytest.mark.parametrize("alias",(False,True))
def test_mma_alias_and_effects(alias):
    op=Mma8x8x8();types=_types(op)
    data=[torch.zeros(t.shape,dtype=to_torch_dtype(t.dtype)) for t in types]
    data[1].copy_(torch.arange(128,dtype=torch.float32).reshape(2,8,8)/16)
    data[2].copy_(torch.eye(8).expand(2,8,8)*2)
    data[3].fill_(3)
    if alias:
        data[0]=data[3]
    before=[t.clone() for t in data]
    args=tuple(TensorValue(t,ty) for t,ty in zip(data,types,strict=True))
    result=schedule_eval_registry.lookup(Mma8x8x8)(EvaluateContext(op=op,args=args,result_type=types[0]))
    expected=before[1]@before[2]+before[3]
    assert torch.equal(result.data,expected)
    assert torch.equal(data[1],before[1]) and torch.equal(data[2],before[2])
    if not alias:
        assert torch.equal(data[3],before[3])
    params=tuple(p for p in collect_param_defs(Mma8x8x8) if p.kind=="input")
    assert tuple(p.effect for p in params)==(E.WRITE,E.READ,E.READ,E.READ)


@pytest.mark.parametrize("declaration",(Reduce,Shuffle,PrefixSum,PrefixProduct,AtomicLoad,AtomicStore))
def test_nonzero_semantics(declaration):
    op=declaration(dtype=DType.i32) if declaration in (AtomicLoad,AtomicStore) else declaration()
    types=_types(op)
    params=tuple(p for p in collect_param_defs(declaration) if p.kind=="input")
    args=tuple(TensorValue(torch.zeros(t.shape,dtype=to_torch_dtype(t.dtype)),t) for t in types)
    data=[v.data for v in args]
    if declaration is Reduce:
        data[1].copy_(torch.arange(data[1].numel(),dtype=data[1].dtype).reshape(data[1].shape))
        wanted=data[1].sum(1,keepdim=True)
    elif declaration is Shuffle:
        data[1].copy_(torch.arange(data[1].numel(),dtype=data[1].dtype).reshape(data[1].shape))
        data[-1].fill_(7)
        wanted=data[1][:,7:8].expand_as(data[1]).clone()
    elif declaration in (PrefixSum,PrefixProduct):
        data[1].fill_(1)
        if declaration is PrefixSum:
            wanted=torch.arange(32,dtype=data[1].dtype).expand_as(data[1]).clone()
        else:
            data[1][:,0]=2
            wanted=torch.full_like(data[1],2);wanted[:,0]=1
    else:
        idx=data[-1];idx.copy_(torch.arange(idx.numel(),dtype=idx.dtype).flip(0))
        if declaration is AtomicLoad:
            data[1].copy_(torch.arange(data[1].numel(),dtype=data[1].dtype))
            wanted=data[1][idx.long()].clone()
        else:
            data[1].fill_(5)
            wanted=data[0].clone();wanted[idx.long()]=5
    before=[x.clone() for x in data]
    result=schedule_eval_registry.lookup(declaration)(EvaluateContext(op=op,args=args,result_type=types[0]))
    assert torch.equal(result.data,wanted)
    for p,x,old in zip(params,data,before,strict=True):
        if not p.effect&E.WRITE:
            assert torch.equal(x,old)


def test_precision_and_selector_rejections():
    from tilefoundry.evaluator import EvalError
    mesh=simd_mesh();frame=VerifyContext(target=MetalTarget("apple.m2_pro.gpu"),current_mesh=mesh)
    for name in ("lhs_dtype","rhs_dtype","acc_dtype","dst_dtype"):
        op=Mma8x8x8(**{name:DType.f16})
        with pytest.raises((TypeError,ValueError)):
            check_call(op,_types(Mma8x8x8()),mesh,frame=frame)
    for cls in (MatMul16x32x16,Mma16x32x16):
        op=cls(a_dtype=DType.f16,b_dtype=DType.f32)
        with pytest.raises((TypeError,ValueError)):
            check_call(op,_types(cls()),mesh,frame=VerifyContext(target=None,current_mesh=mesh))
    for cls in (Shuffle,ShuffleFill):
        op=cls();types=_types(op)
        args=tuple(TensorValue(torch.zeros(t.shape,dtype=to_torch_dtype(t.dtype)),t) for t in types)
        args[0].data.fill_(123)
        args[-1].data.fill_(33)
        before=tuple(v.data.clone() for v in args)
        with pytest.raises(EvalError):
            schedule_eval_registry.lookup(cls)(EvaluateContext(op=op,args=args,result_type=types[0]))
        assert all(torch.equal(v.data,old) for v,old in zip(args,before,strict=True))
