from tilefoundry.visitor_registry.candidates import HirMatch
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.core import OpCapability
from tilefoundry.ir.core.param_def import ParamDef,MemoryEffect
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import TensorPattern,LayoutPattern,SwitchPattern,WildcardPattern,OrPattern,RangePattern
from tilefoundry.ir.pattern import predicates as P
from tilefoundry.ir.types import DType,StorageKind,Mesh,Topology,Layout
from tilefoundry.ir.pattern.predicates import Bits, Table, variable
from .common import (
    instruction_versions,
    AmxOp,
    ProductMode,IndexMode,TransferCount,Repeat,Shuffle,FPMode,IntLane,
    FPALU,IntALU,Bank,ExtractMode,LutMode,
    FLOATS,TYPES,I8,I16,I32,
    bind,encode,amx_mesh_pattern,
    vector_layout,xy_matrix_layout,z_matrix_layout,z_vector_layout,
    vector_input_layout,packed_input_layout,int_matrix_layout,int_vector_layout,
    extract_source_shape,extract_source_layout,
)
R=MemoryEffect.READ
W=MemoryEffect.WRITE
AMX=Mesh((Topology('amx',1),),Layout((1,),(1,)),('unit',))

class Ldx(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={'src': 'x', 'dst': 'result'}),)
    mnemonic='LDX'
    capability=OpCapability('amx_ldx')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('src','dst')
    writes=('dst',)
    register_roles={'src':'memory','dst':'x'}
    count=ParamDef(kind='attribute',
        annotation=TransferCount,
        pattern=WildcardPattern('count',predicates=(SwitchPattern('amx_generation',{1:OrPattern(TransferCount.ONE,
                TransferCount.TWO),2:OrPattern(TransferCount.ONE,
                TransferCount.TWO,
                TransferCount.FOUR),3:OrPattern(TransferCount.ONE,
                TransferCount.TWO,
                TransferCount.FOUR),4:OrPattern(TransferCount.ONE,
                TransferCount.TWO,
                TransferCount.FOUR)}),)),
        default=TransferCount.ONE)
    nonconsecutive=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('nonconsecutive',predicates=(SwitchPattern('count',{TransferCount.ONE:False,
                TransferCount.TWO:SwitchPattern('amx_generation',{1:False,2:False,3:OrPattern(False,True),4:OrPattern(False,True)}),
                TransferCount.FOUR:SwitchPattern('amx_generation',{1:False,2:False,3:OrPattern(False,True),4:OrPattern(False,True)})}),)),
        default=False)
    dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('dtype',predicates=(OrPattern(*TYPES),)),
        default=DType.u8)
    register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('register',predicates=(RangePattern(lo=0,hi=7),)),
        default=0)
    memory_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('memory_offset',predicates=(RangePattern(lo=0,hi=511), variable('transfer_rows') == Table((0, 1, 2, 0, 4))[variable('count')], variable('transfer_lanes') == 512 // Bits('dtype'), variable('transfer_stride') == (512 // Bits('dtype')) * (1 + Table((0, 1))[variable('nonconsecutive')] * (Table((0, 1, 4, 0, 2))[variable('count')] - 1)))),
        default=0)
    src=ParamDef(kind='input',
        effect=R,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.GMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    dst=ParamDef(kind='input',
        effect=W,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_stride'),1))))
    def __init__(self,**attrs): bind(self,attrs)

class Ldy(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={'src': 'x', 'dst': 'result'}),)
    mnemonic='LDY'
    capability=OpCapability('amx_ldy')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('src','dst')
    writes=('dst',)
    register_roles={'src':'memory','dst':'y'}
    count=ParamDef(kind='attribute',
        annotation=TransferCount,
        pattern=WildcardPattern('count',predicates=(SwitchPattern('amx_generation',{1:OrPattern(TransferCount.ONE,
                TransferCount.TWO),2:OrPattern(TransferCount.ONE,
                TransferCount.TWO,
                TransferCount.FOUR),3:OrPattern(TransferCount.ONE,
                TransferCount.TWO,
                TransferCount.FOUR),4:OrPattern(TransferCount.ONE,
                TransferCount.TWO,
                TransferCount.FOUR)}),)),
        default=TransferCount.ONE)
    nonconsecutive=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('nonconsecutive',predicates=(SwitchPattern('count',{TransferCount.ONE:False,
                TransferCount.TWO:SwitchPattern('amx_generation',{1:False,2:False,3:OrPattern(False,True),4:OrPattern(False,True)}),
                TransferCount.FOUR:SwitchPattern('amx_generation',{1:False,2:False,3:OrPattern(False,True),4:OrPattern(False,True)})}),)),
        default=False)
    dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('dtype',predicates=(OrPattern(*TYPES),)),
        default=DType.u8)
    register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('register',predicates=(RangePattern(lo=0,hi=7),)),
        default=0)
    memory_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('memory_offset',predicates=(RangePattern(lo=0,hi=511), variable('transfer_rows') == Table((0, 1, 2, 0, 4))[variable('count')], variable('transfer_lanes') == 512 // Bits('dtype'), variable('transfer_stride') == (512 // Bits('dtype')) * (1 + Table((0, 1))[variable('nonconsecutive')] * (Table((0, 1, 4, 0, 2))[variable('count')] - 1)))),
        default=0)
    src=ParamDef(kind='input',
        effect=R,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.GMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    dst=ParamDef(kind='input',
        effect=W,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_stride'),1))))
    def __init__(self,**attrs): bind(self,attrs)

@register_op(dialect="T", namespace="amx", category="memory", name="ldz")
class Ldz(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={'src': 'x', 'dst': 'result'}),)
    mnemonic='LDZ'
    capability=OpCapability('amx_ldz')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('src','dst')
    writes=('dst',)
    register_roles={'src':'memory','dst':'z'}
    count=ParamDef(kind='attribute',
        annotation=TransferCount,
        pattern=WildcardPattern('count',predicates=(OrPattern(TransferCount.ONE,
                TransferCount.TWO),)),
        default=TransferCount.ONE)
    nonconsecutive=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('nonconsecutive',predicates=(False,)),
        default=False)
    dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('dtype',predicates=(OrPattern(*TYPES),)),
        default=DType.u8)
    register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('register',predicates=(RangePattern(lo=0,hi=63),)),
        default=0)
    memory_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('memory_offset',predicates=(RangePattern(lo=0,hi=511), variable('transfer_rows') == Table((0, 1, 2, 0, 4))[variable('count')], variable('transfer_lanes') == 512 // Bits('dtype'), variable('transfer_stride') == (512 // Bits('dtype')) * (1 + Table((0, 1))[variable('nonconsecutive')] * (Table((0, 1, 4, 0, 2))[variable('count')] - 1)))),
        default=0)
    src=ParamDef(kind='input',
        effect=R,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.GMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    dst=ParamDef(kind='input',
        effect=W,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    def __init__(self,**attrs): bind(self,attrs)

@register_op(dialect="T", namespace="amx", category="memory", name="ldzi")
class Ldzi(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={'src': 'x', 'dst': 'result'}),)
    mnemonic='LDZI'
    capability=OpCapability('amx_ldzi')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('src','dst')
    writes=('dst',)
    register_roles={'src':'memory','dst':'z'}
    count=ParamDef(kind='attribute',
        annotation=TransferCount,
        pattern=WildcardPattern('count',predicates=(TransferCount.ONE,)),
        default=TransferCount.ONE)
    nonconsecutive=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('nonconsecutive',predicates=(False,)),
        default=False)
    dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('dtype',predicates=(OrPattern(DType.u32,
                DType.i32,
                DType.f32),)),
        default=DType.u32)
    register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('register',predicates=(RangePattern(lo=0,hi=63),)),
        default=0)
    memory_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('memory_offset',predicates=(RangePattern(lo=0,hi=511), variable('transfer_rows') == Table((0, 1, 2, 0, 4))[variable('count')], variable('transfer_lanes') == 512 // Bits('dtype'), variable('transfer_stride') == (512 // Bits('dtype')) * (1 + Table((0, 1))[variable('nonconsecutive')] * (Table((0, 1, 4, 0, 2))[variable('count')] - 1)))),
        default=0)
    src=ParamDef(kind='input',
        effect=R,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.GMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    dst=ParamDef(kind='input',
        effect=W,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((1,(2,8)),(0,(16,1)))))
    def __init__(self,**attrs): bind(self,attrs)

@register_op(dialect="T", namespace="amx", category="memory", name="stx")
class Stx(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={'src': 'x', 'dst': 'result'}),)
    mnemonic='STX'
    capability=OpCapability('amx_stx')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('src','dst')
    writes=('dst',)
    register_roles={'src':'x','dst':'memory'}
    count=ParamDef(kind='attribute',
        annotation=TransferCount,
        pattern=WildcardPattern('count',predicates=(OrPattern(TransferCount.ONE,
                TransferCount.TWO),)),
        default=TransferCount.ONE)
    nonconsecutive=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('nonconsecutive',predicates=(False,)),
        default=False)
    dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('dtype',predicates=(OrPattern(*TYPES),)),
        default=DType.u8)
    register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('register',predicates=(RangePattern(lo=0,hi=7),)),
        default=0)
    memory_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('memory_offset',predicates=(RangePattern(lo=0,hi=511), variable('transfer_rows') == Table((0, 1, 2, 0, 4))[variable('count')], variable('transfer_lanes') == 512 // Bits('dtype'), variable('transfer_stride') == (512 // Bits('dtype')) * (1 + Table((0, 1))[variable('nonconsecutive')] * (Table((0, 1, 4, 0, 2))[variable('count')] - 1)))),
        default=0)
    src=ParamDef(kind='input',
        effect=R,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    dst=ParamDef(kind='input',
        effect=W,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.GMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    def __init__(self,**attrs): bind(self,attrs)

@register_op(dialect="T", namespace="amx", category="memory", name="sty")
class Sty(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={'src': 'x', 'dst': 'result'}),)
    mnemonic='STY'
    capability=OpCapability('amx_sty')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('src','dst')
    writes=('dst',)
    register_roles={'src':'y','dst':'memory'}
    count=ParamDef(kind='attribute',
        annotation=TransferCount,
        pattern=WildcardPattern('count',predicates=(OrPattern(TransferCount.ONE,
                TransferCount.TWO),)),
        default=TransferCount.ONE)
    nonconsecutive=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('nonconsecutive',predicates=(False,)),
        default=False)
    dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('dtype',predicates=(OrPattern(*TYPES),)),
        default=DType.u8)
    register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('register',predicates=(RangePattern(lo=0,hi=7),)),
        default=0)
    memory_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('memory_offset',predicates=(RangePattern(lo=0,hi=511), variable('transfer_rows') == Table((0, 1, 2, 0, 4))[variable('count')], variable('transfer_lanes') == 512 // Bits('dtype'), variable('transfer_stride') == (512 // Bits('dtype')) * (1 + Table((0, 1))[variable('nonconsecutive')] * (Table((0, 1, 4, 0, 2))[variable('count')] - 1)))),
        default=0)
    src=ParamDef(kind='input',
        effect=R,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    dst=ParamDef(kind='input',
        effect=W,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.GMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    def __init__(self,**attrs): bind(self,attrs)

@register_op(dialect="T", namespace="amx", category="memory", name="stz")
class Stz(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={'src': 'x', 'dst': 'result'}),)
    mnemonic='STZ'
    capability=OpCapability('amx_stz')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('src','dst')
    writes=('dst',)
    register_roles={'src':'z','dst':'memory'}
    count=ParamDef(kind='attribute',
        annotation=TransferCount,
        pattern=WildcardPattern('count',predicates=(OrPattern(TransferCount.ONE,
                TransferCount.TWO),)),
        default=TransferCount.ONE)
    nonconsecutive=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('nonconsecutive',predicates=(False,)),
        default=False)
    dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('dtype',predicates=(OrPattern(*TYPES),)),
        default=DType.u8)
    register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('register',predicates=(RangePattern(lo=0,hi=63),)),
        default=0)
    memory_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('memory_offset',predicates=(RangePattern(lo=0,hi=511), variable('transfer_rows') == Table((0, 1, 2, 0, 4))[variable('count')], variable('transfer_lanes') == 512 // Bits('dtype'), variable('transfer_stride') == (512 // Bits('dtype')) * (1 + Table((0, 1))[variable('nonconsecutive')] * (Table((0, 1, 4, 0, 2))[variable('count')] - 1)))),
        default=0)
    src=ParamDef(kind='input',
        effect=R,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    dst=ParamDef(kind='input',
        effect=W,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.GMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    def __init__(self,**attrs): bind(self,attrs)

@register_op(dialect="T", namespace="amx", category="memory", name="stzi")
class Stzi(AmxOp):
    hir_matches = (HirMatch(Reshard, operands={'src': 'x', 'dst': 'result'}),)
    mnemonic='STZI'
    capability=OpCapability('amx_stzi')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('src','dst')
    writes=('dst',)
    register_roles={'src':'z','dst':'memory'}
    count=ParamDef(kind='attribute',
        annotation=TransferCount,
        pattern=WildcardPattern('count',predicates=(TransferCount.ONE,)),
        default=TransferCount.ONE)
    nonconsecutive=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('nonconsecutive',predicates=(False,)),
        default=False)
    dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('dtype',predicates=(OrPattern(DType.u32,
                DType.i32,
                DType.f32),)),
        default=DType.u32)
    register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('register',predicates=(RangePattern(lo=0,hi=63),)),
        default=0)
    memory_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('memory_offset',predicates=(RangePattern(lo=0,hi=511), variable('transfer_rows') == Table((0, 1, 2, 0, 4))[variable('count')], variable('transfer_lanes') == 512 // Bits('dtype'), variable('transfer_stride') == (512 // Bits('dtype')) * (1 + Table((0, 1))[variable('nonconsecutive')] * (Table((0, 1, 4, 0, 2))[variable('count')] - 1)))),
        default=0)
    src=ParamDef(kind='input',
        effect=R,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((1,(2,8)),(0,(16,1)))))
    dst=ParamDef(kind='input',
        effect=W,
        pattern=TensorPattern(shape=(WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),
            dtype=WildcardPattern('dtype'),
            storage=StorageKind.GMEM,
            layout=LayoutPattern((WildcardPattern('transfer_rows'),WildcardPattern('transfer_lanes')),(WildcardPattern('transfer_lanes'),1))))
    def __init__(self,**attrs): bind(self,attrs)

LdxV1, LdxV2, LdxV3 = instruction_versions(Ldx, (1, 2, 3), name='ldx', category='memory')

LdyV1, LdyV2, LdyV3 = instruction_versions(Ldy, (1, 2, 3), name='ldy', category='memory')
