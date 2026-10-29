from tilefoundry.ir.core import OpCapability
from tilefoundry.ir.core.param_def import ParamDef,MemoryEffect
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import TensorPattern,LayoutPattern,SwitchPattern,WildcardPattern,OrPattern,RangePattern
from tilefoundry.ir.pattern import predicates as P
from tilefoundry.ir.types import DType,StorageKind,Mesh,Topology,Layout
from tilefoundry.ir.pattern.predicates import Bits, In, Table, variable
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

class Genlut(AmxOp):
    """Packed index source, typed table, and only the written destination register."""
    mnemonic='GENLUT'
    capability=OpCapability('amx_genlut')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('source','table','dst')
    writes=('dst',)
    register_roles={'source':'source_bank','table':'table_bank','dst':'destination'}
    mode=ParamDef(kind='attribute',
        annotation=LutMode,
        pattern=WildcardPattern('mode'))
    bf16=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('bf16', predicates=((variable('bf16') == False) | ((variable('amx_generation') >= 2) & (variable('mode') == LutMode.F16)),)),
        default=False)
    dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('dtype',predicates=(SwitchPattern('mode',{
        LutMode.F32:DType.f32,
                LutMode.F16:SwitchPattern('bf16',{False:DType.f16,True:DType.bf16}),
                LutMode.F64:DType.f64,
                LutMode.I32:DType.i32,
                LutMode.I16:DType.i16,
                LutMode.U32:DType.u32,
                LutMode.U16:DType.u16,
        LutMode.LOOKUP32_2:OrPattern(DType.i32,
                DType.u32,
                DType.f32),
                LutMode.LOOKUP16_2:OrPattern(DType.i16,
                DType.u16,
                DType.f16,
                DType.bf16),
                LutMode.LOOKUP8_2:OrPattern(DType.i8,
                DType.u8),
                LutMode.LOOKUP64_4:DType.f64,
                LutMode.LOOKUP32_4:OrPattern(DType.i32,
                DType.u32,
                DType.f32),
                LutMode.LOOKUP16_4:OrPattern(DType.i16,
                DType.u16,
                DType.f16,
                DType.bf16),
                LutMode.LOOKUP8_4:OrPattern(DType.i8,
                DType.u8),
                LutMode.LOOKUP16_5:OrPattern(DType.i16,
                DType.u16,
                DType.f16,
                DType.bf16),
                LutMode.LOOKUP8_5:OrPattern(DType.i8,
                DType.u8)}),)))
    source_bank=ParamDef(kind='attribute',
        annotation=Bank,
        pattern=WildcardPattern('source_bank',predicates=(OrPattern(Bank.X,
                Bank.Y),)),
        default=Bank.X)
    table_bank=ParamDef(kind='attribute',
        annotation=Bank,
        pattern=WildcardPattern('table_bank',predicates=(OrPattern(Bank.X,
                Bank.Y),)),
        default=Bank.Y)
    destination=ParamDef(kind='attribute',
        annotation=Bank,
        pattern=WildcardPattern('destination', predicates=(OrPattern(Bank.X, Bank.Y, Bank.Z), (variable('destination') != Bank.Z) | (variable('mode') >= LutMode.LOOKUP32_2))),
        default=Bank.X)
    source_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('source_offset',predicates=(RangePattern(lo=0,hi=511),)),
        default=0)
    table_register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('table_register',predicates=(RangePattern(lo=0,hi=7),)),
        default=1)
    destination_register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('destination_register',predicates=(SwitchPattern('destination',{Bank.X:RangePattern(lo=0,hi=7),
                Bank.Y:RangePattern(lo=0,hi=7),
                Bank.Z:RangePattern(lo=0,hi=63)}), variable('table_lanes') == 512 // Bits('dtype'), SwitchPattern('mode', {m: WildcardPattern(predicates=(variable('source_lanes') == 512 // Bits('dtype'), variable('destination_lanes') == 64)) if int(m) <= 6 else WildcardPattern(predicates=(variable('source_lanes') == {7:4,8:8,9:16,10:4,11:8,12:16,13:32,14:20,15:40}[int(m)], variable('destination_lanes') == 512 // Bits('dtype'))) for m in LutMode}))),
        default=2)
    source=ParamDef(kind='input', effect=R,
        pattern=TensorPattern(shape=(WildcardPattern('source_lanes'),),
            dtype=SwitchPattern('mode', {m: WildcardPattern('dtype') if int(m) <= 6 else DType.u8 for m in LutMode}),
            storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('source_lanes'),), (1,))))
    table=ParamDef(kind='input', effect=R,
        pattern=TensorPattern(shape=(WildcardPattern('table_lanes'),), dtype=WildcardPattern('dtype'),
            storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('table_lanes'),), (1,))))
    dst=ParamDef(kind='input', effect=W,
        pattern=TensorPattern(shape=(WildcardPattern('destination_lanes'),),
            dtype=SwitchPattern('mode', {m: DType.u8 if int(m) <= 6 else WildcardPattern('dtype') for m in LutMode}),
            storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('destination_lanes'),), (1,))))
    def __init__(self,**attrs): bind(self,attrs)

GenlutV1, GenlutV2 = instruction_versions(Genlut, (1, 2), name='genlut', category='arith')
