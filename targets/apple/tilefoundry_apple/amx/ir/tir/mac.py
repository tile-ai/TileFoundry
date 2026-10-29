from tilefoundry.visitor_registry.candidates import HirMatch
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.core import OpCapability
from tilefoundry.ir.core.param_def import ParamDef,MemoryEffect
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import TensorPattern,LayoutPattern,SwitchPattern,WildcardPattern,OrPattern,RangePattern
from tilefoundry.ir.types import DType,StorageKind,Mesh,Topology,Layout
from .common import (
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

@register_op(dialect="T", namespace="amx", category="nn", name="mac16")
class Mac16(AmxOp):
    hir_matches = (
        HirMatch(MatMul, when={'a_layout': 'MK', 'b_layout': 'KN'}, attributes={'mode': ProductMode.MATRIX}, operands={'acc': 'result', 'lhs': 'lhs', 'rhs': 'rhs'}),
        HirMatch(Binary, when={'kind': BinaryKind.MUL}, attributes={'mode': ProductMode.VECTOR, 'skip_z': True}, operands={'acc': 'result', 'lhs': 'lhs', 'rhs': 'rhs'}),
        HirMatch(Binary, when={'kind': BinaryKind.ADD}, attributes={'mode': ProductMode.VECTOR, 'skip_y': True, 'skip_z': False}, operands={'acc': 'lhs', 'lhs': 'rhs', 'rhs': 'rhs'}),
    )
    """One 16-bit instruction; relative physical layouts retain only its logical operands."""
    mnemonic='MAC16'
    capability=OpCapability('amx_mac16')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('acc','lhs','rhs')
    writes=('acc',)
    register_roles={'acc':'z','lhs':'y','rhs':'x'}
    mode=ParamDef(kind='attribute',
        annotation=ProductMode,
        pattern=WildcardPattern('mode'),
        default=ProductMode.MATRIX)
    acc_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('acc_dtype',predicates=(SwitchPattern('mode',{ProductMode.MATRIX:OrPattern(DType.i16,
                DType.u16,
                DType.i32,
                DType.u32),
                ProductMode.VECTOR:OrPattern(DType.i16,
                DType.u16)}),)),
        default=DType.i32)
    x_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('x_dtype',predicates=(OrPattern(DType.i8,
                DType.i16),)),
        default=DType.i16)
    y_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('y_dtype',predicates=(OrPattern(DType.i8,
                DType.i16),)),
        default=DType.i16)
    x_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('x_offset',predicates=(RangePattern(lo=0,hi=511),)),
        default=0)
    y_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('y_offset',predicates=(RangePattern(lo=0,hi=511),)),
        default=0)
    z_row=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('z_row',predicates=(RangePattern(lo=0,hi=63),)),
        default=0)
    skip_x=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('skip_x',predicates=(OrPattern(False,True),)),
        default=False)
    skip_y=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('skip_y',predicates=(OrPattern(False,True),)),
        default=False)
    skip_z=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('skip_z',predicates=(OrPattern(False,True),)),
        default=False)
    x_mask=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('x_mask',predicates=(RangePattern(lo=0,hi=127),)),
        default=0)
    y_mask=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('y_mask',predicates=(SwitchPattern('mode',{ProductMode.MATRIX:RangePattern(lo=0,hi=127),
                ProductMode.VECTOR:0}),)),
        default=0)
    acc=ParamDef(kind='input',
        effect=R|W,
        pattern=TensorPattern(shape=(SwitchPattern('mode',{ProductMode.MATRIX:32,
                ProductMode.VECTOR:1}),32),
            dtype=WildcardPattern('acc_dtype'),
            storage=StorageKind.RMEM,
            layout=SwitchPattern('mode',{ProductMode.MATRIX:SwitchPattern('acc_dtype',{DType.i16:LayoutPattern((32,32),(64,1)),
                DType.u16:LayoutPattern((32,32),(64,1)),
                DType.i32:LayoutPattern((32,(2,16)),(32,(16,1))),
                DType.u32:LayoutPattern((32,(2,16)),(32,(16,1)))}),
                ProductMode.VECTOR:LayoutPattern((1,32),(0,1))})))
    lhs=ParamDef(kind='input',
        effect=R,
        pattern=TensorPattern(
            shape=(SwitchPattern('mode', {ProductMode.MATRIX: 32,
                ProductMode.VECTOR: 1}), SwitchPattern('mode', {ProductMode.MATRIX: 1,
                ProductMode.VECTOR: 32})),
            dtype=WildcardPattern('y_dtype'),
            storage=StorageKind.RMEM,
            layout=SwitchPattern('mode', {ProductMode.MATRIX: LayoutPattern((32, 1), (SwitchPattern('y_dtype', {DType.i8: 2,
                DType.i16: 1}), 0)),
                ProductMode.VECTOR: LayoutPattern((1, 32), (0, SwitchPattern('y_dtype', {DType.i8: 2,
                DType.i16: 1})))})))
    rhs=ParamDef(kind='input',
        effect=R,
        pattern=TensorPattern(shape=(1,32),
            dtype=WildcardPattern('x_dtype'),
            storage=StorageKind.RMEM,
            layout=LayoutPattern((1,32),(0,SwitchPattern('x_dtype',{DType.i8:2,
                DType.i16:1})))))
    shift=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('shift',predicates=(RangePattern(lo=0,hi=31),)),
        default=0)
    def __init__(self,**attrs): bind(self,attrs)

