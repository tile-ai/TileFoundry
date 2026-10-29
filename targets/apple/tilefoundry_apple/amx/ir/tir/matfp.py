from tilefoundry.visitor_registry.candidates import HirMatch
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.core import OpCapability
from tilefoundry.ir.core.param_def import ParamDef,MemoryEffect
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import TensorPattern,LayoutPattern,SwitchPattern,WildcardPattern,OrPattern,RangePattern
from tilefoundry.ir.types import DType,StorageKind,Mesh,Topology,Layout
from tilefoundry.ir.pattern.predicates import Bits, Table, In, variable
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

class Matfp(AmxOp):
    hir_matches = tuple(
        HirMatch(MatMul, when={'a_layout': 'MK', 'b_layout': 'KN'}, attributes={'fp_mode': mode, 'alu': FPALU.ADD_PRODUCT}, operands={'acc': 'result', 'lhs': 'lhs', 'rhs': 'rhs'})
        for mode in FPMode
    )
    mnemonic='MATFP'
    capability=OpCapability('amx_matfp')
    resource='amx_engine'
    execution_mesh=AMX
    execution_mesh_pattern=staticmethod(amx_mesh_pattern)
    roles=('acc','lhs','rhs','table')
    writes=('acc',)
    register_roles={'acc':'z','lhs':'y','rhs':'x','table':'indexed_bank'}
    fp_mode=ParamDef(kind='attribute',
        annotation=FPMode,
        pattern=WildcardPattern('fp_mode', predicates=((variable('amx_generation') >= 2) | ~In(variable('fp_mode'), (FPMode.BF16, FPMode.BF16_F32)),)),
        default=FPMode.F32)
    dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('dtype', predicates=((variable('dtype') == Table((DType.bf16, DType.bf16, DType.f16, DType.f16, DType.f32, DType.f64, DType.f64, DType.f64))[variable('fp_mode')]), (variable('x_lanes') == 512 // Bits('dtype')), (variable('y_lanes') == 512 // Bits('dtype')),)),
        default=DType.f32)
    acc_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('acc_dtype', predicates=((variable('acc_dtype') == Table((DType.bf16, DType.f32, DType.f16, DType.f32, DType.f32, DType.f64, DType.f64, DType.f64))[variable('fp_mode')]),)),
        default=DType.f32)
    alu=ParamDef(kind='attribute',
        annotation=FPALU,
        pattern=WildcardPattern('alu', predicates=(In(variable('alu'), (FPALU.ADD_PRODUCT, FPALU.SUB_PRODUCT, FPALU.POSITIVE_SELECT)), (variable('amx_generation') >= 2) | In(variable('alu'), (FPALU.ADD_PRODUCT, FPALU.SUB_PRODUCT, FPALU.POSITIVE_SELECT, FPALU.MIN, FPALU.MAX)),)),
        default=FPALU.ADD_PRODUCT)
    index_mode=ParamDef(kind='attribute',
        annotation=IndexMode,
        pattern=WildcardPattern('index_mode', predicates=((variable('alu') == FPALU.ADD_PRODUCT) | (variable('index_mode') == IndexMode.NONE),)),
        default=IndexMode.NONE)
    index_register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('index_register', predicates=(RangePattern(lo=0, hi=7), (variable('index_mode') != IndexMode.NONE) | (variable('index_register') == 0),)),
        default=0)
    x_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('x_offset',predicates=(RangePattern(lo=0,hi=511),)),
        default=0)
    y_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('y_offset',predicates=(RangePattern(lo=0,hi=511),)),
        default=64)
    z_row=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('z_row',predicates=(RangePattern(lo=0,hi=7),)),
        default=0)
    mask=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('mask',predicates=(RangePattern(lo=0,hi=511),)),
        default=0)
    y_mask=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('y_mask',predicates=(RangePattern(lo=0,hi=511),)),
        default=0)
    shuffle_x=ParamDef(kind='attribute',
        annotation=Shuffle,
        pattern=WildcardPattern('shuffle_x'),
        default=Shuffle.NONE)
    shuffle_y=ParamDef(kind='attribute',
        annotation=Shuffle,
        pattern=WildcardPattern('shuffle_y'),
        default=Shuffle.NONE)
    repeat=ParamDef(kind='attribute',
        annotation=Repeat,
        pattern=WildcardPattern('repeat', predicates=(Repeat.ONE,)),
        default=Repeat.ONE)
    broadcast_mode=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('broadcast_mode', predicates=(RangePattern(lo=0, hi=7), (variable('repeat') != Repeat.ONE) | (variable('broadcast_mode') == 0), (variable('rows') == 1), (variable('z_step') == 512 // Bits('acc_dtype')), (((((variable('acc_rows') == 512 // Bits('dtype'))) & ((variable('acc_lanes') == 512 // Bits('dtype'))) & ((variable('acc_ratio') == Bits('acc_dtype') // Bits('dtype'))) & ((variable('acc_inner') == 512 // Bits('acc_dtype'))) & ((variable('acc_wide') == Table((0, 0, 1, 1, 1))[Bits('acc_dtype') // Bits('dtype')])))) & (((((Bits('acc_dtype') == Bits('dtype')) & ((variable('acc_step') == 64)))) | (((Bits('acc_dtype') != Bits('dtype')) & ((variable('acc_step') == 512 // Bits('dtype')))))))), (variable('table_side') == Table((0, 1, 1, 1, 1))[variable('index_mode')]), (variable('lhs_count') == ((512 // Bits('dtype')) * Table((8, 8, 2, 8, 4))[variable('index_mode')] + 7) // 8), (variable('lhs_packed') == Table((0, 0, 1, 0, 1))[variable('index_mode')]), (variable('rhs_count') == ((512 // Bits('dtype')) * Table((8, 2, 8, 4, 8))[variable('index_mode')] + 7) // 8), (variable('rhs_packed') == Table((0, 1, 0, 1, 0))[variable('index_mode')]),)),
        default=0)
    acc=ParamDef(kind='input', effect=R|W, pattern=TensorPattern(shape=(WildcardPattern('acc_rows'), WildcardPattern('acc_lanes')), dtype=WildcardPattern('acc_dtype'), storage=StorageKind.RMEM, layout=SwitchPattern('acc_wide', {0: LayoutPattern((WildcardPattern('acc_rows'), WildcardPattern('acc_lanes')), (WildcardPattern('acc_step'), 1)), 1: LayoutPattern((WildcardPattern('acc_rows'), (WildcardPattern('acc_ratio'), WildcardPattern('acc_inner'))), (WildcardPattern('acc_step'), (WildcardPattern('z_step'), 1)))})))
    lhs=ParamDef(kind='input', effect=R, pattern=SwitchPattern('lhs_packed', {0: TensorPattern(shape=(WildcardPattern('y_lanes'), 1), dtype=WildcardPattern('dtype'), storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('y_lanes'), 1), (1, 0))), 1: TensorPattern(shape=(WildcardPattern('lhs_count'),), dtype=DType.u8, storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('lhs_count'),), (1,)))}))
    rhs=ParamDef(kind='input', effect=R, pattern=SwitchPattern('rhs_packed', {0: TensorPattern(shape=(1, WildcardPattern('x_lanes')), dtype=WildcardPattern('dtype'), storage=StorageKind.RMEM, layout=LayoutPattern((1, WildcardPattern('x_lanes')), (0, 1))), 1: TensorPattern(shape=(WildcardPattern('rhs_count'),), dtype=DType.u8, storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('rhs_count'),), (1,)))}))
    table=ParamDef(kind='input', effect=R, pattern=SwitchPattern('table_side', {0: None, 1: TensorPattern(shape=(WildcardPattern('x_lanes'),), dtype=WildcardPattern('dtype'), storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('x_lanes'),), (1,)))}), optional=True, default=None)
    def __init__(self,**attrs): bind(self,attrs)

MatfpV1, MatfpV2 = instruction_versions(Matfp, (1, 2), name='matfp', category='nn')
