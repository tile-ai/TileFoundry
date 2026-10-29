from tilefoundry.visitor_registry.candidates import HirMatch
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.core import OpCapability
from tilefoundry.ir.core.param_def import ParamDef,MemoryEffect
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import TensorPattern,LayoutPattern,SwitchPattern,WildcardPattern,OrPattern,RangePattern
from tilefoundry.ir.types import DType,StorageKind
from tilefoundry.ir.pattern.predicates import Bits, Table, In, variable
from .common import (
    instruction_versions,
    AmxOp,
    ProductMode,IndexMode,TransferCount,Repeat,Shuffle,FPMode,IntLane,
    FPALU,IntALU,Bank,ExtractMode,LutMode,
    FLOATS,TYPES,I8,I16,I32,
    bind,encode,
    vector_layout,xy_matrix_layout,z_matrix_layout,z_vector_layout,
    vector_input_layout,packed_input_layout,int_matrix_layout,int_vector_layout,
    extract_source_shape,extract_source_layout,
)
R=MemoryEffect.READ
W=MemoryEffect.WRITE


class Vecint(AmxOp):
    hir_matches = tuple(
        HirMatch(Binary, when={'kind': BinaryKind.MUL}, attributes={'lane_mode': lane, 'alu': IntALU.PRODUCT}, operands={'acc': 'result', 'lhs': 'lhs', 'rhs': 'rhs'})
        for lane in IntLane
    ) + tuple(
        HirMatch(Binary, when={'kind': kind}, attributes={'lane_mode': lane, 'alu': alu}, operands={'acc': 'lhs', 'lhs': 'rhs', 'rhs': 'rhs'})
        for lane in IntLane
        for kind, alu in ((BinaryKind.ADD, IntALU.ADD_X),)
    ) + tuple(
        HirMatch(Binary, when={'kind': BinaryKind.SUB}, attributes={'lane_mode': lane, 'alu': IntALU.SUB_SUM, 'mask': 5, 'shift': 0}, operands={'acc': 'lhs', 'lhs': 'rhs', 'rhs': 'rhs'})
        for lane in IntLane
    )
    mnemonic='VECINT'
    capability=OpCapability('amx_vecint')
    resource='amx_engine'
    roles=('acc','lhs','rhs','table')
    writes=('acc',)
    register_roles={'acc':'z','lhs':'y','rhs':'x','table':'indexed_bank'}
    alu=ParamDef(kind='attribute',
        annotation=IntALU,
        pattern=WildcardPattern('alu', predicates=(In(variable('alu'), (IntALU.ADD_PRODUCT, IntALU.SUB_PRODUCT, IntALU.ADD_SUM, IntALU.SUB_SUM, IntALU.SHIFT, IntALU.SQRDMLAH, IntALU.SQRDMLSH, IntALU.PRODUCT, IntALU.ADD_X, IntALU.ADD_Y)), (variable('amx_generation') >= 2) | (variable('alu') < 10),)),
        default=IntALU.ADD_PRODUCT)
    lane_mode=ParamDef(kind='attribute',
        annotation=IntLane,
        pattern=WildcardPattern('lane_mode', predicates=((((((variable('alu') == IntALU.SHIFT)) & (In(variable('lane_mode'), (IntLane.NARROW16, IntLane.WIDE16, IntLane.I32, IntLane.I8_I8_I32, IntLane.I8_I8_I16, IntLane.I8))))) | (((In(variable('alu'), (IntALU.SQRDMLAH, IntALU.SQRDMLSH))) & ((variable('lane_mode') == IntLane.NARROW16)))) | (((variable('alu') != IntALU.SHIFT) & (~(In(variable('alu'), (IntALU.SQRDMLAH, IntALU.SQRDMLSH)))) & (In(variable('lane_mode'), (IntLane.NARROW16, IntLane.WIDE16, IntLane.I8_I8_I32, IntLane.I8_I8_I16, IntLane.I8_I16_I32, IntLane.I16_I8_I32)))))),)),
        default=IntLane.NARROW16)
    x_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('x_dtype', predicates=((((((variable('alu') == IntALU.SHIFT)) & ((variable('x_dtype') == DType.u8)))) | (((variable('alu') != IntALU.SHIFT) & (((In(variable('x_dtype'), (*I8, *I16, *I32))) & (Bits('x_dtype') == Table((0, 0, 16, 16, 32, 0, 0, 0, 0, 8, 8, 8, 8, 16))[variable('lane_mode')])))))),)),
        default=DType.i16)
    y_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('y_dtype', predicates=((((((variable('alu') == IntALU.SHIFT)) & ((variable('y_dtype') == DType.u8)))) | (((variable('alu') != IntALU.SHIFT) & (((In(variable('y_dtype'), (*I8, *I16, *I32))) & (Bits('y_dtype') == Table((0, 0, 16, 16, 32, 0, 0, 0, 0, 8, 8, 8, 16, 8))[variable('lane_mode')])))))), (variable('x_lanes') == 512 // Bits('x_dtype')), (variable('y_lanes') == 512 // Bits('y_dtype')), ((variable('input_bits') <= Bits('x_dtype')) & (variable('input_bits') <= Bits('y_dtype')) & (((variable('input_bits') == Bits('x_dtype')) | (variable('input_bits') == Bits('y_dtype'))))),)),
        default=DType.i16)
    acc_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=WildcardPattern('acc_dtype', predicates=(((In(variable('acc_dtype'), (*I8, *I16, *I32))) & (Bits('acc_dtype') == Table((0, 0, 16, 32, 32, 0, 0, 0, 0, 8, 32, 16, 32, 32))[variable('lane_mode')])),)),
        default=DType.i16)
    index_mode=ParamDef(kind='attribute',
        annotation=IndexMode,
        pattern=WildcardPattern('index_mode', predicates=(In(variable('alu'), (IntALU.ADD_PRODUCT, IntALU.WIDE_PRODUCT)) | (variable('index_mode') == IndexMode.NONE),)),
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
        pattern=WildcardPattern('z_row',predicates=(RangePattern(lo=0,hi=63),)),
        default=0)
    shift=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('shift',predicates=(RangePattern(lo=0,hi=31),)),
        default=0)
    mask=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('mask',predicates=(RangePattern(lo=0,hi=511),)),
        default=0)
    mask_y=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('mask_y',predicates=(False,)),
        default=False)
    saturate=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('saturate', predicates=(In(variable('saturate'), (False, True)), (variable('alu') == IntALU.SHIFT) | (variable('saturate') == False),)),
        default=False)
    rounding=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('rounding', predicates=(In(variable('rounding'), (False, True)), (variable('alu') == IntALU.SHIFT) | (variable('rounding') == False),)),
        default=False)
    signed_output=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('signed_output', predicates=(In(variable('signed_output'), (False, True)), (variable('alu') == IntALU.SHIFT) | (variable('signed_output') == False),)),
        default=False)
    shuffle_x=ParamDef(kind='attribute',
        annotation=Shuffle,
        pattern=WildcardPattern('shuffle_x', predicates=((variable('alu') != IntALU.SHIFT) | (variable('shuffle_x') == Shuffle.NONE),)),
        default=Shuffle.NONE)
    shuffle_y=ParamDef(kind='attribute',
        annotation=Shuffle,
        pattern=WildcardPattern('shuffle_y', predicates=((variable('alu') != IntALU.SHIFT) | (variable('shuffle_y') == Shuffle.NONE),)),
        default=Shuffle.NONE)
    repeat=ParamDef(kind='attribute',
        annotation=Repeat,
        pattern=WildcardPattern('repeat', predicates=((variable('amx_generation') >= 2) | (variable('repeat') == Repeat.ONE),)),
        default=Repeat.ONE)
    broadcast_mode=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('broadcast_mode', predicates=(RangePattern(lo=0, hi=7), (variable('repeat') != Repeat.ONE) | (variable('broadcast_mode') == 0), (variable('rows') == Table((0, 1, 2, 0, 4))[variable('repeat')]), (variable('z_step') == 512 // Bits('acc_dtype')), (((((variable('alu') == IntALU.SHIFT)) & ((((variable('acc_rows') == Table((0, 1, 2, 0, 4))[variable('repeat')])) & ((variable('acc_lanes') == 512 // Bits('acc_dtype'))) & ((variable('acc_step') == (64 // Table((0, 1, 2, 0, 4))[variable('repeat')]) * 512 // Bits('acc_dtype'))) & ((variable('acc_ratio') == 1)) & ((variable('acc_inner') == 512 // Bits('acc_dtype'))) & ((variable('acc_wide') == 0)))))) | (((variable('alu') != IntALU.SHIFT) & ((((((variable('acc_rows') == Table((0, 1, 2, 0, 4))[variable('repeat')])) & ((variable('acc_lanes') == 512 // variable('input_bits'))) & ((variable('acc_ratio') == Bits('acc_dtype') // variable('input_bits'))) & ((variable('acc_inner') == 512 // Bits('acc_dtype'))) & ((variable('acc_wide') == Table((0, 0, 1, 1, 1))[Bits('acc_dtype') // variable('input_bits')])))) & ((variable('acc_step') == (64 // Table((0, 1, 2, 0, 4))[variable('repeat')]) * 512 // Bits('acc_dtype')))))))), (variable('has_inputs') == Table((1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1))[variable('alu')]), (variable('table_side') == Table((0, 1, 2, 1, 2))[variable('index_mode')]), (variable('lhs_count') == ((512 // Bits('y_dtype')) * Table((8, 8, 2, 8, 4))[variable('index_mode')] + 7) // 8), (variable('lhs_packed') == Table((0, 0, 1, 0, 1))[variable('index_mode')]), (variable('lhs_row') == (((512 // Bits('y_dtype')) * Table((8, 8, 2, 8, 4))[variable('index_mode')] + 7) // 8) * Table((1, 1, 1, 0, 1, 1, 1, 0))[variable('broadcast_mode')]), (variable('rhs_count') == ((512 // Bits('x_dtype')) * Table((8, 2, 8, 4, 8))[variable('index_mode')] + 7) // 8), (variable('rhs_packed') == Table((0, 1, 0, 1, 0))[variable('index_mode')]), (variable('rhs_row') == (((512 // Bits('x_dtype')) * Table((8, 2, 8, 4, 8))[variable('index_mode')] + 7) // 8) * Table((1, 1, 0, 1, 1, 1, 0, 1))[variable('broadcast_mode')]),)),
        default=0)
    acc=ParamDef(kind='input', effect=R|W, pattern=TensorPattern(shape=(WildcardPattern('acc_rows'), WildcardPattern('acc_lanes')), dtype=WildcardPattern('acc_dtype'), storage=StorageKind.RMEM, layout=SwitchPattern('acc_wide', {0: LayoutPattern((WildcardPattern('acc_rows'), WildcardPattern('acc_lanes')), (WildcardPattern('acc_step'), 1)), 1: LayoutPattern((WildcardPattern('acc_rows'), (WildcardPattern('acc_ratio'), WildcardPattern('acc_inner'))), (WildcardPattern('acc_step'), (WildcardPattern('z_step'), 1)))})))
    lhs=ParamDef(kind='input', effect=R, pattern=SwitchPattern('has_inputs', {0: None, 1: TensorPattern(shape=(WildcardPattern('rows'), WildcardPattern('lhs_count')), dtype=SwitchPattern('lhs_packed', {0: WildcardPattern('y_dtype'), 1: DType.u8}), storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('rows'), WildcardPattern('lhs_count')), (WildcardPattern('lhs_row'), 1)))}), optional=True, default=None)
    rhs=ParamDef(kind='input', effect=R, pattern=SwitchPattern('has_inputs', {0: None, 1: TensorPattern(shape=(WildcardPattern('rows'), WildcardPattern('rhs_count')), dtype=SwitchPattern('rhs_packed', {0: WildcardPattern('x_dtype'), 1: DType.u8}), storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('rows'), WildcardPattern('rhs_count')), (WildcardPattern('rhs_row'), 1)))}), optional=True, default=None)
    table=ParamDef(kind='input', effect=R, pattern=SwitchPattern('table_side', {0: None, 1: TensorPattern(shape=(WildcardPattern('x_lanes'),), dtype=WildcardPattern('x_dtype'), storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('x_lanes'),), (1,))), 2: TensorPattern(shape=(WildcardPattern('y_lanes'),), dtype=WildcardPattern('y_dtype'), storage=StorageKind.RMEM, layout=LayoutPattern((WildcardPattern('y_lanes'),), (1,)))}), optional=True, default=None)
    def __init__(self,**attrs): bind(self,attrs)

VecintV1, VecintV2, VecintV3 = instruction_versions(Vecint, (1, 2, 4), name='vecint', category='arith')
