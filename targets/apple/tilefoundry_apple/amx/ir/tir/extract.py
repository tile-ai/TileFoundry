from tilefoundry.visitor_registry.candidates import HirMatch
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.core import OpCapability
from tilefoundry.ir.core.param_def import ParamDef,MemoryEffect
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import TensorPattern,LayoutPattern,SwitchPattern,WildcardPattern,OrPattern,RangePattern
from tilefoundry.ir.types import DType,StorageKind
from tilefoundry.ir.pattern.predicates import Bits, In, Table, variable
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



_EXTRACT_SRC_DTYPE = WildcardPattern('src_dtype', predicates=(
    SwitchPattern('mode', {
        ExtractMode.COPY: OrPattern(*TYPES),
        ExtractMode.LEGACY: OrPattern(DType.i16, DType.u16, DType.f16,
                                     DType.i32, DType.u32, DType.f32, DType.f64),
        ExtractMode.INTEGER: OrPattern(*I8, *I16, *I32),
        ExtractMode.FLOAT: OrPattern(DType.f16, DType.f32, DType.f64),
    }),
))

_EXTRACT_DST_DTYPE = WildcardPattern('dst_dtype', predicates=(
    SwitchPattern('mode', {
        ExtractMode.COPY: WildcardPattern('src_dtype'),
        ExtractMode.LEGACY: WildcardPattern('src_dtype'),
        ExtractMode.INTEGER: WildcardPattern(predicates=(
            In(variable('dst_dtype'), (*I8, *I16, *I32)),
            Bits('dst_dtype') <= Bits('src_dtype'),
        )),
        ExtractMode.FLOAT: SwitchPattern('src_dtype', {
            DType.f16: DType.f16,
            DType.f32: OrPattern(DType.f32, DType.f16, DType.bf16),
            DType.f64: DType.f64,
        }),
    }),
    (variable('mode') != ExtractMode.FLOAT) | (variable('amx_generation') >= 2)
        | (variable('dst_dtype') == variable('src_dtype')),
))

class Extrx(AmxOp):
    hir_matches = (
        HirMatch(Reshard, attributes={'mode': ExtractMode.COPY}, operands={'src': 'x', 'dst': 'result'}),
        HirMatch(Cast, attributes={'mode': ExtractMode.INTEGER, 'shift': 0, 'rounding': False, 'saturate': False}, operands={'src': 'x', 'dst': 'result'}),
        HirMatch(Cast, attributes={'mode': ExtractMode.FLOAT}, operands={'src': 'x', 'dst': 'result'}),
    )
    mnemonic='EXTRX'
    capability=OpCapability('amx_extrx')
    resource='amx_engine'
    roles=('src','dst')
    writes=('dst',)
    register_roles={'src':'extract_source','dst':'destination'}
    mode=ParamDef(kind='attribute',
        annotation=ExtractMode,
        pattern=WildcardPattern('mode'),
        default=ExtractMode.LEGACY)
    src_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=_EXTRACT_SRC_DTYPE,
        default=DType.f32)
    dst_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=_EXTRACT_DST_DTYPE,
        default=DType.f32)
    destination=ParamDef(kind='attribute',
        annotation=Bank,
        pattern=WildcardPattern('destination', predicates=(OrPattern(Bank.X, Bank.Y), In(variable('mode'), (ExtractMode.INTEGER, ExtractMode.FLOAT)) | (variable('destination') == Bank.X))),
        default=Bank.X)
    z_selector=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('z_selector', predicates=(RangePattern(lo=0, hi=63), (variable('mode') != ExtractMode.COPY) | (variable('z_selector') <= 7))),
        default=0)
    destination_register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('destination_register', predicates=(RangePattern(lo=0, hi=7), (variable('mode') == ExtractMode.COPY) | (variable('destination_register') == 0))),
        default=0)
    destination_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('destination_offset', predicates=(RangePattern(lo=0, hi=511), (variable('mode') != ExtractMode.COPY) | (variable('destination_offset') == 0))),
        default=0)
    repeat=ParamDef(kind='attribute',
        annotation=Repeat,
        pattern=WildcardPattern('repeat', predicates=((variable('repeat') == Repeat.ONE) | ((variable('amx_generation') >= 2) & In(variable('mode'), (ExtractMode.INTEGER, ExtractMode.FLOAT))),)),
        default=Repeat.ONE)
    alternating=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('alternating', predicates=((variable('alternating') == False) | (In(variable('mode'), (ExtractMode.INTEGER, ExtractMode.FLOAT)) & (Bits('src_dtype') == 32)),)),
        default=False)
    low_byte_only=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('low_byte_only', predicates=((variable('low_byte_only') == False) | ((variable('mode') == ExtractMode.LEGACY) & (Bits('src_dtype') == 16)),)),
        default=False)
    shift=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('shift', predicates=(RangePattern(lo=0, hi=31), (variable('mode') == ExtractMode.INTEGER) | (variable('shift') == 0))),
        default=0)
    rounding=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('rounding', predicates=((variable('rounding') == False) | (variable('mode') == ExtractMode.INTEGER),)),
        default=False)
    saturate=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('saturate', predicates=((variable('saturate') == False) | (variable('mode') == ExtractMode.INTEGER),)),
        default=False)
    mask=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('mask',predicates=(SwitchPattern('mode',{ExtractMode.COPY:0,
                ExtractMode.LEGACY:RangePattern(lo=0,hi=127),
                ExtractMode.INTEGER:RangePattern(lo=0,hi=511),
                ExtractMode.FLOAT:RangePattern(lo=0,hi=511)}), variable('src_lanes') == 64 // (Bits('src_dtype') // 8),
            variable('dst_lanes') == 64 // (Bits('dst_dtype') // 8),
            variable('rows') == Table((0, 1, 2, 0, 4))[variable('repeat')],
            variable('extract_ratio') == Bits('src_dtype') // Bits('dst_dtype'),
            variable('extract_inner') == 512 // Bits('src_dtype'),
            variable('source_row') == (64 // Table((0, 1, 2, 0, 4))[variable('repeat')]) * 64 // (Bits('src_dtype') // 8),
            SwitchPattern('alternating', {False: WildcardPattern(predicates=(variable('extract_step') == 64 // (Bits('src_dtype') // 8),)),
                True: SwitchPattern('dst_dtype', {d: WildcardPattern(predicates=(variable('extract_step') == (2 if d.bit_width == 16 else 1) * 64 // (Bits('src_dtype') // 8),)) for d in TYPES})}))),
        default=0)
    src=ParamDef(kind='input', effect=R,
        pattern=SwitchPattern('mode', {
            ExtractMode.COPY: TensorPattern(shape=(1, WildcardPattern('src_lanes')), dtype=WildcardPattern('src_dtype'), storage=StorageKind.RMEM,
                layout=LayoutPattern((1, WildcardPattern('src_lanes')), (0, 1))),
            ExtractMode.LEGACY: TensorPattern(shape=(1, 1, WildcardPattern('src_lanes')), dtype=WildcardPattern('src_dtype'), storage=StorageKind.RMEM,
                layout=LayoutPattern((1, 1, WildcardPattern('src_lanes')), (WildcardPattern('source_row'), WildcardPattern('src_lanes'), 1))),
            **{m: TensorPattern(shape=(WildcardPattern('rows'), WildcardPattern('dst_lanes')), dtype=WildcardPattern('src_dtype'), storage=StorageKind.RMEM,
                layout=SwitchPattern('extract_ratio', {
                    1: LayoutPattern((WildcardPattern('rows'), WildcardPattern('dst_lanes')), (WildcardPattern('source_row'), 1)),
                    **{ratio: LayoutPattern((WildcardPattern('rows'), (WildcardPattern('extract_inner'), WildcardPattern('extract_ratio'))),
                        (WildcardPattern('source_row'), (1, WildcardPattern('extract_step')))) for ratio in (2, 4)}}))
                for m in (ExtractMode.INTEGER, ExtractMode.FLOAT)}}))
    dst=ParamDef(kind='input', effect=R|W,
        pattern=TensorPattern(shape=(WildcardPattern('rows'), WildcardPattern('dst_lanes')), dtype=WildcardPattern('dst_dtype'), storage=StorageKind.RMEM,
            layout=LayoutPattern((WildcardPattern('rows'), WildcardPattern('dst_lanes')), (WildcardPattern('dst_lanes'), 1))))
    def __init__(self,**attrs): bind(self,attrs)


class Extry(AmxOp):
    hir_matches = (
        HirMatch(Reshard, attributes={'mode': ExtractMode.COPY}, operands={'src': 'x', 'dst': 'result'}),
        HirMatch(Cast, attributes={'mode': ExtractMode.INTEGER, 'shift': 0, 'rounding': False, 'saturate': False}, operands={'src': 'x', 'dst': 'result'}),
        HirMatch(Cast, attributes={'mode': ExtractMode.FLOAT}, operands={'src': 'x', 'dst': 'result'}),
    )
    mnemonic='EXTRY'
    capability=OpCapability('amx_extry')
    resource='amx_engine'
    roles=('src','dst')
    writes=('dst',)
    register_roles={'src':'extract_source','dst':'destination'}
    mode=ParamDef(kind='attribute',
        annotation=ExtractMode,
        pattern=WildcardPattern('mode'),
        default=ExtractMode.LEGACY)
    src_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=_EXTRACT_SRC_DTYPE,
        default=DType.f32)
    dst_dtype=ParamDef(kind='attribute',
        annotation=DType,
        pattern=_EXTRACT_DST_DTYPE,
        default=DType.f32)
    destination=ParamDef(kind='attribute',
        annotation=Bank,
        pattern=WildcardPattern('destination', predicates=(OrPattern(Bank.X, Bank.Y), In(variable('mode'), (ExtractMode.INTEGER, ExtractMode.FLOAT)) | (variable('destination') == Bank.Y))),
        default=Bank.Y)
    z_selector=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('z_selector', predicates=(RangePattern(lo=0, hi=63), (variable('mode') != ExtractMode.COPY) | (variable('z_selector') <= 7))),
        default=0)
    destination_register=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('destination_register', predicates=(RangePattern(lo=0, hi=7), (variable('mode') == ExtractMode.COPY) | (variable('destination_register') == 0))),
        default=0)
    destination_offset=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('destination_offset', predicates=(RangePattern(lo=0, hi=511), (variable('mode') != ExtractMode.COPY) | (variable('destination_offset') == 0))),
        default=0)
    repeat=ParamDef(kind='attribute',
        annotation=Repeat,
        pattern=WildcardPattern('repeat', predicates=((variable('repeat') == Repeat.ONE) | ((variable('amx_generation') >= 2) & In(variable('mode'), (ExtractMode.INTEGER, ExtractMode.FLOAT))),)),
        default=Repeat.ONE)
    alternating=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('alternating', predicates=((variable('alternating') == False) | (In(variable('mode'), (ExtractMode.INTEGER, ExtractMode.FLOAT)) & (Bits('src_dtype') == 32)),)),
        default=False)
    low_byte_only=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('low_byte_only', predicates=((variable('low_byte_only') == False) | ((variable('mode') == ExtractMode.LEGACY) & (Bits('src_dtype') == 16)),)),
        default=False)
    shift=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('shift', predicates=(RangePattern(lo=0, hi=31), (variable('mode') == ExtractMode.INTEGER) | (variable('shift') == 0))),
        default=0)
    rounding=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('rounding', predicates=((variable('rounding') == False) | (variable('mode') == ExtractMode.INTEGER),)),
        default=False)
    saturate=ParamDef(kind='attribute',
        annotation=bool,
        pattern=WildcardPattern('saturate', predicates=((variable('saturate') == False) | (variable('mode') == ExtractMode.INTEGER),)),
        default=False)
    mask=ParamDef(kind='attribute',
        annotation=int,
        pattern=WildcardPattern('mask',predicates=(SwitchPattern('mode',{ExtractMode.COPY:0,
                ExtractMode.LEGACY:RangePattern(lo=0,hi=127),
                ExtractMode.INTEGER:RangePattern(lo=0,hi=511),
                ExtractMode.FLOAT:RangePattern(lo=0,hi=511)}), variable('src_lanes') == 64 // (Bits('src_dtype') // 8),
            variable('dst_lanes') == 64 // (Bits('dst_dtype') // 8),
            variable('rows') == Table((0, 1, 2, 0, 4))[variable('repeat')],
            variable('extract_ratio') == Bits('src_dtype') // Bits('dst_dtype'),
            variable('extract_inner') == 512 // Bits('src_dtype'),
            variable('source_row') == 64 // Table((0, 1, 2, 0, 4))[variable('repeat')] // (Bits('src_dtype') // 8),
            SwitchPattern('alternating', {False: WildcardPattern(predicates=(variable('extract_step') == 64 // (Bits('src_dtype') // 8),)),
                True: SwitchPattern('dst_dtype', {d: WildcardPattern(predicates=(variable('extract_step') == (2 if d.bit_width == 16 else 1) * 64 // (Bits('src_dtype') // 8),)) for d in TYPES})}))),
        default=0)
    src=ParamDef(kind='input', effect=R,
        pattern=SwitchPattern('mode', {
            ExtractMode.COPY: TensorPattern(shape=(1, WildcardPattern('src_lanes')), dtype=WildcardPattern('src_dtype'), storage=StorageKind.RMEM,
                layout=LayoutPattern((1, WildcardPattern('src_lanes')), (0, 1))),
            ExtractMode.LEGACY: TensorPattern(shape=(WildcardPattern('rows'), 64), dtype=WildcardPattern('src_dtype'), storage=StorageKind.RMEM,
                layout=LayoutPattern((WildcardPattern('rows'), 64), (WildcardPattern('source_row'), WildcardPattern('src_lanes')))),
            **{m: TensorPattern(shape=(WildcardPattern('rows'), WildcardPattern('dst_lanes')), dtype=WildcardPattern('src_dtype'), storage=StorageKind.RMEM,
                layout=SwitchPattern('extract_ratio', {
                    1: LayoutPattern((WildcardPattern('rows'), WildcardPattern('dst_lanes')), (WildcardPattern('source_row'), 64)),
                    **{ratio: LayoutPattern((WildcardPattern('rows'), (WildcardPattern('extract_inner'), WildcardPattern('extract_ratio'))),
                        (WildcardPattern('source_row'), (64, WildcardPattern('extract_step')))) for ratio in (2, 4)}}))
                for m in (ExtractMode.INTEGER, ExtractMode.FLOAT)}}))
    dst=ParamDef(kind='input', effect=R|W,
        pattern=TensorPattern(shape=(WildcardPattern('rows'), WildcardPattern('dst_lanes')), dtype=WildcardPattern('dst_dtype'), storage=StorageKind.RMEM,
            layout=LayoutPattern((WildcardPattern('rows'), WildcardPattern('dst_lanes')), (WildcardPattern('dst_lanes'), 1))))
    def __init__(self,**attrs): bind(self,attrs)

ExtrxV1, ExtrxV2, ExtrxV3 = instruction_versions(Extrx, (1, 2, 4), name='extrx', category='memory')

ExtryV1, ExtryV2, ExtryV3 = instruction_versions(Extry, (1, 2, 4), name='extry', category='memory')
