"""macOS arm64 C oracle and independent physical input bridge.

The original C emulator uses AArch64 scalar/SIMD and FP control assembly, but
never emits AMX instructions. It is intentionally not a portable C reference.
Operand encoding and placement below must remain independent of the Evaluator.
"""
from pathlib import Path
from dataclasses import dataclass, field
import random

import torch

from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
import ctypes
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent
OP_NAMES=('LDX','LDY','LDZ','LDZI','STX','STY','STZ','STZI','EXTRX','EXTRY',
          'FMA16','FMA32','FMA64','FMS16','FMS32','FMS64','MAC16','MATFP','MATINT','VECFP','VECINT','GENLUT')

def build_oracle(output_dir):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'oracle.c',*sorted((ROOT/'c').glob('*.c'))]
    output=output_dir/'libamx_reference.dylib'
    command=['clang','-dynamiclib','-O2','-std=gnu11','-ffp-contract=off','-fno-fast-math',
             '-o',str(output),*[str(p) for p in sources]]
    subprocess.run(command,check=True,capture_output=True,text=True)
    manifest={'command':command,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources,ROOT/'c/emulate.h',ROOT/'c/LICENSE']},
              'fpcr':'RNE, DN=1, FZ=0, FZ16=0, AH=0, FIZ=0; FPCR/FPSR restored per call',
              'execution':'original emulate_* C CPU arithmetic only; no test.c, AMX instruction or hardware probe'}
    (output_dir/'oracle-build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output

class COracle:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(path)); self.fn=self.lib.amx_oracle
        self.fn.argtypes=[ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint64,
                          ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t]
        self.fn.restype=ctypes.c_int
    def execute(self,name,state,operand):
        raw=bytearray(state.x+state.y+state.z)
        mem=bytearray(state.memory)
        sb=(ctypes.c_uint8*len(raw)).from_buffer(raw)
        mb=(ctypes.c_uint8*len(mem)).from_buffer(mem)
        result=self.fn(OP_NAMES.index(name),state.version,operand,sb,mb,len(mem))
        if result: raise RuntimeError(f'C oracle {name} failed: {result}')
        state.x[:]=raw[:512];state.y[:]=raw[512:1024];state.z[:]=raw[1024:]
        state.memory[:]=mem
        return state


@dataclass
class _Machine:
    version:int
    x:bytearray=field(default_factory=lambda:bytearray(512))
    y:bytearray=field(default_factory=lambda:bytearray(512))
    z:bytearray=field(default_factory=lambda:bytearray(4096))
    memory:bytearray=field(default_factory=lambda:bytearray(4096))

def _bytes(size,seed):
    # Fixed reproducible input, never an open-ended fuzz search.
    return bytearray(random.Random(seed).randbytes(size))

def _seed(role,type,index):
    size=type.dtype.bit_width//8
    for extent in type.shape: size*=extent
    if role in ('lifecycle','token'): return bytearray(size)
    out=_bytes(size,400+index)
    if type.dtype.name in ('f16','bf16','f32','f64'):
        bits=type.dtype.bit_width; n=bits//8
        eb,fb={'f16':(5,10),'bf16':(8,7),'f32':(8,23),'f64':(11,52)}[type.dtype.name]
        inf=((1<<eb)-1)<<fb; sign=1<<(bits-1); one=((1<<(eb-1))-1)<<fb
        edges=(0,sign,inf,inf|sign,inf|1,inf|(1<<(fb-1)),1,sign|1,(1<<fb)-1,1<<fb)
        # Aligned finite lanes exercise fused cancellation and rounding instead
        # of letting identical NaN/Inf-heavy rows dominate every arithmetic op.
        # (1+ulp)*(1-ulp)-1 = -ulp**2, an exact fused residual.
        bank={'lhs':0,'rhs':1,'acc':2,'source':0,'table':1,'src':0,'dst':2}.get(role,index)
        finite=(
            (one+1,one+1,one+(1<<(fb-2)),sign|one|(1<<(fb-1))),
            (one-2,one+(1<<(fb-1)),sign|one|3,one+(1<<(fb-3))),
            (sign|one,0,one+3,sign|one|1),
            (one+3,sign|one|5,one+(1<<(fb-1)),one-1),
        )[bank%4]
        for row in range(0,size,64):
            for lane in range(min(64,size-row)//n):
                slot=lane%8
                if slot in (0,4):
                    edge_index=(2*(row//64)+2*bank+(slot==4))%len(edges)
                    value=edges[edge_index]
                elif slot in (1,2,5,6):
                    value=finite[(1,2,5,6).index(slot)]
                else:
                    continue  # retain original seeded bits in two lanes per eight
                out[row+lane*n:row+(lane+1)*n]=value.to_bytes(n,'little')
    return out

def _coordinates(shape):
    from itertools import product
    return product(*(range(int(d)) for d in shape))

def _layout_element_offset(shape,strides,coordinate):
    # CuTe nested modes use least-significant submode first; implement locally
    # rather than importing the Evaluator's placement or layout helper.
    if isinstance(shape,int): return coordinate*strides
    total=0
    for extent,stride in zip(shape,strides):
        count=1
        def leaves(x):
            if isinstance(x,tuple):
                for value in x: yield from leaves(value)
            else: yield x
        for size in leaves(extent): count*=size
        total+=_layout_element_offset(extent,stride,coordinate%count)
        coordinate//=count
    return total

def _relative_bytes(type):
    n=type.dtype.bit_width//8
    return tuple(sum(_layout_element_offset(shape,stride,c)
                     for shape,stride,c in zip(type.layout.shape,type.layout.strides,point))*n
                 for point in _coordinates(type.shape))

def _oracle_locations(op,role,type):
    """C-side packing from public relative layouts and named physical selectors.

    Intentionally does not call evaluator/common placements or layout_offsets.
    """
    bank=op.register_roles[role]
    if bank=='extract_source':
        bank=('x' if op.mnemonic=='EXTRY' else 'y') if int(op.mode)==0 else 'z'
    elif bank=='indexed_bank': bank='y' if int(op.index_mode) in (2,4) else 'x'
    elif bank not in ('x','y','z','memory','lifecycle'):
        bank=getattr(op,bank).name.lower()
    origin=0
    if bank=='x': origin=getattr(op,'x_offset',0)
    elif bank=='y': origin=getattr(op,'y_offset',0)
    elif bank=='z': origin=getattr(op,'z_row',0)*64
    if role=='acc' and (op.mnemonic.startswith(('FMA','FMS')) or op.mnemonic=='MAC16') and int(op.mode)==0:
        bits=16 if op.mnemonic in ('FMA16','FMS16','MAC16') else op.acc_dtype.bit_width
        origin=0 if bits==16 and type.dtype.bit_width==32 else (op.z_row%(bits//8))*64
    if op.mnemonic in ('MATFP','MATINT','VECFP','VECINT'):
        if role=='table': origin=op.index_register*64
        elif role=='acc':
            xybytes=op.dtype.bit_width//8 if hasattr(op,'fp_mode') else op.x_dtype.bit_width//8
            if op.mnemonic.startswith('MAT'):
                if op.mnemonic=='MATINT' and int(op.alu)==4:
                    origin=(op.z_row%(type.dtype.bit_width//8))*64
                else:
                    origin=0 if type.dtype.bit_width>xybytes*8 else (op.z_row%xybytes)*64
            else:
                if not hasattr(op,'fp_mode') and int(op.alu)!=4:
                    xybytes=min(op.x_dtype.bit_width,op.y_dtype.bit_width)//8
                row=op.z_row%(64//int(op.repeat))
                if type.dtype.bit_width>xybytes*8 and not (op.mnemonic=='VECINT' and int(op.alu)==4): row&=~((type.dtype.bit_width//8//xybytes)-1)
                origin=row*64
    if op.mnemonic.startswith(('LD','ST')):
        origin=op.memory_offset if bank=='memory' else ((op.register&~1)*64+(op.register&1)*32) if op.mnemonic.endswith('I') else op.register*64
    if op.mnemonic=='GENLUT':
        origin=op.source_offset if role=='source' else (op.table_register if role=='table' else op.destination_register)*64
    if op.mnemonic in ('EXTRX','EXTRY'):
        if int(op.mode)==0:
            origin=(op.z_selector if role=='src' else op.destination_register)*64
        elif role=='dst':
            origin=op.destination_offset
            if op.generation>=4 and int(op.repeat)>1: origin&=-64
        else:
            selector=op.z_selector%(64//int(op.repeat))
            origin=selector&-(op.src_dtype.bit_width//8)
            if op.mnemonic=='EXTRX': origin=(selector if int(op.mode)==1 else origin)*64
    n=type.dtype.bit_width//8
    locations=[]
    for start in _relative_bytes(type):
        for byte in range(n):
            location=origin+start+byte
            if bank in ('x','y'): location%=512
            elif bank=='z' and op.mnemonic.startswith(('LD','ST')): location%=4096
            locations.append(location)
    return bank,tuple(locations)

def _tensor_bytes(tensor):
    return bytearray(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())

def _prepare(op,params,types):
    machine=_Machine(op.generation)
    banks={name:getattr(machine,name) for name in ('x','y','z','memory')}
    banks['lifecycle']=bytearray(1)
    placements={}
    assigned=set()
    for i,(param,type) in enumerate(zip(params,types)):
        bank,locations=_oracle_locations(op,param.name,type)
        raw=_seed(param.name,type,i)
        if op.mnemonic=='CLR' and param.name=='token': raw[:]=bytes([1])*len(raw)
        for address,value in zip(locations,raw,strict=True):
            if not (param.effect and param.effect & MemoryEffect.READ): continue
            key=(bank,address)
            assert 0<=address<len(banks[bank]), (op.bindings,param.name,address)
            # Physical aliases receive one consistent initial byte value.
            if key not in assigned:
                banks[bank][address]=value;assigned.add(key)
        placements[param.name]=(bank,locations)
    args=[]
    for param,type in zip(params,types):
        bank,locations=placements[param.name]
        raw=bytearray(banks[bank][address] for address in locations)
        tensor=(torch.frombuffer(raw,dtype=to_torch_dtype(type.dtype)).clone()
                if raw else torch.empty(0,dtype=to_torch_dtype(type.dtype)))
        args.append(TensorValue(tensor.reshape(tuple(type.shape)),type))
    return machine,banks,placements,tuple(args)

def _c_operand(op):
    """Encode original C controls independently of Python adapter encoding."""
    name=op.mnemonic
    if name.startswith(('FMA','FMS')):
        width=int(name[3:])
        value=(op.y_offset|(op.x_offset<<10)|(op.z_row<<20)|
               (int(op.skip_z)<<27)|(int(op.skip_y)<<28)|(int(op.skip_x)<<29)|
               (op.y_mask<<32)|(op.x_mask<<41)|(int(op.mode)<<63))
        if width==32:
            if op.y_dtype.bit_width==16: value|=1<<60
            if op.x_dtype.bit_width==16: value|=1<<61
        if width==16 and op.acc_dtype.bit_width==32: value|=1<<62
        return value
    if name=='MAC16':
        return (op.y_offset|op.x_offset<<10|op.z_row<<20|int(op.skip_z)<<27|
                int(op.skip_y)<<28|int(op.skip_x)<<29|op.y_mask<<32|op.x_mask<<41|
                op.shift<<55|int(op.y_dtype.bit_width==8)<<60|
                int(op.x_dtype.bit_width==8)<<61|int(op.acc_dtype.bit_width==32)<<62|
                int(op.mode)<<63)
    if name in ('EXTRX','EXTRY'):
        if int(op.mode)==0:
            return 1<<27|op.z_selector<<20|op.destination_register<<(6 if name=='EXTRY' else 16)
        if int(op.mode)==1:
            lane={64:0,32:1,16:3 if op.low_byte_only else 2}[op.src_dtype.bit_width]
            return op.z_selector<<20|lane<<28|op.mask<<(32 if name=='EXTRY' else 41)|op.destination_offset<<(0 if name=='EXTRY' else 10)
        s=op.src_dtype.bit_width//8;d=op.dst_dtype.bit_width//8
        lane={(1,1):0,(2,2):2,(4,4):8,(4,2):10 if op.alternating else 9,(4,1):11,(2,1):13,(8,8):1}[(s,d)]
        selector=op.z_selector
        value=(op.destination_offset|int(op.destination==1)<<10|lane<<11|
               selector<<20|1<<26|op.mask<<32|int(op.rounding)<<54|
               int(op.saturate)<<55|int(op.dst_dtype.name.startswith('i'))<<56|
               int(op.src_dtype.name.startswith('i'))<<57|
               (16 if op.dst_dtype.name=='bf16' else op.shift)<<58|
               int(int(op.mode)==3)<<63)
        if int(op.repeat)>1:
            value|=1<<31
            if int(op.repeat)==4: value|=1<<25
        return value
    if name in ('MATFP','MATINT','VECFP','VECINT'):
        value=op.y_offset|(op.x_offset<<10)|(op.z_row<<20)|(int(op.shuffle_y)<<27)|(int(op.shuffle_x)<<29)|(op.mask<<32)
        value|=(int(op.fp_mode) if hasattr(op,'fp_mode') else int(op.lane_mode))<<42
        if name=='MATFP':
            value|=(op.y_mask&63)<<58|((op.y_mask>>6)&7)<<23
        elif name=='MATINT':
            value|=int(op.mask_y)<<25
        elif int(op.repeat)>1:
            value|=1<<31
            if int(op.repeat)==4: value|=1<<25
            value=(value&~(511<<32))|(op.broadcast_mode<<32)
        if int(op.index_mode):
            value|=1<<53|op.index_register<<49
            if int(op.index_mode) in (2,4): value|=1<<47
            if int(op.index_mode) in (3,4): value|=1<<48
            if name=='MATINT' and int(op.alu)==8: value|=1<<54
        else: value|=int(op.alu)<<47
        if hasattr(op,'shift'):
            value|=op.shift<<58
            if int(op.alu)==4:
                value|=int(op.acc_dtype.name.startswith('i'))<<63|int(op.signed_output)<<26
            else:
                value|=int(op.x_dtype.name.startswith('i'))<<63|int(op.y_dtype.name.startswith('i'))<<26
            if int(op.alu)==4:
                value=(value&~(3<<29))|int(op.rounding)<<29|int(op.saturate)<<30
        return value
    if name.startswith(('LD','ST')):
        return (op.memory_offset|(op.register<<56)|(int(op.count)>1)<<62|
                (int(op.count)==4)<<60|int(op.nonconsecutive)<<61)
    if name=='GENLUT':
        return (op.source_offset|(int(op.source_bank)==1)<<10|
                op.destination_register<<20|(int(op.destination)==1)<<25|
                (int(op.destination)==2)<<26|int(op.bf16)<<30|
                int(op.mode)<<53|(int(op.table_bank)==1)<<59|op.table_register<<60)
    raise AssertionError(f'missing independent C encoding for {name}')
