"""The feature-dense programs the parser tests read.

Each program is parsed once, at import, and its recorded golden is that program
printed back as DSL source. A parser feature belongs in the program whose
``features`` list names it, so adding one is adding a line to a body here rather
than a new file. What a golden cannot show — node identity, a target's concrete
``Op`` class, a layout the printer renders as the sugar it was written as — is a
named test in ``test_programs.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.fixtures.logical.hir_composition import Expert
from tests.parser.error_cases import CTX_LEN, Callee
from tilefoundry import func, module, prim_func
from tilefoundry.dsl import ConstTensor, DimVar, DimVarRangePat, T, Tensor, ceildiv, tf
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op names used by the bodies
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor as TensorPattern
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, P, ShardLayout, Split, Topology
from tilefoundry.ir.types.shard import Mesh as TirMesh
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target import CpuTarget, CudaTarget
from tilefoundry.visitor_registry import register_typeinfer

SEQ_LEN = DimVar("seq_len", 1, 100)
N_SCALED = DimVar("N", 1, 64)
SEQ_DYN = DimVar("seq_len", 1, 4)

_EPS = 1e-6
_NK, _KD, _VD, _NV = 16, 128, 64, 32


@dataclass(frozen=True)
class _Cfg:
    """A model config read at parse time, so its fields arrive as numbers."""

    head_dim: int = 128
    rms_eps: float = 1e-6


_CFG = _Cfg()
_HALF, _STEP, _ROWS = 4, 2, 3

CTA_MESH = Mesh((Topology("cta", 8),), Layout((8,), (1,)))
M_GPU = Mesh(
    (Topology("gpu", 8192),),
    Layout((32, 2, 8, 32), (2048, 1024, 32, 1)),
    names=("cluster", "cta", "warp", "lane"),
)
M_MULTI = Mesh((Topology("thread", 6 * 32),), Layout((6, 32), (32, 1)), names=("w", "t"))
M_STRIDED = Mesh((Topology("thread", 4 * 32),), Layout((4, 32), (32, 1)), names=("y", "t"))
M_CTA = Mesh((Topology("cta", 128),), Layout((128,), (1,)), names=("cta",))


@module(entry="dim_anchored_twice")
class HirExpressions:
    """Expression, annotation, and subscript surface, in one Module.

    Dim arithmetic in a signature, the string dtype surface, a value literal's
    unmaterialized storage, a declared constant, tuple literals in and out, and
    every subscript form: an index that drops its axis, a slice that keeps it,
    a negative index, a clamped stride, and a tile window that a compile-time
    offset moves.
    """

    @func
    def dim_anchored_twice(
        x: Tensor[(CTX_LEN,), "bf16"],
        y: Tensor[(CTX_LEN + 1,), "bf16"],
    ) -> Tensor[(CTX_LEN,), "bf16"]:
        return x

    @func
    def dim_from_a_static_call(x: Tensor[(CTX_LEN,), "bf16"]):
        return tf.zeros(shape=(ceildiv(CTX_LEN, 128) * 128,), dtype="bf16")

    @func
    def cast_by_dtype_string(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "bf16"]:
        return cast(x, dtype="bf16")  # noqa: F405

    @func
    def reduce_by_kind_string(x: Tensor[(8,), "f32"]) -> Tensor[(1,), "f32"]:
        return reduce(x, axes=(0,), keepdim=True, kind="sum")  # noqa: F405

    @func
    def unmaterialized_surface_storage(
        x: Tensor[(8,), "f32", None, "umat"],
    ) -> Tensor[(8,), "f32"]:
        return x

    @func
    def storage_without_a_layout_slot(
        x: Tensor[(8,), "f32", "umat"],
    ) -> Tensor[(8,), "f32"]:
        return x

    @func
    def literal_meets_bf16(x: Tensor[(1, 8), "bf16"]) -> Tensor[(1, 8), "bf16"]:
        return add(x, 1e-6)  # noqa: F405

    @func
    def captured_float_meets_bf16(x: Tensor[(1, 8), "bf16"]) -> Tensor[(1, 8), "bf16"]:
        return add(x, _EPS)  # noqa: F405

    @func
    def compile_time_operands(x: Tensor[(1, 2048), "bf16"]) -> Tensor[(1, 16, 128), "bf16"]:
        key_dim = _NK * _KD
        scaled = mul(x, _CFG.head_dim**-0.5)  # noqa: F405
        shifted = add(scaled, _CFG.rms_eps)  # noqa: F405
        return reshape(shifted, new_shape=(1, key_dim // _KD, _KD))  # noqa: F405

    @func
    def unpacked_compile_time_values(
        x: Tensor[(1, 32, 128), "f32"],
    ) -> Tensor[(1, 64, 64), "f32"]:
        nv, kd, vd = _NV, _KD, _VD
        return reshape(x, new_shape=(1, nv * kd // vd, vd))  # noqa: F405

    @func
    def offsets_as_a_tuple_literal(
        dst: Tensor[(2, 8, 4), "f32"],
        upd: Tensor[(1, 3, 4), "f32"],
        p: Tensor[(), "i32"],
    ) -> Tensor[(2, 8, 4), "f32"]:
        return insert_slice(dst, upd, (1, p, 0))  # noqa: F405

    @func
    def unpacked_multi_output(x: Tensor[(1, 1536), "bf16"]) -> Tensor[(1, 1536), "fp8e4m3"]:
        x_fp8, x_scale = quant(x)  # noqa: F405, F841
        return x_fp8

    @func
    def index_drops_its_axis(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4), "f32"]:
        return x[:, :, 3]

    @func
    def slice_keeps_its_axis(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4, 1), "f32"]:
        return x[:, :, 3:4]

    @func
    def index_counted_from_the_end(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4), "f32"]:
        return x[:, :, -1]

    @func
    def slice_strided_and_clamped(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4, 3), "f32"]:
        return x[:, :, 1:20:3]

    @func
    def slice_to_symbolic_extents(
        x: Tensor[(CTX_LEN, _KD), "f32"],
    ) -> Tensor[(CTX_LEN, _KD), "f32"]:
        return x[0:CTX_LEN, 0:_KD]

    @func
    def full_tile_window(x: Tensor[(8, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
        out = add(seed, seed)  # noqa: F405
        for row in tile(8, 4):  # noqa: F405
            out = add(x[row, :], seed)  # noqa: F405
        return out

    @func
    def two_windows_a_fixed_distance_apart(
        gu: Tensor[(_ROWS, 2 * _HALF), "f32"],
        seed: Tensor[(_ROWS, _STEP), "f32"],
    ):
        out = add(seed, seed)  # noqa: F405
        for n in tile(_HALF, _STEP):  # noqa: F405
            out = add(out, mul(gu[:, n], gu[:, n + _HALF]))  # noqa: F405
        return out

    @func
    def a_summed_offset_names_the_same_move(
        gu: Tensor[(_ROWS, 2 * _HALF), "f32"],
        seed: Tensor[(_ROWS, _STEP), "f32"],
    ):
        out = add(seed, seed)  # noqa: F405
        for n in tile(_HALF, _STEP):  # noqa: F405
            out = add(out, mul(gu[:, n], gu[:, _HALF + 1 + n - 1]))  # noqa: F405
        return out


@func
def returns_a_pair(a: Tensor[(4,), "f32"], b: Tensor[(4,), "f32"]):
    return (add(a, b), mul(a, b))  # noqa: F405


@func
def doubles_a_constant(w: ConstTensor[(8, 64), "f32"]) -> Tensor[(8, 64), "f32"]:
    return add(w, w)  # noqa: F405


@module(entry="single_carry")
class HirGrid:
    """Grid-region loops and the ``where`` annotations that ride on them.

    Both loop spellings over one domain, every way an extent is written, a
    rebinding lifted to a phi carry, a carry initialized straight from a
    parameter, nested loops, and layout / mesh / storage intent stated on a
    binding, on a parameter, and on a bound tuple element.
    """

    @func
    def range_default_step(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        for i in range(8):
            y = relu(x)  # noqa: F405, F841

    @func
    def tile_extent_step(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        for i in tile(8, 2):  # noqa: F405
            y = relu(x)  # noqa: F405, F841

    @func
    def tile_dimvar_extent(x: Tensor[(SEQ_LEN, 4), "f32"]) -> Tensor[(SEQ_LEN, 4), "f32"]:
        for i in tile(SEQ_LEN, 2):  # noqa: F405
            y = relu(x)  # noqa: F405, F841

    @func
    def range_dim_expr_extent(x: Tensor[(SEQ_LEN, 4), "f32"]) -> Tensor[(SEQ_LEN, 4), "f32"]:
        for i in range(SEQ_LEN // 2):
            y = relu(x)  # noqa: F405, F841

    @func
    def range_start_stop_step(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        for i in range(2, 8, 3):
            y = relu(x)  # noqa: F405, F841

    @func
    def single_carry(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        o = relu(x)  # noqa: F405
        for i in range(8):
            o = add(o, x)  # noqa: F405
        return o

    @func
    def inner_bindings_carry_nothing(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        for i in range(8):
            t = relu(x)  # noqa: F405
            z = add(t, x)  # noqa: F405, F841

    @func
    def carry_reads_old_and_new(x: Tensor[(8,), "f32"]):
        m = relu(x)  # noqa: F405
        o = relu(x)  # noqa: F405
        for i in range(8):
            m_new = maximum(m, x)  # noqa: F405
            correction = sub(m, m_new)  # noqa: F405
            o = add(o, correction)  # noqa: F405
            m = m_new
        return o

    @func
    def carry_initialized_from_a_parameter(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        acc = x
        for i in range(8):
            acc = add(acc, x)  # noqa: F405
        return acc

    @func
    def nested_for(x: Tensor[(8, 4), "f32"]) -> Tensor[(8, 4), "f32"]:
        o = relu(x)  # noqa: F405
        for r in range(8):
            for c in range(4):
                o = add(o, x)  # noqa: F405
        return o

    @func
    def where_on_a_binding(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 16), "bf16"]:
        y: where(  # noqa: F405, F821
            layout=(_, 16 @ cta), mesh=CTA_MESH, storage="gmem"
        ) = tf.add(x, x)
        return y

    @func
    def where_with_a_partial_value_state(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 16), "bf16"]:
        y: where(layout=((_, 16), {cta @ P("sum")})) = tf.add(x, x)  # noqa: F405, F821
        return y

    @func
    def where_on_a_parameter(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 16), "bf16"]:
        x: where(storage="smem")  # noqa: F405, F821
        return x

    @func
    def where_on_a_bound_tuple_element(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 4), "i64"]:
        values = tf.topk(x, k=4, axis=-1)
        ids = values[1]
        ids: where(storage="gmem")  # noqa: F405, F821
        return ids


@module(
    entry="split_inline_and_default_broadcast",
    topologies=(Topology("cta", 8),),
)
class HirSharded:
    """Mesh placement, shard-layout sugar, and topology declaration in one Module.

    Every annotation sugar form the parser canonicalises — an inline ``Split``,
    the ``{...}`` value-state set, a multi-mesh-axis split with a remainder,
    explicit strides, and the single-axis ``int @ mesh`` shorthand — plus a mesh
    axis read as a position coordinate and a reshard whose split extent is
    resolved through the closure.
    """

    @func
    def split_inline_and_default_broadcast(
        a: Tensor[(32, 128), bf16, (32 @ M_GPU.cluster, 2 @ M_GPU.cta, 64), "smem"],  # noqa: F405
    ) -> Tensor[(32, 128), "f32"]:
        return a

    @func
    def partial_brace_value_state(
        a: Tensor[(64, 128), "bf16", ((32 @ M_GPU.cluster, 64), {M_GPU.warp @ P("sum")}), "smem"],
    ) -> Tensor[(64, 128), "f32"]:
        return a

    @func
    def multi_axis_split_with_remainder(
        a: Tensor[(1, 1536), "f32", (1, 1536 @ (M_MULTI.w, M_MULTI.t)), "smem"],
    ) -> Tensor[(1, 1536), "f32"]:
        return a

    @func
    def explicit_strides(
        a: Tensor[(12, 4), "f32", ((12 @ M_STRIDED.y, 4), (4, 1)), "smem"],
    ) -> Tensor[(12, 4), "f32"]:
        return a

    @func
    def int_at_a_single_axis_mesh(
        a: Tensor[(1, 8192), "f32", (1, 8192 @ M_CTA), "smem"],
    ) -> Tensor[(1, 8192), "f32"]:
        return a

    @func
    def mesh_axis_as_a_position_coordinate() -> Tensor[(), "i64"]:
        with Mesh(("cta",), layout=(8,), names=("w",)) as cta:
            return cta.w

    @func
    def reshard_with_a_dynamic_and_a_closure_axis(
        q: Tensor[(1, SEQ_DYN, 32, 128), "bf16"],
    ) -> Tensor[(1, SEQ_DYN, 32, 128), "bf16"]:
        with Mesh(("cta",), layout=Layout((8,), (1,))) as cta:
            return reshard(q, layout=(1, SEQ_DYN, 32 @ cta, 128))  # noqa: F405


@register_op(dialect="tf", category="custom", name="custom_parse_addsq")
class CustomParseAddSq(Op):
    """Test-only custom op that squares the sum of its inputs."""

    lhs = ParamDef(kind="input", pattern=TensorPattern)
    rhs = ParamDef(kind="input", pattern=TensorPattern)


@register_typeinfer(CustomParseAddSq)
def _(call, ctx):
    return ctx.type_of(call.args[0])


custom_parse_addsq = CustomParseAddSq


@module(entry="scale")
class Scaled:
    """A child Module whose entry is shaped by a ``DimVar``."""

    @func
    def scale(x: Tensor[(N_SCALED,), "f32"]) -> Tensor[(N_SCALED,), "f32"]:
        return tf.mul(x, x)


@module(entry="run")
class Grand:
    """The grandchild of ``Deep``: one activation and one declared constant."""

    @func
    def run(x: Tensor[(8,), "f32"], w: ConstTensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return tf.mul(x, w)


@module(entry="mid")
class Deep:
    """A child Module that itself calls a child."""

    grand = Grand

    @func
    def mid(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return grand(x)  # noqa: F821 — the class-body child binding


@module(
    entry="calls_a_child",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 4),),
)
class HirModule:
    """A ``@module`` class body and every call it can make.

    A child Module bound by name, two bindings of one child rebuilt for a
    resharded argument, a grandchild reached through the middle Module, a
    child taking an activation while its constant stays declared, a weight
    converter that calls a child of its own, a specialization variant, and a
    registered custom op as a call target.
    """

    leaf = Callee
    first = Callee
    second = Callee
    mlp = Expert
    deep = Deep
    variant_leaf = Scaled

    @func
    def calls_a_child(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return leaf(x)  # noqa: F821

    @func
    def two_bindings_under_a_reshard(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
            local = tf.reshard(x, (8 @ cta.tile,), "rmem")
            return tf.add(first(local), second(local))  # noqa: F821

    @func
    def through_the_grandchild(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
            local = tf.reshard(x, (8 @ cta.tile,), "gmem")
            return deep(local)  # noqa: F821

    @func
    def carries_activations_only(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return mlp(x)  # noqa: F821

    @func
    def converts_its_weight(
        x: Tensor[(8,), "f32"], w: ConstTensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        return tf.add(x, w)

    @converts_its_weight.converter("w")  # noqa: F821
    def _convert_w(w: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
        return leaf(w)  # noqa: F821

    @func
    def dispatches_to_a_variant(x: Tensor[(N_SCALED,), "f32"]) -> Tensor[(N_SCALED,), "f32"]:
        pass

    @dispatches_to_a_variant.specialize(DimVarRangePat("N", 1, 64))  # noqa: F821
    def scaled_variant(x: Tensor[(N_SCALED,), "f32"]) -> Tensor[(N_SCALED,), "f32"]:
        return variant_leaf(x)  # noqa: F821

    @func
    def uses_a_custom_op(
        a: Tensor[(8,), "f32"], b: Tensor[(8,), "f32"]
    ) -> Tensor[(8,), "f32"]:
        return custom_parse_addsq(a, b)


@dataclass(frozen=True)
class ParserProgram:
    """One feature-dense program and the features it is here to carry."""

    name: str
    parsed: Module | Function
    features: tuple[str, ...]


PROGRAMS: tuple[ParserProgram, ...] = (
    ParserProgram(
        "hir_expressions",
        HirExpressions,
        (
            "dim arithmetic",
            "dim from a static call",
            "str dtype surface",
            "str reduce-kind surface",
            "storage with and without an empty layout slot",
            "value literal dtype",
            "compile-time operands",
            "compile-time tuple unpack",
            "tuple-literal op input",
            "multi-output tuple unpack",
            "subscript indexing and slicing",
            "slice endpoints at symbolic extents",
            "tile window",
            "window move by a compile-time offset",
        ),
    ),
    ParserProgram(
        "hir_grid",
        HirGrid,
        (
            "range and tile share one loop domain",
            "static, DimVar, and dim-expression extents",
            "carry lifting",
            "carry initialized from a parameter",
            "nested for",
            "where layout / mesh / storage constraint",
            "where on a parameter and on a bound tuple element",
        ),
    ),
    ParserProgram(
        "hir_sharded",
        HirSharded,
        (
            "inline Split and default Broadcast",
            "Partial value-state set",
            "multi-mesh-axis split with a remainder",
            "explicit strides",
            "int @ single-axis mesh",
            "mesh placement",
            "mesh axis as a position coordinate",
            "closure-resolved reshard split axis",
        ),
    ),
    ParserProgram(
        "hir_module",
        HirModule,
        (
            "@module class body",
            "child Module bound by name",
            "two bindings of one child rebuilt at a call site",
            "grandchild call through the middle Module",
            "activations carried, constants left declared",
            "weight converter",
            "specialization variant",
            "custom op call target",
        ),
    ),
)


_TILE = 12
_NT = DimVar("Ntile", 1, 64)


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tir_dynamic_device(a: Tensor[(_NT, _TILE), "f32"]):
    with TirMesh((Topology("cta", _NT),), Layout(shape=(_NT,), strides=(1,))) as cta:
        view = T.tensor_view(
            a,
            layout=ShardLayout(
                layout=Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                attrs=(Split(0),),
                mesh=cta,
            ),
        )
        reg = T.alloc_tensor(
            TensorType(
                shape=(_NT, _TILE),
                dtype=DType.f32,
                layout=ShardLayout(
                    layout=Layout(shape=(_NT, _TILE), strides=(_TILE, 1)),
                    attrs=(Split(0),),
                    mesh=cta,
                ),
                storage=StorageKind.RMEM,
            )
        )
        T.copy(view, reg)


@prim_func(target=CpuTarget())
def tir_host_entry(a: Tensor[(_NT, _TILE), "f32"]):
    launch(tir_dynamic_device, a, grid=(_NT, 1, 1), block=(1, 1, 1))  # noqa: F821


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tir_static_device(a: Tensor[(16, 8), "f32"]):
    with TirMesh((Topology("thread", 8),), Layout(shape=(8,), strides=(1,))) as t:
        view = T.tensor_view(
            a,
            layout=ShardLayout(
                layout=Layout(shape=(16, 8), strides=(8, 1)), attrs=(Split(0),), mesh=t
            ),
        )
        reg = T.alloc_tensor(
            TensorType(
                shape=(16, 8),
                dtype=DType.f32,
                layout=ShardLayout(
                    layout=Layout(shape=(16, 8), strides=(8, 1)), attrs=(Split(0),), mesh=t
                ),
                storage=StorageKind.RMEM,
            )
        )
        T.copy(view, reg)


@prim_func(target=CpuTarget())
def tir_effect_form_selector(a: Tensor[(128,), "f32"], b: Tensor[(128,), "f32"]):
    copy_(a, b)  # noqa: F821 — resolved via dispatch.resolve_callable, not the closure


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tir_param_layout_sugar(a: Tensor[(1, 8192), "f32", (1, 8192 @ M_CTA), "smem"]):
    return


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tir_sync_scopes(a: Tensor[(128,), "f32"]):  # noqa: ARG001
    with TirMesh(
        (Topology("thread", 128),), Layout(shape=(4, 32), strides=(32, 1)), ("w", "t")
    ) as m:
        T.sync(m)
        T.sync(m[0, :])
        T.sync(m[1:3, :])


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tir_static_atom_bindings(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
    op = T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN
    atom = T.cuda.mma.atom(op=op)  # noqa: F841


@prim_func(target=CudaTarget("nvidia.h200_sxm"))
def tir_atom_fragment_in_a_warp_scope(a: Tensor[(16, 16), "bf16"]):  # noqa: ARG001
    atom = T.cuda.mma.atom(op=T.cuda.mma.SM80_16x8x16_F32BF16BF16F32_TN)
    with TirMesh(
        (Topology("thread", 32),), Layout(shape=(4, 8), strides=(1, 4)), names=("warp", "lane")
    ) as warp:  # noqa: F841
        frag = T.alloc_tensor(  # noqa: F841
            TensorType(shape=(16, 16), dtype=DType.bf16, layout=atom.A, storage=StorageKind.RMEM)
        )
