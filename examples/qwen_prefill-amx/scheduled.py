"""Qwen HIR with explicit AMX projection layouts and worker schedules."""

from dataclasses import replace
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, ConstTensor, Mesh, T, Topology, tf
from tilefoundry.ir.types import Layout, DType
from tilefoundry.ir.tir.amx.instructions import Z_LAYOUT
from model import build
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.visitor import collect_exprs, ExprCloner, BindingSubstitutionCloner
from tilefoundry.ir.hir.schedule import ScheduleOp
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.types import Mesh as MeshValue
from tilefoundry.ir.tir.cpu.tensor import instruction_for, INSTRUCTIONS

BM = 32
WORKERS = 8


def scheduled_projection(M, N, K):
    if K <= 0 or N <= 0 or K % 32 or N % 32:
        raise ValueError(
            "The native AMX runtime requires positive K and N that are multiples of 32"
        )
    B = 64
    if M != 1 and (M <= 0 or M % 32):
        raise ValueError(
            "This schedule requires a multiple of 32 prompt tokens, or M=1 for the last-token head"
        )
    if M < 128 or N % 64:

        @module(entry="forward", topologies=(Topology("core", 8), Topology("amx", 1)))
        class Head:
            @func
            def forward(x: Tensor[(M, K), "f32"], w: ConstTensor[(K, N), "f16"]):
                return tf.matmul(tf.cast(x, dtype="f16"), w, out_dtype="f32")

            @forward.converter("w")
            def _(raw: ConstTensor[(N, K), "f16"]):
                return tf.transpose(raw, perm=(1, 0))

        return cpu_schedule(Head)

    @module(entry="forward", topologies=(Topology("core", 8), Topology("amx", 1)))
    class Linear:
        @func
        def forward(x: Tensor[(M, K), "f32"], w: ConstTensor[(N // B, K, B), "f16"]):
            with Mesh(("core",), layout=(1,), names=("packer",)) as _packer:
                # One CPU pack, outside all K issues. The physical 32-row
                # panel makes each AMX Y load a contiguous 64-byte vector.
                half = tf.schedule(
                    (x, tf.empty(Tensor[(M, K), "f16", None, "gmem"])),
                    op=T.cpu_cast(dtype=DType.f16),
                )
                view = tf.transpose(
                    tf.reshape(half, new_shape=(M // BM, BM, K)), perm=(0, 2, 1)
                )
                packed = tf.schedule(
                    (
                        view,
                        tf.empty(
                            Tensor[
                                (M // BM, K, BM),
                                "f16",
                                Layout((M // BM, K, BM), (K * BM, BM, 1)),
                                "gmem",
                            ]
                        ),
                    ),
                    op=T.copy(),
                )
                out = tf.zeros(Tensor[(M, N), "f32"])
            with Mesh(("core",), layout=(WORKERS,), names=("worker",)) as cores:
                for panel in range(cores.worker, N // B, WORKERS):
                    for half_n in range(B // 32):
                        for mb in range(M // BM):
                            with Mesh(("amx",), layout=(1,), names=("unit",)) as _amx:
                                acc = tf.zeros(
                                    Tensor[(BM, 32), "f32", Z_LAYOUT, "rmem"]
                                )
                                for k in range(K):
                                    a = tf.schedule(
                                        (
                                            tf.reshape(
                                                packed[mb : mb + 1, k : k + 1, :],
                                                new_shape=(BM, 1),
                                            ),
                                            tf.empty(
                                                Tensor[(BM, 1), "f16", None, "rmem"]
                                            ),
                                        ),
                                        op=T.amx_load_y(),
                                    )
                                    b = tf.schedule(
                                        (
                                            tf.reshape(
                                                w[
                                                    panel : panel + 1,
                                                    k : k + 1,
                                                    half_n * 32 : half_n * 32 + 32,
                                                ],
                                                new_shape=(1, 32),
                                            ),
                                            tf.empty(
                                                Tensor[(1, 32), "f16", None, "rmem"]
                                            ),
                                        ),
                                        op=T.amx_load_x(),
                                    )
                                    acc = tf.schedule(
                                        (acc, a, b),
                                        op=T.amx_fma16(),
                                        repeat=(1, 1, 1),
                                        order=(0, 1, 2),
                                    )
                                tile = tf.schedule(
                                    (
                                        acc,
                                        tf.empty(Tensor[(BM, 32), "f32", None, "gmem"]),
                                    ),
                                    op=T.amx_store(),
                                )
                                out = tf.insert_slice(
                                    out, tile, (mb * BM, panel * B + half_n * 32)
                                )
                return out

        @forward.converter("w")
        def _(raw: ConstTensor[(N, K), "f16"]):
            return tf.transpose(
                tf.reshape(raw, new_shape=(N // B, B, K)), perm=(0, 2, 1)
            )

    return Linear


class CPUChoices(ExprCloner):
    """Record explicit software-kernel choices without wrapping whole functions."""

    def visit_Call(self, call, ctx=None):
        args = tuple(self.visit(arg, ctx) for arg in call.args)
        if (type(call.target), len(args)) in INSTRUCTIONS and not (
            call.type.shape == () and call.type.dtype in (DType.i64, DType.i32)
        ):
            op = instruction_for(call.target, len(args), call.type.dtype)
            from tilefoundry.ir.hir.tensor.empty import Empty
            from tilefoundry.ir.types import StorageKind
            from tilefoundry.ir.types.stride import compact_row_major

            ty = replace(
                call.type,
                storage=StorageKind.GMEM,
                layout=Layout(
                    call.type.shape, tuple(compact_row_major(call.type.shape))
                ),
            )
            dst = Call(Empty(type=ty), (), type=ty)
            return Call(ScheduleOp(op=op), (*args, dst), type=ty)
        return replace(call, args=args)


def cpu_schedule(module, memo=None):
    from tilefoundry.dsl import ModuleList
    from tilefoundry.ir.core import Op

    memo = {} if memo is None else memo
    if id(module) in memo:
        return memo[id(module)]
    functions = []
    for fn in module.functions:
        if fn.body is None:
            raise ValueError("No dispatch prototypes")
        if isinstance(fn.body, MeshRegion) or any(
            isinstance(v, Call) and not isinstance(v.target, Op)
            for v in collect_exprs(fn.body)
        ):
            functions.append(fn)
            continue
        body = CPUChoices().visit(fn.body)
        mesh = MeshValue((Topology("core", 8),), Layout((1,), (1,)), ("worker",))
        params = tuple(Var(p.name, type=p.type) for p in fn.params)
        body = BindingSubstitutionCloner().visit(
            body, {id(p): v for p, v in zip(fn.params, params)}
        )
        body = MeshRegion(
            mesh, body, params=params, args=fn.params, type=fn.return_type
        )
        functions.append(replace(fn, body=body))
    mapping = {id(old): new for old, new in zip(module.functions, functions)}

    class Retarget(ExprCloner):
        def visit_Call(self, call, ctx=None):
            result = replace(
                call,
                args=tuple(self.visit(arg, ctx) for arg in call.args),
                target=self.visit(call.target, ctx)
                if isinstance(call.target, Call)
                else call.target,
            )
            return replace(result, target=mapping.get(id(result.target), result.target))

    for fn in functions:
        fn.body = Retarget().visit(fn.body)
    children = {
        key: ModuleList(tuple(cpu_schedule(child, memo) for child in slot.elements))
        if slot.indexed
        else ModuleList(cpu_schedule(slot.elements[0], memo))
        for key, slot in module.children.items()
    }
    result = replace(module, functions=tuple(functions), children=children)
    memo[id(module)] = result
    return result


def make_scheduled(config, sequence=128):
    return replace(
        cpu_schedule(build(config, sequence, linear=scheduled_projection)),
        topologies=(Topology("core", 8), Topology("amx", 1)),
    )


from config import config

Qwen3_1_7B = make_scheduled(config("1.7b"), 128)
Qwen3_5_4B = make_scheduled(config("4b"), 128)
