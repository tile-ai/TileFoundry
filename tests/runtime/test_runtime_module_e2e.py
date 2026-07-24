"""GPU end-to-end for the nested ``RuntimeModule`` ABI: handwritten CUDA
kernels assembled by hand (no build/compile/jit), with weights and a
persistent state resolved by ``ParamABI`` name from a ``SafetensorsResource``.

``TinyMlp`` (root) holds ``mlp_head`` + weight ``w_head``, and one child
module ``proj`` (holds ``up_down`` + weights ``w_up``/``w_down`` + state
``kv``). The candidate composition ``root_rm.mlp_head(root_rm.proj.up_down(x))``
takes only the activation ``x`` positionally — weights auto-inject by name —
and is checked against the same composition run through the plain HIR
evaluator with the identical weight tensors passed positionally.
"""
from __future__ import annotations

import json

import pytest
import torch
from torch.utils.cpp_extension import load_inline

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.runtime import (
    RuntimeFunction,
    RuntimeModule,
    SafetensorsResource,
    bench,
    callable_type_of,
    check,
)
from tilefoundry.target.cuda import H200SXM

_B, _D, _H, _V = 4, 64, 128, 32


@func
def mlp_head(
    hidden: Tensor[(_B, _D), "f32"], w_head: Tensor[(_D, _V), "f32"]
) -> Tensor[(_B, _V), "f32"]:
    return tf.matmul(hidden, w_head)


@func
def up_down(
    x: Tensor[(_B, _D), "f32"],
    w_up: Tensor[(_D, _H), "f32"],
    w_down: Tensor[(_H, _D), "f32"],
) -> Tensor[(_B, _D), "f32"]:
    return tf.matmul(tf.relu(tf.matmul(x, w_up)), w_down)


def _tt(shape: tuple[int, ...]) -> TensorType:
    return TensorType(shape=shape, dtype=DType.f32, layout=None, storage="gmem")


# ── Handwritten CUDA kernels (naive f32 matmul / matmul+relu) ────────────────

_CUDA_DECLS = r"""
torch::Tensor matmul(torch::Tensor x, torch::Tensor w);
torch::Tensor matmul_relu(torch::Tensor x, torch::Tensor w);
"""

_CUDA_SRC = r"""
__global__ void matmul_kernel(const float* x, const float* w, float* out,
                               int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += x[row * K + k] * w[k * N + col];
        out[row * N + col] = acc;
    }
}

__global__ void matmul_relu_kernel(const float* x, const float* w, float* out,
                                    int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += x[row * K + k] * w[k * N + col];
        out[row * N + col] = acc > 0.0f ? acc : 0.0f;
    }
}

torch::Tensor matmul(torch::Tensor x, torch::Tensor w) {
    int M = x.size(0), K = x.size(1), N = w.size(1);
    auto out = torch::empty({M, N}, x.options());
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), M, K, N);
    return out;
}

torch::Tensor matmul_relu(torch::Tensor x, torch::Tensor w) {
    int M = x.size(0), K = x.size(1), N = w.size(1);
    auto out = torch::empty({M, N}, x.options());
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), M, K, N);
    return out;
}
"""

_ext = load_inline(
    name="tf_runtime_module_e2e_kernels",
    cpp_sources=_CUDA_DECLS,
    cuda_sources=_CUDA_SRC,
    functions=["matmul", "matmul_relu"],
    verbose=False,
)


def _up_down_impl(x, w_up, w_down):
    return _ext.matmul(_ext.matmul_relu(x, w_up), w_down)


def _mlp_head_impl(hidden, w_head):
    return _ext.matmul(hidden, w_head)


# ── Fixture-ish assembly (fresh per test — cheap: no recompilation) ──────────

def _build(tmp_path):
    torch.manual_seed(0)
    w_head = torch.randn(_D, _V, dtype=torch.float32)
    w_up = torch.randn(_D, _H, dtype=torch.float32)
    w_down = torch.randn(_H, _D, dtype=torch.float32)
    kv_init = torch.randn(_B, _D, dtype=torch.float32)

    from safetensors.torch import save_file  # noqa: PLC0415 -- test-only dep

    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"w_head": w_head}, str(shard_a))
    save_file({"proj.w_up": w_up, "proj.w_down": w_down, "proj.kv": kv_init}, str(shard_b))
    index = {
        "weight_map": {
            "w_head": shard_a.name,
            "proj.w_up": shard_b.name,
            "proj.w_down": shard_b.name,
            "proj.kv": shard_b.name,
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    child_ir = Module(
        name="proj",
        functions=(up_down,),
        entry="up_down",
        weights={"w_up": _tt((_D, _H)), "w_down": _tt((_H, _D))},
        states={"kv": _tt((_B, _D))},
    )
    root_ir = Module(
        name="TinyMlp",
        functions=(mlp_head,),
        entry="mlp_head",
        modules=(child_ir,),
        weights={"w_head": _tt((_D, _V))},
    )

    resource = SafetensorsResource(tmp_path, device="cuda")
    child_rm = RuntimeModule(
        name=child_ir.name,
        entry=child_ir.entry,
        functions={"up_down": RuntimeFunction(callable_type_of(up_down), _up_down_impl)},
        resource=resource.subtree("proj"),
        weight_names=tuple(child_ir.weights),
        state_specs=child_ir.states,
        device=H200SXM(),
    )
    root_rm = RuntimeModule(
        name=root_ir.name,
        entry=root_ir.entry,
        functions={"mlp_head": RuntimeFunction(callable_type_of(mlp_head), _mlp_head_impl)},
        resource=resource,
        weight_names=tuple(root_ir.weights),
        modules=(child_rm,),
        device=H200SXM(),
    )
    return root_rm, w_head, w_up, w_down, kv_init


def test_candidate_matches_evaluator_reference(tmp_path) -> None:
    root_rm, w_head, w_up, w_down, _kv = _build(tmp_path)
    x = torch.randn(_B, _D, dtype=torch.float32, device="cuda")

    def candidate(x):
        return root_rm.mlp_head(root_rm.proj.up_down(x))

    def reference(x):
        mid = evaluate(up_down, x, w_up, w_down, device="cuda")
        return evaluate(mlp_head, mid, w_head, device="cuda")

    report = check(candidate, reference, (x,))
    assert report.passed is True


def test_child_submodule_matches_evaluator_reference(tmp_path) -> None:
    root_rm, _w_head, w_up, w_down, _kv = _build(tmp_path)
    x = torch.randn(_B, _D, dtype=torch.float32, device="cuda")

    def reference(x):
        return evaluate(up_down, x, w_up, w_down, device="cuda")

    report = check(root_rm.proj.up_down, reference, (x,))
    assert report.passed is True


def test_state_from_resource_bench_and_missing_attr(tmp_path) -> None:
    root_rm, _w_head, _w_up, _w_down, kv_init = _build(tmp_path)

    kv = root_rm.proj.states["kv"]
    assert kv.device.type == "cuda"
    assert kv.shape == (_B, _D)
    assert torch.allclose(kv.cpu(), kv_init)

    x = torch.randn(_B, _D, dtype=torch.float32, device="cuda")

    def candidate(x):
        return root_rm.mlp_head(root_rm.proj.up_down(x))

    report = bench(candidate, (x,), iters=10, device=H200SXM())
    assert report.metrics["mean_ms"] > 0

    with pytest.raises(AttributeError):
        root_rm.no_such_fn
