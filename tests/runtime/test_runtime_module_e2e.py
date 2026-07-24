"""The authoring pattern end to end: a semantic module and its runtime twin,
written separately but structurally identical, agreeing numerically.

1. Write the **semantic module** — compute @funcs plus a sibling ``convert``
   @func (the weight converter, peer to the compute function) + weights/states
   declarations.
2. ``semantic.prepare(raw, dir)`` runs every ``convert`` once and writes the
   canonical weights to a prepared directory.
3. Write the **runtime module** — same names / same child tree; its function
   bodies are handwritten CUDA ``RuntimeFunction`` subclasses. It does not
   prepare; it ``load``s straight from the prepared directory.
4. Both sides load the same directory and run; the runtime ``forward`` and the
   semantic module's ``forward`` (via the evaluator) agree (``check``,
   cosine >= 0.999).

Needs nvcc + a GPU (like the rest of tests/e2e).
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.cpp_extension import load_inline

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.types.utils import make_tensor_type
from tilefoundry.runtime import (
    DictResource,
    RuntimeFunction,
    RuntimeModule,
    SafetensorsResource,
    bench,
    callable_type_of,
    check,
)
from tilefoundry.target.cuda import H200SXM

_B, _D, _H, _V = 4, 64, 128, 32


# ── 1. semantic module: compute @funcs + sibling convert @func ───────────────

@func
def up_down(
    x: Tensor[(_B, _D), "f32"],
    w_up: Tensor[(_D, _H), "f32"],
    w_down: Tensor[(_H, _D), "f32"],
) -> Tensor[(_B, _D), "f32"]:
    return tf.matmul(tf.relu(tf.matmul(x, w_up)), w_down)


@func
def convert(
    w_up: Tensor[(_H, _D), "f32"],
    w_down: Tensor[(_H, _D), "f32"],
):
    # convert's params are the RAW checkpoint names; raw ``w_up`` is stored
    # (H, D) and the canonical form ``up_down`` consumes is its transpose
    # (D, H) — a layout convert, non-identity, so a passing ``check`` plus the
    # transpose assertion proves the converter ran. ``w_down`` passes through.
    return tf.transpose(w_up, perm=(1, 0)), w_down


@func
def mlp_head(
    hidden: Tensor[(_B, _D), "f32"],
    w_head: Tensor[(_D, _V), "f32"],
) -> Tensor[(_B, _V), "f32"]:
    return tf.matmul(hidden, w_head)


def _tt(shape: tuple[int, ...]):
    return make_tensor_type(shape)


proj_ir = Module(
    name="proj",
    functions=(up_down, convert),
    entry="up_down",
    weights={"w_up": _tt((_D, _H)), "w_down": _tt((_H, _D))},
    states={"kv": _tt((_B, _D))},
)
tinymlp_ir = Module(
    name="TinyMlp",
    functions=(mlp_head,),
    entry="mlp_head",
    modules=(proj_ir,),
    weights={"w_head": _tt((_D, _V))},
)


# ── 2. handwritten CUDA kernels + runtime module twin ────────────────────────

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


class UpDownFn(RuntimeFunction):
    """CUDA body for ir ``up_down`` — canonical weights taken at load time."""

    def __init__(self, w_up: torch.Tensor, w_down: torch.Tensor) -> None:
        super().__init__(callable_type_of(up_down))
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _ext.matmul(_ext.matmul_relu(x, self.w_up), self.w_down)


class MlpHeadFn(RuntimeFunction):
    """CUDA body for ir ``mlp_head``."""

    def __init__(self, w_head: torch.Tensor) -> None:
        super().__init__(callable_type_of(mlp_head))
        self.w_head = w_head

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return _ext.matmul(hidden, self.w_head)


class ProjRT(RuntimeModule):
    def __init__(self) -> None:
        super().__init__("proj", entry="up_down")

    def load(self, resource) -> None:
        self.up_down = UpDownFn(resource.load("w_up"), resource.load("w_down"))
        self.kv = resource.load("kv")
        super().load(resource)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_down(x)


class TinyMlpRT(RuntimeModule):
    def __init__(self) -> None:
        self.proj = ProjRT()
        super().__init__("TinyMlp", entry="mlp_head", modules=(self.proj,))

    def load(self, resource) -> None:
        self.mlp_head = MlpHeadFn(resource.load("w_head"))
        super().load(resource)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_head(self.proj(x))


# ── 3. raw weights → prepare → prepared directory ────────────────────────────

def _raw_weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "w_head": torch.randn(_D, _V, dtype=torch.float32),
        "proj.w_up": torch.randn(_H, _D, dtype=torch.float32),  # RAW (H, D)
        "proj.w_down": torch.randn(_H, _D, dtype=torch.float32),
        "proj.kv": torch.randn(_B, _D, dtype=torch.float32),
    }


def _prepared(tmp_path) -> dict[str, torch.Tensor]:
    raw = _raw_weights()
    tinymlp_ir.prepare(DictResource(raw), str(tmp_path))
    return raw


# ── tests ────────────────────────────────────────────────────────────────────

def test_prepare_runs_converter(tmp_path) -> None:
    raw = _prepared(tmp_path)
    proj = SafetensorsResource(str(tmp_path), device="cpu").subtree("proj")
    # canonical w_up == transpose(raw w_up): the convert @func actually ran
    assert torch.allclose(proj.load("w_up"), raw["proj.w_up"].t())
    assert torch.allclose(proj.load("w_down"), raw["proj.w_down"])


def test_runtime_twin_matches_reference(tmp_path) -> None:
    _prepared(tmp_path)
    rt = TinyMlpRT()
    rt.load(SafetensorsResource(str(tmp_path), device="cuda"))
    x = torch.randn(_B, _D, dtype=torch.float32, device="cuda")
    ref = SafetensorsResource(str(tmp_path), device="cuda")

    # structural mirror
    assert rt.name == tinymlp_ir.name
    assert rt.entry == tinymlp_ir.entry
    assert [m.name for m in rt.modules] == [m.name for m in tinymlp_ir.modules]

    # child node: runtime forward vs the semantic module's own forward (evaluator)
    def proj_semantic(x):
        return proj_ir.forward(ref.subtree("proj"), x)

    assert check(rt.proj, proj_semantic, (x,)).passed is True

    # root (composite): chain the semantic forward per node, mirroring the twin
    def root_semantic(x):
        mid = proj_ir.forward(ref.subtree("proj"), x)
        return tinymlp_ir.forward(ref, mid)

    assert check(rt, root_semantic, (x,)).passed is True


def test_state_from_prepared_dir_bench_and_missing_attr(tmp_path) -> None:
    raw = _prepared(tmp_path)
    rt = TinyMlpRT()
    rt.load(SafetensorsResource(str(tmp_path), device="cuda"))

    kv = rt.proj.kv
    assert kv.device.type == "cuda"
    assert kv.shape == (_B, _D)
    assert torch.allclose(kv.cpu(), raw["proj.kv"])

    x = torch.randn(_B, _D, dtype=torch.float32, device="cuda")
    assert bench(rt, (x,), iters=10, device=H200SXM()).metrics["mean_ms"] > 0

    with pytest.raises(AttributeError):
        rt.no_such_fn
