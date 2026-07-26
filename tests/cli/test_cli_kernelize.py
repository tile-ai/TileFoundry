"""CLI tests for the `kernelize` verb -- the kernelize-path analogue of
`test_cli.py`'s `analyze` coverage: same `load_authored_ir` source loading,
same `#`-headed machine-parsable output style, exercised end to end
(`extract` -> `build_schedule_tree` -> `solve_resources` -> `emit_scaffold`) over a
matmul-based HIR Function (kernelize's `extract` is only real for
matmul-containing HIR today -- see `tests/schedule/test_solve.py`'s
own `bf16_gemm_rmsnorm`, reused here verbatim as the CLI fixture).
"""
from __future__ import annotations

import textwrap

from tilefoundry import cli

_BF16_GEMM_RMSNORM_MODULE = """
    from tilefoundry import func
    from tilefoundry.dsl import Tensor
    from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically

    @func(target="cuda")
    def bf16_gemm_rmsnorm(
        x: Tensor[(64, 128), "bf16"],
        w: Tensor[(128, 64), "bf16"],
        weight: Tensor[(64,), "f32"],
    ) -> Tensor[(64, 64), "bf16"]:
        h = matmul(x, w)
        y = rms_norm(h, weight)
        return y
"""

_UNTARGETED_MODULE = _BF16_GEMM_RMSNORM_MODULE.replace('@func(target="cuda")', "@func")


def _write_module(tmp_path, source: str = _BF16_GEMM_RMSNORM_MODULE):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_kernelize_prints_skeleton_swimlane_and_hole_contracts(tmp_path, capsys) -> None:
    path = _write_module(tmp_path)

    assert cli.main(["kernelize", f"{path}:bf16_gemm_rmsnorm"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    out = captured.out

    # Header: `#`-headed, machine-parsable, analyze-style summary.
    assert out.startswith(
        "# kernelize target=cuda function=bf16_gemm_rmsnorm statements=MM,RN"
    )
    assert "# decisions status=" in out
    assert "makespan=" in out
    assert "# decisions statement=MM atom=SM80_16x8x16_F32BF16BF16F32_TN" in out
    assert "# decisions statement=RN atom=none" in out
    assert "# ring " in out

    # Skeleton: holed C-like loop nest.
    assert "# skeleton" in out
    assert "HOLE_MM(" in out
    assert "HOLE_RN(" in out

    # Swimlane: Mermaid gantt.
    assert "# swimlane" in out
    assert "```mermaid" in out
    assert "gantt" in out

    # Hole contracts: op type / coords / input tensor names / output.
    assert "# holes" in out
    assert "hole=HOLE_MM op=MatMul" in out
    assert "hole=HOLE_RN op=RMSNorm" in out
    assert "inputs=x,w" in out
    assert "output=h" in out
    assert "inputs=h,weight" in out
    assert "output=y" in out


def test_kernelize_explicit_target_matches_the_function_default(tmp_path, capsys) -> None:
    path = _write_module(tmp_path)

    assert cli.main(["kernelize", f"{path}:bf16_gemm_rmsnorm", "--target", "cuda"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("# kernelize target=cuda")


def test_kernelize_falls_back_to_default_target_when_function_has_none(tmp_path, capsys) -> None:
    path = _write_module(tmp_path, _UNTARGETED_MODULE)

    assert cli.main(["kernelize", f"{path}:bf16_gemm_rmsnorm"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("# kernelize target=cuda")


def test_kernelize_rejects_an_unknown_target(tmp_path, capsys) -> None:
    path = _write_module(tmp_path)

    assert cli.main(["kernelize", f"{path}:bf16_gemm_rmsnorm", "--target", "tpu"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unknown target" in captured.err
