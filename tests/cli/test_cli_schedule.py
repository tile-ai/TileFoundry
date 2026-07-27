"""CLI tests for the `schedule` verb -- the schedule-path analogue of
`test_cli.py`'s `analyze` coverage: same `load_authored_ir` source loading,
same `#`-headed machine-parsable output style, exercised end to end
(`extract` -> `build_schedule_tree` -> `select_atoms` -> `emit_scaffold`).

Two fixtures, because the verb's two arguments are the target the Function was
authored against and the level to schedule it at: a CUDA kernel scheduled at
`cta`, and an AMX kernel blocked to the AMX register files scheduled at `core`.
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

_AMX_BLOCKED_MODULE = """
    from tilefoundry import func
    from tilefoundry.dsl import Tensor
    from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul resolved dynamically

    @func(target="amx")
    def blocked_matmul(
        x: Tensor[(32, 64), "f32"],
        w: Tensor[(64, 32), "f32"],
    ) -> Tensor[(32, 32), "f32"]:
        return matmul(x, w)
"""

_UNTARGETED_MODULE = _BF16_GEMM_RMSNORM_MODULE.replace('@func(target="cuda")', "@func")


def _write_module(tmp_path, source: str = _BF16_GEMM_RMSNORM_MODULE):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_schedule_prints_skeleton_swimlane_and_hole_contracts(tmp_path, capsys) -> None:
    path = _write_module(tmp_path)

    assert cli.main(["schedule", f"{path}:bf16_gemm_rmsnorm", "--stage", "cta"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    out = captured.out

    # Header: `#`-headed, machine-parsable, analyze-style summary.
    assert out.startswith(
        "# schedule target=cuda stage=cta function=bf16_gemm_rmsnorm statements=MM,RN"
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


def test_schedule_reaches_the_amx_atom_at_the_core_level(tmp_path, capsys) -> None:
    """The same verb over the other target: a 32x32 f32 accumulator is exactly
    the AMX Z file, so the core level's decisions name the register-resident
    atom rather than the cache-streaming one."""
    path = _write_module(tmp_path, _AMX_BLOCKED_MODULE)

    assert cli.main(["schedule", f"{path}:blocked_matmul", "--stage", "core"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("# schedule target=amx stage=core function=blocked_matmul")
    assert "# decisions statement=MM atom=AMX_FMA32_16x16x1_F32" in captured.out


def test_schedule_rejects_a_selection_that_declares_no_execution_context(
    tmp_path, capsys
) -> None:
    """A kernel is scheduled against the device it was authored for. A bare
    ``@func`` declares no context, so it resolves to a Function and is rejected
    rather than scheduled against a default target the author never named."""
    path = _write_module(tmp_path, _UNTARGETED_MODULE)

    assert cli.main(["schedule", f"{path}:bf16_gemm_rmsnorm", "--stage", "cta"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "select the Module that declares it" in captured.err


def test_schedule_names_the_levels_a_target_owns_when_the_stage_is_not_one(
    tmp_path, capsys
) -> None:
    path = _write_module(tmp_path, _AMX_BLOCKED_MODULE)

    assert cli.main(["schedule", f"{path}:blocked_matmul", "--stage", "cta"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "has no topology level 'cta'" in captured.err
    assert "must be one of core, amx" in captured.err


_NESTED_MODULE = """
    from tilefoundry import func, module
    from tilefoundry.dsl import Tensor
    from tilefoundry.dsl.tf import *  # noqa: F401,F403 -- matmul/rms_norm resolved dynamically
    from tilefoundry.ir.types.shard import Topology

    @module(entry="bf16_gemm_rmsnorm", target="cuda")
    class Model:
        topologies = (Topology("cta", 132),)

        @func
        def bf16_gemm_rmsnorm(
            x: Tensor[(64, 128), "bf16"],
            w: Tensor[(128, 64), "bf16"],
            weight: Tensor[(64,), "f32"],
        ) -> Tensor[(64, 64), "bf16"]:
            h = matmul(x, w)
            y = rms_norm(h, weight)
            return y

        @module(entry="inner")
        class child:
            @func
            def inner(
                x: Tensor[(64, 128), "bf16"],
                w: Tensor[(128, 64), "bf16"],
                weight: Tensor[(64,), "f32"],
            ) -> Tensor[(64, 64), "bf16"]:
                h = matmul(x, w)
                y = rms_norm(h, weight)
                return y
"""


def test_schedule_selects_a_function_through_a_nested_module_path(
    tmp_path, capsys
) -> None:
    """A leaf is selectable by the path of owners it hangs under, and the
    selection still resolves the Target and hierarchy it inherits from them
    rather than the empty context a detached copy would carry."""
    path = _write_module(tmp_path, _NESTED_MODULE)

    assert cli.main(["schedule", f"{path}:Model.child.inner", "--stage", "cta"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("# schedule target=cuda stage=cta function=inner")
