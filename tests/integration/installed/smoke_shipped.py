"""What the wheel put on disk, and where the commands find it."""

from __future__ import annotations

from pathlib import Path

MODELS = (
    "deepseek_v4_flash",
    "gemma2_2b",
    "kimi_linear_48b_a3b",
    "minicpm3_4b",
    "qwen2_5_1_5b",
    "qwen3_1_7b",
    "qwen3_5_35b_a3b",
)


def test_every_shipped_kind_resolves_inside_the_installation(installation, shipped) -> None:
    """A kind resolving outside the environment is reading somebody else's files."""
    prefix = installation.resolve()
    astray = {
        kind: where
        for kind, where in shipped.items()
        if not Path(where).resolve().is_relative_to(prefix)
    }
    assert not astray, f"resolved outside {installation}: {astray}"


def test_each_model_ships_its_source_and_its_config(shipped) -> None:
    """`model.py` alone is not a model: it reads the config beside it."""
    models = Path(shipped["models"])
    assert (models / "catalog.json").is_file()
    for name in MODELS:
        assert (models / name / "model.py").is_file(), name
        assert (models / name / "config.json").is_file(), name
