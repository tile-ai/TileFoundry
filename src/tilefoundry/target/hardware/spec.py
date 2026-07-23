"""Load and render compact, source-attributed hardware facts."""

from __future__ import annotations

import tomllib
from importlib.resources import files
from typing import Any


def load_h200_sxm_sm90() -> dict[str, Any]:
    """Return the installed H200 SXM / SM90 hardware specification."""
    resource = files("tilefoundry.target.hardware").joinpath("h200_sxm_sm90.toml")
    return tomllib.loads(resource.read_text(encoding="utf-8"))


def load_hardware_spec(target: object) -> dict[str, Any]:
    """Resolve an installed hardware spec for one exact authored target."""
    from tilefoundry.target.cuda import H200SXM, SM90, CudaTarget  # noqa: PLC0415

    if (
        isinstance(target, CudaTarget)
        and type(target.device) is H200SXM
        and type(target.architecture) is SM90
        and target.device.name == "h200_sxm"
        and target.architecture.name == "sm_90"
    ):
        return load_h200_sxm_sm90()
    device = getattr(getattr(target, "device", None), "name", "unknown")
    architecture = getattr(getattr(target, "architecture", None), "name", "unknown")
    raise ValueError(
        "no installed authored-analysis hardware spec for "
        f"device={device!r}, architecture={architecture!r}"
    )


def format_capabilities(
    spec: dict[str, Any], *, grid_cta_count: int | None = None,
) -> str:
    """Format the stable, intentionally compact capabilities report."""
    target = spec["target"]
    lines = [
        f"target: {target['name']}",
        f"device: {target['device']}",
        f"architecture: {target['architecture']}",
        f"grid_cta_count: {grid_cta_count if grid_cta_count is not None else 'unspecified'}",
        "facts:",
    ]
    for name, fact in spec["facts"].items():
        value = fact["value"]
        rendered = str(value) if not fact["unit"] else f"{value} {fact['unit']}"
        lines.append(f"  {name}: {rendered} [{fact['provenance']}]")
        if fact["conditions"]:
            lines.append(f"    conditions: {fact['conditions']}")
        if fact["source"]:
            lines.append(f"    source: {fact['source']}")
    return "\n".join(lines)


__all__ = ["format_capabilities", "load_h200_sxm_sm90", "load_hardware_spec"]
