"""FSDP training cases shared by numerical and storage checks."""

from dataclasses import dataclass
from pathlib import Path

from tests.core._corpus import Case


@dataclass(frozen=True)
class TrainingCase(Case):
    participants: int = 2
    retain: bool = False


FSDP_CASES = tuple(
    TrainingCase(
        path=Path(__file__).parent / "fixtures" / "evaluator" / f"fsdp_mlp_{participants}.py",
        name=f"Training.{policy}_step",
        participants=participants,
        retain=policy == "retain",
    )
    for participants in (1, 2, 4)
    for policy in ("release", "retain")
)
