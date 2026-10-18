"""Model config snapshots and local resource paths."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def config(model):
    data = json.loads((ROOT / "configs" / f"{model}.json").read_text())
    return data.get("text_config", data)
