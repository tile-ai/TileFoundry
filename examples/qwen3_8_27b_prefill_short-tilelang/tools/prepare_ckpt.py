#!/usr/bin/env python3
"""Write the canonical-name checkpoint `check --weights ckpt:DIR` reads.

    python tools/prepare_ckpt.py [OUT_DIR]      # reads $QWEN38_CKPT, writes ./ckpt

`Module.prepare` walks the authored Module, reads each declared weight from the
published checkpoint through `hf_alias`, validates shape and dtype against the
declaration and writes it under its canonical dotted name.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model  # noqa: E402
import paths  # noqa: E402
from hf_alias import hf_alias  # noqa: E402
from tilefoundry.runtime import SafetensorsResource  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "ckpt")
t0 = time.time()
raw = SafetensorsResource(str(paths.need("ckpt")), device="cpu", alias=hf_alias(model.config))
model.Qwen3_8_27B.prepare(raw, str(out), device="cpu")
print(f"prepared {out} in {time.time() - t0:.0f} s")
