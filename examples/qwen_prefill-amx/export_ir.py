"""Export whole-model HIR, scheduled HIR and retained-call TIR."""

import argparse, json
from pathlib import Path
from tilefoundry.inspection import as_script
from tilefoundry.schedule import finalize
from tilefoundry.analysis.check import check_program
from config import config, ROOT
from model import build
from scheduled import make_scheduled


def export(model, sequence=128, output=None):
    out = Path(output) if output else ROOT / "artifacts" / ("ir-" + model)
    out.mkdir(parents=True, exist_ok=True)
    c = config(model)
    plain = build(c, sequence)
    plain, _ = check_program(plain, plain.entry_function())
    scheduled = make_scheduled(c, sequence)

    def registrations(source):
        future = "from __future__ import annotations\n"
        imports = "import instructions, host_instructions\n"
        return (
            source.replace(future, future + imports, 1)
            if future in source
            else imports + source
        )

    (out / "model.py").write_text(as_script(plain))
    (out / "scheduled.py").write_text(registrations(as_script(scheduled)))
    lowered = finalize(scheduled)
    (out / "finalized.py").write_text(registrations(as_script(lowered)))
    report = dict(
        model=model,
        sequence=sequence,
        layers=len(scheduled.layers),
        shared_layer_definitions=len({id(layer) for layer in scheduled.layers}),
        codegen=False,
        performance_measured=False,
    )
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("1.7b", "4b"), required=True)
    p.add_argument("--sequence", type=int, default=128)
    p.add_argument("--output")
    a = p.parse_args()
    export(a.model, a.sequence, a.output)
