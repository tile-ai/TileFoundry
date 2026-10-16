#!/usr/bin/env python3
"""Check one definition against file inputs using an instance's checkpoint subtree.

This example adapter uses the public check API; SOURCE remains Module.function.
"""
import argparse
import json
from pathlib import Path

from tilefoundry.cli.check import (
    CheckRequest,
    add_arguments,
    check_concrete,
    expectations,
    parse_dims,
    read_inputs,
    select,
)
from tilefoundry.runtime import SafetensorsResource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--weight-scope", default="", help="canonical checkpoint subtree, e.g. layers.3.mixer")
    parser.add_argument("--instance-name", help="runtime twin's instance name, e.g. layer3")
    args = parser.parse_args()
    if not args.inputs or not args.inputs.startswith("files:"):
        parser.error("this adapter requires --inputs files:PATH[,PATH...]")
    if not args.weights or not args.weights.startswith("ckpt:"):
        parser.error("this adapter requires --weights ckpt:DIR")
    selection = select(args.source)
    device = args.device or "cuda"
    resource = SafetensorsResource(args.weights[5:], device=device)
    for segment in args.weight_scope.split(".") if args.weight_scope else ():
        resource = resource.subtree(segment)
    if selection.twin is not None and args.instance_name:
        selection.twin.name = args.instance_name
    report = check_concrete(CheckRequest(
        selection.module, selection.twin,
        read_inputs(args.inputs[6:].split(","), device), resource,
        read_inputs(args.expected, device) if args.expected else None,
        device, expectations(args.comparison),
    ))
    outputs = [{
        "path": output.path, "shape": list(output.shape), "dtype": output.dtype,
        "ref_norm": output.ref_norm,
        "fns": [{
            "fn": result.predicate.name,
            **{bound: getattr(result.predicate, bound) for bound in result.predicate.bounds},
            **dict(result.values), "passed": result.passed,
            **({"note": result.note} if result.note is not None else {}),
        } for result in output.results],
    } for output in report.outputs]
    payload = {"target": args.source, "passed": report.passed, "runs": [{
        "passed": report.passed, "outputs": outputs, "dims": parse_dims(args.dim) or {},
    }]}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
    else:
        for output in outputs:
            print(f"  {output['path']}")
            for measured in output["fns"]:
                print(f"    {measured}")
        print("PASS" if report.passed else "FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
