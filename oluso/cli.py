"""Command line entry point: `oluso <command>`."""

from __future__ import annotations

import argparse
import json
import sys


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_env(args) -> int:
    from oluso.env import report

    _print(report())
    return 0


def cmd_smoke(args) -> int:
    from oluso.generate import ModelAccessError
    from oluso.smoke import run

    try:
        res = run(backend=args.backend, max_new_tokens=args.max_new_tokens, device=args.device)
    except ModelAccessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for r in res["rows"]:
        print(f"\n> {r['prompt']}\n{r['output']}\n  [looks_yoruba={r['looks_yoruba']}]")
    summary = {
        k: res[k] for k in ("backend", "model_id", "device", "yoruba_outputs", "gate_passed")
    }
    print()
    _print({**summary, "written": res["path"]})
    if args.backend == "hf":
        return 0 if res["gate_passed"] else 1
    return 0


def cmd_hub_init(args) -> int:
    from oluso.hub import init_collections

    _print(init_collections(apply=args.apply))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oluso")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="print runtime environment").set_defaults(fn=cmd_env)

    s = sub.add_parser("smoke", help="P0 gate: generate Yoruba with the target model")
    s.add_argument("--backend", choices=["hf", "tiny", "mock"], default="hf")
    s.add_argument("--max-new-tokens", type=int, default=128)
    s.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    s.set_defaults(fn=cmd_smoke)

    h = sub.add_parser("hub-init", help="create public HF collections (dry run by default)")
    h.add_argument("--apply", action="store_true")
    h.set_defaults(fn=cmd_hub_init)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
