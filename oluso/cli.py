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


def cmd_data(args) -> int:
    from collections import Counter

    from oluso.data import pidgin, sources, testset

    if args.step == "tukabench":
        rows = [r.to_dict() for r in sources.tukabench_rows()]
        _print(testset.record_part("tukabench", testset.TEST_DIR / "tukabench.jsonl", rows))
    elif args.step == "pcm-draft":
        goals = sources.english_goals(sources.tukabench_rows())
        rows = pidgin.build_pcm(goals, mock=args.mock, device=args.device)
        path = pidgin.write_draft("pcm_draft_mock" if args.mock else "pcm_draft", rows)
        fails = Counter(reason.split(":")[0] for r in rows for reason in r["gate_reasons"])
        passed = sum(r["gate_pass"] for r in rows)
        _print({"written": path, "rows": len(rows), "passed": passed, "fail_reasons": fails})
    elif args.step == "pcm-rewrite":
        goals = sources.english_goals(sources.tukabench_rows())
        drafts = pidgin.read_draft("pcm_draft_mock" if args.mock else "pcm_draft")
        pcm_rows, mix_rows = pidgin.build_rewrites(
            goals, drafts, backend=args.backend, mock=args.mock, device=args.device
        )
        sfx = "_mock" if args.mock else ""
        out = {}
        for name, rows in (("pcm_candidates", pcm_rows), ("pcmen_draft", mix_rows)):
            fails = Counter(x.split(":")[0] for r in rows for x in r["gate_reasons"])
            out[name] = {
                "written": pidgin.write_draft(name + sfx, rows),
                "passed": sum(r["gate_pass"] for r in rows),
                "fail_reasons": fails,
            }
        out["pcm_origin"] = Counter(r["origin"] for r in pcm_rows)
        _print(out)
    elif args.step == "pcm-finalize":
        pcm_rows, mix_rows = pidgin.finalize(mock=args.mock, device=args.device)
        out = {}
        for name, rows in (("pcm", pcm_rows), ("pcm-en", mix_rows)):
            out[name] = {
                "rows": len(rows),
                "passed": sum(r["gate_pass"] for r in rows),
                "origin": Counter(r["origin"] for r in rows),
                "fail_reasons": Counter(x.split(":")[0] for r in rows for x in r["gate_reasons"]),
            }
        out["review_sheet"] = "data/review/pidgin_review.csv"
        _print(out)
    elif args.step == "pcm-lock":
        rows = pidgin.lock_rows()
        _print(testset.record_part("pidgin", testset.TEST_DIR / "pidgin.jsonl", rows))
    elif args.step == "fitset":
        from oluso.data import fitset

        test = testset.read_jsonl(testset.TEST_DIR / "tukabench.jsonl")
        goals = sorted({r["goal_en"] for r in test if r["setting"] == "en"})
        rows, report = fitset.build_english(goals)
        out = fitset.translate_rows(rows, mock=args.mock, device=args.device)
        path = pidgin.write_draft("fit_mt_mock" if args.mock else "fit_mt", out)
        by_lang = Counter(r["setting"] for r in out if r.get("gate_pass", True))
        _print({**report, "written": path, "passing_rows_by_setting": by_lang})
    elif args.step == "fitset-finalize":
        from oluso import config
        from oluso.data import fitset

        out = config.ROOT / "data" / "fit" / "fit.jsonl"
        report = fitset.finalize(pidgin.BUILD_DIR, out)
        report["sha256"] = testset.sha256(out)
        _print({**report, "written": str(out.relative_to(config.ROOT))})
    elif args.step == "verify":
        res = testset.verify()
        _print(res)
        return 0 if all(res.values()) else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oluso")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="build and lock datasets")
    d.add_argument(
        "step",
        choices=[
            "tukabench",
            "pcm-draft",
            "pcm-rewrite",
            "pcm-finalize",
            "pcm-lock",
            "fitset",
            "fitset-finalize",
            "verify",
        ],
    )
    d.add_argument("--mock", action="store_true", help="offline: mock MT and language ID")
    d.add_argument("--device", default=None)
    d.add_argument("--backend", choices=["global", "tiny"], default="global",
                   help="rewrite model for pcm-rewrite")  # fmt: skip
    d.set_defaults(fn=cmd_data)

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
