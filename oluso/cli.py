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
    elif args.step == "publish":
        from oluso import config
        from oluso.data import publish

        if not all(testset.verify().values()):
            raise SystemExit("locked test set fails verification; refusing to publish")
        test = [r for part in ("tukabench", "pidgin")
                for r in testset.read_jsonl(testset.TEST_DIR / f"{part}.jsonl")]  # fmt: skip
        fit = [
            {
                "id": r["id"],
                "setting": r["setting"],
                "side": r["side"],
                "split": r["split"],
                "prompt": r["text"],
                "goal_en": r["en"],
                "source": r["source"],
            }  # fmt: skip
            for r in testset.read_jsonl(config.ROOT / "data" / "fit" / "fit.jsonl")
        ]
        ns = json.loads((config.RECORDS_DIR / "hub.json").read_text())["namespace"]
        out = {}
        for name, rows, splits in (("oluso-safety-ng", test, ["test"]),
                                   ("oluso-steer-fit-ng", fit, ["fit", "dev"])):  # fmt: skip
            curated, manifest = publish.curate(rows, name)
            out[name] = {s: m["out"] for s, m in manifest["shards"].items()}
            if args.apply:
                card = (config.ROOT / "cards" / f"{name}.md").read_text(encoding="utf-8")
                out[name]["url"] = publish.push(curated, f"{ns}/{name}", card, splits)
        out["dry_run"] = not args.apply
        _print(out)
    elif args.step == "verify":
        res = testset.verify()
        _print(res)
        return 0 if all(res.values()) else 1
    return 0


def cmd_p2(args) -> int:
    from pathlib import Path

    from oluso import p2, report

    run_dir = Path(args.run_dir)
    if args.step == "generate":
        meta = p2.run_generate(
            backend=args.backend,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            out_dir=run_dir,
        )
        _print(meta)
    elif args.step == "calib":
        _print(p2.run_calib(backend=args.backend, device=args.device, limit=args.limit,
                            out_dir=run_dir))  # fmt: skip
    elif args.step == "judge":
        _print(p2.run_judge(out_dir=run_dir, mock=args.mock, device=args.device, limit=args.limit))
    elif args.step == "relabel":
        from collections import Counter

        from oluso import judge
        from oluso.baseline import write_jsonl

        path = run_dir / "judged.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines() if x]
        for r in rows:
            if r["resp_lid"]:
                r["wrong_lang"] = judge.wrong_language(
                    r["setting"], (r["resp_lid"], r["resp_lid_prob"])
                )
            r["final"] = judge.final_label(r)
        rows = judge.apply_lexicon(rows)
        write_jsonl(path, rows)
        _print({"rows": len(rows), "pl_agree": sum(r["pl_agree"] for r in rows),
                "final": Counter(r["final"] for r in rows)})  # fmt: skip
    elif args.step == "openings":
        from collections import Counter

        from oluso import judge

        rows = [json.loads(x) for x in (run_dir / "judged.jsonl").read_text().splitlines() if x]
        for s in ("en", "yo", "ig", "ha", "pcm", "pcm-en"):
            texts = [r["response"] for r in rows if r["setting"] == s]
            missed = [t for t in texts if not judge.lexicon_refusal(t)]
            heads = Counter(" ".join(judge.fold(t).split()[:5]) for t in missed)
            print(f"== {s}: most common openings NOT matched by the lexicon")
            for h, n in heads.most_common(args.limit or 15):
                print(f"  {n:3d}  {h}")
    elif args.step == "spotcheck":
        from oluso import spotcheck

        judged = [json.loads(x) for x in (run_dir / "judged.jsonl").read_text().splitlines() if x]
        _print(spotcheck.build_sheet(judged, run_dir / "spotcheck"))
    elif args.step == "kappa":
        from oluso import spotcheck

        d = run_dir / "spotcheck"
        res = spotcheck.kappa_report(d / "judge_spotcheck.csv", d / "judge_spotcheck_key.jsonl")
        (run_dir / "kappa.json").write_text(json.dumps(res, indent=2) + "\n")
        _print(res)
    elif args.step == "table":
        out = Path(args.out_dir) if args.out_dir else None
        before = report.before_table(run_dir, out)
        print(report.markdown(before))
    return 0


def cmd_p3(args) -> int:
    from pathlib import Path

    from oluso import p3

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu = {"backend": args.backend, "device": args.device, "out_dir": run_dir}
    if args.step == "fit":
        _print(p3.run_fit(**gpu, limit=args.limit))
    elif args.step == "screen":
        _print(p3.run_screen(**gpu, batch_size=args.batch_size, limit=args.limit))
    elif args.step in ("dev", "test"):
        _print(p3.run_answer(args.step, **gpu, batch_size=args.batch_size, limit=args.limit,
                             kinds=args.arms))  # fmt: skip
    elif args.step.startswith("judge-"):
        stage = args.step.removeprefix("judge-")
        _print(p3.run_judge(stage, out_dir=run_dir, mock=args.mock, device=args.device))
    elif args.step == "select":
        res = p3.select(p3._read(run_dir / "dev_judged.jsonl"))
        (run_dir / "selection.json").write_text(json.dumps(res, indent=2) + "\n")
        _print(res)
    elif args.step == "calib":
        _print(p3.run_calib(**gpu, limit=args.limit, kinds=args.arms))
    elif args.step.startswith("gate-"):
        _print(p3.run_gate(args.step.removeprefix("gate-"), **gpu, limit=args.limit))
    elif args.step == "bench":
        _print(p3.run_bench(**gpu, dtype=args.dtype, batch_size=64, limit=args.limit))
    elif args.step == "table":
        from oluso import report_after

        if args.from_hub:  # rebuild from the public label-only results; results/ is untouched
            import tempfile

            res = report_after.from_hub(Path(tempfile.mkdtemp(prefix="oluso-tables-")))
        else:
            out = Path(args.out_dir) if args.out_dir else None
            res = report_after.after_table(Path(args.p2_dir), run_dir, out)
        print(report_after.markdown(res))
    return 0


def cmd_guard(args) -> int:
    from pathlib import Path

    from oluso import config, guard

    if args.step == "export":
        t = args.temperature
        if t is None:  # the global temperature from the after-table (all languages' validation)
            cal = json.loads((config.RESULTS_DIR / "p3_after.json").read_text())["calibration"]
            t = round(cal["base"]["en"]["global_T"], 3)  # one T, shared by every language
        prov = {"selection": "dev split, pre-stated rule (see results/p3_selection.json)",
                "fit_set": "alanmik/oluso-steer-fit-ng (fit split)",
                "test_set": "alanmik/oluso-safety-ng (locked, run once)",
                "temperature": "one global T fitted by NLL on AfriMMLU validation (en/yo/ig/ha)",
                "code": "https://github.com/big14way/oluso"}  # fmt: skip
        _print(guard.export(Path(args.run_dir), Path(args.out_dir), t, prov))
    else:
        _print(guard.publish(Path(args.out_dir), apply=args.apply))
    return 0


def cmd_demo(args) -> int:
    from pathlib import Path

    from oluso import demo

    out = Path(args.out_dir)
    _print(demo.write(out) if args.step == "build" else demo.publish(out, apply=args.apply))
    return 0


def cmd_export(args) -> int:
    from pathlib import Path

    from oluso import export

    _print(export.run(Path(args.p2_dir), Path(args.p3_dir), apply=args.apply))
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
            "publish",
            "verify",
        ],
    )
    d.add_argument("--mock", action="store_true", help="offline: mock MT and language ID")
    d.add_argument("--device", default=None)
    d.add_argument("--apply", action="store_true", help="publish: push to the Hub (else dry run)")
    d.add_argument("--backend", choices=["global", "tiny"], default="global",
                   help="rewrite model for pcm-rewrite")  # fmt: skip
    d.set_defaults(fn=cmd_data)

    sub.add_parser("env", help="print runtime environment").set_defaults(fn=cmd_env)

    q = sub.add_parser("p2", help="baseline: generate, judge, before-table")
    steps = ["generate", "calib", "judge", "relabel", "openings", "spotcheck", "kappa", "table"]
    q.add_argument("step", choices=steps)
    q.add_argument("--backend", choices=["hf", "tiny", "mock"], default="hf")
    q.add_argument("--mock", action="store_true", help="judge: offline mock judges")
    q.add_argument("--device", default=None)
    q.add_argument("--batch-size", type=int, default=32)
    q.add_argument("--limit", type=int, default=None, help="rows per stratum (smoke runs)")
    q.add_argument("--run-dir", default="runs/p2")
    q.add_argument("--out-dir", default=None, help="table: where to write (default results/)")
    q.set_defaults(fn=cmd_p2)

    q = sub.add_parser("p3", help="repair: CAST steering fit, screen, dev selection, test arms")
    steps = [
        "fit",
        "screen",
        "dev",
        "judge-dev",
        "select",
        "test",
        "judge-test",
        "calib",
        "bench",
        "gate-dev",
        "gate-test",
        "table",
    ]
    q.add_argument("step", choices=steps)
    q.add_argument("--backend", choices=["hf", "tiny", "mock"], default="hf")
    q.add_argument("--mock", action="store_true", help="judge: offline mock judges")
    q.add_argument("--device", default=None)
    q.add_argument("--batch-size", type=int, default=32)
    q.add_argument("--limit", type=int, default=None, help="rows per stratum (smoke runs)")
    q.add_argument("--run-dir", default="runs/p3")
    q.add_argument("--p2-dir", default="runs/p2", help="table: the baseline run")
    q.add_argument("--arms", default=None, help="test/calib: comma list of arm kinds to run")
    q.add_argument("--out-dir", default=None, help="table: where to write (default results/)")
    q.add_argument("--dtype", default="bfloat16", help="bench: serving dtype to time")
    q.add_argument("--from-hub", action="store_true",
                   help="table: rebuild from the public results dataset")  # fmt: skip
    q.set_defaults(fn=cmd_p3)

    g = sub.add_parser("guard", help="export the guard locally, or publish it (dry run by default)")
    g.add_argument("step", choices=["export", "publish"])
    g.add_argument("--run-dir", default="runs/p3")
    g.add_argument("--out-dir", default="dist/guard")
    g.add_argument("--temperature", type=float, default=None, help="export: default global T")
    g.add_argument("--apply", action="store_true", help="publish: push it public to the Hub")
    g.set_defaults(fn=cmd_guard)

    m = sub.add_parser("demo", help="build the static results page, or publish it as a Space")
    m.add_argument("step", choices=["build", "publish"])
    m.add_argument("--out-dir", default="dist/space")
    m.add_argument("--apply", action="store_true", help="publish: create the public Space")
    m.set_defaults(fn=cmd_demo)

    x = sub.add_parser("export", help="label-only results dataset (dry run by default)")
    x.add_argument("--p2-dir", default="runs/p2")
    x.add_argument("--p3-dir", default="runs/p3")
    x.add_argument("--apply", action="store_true", help="push it public to the Hub")
    x.set_defaults(fn=cmd_export)

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
