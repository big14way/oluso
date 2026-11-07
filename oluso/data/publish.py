"""Curate with CuratorKIT and publish to the Hugging Face Hub (public).

Each language setting is curated as its own shard (so deduplication never collapses
a Pidgin row into its English source): JSONL reader with a preprocessing hook,
schema gate, exact dedup, PPO export. CuratorKIT's exporters drop metadata, so the
passed samples (which keep it) are written and pushed by this module.
"""

from __future__ import annotations

import json
import tempfile
from importlib.metadata import version
from pathlib import Path

from oluso import config

META_KEYS = ("id", "setting", "label", "index", "goal_en", "source", "split", "side", "origin")


def to_sample(row: dict):
    from curatorkit.interfaces import DataSample

    text = row.get("prompt") or row.get("text")
    meta = {k: row[k] for k in META_KEYS if k in row}
    return DataSample(
        id=row["id"],
        source_uri=row.get("source") or "oluso",
        instruction=text,
        task_type="prompt_only",
        metadata=meta,
    )


def curate_shard(rows: list[dict], out_dir: Path) -> tuple[list[dict], dict]:
    from curatorkit.connectors.jsonl import JSONLReader
    from curatorkit.exporters.ppo import PPOExporter
    from curatorkit.gates.schema import SchemaGate
    from curatorkit.normalizers.dedup import ExactDeduplicator
    from curatorkit.pipeline import Pipeline

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        src = f.name
    result = Pipeline(
        steps=[
            JSONLReader(src, preprocessing_fn=to_sample),
            SchemaGate(required_fields=["instruction"], min_tokens=3),
            ExactDeduplicator(),
            PPOExporter(),
        ],
        output_dir=out_dir,
    ).run()
    Path(src).unlink(missing_ok=True)
    passed = []
    for s in result.passed:
        row = {**s.metadata, "prompt": s.instruction}
        if "index" in row:
            row["index"] = int(row["index"])  # the JSONL reader stringifies values
        passed.append(row)
    rejected = [f"{r.metadata.get('raw_preview', {}).get('id')}: {r.rejection_reason}"
                for r in result.rejected]  # fmt: skip
    return passed, {"stage_counts": result.stage_counts, "rejected": rejected}


def curate(rows: list[dict], name: str, key: str = "setting") -> tuple[dict[str, list[dict]], dict]:
    """Curate rows shard by shard; returns ({shard: rows}, manifest)."""
    out_root = config.ROOT / "data" / "curated" / name
    shards: dict[str, list[dict]] = {}
    for r in rows:
        shards.setdefault(r[key], []).append(r)
    curated, manifest = {}, {"curatorkit": version("curatorkit"), "shards": {}}
    for shard, shard_rows in sorted(shards.items()):
        passed, info = curate_shard(shard_rows, out_root / shard)
        curated[shard] = sorted(passed, key=lambda r: r["id"])
        manifest["shards"][shard] = {"in": len(shard_rows), "out": len(passed), **info}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return curated, manifest


def push(curated: dict[str, list[dict]], repo_id: str, card: str, splits=("test",)) -> str:
    """Push one config per shard (rows routed to splits by their `split` field), then the card."""
    from datasets import Dataset
    from huggingface_hub import DatasetCard

    from oluso.env import hf_token

    token = hf_token()
    for shard, rows in curated.items():
        for split in splits:
            part = rows if len(splits) == 1 else [r for r in rows if r["split"] == split]
            Dataset.from_list(part).push_to_hub(
                repo_id, config_name=shard, split=split, private=False, token=token
            )
    # Keep the `configs:` YAML that push_to_hub wrote; replace only the body.
    hub_card = DatasetCard.load(repo_id, repo_type="dataset", token=token)
    hub_card.data.license = "cc-by-nc-4.0"
    hub_card.data.language = ["en", "yo", "ig", "ha", "pcm"]
    hub_card.text = card
    hub_card.push_to_hub(repo_id, repo_type="dataset", token=token)
    return f"https://huggingface.co/datasets/{repo_id}"
