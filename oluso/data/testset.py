"""The locked test set.

Six settings, 1,600 prompts:
  en, ha, ig, yo   TukaBench harm + benign + culture (100 each per language) = 1,200
  pcm              Nigerian Pidgin versions of the 200 English harm + benign goals
  pcm-en           Pidgin-English code-mixed versions of the same 200 goals

Each part is written as canonical JSONL and its SHA-256 recorded in LOCK.json.
Steering vectors and temperatures are never fitted on these prompts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from oluso import config

TEST_DIR = config.ROOT / "data" / "test"
LOCK = TEST_DIR / "LOCK.json"


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["id"])
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_part(name: str, path: Path, rows: list[dict]) -> dict:
    """Write one part and record its hash and counts in LOCK.json."""
    digest = write_jsonl(path, rows)
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {"parts": {}}
    counts: dict[str, int] = {}
    for r in rows:
        key = f"{r['setting']}/{r['label']}"
        counts[key] = counts.get(key, 0) + 1
    lock["parts"][name] = {
        "file": str(path.relative_to(config.ROOT)),
        "sha256": digest,
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "locked_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    lock["total_rows"] = sum(p["rows"] for p in lock["parts"].values())
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    return lock["parts"][name]


def verify() -> dict:
    """Check every locked part against its recorded hash."""
    lock = json.loads(LOCK.read_text())
    out = {}
    for name, part in lock["parts"].items():
        out[name] = sha256(config.ROOT / part["file"]) == part["sha256"]
    return out
