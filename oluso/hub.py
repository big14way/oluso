"""Hugging Face collections for all public artefacts.

Dry run by default. With --apply and a token, creates (or reuses) two public
collections and records their slugs in records/hub.json.
"""

from __future__ import annotations

import json

from oluso import config
from oluso.env import hf_token

COLLECTIONS = {
    "models": (
        "oluso-models",
        "Oluso: steering vectors, temperature tables and steered checkpoints for "
        "Tiny Aya Earth on Yoruba, Igbo, Hausa and Nigerian Pidgin.",
    ),
    "datasets": (
        "oluso-datasets",
        "Oluso: harmful/benign prompt sets and calibration subsets across English, Yoruba, "
        "Igbo, Hausa, Nigerian Pidgin and Pidgin-English code-mixed text.",
    ),
}

RECORD = config.RECORDS_DIR / "hub.json"


def init_collections(apply: bool = False) -> dict:
    token = hf_token()
    plan = {k: {"title": t, "private": False} for k, (t, _) in COLLECTIONS.items()}
    if not apply or not token:
        reason = "dry run" if not apply else "no HF token; dry run"
        return {"dry_run": True, "reason": reason, "plan": plan}

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    out: dict = {"namespace": user}
    for key, (title, desc) in COLLECTIONS.items():
        existing = [c for c in api.list_collections(owner=user, limit=100) if c.title == title]
        coll = (
            existing[0]
            if existing
            else api.create_collection(title=title, namespace=user, description=desc, private=False)
        )
        out[key] = {"title": title, "slug": coll.slug, "url": coll.url}
    config.RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(out, indent=2) + "\n")
    return out


def add_to_collection(repo_id: str, repo_type: str, key: str, note: str | None = None) -> str:
    """Add a public repo to the models or datasets collection recorded in records/hub.json."""
    from huggingface_hub import HfApi

    slug = json.loads(RECORD.read_text())[key]["slug"]
    api = HfApi(token=hf_token())
    api.add_collection_item(slug, item_id=repo_id, item_type=repo_type, note=note,
                            exists_ok=True)  # fmt: skip
    return slug
