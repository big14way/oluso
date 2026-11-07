#!/usr/bin/env bash
# SSA-COMET-QE scores for (1) the NLLB fit-set translations into ha/ig/yo and
# (2) TukaBench's human-corrected ha/ig/yo translations, which set the threshold.
# Uploads scores to the PRIVATE scratch dataset. Needs HF_TOKEN.
set -euo pipefail

REF="${OLUSO_REF:-main}"
WORK="${WORK:-/tmp/oluso}"
git clone --depth 1 --branch "$REF" https://github.com/big14way/oluso.git "$WORK"
cd "$WORK"

python3 - <<'EOF'
import json
rows = []
for line in open("data/build/fit_mt.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["setting"] in ("ha", "ig", "yo"):
        rows.append({"id": r["id"], "src": r["en"], "mt": r["text"], "kind": "fit_nllb"})
for line in open("data/test/tukabench.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["setting"] in ("ha", "ig", "yo") and r["label"] in ("harm", "benign"):
        rows.append({"id": r["id"], "src": r["goal_en"], "mt": r["prompt"], "kind": "tukabench_human"})
with open("/tmp/qe_in.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "rows to score")
EOF

uv run --isolated --no-project --python 3.11 --with "unbabel-comet==2.2.7" --with "setuptools<70" \
  python jobs/qe_score.py /tmp/qe_in.jsonl /tmp/qe_scores.jsonl

NS=$(python3 -c "import json; print(json.load(open('records/hub.json'))['namespace'])")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
uvx --from "huggingface_hub[cli]>=1.5" hf upload "$NS/oluso-build" /tmp/qe_scores.jsonl \
  "p1qe/$STAMP/qe_scores.jsonl" --repo-type dataset --private
echo "uploaded to $NS/oluso-build p1qe/$STAMP"
