#!/usr/bin/env bash
# P2 baseline on a GPU machine: generate (Tiny Aya Earth answers + AfriMMLU option scores),
# then judge (MT -> WildGuard, AfriGuard refusal classifier, English anchors, flip test).
# Uploads runs/p2/ to the PRIVATE scratch dataset <ns>/oluso-build under p2/<stamp>/.
#
#   STEPS="generate judge" (default) | "judge" or "calib" with P2_FROM=p2/<stamp> to redo one part
#   P2_LIMIT=2 for a smoke run (rows per stratum)
#
# HF Jobs (pin the script URL to a commit SHA; branch URLs are CDN-cached):
#   hf jobs run --flavor a10g-small --timeout 3h --secrets HF_TOKEN \
#     -e OLUSO_REF=<sha> ghcr.io/astral-sh/uv:python3.12-bookworm \
#     bash -c "curl -fsSL https://raw.githubusercontent.com/big14way/oluso/<sha>/jobs/p2.sh | bash"
set -euo pipefail

REF="${OLUSO_REF:-main}"
WORK="${WORK:-/tmp/oluso}"
STEPS="${STEPS:-generate judge}"
LIMIT_ARG=${P2_LIMIT:+--limit $P2_LIMIT}
export OLUSO_DTYPE="${OLUSO_DTYPE:-float32}"  # bf16 left-padded batches drift from unbatched greedy

git clone --quiet https://github.com/big14way/oluso.git "$WORK"
cd "$WORK"
git checkout --quiet "$REF"
uv sync --frozen --extra data
uv run oluso env

NS=$(uv run python -c "import json; print(json.load(open('records/hub.json'))['namespace'])")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DEST="p2/$STAMP${P2_LIMIT:+-smoke}"
upload() { uv run hf upload "$NS/oluso-build" runs/p2 "$DEST" --repo-type dataset --private; }
trap 'echo "step failed; uploading partial results"; upload || true' ERR

if [[ " $STEPS " != *" generate "* ]]; then
  uv run hf download "$NS/oluso-build" --repo-type dataset --include "$P2_FROM/*" --local-dir /tmp/p2from
  mkdir -p runs/p2 && cp /tmp/p2from/"$P2_FROM"/*.jsonl /tmp/p2from/"$P2_FROM"/gen_meta.json runs/p2/
fi
for step in $STEPS; do
  echo "=== p2 $step ==="
  uv run oluso p2 "$step" $LIMIT_ARG
  upload
done
echo "uploaded to $NS/oluso-build $DEST"
