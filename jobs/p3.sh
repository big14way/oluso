#!/usr/bin/env bash
# P3 steering on a GPU machine. Uploads runs/p3/ to the PRIVATE scratch dataset
# <ns>/oluso-build under p3/<stamp>/ after every step.
#
#   STEPS="fit screen dev judge-dev select" (default): fit CAST, screen, judge finalists, select
#   STEPS="test judge-test calib" with P3_FROM=p3/<stamp>: the locked test set, once
#   P3_LIMIT=2 for a smoke run (rows per stratum)
#
# HF Jobs (pin the script URL to a commit SHA; branch URLs are CDN-cached):
#   hf jobs run --flavor l4x1 (or a10g-small, l40sx1) --timeout 4h --secrets HF_TOKEN \
#     -e OLUSO_REF=<sha> ghcr.io/astral-sh/uv:python3.12-bookworm \
#     bash -c "curl -fsSL https://raw.githubusercontent.com/big14way/oluso/<sha>/jobs/p3.sh | bash"
set -euo pipefail

REF="${OLUSO_REF:-main}"
WORK="${WORK:-/tmp/oluso}"
STEPS="${STEPS:-fit screen dev judge-dev select}"
LIMIT_ARG=${P3_LIMIT:+--limit $P3_LIMIT}
export OLUSO_DTYPE="${OLUSO_DTYPE:-float32}"  # same batch-invariant setting as the P2 baseline

git clone --quiet https://github.com/big14way/oluso.git "$WORK"
cd "$WORK"
git checkout --quiet "$REF"
uv sync --frozen --extra data
uv run oluso env

NS=$(uv run python -c "import json; print(json.load(open('records/hub.json'))['namespace'])")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DEST="p3/$STAMP${P3_LIMIT:+-smoke}"
mkdir -p runs/p3
upload() { uv run hf upload "$NS/oluso-build" runs/p3 "$DEST" --repo-type dataset --private; }
trap 'echo "step failed; uploading partial results"; upload || true' ERR
# Our own log, uploaded every 5 minutes: the Jobs log endpoint has returned nothing for some runs.
exec > >(tee -a runs/p3/job.log) 2>&1
upload_log() { uv run hf upload "$NS/oluso-build" runs/p3/job.log "$DEST/job.log" --repo-type dataset --private >/dev/null 2>&1; }
( while sleep 300; do upload_log || true; done ) &
echo "start $(date -u +%FT%TZ) ref=$REF steps=[$STEPS] limit=${P3_LIMIT:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ -n "${P3_FROM:-}" ]]; then
  uv run hf download "$NS/oluso-build" --repo-type dataset --include "$P3_FROM/*" --local-dir /tmp/p3from
  cp /tmp/p3from/"$P3_FROM"/* runs/p3/
fi
for step in $STEPS; do
  echo "=== p3 $step $(date -u +%T) ==="
  case "$step" in
    judge-*|select) uv run oluso p3 "$step" ;;
    *) uv run oluso p3 "$step" $LIMIT_ARG ;;
  esac
  upload
done
echo "uploaded to $NS/oluso-build $DEST"
