#!/usr/bin/env bash
# P3 steering on a GPU machine. Uploads runs/p3/ to the PRIVATE scratch dataset
# <ns>/oluso-build under p3/<stamp>/ after every step.
#
#   STEPS="fit screen dev judge-dev select" (default): fit CAST, screen, judge finalists, select
#   STEPS="test judge-test calib" with P3_FROM=p3/<stamp>: the locked test set, once
#   STEPS="bench" after select: bf16 serving throughput (ROI); every generation step also
#   writes <step>_timing.json with per-arm requests/hour and new tokens/s
#   P3_LIMIT=2 for a smoke run (rows per stratum); P3_ARMS=base,cast to run only some test arms
#   OLUSO_RUN=<name> names the output folder p3/<name>; a restarted container sees its log and exits
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
ARMS_ARG=${P3_ARMS:+--arms $P3_ARMS}  # e.g. P3_ARMS=base,cast to split the test set over jobs
export OLUSO_DTYPE="${OLUSO_DTYPE:-float32}"  # same batch-invariant setting as the P2 baseline

git clone --quiet https://github.com/big14way/oluso.git "$WORK"
cd "$WORK"
git checkout --quiet "$REF"
NS=$(python3 -c "import json; print(json.load(open('records/hub.json'))['namespace'])")

# A job killed for memory is silently restarted by the platform (seen: 9 restarts, 9 hours).
# OLUSO_RUN names the run; if its log already exists, this is a restart: stop, do not rerun.
RUN="${OLUSO_RUN:-$(date -u +%Y%m%dT%H%M%SZ)${P3_LIMIT:+-smoke}}"
DEST="p3/$RUN"
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/datasets/$NS/oluso-build/resolve/main/$DEST/job.log")
if [[ "$code" == "200" ]]; then
  echo "run $RUN already started once (restarted container); exiting without rerunning"
  exit 0
fi

uv sync --frozen --extra data
uv run oluso env
mkdir -p runs/p3
upload() { uv run hf upload "$NS/oluso-build" runs/p3 "$DEST" --repo-type dataset --private; }
trap 'echo "step failed; uploading partial results"; upload || true' ERR
# Our own log, uploaded every 5 minutes: the Jobs log endpoint has returned nothing for some runs.
exec > >(tee -a runs/p3/job.log) 2>&1
upload_log() { uv run hf upload "$NS/oluso-build" runs/p3/job.log "$DEST/job.log" --repo-type dataset --private >/dev/null 2>&1; }
echo "start $(date -u +%FT%TZ) run=$RUN ref=$REF steps=[$STEPS] limit=${P3_LIMIT:-none}"
upload_log || true  # marks the run as started (the restart check above)
( while sleep 300; do upload_log || true; done ) &
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
free -g | head -2 || true

if [[ -n "${P3_FROM:-}" ]]; then
  uv run hf download "$NS/oluso-build" --repo-type dataset --include "$P3_FROM/*" --local-dir /tmp/p3from
  for f in /tmp/p3from/"$P3_FROM"/*; do  # not the earlier run's log: this run tees its own
    [[ "$(basename "$f")" == job.log ]] || cp "$f" runs/p3/
  done
fi
for step in $STEPS; do
  echo "=== p3 $step $(date -u +%T) ==="
  case "$step" in
    judge-*|select) uv run oluso p3 "$step" ;;
    test|calib) uv run oluso p3 "$step" $LIMIT_ARG $ARMS_ARG ;;
    *) uv run oluso p3 "$step" $LIMIT_ARG ;;
  esac
  upload
done
echo "uploaded to $NS/oluso-build $DEST"
