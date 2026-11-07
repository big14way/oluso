#!/usr/bin/env bash
# P1 heavy data steps on a GPU machine (HF Jobs, Kaggle or Colab).
#
#   Pidgin + code-mixed rewrites with Tiny Aya Global, steering fit-set translation
#   with NLLB-200, then upload of data/build/ to a PRIVATE scratch dataset for review.
#
# Needs HF_TOKEN in the environment (an account that accepted the Tiny Aya licence).
# HF Jobs example (image includes git and uv):
#   hf jobs run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
#     ghcr.io/astral-sh/uv:python3.12-bookworm \
#     bash -c "curl -fsSL https://raw.githubusercontent.com/big14way/oluso/${OLUSO_REF:-main}/jobs/p1_data.sh | bash"
set -euo pipefail

REF="${OLUSO_REF:-main}"
WORK="${WORK:-/tmp/oluso}"

git clone --depth 1 --branch "$REF" https://github.com/big14way/oluso.git "$WORK"
cd "$WORK"
uv sync --frozen --extra data
uv run oluso env

uv run oluso data pcm-rewrite --backend global
uv run oluso data fitset

NS=$(uv run python -c "import json; print(json.load(open('records/hub.json'))['namespace'])")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
uv run hf upload "$NS/oluso-build" data/build "p1/$STAMP" --repo-type dataset --private
echo "uploaded to $NS/oluso-build p1/$STAMP"
