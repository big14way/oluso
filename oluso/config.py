"""Project-wide constants: model ids, language settings, paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Target model. Gated with automatic approval: accept the licence on the model page,
# then provide HF_TOKEN. Revision is pinned for reproducibility.
MODEL_ID = os.environ.get("OLUSO_MODEL", "CohereLabs/tiny-aya-earth")
MODEL_REVISION = os.environ.get("OLUSO_MODEL_REVISION", "0624fafbe200f930b89566fc4ec4da314ba48d89")

# Ungated, randomly initialised model with the same architecture (cohere2).
# Used by tests and offline runs to exercise the real transformers code path.
TINY_TEST_MODEL = "hf-tiny-v2/tiny-random-Cohere2ForCausalLM"

# The six language settings evaluated throughout.
LANGS: dict[str, str] = {
    "en": "English",
    "yo": "Yoruba",
    "ig": "Igbo",
    "ha": "Hausa",
    "pcm": "Nigerian Pidgin",
    "pcm-en": "Pidgin-English code-mixed",
}

RUNS_DIR = ROOT / "runs"  # scratch outputs, not committed
RESULTS_DIR = ROOT / "results"  # gate evidence and tables, committed
RECORDS_DIR = ROOT / "records"  # ids of things created on external services, committed
