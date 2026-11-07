"""Runtime environment detection: device, dtype, credentials."""

from __future__ import annotations

import os
import platform

import torch

from oluso import config  # noqa: F401  (importing config loads .env)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    forced = os.environ.get("OLUSO_DTYPE")  # e.g. float32 for batch-invariant evaluation runs
    if forced:
        return getattr(torch, forced)
    if device == "cuda":
        # T4 (Colab/Kaggle free tier) has no native bf16.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def hf_token() -> str | None:
    """Return the Hugging Face token from the environment or the local login cache."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def report() -> dict:
    device = pick_device()
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "dtype": str(pick_dtype(device)).removeprefix("torch."),
        "hf_token": "present" if hf_token() else "missing",
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    for pkg in ("transformers", "safetune", "auditkit", "curatorkit"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "installed")
        except Exception as exc:  # pragma: no cover - depends on install
            info[pkg] = f"unavailable ({type(exc).__name__})"
    return info
