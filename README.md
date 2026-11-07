# Oluso

**Measuring and repairing safety and calibration drift in Nigerian languages, without retraining.**

Oluso (Olùṣọ́, guardian) evaluates a small open-weight multilingual model on Yoruba, Igbo, Hausa and Nigerian Pidgin, including code-mixed prompts, on two failure modes at once: safety drift (refusal collapse and over-refusal across languages) and calibration drift (confidence that no longer tracks accuracy). It then repairs both at inference time with activation steering and per-language temperature scaling, and reports before and after results on four metrics.

## Status

In active development. Results, the write-up and a demo will be added here as each stage lands.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # optional; only needed for the real model
make check             # lint + tests, no credentials needed
```

Everything has a credential-free mode:

| Command | Needs | What it does |
| --- | --- | --- |
| `uv run oluso env` | nothing | prints device, dtype, library versions, token status |
| `uv run oluso smoke --backend mock` | nothing | canned replies, exercises the pipeline end to end |
| `uv run oluso smoke --backend tiny` | internet | random-weight cohere2 model, real transformers code path |
| `uv run oluso smoke --backend hf` | HF token | Tiny Aya Earth generates Yoruba; writes `results/p0_smoke.json` |
| `uv run oluso hub-init` | nothing | dry run of the public Hugging Face collections |
| `uv run oluso hub-init --apply` | HF write token | creates the collections, records slugs in `records/hub.json` |

### Using the real model

[Tiny Aya Earth](https://huggingface.co/CohereLabs/tiny-aya-earth) (3.35B, CC-BY-NC-4.0) is gated with automatic approval. Accept the licence on the model page, then either put `HF_TOKEN=...` in `.env` or run `uv run hf auth login`. The model revision is pinned in `oluso/config.py`.

Hardware: a free Colab or Kaggle T4 (fp16), or an Apple Silicon Mac with 16 GB or more (bf16 on MPS). See `notebooks/00_env_smoke.ipynb` for the Colab runner.

## License

Code: MIT. Tiny Aya model weights are CC-BY-NC-4.0 and are not redistributed here.
