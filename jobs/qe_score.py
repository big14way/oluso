"""Reference-free translation quality estimation with SSA-COMET-QE.

Runs in its own environment (unbabel-comet needs numpy<2 and transformers<5):
  uv run --isolated --no-project --python 3.11 --with unbabel-comet \
      python jobs/qe_score.py in.jsonl out.jsonl
Input rows: {"id", "src", "mt"}. Output rows add "qe" (0-1, higher is better).
"""

import json
import sys

MODEL = "McGill-NLP/ssa-comet-qe-final"


def main(src_path: str, out_path: str) -> None:
    from comet import download_model, load_from_checkpoint

    rows = [json.loads(line) for line in open(src_path, encoding="utf-8") if line.strip()]
    model = load_from_checkpoint(download_model(MODEL))
    data = [{"src": r["src"], "mt": r["mt"]} for r in rows]
    scores = model.predict(data, batch_size=32, gpus=1).scores
    with open(out_path, "w", encoding="utf-8") as f:
        for r, s in zip(rows, scores, strict=True):
            f.write(json.dumps({**r, "qe": round(float(s), 4)}, ensure_ascii=False) + "\n")
    print(f"scored {len(rows)} rows with {MODEL}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
