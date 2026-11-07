"""P0 gate: the target model generates Yoruba."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime

from oluso import config
from oluso.env import report
from oluso.generate import load_generator
from oluso.langcheck import looks_yoruba, yoruba_signals

# Benign Yoruba prompts written in standard orthography.
YO_PROMPTS = [
    "Kí ni olú-ìlú orílẹ̀-èdè Nàìjíríà?",  # What is the capital of Nigeria?
    "Kọ gbólóhùn méjì nípa pàtàkì ẹ̀kọ́.",  # Write two sentences on the importance of education.
    "Ṣàlàyé bí a ṣe ń se ẹ̀wà ní ṣókí.",  # Briefly explain how beans are cooked.
]


def run(backend: str = "hf", max_new_tokens: int = 128, device: str | None = None) -> dict:
    t0 = time.perf_counter()
    gen = load_generator(backend, device=device)
    load_s = time.perf_counter() - t0
    outs = gen.chat(YO_PROMPTS, max_new_tokens=max_new_tokens, lang="yo")
    rows = []
    for g in outs:
        rows.append(
            {**asdict(g), "looks_yoruba": looks_yoruba(g.output), **yoruba_signals(g.output)}
        )
    n_yo = sum(r["looks_yoruba"] for r in rows)
    is_target = backend == "hf" and gen.model_id == "CohereLabs/tiny-aya-earth"
    result = {
        "gate": "P0: Tiny Aya Earth generates in Yoruba",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": backend,
        "model_id": gen.model_id,
        "revision": gen.revision,
        "device": gen.device,
        "load_seconds": round(load_s, 1),
        "env": report(),
        "yoruba_outputs": f"{n_yo}/{len(rows)}",
        "gate_passed": bool(is_target and n_yo >= 2),
        "rows": rows,
    }
    out_dir = config.RESULTS_DIR if is_target else config.RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("p0_smoke.json" if is_target else f"p0_smoke_{backend}.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    result["path"] = str(path.relative_to(config.ROOT))
    return result
