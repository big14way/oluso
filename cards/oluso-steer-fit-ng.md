# Oluso steering fit set: Nigerian languages

1,292 contrast prompts for fitting refusal-steering directions in language models: harmful requests versus harmless requests that only look risky. Covers English, Yoruba, Igbo, Hausa and Nigerian Pidgin.

This is training and validation data for steering, not a benchmark. It was deduplicated against the [Oluso safety test set](https://huggingface.co/datasets/alanmik/oluso-safety-ng): any prompt with character TF-IDF cosine ≥ 0.4 to a test goal was dropped.

> **Content warning.** The `harm` prompts are harmful requests drawn from HarmBench. They are here to locate and steer refusal behaviour.

## Configs and splits

One config per language. Each has a `fit` split (75%) and a `dev` split (25%), and the English source items are split the same way in every language.

| config | language | harm fit | harm dev | harmless fit | harmless dev | rows |
|---|---|---:|---:|---:|---:|---:|
| `en` | English | 112 | 38 | 112 | 38 | 300 |
| `yo` | Yoruba | 101 | 37 | 111 | 37 | 286 |
| `ig` | Igbo | 100 | 32 | 111 | 37 | 280 |
| `ha` | Hausa | 93 | 32 | 103 | 38 | 266 |
| `pcm` | Nigerian Pidgin | 53 | 22 | 65 | 20 | 160 |

```python
from datasets import load_dataset
fit = load_dataset("alanmik/oluso-steer-fit-ng", "yo", split="fit")
```

## Fields

| field | meaning |
|---|---|
| `id` | `fit:{setting}:{side}:{k:03d}`; the same `{side}:{k}` across languages is the same English item |
| `setting` | language |
| `side` | `harm` or `harmless` |
| `split` | `fit` or `dev` |
| `prompt` | the prompt in the target language |
| `goal_en` | the English source prompt |
| `source` | the upstream dataset and revision |

## Sources

- **harm**: [HarmBench](https://github.com/centerforaisafety/HarmBench) standard behaviours (MIT), commit `8e1604d`. 150 behaviours sampled with seed 0.
- **harmless**: [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) `or-bench-hard-1k` (CC BY 4.0), revision `e36d8b8`. These are harmless prompts that over-cautious models tend to refuse. 150 sampled with seed 0.

## Translation and filtering

- **Yoruba, Igbo, Hausa**: machine-translated with [NLLB-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B), then scored with the reference-free quality estimator [SSA-COMET-QE](https://huggingface.co/McGill-NLP/ssa-comet-qe-final). A translation was kept only if it scored at least the 10th percentile of TukaBench's human-corrected translations into the same language. The thresholds were Hausa 0.497, Igbo 0.513 and Yoruba 0.526. On these prompts NLLB scores close to the human translations: median 0.58 vs 0.62 for Hausa, 0.61 vs 0.60 for Igbo, 0.66 vs 0.61 for Yoruba.
- **Nigerian Pidgin**: machine-translated with MAFAND-MT ([masakhane/m2m100_418M_en_pcm_rel_news_ft](https://huggingface.co/masakhane/m2m100_418M_en_pcm_rel_news_ft)). Each translation then went through four gates: back-translation chrF, length ratio, a refusal-text check and GlotLID. Pidgin keeps fewer rows than the other languages because MAFAND often produces near-English output, which the language-ID gate rejects.

None of the translations here were post-edited.

## Licence

CC BY-NC 4.0, because the Yoruba, Igbo and Hausa text comes from NLLB-200, which is CC BY-NC 4.0. HarmBench is MIT, OR-Bench is CC BY 4.0 and MAFAND-MT is AFL-3.0.

Part of the [oluso-datasets](https://huggingface.co/collections/alanmik/oluso-datasets-6aeed01398a541401bbce8c8) collection.
