# Oluso safety test set: Nigerian languages

1,600 prompts for measuring how often a language model refuses harmful requests, and how often it wrongly refuses harmless ones, in English, Yoruba, Igbo, Hausa, Nigerian Pidgin and Pidgin-English code-mixed text.

> **Content warning.** The `harm` prompts are requests for harmful content: harassment, malware, physical harm, fraud, disinformation, sexual content and more. They are here to evaluate model safety. Do not use them to train a model to comply.

## Configs

One config per language setting, each with a single `test` split.

| config | language | harm | benign | culture | rows |
|---|---|---:|---:|---:|---:|
| `en` | English | 100 | 100 | 100 | 300 |
| `yo` | Yoruba | 100 | 100 | 100 | 300 |
| `ig` | Igbo | 100 | 100 | 100 | 300 |
| `ha` | Hausa | 100 | 100 | 100 | 300 |
| `pcm` | Nigerian Pidgin | 100 | 100 | – | 200 |
| `pcm-en` | Pidgin-English code-mix | 100 | 100 | – | 200 |

```python
from datasets import load_dataset
ds = load_dataset("alanmik/oluso-safety-ng", "pcm", split="test")
```

## Fields

| field | meaning |
|---|---|
| `id` | `{setting}:{label}:{index:03d}` |
| `setting` | `en`, `yo`, `ig`, `ha`, `pcm` or `pcm-en` |
| `label` | `harm` (should be refused), `benign` (should be answered), `culture` (culturally grounded benign request) |
| `index` | JailbreakBench behaviour index. It pairs each item across languages, and pairs each harmful request with its benign counterpart. |
| `prompt` | the prompt in the target language |
| `goal_en` | the English source prompt |
| `source` | where the prompt text comes from (see below) |

## Sources

- **`en`, `yo`, `ig`, `ha`**: [TukaBench](https://huggingface.co/datasets/McGill-NLP/tukabench) at revision `d9194efc`, used unchanged. The `harm` and `benign` items are human-corrected translations of [JailbreakBench JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors). The `culture` items are TukaBench's own.
- **`pcm`** (new): each English goal got two candidate translations:
  - one from MAFAND-MT ([masakhane/m2m100_418M_en_pcm_rel_news_ft](https://huggingface.co/masakhane/m2m100_418M_en_pcm_rel_news_ft));
  - one from a few-shot rewrite by [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global), run in plain completion mode with eight reviewed example triples.

  The better candidate under the quality gates below was kept, then post-edited where needed. `source` records which one survived: the MT model, the rewrite model, or `post-edit`.
- **`pcm-en`** (new): Tiny Aya Global few-shot rewrites into the mixed register Nigerians use online, gated and then post-edited where needed.

The model under evaluation (Tiny Aya Earth) was never used to write any test prompt.

## Quality control

Each Pidgin prompt was checked four ways:
- its back-translation to English (MAFAND pcm→en) was scored with chrF;
- its length ratio to the English was checked;
- it was checked for refusal text;
- [GlotLID](https://huggingface.co/cis-lmu/glotlid) v4 language ID was run on it.

Each code-mixed prompt had to:
- contain Nigerian Pidgin grammar markers;
- stay close to the English meaning;
- differ from the pure Pidgin version.

Post-editing ran in two passes: 137 Pidgin and 95 code-mixed prompts were rewritten by hand. The passes fixed meaning drift (the harmful or harmless intent must match the English), garbled grammar and pronoun errors. Spelling mostly follows BBC News Pidgin (`di`, `pipo`, `pesin`, `wetin`), with some Nigerian social-media spellings kept, because that is how people actually type.

## Known limitations

- **Native review is incomplete.** A native speaker approved the eight example triples used for the few-shot rewrites. The post-edits themselves have not yet had a full native-speaker pass. Corrections are welcome in the Community tab.
- **Language ID is unreliable on short Pidgin prompts.** GlotLID labels 33 of the 200 `pcm` prompts as English or gives Pidgin low confidence. These prompts all contain Pidgin grammar, so they were kept, and the flag was treated as advisory.
- **Translationese.** Pidgin and code-mixed text translated from English sounds more formal than Pidgin people write from scratch.
- **Spelling varies.** Nigerian Pidgin has no single standard spelling.

## Licence

CC BY-NC 4.0. TukaBench, Tiny Aya Global and NLLB-200 are all CC BY-NC 4.0, JBB-Behaviors is MIT, and the MAFAND-MT models are AFL-3.0. Please cite TukaBench and JailbreakBench as well as this dataset.

Part of the [oluso-datasets](https://huggingface.co/collections/alanmik/oluso-datasets-6aeed01398a541401bbce8c8) collection.
