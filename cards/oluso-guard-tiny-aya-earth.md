# Oluso guard for Tiny Aya Earth

An inference-time safety and calibration patch for [CohereLabs/tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth) (revision `0624fafb`), aimed at Yoruba, Igbo, Hausa, Nigerian Pidgin and Pidgin-English code-mix. **There are no weights here.** The base model is used unchanged, and this repo holds under 20 KB:

| File | Contents |
|---|---|
| `steering.pt` | the CAA refusal vector for layer 18, and the CAST gate's condition vector for layer 15 |
| `guard.json` | model revision, layer, α = 1.0, the gate (layer 15, fires when similarity exceeds 0.14; SafeTune comparator `"smaller"`; held-out F1 0.91), the temperature T = 2.0, and provenance |

The vectors were fitted with [SafeTune](https://pypi.org/project/safetune/) 0.1.12 on [alanmik/oluso-steer-fit-ng](https://huggingface.co/datasets/alanmik/oluso-steer-fit-ng). The configuration was selected on that set's dev split by a rule written before any steered output was seen.

## Use

```python
# pip install "git+https://github.com/big14way/oluso"   (needs the Tiny Aya Earth licence accepted)
from oluso.guard import Guard

g = Guard.load("alanmik/oluso-guard-tiny-aya-earth")
g.chat(["Abeg, how I fit ..."])                  # steered when the gate fires on the prompt
g.multiple_choice(question, options)             # option probabilities at temperature 2.0
```

## What it does, measured on the locked test set

Evaluated once on [alanmik/oluso-safety-ng](https://huggingface.co/datasets/alanmik/oluso-safety-ng), 1,600 prompts, pooled over the non-English settings. Changes are in points against the unchanged model on the same items, with 95% confidence intervals:

| | Change |
|---|---|
| Harmful answers to harmful requests | −6 [−9, −3] |
| Harmful answers to culturally grounded harmful requests | −17 [−22, −12] |
| Refusals of harmful requests | +10 [+6, +13] |
| **Refusals of benign requests (over-refusal)** | **+13 [+10, +16]** |

A random direction of the same norm behind the same gate changes nothing. Hausa harmful answers fall from 15% to 4%, Yoruba from 35% to 25%.

**Read the cost before using it.** On this test set the gate fires on 96% of benign prompts in Yoruba and Igbo, so in those languages it acts almost like always-on steering. Over-refusal rises from 10% to 27% in Yoruba and from 5% to 17% in Igbo. In English, Pidgin and code-mix the gate is more selective. Even there over-refusal rises (13% → 20% in English), but by less than half as much as ungated steering.

**Calibration.** The temperature brings AfriMMLU calibration error from 0.19–0.22 to 0.05 (the noise floor) in Yoruba, Igbo and Hausa, and from 0.23 to 0.08 in English. The gate never fired on AfriMMLU prompts, so multiple-choice accuracy is unchanged.

**Cost.** 12% of generation throughput. On one A10G in bfloat16: about 7,400 requests per hour.

Full method, controls, statistics and limitations: [github.com/big14way/oluso](https://github.com/big14way/oluso).

## Limitations

- One model revision. The gate was fitted on HarmBench vs OR-Bench prompts, and it does not separate benign near-twins of harmful requests in Yoruba and Igbo.
- The judge reads machine translations. It was validated on English human labels but not by native speakers.
- Not a substitute for a moderation layer. It moves refusal rates; it does not guarantee them.

## License

CC BY-NC 4.0, following Tiny Aya Earth (the vectors are derived from its activations) and the NLLB-translated fit data.
